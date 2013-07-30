# Written by Willem Onderwaater and Sander Roobol as part of a collaboration
# between the ID03 beamline at the European Synchrotron Radiation Facility and
# the Interface Physics group at Leiden University.

import sys
import os
import time
import copy
import itertools
import subprocess
import random
import glob
import cPickle as pickle
import gzip
import argparse
import numbers

import numpy


from PyMca import SixCircle
from PyMca import specfilewrapper
import EdfFile

import getconfig

import inspect
import re
import scipy.optimize, scipy.special
from scipy.stats import linregress


def sum_onto(a, axis):
    for i in reversed(range(len(a.shape))):
        if axis != i:
            a = a.sum(axis=i)
    return a


class Axis(object):
    def __init__(self, min, max, res, label=None):
        self.res = float(res)
        if round(min / self.res) != round(min / self.res,6) or round(max / self.res) != round(max / self.res,6):
            self.min = numpy.floor(float(min)/self.res)*self.res
            self.max = numpy.ceil(float(max)/self.res)*self.res
        else:
            self.min = min
            self.max = max
        self.label = label
    
    def __len__(self):
        return int(round((self.max - self.min) / self.res)) + 1

    def __getitem__(self, index):
        if index >= len(self):  # to support iteration
            raise IndexError('index out of range')
        return self.min + index * self.res

    def get_index(self, value):
        if isinstance(value, numbers.Number):
            if self.min <= value <= self.max:
                return int(round((value - self.min) / self.res))
            raise ValueError('cannot get index: value {0} not in range [{1}, {2}]'.format(value, self.min, self.max))
        else:
            if ((self.min <= value) & (value <= self.max)).all():
                return numpy.around((value - self.min) / self.res).astype(int)
            raise ValueError('cannot get indices, values from [{0}, {1}], axes range [{2}, {3}]'.format(value.min(), value.max(), self.min, self.max))

    def __or__(self, other): # union operation
        if not isinstance(other, Axis):
            return NotImplemented
        if not self.is_compatible(other):
            raise ValueError('cannot unite axes with different resolution/label')
        return self.__class__(min(self.min, other.min), max(self.max, other.max), self.res, self.label)

    def __equal__(self, other):
        if not isinstance(other, Axis):
            return NotImplemented
        return self.res == other.res and self.min == other.min and self.max == other.max and self.label == other.label

    def __hash__(self):
        return hash(self.min) ^ hash(self.max) ^ hash(self.res) ^ hash(self.label)

    def is_compatible(self, other):
        if not isinstance(other, Axis):
            return False
        return self.res == other.res and self.label == other.label

    def __contains__(self, other):
        if isinstance(other, numbers.Number):
            return self.min <= other <= self.max
        elif isinstance(other, Axis):
            return self.is_compatible(other) and self.min <= other.min and self.max >= other.max

    def rebound(self, min, max):
        return self.__class__(min, max, self.res, self.label)

    def rebin(self, factor):
        newres = self.res*factor
        left = int(round(self.min/self.res))
        right = int(round(self.max/self.res))
        new = self.__class__(newres * numpy.floor(round(self.min / newres, 3)), newres * numpy.ceil(round(self.max / newres, 3)), newres, self.label)
        return left % factor, -right % factor, new

    def __repr__(self):
        return '{0.__class__.__name__} {0.label} (min={0.min}, max={0.max}, res={0.res}, count={1})'.format(self, len(self))


class EmptySpace(object):
    def __iadd__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        return other


class Space(object):
    def __init__(self, axes):
        self.axes = tuple(axes)
        
        self.photons = numpy.zeros([len(ax) for ax in self.axes], order='C')
        self.contributions = numpy.zeros(self.photons.shape, dtype=numpy.uint32, order='C')

    @classmethod
    def fromcfg(cls, cfg): # FIXME: to be removed once automatic HKL limits detection is working
        return cls((
            Axis(cfg.Hmin, cfg.Hmax, cfg.Hres, 'H'),
            Axis(cfg.Kmin, cfg.Kmax, cfg.Kres, 'K'),
            Axis(cfg.Lmin, cfg.Lmax, cfg.Lres, 'L'),
        ))

    def get(self):
        return self.photons/self.contributions
    
    def __getitem__(self, key):
        if key is Ellipsis:
            return self.get_masked()
        if isinstance(key, numbers.Number) or isinstance(key, slice):
            if not len(self.axes) == 1:
                raise IndexError('dimension mismatch')
            else:
                key = [key]
        elif not isinstance(key, tuple) or not len(key) == len(self.axes):
            raise IndexError('dimension mismatch')

        newkey = tuple(self._convertindex(k,ax) for k, ax in zip(key, self.axes))
        return self.get_masked()[newkey]

    def _convertindex(self,key,ax):
        if isinstance(key,slice):
            if key.step is not None:
                raise IndexError('stride not supported')
            if key.start is None:
                start = None
            else:
                start = ax.get_index(key.start)
            if key.stop is None:
                stop = None
            else:
                stop = ax.get_index(key.stop)
            return slice(start, stop)
        if isinstance(key, numbers.Number):
            return ax.get_index(key)
        raise TypeError('unrecognized slice')

    def get_masked(self):
        return numpy.ma.array(data=self.get(), mask=(self.contributions == 0))
        
    def __add__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        if not len(self.axes) == len(other.axes) or not all(a.is_compatible(b) for (a, b) in zip(self.axes, other.axes)):
            raise ValueError('cannot add spaces with different dimensionality or resolution')

        new = Space([a | b for (a, b) in zip(self.axes, other.axes)])
        new += self
        new += other
        return new

    def __iadd__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        if not len(self.axes) == len(other.axes) or not all(a.is_compatible(b) for (a, b) in zip(self.axes, other.axes)):
            raise ValueError('cannot add spaces with different dimensionality or resolution')

        if not all(other_ax in self_ax for (self_ax, other_ax) in zip(self.axes, other.axes)):
            return self.__add__(other)

        index = tuple(slice(self_ax.get_index(other_ax.min), self_ax.get_index(other_ax.min) + len(other_ax)) for (self_ax, other_ax) in zip(self.axes, other.axes))
        self.photons[index] += other.photons
        self.contributions[index] += other.contributions
        return self

    def trim(self):
        mask = self.contributions > 0
        lims = (numpy.flatnonzero(sum_onto(mask, i)) for (i, ax) in enumerate(self.axes))
        lims = tuple((i.min(), i.max()) for i in lims)
        self.axes = tuple(ax.rebound(ax[min], ax[max]) for (ax, (min, max)) in zip(self.axes, lims))
        slices = tuple(slice(min, max+1) for (min, max) in lims)
        self.photons = self.photons[slices].copy()
        self.contributions = self.contributions[slices].copy()

    def rebin(self, factors):
        if len(factors) != len(self.axes):
            raise ValueError('dimension mismatch between factors and axes')
        if not all(isinstance(factor, int) for factor in factors) or not all(factor % 2 == 0 for factor in factors):
            raise ValueError('binning factors must be even integers')

        lefts, rights, newaxes = zip(*[ax.rebin(factor) for ax, factor in zip(self.axes, factors)])
        tempshape = tuple(size + left + right + factor for size, left, right, factor in zip(self.photons.shape, lefts, rights, factors))

        photons = numpy.zeros(tempshape, order='C')
        contributions = numpy.zeros(tempshape, dtype=numpy.uint32, order='C')
        pad = tuple(slice(left+factor/2, left+factor/2+size) for left, factor, size in zip(lefts, factors, self.photons.shape))
        photons[pad] = self.photons
        contributions[pad] = self.contributions

        new = Space(newaxes)
        for offsets in itertools.product(*[range(factor) for factor in factors]):
            stride = tuple(slice(offset, offset+size+left+factor/2, factor) for offset, size, factor, left in zip(offsets, self.photons.shape, factors, lefts))
            new.photons += photons[stride]
            new.contributions += contributions[stride]

        return new
    
    def process_image(self, coordinates, intensity):
        # note: coordinates must be tuple of arrays, not a 2D array
        if len(coordinates) != len(self.axes):
            raise ValueError('dimension mismatch between coordinates and axes')

        valid = numpy.isfinite(intensity)
        intensity = intensity[valid]
        if not intensity.size:
            return
        coordinates = tuple(coord[valid] for coord in coordinates)

        indices = numpy.array(tuple(ax.get_index(coord) for (ax, coord) in zip(self.axes, coordinates)))
        for i in range(0, len(self.axes)):
            for j in range(i+1, len(self.axes)):
                indices[i,:] *= len(self.axes[j])
        indices = indices.sum(axis=0).astype(int)
        photons = numpy.bincount(indices, weights=intensity)
        contributions = numpy.bincount(indices)
    
        self.photons.ravel()[:photons.size] += photons
        self.contributions.ravel()[:contributions.size] += contributions

    def tofile(self, filename):
        tmpfile = '{0}-{1:x}.tmp'.format(os.path.splitext(filename)[0], random.randint(0, 2**32-1))
        fp = gzip.open(tmpfile, 'wb')
        try:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)
        finally:
            fp.close()
        os.rename(tmpfile, filename)
    
    @classmethod
    def fromfile(cls, filename):
        fp = gzip.open(filename,'rb')
        try:
            return pickle.load(fp)
        finally:
            fp.close()


class ProjectionBase(object):
    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def fromcfg(cls, cfg):
        if cfg.projection == 'hkl':
            return HKLProjection(cfg)
        elif cfg.projection == 'twotheta':
            return TwoThetaProjection(cfg)
        else:
            raise ValueError('unknown projection {0}'.format(cfg.projection))

    def space_from_bounds(self, *args, **kwargs):
        limits = self.get_bounds(*args, **kwargs)
        try:
            iter(cfg.resolution)
        except TypeError:
            resolution = itertools.repeat(cfg.resolution)
        else:
            if len(cfg.resolution) != len(limits):
                raise ValueError('not enough values in given for resolution')
            resolution = cfg.resolution
        return Space(Axis(min, max, res, label) for ((min, max), res, label) in zip(limits, resolution, self._get_labels()))


class HKLProjection(ProjectionBase):
    def get_bounds(self, scan):
        h = []
        k = []
        l = []
        for n in range(self.length):
            coordinates , intensity = self.getImdata(n)
            h.extend([coordinates[0].min(),coordinates[0].max()])
            k.extend([coordinates[1].min(),coordinates[1].max()])
            l.extend([coordinates[2].min(),coordinates[2].max()])
        offset = 0.0001
        return (min(h) - offset, max(h) + offset), (min(k)- offset, max(k) + offset), (min(l)- offset, max(l) + offset)

    # arrays: gamma, delta
    # scalars: theta, mu, chi, phi
    def project(self, **kwargs):
        R = SixCircle.getHKL(self.wavelength, self.UB, **kwargs)
        H = R[0,:]
        K = R[1,:]
        L = R[2,:]
        return (H,K,L)

    def _get_labels(self):
        return 'H', 'K', 'L'


class TwoThetaProjection(HKLProjection):
    def get_bounds(self, *args, **kwargs):
        (hmin, hmax), (kmin, kmax), (lmin, lmax) = super(TwoThetaProjection, self).get_bounds(*args, **kwargs)
        return ((self._hkl_to_tth(hmin, kmin, lmin), self._hkl_to_tth(hmax, kmax, lmax)),)

    def project(self, **kwargs):
        h,k,l = super(TwoThetaProjection, self).project(**kwargs)
        return self._hkl_to_tth(h, k, l)

    def _hkl_to_tth(self, h, k, l):
        return 2 * numpy.arcsin(self.wavelength * numpy.sqrt(h**2+k**2+l**2) / 4 / numpy.pi)

    def _get_labels(self):
        return 'TwoTheta'



class ScanBase(object):
    def __init__(self, cfg, spec, scannumber, scan=None):
        self.cfg = cfg
        self.projection = ProjectionBase.fromcfg(cfg)
        #self.background = BackgroundBase.fromcfg(cfg)
   
        self.scannumber = scannumber
        if scan:
            self.scan = scan
        else:
            self.scan = self.get_scan(scannumber)
                
    def initImdata(self):
        self.buildfilelist()
    
    def buildfilelist(self):
        allfiles =  glob.glob(self.imagepattern)
        if len(allfiles) == 0:
            raise ValueError('Empty filelist, check if the specified imagefolder corresponds to the location of the images, looking for images in: {0}'.format(self.imagepattern))
        filelist = list()
        imagedict = {}
        for file in allfiles:
            filename = os.path.basename(file).split('.')[0]
            scanno, pointno, imageno = filename.split('_')[-3:]
            scanno, pointno, imageno = int(scanno), int(pointno), int(imageno)
            if not scanno in imagedict:
                imagedict[scanno] = {}
            imagedict[scanno][pointno] = file
        try:
            filedict = imagedict[self.scannumber]
        except:
            raise ValueError('Scannumber {0} not in this folder. Folder contains scannumbers {1}-{2}'.format(self.scannumber, min(imagedict.keys()), max(imagedict.keys())))
        points = sorted(filedict.iterkeys())
        self.filelist = [filedict[i] for i in points]

    
    def apply_roi(self, data):
        roi = data[self.cfg.ymask, :]
        return roi[:, self.cfg.xmask]

    def getCorrectedData(self,n):
        return self.GetData(n)/(self.mon[n]*self.transm[n])
    
    def getImdata(self,n):
        data = self.getCorrectedData(n)
        app = self.cfg.app #angle per pixel (delta,gamma)
        centralpixel = self.cfg.centralpixel #(row,column)=(delta,gamma)
        gamma = -app[1]*(numpy.arange(data.shape[1])-centralpixel[1])+self.gamma[n]
        delta = app[0]*(numpy.arange(data.shape[0])-centralpixel[0])+self.delta[n]
        gamma = gamma[self.cfg.ymask]
        delta = delta[self.cfg.xmask]
        
        coordinates = self.projection.project(delta=delta, theta=self.theta[n], chi=self.chi, phi=self.phi, mu=self.mu, gamma=gamma)
        
        roi = self.apply_roi(data)
        intensity = roi.flatten()
        #intensity = self.background.correct(roi, self.fitdata).flatten()
                
        return coordinates, intensity

    
    @staticmethod
    def get_scan(spec, scannumber):
        return spec.select('{0}.1'.format(scannumber))

    @classmethod
    def detect_scan(cls, cfg, spec, scanno):
        scan = cls.get_scan(spec, scanno)
        scantype = scan.header('S')[0].split()[2]
        if cfg.hutch == 'EH1':
            if scantype.startswith('zap'):
                return ZapScan(cfg, spec, scanno, scan)
            else:
                return ClassicScan(cfg, spec, scanno, scan)
        elif cfg.hutch =='EH2':
            if scantype.startswith('zap'):
                return EH2ZapScan(cfg, spec, scanno, scan)
            else:
                return EH2ClassicScan(cfg, spec, scanno, scan)
        else:
            raise ValueError('Hutch type not recognized: {0}'.format(cfg.hutch))

    def get_space(self):
        self.projection.getImdata = self.getImdata
        self.projection.length = self.length
        return self.projection.space_from_bounds(self.scan)


class ZapScan(ScanBase):
    def __init__(self, cfg, spec, scanno, scan=None):
        super(ZapScan, self).__init__(cfg, spec, scanno, scan)

        scanheaderC = self.scan.header('C')
        folder = os.path.split(scanheaderC[0].split(' ')[-1])[-1]
        scanname = scanheaderC[1].split(' ')[-1]
        self.imagepattern = os.path.join(cfg.imagefolder, folder,'*{0}*mpx*'.format(scanname))
        self.scannumber = int(scanheaderC[2].split(' ')[-1])#is different from scanno should be changed in spec!
        
        #UB matrix will be installed in new versions of the zapline, it has to come from the configfile
        if 'UB' not in cfg.__dict__.keys():
            raise getconfig.ConfigError('UB')
        
        self.projection.UB = numpy.array(cfg.UB)
        
        self.projection.wavelength = float(self.scan.header('G')[1].split(' ')[-1])

        delta, theta, self.chi, self.phi, self.mu, gamma = numpy.array(self.scan.header('P')[0].split(' ')[1:7],dtype=numpy.float)
        
    
        self.theta = self.scan.datacol('th')        
        self.length = numpy.alen(self.theta)
                
        #correction for difference between back and forth in th motor
        correction = (self.theta[1] - self.theta[0]) / (self.length * 1.0) / 2
        self.theta -= correction
                
        self.gamma = gamma.repeat(self.length)
        self.delta = delta.repeat(self.length)

        self.mon = self.scan.datacol('zap_mon')
        self.transm = self.scan.datacol('zap_transm')
        self.transm[-1]=self.transm[-2] #bug in specfile

    def initImdata(self):
        super(ZapScan, self).initImdata()
        self.edf = EdfFile.EdfFile(self.filelist[0])

    def GetData(self,n):
        return self.edf.GetData(n)
        
class ClassicScan(ScanBase):
    def __init__(self, cfg, spec, scanno, scan=None):
        super(ClassicScan, self).__init__(cfg, spec, scanno, scan)

        try:
            UCCD = os.path.split(self.scan.header('UCCD')[0].split(' ')[-1])
            folder = os.path.split(UCCD[0])[-1]
            scanname = UCCD[-1].split('_')[0]
            self.imagepattern = os.path.join(cfg.imagefolder, folder, '*{0}*'.format(scanname))
        except:
            print 'Warning: No UCCD tag was found. Searching folder {0} for all images'.format(cfg.imagefolder)
            self.imagepattern = '*/*'
                
        self.projection.UB = numpy.array(self.scan.header('G')[2].split(' ')[-9:],dtype=numpy.float)
        self.projection.wavelength = float(self.scan.header('G')[1].split(' ')[-1])

        delta, theta, self.chi, self.phi, self.mu, gamma = numpy.array(self.scan.header('P')[0].split(' ')[1:7],dtype=numpy.float)
        self.theta = self.scan.datacol('thcnt')
        self.gamma = self.scan.datacol('gamcnt')
        self.delta = self.scan.datacol('delcnt')

        self.mon = self.scan.datacol('mon')
        self.transm = self.scan.datacol('transm')
        self.length = numpy.alen(self.theta)

    def GetData(self,n):
        edf = EdfFile.EdfFile(self.filelist[n])
        return edf.GetData(0)

class EH2ScanBase(ScanBase):
    def getImdata(self,n):
        sdd = self.cfg.sdd / numpy.cos(self.gamma[n] *numpy.pi /180)
        areacorrection = (self.cfg.sdd / sdd)**2
        data = self.getCorrectedData(n) * areacorrection
        pixelsize = self.cfg.sdd * numpy.tan(self.cfg.app[0] *numpy.pi /180 )
        app = numpy.arctan(pixelsize/sdd) * 180 / numpy.pi
        centralpixel = self.cfg.centralpixel #(row,column)=(delta,gamma)
        gamma = app*(numpy.arange(data.shape[1])-centralpixel[1])+self.gamma[n]
        delta = app*(numpy.arange(data.shape[0])-centralpixel[0])+self.delta[n]
        gamma = gamma[self.cfg.xmask]
        delta = delta[self.cfg.ymask]
        coordinates = self.projection.project(delta=delta, theta=self.theta[n], chi=self.chi, phi=self.phi, mu=self.mu, gamma=gamma)
        roi = self.apply_roi(data)
        intensity = numpy.rot90(roi).flatten()
        return coordinates, intensity


class EH2ClassicScan(EH2ScanBase):
    def __init__(self, cfg, spec, scanno, scan=None):
        super(EH2ClassicScan, self).__init__(cfg, spec, scanno, scan)
        
        try:
            UCCD = os.path.split(self.scan.header('UCCD')[0].split(' ')[-1])
            folder = os.path.split(UCCD[0])[-1]
            scanname = UCCD[-1].split('_')[0]
            self.imagepattern = os.path.join(cfg.imagefolder, folder, '*{0}*'.format(scanname))
        except:
            print 'Warning: No UCCD tag was found. Searching folder {0} for all images'.format(cfg.imagefolder)
            self.imagepattern = '*/*.edf*'
        
        self.projection.UB = numpy.array(self.scan.header('G')[2].split(' ')[-9:],dtype=numpy.float)
        self.projection.wavelength = float(self.scan.header('G')[1].split(' ')[-1])
        
        delta, theta, self.chi, self.phi, self.mu, gamma = numpy.array(self.scan.header('P')[0].split(' ')[1:7],dtype=numpy.float)
        self.theta = self.scan.datacol('thcnt')
        self.gamma = self.scan.datacol('gamcnt')
        self.delta = self.scan.datacol('delcnt')
        
        self.mon = self.scan.datacol('Monitor')
        self.transm = self.scan.datacol('transm')
        self.length = numpy.alen(self.theta)
    
    def GetData(self,n):
        edf = EdfFile.EdfFile(self.filelist[n])
        return edf.GetData(0)

class EH2ZapScan(EH2ScanBase):
    def __init__(self, cfg, spec, scanno, scan=None):
        super(EH2ZapScan, self).__init__(cfg, spec, scanno, scan)
        
        scanheaderC = self.scan.header('C')
        folder = os.path.split(scanheaderC[0].split(' ')[-1])[-1]
        scanname = scanheaderC[1].split(' ')[-1].split('_')[0]
        self.imagepattern = os.path.join(cfg.imagefolder, folder,'*{0}*_mpx*'.format(scanname))
        self.scannumber = int(scanheaderC[2].split(' ')[-1])#is different from scanno should be changed in spec!
        
        #UB matrix will be installed in new versions of the zapline, it has to come from the configfile
        if 'UB' not in cfg.__dict__.keys():
            raise getconfig.ConfigError('UB')
        
        self.projection.UB = numpy.array(cfg.UB)
        
        self.projection.wavelength = float(self.scan.header('G')[1].split(' ')[-1])
        
        delta, theta, self.chi, self.phi, self.mu, gamma = numpy.array(self.scan.header('P')[0].split(' ')[1:7],dtype=numpy.float)
        
        
        self.theta = self.scan.datacol('th')
        self.length = numpy.alen(self.theta)
    
        #correction for difference between back and forth in th motor
        correction = (self.theta[1] - self.theta[0]) / (self.length * 1.0) / 2
        self.theta -= correction
        
        self.gamma = gamma.repeat(self.length)
        self.delta = delta.repeat(self.length)
        
        self.mon = self.scan.datacol('zap_mon')
        self.transm = self.scan.datacol('zap_transm')
        self.transm[-1]=self.transm[-2] #bug in specfile
    
    def initImdata(self):
        super(EH2ZapScan, self).initImdata()
        self.edf = EdfFile.EdfFile(self.filelist[0])
    
    def GetData(self,n):
        return self.edf.GetData(n)

def process(scanno):
    print scanno
    a = ScanBase.detect_scan(cfg, spec, scanno)
    a.initImdata()
    mesh = a.get_space()
    for m in range(a.length):
        coordinates , intensity = a.getImdata(m)
        mesh.process_image(coordinates, intensity)
    return mesh

def checkscan(scanno):
    try:
        spec = specfilewrapper.Specfile(cfg.specfile)
        a = ScanBase.detect_scan(cfg, spec, scanno)
        a.initImdata()
    except Exception as e:
        print 'Unable to load scan {0}\n'.format(scanno)
        print e.message
        return False
    return True

def makeplot(space, args):
    import matplotlib.pyplot as pyplot
    import matplotlib.colors
    
    clipping = float(args.clip)
    mesh = space.get_masked()
    axes = space.axes
    del space

    # project automatically onto the smallest dimension or from command line argument

    remaining = range(len(axes))
    
    if args.slice and len(axes) == 3:
        s = [slice(None)] * 3
        axlabels = [ax.label.lower() for ax in axes]
        if args.slice[0].lower() in axlabels:
            projected = axlabels.index(args.slice[0].lower())
        if ':' in args.slice[1]:
            mi,ma = args.slice[1].split(':')
            if mi == '':
                mi = axes[projected].min
            if ma == '':
                ma = axes[projected].max
            s[projected] = slice(axes[projected].get_index(float(mi)), axes[projected].get_index(float(ma)))
            data = mesh[s].mean(axis = projected)
        else:
            index = axes[projected].get_index(float(args.slice[1]))
            s[projected] = index
            data = mesh[s].copy() # make a copy so the rest of the mesh can be released from memory
        info = ' sliced at {0} = {1}'.format(axes[projected].label, args.slice[1])
        remaining.pop(projected)

    elif len(remaining) == 3:
        if args.project:
            axlabels = [ax.label.lower() for ax in axes]
            if args.project.lower() in axlabels:
                projected = axlabels.index(args.project.lower())
        else:
            projected = numpy.argmin(mesh.shape)
        info = ' projected on {0}'.format(axes[projected].label)
        remaining.pop(projected)
 
        data = mesh.mean(axis=projected)
    else:
        data = mesh

    if args.ecs:
        colordata = mesh
    else:
        colordata = data
    del mesh

    colordata.mask[colordata == 0] = 1
    colordata = colordata.compressed()

    if args.clip:
        chop = int(round(colordata.size * clipping))
        clip = sorted(colordata)[chop:-(1+chop)]
        vmin, vmax = clip[0], clip[-1]
        del clip
    else:
        vmin, vmax = colordata.min(),colordata.max()
    del colordata

    xmin = axes[remaining[0]].min
    xmax = axes[remaining[0]].max
    ymin = axes[remaining[1]].min
    ymax = axes[remaining[1]].max
    
    pyplot.figure(figsize=(12,9))
    pyplot.imshow(data.transpose(), origin='lower', extent=(xmin, xmax, ymin, ymax), aspect='auto', norm=matplotlib.colors.LogNorm(vmin,vmax))

    pyplot.xlabel(axes[remaining[0]].label)
    pyplot.ylabel(axes[remaining[1]].label)
    pyplot.suptitle('{0}{1}'.format(os.path.splitext(args.outfile)[0], info))
    pyplot.colorbar()
    
    if args.savepdf or args.savefile:
        if args.savefile:
            pyplot.savefig(args.savefile)
            print 'saved at {0}'.format(args.savefile)
        else:
            pyplot.savefig('{0}.pdf'.format(os.path.splitext(args.outfile)[0]))
            print 'saved at {0}.pdf'.format(os.path.splitext(args.outfile)[0])
    else:
        pyplot.show()


def mfinal(filename, scanrange):
    base, ext = os.path.splitext(filename)
    return ('{0}_{2}{1}').format(base,ext,scanrange)


def wait_for_files(filelist):
    filelist = filelist[:] # make copy
    i = 0
    while filelist:
        i = i % len(filelist)
        if os.path.exists(filelist[i]):
            yield filelist.pop(i)
        else:
            time.sleep(5)
            i == 1


def wait_for_file(filename):
    return bool(list(wait_for_files([filename])))


def gbkg(spec, scanno,cfg):
    print scanno
    spec = specfilewrapper.Specfile(cfg.specfile)
    a = ScanBase.detect_scan(cfg, spec, scanno)
    a.initImdata()
    bkg = a.getbkg()
    return bkg


_status_line_length = 0
def status(line, eol=False):
    """Prints a status line to sys.stdout, overwriting the previous one.
    Set eol to True to append a newline to the end of the line"""

    global _status_line_length
    sys.stdout.write('\r{0}\r{1}'.format(' '*_status_line_length, line))
    if eol:
        sys.stdout.write('\n')
        _status_line_length = 0
    else:
        _status_line_length = len(line)

    sys.stdout.flush()

def statusnl(line):
    """Shortcut for status(..., eol=True)"""
    return status(line, eol=True)

def statuseol():
    """Starts a new status line, keeping the previous one intact"""
    global _status_line_length
    _status_line_length = 0
    sys.stdout.write('\n')
    sys.stdout.flush()

def statuscl():
    """Clears the status line, shortcut for status('')"""
    return status('')


if __name__ == "__main__":    
    
    def run(*command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, unused_err = process.communicate()
        retcode = process.poll()
        return retcode, output


    def oarsub(*args, **kwargs):
        opts = kwargs.pop('options', 'walltime=0:15')
        if kwargs:
            raise ValueError('invalid keyword parameter(s): {0}'.format(kwargs))
        scriptname = './blisspython /users/onderwaa/iVoxOar/iVoxOar.py '
        command = '{0} {1}'.format(scriptname, ' '.join(args))
        ret, output = run('oarsub', '-l {0}'.format(opts), command)
        if ret == 0:
            lines = output.split('\n')
            for line in lines:
                if line.startswith('OAR_JOB_ID='):
                    void, jobid = line.split('=')
                    return jobid
        return False


    def oarstat(jobid):
        # % oarstat -s -j 5651374
        # 5651374: Running
        # % oarstat -s -j 5651374
        # 5651374: Finishing
        ret, output = run('oarstat', '-s', '-j', str(jobid))
        if ret == 0:
            for n in output.split('\n'):
                if n.startswith(str(jobid)):
                    job, status = n.split(':')
            return status.strip()
        else:
            return 'Unknown'

    def oarwait(jobs, remaining=0):
        linelen = 0
        if len(jobs) > remaining:
            status('{0}: getting status of {1} jobs...'.format(time.ctime(), len(jobs)))
        else:
            return
     
        while 1:
            i = 0
            R = 0
            W = 0
            U = 0
            while i < len(jobs):
                state = oarstat(jobs[i])
                if state == 'Running':
                    R += 1
                elif state == 'Waiting':
                    W += 1
                elif state == 'Unknown':
                    U += 1
                else: # assume state == 'Finishing' or 'Terminated' but don't wait on something unknown
                    del jobs[i]
                    i -= 1 #otherwise it skips a job
                i += 1
            status('{0}: {1} jobs to go. {2} waiting, {3} running, {4} unknown.'.format(time.ctime(),len(jobs),W,R,U))
            if len(jobs) <= remaining:
                statuseol()
                return
            else:
                time.sleep(30) # only sleep if we're not done yet
            
    def cluster(args):
        if args.single:
            job = oarsub('--config', args.config, 'local', '--multiprocessing', '-o', mfinal(cfg.outfile, args.scanrange), args.scanrange, options='nodes=1/core=4,walltime=0:30')
            oarwait([job])
            print "done"
            return

        prefix = 'iVoxOar-{0:x}'.format(random.randint(0, 2**32-1))
        jobs = []
        scanrange = getconfig.parsemultirange(args.scanrange)
                
        if len(scanrange) == 1:
            jobs.append(oarsub('--config', args.config, '_part', '--trim', '-o', mfinal(cfg.outfile, args.scanrange), str(scanrange[0])))
            print 'submitted 1 job, waiting...'
            oarwait(jobs)
            print 'done'
            return

        if args.split:
            for scanno in scanrange:
                jobs.append(oarsub('--config', args.config, '_part', '--trim', '-o', mfinal(cfg.outfile, scanno), str(scanno)))
            print 'submitted {0} jobs, waiting...'.format(len(jobs))
            oarwait(jobs)
            print 'done!'
            return
                    
        parts = []
        for i, scanno in enumerate(scanrange):
            part = '{0}/{1}-part-{2}.zpi'.format(args.tmpdir, prefix, scanno)
            jobs.append(oarsub('--config', args.config,'_part','-o', part, str(scanno)))
            status('submitted {0} jobs...'.format(i+1))
            parts.append(part)
        statusnl('submitted {0} jobs, waiting...'.format(len(jobs)))
        oarwait(jobs, 25)

        count = len(scanrange)
        chunkcount = int(numpy.ceil(float(count) / args.chunksize))
        if chunkcount == 1:
            jobs.append(oarsub('--config', args.config,'--wait','_sum' ,'--trim', '--delete', '-o', mfinal(cfg.outfile,args.scanrange), *parts))
        else:
            chunksize = int(numpy.ceil(float(count) / chunkcount))
            chunks = []
            for i in range(chunkcount):
                chunk = '{0}/{1}-chunk-{2}.zpi'.format(args.tmpdir, prefix, i+1)
                jobs.append(oarsub('--config', args.config,'--wait','_sum', '--delete', '-o', chunk, *parts[i*chunksize:(i+1)*chunksize]))
                chunks.append(chunk)
             
            jobs.append(oarsub('--config', args.config,'--wait','_sum', '--trim', '--delete', '-o', mfinal(cfg.outfile,args.scanrange), *chunks))
        print 'submitted {0} jobs, waiting...'.format(len(jobs))
        oarwait(jobs)
        print 'done!'


    def part(args):
        global spec
        spec = specfilewrapper.Specfile(cfg.specfile)
        space = process(args.scan)
        
        if args.trim:
            space.trim()
        space.tofile(args.outfile)


    def cmd_sum(args):
        globalspace = EmptySpace()

        if args.wait:
            fileiter = wait_for_files(args.infiles)
        else:
            fileiter = args.infiles

        for fn in fileiter:
            print fn
            result = Space.fromfile(fn)
            if result is not None:
                globalspace += result

        if args.trim:
            globalspace.trim()
        
        globalspace.tofile(args.outfile)
                    
        if args.delete:
            for fn in args.infiles:
                try:
                    os.remove(fn)
                except:
                    pass


    def local(args):
        global spec
        spec = specfilewrapper.Specfile(cfg.specfile)
        
        scanlist = getconfig.parsemultirange(args.scanrange)
        globalspace = EmptySpace()
     
        if args.multiprocessing:
            import multiprocessing
            pool = multiprocessing.Pool()
            iter = pool.imap_unordered(process, scanlist, 1)
        else:
            iter = itertools.imap(process, scanlist)
     
        for result in iter:
            if result is not None:
                globalspace += result

        globalspace.trim()
        if args.outfile:
            outfile = args.outfile
        else:
            outfile = mfinal(cfg.outfile, args.scanrange)
        globalspace.tofile(outfile)

        if args.plot:
            if args.plot is True:
                makeplot(globalspace, None)
            else:
                makeplot(globalspace, args.plot)

    def plot(args):
        if args.wait:
            wait_for_file(args.outfile)
        space = Space.fromfile(args.outfile)
        makeplot(space, args)
    
    def export(args):
        if args.wait:
            wait_for_file(args.outfile)
        space = Space.fromfile(args.outfile)
        ext = os.path.splitext(args.savefile)[-1]
        
        if args.rebin:
            if ',' in args.rebin:
                factors = tuple(int(i) for i in args.rebin.split(','))
            else:
                factors = (int(args.rebin),)
            space = space.rebin(factors)
 
        if ext == '.edf':
            header = {}
            for a in space.axes:
                header[str(a.label)] = '{0} {1} {2}'.format(a.min,a.max,a.res)
            edf = EdfFile.EdfFile(args.savefile)
            edf.WriteImage(header,space.get_masked().filled(0),DataType="Float")
            print 'saved at {0}'.format(args.savefile)

        elif ext == '.txt':
            tmpfile = '{0}-{1:x}.tmp'.format(os.path.splitext(args.savefile)[0], random.randint(0, 2**32-1))
            fp = open(tmpfile,'w')
            try:
                grid = numpy.mgrid[tuple(slice(0, len(a)) for a in space.axes)]
                columns = tuple((grid[n] * space.axes[n].res + space.axes[n].min).flatten() for n in range(grid.ndim-1))
                data = space.get_masked().filled(0).flatten()
                for a in space.axes:
                    fp.write('{0}\t'.format(a.label))
                fp.write('intensity')
                fp.write('\n')
                for n in range(len(data)):
                    for m in range(grid.ndim-1):
                        fp.write(str(columns[m][n]))
                        fp.write('\t')
                    fp.write(str(data[n]))
                    fp.write('\n')
                    fp.flush()
            finally:
                fp.close()
                print 'saved at {0}'.format(args.savefile)
                os.rename(tmpfile, args.savefile)

        elif ext == '.zpi':
            space.tofile(args.savefile)
            print 'saved at {0}'.format(args.savefile)

        else:
            print 'unknown extension, unable to save!'
            sys.exit(1)

    def check(args):
        print 'checking scans'
        for scanno in getconfig.parsemultirange(args.scanrange):
            print 'Checking scan: {0}'.format(scanno)
            if not checkscan(scanno):
                print 'exiting...'
                return
        print 'all good'
        


    parser = argparse.ArgumentParser(prog='iVoxOar')
    parser.add_argument('--config',default='./config')
    parser.add_argument('--projection')
    parser.add_argument('--wait', action='store_true', help='wait for input files to appear')
    subparsers = parser.add_subparsers()

    parser_cluster = subparsers.add_parser('cluster')
    parser_cluster.add_argument('scanrange')
    parser_cluster.add_argument('-o', '--outfile')
    parser_cluster.add_argument('--tmpdir', default='.')
    parser_cluster.add_argument('--chunksize', default=20, type=int)
    parser_cluster.add_argument('--split', action='store_true', help='dont sum files (for timescans)')
    parser_cluster.add_argument('--single', action='store_true', help='run entire job on a single node')
    parser_cluster.set_defaults(func=cluster)

    parser_part = subparsers.add_parser('_part')
    parser_part.add_argument('scan', type=int)
    parser_part.add_argument('-o', '--outfile',required=True)
    parser_part.add_argument('--trim', action='store_true')
    parser_part.set_defaults(func=part)
    
    parser_sum = subparsers.add_parser('_sum')
    parser_sum.add_argument('-o', '--outfile',required=True)
    parser_sum.add_argument('--delete', action='store_true')
    parser_sum.add_argument('--trim', action='store_true')
    parser_sum.add_argument('infiles', nargs='+')
    parser_sum.set_defaults(func=cmd_sum)

    parser_local = subparsers.add_parser('local')
    parser_local.add_argument('scanrange')
    parser_local.add_argument('-o', '--outfile')
    parser_local.add_argument('-p', '--plot', nargs='?', const=True)
    parser_local.add_argument('-m', '--multiprocessing', action='store_true')
    parser_local.set_defaults(func=local)

    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('outfile')
    parser_plot.add_argument('-s', '--savepdf', action='store_true')
    parser_plot.add_argument('-c', '--clip', default = 0.00)
    parser_plot.add_argument('-p', '--project', default=False)
    parser_plot.add_argument('--ecs', action='store_true', help='extend color scale to whole mesh not just the to be plotted data')
    parser_plot.add_argument('--slice', nargs=2, default=False)
    parser_plot.add_argument('--savefile')
    parser_plot.set_defaults(func=plot)

    parser_export = subparsers.add_parser('export')
    parser_export.add_argument('--rebin', default=None)
    parser_export.add_argument('outfile')
    parser_export.add_argument('savefile')
    parser_export.set_defaults(func=export)

    parser_check = subparsers.add_parser('check')
    parser_check.add_argument('scanrange')
    parser_check.add_argument('-o', '--outfile')
    parser_check.set_defaults(func=check)
    
    
    args = parser.parse_args()
    
    cfg = getconfig.cfg(args.config)

    if args.projection:
        cfg.projection = args.projection
    

    args.func(args)
