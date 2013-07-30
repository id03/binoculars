# Written by Willem Onderwaater and Sander Roobol as part of a collaboration
# between the ID03 beamline at the European Synchrotron Radiation Facility and
# the Interface Physics group at Leiden University.

import sys
import os
import itertools
import glob
import numpy

from PyMca import SixCircle
from PyMca import specfilewrapper
try:
	import EdfFile # allow user to provide a local version of EdfFile if the PyMca one is too old
except:
	from PyMca import EdfFile

from . import config


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
            raise config.ConfigError('UB')
        
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
            raise config.ConfigError('UB')
        
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
