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
import Fitscurve


def sum_onto(a, axis):
    for i in reversed(range(len(a.shape))):
        if axis != i:
            a = a.sum(axis=i)
    return a


class Axis(object):
    def __init__(self, min, max, res, label=None):
        self.res = float(res)
        if round(min / self.res) != min / self.res or round(max / self.res) != max / self.res:
            raise ValueError('min/max must be multiple of res (got min={0}, max={1}, res={2}'.format(min, max, res))
        self.min = min
        self.max = max
        self.label = label

    def __len__(self):
        return int(round((self.max - self.min) / res)) + 1

    def __getitem__(self, index):
        return self.min + index * self.res

    def get_index(self, value):
        if isinstance(x, numbers.Numbers) and not self.min <= value <= self.max:
            raise ValueError('cannot get index: value {0} not in range [{1}, {2}]'.format(value, self.min, self.max))
        elif not (self.min <= value <= self.max).all():
            raise ValueError('cannot get indices, values from [{0}, {1}], axes range [{2}, {3}]'.format(value.min(), value.max(), self.min, self.max))
        return int(round((value - self.min) / self.res))

    def __or__(self, other): # union operation
        if not isinstance(other, Axis):
            return NotImplemented
        if not self.is_compatible(other):
            raise ValueError('cannot unite axes with different resolution/label')
        return self.__class__(min(self.min, other.min), max(self.max, other.max), res)

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
        elif isinstance(other, Axes):
            return self.is_compatible(other) and self.min <= other.min and self.max >= other.max

    def rebound(self, min, max):
        return self.__class__(min, max, self.res, self.label)

	def __repr__(self):
		return '{0.__class__.__name__} {0.label} (min={0.min}, max={0.max}, res={0.res})'.format(self)


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

    def get_masked(self):
        return numpy.ma.array(data=self.photons/self.contributions, mask=(self.contributions == 0))
        
    def __add__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        if not len(self.axes) != len(other.axes) or not all(a.is_compatible(b) for (a, b) in zip(self.axes, other.axes)):
            raise ValueError('cannot add spaces with different dimensionality or resolution')

        new = Space([a | b for (a, b) in zip(self.axes, other.axes)])
        new += self
        new += other
        return new

    def __iadd__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        if not len(self.axes) != len(other.axes) or not all(a.is_compatible(b) for (a, b) in zip(self.axes, other.axes)):
            raise ValueError('cannot add spaces with different dimensionality or resolution')

        if not all(other_ax in self_ax for (self_ax, other_ax) in zip(self.axes, other.axes)):
            return self.__add__(self, other)

        index = tuple(slice(self_ax.get_index(other_ax.min), len(other_ax)+1) for (self_ax, other_ax) in zip(self.axes, other.axes))
        self.photons[index] += other.photons
        self.contributions[index] += other.contributions
        return self

    def trim(self):
        mask = self.contributions > 0
        lims = (numpy.flatnonzero(sum_onto(mask, i)) for (i, ax) in enumerate(self.axes))
        lims = tuple((i.min(), i.max()) for i in lims)
        self.axes = tuple(ax.rebound(ax[min], ax[max]) for ax in self.axes)
        slices = tuple(slice(min, max+1) for (min, max) in lims)
        self.photons = self.photons[slices].copy()
        self.contributions = self.contributions[slices].copy()

    def process_image(self, coordinates, intensity):
        # note: coordinates must be tuple of arrays, not a 2D array
        if len(coordinates) != len(self.axes):
            raise ValueError('dimension mismatch between coordinates and axes')

        indices = numpy.array(tuple(ax.get_index(coord.flatten()) for (ax, coord) in zip(self.axes, coordinates)))
        for i in range(1, len(self.axes)):
            for j in range(0, i):
                indices[j,:] *= len(self.axes[i])
        indices = indices.sum(axis=0).astype(int)

        photons = numpy.bincount(indices, weights=Intensity.flatten())
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


class NotAZaplineError(Exception):
        pass


class Arc(object):
    def __init__(self,spec,scanno,cfg):
        scan = spec.select('{0}.1'.format(scanno))
        self.cfg = cfg
        if not scan.header('S')[0].split()[2] == 'zapline':
            raise NotAZaplineError

        scanheaderC = scan.header('C')
        folder = os.path.split(scanheaderC[0].split(' ')[-1])[-1]
        self.imagefolder = os.path.join(cfg.imagefolder,folder)
        self.scannumber = int(scanheaderC[2].split(' ')[-1])
        self.imagenumber = int(scanheaderC[3].split(' ')[-1])
        self.scanname = scanheaderC[1].split(' ')[-1]
        self.delt,self.theta,self.chi,self.phi,self.mu,self.gam = numpy.array(scan.header('P')[0].split(' ')[1:7],dtype=numpy.float)
        #UB matrix will be installed in new versions of the zapline, until then i keep this here.
    
        if scanno < 405:
            self.UB = numpy.array([2.628602629,0.2730763688,-0.001032444885,1.202301748,2.877587966,-0.001081570571,0.002600281749,0.002198663001,1.54377945])
        else:
            self.UB = numpy.array([2.624469378,0.2632191474,-0.001028869827,1.211297551,2.878506363,-0.001084906521,0.002600359765,0.002198324744,1.54377945])
        self.wavelength = float(scan.header('G')[1].split(' ')[-1])
        self.theta = scan.datacol('th')
        self.mon = scan.datacol('zap_mon')
        self.transm = scan.datacol('zap_transm')
        self.length = numpy.alen(self.theta)
            
        self.gam = self.gam.repeat(self.length)
        self.delt = self.delt.repeat(self.length)
            
        self.imagecode = os.path.join(self.imagefolder,'*{0}_mpx*'.format(self.scanname))
            
        self.transm[-1]=self.transm[-2] #bug in specfile

    def initImdata(self):
        self.buildfilelist()
        self.edf = EdfFile.EdfFile(self.filelist[0])
         
    def buildfilelist(self):
        allfiles =  glob.glob(self.imagecode)
        filelist = list()
        imagedict = {}
        for file in allfiles:        
            filename = os.path.basename(file).split('.')[0]
            scanno, pointno, imageno = filename.split('_')[-3:]
            scanno, pointno, imageno = int(scanno), int(pointno), int(imageno)
            if not scanno in imagedict:
                imagedict[scanno] = {}
            imagedict[scanno][pointno] = file
        filedict = imagedict[self.scannumber]
        points = sorted(filedict.iterkeys())
        self.filelist = [filedict[i] for i in points]
        if len(self.filelist) == 0:
            raise NameError('Empty filelist, check if the specified imagefolder corresponds to the location of the images')
        
    def getImdata(self,n):
        ymask = numpy.asarray(self.cfg.ymask)
        xmask = numpy.asarray(self.cfg.xmask)

        self.data = self.GetData(n)/(self.mon[n]*self.transm[n])
        app = self.cfg.app #angle per pixel (delta,gamma)
        centralpixel = self.cfg.centralpixel #(row,column)=(delta,gamma)
        self.gamma = app[1]*(numpy.arange(self.data.shape[1])-centralpixel[1])+self.gam[n]
        self.delta = app[0]*(numpy.arange(self.data.shape[0])-centralpixel[0])+self.delt[n]
        self.gamma = self.gamma[ymask]
        self.delta = self.delta[xmask]

        R = SixCircle.getHKL(self.wavelength, self.UB, delta=self.delta, theta=self.theta[n],chi=self.chi,phi=self.phi,mu=self.mu,gamma=self.gamma)
#        R.shape = 3,numpy.alen(self.gamma),mp.alen(self.delta)
        H = R[0,:]
        K = R[1,:]
        L = R[2,:]
        roi = self.data[ymask, :]
        roi = roi[:,xmask]
        intensity = roi.flatten()
        return H,K,L, intensity

    def getmean(self,n):
        ymask = numpy.asarray(self.cfg.ymask)
        xmask = numpy.asarray(self.cfg.xmask)        
        self.data = self.GetData(n)#/self.mon[n]
        roi = self.data[ymask, :]
        roi = roi[:,xmask]
        return roi.mean(axis = 0)
            
    def getHKLbounds(self, full=False):
        if full:
            thetas = self.theta
        else:
            thetas = self.theta[0], self.theta[-1]
        
        hkls = []
        for th in thetas:
            hkl = SixCircle.getHKL(self.wavelength, self.UB, delta=self.delta, theta=th,chi=self.chi,phi=self.phi,mu=self.mu,gamma=self.gamma)
            hkls.append(hkl.reshape(3))
        return hkls

    def getbkg(self):
        abkg = numpy.vstack(self.getmean(m) for m in range(self.length)).mean(axis = 0)
        fit = Fitscurve.fitbkg(numpy.arange(abkg.shape[0]), abkg )
        #self.bkg = fit.reshape(1,avg.shape[0]).repeat(im.shape[0],axis = 0)
        return abkg , fit

    def GetData(self,n):
        return self.edf.GetData(n)
        

class hklmesh(Arc):
    def __init__(self,spec,scanno,cfg):
        scan = spec.select('{0}.1'.format(scanno))
        self.cfg = cfg
        if not scan.header('S')[0].split()[2] == 'hklmesh':
            raise NotAZaplineError
        
        UCCD = os.path.split(scan.header('UCCD')[0].split(' ')[-1])
        folder = os.path.split(UCCD[0])[-1]
        self.scanname = UCCD[-1].split('_')[0]
        self.imagefolder = os.path.join(cfg.imagefolder,folder)
        self.scannumber = scanno

        self.chi = 0
        self.phi = 0
                
        self.theta = scan.datacol('thcnt')
        self.gam = scan.datacol('gamcnt')
        self.delt = scan.datacol('delcnt')
        self.mu = float(scan.header('P')[0].split(' ')[5])
                
        self.UB = numpy.array(scan.header('G')[2].split(' ')[-9:],dtype=numpy.float)
        self.wavelength = float(scan.header('G')[1].split(' ')[-1])
        
        self.mon = scan.datacol('mon')
        self.transm = scan.datacol('transm')
        self.length = numpy.alen(self.theta)

        self.imagecode = os.path.join(self.imagefolder,'*{0}*'.format(self.scanname))

    def initImdata(self):
        self.buildfilelist()

    def GetData(self,n):        
        edf = EdfFile.EdfFile(self.filelist[n])
        return edf.GetData(0)



def process(scanno):
    mesh = Space.fromcfg(cfg)
    try:
        if cfg.hklmesh:
            a = hklmesh(spec, scanno,cfg)
        else:
            a = Arc(spec, scanno,cfg)
    except NotAZaplineError:
        return None
    print scanno
    a.initImdata()
    for m in range(a.length):
        H, K, L, intensity = a.getImdata(m)
        mesh.process_image((H, K, L), Intensity)
    return mesh


def makeplot(space, args):
    import matplotlib.pyplot as pyplot
    import matplotlib.colors

    clipping = 0.02
   
    mesh = space.get_masked()
    data = numpy.log(mesh[...,0])
    compresseddata = data.compressed()
    chop = int(round(compresseddata.size * clipping))
    clip = sorted(compresseddata)[chop:-chop]
    vmin, vmax = clip[0], clip[-1]
    
    pyplot.figure(figsize=(12,9))
    #pyplot.imshow(space.contributions.reshape((space.Hcount, space.Kcount, space.Lcount), order='C')[:,:,0].transpose(), origin='lower', extent=(space.set['minH'], space.set['maxH'], space.set['minK'], space.set['maxK']), aspect='auto')#, norm=matplotlib.colors.Normalize(vmin, vmax))

    pyplot.imshow(data.transpose(), origin='lower', extent=(space.cfg.Hmin, space.cfg.Hmax, space.cfg.Kmin, space.cfg.Kmax), aspect='auto', norm=matplotlib.colors.Normalize(vmin, vmax))
    
    
    #pyplot.imshow(data.transpose())
    
    #xgrid, ygrid = numpy.meshgrid(numpy.arange(data.shape[0]+1), numpy.arange(data.shape[1]+1))
    #ax=pyplot.subplot(111)
    #ax.pcolorfast(numpy.sin(60. /180 * numpy.pi) * xgrid+numpy.cos(60. /180 * numpy.pi) * ygrid, ygrid , data.transpose(),norm=matplotlib.colors.Normalize(vmin, vmax))
    
    pyplot.xlabel('H')
    pyplot.ylabel('K')
    #pyplot.suptitle() TODO
    pyplot.colorbar()
    #ax.set_xlim(200,1500)
    if args.s:
        if args.savefile != None:
            pyplot.savefig(args.savefile)
            print 'saved at {0}'.format(args.savefile)
        else:
            pyplot.savefig('{0}.png'.format(os.path.splitext(args.outfile)[0]))
            print 'saved at {0}.png'.format(os.path.splitext(args.outfile)[0])
    else:
        pyplot.show()



def mfinal(filename, first, last=None):
    base, ext = os.path.splitext(filename)
    if last is None:
        return ('{0}_{2}{1}').format(base,ext,first)
    else:
        return ('{0}_{2}-{3}{1}').format(base,ext,first,last)


def detect_hkllimits(cfg, firstscan, lastscan):
    spec = specfilewrapper.Specfile(cfg.specfile)

    arcs = []
    for scanno in range(firstscan, lastscan+1):
        try:
            a = Arc(spec, scanno,cfg)
        except NotAZaplineError:
            continue
        arcs.append(a)

    hkls = []
    for i, a in enumerate(arcs):
        hkls.extend(a.getHKLbounds(i == 0 or (i + 1) == len(arcs)))

    hkls = numpy.array(hkls)
    return hkls.min(axis=0), hkls.max(axis=0)


def wait_for_files(filelist):
    i = 0
    while filelist:
        if os.path.exists(filelist[i]):
            yield filelist.pop(i)
            i = i % len(arcs)
        else:
            time.sleep(5)
            i = (i + 1) % len(arcs)


def wait_for_file(filename):
    return bool(list(wait_for_files([filename])))


if __name__ == "__main__":    
    
    def run(*command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, unused_err = process.communicate()
        retcode = process.poll()
        return retcode, output


    def oarsub(*args):
        scriptname = './blisspython /data/id03/inhouse/2012/Sep12/si2515/iVoxOar/iVoxOar.py '
        command = '{0} {1}'.format(scriptname, ' '.join(args))
        ret, output = run('oarsub', command)
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
            job, status = output.split(':')
            return status.strip()
        else:
            return 'Unknown'


    def oarwait(jobs):
        i = 0
        while jobs:
            status = oarstat(jobs[i])
            if status == 'Running' or status == 'Waiting' or status == 'Unknown':
                i += 1
                time.sleep(5)
            else: # assume status == 'Finishing' or 'Terminated' but don't wait on something unknown
                del jobs[i]
                print '{0} {1} jobs to go'.format(time.ctime(), len(jobs))
            if i == len(jobs):
                i = 0


    def cluster(args):
        prefix = 'iVoxOar-{0:x}'.format(random.randint(0, 2**32-1)) 
        jobs = []
        
        if firstscan == lastscan:
            jobs.append(oarsub('--config', args.config, '_part', '--trim', '-o', mfinal(cfg.outfile, args.firstscan), str(scanno)))
            print 'submitted 1 job, waiting...'
            oar.wait(jobs)
            print 'done'
            return

        parts = []
        for scanno in range(args.firstscan, args.lastscan+1):
            part = '{0}/{1}-part-{2}.zpi'.format(args.tmpdir, prefix, scanno)
            jobs.append(oarsub('--config', args.config,'_part','-o', part, str(scanno)))
            parts.append(part)

        count = args.lastscan - args.firstscan + 1
        chunkcount = int(numpy.ceil(float(count) / args.chunksize))
        if chunkcount == 1:
            jobs.append(oarsub('--config', args.config,'_sum', '--trim', '--delete', '-o', mfinal(cfg.outfile,args.firstscan,args.lastscan), *parts))
        else:
            chunksize = int(numpy.ceil(float(count) / chunkcount))
            chunks = []
            for i in range(chunkcount):
                chunk = '{0}/{1}-chunk-{2}.zpi'.format(args.tmpdir, prefix, i+1)
                jobs.append(oarsub('--config', args.config,'_sum', '--delete', '-o', chunk, *parts[i*chunksize:(i+1)*chunksize]))
                chunks.append(chunk)
             
            jobs.append(oarsub('--config', args.config,'_sum', '--trim', '--delete', '-o', mfinal(cfg.outfile,args.firstscan,args.lastscan), *chunks))
            print 'submitted final job, waiting...'
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


    def sum(args):
        globalspace = Space.fromcfg(cfg)

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
        
        scanlist = range(args.firstscan, args.lastscan+1)
        globalspace = Space.fromcfg(cfg)
     
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
        globalspace.tofile(mfinal(cfg.outfile, args.firstscan, args.lastscan))

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
    
    def test(args):
        scanlist = range(args.firstscan, args.lastscan+1)
        spec = specfilewrapper.Specfile(cfg.specfile)
        for n in scanlist:
            print n
            a = Arc(spec, n,cfg)
            a.initImdata()
            abkg,fit = a.getbkg()
            numpy.savetxt('{0}-bkg.txt'.format(str(n)) ,abkg)
            fit = Fitscurve.fitbkg(numpy.arange(abkg.shape[0]), abkg )
            #bkg = fit.reshape(1,avg.shape[0]).repeat(im.shape[0],axis = 0)
            pyplot.figure()
            #pyplot.figure(figsize = (8,10))
            #pyplot.subplot(221)
            #pyplot.imshow(im-bkg)
            #pyplot.axis('off')
            #pyplot.colorbar()
            #pyplot.subplot(222)
            #pyplot.imshow(im)
            #pyplot.axis('off')
            #pyplot.colorbar()
            pyplot.subplot(111)
            pyplot.plot(abkg,'wo')
            pyplot.plot(fit,'r')
            pyplot.savefig('{0}-bkg.pdf'.format(str(n)))
            pyplot.close()

    parser = argparse.ArgumentParser(prog='iVoxOar')
    parser.add_argument('--config',default='./config')
    parser.add_argument('--wait', action='store_true', help='wait for input files to appear')
    parser.add_argument('--hklmesh', action='store_true')
    subparsers = parser.add_subparsers()

    parser_cluster = subparsers.add_parser('cluster')
    parser_cluster.add_argument('firstscan', type=int)
    parser_cluster.add_argument('lastscan', type=int, default=None, nargs='?')
    parser_cluster.add_argument('-o', '--outfile')
    parser_cluster.add_argument('--tmpdir', default='.')
    parser_cluster.add_argument('--chunksize', default=20, type=int)
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
    parser_sum.set_defaults(func=sum)

    parser_local = subparsers.add_parser('local')
    parser_local.add_argument('firstscan', type=int)
    parser_local.add_argument('lastscan', type=int)
    parser_local.add_argument('-o', '--outfile')
    parser_local.add_argument('-p', '--plot', nargs='?', const=True)
    parser_local.add_argument('-m', '--multiprocessing', action='store_true')
    parser_local.set_defaults(func=local)

    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('outfile')
    parser_plot.add_argument('-s',action='store_true')
    parser_plot.add_argument('--savefile')
    parser_plot.set_defaults(func=plot)

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('firstscan', type=int)
    parser_test.add_argument('lastscan', type=int)
    parser_test.add_argument('--outfile', default = 'test.pdf')
    
    parser_test.set_defaults(func=test)

    args = parser.parse_args()

    if args.lastscan is None:
        args.lastscan = args.firstscan
    
    cfg = getconfig.cfg(args.config)

    if args.outfile:
        cfg.outfile = args.outfile
    
    cfg.__dict__['hklmesh'] = args.hklmesh

    args.func(args)
