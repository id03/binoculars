import numpy
import os
import time
import subprocess
import random
import glob
from PyMca import SixCircle
from PyMca import specfilewrapper
import EdfFile
import getconfig

import Fitscurve
import matplotlib.pyplot as pyplot
import matplotlib.colors

import multiprocessing
import cPickle as pickle
import itertools
import sys
import gzip
import argparse
import copy


# example usage on Oar:
# ssh to onderwaa@nice
# cd to iVoxOar
# ./blisspython /data/id03/inhouse/2012/Sep12/si2515/iVox/iVoxOar.py cluster -o OUTPUT.zpi /data/id03/inhouse/2012/Sep12/si2515/iVox/test.spec 217 287
# to plot, run (not on nice): python iVoxOar.py plot file.zpi


class Space(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.Hmin, self.Hmax, self.Hres = cfg.Hmin, cfg.Hmax, cfg.Hres
        self.Kmin, self.Kmax, self.Kres = cfg.Kmin, cfg.Kmax, cfg.Kres
        self.Lmin, self.Lmax, self.Lres = cfg.Lmin, cfg.Lmax, cfg.Lres
        
        self.Hcount = int(round((self.Hmax-self.Hmin)/Hres))+1
        self.Kcount = int(round((self.Kmax-self.Kmin)/Kres))+1
        self.Lcount = int(round((self.Lmax-self.Lmin)/Lres))+1

        self.photons = numpy.zeros((self.Hcount, self.Kcount, self.Lcount), order='C')
        self.contributions = numpy.zeros(self.photons.shape, dtype=numpy.uint32, order='C')

    def get_masked(self):
        return numpy.ma.array(data=self.photons/self.contributions, mask=(self.contributions == 0))
        
    def __add__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        if self.Hres != other.Hres or self.Kres != other.Kres or self.Lres != other.Lres:
            raise ValueError('cannot add spaces with different H/K/L resolution')

        newcfg = copy.copy(cfg)
        newcfg.Hmin = min(self.Hmin, other.Hmin)
        newcfg.Hmax = max(self.Hmax, other.Hmax)
        newcfg.Kmin = min(self.Kmin, other.Kmin)
        newcfg.Kmax = max(self.Kmax, other.Kmax)
        newcfg.Lmin = min(self.Lmin, other.Lmin)
        newcfg.Lmax = max(self.Lmax, other.Lmax)

        new = Space(newcfg)
        new += self
        new += other
        return new

    def __iadd__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        if self.Hres != other.Hres or self.Kres != other.Kres or self.Lres != other.Lres:
            raise ValueError('cannot add spaces with different H/K/L resolution')

        if self.Hmin > other.Hmin or self.Hmax < other.Hmax or self.Kmin > other.Kmin or self.Kmax < other.Kmax or self.Lmin > other.Lmin or self.Lmax < other.Lmax:
            return self.__add__(self, other)

        Hi = int(round((other.Hmin - self.Hmin) / self.Hres))
        Ki = int(round((other.Kmin - self.Kmin) / self.Kres))
        Li = int(round((other.Lmin - self.Lmin) / self.Lres))
        self.photons[Hi:Hi+other.shape[0], Ki:Ki+other.shape[1], Li:Li+other.shape[2]] += other.photons
        self.contributions[Hi:Hi+other.shape[0], Ki:Ki+other.shape[1], Li:Li+other.shape[2]] += other.contributions
        return self

    def trim(self):
        mask = self.contributions > 0
		sum2 = mask.sum(axis=2)

        Hlims = numpy.flatnonzero(sum2.sum(axis=1))
        Hmini, Hmaxi = Hlims.min(), Hlims.max()
        self.Hmin = self.Hmin + self.Hres * Hmini
        self.Hmax = self.Hmin + self.Hres * Hmaxi

        Klims = numpy.flatnonzero(sum2.sum(axis=0))
        Kmini, Kmaxi = Klims.min(), Klims.max()
        self.Kmin = self.Kmin + self.Kres * Kmini
        self.Kmax = self.Kmin + self.Kres * Kmaxi

        Llims = numpy.flatnonzero(mask.sum(axis=1).sum(axis=0))
        Lmini, Lmaxi = Llims.min(), Llims.max()
        self.Lmin = self.Lmin + self.Lres * Lmini
        self.Lmax = self.Lmin + self.Lres * Lmaxi

        self.photons = self.photons[Hmini:Hmaxi+1, Kmini:Kmaxi+1, Lmini:Lmaxi+1].copy()
        self.contributions = self.contributions[Hmini:Hmaxi+1, Kmini:Kmaxi+1, Lmini:Lmaxi+1].copy()

    def process_image(self, H, K, L, intensity):
        hkl = (numpy.round((H-self.Hmin)/self.Hres)*self.Kcount*self.Lcount + numpy.round((K-self.Kmin)/self.Kres)*self.Lcount + numpy.round((L-self.Lmin)/self.Lres)).astype(int)

        photons = numpy.bincount(hkl.flatten(), weights=Intensity.flatten())
        contributions = numpy.bincount(hkl.flatten())

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
    mesh = Space(cfg)
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
        mesh.process_image(H, K, L, Intensity)
    return mesh


def makeplot(space, args):
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



def mfinal(filename,first,last):
    base, ext = os.path.splitext(filename)
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
        parts = []
        for scanno in range(args.firstscan, args.lastscan+1):
            part = '{0}/{1}-part-{2}.zpi'.format(args.tmpdir, prefix, scanno)
            jobs.append(oarsub('--config', args.config,'_part','-o', part, str(scanno)))
            parts.append(part)
        print 'submitted {0} jobs, waiting...'.format(len(jobs))
        oarwait(jobs)

        count = args.lastscan - args.firstscan + 1
        chunkcount = int(numpy.ceil(float(count) / args.chunksize))
        if chunkcount == 1:
            job = oarsub('--config', args.config,'_sum', '--trim', '--delete', '-o', mfinal(cfg.outfile,args.firstscan,args.lastscan), *parts)
            print 'submitted final job, waiting...'
            oarwait([job])
        else:
            chunksize = int(numpy.ceil(float(count) / chunkcount))
            jobs = []
            chunks = []
            for i in range(chunkcount):
                chunk = '{0}/{1}-chunk-{2}.zpi'.format(args.tmpdir, prefix, i+1)
                jobs.append(oarsub('--config', args.config,'_sum', '--delete', '-o', chunk, *parts[i*chunksize:(i+1)*chunksize]))
                chunks.append(chunk)
            print 'submitted {0} jobs, waiting...'.format(len(jobs))
            oarwait(jobs)
                       
            job = oarsub('--config', args.config,'_sum', '--trim', '--delete', '-o', mfinal(cfg.outfile,args.firstscan,args.lastscan), *chunks)
            print 'submitted final job, waiting...'
            oarwait([job])
        print 'done!'


    def part(args):
        global spec
        spec = specfilewrapper.Specfile(cfg.specfile)
        space = process(args.scan)
        
        space.tofile(args.outfile)


    def sum(args):
        globalspace = Space(cfg)

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
        globalspace = Space(cfg)
     
        if args.multiprocessing:
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
    parser_cluster.add_argument('lastscan', type=int)
    parser_cluster.add_argument('-o', '--outfile')
    parser_cluster.add_argument('--tmpdir', default='.')
    parser_cluster.add_argument('--chunksize', default=20, type=int)
    parser_cluster.set_defaults(func=cluster)

    parser_part = subparsers.add_parser('_part')
    parser_part.add_argument('scan', type=int)
    parser_part.add_argument('-o', '--outfile',required=True)
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
    
    cfg = getconfig.cfg(args.config)

    if args.outfile:
        cfg.outfile = args.outfile
    
    cfg.__dict__['hklmesh'] = args.hklmesh

    args.func(args)
