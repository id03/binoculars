# -*- encoding: utf-8 -*-
'''
 This file is part of the binoculars project.

  The BINoculars library is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  The BINoculars library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the hkl library.  If not, see <http://www.gnu.org/licenses/>.

  Copyright (C) 2012-2015 European Synchrotron Radiation Facility
                          Grenoble, France

  Authors: Willem Onderwaater <onderwaa@esrf.fr>
           Jonathan Rawle

'''
import sys
import os
import itertools
import numpy
import time
import math
import json
from scipy.misc import imread
import scisoftpy as dnp

from scisoftpy import sin,cos

from .. import backend, errors, util

PY3 = sys.version_info > (3,)
if PY3:
    from functools import reduce
else:
    from itertools import izip as zip


class HKLProjection(backend.ProjectionBase):
    # scalars: mu, theta, [chi, phi, "omitted"] delta, gamR, gamT, ty, wavelength
    # 3x3 matrix: UB
    def project(self, energy, UB, pixels, gamma, delta, omega, alpha, nu):
        # put the detector at the right position

        dx,dy,dz = pixels

        # convert angles to radians
        gamma, delta, alpha, omega, nu = numpy.radians((gamma, delta, alpha, omega, nu))

        RGam = numpy.matrix([[1,0,0],[0,cos(gamma),-sin(gamma)],[0,sin(gamma),cos(gamma)]])
        RDel = (numpy.matrix([[cos(delta),-sin(delta),0],[sin(delta),cos(delta),0],[0,0,1]])).getI()
        RNu = numpy.matrix([[cos(nu),0,sin(nu)],[0,1,0],[-sin(nu),0,cos(nu)]])

        # calculate Cartesian coordinates for each pixel using clever matrix stuff
        M = numpy.mat(numpy.concatenate((dx.flatten(0), dy.flatten(0), dz.flatten(0))).reshape(3,dx.shape[0]*dx.shape[1])) 
        XYZp = RGam * RDel * RNu * M
        xp = dnp.array(XYZp[0]).reshape(dx.shape)
        yp = dnp.array(XYZp[1]).reshape(dy.shape)
        zp = dnp.array(XYZp[2]).reshape(dz.shape)        
        # don't bother with the part about slits...

        # Calculate effective gamma and delta for each pixel
        d_ds = dnp.sqrt(xp**2 + yp**2 + zp**2)
        Gam = dnp.arctan2(zp, yp)
        Del = -1 * dnp.arcsin(-xp/d_ds)
    
        # wavenumber
        k = 2 * math.pi / 12.398 * energy

        # Define the needed matrices. The notation follows the article by Bunk &
        # Nielsen. J.Appl.Cryst. (2004) 37, 216-222.        
        M1 = k * numpy.matrix(cos(omega) * sin(Del) - sin(omega) * (cos(alpha) * (cos(Gam) * cos(Del)-1) + sin(alpha) * sin(Gam) * cos(Del)))
        M2 = k * numpy.matrix(sin(omega) * sin(Del) + cos(omega) * (cos(alpha) * (cos(Gam) * cos(Del)-1) + sin(alpha) * sin(Gam) * cos(Del)))
        M3 = k * numpy.matrix(-sin(alpha) * (cos(Gam) * cos(Del)-1) + cos(alpha) * sin(Gam) * cos(Del))

        # invert UB matrix
        UBi = numpy.matrix(UB).getI()
    
        # calculate HKL
        H = UBi[0,0]*M1 + UBi[0,1]*M2 + UBi[0,2]*M3
        K = UBi[1,0]*M1 + UBi[1,1]*M2 + UBi[1,2]*M3
        L = UBi[2,0]*M1 + UBi[2,1]*M2 + UBi[2,2]*M3

        return (H, K, L)

    def get_axis_labels(self):
        return 'H', 'K', 'L'

class GammaDelta(HKLProjection):  # just passing on the coordinates, makes it easy to accurately test the theta correction
    def project(self, beamenergy, UB, gamma, delta, omega, alpha):
        delta, gamma = numpy.meshgrid(delta, gamma)
        return (gamma, delta)

    def get_axis_labels(self):
        return 'Gamma', 'Delta'

class pixels(backend.ProjectionBase):
    def project(self, beamenergy, UB, gamma, delta, omega, alpha):
        y, x = numpy.mgrid[slice(None, gamma.shape[0]), slice(None, delta.shape[0])]
        return (y, x)

    def get_axis_labels(self):
        return 'y', 'x'

class IO7Input(backend.InputBase):
    # OFFICIAL API

    dbg_scanno = None
    dbg_pointno = None

    def generate_jobs(self, command):
        scans = util.parse_multi_range(','.join(command).replace(' ', ','))
        if not len(scans):
            sys.stderr.write('error: no scans selected, nothing to do\n')
        for scanno in scans:
            util.status('processing scan {0}...'.format(scanno))
            if self.config.pr:
                pointcount = self.config.pr[1] - self.config.pr[0] + 1
                start = self.config.pr[0]
            else:
                scan = self.get_scan(scanno)
                pointcount = len(scan.file)
                start = 0
            if pointcount > self.config.target_weight * 1.4:
                for s in util.chunk_slicer(pointcount, self.config.target_weight):
                    yield backend.Job(scan=scanno, firstpoint=start+s.start, lastpoint=start+s.stop-1, weight=s.stop-s.start)
            else:
                yield backend.Job(scan=scanno, firstpoint=start, lastpoint=start+pointcount-1, weight=pointcount)

    def process_job(self, job):
        super(IO7Input, self).process_job(job)
        scan = self.get_scan(job.scan)
        self.metadict = dict()
        try:
            scanparams = self.get_scan_params(scan)  # wavelength, UB
            pointparams = self.get_point_params(scan, job.firstpoint, job.lastpoint)  # 2D array of diffractometer angles + mon + transm
            images = self.get_images(scan, job.firstpoint, job.lastpoint)  # iterator!
            for pp, image in zip(pointparams, images):
                yield self.process_image(scan, scanparams, pp, image)
            util.statuseol()
        except Exception as exc:
            exc.args = errors.addmessage(exc.args, ', An error occured for scan {0} at point {1}. See above for more information'.format(self.dbg_scanno, self.dbg_pointno))
            raise
        self.metadata.add_section('id7_backend', self.metadict)

    def get_scan_params(self, scan):
        energy = scan.metadata.dcm1energy
        UB = numpy.array(json.loads(scan.metadata.diffcalc_ub))

        self.metadict['UB'] = UB
        self.metadict['energy'] = energy

        return energy, UB

    def get_point_params(self, scan, first, last):
        sl = slice(first, last+1)

        GAM, DEL, OMG, CHI, PHI, ALF, MON, TRANSM = list(range(8))
        params = numpy.zeros((last - first + 1, 8))  # gamma delta theta chi phi mu mon transm
        params[:, CHI] = 0
        params[:, PHI] = 0

        params[:, OMG] = scan['omega'][sl]
        params[:, GAM] = scan['gamma'][sl]
        params[:, DEL] = scan['delta'][sl]
        params[:, ALF] = scan['alpha'][sl]

        return params

    def get_images(self, scan, first, last, dry_run=False):
        sl = slice(first, last+1)
        for fn in scan.file[sl]:
            yield imread(self.get_imagefilename(fn))

    def get_imagefilename(self, filename):
        if self.config.imagefolder is None:
            if os.path.exists(filename):
                return filename
            else:
                raise errors.ConfigError("image filename specified in the datafile does not exist '{0}'".format(filename))           
        else:
            head, tail = os.path.split(filename)
            folders = head.split('/')
            try:
                imagefolder = self.config.imagefolder.format(folders=folders, rfolders=list(reversed(folders)))
            except Exception as e:
                raise errors.ConfigError("invalid 'imagefolder' specification '{0}': {1}".format(self.config.imagefolder, e))
            else:
                if not os.path.exists(imagefolder):
                    raise errors.ConfigError("invalid 'imagefolder' specification '{0}'. Path {1} does not exist".format(self.config.imagefolder, imagefolder))
            fn = os.path.join(imagefolder, tail)    
            if os.path.exists(fn):
                return fn
            else:
                raise errors.ConfigError("image filename does not exist '{0}', either imagefolder is wrongly specified or image file does not exist".format(filename))    

    def parse_config(self, config):
        super(IO7Input, self).parse_config(config)
        self.config.xmask = util.parse_multi_range(config.pop('xmask', None))#Optional, select a subset of the image range in the x direction. all by default
        self.config.ymask = util.parse_multi_range(config.pop('ymask', None))#Optional, select a subset of the image range in the y direction. all by default
        self.config.datafilefolder = config.pop('datafilefolder')#Folder with the datafiles
        self.config.imagefolder = config.pop('imagefolder', None)  # Optional, takes datafile folder tag by default
        self.config.pr = config.pop('pr', None) #Optional, all range by default
        if self.config.xmask is None:
            self.config.xmask = slice(None)
        if self.config.ymask is None:
            self.config.ymask = slice(None)
        if self.config.pr:
            self.config.pr = util.parse_tuple(self.config.pr, length=2, type=int)
        self.config.centralpixel = util.parse_tuple(config.pop('centralpixel'), length=2, type=int)  #x,y
        self.config.maskmatrix = config.pop('maskmatrix', None)#Optional, if supplied pixels where the mask is 0 will be removed
        self.config.pixelsize = util.parse_tuple(config.pop('pixelsize'), length=2, type=float)  # pixel size x/y (mm) (same dimension as sdd)

    def get_destination_options(self, command):
        if not command:
            return False
        command = ','.join(command).replace(' ', ',')
        scans = util.parse_multi_range(command)
        return dict(first=min(scans), last=max(scans), range=','.join(str(scan) for scan in scans))

    # CONVENIENCE FUNCTIONS
    def get_scan(self, scanno):
        filename = os.path.join(self.config.datafilefolder, str(scanno) + '.dat')
        if not os.path.exists(filename):
	        raise errors.ConfigError('datafile filename does not exist: {0}'.format(filename))
	return dnp.io.load(filename)

    @staticmethod
    def apply_mask(data, xmask, ymask):
        roi = data[ymask, :]
        return roi[:, xmask]

class EH2(IO7Input):
    def parse_config(self, config):
        super(IO7Input, self).parse_config(config)
        self.config.sdd = float(config.pop('sdd'), None)#Sample to detector distance (mm)
        if self.config.sdd is not None:
            self.config.sdd = float(self.config.sdd)


    def process_image(self, scan, scanparams, pointparams, image):
        gamma, delta, omega, chi, phi, alpha, mon, transm = pointparams#GAM, DEL, OMG, CHI, PHI, ALF, MON, TRANSM
        energy, UB = scanparams

        weights = numpy.ones_like(image)

        util.status('{4}| gamma: {0}, delta: {1}, omega: {2}, mu: {3}'.format(gamma, delta, omega, alpha, time.ctime(time.time())))

        # pixels to angles
        pixelsize = numpy.array(self.config.pixelsize)
            
        if self.config.sdd is None:
            sdd = scan.metadata.diff1detdist
        else:
            sdd = self.config.sdd

        nu = scan.metadata.diff2prot

        centralpixel = self.config.centralpixel  # (column, row) = (delta, gamma)

        dz = (numpy.indices(image.shape)[1] - centralpixel[1]) * pixelsize[1]
        dx = (numpy.indices(image.shape)[0] - centralpixel[0]) * pixelsize[0]
        dy = numpy.ones(image.shape) * sdd

        # masking
        if self.config.maskmatrix is not None:
            if self.config.maskmatrix.shape != data.shape:
                raise errors.BackendError('The mask matrix does not have the same shape as the images')
            weights *= self.config.maskmatrix

        intensity = self.apply_mask(image, self.config.xmask, self.config.ymask)
        weights = self.apply_mask(weights, self.config.xmask, self.config.ymask)
        dx = self.apply_mask(dx, self.config.xmask, self.config.ymask)
        dy = self.apply_mask(dy, self.config.xmask, self.config.ymask)
        dz = self.apply_mask(dz, self.config.xmask, self.config.ymask)


        #X,Y = numpy.meshgrid(x,y)
        #Z = numpy.ones(X.shape) * sdd

        pixels = dx,dy,dz

        return intensity, weights, (energy, UB, pixels, gamma, delta, omega, alpha, nu)


class EH1(IO7Input):
    def parse_config(self, config):
        super(EH1, self).parse_config(config)
        self.config.sdd = float(config.pop('sdd'))#Sample to detector distance (mm)


    def process_image(self, scan, scanparams, pointparams, image):
        gamma, delta, omega, chi, phi, alpha, mon, transm = pointparams#GAM, DEL, OMG, CHI, PHI, ALF, MON, TRANSM
        energy, UB = scanparams

        weights = numpy.ones_like(image)

        util.status('{4}| gamma: {0}, delta: {1}, omega: {2}, mu: {3}'.format(gamma, delta, omega, alpha, time.ctime(time.time())))

        # pixels to angles
        pixelsize = numpy.array(self.config.pixelsize)

        sdd = self.config.sdd

        nu = scan.metadata.diff1prot

        centralpixel = self.config.centralpixel  # (column, row) = (delta, gamma)

        dz = (numpy.indices(image.shape)[1] - centralpixel[1]) * pixelsize[1]
        dx = (numpy.indices(image.shape)[0] - centralpixel[0]) * pixelsize[0]
        dy = numpy.ones(image.shape) * sdd

        # masking
        if self.config.maskmatrix is not None:
            if self.config.maskmatrix.shape != data.shape:
                raise errors.BackendError('The mask matrix does not have the same shape as the images')
            weights *= self.config.maskmatrix

        intensity = self.apply_mask(image, self.config.xmask, self.config.ymask)
        weights = self.apply_mask(weights, self.config.xmask, self.config.ymask)
        dx = self.apply_mask(dx, self.config.xmask, self.config.ymask)
        dy = self.apply_mask(dy, self.config.xmask, self.config.ymask)
        dz = self.apply_mask(dz, self.config.xmask, self.config.ymask)

        pixels = dx,dy,dz

        return intensity, weights, (energy, UB, pixels, gamma, delta, omega, alpha, nu)


def load_matrix(filename):
    if filename == None:
        return None
    if os.path.exists(filename):
        ext = os.path.splitext(filename)[-1]
        if ext == '.txt':
            return numpy.array(numpy.loadtxt(filename), dtype = numpy.bool)
        elif ext == '.npy':
            return numpy.array(numpy.load(filename), dtype = numpy.bool)
        else:
            raise ValueError('unknown extension {0}, unable to load matrix!\n'.format(ext))
    else:
       raise IOError('filename: {0} does not exist. Can not load matrix'.format(filename))
