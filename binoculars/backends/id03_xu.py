"""
BINocular backend for beamline ID03:EH2 
This backend should serve as a basic example of a backend based on
xrayutilities [1]. It still uses PyMCA for parsing the spec,edf files.
The 'original' ID03 backend was used as a template.

Created on 2014-10-16

[1] http://xrayutilities.sourceforge.net/

author: Dominik Kriegner (dominik.kriegner@gmail.com)
"""

import sys
import os
import glob
import numpy

import xrayutilities as xu
from PyMca import specfile

#python3 support
PY3 = sys.version_info > (3,)
if PY3:
    pass
else:
    from itertools import izip as zip

try:
    from PyMca import specfilewrapper, EdfFile
except ImportError:
    from PyMca.PyMcaIO import specfilewrapper, EdfFile

from .. import backend, errors, util

class HKLProjection(backend.ProjectionBase):
    # scalars: mu, theta, [chi, phi, "omitted"] delta, gamR, gamT, ty, wavelength
    # 3x3 matrix: UB
    def project(self, mu, theta, delta, gamR, gamT, ty, wavelength, UB, qconv):
        qconv.wavelength = wavelength
        h, k, l = qconv.area(mu, theta, 
                             mu, delta, ty, gamT, gamR,
                             UB=UB.reshape((3,3)))
        return (h, k, l)

    def get_axis_labels(self):
        return 'H', 'K', 'L'

class HKProjection(HKLProjection):
    def project(self, mu, theta, delta, gamR, gamT, ty, wavelength, UB, qconv):
        H, K, L = super(HKProjection, self).project(mu, theta, delta, gamR, gamT, ty, wavelength, UB, qconv)
        return (H, K)

    def get_axis_labels(self):
        return 'H', 'K'

class QProjection(backend.ProjectionBase):
    def project(self, mu, theta, delta, gamR, gamT, ty, wavelength, UB, qconv):
        qconv.wavelength = wavelength
        qx, qy, qz = qconv.area(mu, theta, 
                                mu, delta, ty, gamT, gamR,
                                UB=numpy.identity(3))
        return (qx, qy, qz)

    def get_axis_labels(self):
        return 'qx', 'qy', 'qz'

class ID03Input(backend.InputBase):
    # OFFICIAL API
    def generate_jobs(self, command):
        scans = util.parse_multi_range(','.join(command).replace(' ', ','))
        if not len(scans):
            sys.stderr.write('error: no scans selected, nothing to do\n')
        for scanno in scans:
            scan = self.get_scan(scanno)
            try:
                pointcount = scan.lines()
            except specfile.error: # no points
                continue
            next(self.get_images(scan, 0, pointcount-1, dry_run=True))# dryrun

            if self.config.target_weight and pointcount > self.config.target_weight * 1.4:
                for s in util.chunk_slicer(pointcount, self.config.target_weight):
                    yield backend.Job(scan=scanno, firstpoint=s.start, lastpoint=s.stop-1, weight=s.stop-s.start)
            else:
                yield backend.Job(scan=scanno, firstpoint=0, lastpoint=pointcount-1, weight=pointcount)

    def process_job(self, job):
        super(ID03Input, self).process_job(job)
        scan = self.get_scan(job.scan)
        
        scanparams = self.get_scan_params(scan) # wavelength, UB
        pointparams = self.get_point_params(scan, job.firstpoint, job.lastpoint) # 1D array of diffractometer angles + mon + transm
        images = self.get_images(scan, job.firstpoint, job.lastpoint) # iterator!
        
        for pp, image in zip(pointparams, images):
            yield self.process_image(scanparams, pp, image)

    def parse_config(self, config):
        super(ID03Input, self).parse_config(config)
        self.config.xmask = util.parse_multi_range(config.pop('xmask'))
        self.config.ymask = util.parse_multi_range(config.pop('ymask'))
        self.config.specfile = config.pop('specfile')
        self.config.imagefolder = config.pop('imagefolder', None)
        self.config.UB = config.pop('ub', None)
        if self.config.UB:
            self.config.UB = util.parse_tuple(self.config.UB, length=9, type=float)
        self.config.sdd = float(config.pop('sdd'))
        self.config.pixelsize = util.parse_tuple(config.pop('pixelsize'), length=2, type=float)
        self.config.centralpixel = util.parse_tuple(config.pop('centralpixel'), length=2, type=int)

    def get_destination_options(self, command):
        if not command:
            return False
        command = ','.join(command).replace(' ', ',')
        scans = util.parse_multi_range(command)
        return dict(first=min(scans), last=max(scans), range=','.join(command))

    # CONVENIENCE FUNCTIONS
    _spec = None
    def get_scan(self, scannumber):
        if self._spec is None:
            self._spec = specfilewrapper.Specfile(self.config.specfile) 
        return self._spec.select('{0}.1'.format(scannumber))

    def find_edfs(self, pattern, scanno):
        files = glob.glob(pattern)
        ret = {}
        for file in files:
            try:
                filename = os.path.basename(file).split('.')[0]
                scan, point, image = filename.split('_')[-3:]
                scan, point, image = int(scan), int(point), int(image)
                if scan == scanno and point not in list(ret.keys()):
                    ret[point] = file
            except ValueError:
                continue
        return ret

    @staticmethod 
    def apply_mask(data, xmask, ymask):
        roi = data[ymask, :]
        return roi[:, xmask]


    # MAIN LOGIC
    def get_scan_params(self, scan):
        UB = numpy.array(scan.header('G')[2].split(' ')[-9:],dtype=numpy.float)
        wavelength = float(scan.header('G')[1].split(' ')[-1])

        return wavelength, UB

    def get_images(self, scan, first, last, dry_run=False):
        try:
            uccdtagline = scan.header('UCCD')[0]
            UCCD = os.path.split(os.path.dirname(uccdtagline.split()[-1]))
        except:
            print('warning: UCCD tag not found, use imagefolder for proper file specification')
            UCCD = []
        pattern = self._get_pattern(UCCD) 
        matches = self.find_edfs(pattern, scan.number())
        if set(range(first, last + 1)) > set(matches.keys()):
            raise errors.FileError("incorrect number of matches for scan {0} using pattern {1}".format(scan.number(), pattern))
        if dry_run:
            yield
        else:
            for i in range(first, last+1):
                edf = EdfFile.EdfFile(matches[i])
                yield edf.GetData(0)

    def _get_pattern(self,UCCD):
       imagefolder = self.config.imagefolder
       if imagefolder:
           try:
               imagefolder = imagefolder.format(UCCD=UCCD, rUCCD=list(reversed(UCCD)))
           except Exception as e:
               raise errors.ConfigError("invalid 'imagefolder' specification '{0}': {1}".format(self.config.imagefolder, e))
       else:
           imagefolder = os.path.join(*UCCD)

       if not os.path.exists(imagefolder):
           raise ValueError("invalid 'imagefolder' specification '{0}'. Path {1} does not exist".format(self.config.imagefolder, imagefolder))
       return os.path.join(imagefolder, '*')
       

class EH2(ID03Input):
    monitor_counter = 'Monitor'
    # define ID03 goniometer, SIXC geometry with 2D detector mounted on a
    # translation-axis (distance changing with changing Gamma)
    # The geometry is: 1+3S+2D
    # sample axis mu, th, chi, phi -> here chi,phi are omitted
    # detector axis mu, del, gam 
    # gam is realized by a translation along z (gamT) and rotation around x+ (gamR)
    qconv = xu.experiment.QConversion(['x+', 'z-'],  # 'y+', 'z+'
                                      ['x+', 'z-', 'ty', 'tz', 'x+'],
                                      [0, 1, 0])
    # convention for coordinate system: y downstream; z outwards; x upwards
    # (righthanded)
    # QConversion will set up the goniometer geometry. So the first argument
    # describes the sample rotations, the second the detector rotations and the
    # third the primary beam direction.
    ty = 600. # mm

    def parse_config(self, config):
        super(EH2, self).parse_config(config)
        centralpixel = self.config.centralpixel # (row, column) = (gamma, delta)
        # define detector parameters
        roi = (self.config.ymask[0], self.config.ymask[-1]+1,
               self.config.xmask[0], self.config.xmask[-1]+1)
        self.qconv.init_area('x+', 'z-',
                             cch1=centralpixel[1], cch2=centralpixel[0],
                             Nch1=516, Nch2=516,
                             pwidth1=self.config.pixelsize[1],
                             pwidth2=self.config.pixelsize[0],
                             distance=self.config.sdd-self.ty,
                             roi=roi)
        # distance sdd-600 corresponds to distance of the detector chip from
        # the gamR rotation axis (rest is handled by the translations ty and
        # gamT (along z))
        print(('{:>9} {:>10} {:>9} {:>9}'.format('Mu', 'Theta', 'Delta', 'Gamma')))

    def process_image(self, scanparams, pointparams, image):
        mu, theta, chi, phi, delta, gamma, mon, transm = pointparams
        wavelength, UB = scanparams
        data = image / mon / transm
        print(('{:9.4f} {:10.4f} {:9.4f} {:9.4f}'.format(mu, theta, delta, gamma)))

        # recalculate detector translation (which should be saved!)
        gamT = self.ty * numpy.tan(numpy.radians(gamma))

        # masking
        intensity = self.apply_mask(data, self.config.xmask, self.config.ymask)

        # no polarization correction for the moment!
        
        return intensity, numpy.ones_like(intensity), (mu, theta, delta, gamma, gamT,#weights added to API. keeps functionality identical with wights of one
                           self.ty, wavelength, UB, self.qconv)

    def get_point_params(self, scan, first, last):
        sl = slice(first, last+1)

        MU, TH, CHI, PHI, DEL, GAM, MON, TRANSM = list(range(8))
        params = numpy.zeros((last - first + 1, 8)) 
        # Mu, Theta, Chi, Phi, Delta, Gamma, MON, transm
        params[:, CHI] = scan.motorpos('Chi')
        params[:, PHI] = scan.motorpos('Phi')
        params[:, TH] = scan.datacol('thcnt')[sl]
        params[:, GAM] = scan.datacol('gamcnt')[sl]
        params[:, DEL] = scan.datacol('delcnt')[sl]
        params[:, MON] = scan.datacol(self.monitor_counter)[sl] 
        params[:, TRANSM] = scan.datacol('transm')[sl]
        params[:, MU] = scan.datacol('mucnt')[sl]
        return params

