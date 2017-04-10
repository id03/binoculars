"""
BINocular backend for beamline BM25, branch B first endstation [1] 
This backend should serve as a basic implementation of a backend based on
xrayutilities [2]. It uses the information from the edf files (motors position
and detector image) ignoring the spec file, except for using its scan numbers
to identify images belonging to the same scan.

You should use CCD file names generated with the following pattern:
filename_#n_#p_#r.edf  (n: spec-scan number, p: point number, r: image number)
Binning (2,2)

The backend is called 'EH2SCD'.

Created on 2014-10-28

[1] http://www.esrf.eu/UsersAndScience/Experiments/CRG/BM25/BeamLine/experimentalstations/Single_Crystal_Diffraction
[2] http://xrayutilities.sourceforge.net/

author: Dominik Kriegner (dominik.kriegner@gmail.com)
"""

import sys
import os
import glob
import numpy
import xrayutilities as xu

from .. import backend, errors, util

class HKLProjection(backend.ProjectionBase):
    # scalars: mu, theta, phi, chi, ccdty, ccdtx, ccdtz, ccdth, wavelength
    # 3x3 matrix: UB
    def project(self, mu, theta, phi, chi, ccdty, ccdtx, ccdtz, ccdth, ccdtr, wavelength, UB, qconv):
        qconv.wavelength = wavelength
        h, k, l = qconv.area(mu, theta, phi, chi, 
                             ccdty, ccdtx, ccdtz, ccdth, ccdtr,
                             UB=UB.reshape((3, 3)))
        return (h, k, l)

    def get_axis_labels(self):
        return 'H', 'K', 'L'

class HKProjection(HKLProjection):
    def project(self, mu, theta, phi, chi, ccdty, ccdtx, ccdtz, ccdth, ccdtr, wavelength, UB, qconv):
        H, K, L = super(HKProjection, self).project(mu, theta, phi, chi, ccdty, ccdtx, ccdtz, ccdth, ccdtr, wavelength, UB, qconv)
        return (H, K)

    def get_axis_labels(self):
        return 'H', 'K'

class QProjection(backend.ProjectionBase):
    def project(self, mu, theta, phi, chi, ccdty, ccdtx, ccdtz, ccdth, ccdtr, wavelength, UB, qconv):
        qconv.wavelength = wavelength
        qx, qy, qz = qconv.area(mu, theta, phi, chi, 
                                ccdty, ccdtx, ccdtz, ccdth, ccdtr,
                                UB=numpy.identity(3))
        return (qx, qy, qz)

    def get_axis_labels(self):
        return 'qx', 'qy', 'qz'

class QinpProjection(backend.ProjectionBase):
    def project(self, mu, theta, phi, chi, ccdty, ccdtx, ccdtz, ccdth, ccdtr, wavelength, UB, qconv):
        qconv.wavelength = wavelength
        qx, qy, qz = qconv.area(mu, theta, phi, chi, 
                                ccdty, ccdtx, ccdtz, ccdth, ccdtr,
                                UB=numpy.identity(3))
        return (numpy.sqrt(qx**2+qy**2), qz)

    def get_axis_labels(self):
        return 'qinp', 'qz'

class EDFInput(backend.InputBase):
    # OFFICIAL API
    def generate_jobs(self, command):
        scans = util.parse_multi_range(','.join(command).replace(' ', ','))
        imgs = self.list_images(scans)
        imgcount = len(imgs)
        if not len(imgs):
            sys.stderr.write('error: no images selected, nothing to do\n')
        #next(self.get_images(imgs, 0, imgcount-1, dry_run=True))# dryrun

        for s in util.chunk_slicer(imgcount, self.config.target_weight):
            yield backend.Job(images=imgs, firstimage=s.start, lastimage=s.stop-1, weight=s.stop-s.start)

    def process_job(self, job):
        super(EDFInput, self).process_job(job)
        images = self.get_images(job.images, job.firstimage, job.lastimage) # iterator!
        
        for image in images:
            yield self.process_image(image)

    def parse_config(self, config):
        super(EDFInput, self).parse_config(config)
        self.config.xmask = util.parse_multi_range(config.pop('xmask'))
        self.config.ymask = util.parse_multi_range(config.pop('ymask'))
        self.config.imagefile = config.pop('imagefile')
        self.config.UB = config.pop('ub', None)
        if self.config.UB:
            self.config.UB = util.parse_tuple(self.config.UB, length=9, type=float)
        self.config.sddx = float(config.pop('sddx_offset'))
        self.config.sddy = float(config.pop('sddy_offset'))
        self.config.sddz = float(config.pop('sddz_offset'))
        self.config.ccdth0 = float(config.pop('ccdth_offset'))
        self.config.pixelsize = util.parse_tuple(config.pop('pixelsize'), length=2, type=float)
        self.config.centralpixel = util.parse_tuple(config.pop('centralpixel'), length=2, type=float)

    def get_destination_options(self, command):
        if not command:
            return False
        command = ','.join(command).replace(' ', ',')
        scans = util.parse_multi_range(command)
        return dict(first=min(scans), last=max(scans), range=','.join(command))

    # CONVENIENCE FUNCTIONS
    @staticmethod 
    def apply_mask(data, xmask, ymask):
        roi = data[ymask, :]
        return roi[:, xmask]
                
    # MAIN LOGIC
    def list_images(self, scannrs):
        pattern = self.config.imagefile
        imgfiles = []
        # check if necessary image-files exist
        for nr in scannrs: 
            try:
                fpattern = pattern.format(scannr=nr)
            except Exception as e:
                raise errors.ConfigError("invalid 'imagefile' specification '{0}': {1}".format(self.config.imagefile, e))

            files = glob.glob(fpattern)
            if len(files)==0:
                raise errors.FileError("needed file do not exist: scannr {0}".format(nr))
            else:
                imgfiles += files
        return imgfiles

    def get_images(self, imgs, first, last, dry_run=False):
        for i in range(first,last+1): 
            img = imgs[i]
            if dry_run:
                yield
            else:
                edf = xu.io.EDFFile(img)
                yield edf

class EH2SCD(EDFInput):
    monitor_counter = 'C_mont'
    # define BM25 goniometer, SIXC geometry? with 2D detector mounted on
    # translation-axes
    # see http://www.esrf.eu/UsersAndScience/Experiments/CRG/BM25/BeamLine/experimentalstations/Single_Crystal_Diffraction
    # The geometry is: 4S + translations and one det. rotation
    # sample axis: mu, th, chi, phi
    # detector axis: translations + theta rotation (to make beam perpendicular
    #                to the detector plane in symmetric arrangement)
    qconv = xu.experiment.QConversion(['x+', 'z+', 'y+', 'x+'],
                                      ['ty', 'tx', 'tz', 'x+', 'ty'],
                                      [0, 1, 0])
    # convention for coordinate system: y downstream; x in bound; z upwards
    # (righthanded)
    # QConversion will set up the goniometer geometry. So the first argument
    # describes the sample rotations, the second the detector rotations and the
    # third the primary beam direction.

    def parse_config(self, config):
        super(EH2SCD, self).parse_config(config)
        centralpixel = self.config.centralpixel
        # define detector parameters
        roi = (self.config.ymask[0], self.config.ymask[-1]+1,
               self.config.xmask[0], self.config.xmask[-1]+1)
        self.qconv.init_area('z-', 'x+',
                             cch1=centralpixel[1], cch2=centralpixel[0],
                             Nch1=1912, Nch2=3825,
                             pwidth1=self.config.pixelsize[1],
                             pwidth2=self.config.pixelsize[0],
                             distance=1e-10,
                             roi=roi)
        print(('{:>20} {:>9} {:>10} {:>9} {:>9} {:>9}'.format(' ', 'Mu', 'Theta', 'CCD_Y', 'CCD_X', 'CCD_Z')))

    def process_image(self, image):
        # motor positions
        mu = float(image.header['M_mu'])
        th = float(image.header['M_th'])
        chi = float(image.header['M_chi'])
        phi = float(image.header['M_phi'])
        # distance 'ctr' corresponds to distance of the detector chip from
        # the CCD_TH rotation axis. The rest is handled by the translations
        ctr = -270.0 # measured by ruler only!!!
        cty = float(image.header['M_CCD_Y'])-self.config.sddy-ctr
        ctx = float(image.header['M_CCD_X'])-self.config.sddx
        ctz = float(image.header['M_CCD_Z'])-self.config.sddz
        cth = float(image.header['M_CCD_TH'])-self.config.ccdth0

        # filter correction
        transm = 1. # no filter correction! (Filters are manual on BM25!)

        mon = float(image.header[self.monitor_counter])
        wavelength = float(image.header['WAVELENGTH'])
        if self.config.UB:
            UB = self.config.UB
        else:
            UB = self._get_UB(image.header)

        # normalization
        data = image.data / mon / transm
        print(('{:>20} {:9.4f} {:10.4f} {:9.1f} {:9.1f} {:9.1f}'.format(os.path.split(image.filename)[-1] ,mu, th, cty, ctx, ctz)))

        # masking
        intensity = self.apply_mask(data, self.config.xmask, self.config.ymask)
 
        return intensity, numpy.ones_like(intensity), (mu, th, phi, chi, cty, ctx, ctz, cth, ctr,## weights added to API. Treated here like before
                           wavelength, UB, self.qconv)

    @staticmethod
    def _get_UB(header):
        ub = numpy.zeros(9)
        for i in range(9):
            ub[i] = float(header['UB{:d}'.format(i)])
        return ub

