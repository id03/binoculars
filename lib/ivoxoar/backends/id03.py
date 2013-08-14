import sys
import os
import itertools
import glob
import numpy

from PyMca import SixCircle, specfilewrapper, specfile
try:
    import EdfFile # allow user to provide a local version of EdfFile if the PyMca one is too old
except:
    from PyMca import EdfFile

from .. import space, backend, errors, util


class HKLProjection(backend.ProjectionBase):
    # arrays: gamma, delta
    # scalars: theta, mu, chi, phi
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        R = SixCircle.getHKL(wavelength, UB, gamma=gamma, delta=delta, theta=theta, mu=mu, chi=chi, phi=phi)
        H = R[0,:]
        K = R[1,:]
        L = R[2,:]
        return (H,K,L)

    def get_axis_labels(self):
        return 'H', 'K', 'L'


class TwoThetaProjection(HKLProjection):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        h,k,l = super(TwoThetaProjection, self).project(wavelength, UB, gamma, delta, theta, mu, chi, phi)
        return 2 * numpy.arcsin(wavelength * numpy.sqrt(h**2+k**2+l**2) / 4 / numpy.pi) # TODO: shouldn't this be a 1-tuple?

    def get_axis_labels(self):
        return 'TwoTheta'


class ID03Input(backend.InputBase):
    # OFFICIAL API
    def generate_jobs(self, command):
        scans = util.parse_multi_range(' '.join(command))
        if not len(scans):
            sys.stderr.write('error: no scans selected, nothing to do\n')
        for scanno in scans:
            scan = self.get_scan(scanno)
            try:
                pointcount = scan.lines()
            except specfile.error: # no points
                continue
            if self.config.target_weight and pointcount > self.config.target_weight * 1.4:
                for s in util.chunk_slicer(pointcount, self.config.target_weight):
                    yield backend.Job(scan=scanno, firstpoint=s.start, lastpoint=s.stop-1, weight=s.stop-s.start)
            else:
                yield backend.Job(scan=scanno, firstpoint=0, lastpoint=pointcount-1, weight=pointcount)

    def process_job(self, job):
        scan = self.get_scan(job.scan)
        
        scanparams = self.get_scan_params(scan) # wavelength, UB
        pointparams = self.get_point_params(scan, job.firstpoint, job.lastpoint) # 2D array of diffractometer angles + mon + transm
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
        self.config.app = util.parse_tuple(config.pop('app'), length=2, type=float)
        self.config.centralpixel = util.parse_tuple(config.pop('centralpixel'), length=2, type=int)


    # CONVENIENCE FUNCTIONS
    _spec = None
    def get_scan(self, scannumber):
        if self._spec is None:
            self._spec = specfilewrapper.Specfile(self.config.specfile) 
        return self._spec.select('{0}.1'.format(scannumber))

    @staticmethod
    def is_zap(scan):
        return scan.header('S')[0].split()[2].startswith('zap')

    def find_edfs(self, pattern, scanno):
        files = glob.glob(pattern)
        ret = {}
        for file in files:
            filename = os.path.basename(file).split('.')[0]
            scan, point, image = filename.split('_')[-3:]
            scan, point, image = int(scan), int(point), int(image)
            if scan == scanno:
                ret[point] = file
        return ret

    @staticmethod 
    def apply_mask(data, xmask, ymask):
        roi = data[ymask, :]
        return roi[:, xmask]


    # MAIN LOGIC
    def get_scan_params(self, scan):
        if self.is_zap(scan):
            # UB matrix will be installed in new versions of the zapline, it has to come from the configfile
            if not self.config.UB:
                raise errors.ConfigError('UB matrix must be specified in configuration file when processing zapscans')
            UB = numpy.array(self.config.UB)
        else:
            UB = numpy.array(scan.header('G')[2].split(' ')[-9:],dtype=numpy.float)
        wavelength = float(scan.header('G')[1].split(' ')[-1])

        return wavelength, UB

    def get_point_params(self, scan, first, last):
        delta, theta, chi, phi, mu, gamma = numpy.array(scan.header('P')[0].split(' ')[1:7],dtype=numpy.float)

        sl = slice(first, last+1)

        GAM, DEL, TH, CHI, PHI, MU, MON, TRANSM = range(8)
        params = numpy.zeros((last - first + 1, 8)) # gamma delta theta chi phi mu mon transm
        params[:, CHI] = chi
        params[:, PHI] = phi
        params[:, MU] = mu

        if self.is_zap(scan):
            th = scan.datacol('th')
            # correction for difference between back and forth in th motor
            th -= (th[1] - th[0]) / (len(th) * 1.0) / 2 # FIXME is this right?
            params[:, TH] = th[sl]

            params[:, GAM] = gamma
            params[:, DEL] = delta

            params[:, MON] = scan.datacol('zap_mon')[sl]

            transm = scan.datacol('zap_transm')
            transm[-1] = transm[-2] # bug in specfile
            params[:, TRANSM] = transm[sl]
        else:
            params[:, TH] = scan.datacol('thcnt')[sl]
            params[:, GAM] = scan.datacol('gamcnt')[sl]
            params[:, DEL] = scan.datacol('delcnt')[sl]
            params[:, MON] = scan.datacol(self.monitor_counter)[sl] # differs in EH1/EH2
            params[:, TRANSM] = scan.datacol('transm')[sl]

        return params

    def get_images(self, scan, first, last):
        if self.is_zap(scan):
            scanheaderC = scan.header('C')
            folder = os.path.split(scanheaderC[0].split(' ')[-1])[-1]
            scanname = scanheaderC[1].split(' ')[-1]
            pattern = os.path.join(self.config.imagefolder, folder, '*{0}*mpx*'.format(scanname))
            zapscanno = int(scanheaderC[2].split(' ')[-1]) # is different from scanno should be changed in spec!

            matches = self.find_edfs(pattern, zapscanno)
            if 0 not in matches:
                raise errors.FileError('could not find matching edf for zapscannumber {0}'.format(zapscannumber))
            edf = EdfFile.EdfFile(matches[0])
        
            for i in range(first, last+1):
                yield edf.GetData(i)

        else:
            uccdtagline = scan.header('UCCD')[0]
            UCCD = os.path.dirname(uccdtagline[6:]).split(os.sep)
            imagefolder = self.config.imagefolder
            if imagefolder:
                try:
                    imagefolder = imagefolder.format(UCCD=UCCD, rUCCD=list(reversed(UCCD)))
                except Exception as e:
                    raise errors.ConfigError("invalid 'imagefolder' specification '{0}': {1}".format(self.config.imagefolder, e))
            else:
                imagefolder = os.path.join(UCCD[:-1])

            if not os.path.exists(imagefolder):
                raise ValueError("invalid 'imagefolder' specification '{0}'. Path {1} does not exist".format(self.config.imagefolder, imagefolder))
            pattern = os.path.join(imagefolder, '*')
            matches = self.find_edfs(pattern, scan.number())
            if not matches:
                print "error: no matches for scan {0} using pattern {1}".format(scan.number(), pattern)
            for i in range(first, last+1):
                edf = EdfFile.EdfFile(matches[i])
                yield edf.GetData(0)


class EH1(ID03Input):
    monitor_counter = 'mon'

    def process_image(self, scanparams, pointparams, image):
        gamma, delta, theta, mu, chi, phi, mon, transm = pointparams
        wavelength, UB = scanparams
        data = image / mon / transm

        # pixels to angles
        app = self.config.app # angle per pixel (delta, gamma)
        centralpixel = self.config.centralpixel # (row, column) = (delta, gamma)
        gamma_range= -app[1]*(numpy.arange(data.shape[1])-centralpixel[1])+gamma
        delta_range= app[0]*(numpy.arange(data.shape[0])-centralpixel[0])+delta

        # masking
        gamma_range = gamma_range[self.config.ymask]
        delta_range = delta_range[self.config.xmask]
        intensity = self.apply_mask(data, self.config.xmask, self.config.ymask)

        return intensity.flatten(), (wavelength, UB, gamma_range, delta_range, theta, mu, chi, phi)


class EH2(ID03Input):
    monitor_counter = 'Monitor'

    def parse_config(self, config):
        super(EH2, self).parse_config(config)
        self.config.sdd = float(config.pop('sdd'))

    def process_image(self, scanparams, pointparams, image):
        gamma, delta, theta, mu, chi, phi, mon, transm = pointparams
        wavelength, UB = scanparams
        data = image / mon / transm

        # area correction
        sdd = self.config.sdd / numpy.cos(gamma * numpy.pi / 180)
        data *= (self.config.sdd / sdd)**2

        # pixels to angles
        pixelsize = self.config.sdd * numpy.tan(self.config.app[0] * numpy.pi / 180)
        app = numpy.arctan(pixelsize / sdd) * 180 / numpy.pi
        centralpixel = self.config.centralpixel # (row, column) = (delta, gamma)
        gamma_range = app*(numpy.arange(data.shape[1])-centralpixel[1])+gamma
        delta_range = app*(numpy.arange(data.shape[0])-centralpixel[0])+delta

        # masking
        gamma_range = gamma_range[self.config.xmask]
        delta_range = delta_range[self.config.ymask]
        intensity = self.apply_mask(data, self.config.xmask, self.config.ymask)
        intensity = numpy.roi90(intensity)

        return intensity.flatten(), (wavelength, UB, gamma_range, delta_range, theta, mu, chi, phi)
