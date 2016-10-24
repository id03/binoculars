import sys
import os
import glob
import numpy
import time

#python3 support
PY3 = sys.version_info > (3,)
if PY3:
    pass
else:
    from itertools import izip as zip

try:
    from PyMca import specfilewrapper, EdfFile, SixCircle, specfile
except ImportError:
    from PyMca5.PyMca import specfilewrapper, EdfFile, SixCircle, specfile


from .. import backend, errors, util


class pixels(backend.ProjectionBase):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        y,x = numpy.mgrid[slice(None,gamma.shape[0]), slice(None,delta.shape[0])]
        return (y, x)        

    def get_axis_labels(self):
        return 'y','x'

class HKLProjection(backend.ProjectionBase):
    # arrays: gamma, delta
    # scalars: theta, mu, chi, phi
    def project(self, wavelength, UB, beta, delta, omega, alfa, chi, phi):
        R = SixCircle.getHKL(wavelength, UB, gamma=beta, delta=delta, theta=omega, mu=alfa, chi=chi, phi=phi)
        shape = beta.size, delta.size
        H = R[0,:].reshape(shape)
        K = R[1,:].reshape(shape)
        L = R[2,:].reshape(shape)
        return (H, K, L)

    def get_axis_labels(self):
        return 'H', 'K', 'L'

class HKProjection(HKLProjection):
    def project(self, wavelength, UB, beta, delta, omega, alfa, chi, phi):
        H, K, L = super(HKProjection, self).project( wavelength, UB, beta, delta, omega, alfa, chi, phi)
        return (H, K)

    def get_axis_labels(self):
        return 'H', 'K'

class ThetaLProjection(backend.ProjectionBase):
    # arrays: gamma, delta
    # scalars: theta, mu, chi, phi
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        R = SixCircle.getHKL(wavelength, UB, gamma=gamma, delta=delta, theta=theta, mu=mu, chi=chi, phi=phi)
        shape = gamma.size, delta.size
        L = R[2,:].reshape(shape)
        theta_array = numpy.ones_like(L) * theta
        return (theta_array,L)

    def get_axis_labels(self):
        return 'Theta', 'L'

class QProjection(backend.ProjectionBase):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        shape = gamma.size, delta.size
        sixc = SixCircle.SixCircle()
        sixc.setLambda(wavelength)
        sixc.setUB(UB)
        R = sixc.getQSurface(gamma=gamma, delta=delta, theta=theta, mu=mu, chi=chi, phi=phi)
        qx = R[0,:].reshape(shape)
        qy = R[1,:].reshape(shape)
        qz = R[2,:].reshape(shape)
        return (qx, qy, qz)

    def get_axis_labels(self):
        return 'qx', 'qy', 'qz'

class SphericalQProjection(QProjection):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        qx, qy, qz = super(SphericalQProjection, self).project(wavelength, UB, gamma, delta, theta, mu, chi, phi)
        q = numpy.sqrt(qx**2 + qy**2 + qz**2)
        theta = numpy.arccos(qz / q)
        phi = numpy.arctan2(qy, qx)
        return (q, theta, phi)

    def get_axis_labels(self):
        return 'Q', 'Theta', 'Phi'

class CylindricalQProjection(QProjection):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        qx, qy, qz = super(CylindricalQProjection, self).project(wavelength, UB, gamma, delta, theta, mu, chi, phi)
        qpar = numpy.sqrt(qx**2 + qy**2)
        phi = numpy.arctan2(qy, qx)
        return (qpar, qz, phi)

    def get_axis_labels(self):
        return 'qpar', 'qz', 'Phi'

class nrQProjection(backend.ProjectionBase):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        k0 = 2 * numpy.pi / wavelength
        delta, gamma = numpy.meshgrid(delta, gamma)
        mu *= numpy.pi/180
        delta *= numpy.pi/180
        gamma *= numpy.pi/180

        qy = k0 * (numpy.cos(gamma) * numpy.cos(delta) - numpy.cos(mu)) ## definition of qx, and qy same as spec at theta = 0
        qx = k0 * (numpy.cos(gamma) * numpy.sin(delta))
        qz = k0 * (numpy.sin(gamma) + numpy.sin(mu))
        return (qx, qy, qz)

    def get_axis_labels(self):
        return 'qx', 'qy', 'qz'

class TwoThetaProjection(SphericalQProjection):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        q, theta, phi = super(TwoThetaProjection, self).project(wavelength, UB, gamma, delta, theta, mu, chi, phi)
        return 2 * numpy.arcsin(q * wavelength / (4 * numpy.pi)) / numpy.pi * 180, # note: we need to return a 1-tuple?

    def get_axis_labels(self):
        return 'TwoTheta'

class Qpp(nrQProjection):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        qx, qy, qz = super(Qpp, self).project(wavelength, UB, gamma, delta, theta, mu, chi, phi)
        qpar = numpy.sqrt(qx**2 + qy**2)
        qpar[numpy.sign(qx) == -1] *= -1
        return (qpar, qz)

    def get_axis_labels(self):
        return 'Qpar', 'Qz'

class GammaDeltaTheta(HKLProjection):#just passing on the coordinates, makes it easy to accurately test the theta correction
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        delta,gamma = numpy.meshgrid(delta,gamma)
        theta = theta * numpy.ones_like(delta)
        return (gamma, delta, theta)        

    def get_axis_labels(self):
        return 'Gamma','Delta','Theta'

class GammaDelta(HKLProjection):#just passing on the coordinates, makes it easy to accurately test the theta correction
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        delta,gamma = numpy.meshgrid(delta,gamma)
        return (gamma, delta)        

    def get_axis_labels(self):
        return 'Gamma','Delta'


class GammaDeltaMu(HKLProjection):#just passing on the coordinates, makes it easy to accurately test the theta correction
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        delta,gamma = numpy.meshgrid(delta,gamma)
        mu = mu * numpy.ones_like(delta)
        return (gamma, delta, mu)        

    def get_axis_labels(self):
        return 'Gamma','Delta','Mu'

class BM32Input(backend.InputBase):
    # OFFICIAL API

    dbg_scanno = None
    dbg_pointno = None

    def generate_jobs(self, command):
        scans = util.parse_multi_range(','.join(command).replace(' ', ','))
        if not len(scans):
            sys.stderr.write('error: no scans selected, nothing to do\n')
        for scanno in scans:
            util.status('processing scan {0}...'.format(scanno))
            scan = self.get_scan(scanno)
            if self.config.pr:
                pointcount = self.config.pr[1] - self.config.pr[0] + 1
                start = self.config.pr[0]
            else:
                start = 0
                try:
                    pointcount = scan.lines()
                except specfile.error: # no points
                    continue
            next(self.get_images(scan, 0, pointcount-1, dry_run=True))# dryrun
            if pointcount > self.config.target_weight * 1.4:
                for s in util.chunk_slicer(pointcount, self.config.target_weight):
                    yield backend.Job(scan=scanno, firstpoint=start+s.start, lastpoint=start+s.stop-1, weight=s.stop-s.start)
            else:
                yield backend.Job(scan=scanno, firstpoint=start, lastpoint=start+pointcount-1, weight=pointcount)

    def process_job(self, job):
        super(BM32Input, self).process_job(job)
        scan = self.get_scan(job.scan)
        self.metadict = dict()
        try:
            scanparams = self.get_scan_params(scan) # wavelength, UB
            pointparams = self.get_point_params(scan, job.firstpoint, job.lastpoint) # 2D array of diffractometer angles + mon + transm
            images = self.get_images(scan, job.firstpoint, job.lastpoint) # iterator!
        
            for pp, image in zip(pointparams, images):
                yield self.process_image(scanparams, pp, image)
            util.statuseol()
        except Exception as exc:
            #exc.args = errors.addmessage(exc.args, ', An error occured for scan {0} at point {1}. See above for more information'.format(self.dbg_scanno, self.dbg_pointno))
            raise
        self.metadata.add_section('id03_backend', self.metadict)

    def parse_config(self, config):
        super(BM32Input, self).parse_config(config)
        self.config.xmask = util.parse_multi_range(config.pop('xmask', None))#Optional, select a subset of the image range in the x direction. all by default
        self.config.ymask = util.parse_multi_range(config.pop('ymask', None))#Optional, select a subset of the image range in the y direction. all by default
        self.config.specfile = config.pop('specfile')#Location of the specfile
        self.config.imagefolder = config.pop('imagefolder', None) #Optional, takes specfile folder tag by default
        self.config.pr = util.parse_tuple(config.pop('pr', None), length=2, type=int) #Optional, all range by default
        self.config.background = config.pop('background', None) #Optional, if supplied a space of this image is constructed
        if self.config.xmask is None:
            self.config.xmask = slice(None)
        if self.config.ymask is None:
            self.config.ymask = slice(None)
        self.config.maskmatrix = load_matrix(config.pop('maskmatrix', None)) #Optional, if supplied pixels where the mask is 0 will be removed
        self.config.sdd = config.pop('sdd', None)# sample to detector distance (mm)
        if self.config.sdd is not None:
            self.config.sdd = float(self.config.sdd)
        self.config.pixelsize = util.parse_tuple(config.pop('pixelsize', None), length=2, type=float)# pixel size x/y (mm) (same dimension as sdd)

    def get_destination_options(self, command):
        if not command:
            return False
        command = ','.join(command).replace(' ', ',')
        scans = util.parse_multi_range(command)
        return dict(first=min(scans), last=max(scans), range=','.join(str(scan) for scan in scans))

    # CONVENIENCE FUNCTIONS
    _spec = None
    def get_scan(self, scannumber):
        if self._spec is None:
            self._spec = specfilewrapper.Specfile(self.config.specfile) 
        return self._spec.select('{0}.1'.format(scannumber))

    def find_edfs(self, pattern):
        files = glob.glob(pattern)
        ret = {}
        for file in files:
            try:
                filename = os.path.basename(file).split('.')[0]
                imno = int(filename.split('_')[-1].split('-')[-1])
                ret[imno] = file
            except ValueError:
                continue
        return ret

    @staticmethod 
    def apply_mask(data, xmask, ymask):
        roi = data[ymask, :]
        return roi[:, xmask]

    # MAIN LOGIC
    def get_scan_params(self, scan):
        self.dbg_scanno = scan.number()
        UB = numpy.array(scan.header('G')[2].split(' ')[-9:],dtype=numpy.float)
        wavelength = float(scan.header('G')[1].split(' ')[-1])

        self.metadict['UB'] = UB
        self.metadict['wavelength'] = wavelength

        return wavelength, UB

    def get_images(self, scan, first, last, dry_run=False):
        imagenos = numpy.array(scan.datacol('img')[slice(first, last + 1)], dtype = numpy.int)
        if self.config.background:
            if not os.path.exists(self.config.background):
                raise errors.FileError('could not find background file {0}'.format(self.config.background))
            if dry_run:
                yield
            else:
                edf = EdfFile.EdfFile(self.config.background)
                for i in range(first, last+1):
                    self.dbg_pointno = i
                    yield edf
        else:
            try:
                uccdtagline = scan.header('M')[0].split()[-1]
                UCCD = os.path.dirname(uccdtagline).split(os.sep)
            except:
                print('warning: UCCD tag not found, use imagefolder for proper file specification')
                UCCD = []
            pattern = self._get_pattern(UCCD) 
            matches = self.find_edfs(pattern)
            if not set(imagenos).issubset(set(matches.keys())):
                raise errors.FileError("incorrect number of matches for scan {0} using pattern {1}".format(scan.number(), pattern))
            if dry_run:
                yield
            else:
                for i in imagenos:
                    self.dbg_pointno = i
                    edf = EdfFile.EdfFile(matches[i])
                    yield edf

    def _get_pattern(self,UCCD):
       imagefolder = self.config.imagefolder
       if imagefolder:
           try:
               imagefolder = imagefolder.format(UCCD=UCCD, rUCCD=list(reversed(UCCD)))
           except Exception as e:
               raise errors.ConfigError("invalid 'imagefolder' specification '{0}': {1}".format(self.config.imagefolder, e))
           else:
               if not os.path.exists(imagefolder):
                   raise errors.ConfigError("invalid 'imagefolder' specification '{0}'. Path {1} does not exist".format(self.config.imagefolder, imagefolder))               
       else:
           imagefolder = os.path.join(*UCCD)
           if not os.path.exists(imagefolder):
               raise errors.ConfigError("invalid UCCD tag '{0}'. The UCCD tag in the specfile does not point to an existing folder. Specify the imagefolder in the configuration file.".format(imagefolder))
       return os.path.join(imagefolder, '*')
       
class EH1(BM32Input):
    def parse_config(self, config):
        super(EH1, self).parse_config(config)
        self.config.centralpixel = util.parse_tuple(config.pop('centralpixel', None), length=2, type=int)
        self.config.UB = util.parse_tuple(config.pop('ub', None), length=9, type=float)
      
    def process_image(self, scanparams, pointparams, edf):
        delta, omega, alfa, beta, chi, phi, mon, transm = pointparams
        wavelength, UB = scanparams

        image = edf.GetData(0)
        header = edf.GetHeader(0)

        weights = numpy.ones_like(image)

        if not self.config.centralpixel:
            self.config.centralpixel = (int(header['y_beam']), int(header['x_beam']))
        if not self.config.sdd:
            self.config.sdd = float(header['det_sample_dist'])

        if self.config.background:
            data = image / mon
        else:
            data = image / mon / transm

        if mon == 0:
            raise errors.BackendError('Monitor is zero, this results in empty output. Scannumber = {0}, pointnumber = {1}. Did you forget to open the shutter?'.format(self.dbg_scanno, self.dbg_pointno)) 

        util.status('{4}| beta: {0:.3f}, delta: {1:.3f}, omega: {2:.3f}, alfa: {3:.3f}'.format(beta, delta, omega, alfa, time.ctime(time.time())))

        # pixels to angles
        pixelsize = numpy.array(self.config.pixelsize)
        sdd = self.config.sdd 

        app = numpy.arctan(pixelsize / sdd) * 180 / numpy.pi

        centralpixel = self.config.centralpixel # (column, row) = (delta, gamma)
        beta_range= -app[1] * (numpy.arange(data.shape[1]) - centralpixel[1]) + beta
        delta_range= app[0] * (numpy.arange(data.shape[0]) - centralpixel[0]) + delta

        # masking
        if self.config.maskmatrix is not None:
            if self.config.maskmatrix.shape != data.shape:
                raise errors.BackendError('The mask matrix does not have the same shape as the images')
            weights *= self.config.maskmatrix

        delta_range = delta_range[self.config.ymask]
        beta_range = beta_range[self.config.xmask]

        weights = self.apply_mask(weights, self.config.xmask, self.config.ymask)
        intensity = self.apply_mask(data, self.config.xmask, self.config.ymask)

        intensity = numpy.rot90(intensity)
        intensity = numpy.fliplr(intensity)
        intensity = numpy.flipud(intensity)

        weights = numpy.rot90(weights)
        weights = numpy.fliplr(weights)
        weights = numpy.flipud(weights)

        #polarisation correction
        delta_grid, beta_grid = numpy.meshgrid(delta_range, beta_range)
        Pver = 1 - numpy.sin(delta_grid * numpy.pi / 180.)**2 * numpy.cos(beta_grid * numpy.pi / 180.)**2
        #intensity /= Pver
 
        return intensity, weights, (wavelength, UB, beta_range, delta_range, omega, alfa, chi, phi)

    def get_point_params(self, scan, first, last):
        sl = slice(first, last+1)

        DEL, OME, ALF, BET, CHI, PHI, MON, TRANSM = list(range(8))
        params = numpy.zeros((last - first + 1, 8)) # gamma delta theta chi phi mu mon transm
        params[:, CHI] = 0    #scan.motorpos('CHI')
        params[:, PHI] = 0    #scan.motorpos('PHI')

        params[:, OME] = scan.datacol('omecnt')[sl]
        params[:, BET] = scan.datacol('betcnt')[sl]
        params[:, DEL] = scan.datacol('delcnt')[sl]
        params[:, MON] = scan.datacol('Monitor')[sl]

        #params[:, TRANSM] = scan.datacol('transm')[sl]
        params[:, TRANSM] = 1

        params[:, ALF] = scan.datacol('alfcnt')[sl]
        return params
        
def load_matrix(filename):
    if filename == None:
        return None
    if os.path.exists(filename):
        ext = os.path.splitext(filename)[-1]
        if ext == '.txt':
            return numpy.array(numpy.loadtxt(filename), dtype = numpy.bool)
        elif ext == '.npy':
            return numpy.array(numpy.load(filename), dtype = numpy.bool)
        elif ext == '.edf':
            return numpy.array(EdfFile.EdfFile(filename).getData(0),dtype = numpy.bool)
        else:
            raise ValueError('unknown extension {0}, unable to load matrix!\n'.format(ext))        
    else:
       raise IOError('filename: {0} does not exist. Can not load matrix'.format(filename))


