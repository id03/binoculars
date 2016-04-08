import sys
import os
import itertools
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
        y, x = numpy.mgrid[slice(None, gamma.shape[0]), slice(None, delta.shape[0])]
        return (y, x)

    def get_axis_labels(self):
        return 'y', 'x'


class HKLProjection(backend.ProjectionBase):
    # arrays: gamma, delta
    # scalars: theta, mu, chi, phi
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        R = SixCircle.getHKL(wavelength, UB, gamma=gamma, delta=delta, theta=theta, mu=mu, chi=chi, phi=phi)
        shape = gamma.size, delta.size
        H = R[0, :].reshape(shape)
        K = R[1, :].reshape(shape)
        L = R[2, :].reshape(shape)
        return (H, K, L)

    def get_axis_labels(self):
        return 'H', 'K', 'L'


class HKProjection(HKLProjection):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        H, K, L = super(HKProjection, self).project(wavelength, UB, gamma, delta, theta, mu, chi, phi)
        return (H, K)

    def get_axis_labels(self):
        return 'H', 'K'


class specularangles(backend.ProjectionBase):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        delta, gamma = numpy.meshgrid(delta, gamma)
        mu *= numpy.pi/180
        delta *= numpy.pi/180
        gamma *= numpy.pi/180
        chi *= numpy.pi/180
        phi *= numpy.pi/180
        theta *= numpy.pi/180

        def mat(u, th):
            ux, uy, uz = u[0], u[1], u[2]
            sint = numpy.sin(th)
            cost = numpy.cos(th)
            mcost = (1 - numpy.cos(th))

            return numpy.matrix([[cost + ux**2 * mcost, ux * uy * mcost - uz * sint, ux * uz * mcost + uy * sint],
                             [uy * ux * mcost + uz * sint, cost + uy**2 * mcost, uy * uz - ux * sint],
                             [uz * ux * mcost - uy * sint, uz * uy * mcost + ux * sint, cost + uz**2 * mcost]])

        def rot(vx, vy, vz, u, th):
            R = mat(u, th)
            return R[0, 0] * vx + R[0, 1] * vy + R[0, 2] * vz, R[1, 0] * vx + R[1, 1] * vy + R[1, 2] * vz, R[2, 0] * vx + R[2, 1] * vy + R[2, 2] * vz

        #what are the angles of kin and kout in the sample frame?

        #angles in the hexapod frame
        koutx, kouty, koutz = numpy.sin(- numpy.pi / 2 + gamma) * numpy.cos(delta), numpy.sin(- numpy.pi / 2 + gamma) * numpy.sin(delta), numpy.cos(- numpy.pi / 2 + gamma)
        kinx, kiny, kinz = numpy.sin(numpy.pi / 2 - mu), 0, numpy.cos(numpy.pi / 2 - mu)

        #now we rotate the frame around hexapod rotation th
        xaxis = numpy.array(rot(1, 0, 0, numpy.array([0, 0, 1]), theta))
        yaxis = numpy.array(rot(0, 1, 0, numpy.array([0, 0, 1]), theta))

        #first we rotate the sample around the xaxis
        koutx, kouty, koutz = rot(koutx, kouty, koutz, xaxis,  chi)
        kinx, kiny, kinz = rot(kinx, kiny, kinz, xaxis, chi)
        yaxis = numpy.array(rot(yaxis[0], yaxis[1], yaxis[2], xaxis, chi))  # we also have to rotate the yaxis

        #then we rotate the sample around the yaxis
        koutx, kouty, koutz = rot(koutx, kouty, koutz, yaxis,  phi)
        kinx, kiny, kinz = rot(kinx, kiny, kinz, yaxis, phi)

        #to calculate the equivalent gamma, delta and mu in the sample frame we rotate the frame around the sample z which is 0,0,1
        back = numpy.arctan2(kiny, kinx)
        koutx, kouty, koutz = rot(koutx, kouty, koutz, numpy.array([0, 0, 1]),  -back)
        kinx, kiny, kinz = rot(kinx, kiny, kinz, numpy.array([0, 0, 1]), -back)

        mu = numpy.arctan2(kinz, kinx) * numpy.ones_like(delta)
        delta = numpy.pi - numpy.arctan2(kouty, koutx)
        gamma = numpy.pi - numpy.arctan2(koutz, koutx)

        delta[delta > numpy.pi] -= 2 * numpy.pi
        gamma[gamma > numpy.pi] -= 2 * numpy.pi

        mu *= 1 / numpy.pi * 180
        delta *= 1 / numpy.pi * 180
        gamma *= 1 / numpy.pi * 180

        return (gamma - mu, gamma + mu, delta)

    def get_axis_labels(self):
        return 'g-m', 'g+m', 'delta'


class ThetaLProjection(backend.ProjectionBase):
    # arrays: gamma, delta
    # scalars: theta, mu, chi, phi
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        R = SixCircle.getHKL(wavelength, UB, gamma=gamma, delta=delta, theta=theta, mu=mu, chi=chi, phi=phi)
        shape = gamma.size, delta.size
        L = R[2, :].reshape(shape)
        theta_array = numpy.ones_like(L) * theta
        return (theta_array, L)

    def get_axis_labels(self):
        return 'Theta', 'L'


class QProjection(backend.ProjectionBase):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        shape = gamma.size, delta.size
        sixc = SixCircle.SixCircle()
        sixc.setLambda(wavelength)
        sixc.setUB(UB)
        R = sixc.getQSurface(gamma=gamma, delta=delta, theta=theta, mu=mu, chi=chi, phi=phi)
        qz = R[0, :].reshape(shape)
        qy = R[1, :].reshape(shape)
        qx = R[2, :].reshape(shape)
        return (qz, qy, qx)

    def get_axis_labels(self):
        return 'qx', 'qy', 'qz'



class SphericalQProjection(QProjection):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        qz, qy, qx = super(SphericalQProjection, self).project(wavelength, UB, gamma, delta, theta, mu, chi, phi)
        q = numpy.sqrt(qx**2 + qy**2 + qz**2)
        theta = numpy.arccos(qz / q)
        phi = numpy.arctan2(qy, qx)
        return (q, theta, phi)

    def get_axis_labels(self):
        return 'Q', 'Theta', 'Phi'


class CylindricalQProjection(QProjection):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        qz, qy, qx = super(CylindricalQProjection, self).project(wavelength, UB, gamma, delta, theta, mu, chi, phi)
        qpar = numpy.sqrt(qx**2 + qy**2)
        phi = numpy.arctan2(qy, qx)
        return (qpar, qz, phi)

    def get_axis_labels(self):
        return 'qpar', 'qz', 'Phi'


class nrQProjection(QProjection):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        qx, qy, qz = super(nrQProjection, self).project(wavelength, UB, gamma, delta, 0, mu, chi, phi)
        return (qx, qy, qz)

    def get_axis_labels(self):
        return 'qx', 'qy', 'qz'


class TwoThetaProjection(SphericalQProjection):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        q, theta, phi = super(TwoThetaProjection, self).project(wavelength, UB, gamma, delta, theta, mu, chi, phi)
        return 2 * numpy.arcsin(q * wavelength / (4 * numpy.pi)) / numpy.pi * 180,  # note: we need to return a 1-tuple?

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


class GammaDeltaTheta(HKLProjection):  # just passing on the coordinates, makes it easy to accurately test the theta correction
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        delta, gamma = numpy.meshgrid(delta, gamma)
        theta = theta * numpy.ones_like(delta)
        return (gamma, delta, theta)

    def get_axis_labels(self):
        return 'Gamma', 'Delta', 'Theta'


class GammaDelta(HKLProjection):  # just passing on the coordinates, makes it easy to accurately test the theta correction
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        delta, gamma = numpy.meshgrid(delta, gamma)
        return (gamma, delta)

    def get_axis_labels(self):
        return 'Gamma', 'Delta'


class GammaDeltaMu(HKLProjection):  # just passing on the coordinates, makes it easy to accurately test the theta correction
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        delta, gamma = numpy.meshgrid(delta, gamma)
        mu = mu * numpy.ones_like(delta)
        return (gamma, delta, mu)

    def get_axis_labels(self):
        return 'Gamma', 'Delta', 'Mu'

class QTransformation(QProjection):
    def project(self, wavelength, UB, gamma, delta, theta, mu, chi, phi):
        qx, qy, qz = super(QTransformation, self).project(wavelength, UB, gamma, delta, theta, mu, chi, phi)

        M = self.config.matrix
        q1 = qx * M[0] + qy * M[1] + qz * M[2]
        q2 = qx * M[3] + qy * M[4] + qz * M[5]
        q3 = qx * M[6] + qy * M[7] + qz * M[8]

        return (q1, q2, q3)

    def get_axis_labels(self):
        return 'q1', 'q2', 'q3'

    def parse_config(self, config):
        super(QTransformation, self).parse_config(config)
        self.config.matrix = util.parse_tuple(config.pop('matrix'), length=9, type=float)

class ID03Input(backend.InputBase):
    # OFFICIAL API

    dbg_scanno = None
    dbg_pointno = None

    def generate_jobs(self, command):
        scans = util.parse_multi_range(','.join(command).replace(' ', ','))
        if not len(scans):
            sys.stderr.write('error: no scans selected, nothing to do\n')
        for scanno in scans:
            util.status('processing scan {0}...'.format(scanno))
            if self.config.wait_for_data:
                for job in self.get_delayed_jobs(scanno):
                    yield job
            else:
                scan = self.get_scan(scanno)
                if self.config.pr:
                    pointcount = self.config.pr[1] - self.config.pr[0] + 1
                    start = self.config.pr[0]
                else:
                    start = 0
                    try:
                        pointcount = scan.lines()
                    except specfile.error:  # no points
                        continue
                next(self.get_images(scan, 0, pointcount-1, dry_run=True))  # dryrun
                if pointcount > self.config.target_weight * 1.4:
                    for s in util.chunk_slicer(pointcount, self.config.target_weight):
                        yield backend.Job(scan=scanno, firstpoint=start+s.start, lastpoint=start+s.stop-1, weight=s.stop-s.start)
                else:
                    yield backend.Job(scan=scanno, firstpoint=start, lastpoint=start+pointcount-1, weight=pointcount)

    def get_delayed_jobs(self, scanno):
        scan = self.get_delayed_scan(scanno)

        if self.config.pr:
            firstpoint, lastpoint = self.config.pr  # firstpoint is the first index to be included, lastpoint the last index to be included.
        else:
            firstpoint, lastpoint = 0, self.target(scan) - 1

        pointcount = lastpoint - firstpoint + 1

        if self.is_zap(scan):  # wait until the scan is finished.
            if not self.wait_for_points(scanno, self.target(scan), timeout=self.config.timeout):  # wait for last datapoint
                for s in util.chunk_slicer(pointcount, self.config.target_weight):
                    yield backend.Job(scan=scanno, firstpoint=firstpoint+s.start, lastpoint=firstpoint+s.stop-1, weight=s.stop-s.start)
            else:
                raise errors.BackendError('Image collection timed out. Zapscan was probably aborted')
        elif lastpoint >= 0:  # scanlength is known
            for s in util.chunk_slicer(pointcount, self.config.target_weight):
                if self.wait_for_points(scanno, firstpoint + s.stop, timeout=self.config.timeout):
                    stop = self.get_scan(scanno).lines()
                    yield backend.Job(scan=scanno, firstpoint=firstpoint+s.start, lastpoint=stop-1, weight=s.stop-s.start)
                    break
                else:
                    yield backend.Job(scan=scanno, firstpoint=firstpoint+s.start, lastpoint=firstpoint+s.stop-1, weight=s.stop-s.start)
        else:  # scanlength is unknown
            step = int(self.config.target_weight / 1.4)
            for start, stop in zip(itertools.count(0, step), itertools.count(step, step)):
                if self.wait_for_points(scanno, stop, timeout=self.config.timeout):
                    stop = self.get_scan(scanno).lines()
                    yield backend.Job(scan=scanno, firstpoint=start, lastpoint=stop-1, weight=stop-start)
                    break
                else:
                    yield backend.Job(scan=scanno, firstpoint=start, lastpoint=stop-1, weight=stop-start)

    def process_job(self, job):
        super(ID03Input, self).process_job(job)
        scan = self.get_scan(job.scan)
        self.metadict = dict()
        try:
            scanparams = self.get_scan_params(scan)  # wavelength, UB
            pointparams = self.get_point_params(scan, job.firstpoint, job.lastpoint)  # 2D array of diffractometer angles + mon + transm
            images = self.get_images(scan, job.firstpoint, job.lastpoint)  # iterator!

            for pp, image in zip(pointparams, images):
                yield self.process_image(scanparams, pp, image)
            util.statuseol()
        except Exception as exc:
            exc.args = errors.addmessage(exc.args, ', An error occured for scan {0} at point {1}. See above for more information'.format(self.dbg_scanno, self.dbg_pointno))
            raise
        self.metadata.add_section('id03_backend', self.metadict)

    def parse_config(self, config):
        super(ID03Input, self).parse_config(config)
        self.config.xmask = util.parse_multi_range(config.pop('xmask', None))  # Optional, select a subset of the image range in the x direction. all by default
        self.config.ymask = util.parse_multi_range(config.pop('ymask', None))  # Optional, select a subset of the image range in the y direction. all by default
        self.config.specfile = config.pop('specfile')  # Location of the specfile
        self.config.imagefolder = config.pop('imagefolder', None)  # Optional, takes specfile folder tag by default
        self.config.pr = config.pop('pr', None)  # Optional, all range by default
        self.config.background = config.pop('background', None)  # Optional, if supplied a space of this image is constructed
        self.config.th_offset = float(config.pop('th_offset', 0))  # Optional; Only used in zapscans, zero by default.
        self.config.wavelength = config.pop('wavelength', None)  # Optional; Overrides wavelength from specfile.
        if self.config.wavelength is not None:
            self.config.wavelength = float(self.config.wavelength)
        if self.config.xmask is None:
            self.config.xmask = slice(None)
        if self.config.ymask is None:
            self.config.ymask = slice(None)
        self.config.maskmatrix = load_matrix(config.pop('maskmatrix', None))  # Optional, if supplied pixels where the mask is 0 will be removed
        if self.config.pr:
            self.config.pr = util.parse_tuple(self.config.pr, length=2, type=int)
        self.config.sdd = float(config.pop('sdd'))  # sample to detector distance (mm)
        self.config.pixelsize = util.parse_tuple(config.pop('pixelsize'), length=2, type=float)  # pixel size x/y (mm) (same dimension as sdd)
        self.config.wait_for_data = util.parse_bool(config.pop('wait_for_data', 'false'))  # Optional, if true wait until the data appears
        self.config.timeout = int(config.pop('timeout', 180))  # Optional, how long the script wait until it assumes the scan is not continuing

    def get_destination_options(self, command):
        if not command:
            return False
        command = ','.join(command).replace(' ', ',')
        scans = util.parse_multi_range(command)
        return dict(first=min(scans), last=max(scans), range=','.join(str(scan) for scan in scans))

    # CONVENIENCE FUNCTIONS
    def get_scan(self, scannumber):
        spec = specfilewrapper.Specfile(self.config.specfile)
        return spec.select('{0}.1'.format(scannumber))

    def get_delayed_scan(self, scannumber, timeout=None):
        delay = util.loop_delayer(5)
        start = time.time()
        while 1:
            try:
                return self.get_scan(scannumber)  # reload entire specfile
            except specfile.error:
                if timeout is not None and time.time() - start > timeout:
                    raise errors.BackendError('Scan timed out. There is no data to process')
                else:
                    util.status('waiting for scan {0}...'.format(scannumber))
                    next(delay)

    def wait_for_points(self, scannumber, stop, timeout=None):
        delay = util.loop_delayer(1)
        start = time.time()

        while 1:
            scan = self.get_scan(scannumber)
            try:
                if scan.lines() >= stop:
                    next(delay)  # time delay between specfile and edf file
                    return False
            except specfile.error:
                pass
            finally:
                next(delay)

            util.status('waiting for scan {0}, point {1}...'.format(scannumber, stop))
            if (timeout is not None and time.time() - start > timeout) or self.is_aborted(scan):
                try:
                    util.statusnl('scan {0} aborted at point {1}'.format(scannumber, scan.lines()))
                    return True
                except specfile.error:
                    raise errors.BackendError('Scan was aborted before images were collected. There is no data to process')

    def target(self, scan):
        if any(tuple(scan.command().startswith(pattern) for pattern in ['hklscan', 'a2scan', 'ascan', 'ringscan'])):
            return int(scan.command().split()[-2]) + 1
        elif scan.command().startswith('mesh'):
            return int(scan.command().split()[-6]) * int(scan.command().split()[-2]) + 1
        elif scan.command().startswith('loopscan'):
            return int(scan.command().split()[-3])
        elif scan.command().startswith('xascan'):
            params = numpy.array(scan.command().split()[-6:]).astype(float)
            return int(params[2] + 1 + (params[4] - 1) / params[5] * params[2])
        elif self.is_zap(scan):
            return int(scan.command().split()[-2])
        else:
            return -1

    @staticmethod
    def is_zap(scan):
        return scan.command().startswith('zap')

    @staticmethod
    def is_aborted(scan):
        for line in scan.header('C'):
            if 'Scan aborted' in line:
                return True
        return False

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

    def get_wavelength(self, G):
        for line in G:
            if line.startswith('#G4'):
                return float(line.split(' ')[4])
        return None

    # MAIN LOGIC
    def get_scan_params(self, scan):
        self.dbg_scanno = scan.number()
        if self.is_zap(scan):
            # zapscans don't contain the UB matrix, this needs to be fixed at ID03
            scanno = scan.number()
            UB = None
            while 1:  # look back in spec file to locate a UB matrix
                try:
                    ubscan = self.get_scan(scanno)
                except specfilewrapper.specfile.error:
                    break
                try:
                    UB = numpy.array(ubscan.header('G')[2].split(' ')[-9:], dtype=numpy.float)
                except:
                    scanno -= 1
                else:
                    break
            if UB is None:
                # fall back to UB matrix from the configfile
                if not self.config.UB:
                    raise errors.ConfigError('UB matrix must be specified in configuration file when processing zapscans')
                UB = numpy.array(self.config.UB)
        else:
            UB = numpy.array(scan.header('G')[2].split(' ')[-9:], dtype=numpy.float)

        if self.config.wavelength is None:
            wavelength = self.get_wavelength(scan.header('G'))
            if wavelength is None or wavelength == 0:
                raise errors.BackendError('No or incorrect wavelength specified in the specfile. Please add wavelength to the configfile in the input section')
        else:
            wavelength = self.config.wavelength
    
        self.metadict['UB'] = UB
        self.metadict['wavelength'] = wavelength

        return wavelength, UB

    def get_images(self, scan, first, last, dry_run=False):
        if self.config.background:
            if not os.path.exists(self.config.background):
                raise errors.FileError('could not find background file {0}'.format(self.config.background))
            if dry_run:
                yield
            else:
                edf = EdfFile.EdfFile(self.config.background)
                for i in range(first, last+1):
                    self.dbg_pointno = i
                    yield edf.GetData(0)
        else:
            if self.is_zap(scan):
                scanheaderC = scan.header('C')
                zapscanno = int(scanheaderC[2].split(' ')[-1])  # is different from scanno should be changed in spec!
                try:
                    uccdtagline = scanheaderC[0]
                    UCCD = os.path.split(uccdtagline.split()[-1])
                except:
                    print('warning: UCCD tag not found, use imagefolder for proper file specification')
                    UCCD = []
                pattern = self._get_pattern(UCCD)
                matches = self.find_edfs(pattern, zapscanno)
                if 0 not in matches:
                    raise errors.FileError('could not find matching edf for zapscannumber {0} using pattern {1}'.format(zapscanno, pattern))
                if dry_run:
                    yield
                else:
                    edf = EdfFile.EdfFile(matches[0])
                    for i in range(first, last+1):
                        self.dbg_pointno = i
                        yield edf.GetData(i)

            else:
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
                        self.dbg_pointno = i
                        edf = EdfFile.EdfFile(matches[i])
                        yield edf.GetData(0)

    def _get_pattern(self, UCCD):
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


class EH1(ID03Input):
    monitor_counter = 'mon'

    def parse_config(self, config):
        super(EH1, self).parse_config(config)
        self.config.centralpixel = util.parse_tuple(config.pop('centralpixel'), length=2, type=int)  # x,y
        self.config.hr = config.pop('hr', None)  # Optional, hexapod rotations in miliradians. At the entered value the sample is assumed flat, if not entered the sample is assumed flat at the spec values.
        self.config.UB = config.pop('ub', None)  # Optional, takes specfile matrix by default
        if self.config.UB:
            self.config.UB = util.parse_tuple(self.config.UB, length=9, type=float)
        if self.config.hr:
            self.config.hr = util.parse_tuple(self.config.hr, length=2, type=float)

    def process_image(self, scanparams, pointparams, image):
        gamma, delta, theta, chi, phi, mu, mon, transm, hrx, hry = pointparams
        wavelength, UB = scanparams

        weights = numpy.ones_like(image)

        if self.config.hr:
            zerohrx, zerohry = self.config.hr
            chi = (hrx - zerohrx) / numpy.pi * 180. / 1000
            phi = (hry - zerohry) / numpy.pi * 180. / 1000

        if self.config.background:
            data = image / mon
        else:
            data = image / mon / transm

        if mon == 0:
            raise errors.BackendError('Monitor is zero, this results in empty output. Scannumber = {0}, pointnumber = {1}. Did you forget to open the shutter?'.format(self.dbg_scanno, self.dbg_pointno))

        util.status('{4}| gamma: {0}, delta: {1}, theta: {2}, mu: {3}'.format(gamma, delta, theta, mu, time.ctime(time.time())))

        # pixels to angles
        pixelsize = numpy.array(self.config.pixelsize)
        sdd = self.config.sdd

        app = numpy.arctan(pixelsize / sdd) * 180 / numpy.pi

        centralpixel = self.config.centralpixel  # (column, row) = (delta, gamma)
        gamma_range = -app[1] * (numpy.arange(data.shape[1]) - centralpixel[1]) + gamma
        delta_range = app[0] * (numpy.arange(data.shape[0]) - centralpixel[0]) + delta

        # masking
        if self.config.maskmatrix is not None:
            if self.config.maskmatrix.shape != data.shape:
                raise errors.BackendError('The mask matrix does not have the same shape as the images')
            weights *= self.config.maskmatrix

        gamma_range = gamma_range[self.config.ymask]
        delta_range = delta_range[self.config.xmask]
        intensity = self.apply_mask(data, self.config.xmask, self.config.ymask)
        weights = self.apply_mask(weights, self.config.xmask, self.config.ymask)

        #polarisation correction
        delta_grid, gamma_grid = numpy.meshgrid(delta_range, gamma_range)
        Pver = 1 - numpy.sin(delta_grid * numpy.pi / 180.)**2 * numpy.cos(gamma_grid * numpy.pi / 180.)**2
        intensity /= Pver

        return intensity, weights, (wavelength, UB, gamma_range, delta_range, theta, mu, chi, phi)

    def get_point_params(self, scan, first, last):
        sl = slice(first, last+1)

        GAM, DEL, TH, CHI, PHI, MU, MON, TRANSM, HRX, HRY = list(range(10))
        params = numpy.zeros((last - first + 1, 10))  # gamma delta theta chi phi mu mon transm
        params[:, CHI] = scan.motorpos('Chi')
        params[:, PHI] = scan.motorpos('Phi')

        try:
            params[:, HRX] = scan.motorpos('hrx')
            params[:, HRY] = scan.motorpos('hry')
        except:
            raise errors.BackendError('The specfile does not accept hrx and hry as a motor label. Have you selected the right hutch? Scannumber = {0}, pointnumber = {1}'.format(self.dbg_scanno, self.dbg_pointno))

        if self.is_zap(scan):
            if 'th' in scan.alllabels():
                th = scan.datacol('th')[sl]
                if len(th) > 1:
                    sign = numpy.sign(th[1] - th[0])
                else:
                    sign = 1
                # correction for difference between back and forth in th motor
                params[:, TH] = th + sign * self.config.th_offset
            else:
                params[:, TH] = scan.motorpos('Theta')

            params[:, GAM] = scan.motorpos('Gam')
            params[:, DEL] = scan.motorpos('Delta')
            params[:, MU] = scan.motorpos('Mu')

            params[:, MON] = scan.datacol('zap_mon')[sl]

            transm = scan.datacol('zap_transm')
            transm[-1] = transm[-2]  # bug in specfile
            params[:, TRANSM] = transm[sl]
        else:
            if 'hrx' in scan.alllabels():
                params[:, HRX] = scan.datacol('hrx')[sl]
            if 'hry' in scan.alllabels():
                params[:, HRY] = scan.datacol('hry')[sl]

            params[:, TH] = scan.datacol('thcnt')[sl]
            params[:, GAM] = scan.datacol('gamcnt')[sl]
            params[:, DEL] = scan.datacol('delcnt')[sl]

            try:
                params[:, MON] = scan.datacol(self.monitor_counter)[sl]  # differs in EH1/EH2
            except:
                raise errors.BackendError('The specfile does not accept {2} as a monitor label. Have you selected the right hutch? Scannumber = {0}, pointnumber = {1}'.format(self.dbg_scanno, self.dbg_pointno, self.monitor_counter))

            params[:, TRANSM] = scan.datacol('transm')[sl]
            params[:, MU] = scan.datacol('mucnt')[sl]

        return params


class EH2(ID03Input):
    monitor_counter = 'Monitor'

    def parse_config(self, config):
        super(EH2, self).parse_config(config)
        self.config.centralpixel = util.parse_tuple(config.pop('centralpixel'), length=2, type=int)  # x,y
        self.config.UB = config.pop('ub', None)  # Optional, takes specfile matrix by default
        if self.config.UB:
            self.config.UB = util.parse_tuple(self.config.UB, length=9, type=float)

    def process_image(self, scanparams, pointparams, image):
        gamma, delta, theta, chi, phi, mu, mon, transm = pointparams
        wavelength, UB = scanparams

        weights = numpy.ones_like(image)

        if self.config.background:
            data = image / mon
        else:
            data = image / mon / transm

        if mon == 0:
            raise errors.BackendError('Monitor is zero, this results in empty output. Scannumber = {0}, pointnumber = {1}. Did you forget to open the shutter?'.format(self.dbg_scanno, self.dbg_pointno))

        util.status('{4}| gamma: {0}, delta: {1}, theta: {2}, mu: {3}'.format(gamma, delta, theta, mu, time.ctime(time.time())))

        # area correction
        sdd = self.config.sdd / numpy.cos(gamma * numpy.pi / 180)
        data *= (self.config.sdd / sdd)**2

        # pixels to angles
        pixelsize = numpy.array(self.config.pixelsize)
        app = numpy.arctan(pixelsize / sdd) * 180 / numpy.pi

        centralpixel = self.config.centralpixel  # (row, column) = (gamma, delta)
        gamma_range = - 1 * app[0] * (numpy.arange(data.shape[0]) - centralpixel[0]) + gamma
        delta_range = app[1] * (numpy.arange(data.shape[1]) - centralpixel[1]) + delta

        # masking
        if self.config.maskmatrix is not None:
            if self.config.maskmatrix.shape != data.shape:
                raise errors.BackendError('The mask matrix does not have the same shape as the images')
            weights *= self.config.maskmatrix

        gamma_range = gamma_range[self.config.xmask]
        delta_range = delta_range[self.config.ymask]
        intensity = self.apply_mask(data, self.config.xmask, self.config.ymask)
        weights = self.apply_mask(weights, self.config.xmask, self.config.ymask)

        intensity = numpy.fliplr(intensity)
        intensity = numpy.rot90(intensity)
        weights = numpy.fliplr(weights)  # TODO: should be done more efficiently. Will prob change with new HKL calculations
        weights = numpy.rot90(weights)

        #polarisation correction
        delta_grid, gamma_grid = numpy.meshgrid(delta_range, gamma_range)
        Phor = 1 - (numpy.sin(mu * numpy.pi / 180.) * numpy.sin(delta_grid * numpy.pi / 180.) * numpy.cos(gamma_grid * numpy.pi / 180.) + numpy.cos(mu * numpy.pi / 180.) * numpy.sin(gamma_grid * numpy.pi / 180.))**2
        intensity /= Phor

        return intensity, weights, (wavelength, UB, gamma_range, delta_range, theta, mu, chi, phi)

    def get_point_params(self, scan, first, last):
        sl = slice(first, last+1)

        GAM, DEL, TH, CHI, PHI, MU, MON, TRANSM = list(range(8))
        params = numpy.zeros((last - first + 1, 8))  # gamma delta theta chi phi mu mon transm
        params[:, CHI] = scan.motorpos('Chi')
        params[:, PHI] = scan.motorpos('Phi')

        if self.is_zap(scan):
            if 'th' in scan.alllabels():
                th = scan.datacol('th')[sl]
                if len(th) > 1:
                    sign = numpy.sign(th[1] - th[0])
                else:
                    sign = 1
                # correction for difference between back and forth in th motor
                params[:, TH] = th + sign * self.config.th_offset
            else:
                params[:, TH] = scan.motorpos('Theta')

            params[:, GAM] = scan.motorpos('Gamma')
            params[:, DEL] = scan.motorpos('Delta')
            params[:, MU] = scan.motorpos('Mu')
            params[:, MON] = scan.datacol('zap_mon')[sl]

            transm = scan.datacol('zap_transm')
            transm[-1] = transm[-2]  # bug in specfile
            params[:, TRANSM] = transm[sl]
        else:
            params[:, TH] = scan.datacol('thcnt')[sl]
            params[:, GAM] = scan.datacol('gamcnt')[sl]
            params[:, DEL] = scan.datacol('delcnt')[sl]

            try:
                params[:, MON] = scan.datacol(self.monitor_counter)[sl]  # differs in EH1/EH2
            except:
                raise errors.BackendError('The specfile does not accept {2} as a monitor label. Have you selected the right hutch? Scannumber = {0}, pointnumber = {1}'.format(self.dbg_scanno, self.dbg_pointno, self.monitor_counter))

            params[:, TRANSM] = scan.datacol('transm')[sl]
            params[:, MU] = scan.datacol('mucnt')[sl]
        return params


class GisaxsDetector(ID03Input):
    monitor_counter = 'mon'

    def process_image(self, scanparams, pointparams, image):
        ccdy, ccdz, theta, chi, phi, mu, mon, transm = pointparams

        weights = numpy.ones_like(image)

        wavelength, UB = scanparams

        if self.config.background:
            data = image / mon
        else:
            data = image / mon / transm

        if mon == 0:
            raise errors.BackendError('Monitor is zero, this results in empty output. Scannumber = {0}, pointnumber = {1}. Did you forget to open the shutter?'.format(self.dbg_scanno, self.dbg_pointno))

        util.status('{4}| ccdy: {0}, ccdz: {1}, theta: {2}, mu: {3}'.format(ccdy, ccdz, theta, mu, time.ctime(time.time())))

        # pixels to angles
        pixelsize = numpy.array(self.config.pixelsize)
        sdd = self.config.sdd

        directbeam = (self.config.directbeam[0] - (ccdy - self.config.directbeam_coords[0]) * pixelsize[0], self.config.directbeam[1] - (ccdz - self.config.directbeam_coords[1]) * pixelsize[1])
        gamma_distance = - pixelsize[1] * (numpy.arange(data.shape[1]) - directbeam[1])
        delta_distance = - pixelsize[0] * (numpy.arange(data.shape[0]) - directbeam[0])

        gamma_range = numpy.arctan2(gamma_distance, sdd) / numpy.pi * 180 - mu
        delta_range = numpy.arctan2(delta_distance, sdd) / numpy.pi * 180

        #sample pixel distance
        spd = numpy.sqrt(gamma_distance**2 + delta_distance**2 + sdd**2)
        data *= spd**2 / sdd

        # masking
        if self.config.maskmatrix is not None:
            if self.config.maskmatrix.shape != data.shape:
                raise errors.BackendError('The mask matrix does not have the same shape as the images')
            weights *= self.config.maskmatrix

        gamma_range = gamma_range[self.config.ymask]
        delta_range = delta_range[self.config.xmask]
        intensity = self.apply_mask(data, self.config.xmask, self.config.ymask)
        weights = self.apply_mask(weights, self.config.xmask, self.config.ymask)

        return intensity, weights, (wavelength, UB, gamma_range, delta_range, theta, mu, chi, phi)

    def parse_config(self, config):
        super(GisaxsDetector, self).parse_config(config)
        self.config.directbeam = util.parse_tuple(config.pop('directbeam'), length=2, type=int)
        self.config.directbeam_coords = util.parse_tuple(config.pop('directbeam_coords'), length=2, type=float)  # Coordinates of ccdy and ccdz at the direct beam position

    def get_point_params(self, scan, first, last):
        sl = slice(first, last+1)

        CCDY, CCDZ, TH, CHI, PHI, MU, MON, TRANSM = list(range(8))
        params = numpy.zeros((last - first + 1, 8))  # gamma delta theta chi phi mu mon transm
        params[:, CHI] = scan.motorpos('Chi')
        params[:, PHI] = scan.motorpos('Phi')
        params[:, CCDY] = scan.motorpos('ccdy')
        params[:, CCDZ] = scan.motorpos('ccdz')

        params[:, TH] = scan.datacol('thcnt')[sl]

        try:
            params[:, MON] = scan.datacol(self.monitor_counter)[sl]  # differs in EH1/EH2
        except:
            raise errors.BackendError('The specfile does not accept {2} as a monitor label. Have you selected the right hutch? Scannumber = {0}, pointnumber = {1}'.format(self.dbg_scanno, self.dbg_pointno, self.monitor_counter))

        params[:, TRANSM] = scan.datacol('transm')[sl]
        params[:, MU] = scan.datacol('mucnt')[sl]
        return params

    def find_edfs(self, pattern, scanno):
        files = glob.glob(pattern)
        ret = {}
        for file in files:
            try:
                filename = os.path.basename(file).split('.')[0]
                scan, point = filename.split('_')[-2:]
                scan, point = int(scan), int(point)
                if scan == scanno and point not in list(ret.keys()):
                    ret[point] = file
            except ValueError:
                continue
        return ret

def load_matrix(filename):
    if filename == None:
        return None
    if os.path.exists(filename):
        ext = os.path.splitext(filename)[-1]
        if ext == '.txt':
            return numpy.array(numpy.loadtxt(filename), dtype=numpy.bool)
        elif ext == '.npy':
            return numpy.array(numpy.load(filename), dtype=numpy.bool)
        elif ext == '.edf':
            return numpy.array(EdfFile.EdfFile(filename).getData(0), dtype=numpy.bool)
        else:
            raise ValueError('unknown extension {0}, unable to load matrix!\n'.format(ext))
    else:
        raise IOError('filename: {0} does not exist. Can not load matrix'.format(filename))
