import numpy as np
import scipy.optimize
import scipy.special
import re

class FitBase(object):
    """Important attributes:
        xdata       2-tuple of 2D arrays with all x-values of the space (up to axis limits)
        ydata       masked 2D-array of space (mask provided by space)
        cxdata      2-tuple of 1D arrays with x-values from data points (all
                    points not masked in ydata, this is most likely less then xdata)
        cydata      1D array with intensities from the smallest rectangle
                    containing data points (as defined by cxdata)"""
    parameters = None
    guess = None
    result = None
    summary = None
    fitdata = None

    def __init__(self, space, guess=None):
        self.space = space
        args = self.get_param_names()

        if space.dimension != len(args[0]):
            raise ValueError('dimension mismatch: space has {0}, {1.__class__.__name__} expects {2}'.format(space.dimension, self, len(args[0])))
        self.parameters = args[1]

        self.xdata, self.ydata, self.cxdata, self.cydata = self._prepare(self.space)
        if guess is not None:
            if len(guess) != len(self.parameters):
                raise ValueError('invalid number of guess parameters {0!r} for {1!r}'.format(guess, self.parameters))
            self.guess = guess
        else:
            self._guess()
        self.success = self._fit()

    @staticmethod
    def _prepare(space):
        ydata = space.get_norm_intensity()
        cydata = ydata.compressed()
        imask = ~ydata.mask
        xdata = space.get_grid()
        cxdata = tuple(d[imask] for d in xdata)
        return xdata, ydata, cxdata, cydata

    def _guess(self):
        # the implementation should use space and/or self.xdata/self.ydata
        # and/or the cxdata/cydata maskless versions to obtain guess
        raise NotImplementedError

    def _fitfunc(self, params):
        return self.cydata - self.func(self.cxdata, params)

    def _fit(self):
        # check here if the number of data points is greater than the number of fit parameters
        ndatapoints = np.count_nonzero(self.cydata)
        if ndatapoints < len(self.parameters):
            print('not enough data points ({}) to fit function ({}) with {} parameters'.format(ndatapoints, self._fitfunc, len(self.parameters)))
            return False
        result = scipy.optimize.leastsq(self._fitfunc, self.guess, full_output=True, epsfcn=1e-6)

        self.message = re.sub('\s{2,}', ' ', result[3].strip()) # cause of failure
        self.result = result[0]             # best parameter values
        errdata = result[2]['fvec']         # best function value
        if result[1] is None:
            self.variance = np.zeros(len(self.result))
        else:
            self.variance = np.diagonal(result[1] * (errdata**2).sum() / (len(errdata) - len(self.result)))

        self.fitdata = np.ma.array(self.func(self.xdata, self.result), mask=self.ydata.mask)
        self.summary = '\n'.join('{}: {:.4g} +/- {:.4g}'.format(p, r, v) for (p, r, v) in zip(self.parameters, self.result, self.variance))

        return result[4] in (1, 2, 3, 4)  # corresponds to True on success, False on failure

    def __str__(self):
        return '{0.__class__.__name__} fit on {1}\n{2}\n{3}'.format(self, self.space, self.message, self.summary)


class PeakFitBase(FitBase):
    def __init__(self, space, guess=None, loc=None):
        if loc is not None:
            self.argmax = tuple(loc)
        else:
            self.argmax = None
        super(PeakFitBase, self).__init__(space, guess)

    def _guess(self):
        maximum = self.cydata.max()  # for background determination

        background = self.cydata < (np.median(self.cydata) + maximum) / 2

        if any(background == True):  # the fit will fail if background is flat for all
            linparams = self._linfit(list(grid[background] for grid in self.cxdata), self.cydata[background])
        else:
            linparams = np.zeros(len(self.cxdata) + 1)

        simbackground = linparams[-1] + np.sum(np.vstack([param * grid.flatten() for (param, grid) in zip(linparams[:-1], self.cxdata)]), axis=0)
        signal = self.cydata - simbackground

        if self.argmax is not None:
            argmax = self.argmax
        else:
            argmax = tuple((signal * grid).sum() / signal.sum() for grid in self.cxdata)

        argmax_bkg = linparams[-1] + np.sum(np.vstack([param * grid.flatten() for (param, grid) in zip(linparams[:-1], argmax)]))

        try:
            maximum = self.space[tuple(float(l) for l in argmax)] - argmax_bkg
        except ValueError:
            maximum = self.cydata.max()

        if np.isnan(maximum):
            maximum = self.cydata.max()
        self.set_guess(maximum, argmax, linparams)

    def _linfit(self, coordinates, intensity):
        coordinates = list(coordinates)
        coordinates.append(np.ones_like(coordinates[0]))
        matrix = np.vstack([coords.flatten() for coords in coordinates]).T
        return np.linalg.lstsq(matrix, intensity, rcond=-1)[0]


class AutoDimensionFit(FitBase):
    def __new__(cls, space, guess=None):
        if space.dimension in cls.dimensions:
            return cls.dimensions[space.dimension](space, guess)
        else:
            raise TypeError('{0}-dimensional space not supported for {1.__name__}'.format(space.dimension, cls))


# utility functions
def rot2d(x, y, th):
    xrot = x * np.cos(th) + y * np.sin(th)
    yrot = - x * np.sin(th) + y * np.cos(th)
    return xrot, yrot


def rot3d(x, y, z, th, ph):
    xrot = np.cos(th) * x + np.sin(th) * np.sin(ph) * y + np.sin(th) * np.cos(ph) * z
    yrot = np.cos(ph) * y - np.sin(ph) * z
    zrot = -np.sin(th) * x + np.cos(th) * np.sin(ph) * y + np.cos(th) * np.cos(ph) * z
    return xrot, yrot, zrot


def get_class_by_name(name):
    options = {}
    for k, v in globals().items():
        if isinstance(v, type) and issubclass(v, FitBase):
            options[k.lower()] = v
    if name.lower() in options:
        return options[name.lower()]
    else:
        raise ValueError("unsupported fit function '{0}'".format(name))


# fitting functions
class Lorentzian1D(PeakFitBase):
    @staticmethod
    def func(grid, params):
        (x, ) = grid
        (I, loc, gamma, slope, offset) = params
        return I / ((x - loc)**2 + gamma**2) + offset + x * slope

    def set_guess(self, maximum, argmax, linparams):
        gamma0 = 5 * self.space.axes[0].res  # estimated FWHM on 10 pixels
        self.guess = [maximum, argmax[0], gamma0, linparams[0], linparams[1]]

    def get_param_names(self):
        return (('x', ), ('I', 'loc', 'gamma', 'slope', 'offset'))


class Lorentzian1DNoBkg(PeakFitBase):   # should work now
    @staticmethod
    def func(grid, params):
        (x, ) = grid
        (I, loc, gamma) = params
        return I / ((x - loc)**2 + gamma**2)

    def set_guess(self, maximum, argmax, linparams):
        gamma0 = 5 * self.space.axes[0].res  # estimated FWHM on 10 pixels
        self.guess = [maximum, argmax[0], gamma0]

    def get_param_names(self):
        return (('x', ), ('I', 'loc', 'gamma'))


class PolarLorentzian2Dnobkg(PeakFitBase):
    @staticmethod
    def func(grid, params):
        (x, y) = grid
        (I, loc0, loc1, gamma0, gamma1, th) = params
        a, b = tuple(grid - center for grid, center in zip(rot2d(x, y, th), rot2d(loc0, loc1, th)))
        return (I / (1 + (a / gamma0)**2 + (b / gamma1)**2))

    def set_guess(self, maximum, argmax, linparams):
        gamma0 = self.space.axes[0].res  # estimated FWHM on 10 pixels
        gamma1 = self.space.axes[1].res
        self.guess = [maximum, argmax[0], argmax[1], gamma0, gamma1, 0]

    def get_param_names(self):
        return (('x', 'y'), ('I', 'loc0', 'loc1', 'gamma0', 'gamma1', 'th'))

class PolarLorentzian2D(PeakFitBase):
    @staticmethod
    def func(grid, params):
        (x, y) = grid
        (I, loc0, loc1, gamma0, gamma1, th, slope1, slope2, offset) = params
        a, b = tuple(grid - center for grid, center in zip(rot2d(x, y, th), rot2d(loc0, loc1, th)))
        return (I / (1 + (a / gamma0)**2 + (b / gamma1)**2) + x * slope1 + y * slope2 + offset)

    def set_guess(self, maximum, argmax, linparams):
        gamma0 = self.space.axes[0].res  # estimated FWHM on 10 pixels
        gamma1 = self.space.axes[1].res
        self.guess = [maximum, argmax[0], argmax[1], gamma0, gamma1, 0, linparams[0], linparams[1], linparams[2]]

    def get_param_names(self):
        return (('x', 'y'), ('I', 'loc0', 'loc1', 'gamma0', 'gamma1', 'th', 'slope1', 'slope2', 'offset'))

    def integrate_signal(self):
        return self.func(self.cxdata, (self.result[0], self.result[1], self.result[2], self.result[3], self.result[4], self.result[5], 0, 0, 0)).sum()


class Lorentzian2D(PeakFitBase):
    @staticmethod
    def func(grid, params):
        (x, y) = grid
        (I, loc0, loc1, gamma0, gamma1, th, slope1, slope2, offset) = params
        a, b = tuple(grid - center for grid, center in zip(rot2d(x, y, th), rot2d(loc0, loc1, th)))
        return (I / (1 + (a/gamma0)**2) * 1 / (1 + (b/gamma1)**2) + x * slope1 + y * slope2 + offset)

    def set_guess(self, maximum, argmax, linparams):
        gamma0 = 5 * self.space.axes[0].res  # estimated FWHM on 10 pixels
        gamma1 = 5 * self.space.axes[1].res
        self.guess = [maximum, argmax[0], argmax[1], gamma0, gamma1, 0, linparams[0], linparams[1], linparams[2]]

    def get_param_names(self):
        return (('x', 'y'), ('I', 'loc0', 'loc1', 'gamma0', 'gamma1', 'th', 'slope1', 'slope2', 'offset'))


class Lorentzian2Dnobkg(PeakFitBase):
    @staticmethod
    def func(grid, params):
        (x, y) = grid
        (I, loc0, loc1, gamma0, gamma1, th) = params
        a, b = tuple(grid - center for grid, center in zip(rot2d(x, y, th), rot2d(loc0, loc1, th)))
        return (I / (1 + (a/gamma0)**2) * 1 / (1 + (b/gamma1)**2))

    def set_guess(self, maximum, argmax, linparams):
        gamma0 = 5 * self.space.axes[0].res  # estimated FWHM on 10 pixels
        gamma1 = 5 * self.space.axes[1].res
        self.guess = [maximum, argmax[0], argmax[1], gamma0, gamma1, 0]

    def get_param_names(self):
        return (('x', 'y'), ('I', 'loc0', 'loc1', 'gamma0', 'gamma1', 'th'))

class Lorentzian(AutoDimensionFit):
    dimensions = {1: Lorentzian1D, 2: PolarLorentzian2D}


class Gaussian1D(PeakFitBase):
    @staticmethod
    def func(grid, params):
        (x,) = grid
        (loc, I, sigma, offset, slope) = params
        return I * np.exp(-((x-loc)/sigma)**2/2) + offset + x * slope

    def get_param_names(self):
        return (('x', ), ('loc', 'I', 'sigma', 'offset', 'slope'))


class Voigt1D(PeakFitBase):
    @staticmethod
    def func(grid, params):
        (x, ) = grid
        (I, loc, sigma, gamma, slope, offset) = params
        z = (x - loc + np.complex(0, gamma)) / (sigma * np.sqrt(2))
        return I * np.real(scipy.special.wofz(z))/(sigma * np.sqrt(2 * np.pi)) + offset + x * slope

    def set_guess(self, maximum, argmax, linparams):
        gamma0 = 5 * self.space.axes[0].res  # estimated FWHM on 10 pixels
        self.guess = [maximum, argmax[0], 0.01, gamma0, linparams[0], linparams[1]]

    def get_param_names(self):
        return (('x', ), ('I', 'loc', 'sigma', 'gamma', 'slope', 'offset'))
