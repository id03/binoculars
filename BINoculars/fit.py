import numpy
import scipy.optimize, scipy.special
import inspect
import re


class FitBase(object):
    parameters = None
    guess = None
    result = None
    summary = None
    fitdata = None

    def __init__(self, space, guess=None):
        self.space  = space
        args = inspect.getargspec(self.func).args
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
        ydata = space.get_masked()
        cydata = ydata.compressed()
        imask = ~ydata.mask
        xdata = space.get_grid()
        cxdata = tuple(d[imask] for d in xdata)
        return xdata, ydata, cxdata, cydata

    def _guess(self):
        # the implementation should use space and/or self.xdata/self.ydata and/or the cxdata/cydata maskless versions to obtain guess
        raise NotImplementedError

    def _fitfunc(self, params):
        return self.cydata - self.func(self.cxdata, params)
 
    def _fit(self):
        result = scipy.optimize.leastsq(self._fitfunc, self.guess, full_output=True)

        self.message = re.sub('\s{2,}', ' ', result[3].strip())
        self.result = result[0]
        errdata = result[2]['fvec']
        if result[1] is None:
            self.variance = numpy.zeros(len(self.result))
        else:
            self.variance = numpy.diagonal(result[1] * (errdata**2).sum() / (len(errdata) - len(self.result)))

        self.fitdata = numpy.ma.array(self.func(self.xdata, self.result), mask=self.ydata.mask)
        self.summary = '\n'.join('%s: %.4g +/- %.4g' % (n, p,v) for (n, p, v) in zip(self.parameters, self.result, self.variance))

        return result[4] in (1,2,3,4) # corresponds to True on success, False on failure

    def __str__(self):
        return '{0.__class__.__name__} fit on {1}\n{2}\n{3}'.format(self, self.space, self.message, self.summary)

class PeakFitBase(FitBase):
    def __init__(self, space, guess = None, loc = None):
        if loc != None:
            self.argmax = tuple(loc)
        else:
            self.argmax = None
        super(PeakFitBase, self).__init__(space, guess)

    def _guess(self):
        maximum = self.cydata.max() # for background determination

        background = self.cydata < (numpy.median(self.cydata) + maximum) / 2 
        linparams = self._linfit(list(grid[background] for grid in self.cxdata), self.cydata[background])

        simbackground = linparams[-1] + numpy.sum(numpy.vstack(param * grid.flatten() for (param, grid) in zip(linparams[:-1], self.cxdata)) , axis = 0)
        signal = self.cydata - simbackground

        if self.argmax != None:
            argmax = self.argmax
        else:
            argmax = tuple((signal * grid).sum() / signal.sum() for grid in self.cxdata)
        
        argmax_bkg = linparams[-1] + numpy.sum(numpy.vstack(param * grid.flatten() for (param, grid) in zip(linparams[:-1], argmax)))

        try:
            maximum = self.space.get_value(argmax) - argmax_bkg
        except ValueError:
            maximum = self.cydata.max() 
        
        if numpy.isnan(maximum):
            maximum = self.cydata.max() 

        self.set_guess(maximum, argmax, linparams)

    def _linfit(self, coordinates, intensity):
        coordinates = list(coordinates)
        coordinates.append(numpy.ones_like(coordinates[0]))
        matrix = numpy.vstack(coords.flatten() for coords in coordinates).T
        return numpy.linalg.lstsq(matrix, intensity)[0]

class AutoDimensionFit(FitBase):
    def __new__(cls, space, guess=None):
        if space.dimension in cls.dimensions:
            return cls.dimensions[space.dimension](space, guess)
        else:
            raise TypeError('{0}-dimensional space not supported for {1.__name__}'.format(space.dimension, cls))


# utility functions
def rot2d(x,y,th):
    xrot = x * numpy.cos(th) + y * numpy.sin(th) 
    yrot = - x * numpy.sin(th) + y * numpy.cos(th) 
    return xrot,yrot

def rot3d(x, y ,z , th, ph):
    xrot = numpy.cos(th) * x + numpy.sin(th) * numpy.sin(ph) * y + numpy.sin(th) * numpy.cos(ph) * z
    yrot = numpy.cos(ph) * y - numpy.sin(ph) * z
    zrot = -numpy.sin(th) * x + numpy.cos(th) * numpy.sin(ph) * y + numpy.cos(th) * numpy.cos(ph) * z
    return xrot, yrot, zrot

def get_class_by_name(name):
    options = {}
    for k, v in globals().iteritems():
        if isinstance(v, type) and issubclass(v, FitBase):
            options[k.lower()] = v
    if name in options:
        return options[name]
    else:
        raise ValueError("unsupported fit function '{0}'".format(name))


# fitting functions
class Lorentzian1D(PeakFitBase):
    @staticmethod
    def func((x,), (I, loc ,gamma, slope, offset)):
        return I / ((x - loc)**2 + gamma**2)+ offset + x * slope

    def set_guess(self, maximum , argmax, linparams):
        gamma0 = 5 * self.space.axes[0].res #estimated FWHM on 10 pixels
        self.guess =  [maximum , argmax[0], gamma0, linparams[0], linparams[1]] 

class PolarLorentzian2D(PeakFitBase):
    @staticmethod
    def func((x,y), (I, loc0, loc1, gamma0, gamma1, th, slope1, slope2, offset)):
        a,b = tuple(grid - center for grid, center in zip(rot2d(x,y,th),rot2d(loc0,loc1,th)))
        return (I  / (1 + (a / gamma0)**2 + (b / gamma1)**2 ) + x * slope1 + y * slope2 + offset)

    def set_guess(self, maximum , argmax, linparams):
        gamma0 = self.space.axes[0].res#estimated FWHM on 10 pixels
        gamma1 = self.space.axes[1].res
        self.guess = [maximum , argmax[0], argmax[1], gamma0, gamma1, 0, linparams[0], linparams[1], linparams[2]]

class Lorentzian2D(PeakFitBase):
    @staticmethod
    def func((x,y), (I, loc0, loc1, gamma0, gamma1, th, slope1, slope2, offset)):
        a,b = tuple(grid - center for grid, center in zip(rot2d(x,y,th),rot2d(loc0,loc1,th)))
        return (I  / (1 + (a/gamma0)**2) * 1 / (1 + (b/gamma1)**2) + x * slope1 + y * slope2 + offset)

    def set_guess(self, maximum , argmax, linparams):
        gamma0 = 5 * self.space.axes[0].res #estimated FWHM on 10 pixels
        gamma1 = 5 * self.space.axes[1].res 
        self.guess = [maximum , argmax[0], argmax[1],gamma0, gamma1, 0, linparams[0], linparams[1], linparams[2]] 

class Lorentzian(AutoDimensionFit):
    dimensions = {1: Lorentzian1D, 2: PolarLorentzian2D}

class Gaussian1D(PeakFitBase):
    @staticmethod
    def func((x,), (loc, I, sigma, offset, slope)):
        return I * numpy.exp(-((x-loc)/sigma)**2/2) + offset + x * slope

class Voigt1D(PeakFitBase):
    @staticmethod
    def func((x,), (loc, I, sigma, gamma, offset, slope)):
        z = (x - loc + numpy.complex(0, gamma)) / (sigma * numpy.sqrt(2))
        return I * numpy.real(scipy.special.wofz(z))/(sigma * numpy.sqrt(2 *  numpy.pi)) + offset + x * slope
