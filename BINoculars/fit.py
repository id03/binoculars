import numpy
import scipy.optimize, scipy.fftpack
from scipy.special import wofz
import inspect
import re
import pdb
from scipy.stats import linregress

import matplotlib.pyplot as plt


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
        if guess:
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
    def _guess(self):
        iguess = []
        guess = []
        x0= list(self.space.argmax())
        skey = []
        for i in range(self.space.dimension):
            skey.append(x0[:])
            skey[i][i] = slice(None)
        for s in skey:
            iguess.append(self._1Dguess(self.space[tuple(s)]))
        for arg in self.parameters:
            if not arg.isalpha():
                argname = arg.rstrip('0123456789')
                index = int(arg.split(argname)[-1])
                guess.append(iguess[index][argname])
            elif arg in iguess[0].keys():
                guess.append(iguess[0][arg])
            else:
                guess.append(0)
        self.guess = guess
 
    def _1Dguess(self, space):
        xdata, ydata, cxdata, cydata = self._prepare(space)
        xdata = xdata[0]
        cxdata = cxdata[0]
        background = cydata < numpy.median(cydata)
        slope, offset, r , p, std = linregress(cxdata[background], cydata[background])
        loc = cxdata[cydata.argmax()]
        I =  cydata.max() - loc * slope - offset
        gydata = ydata - xdata * slope - offset
        gamma = (gydata.compressed() > I/2).sum() * space.axes[0].res / 2 
        return {'I':I,'gamma':gamma,'slope':slope,'offset':offset,'loc':loc}


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
    def func((x,), (I, loc ,gamma, offset, slope)):
        return I * gamma**2 / ((x - loc)**2 + gamma**2)+ offset + x * slope

class Lorentzian2D(PeakFitBase):
    @staticmethod
    def func((x,y), (I, loc0, loc1, gamma0, gamma1, offset, th)):
        a,b = rot2d(x,y,th)
        a0,b0 = rot2d(loc0,loc1,th)
        return (I  / (1 + ((a-a0)/gamma0)**2) * 1 / (1 + ((b-b0)/gamma1)**2) + offset)

class Lorentzian(AutoDimensionFit):
    dimensions = {1: Lorentzian1D, 2: Lorentzian2D}

class Gaussian1D(PeakFitBase):
    @staticmethod
    def func((x,), (loc, I, sigma, offset, slope)):
        return I * numpy.exp(-((x-loc)/sigma)**2/2) + offset + x * slope

class Voigt1D(PeakFitBase):
    @staticmethod
    def func((x,), (loc, I, sigma, gamma, offset, slope)):
        z = (x - loc + numpy.complex(0, gamma)) / (sigma * numpy.sqrt(2))
        return I * numpy.real(wofz(z))/(sigma * numpy.sqrt(2 *  numpy.pi)) + offset + x * slope
