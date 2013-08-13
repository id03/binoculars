import numpy
import scipy.optimize, scipy.fftpack
from scipy.special import wofz
import inspect
import re
import pdb
from scipy.stats import linregress

import matplotlib.pyplot as plt

def simplefit(func, x, y, guess):
	"""Non-linear least squares fit of y=func(x).
        
        Example:
        # note the double brackets in the function definition:
        def gaussian(x, (A, x0, w)):
	    return A*exp(-((x-x0)/w)**2)
        simplefit(gaussian, [-1, 0, 1], [0, 1, 0], [1, 0, 1])
        # This is equivalent to:
        nonlinfit(lambda p: y-func(x, p), guess)
        
        parameters:
        func:  callable
        function to fit, prototype: func(x, (param1, param2, ...))
        x:     array
        x dataset
        y:     array
        y dataset
        guess: tuple or array
        guess of fit parameters
        
        returns:
        params:   tuple of best fit parameters
        variance: variance (error estimate) of fit parameters
        msg:      human-readable message from scipy.optimize.leastsq
        summary:  human-readable summary of fit parameters
        """
    
	args, varargs, varkw, defaults = inspect.getargspec(func)
	paramnames = args[1]
	return nonlinfit(lambda p: y-func(x,p), guess, paramnames)


def nonlinfit(func, guess, paramnames=None):
    """Non-linear least squares optimization. Finds parameters that minimize the sum of squares.

    parameters:
    func:       callable
    function to optimize, prototype: func(params)
    the return value of this function is squared and summed
    guess:      tuple or array
    guess of fit parameters
    paramnames: tuple of strings, optional
    used to build the human-readable summary (see below)

    returns:
    params:   tuple of best fit parameters
    variance: variance (error estimate) of fit parameters
    msg:      human-readable message from scipy.optimize.leastsq
    summary:  human-readable summary of fit parameters
    """
    
    result = scipy.optimize.leastsq(func, guess, full_output=True)

    msg = re.sub('\s{2,}', ' ', result[3].strip())
    if result[4] not in (1,2,3,4):
        raise ValueError("no solution found (%d): %s" % (result[4], msg))

    params = result[0]
    errdata = result[2]['fvec']
    if result[1] is None:
        variance = numpy.zeros(len(params))
    else:
        variance = numpy.diagonal(result[1] * (errdata**2).sum() / (len(errdata) - len(params)))

    if not paramnames:
        paramnames = [str(i+1) for i in range(len(guess))]
    summary = '\n'.join('%s: %.4e +/- %.4e' % (n, p,v) for (n, p,v) in zip(paramnames, params, variance))
    return params, variance, paramnames, summary

def gaussian((x,), (loc, I, sigma, offset, slope)):
	return I * numpy.exp(-((x-loc)/sigma)**2/2) + offset + x * slope

def voigt((x,), (loc, I, sigma, gamma, offset, slope)):
    z = (x - loc + numpy.complex(0, gamma)) / (sigma * numpy.sqrt(2))
    return I * numpy.real(wofz(z))/(sigma * numpy.sqrt(2 *  numpy.pi)) + offset + x * slope

def lorentzian((x,), (I, loc ,gamma, offset, slope)):
    return I * gamma**2 / ((x - loc)**2 + gamma**2)+ offset + x * slope

def lorentzian2Dpolar((x,y), (I, loc0, loc1,gamma0, gamma1, offset ,th)):
    a,b = rot2d(x,y,th)
    a0,b0 = rot2d(loc0,loc1,th)
    return (I  / (1 + gamma0 * (a-a0)**2 + gamma1 * (b-b0)**2) + offset)

def lorentzian2Dcart((x,y), (I, loc0, loc1, gamma0, gamma1, offset, th )):
    a,b = rot2d(x,y,th)
    a0,b0 = rot2d(loc0,loc1,th)
    return (I  / (1 + ((a-a0)/gamma0)**2) * 1 / (1 + ((b-b0)/gamma1)**2) + offset)

def rot2d(x,y,th):
    xrot = x * numpy.cos(th) + y * numpy.sin(th) 
    yrot = - x * numpy.sin(th) + y * numpy.cos(th) 
    return xrot,yrot

def rot3d(x, y ,z , th, ph):
    xrot = numpy.cos(th) * x + numpy.sin(th) * numpy.sin(ph) * y + numpy.sin(th) * numpy.cos(ph) * z
    yrot = numpy.cos(ph) * y - numpy.sin(ph) * z
    zrot = -numpy.sin(th) * x + numpy.cos(th) * numpy.sin(ph) * y + numpy.cos(th) * numpy.cos(ph) * z
    return xrot, yrot, zrot
         
def noblorentz3d((x,y,z), (x0, y0,z0, A, gammax, gammay ,gammaz, th , ph, B)):
    print x0, y0,z0, A, gammax, gammay ,gammaz, th , ph, B
    a,b,c = rot3d(x,y,z,th,ph)
    a0,b0,c0 = rot3d(x0,y0,z0,th,ph)
    sim = (A  / (1 + gammax * (a-a0)**2 + gammay * (b-b0)**2 + gammaz * (c-c0)**2) + B).flatten()
    return sim


class PeakFit(object):
    def __init__(self,space,func = lorentzian2Dcart):
        self.space  = space
        self.func = func
        self.args = inspect.getargspec(func)[0][1]
 
    def fit(self, guess):
        if not len(guess):
            guess = self._get_guess()
        xdata = self.space.get_grid()
        ydata = self.space.get_masked()
        cxdata = tuple(numpy.ma.array(array, mask = ydata.mask).compressed() for array in xdata)
        params, variance, paramnames, summary = simplefit(self.func, cxdata, ydata.compressed(), guess)
        fit = numpy.ma.array(self.func(xdata,params),mask = ydata.mask)
        return params, variance,fit, paramnames

    def _get_guess(self):
       iguess = []
       guess = []
       x0= list(self.space.argmax())
       skey = []
       for i in range(self.space.dimension):
           skey.append(x0[:])
           skey[i][i] = slice(None)
       for s in skey:
           iguess.append(self._get_1Dguess(self.space[tuple(s)]))
       for arg in self.args:
           if not arg.isalpha():
               argname = arg.rstrip('0123456789')
               index = int(arg.split(argname)[-1])
               guess.append(iguess[index][argname])
           elif arg in iguess[0].keys():
               guess.append(iguess[0][arg])
           else:
               guess.append(0)
       return guess
 
    @staticmethod
    def _get_1Dguess(space):
        xdata = space.get_grid()[0]
        ydata = space.get_masked()
        cxdata = numpy.ma.array(xdata, mask = ydata.mask).compressed()
        cydata = ydata.compressed()
        slope,offset,r,p,std = linregress(cxdata[cydata < numpy.median(cydata)],cydata[cydata < numpy.median(cydata)])
        loc = space.argmax()[0]
        I =  space.max() - loc * slope - offset
        gydata = ydata - xdata * slope - offset
        gamma = (gydata.compressed() > I/2).sum() * space.axes[0].res / 2 
        return {'I':I,'gamma':gamma,'slope':slope,'offset':offset,'loc':loc}
      
  
def fit(space, func, guess = []):
    if space.dimension == 1:
        if func == 'lorentzian':
            fit = PeakFit(space,lorentzian)
        elif func == 'gaussian':
            fit = PeakFit(space,gaussian)
        elif func == 'voigt':
            fit = PeakFit(space,voigt)
        else:
            raise ValueError('Unknown fit function')
        return fit.fit(guess)

    elif space.dimension == 2:
        if func == 'lorentzian':
            fit = PeakFit(space)
        else:
            raise ValueError('Unknown fit function')
        return fit.fit(guess)
    elif space.dimension > 2:
        raise ValueError("Cannot plot 3 or higher dimensional spaces, use projections or slices to decrease dimensionality.")
