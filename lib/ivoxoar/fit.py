import numpy
import scipy.optimize, scipy.fftpack
from scipy.special import wofz
import inspect
import re
import pdb

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


def gaussian(x, (x0, I, sigma, offset, slope)):
    return I * numpy.exp(-((x-x0)/sigma)**2/2) + offset + x * slope

def lorentzian(x, (x0, I, gamma, offset, slope)):
    return I * gamma**2 / ((x - x0)**2 + gamma**2)+ offset + x * slope

def voigt(x, (x0, I, sigma, gamma, offset, slope)):
    z = (x - x0 + numpy.complex(0, gamma)) / (sigma * numpy.sqrt(2))
    return I * numpy.real(wofz(z))/(sigma * numpy.sqrt(2 *  numpy.pi)) + offset + x * slope

def lin(x,(slope,baseline)):
    return slope*x+baseline

def scurve(x,(x0, slope , height , offset)):
    return height * scipy.special.erf(slope * (x - x0)) + offset

def linfit(xdata,ydata, guess):
    if len(guess) == 0:
        slope = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        baseline = ydata[0]-xdata[0]*slope
        params, variance, msg, summary = simplefit(lin, xdata, ydata, (slope, baseline))
    else:
        params, variance, msg, summary = simplefit(lin, xdata, ydata, guess)
    fit = lin(xdata,params)
    return params, summary , fit

def fitgaussian(xdata, ydata,guess):
    xdata.mask = ydata.mask
    if len(guess) == 0:
        left,right = numpy.where(numpy.bitwise_not(ydata.mask))[0][0],numpy.where(numpy.bitwise_not(ydata.mask))[0][-1] 
        slope = (ydata[right] - ydata[left]) / (xdata[right] - xdata[left])
        offset = numpy.mean((ydata - slope * xdata)[(ydata - slope * xdata) < numpy.median(ydata - slope * xdata)])
        gydata = ydata - slope * xdata - offset
        x0 = xdata[numpy.argmax(gydata)]
        I =  gydata[numpy.argmax(gydata)]
        sigma = (gydata.compressed() > I/2).sum() * (xdata.data[1] - xdata.data[0]) / 2 /2.35
        xdata.mask = ydata.mask
        params, variance, msg, summary = simplefit(gaussian, xdata.compressed(), ydata.compressed(), (x0, I, sigma, offset, slope))
    else:
        params, variance, msg, summary = simplefit(gaussian, xdata.compressed(), ydata.compressed(), guess)
    fit = gaussian(xdata,params)
    return params, summary , fit

def fitscurve(xdata,ydata,guess):
    if len(guess) == 0:
        height = (numpy.max(ydata) - numpy.min(ydata))/2.
        offset = numpy.min(ydata) + height
        sign = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0]) > 0
        if sign:
            x0 = int((xdata[numpy.amin(numpy.where(ydata>offset))] + xdata[numpy.amax(numpy.where(ydata<offset))])/2)
        else:
            x0 = int((xdata[numpy.amax(numpy.where(ydata>offset))] + xdata[numpy.amin(numpy.where(ydata<offset))])/2)
        try:
            slope = (ydata[x0+5] - ydata[x0-5]) / (xdata[x0+5] - xdata[x0-5]) * numpy.sqrt(numpy.pi) / (2 * height)
        except:
            print 'x0 on edge'
            slope = 0.001
        print x0, slope , height , offset
        params, variance, msg, summary = simplefit(scurve, xdata, ydata, (x0, slope , height , offset))
    else:
        params, variance, msg, summary = simplefit(scurve, xdata, ydata, guess)
    fit = scurve(xdata,params)
    return params, summary , fit

def fitlorentzian(space,guess):
    xdata = space.get_grid()[0]
    ydata = space.get_masked()
    if len(guess) == 0:
        offset, max_values, gamma, intensity =  get_lorentzparams(space)
        left,right = numpy.where(numpy.bitwise_not(ydata.mask))[0][0],numpy.where(numpy.bitwise_not(ydata.mask))[0][-1] 
        slope = (ydata[right] - ydata[left]) / (xdata[right] - xdata[left])
        cxdata = numpy.ma.array(xdata, mask = ydata.mask)
        params, variance, paramnames, summary = simplefit(lorentzian, cxdata.compressed(), ydata.compressed(), (max_values[0], intensity, gamma[0], offset, slope))
    else:
        params, variance, paramnames, summary = simplefit(lorentzian, xdata.compressed(), ydata.compressed(), guess)
    fit = lorentzian(xdata,params)
    return params, variance, fit, paramnames

def get_lorentzparams(space):
    xdata = space.get_grid()
    ydata = space.get_masked()
    offset = numpy.mean(ydata[ydata < numpy.ma.median(ydata)])
    max_indices = numpy.unravel_index(numpy.argmax(ydata), ydata.shape)
    max_values = list(grid[max_indices] for grid in xdata)
    intensity = ydata[max_indices] - offset
    skey = [max_values[:]] * space.dimension 
    for i,key in enumerate(skey):
       key[i] = slice(None)
    gamma = tuple((ydata[key].compressed() > intensity/2).sum() * space.axes[i].res / 2 for i,key in enumerate(skey))
    return offset, max_values, gamma, intensity


def fitlorentzian2D(space,guess):
    xdata = space.get_grid()
    ydata = space.get_masked()
    cxdata = tuple(numpy.ma.array(array, mask = ydata.mask).compressed() for array in xdata)
    if len(guess) == 0:
        offset, max_values, gamma, intensity =  get_lorentzparams(space)
        th = 0
        guess = []
        for n in [max_values,intensity,gamma,offset,th]:
            if numpy.iterable(n):
                guess.extend(n)
            else:
                guess.append(n)
        print guess
    params, variance, paramnames, summary = simplefit(lorentzian2Dcart, cxdata, ydata.compressed(), guess)
    fit = numpy.ma.array(lorentzian2Dcart(xdata,params),mask = ydata.mask)
   
    return params, variance,fit, paramnames

def lorentzian2Dpolar((x,y), (x0, y0,I,gammax, gammay, offset ,th)):
    a,b = rot2d(x,y,th)
    a0,b0 = rot2d(x0,y0,th)
    return (A  / (1 + gammax * (a-a0)**2 + gammay * (b-b0)**2) + B)

def lorentzian2Dcart((x,y), (x0, y0, I, gammax, gammay, th , B)):
    a,b = rot2d(x,y,th)
    a0,b0 = rot2d(x0,y0,th)
    return (I  / (1 + ((a-a0)/gammax)**2) * 1 / (1 + ((b-b0)/gammay)**2) + B)

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

def noblorentz3dfit(xdata,ydata, guess):
    params, variance, msg, summary = simplefit(noblorentz3d, xdata, ydata.flatten(), guess)
    return params, summary

  
def fit(space, func, guess = []):
    if space.dimension == 1:
        if func == 'gaussian':
            fitfunc = fitgaussian
        elif func == 'lorentzian':
            fitfunc = fitlorentzian
        elif func == 'voigt':
            fitfunc = fitvoigt
        else:
            raise ValueError('Unknown fit function')
        return fitfunc(space,guess)

    elif space.dimension == 2:
        if func == 'gaussian':
            fitfunc = fitgaussian
        elif func == 'lorentzian':
            fitfunc = fitlorentzian2D
        elif func == 'voigt':
            fitfunc = fitvoigt
        else:
            raise ValueError('Unknown fit function')
        return fitfunc(space,guess)

    elif space.dimension > 2:
        raise ValueError("Cannot plot 3 or higher dimensional spaces, use projections or slices to decrease dimensionality.")
