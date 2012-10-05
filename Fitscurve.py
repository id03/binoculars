import inspect
import re
import random
import numpy
import scipy.optimize, scipy.special


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
    
    result = scipy.optimize.leastsq(func, guess, full_output=True, maxfev=3000)
    
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
    return params, variance, msg, summary


def gaussian(x, (x0, A, sigma, offset, slope)):
	return A * numpy.exp(-((x-x0)/2/sigma)**2) + offset + x * slope

def scurve(x,(x0, slope , height , offset)):
    return height * scipy.special.erf(slope * (x - x0)) + offset

def lin(x,(a,b)):
    print x
    return a * x + b

def fitscurve(xdata,ydata):
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
    return params, summary


def fitlin(xdata,ydata):
    slope = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
    offset = ydata[0] - slope * xdata[0]
    params, variance, msg, summary = simplefit(lin, xdata, ydata, (slope,offset))
    return params, summary

def fitbkg(xdata,ydata):
    try:
        params, summary = fitscurve(xdata,ydata)
        return scurve(xdata,params)
    except:
        params, summary = fitlin(xdata,ydata)
        return lin(xdata,params)

#def fitTwoD(xdata,ydata,zdata):



#def TwoDfunction(x,y,()):
    
    

def fitgaussian(xdata, ydata):
	x0 = xdata[numpy.argmax(ydata)]
	A = numpy.max(ydata)
	gamma = (xdata[-1] - xdata[0]) / 10.
	offset = (ydata[0] + ydata[-1]) / 2.
	slope = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
	params, variance, msg, summary = simplefit(gaussian, xdata, ydata, (x0, A, gamma, offset, slope))
	return params, summary

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    xdata = numpy.arange(100)
    x0 = random.randint(0,100)
    offset = random.randint(-50,50)
    height = random.randint(0,100)
    slope = random.random()
    print x0, slope , height , offset
    ydata = numpy.abs(numpy.arange(-50,50))#scurve(xdata,(x0,slope,height,offset)) + numpy.random.randn(100)
    yfit = fitbkg(xdata, ydata)
    plt.plot(xdata,ydata,'wo')
    plt.plot(xdata,yfit,'r')

    plt.show()
