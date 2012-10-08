import inspect
import re
import random
import numpy
import scipy.optimize, scipy.special
from scipy.stats import linregress

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
    
    result = scipy.optimize.leastsq(func, guess, full_output=True, maxfev=3000,epsfcn=0.1)
    
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
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xdata,ydata)
        return slope * xdata + intercept

def fitTwoDscurve((x,y),zdata):
    parameters = numpy.ma.zeros((x.shape[0],4))
    parameters.mask = numpy.zeros_like(parameters.data.shape)
    for n in range(x.shape[0]):
        try:
            params, summary = fitscurve(y,zdata[n,:])
            parameters[n,:] = params
        except:
            parameters.mask[n,:] = True

    slope,height,offset =  numpy.median(parameters[:,1:],axis = 0)
    slope = 0.1 * slope
    x0 = numpy.ma.masked_outside(parameters[:,0],-5 * numpy.median(parameters[:,0]),5 * numpy.median(parameters[:,0]))
    a,b,c,d = cubefit(x,x0)

    params, variance, msg, summary = simplefit(TwoDscurve, (x,y), zdata.flatten(), (a ,b ,c,d,slope,height,offset))
    return params, summary

def cubefit(xdata,ydata):
    guess= (1,1,1,1)
    params, variance, msg, summary = simplefit(cube, xdata, ydata, guess)
    return params

def cube(x,(a,b,c,d)):
    return a * x**3 + b * x**2 + c * x + d

def TwoDscurve((x,y),(a ,b ,c,d,slope,height,offset)):
    parameters = numpy.zeros((x.shape[0],4))
    x0 = a * x**3 + b * x**2 + c * x + d
    parameters[:,0] = x0
    parameters[:,1] = slope
    parameters[:,2] = height
    parameters[:,3] = offset
    fit = numpy.vstack(scurve(y,parameters[n,:])for n in range(x.shape[0]))
    return fit.flatten()    

def fitgaussian(xdata, ydata):
	x0 = xdata[numpy.argmax(ydata)]
	A = numpy.max(ydata)
	gamma = (xdata[-1] - xdata[0]) / 10.
	offset = (ydata[0] + ydata[-1]) / 2.
	slope = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
	params, variance, msg, summary = simplefit(gaussian, xdata, ydata, (x0, A, gamma, offset, slope))
	return params, summary

def bgs_plane(data):
	"""Perform linear fit planar background subtraction.
        
        parameters:
        data: array
        
        returns:
        params: tuple of three floats
        the parameters of the plane a + bx + cy
        data:   array
        data with background subtracted
        """
    
	nx, ny = data.shape
	
	sumxi = (nx-1)/2.;
	sumxixi = (2*nx-1)*(nx-1)/6.;
	sumyi = (ny-1)/2.;
	sumyiyi = (2*ny-1)*(ny-1)/6.;
    
	xgrid, ygrid = numpy.meshgrid(numpy.arange(ny), numpy.arange(nx))
	sumsi = data.mean()
	sumsixi = (data*xgrid).mean()
	sumsiyi = (data*ygrid).mean()
    
	bx = (sumsixi - sumsi*sumxi) / (sumxixi - sumxi*sumxi)
	by = (sumsiyi - sumsi*sumyi) / (sumyiyi - sumyi*sumyi)
	a = sumsi - bx*sumxi - by*sumyi
	return (a, bx, by), data - a - bx*xgrid - by*ygrid


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    zdata = numpy.log(numpy.loadtxt('background.txt'))
    x = numpy.arange(zdata.shape[0])
    y = numpy.arange(zdata.shape[1])
    #slope,fit = bgs_plane(zdata)
    params, summary =  fitTwoDscurve((x,y),zdata)
    print params,summary
    fit = TwoDscurve((x,y),params).reshape(x.shape[0],y.shape[0])
    for n in range(x.shape[0]):
        plt.plot(zdata[n,:],'wo')
        plt.plot(fit[n,:],'r')
        plt.savefig('{0}.png'.format(str(n)))
        plt.close()
    numpy.savetxt('fit.txt',numpy.exp(fit))



