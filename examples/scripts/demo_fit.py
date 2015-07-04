import numpy
import matplotlib.pyplot as pyplot
import BINoculars
import BINoculars.space

space = BINoculars.load('test.hdf5').slice('delta', slice(-0.035, 0.035)).project('delta').slice('g-m', slice(None, 0.4))

results = []
x = BINoculars.space.get_axis_values(space, 'g+m')

for index, curve in enumerate(BINoculars.space.iterate_over_axis(space, 'g+m')):
    try:
        fit = BINoculars.fitspace(curve, 'lorentzian')
        results.append(numpy.append(x[index], fit.result))
        #pyplot.figure()
        #BINoculars.plotspace(curve, fit = fit.fitdata, log = False)
        #pyplot.savefig('fit_{0}'.format(index))
        #pyplot.close()
    except Exception as e:
        print 'warning: fit {0} failed: {1}'.format(index, e)

result = numpy.vstack(results)
numpy.savetxt('fitdata.txt', result)


