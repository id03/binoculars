import binoculars

# to load a binoculars space file in a python script use the load function. You can supply a key if you do not
# want to load the entire space. By default the space is loaded entirely.
space = binoculars.load('test.hdf5')

# if you quickly want to see what the content of the space is you can print the space. This will provide information about the 
# dimension, the labels, and the range of the space, but not of the data in the space.
print space

# you can slice a space. This will return a new space with the limits set by the sliced values. The sliced values are given in terms of the
# labele values, not in terms of the indices of the space. You can use the same language as is used in slicing numpy arrays. 
newspace = space[-0.01:0.01,-0.05:0.05,:]
# this is equivalent to
newspace = space.slice('qx', slice(-0.01, 0.01)).slice('qy', slice(-0.05, 0.05))

# You can project the space on its axes by supplying an arbitrary number of axes in the project function.
specular = newspace.project('qx', 'qy')

# to explore the data inside a space you can call the get_masked() function. This will return you the data as an ndimensional masked array with 
# the values masked where there are no datapoints.
data = space.get_masked()

# if you need the underlying cooridnates of the data you can ask extract the grid with
grid = space.get_grid()

# you can add two spaces provided they have the same labels and the same resolution. The range of the space can be arbitrary. 
total = binoculars.load('test.hdf5') + binoculars.load('test1.hdf5')

# you can view the configuration settings used to create the space.
binoculars.info(space)

# if you now want to reuse this configuration file as an input you can extract the configuration file by
space.config.totxtfile('config.txt')

# You can save the space in another format by changing the extension, currently only txt and hdf5 supported, and EDF if you have PyMca installed
binoculars.save('test.txt', space)

# you can plot a space in a python script, or in an interactive terminal using the binoculars.plotspace function. This function
# automatically puts the right coordinates on the axes. Check the advanced options by typing 'help(binoculars.plotspace)'
import matplotlib.pyplot as pyplot
binoculars.plotspace(specular)
pyplot.show()
