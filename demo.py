import BINoculars

# to load a binoculars space file in a python script use the load function. You can supply a key if you do not
# want to load the entire space. By default the space is loaded entirely.
space = BINoculars.load('test.hdf5')

# if you quickly want to see what the content of the space is you can print the space. This will provide information about the 
# dimension, the labels, and the range of the space, but not of the data in the space.
print space

# you can slice a space. This will return a new space with the limits set by the sliced values. The sliced values are given in terms of the
# labele values, not in terms of the indices of the space. You can use the same language as is used in slicing numpy arrays. 
newspace = space[-0.01:0.01,-0.01:0.01,:]

# you can project the space on its axes. You supply an arbitrary number of axes in the project function.
specular = newspace.project('qx', 'qy')

# to explore the data inside a space you can call the get_masked() function. This will return you the data as an ndimensional masked array with 
# the values masked where there are no datapoints.
data = space.get_masked()

# you can add two spaces provided they have the same labels and the same resolution. The range of the space can be arbitrary. 
total = BINoculars.load('test.hdf5') + BINoculars.load('test1.hdf5')


# you can view the configuration settings used to create the space.
print space.config

# if you now want to reuse this configuration file as an input you can extract the configuration file by
space.config.totxtfile('filename.txt')




