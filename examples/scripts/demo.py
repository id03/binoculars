import binoculars

# To load a binoculars space file in a python script use the load function.
# You can supply a key if you do not want to load the entire space. By default
# the space is loaded entirely.
space = binoculars.load('test.hdf5')

# If you quickly want to see what the content of the space is you can print
# the space. This will provide information about the dimension, the labels,
# and the range of the space, but not of the data in the space.
print(space)

# You can slice (cut) a space. This will return a new space with the limits
# set by the sliced values. The sliced values are given in terms of the label
# values, not in terms of the indices of the space. You can use the same
# language as is used in slicing numpy arrays.
newspace = space[-0.01:0.01, -0.05:0.05, :]
# this is equivalent to
newspace = space.slice('qx', slice(-0.01, 0.01)).slice('qy', slice(-0.05, 0.05))

# You can project the space on its axes by supplying an arbitrary number of
# axes in the project function.
specular = newspace.project('qx', 'qy')

# To explore the data inside a space you can call the get_norm_intensity()
# function. This will return you the data as an ndimensional masked array with
# the values masked where there are no datapoints.
data = space.get_norm_intensity()
# There is also get_norm_variances() to get the masked variances. And the
# non normalized data with
data = space.get_masked_photons()
data = space.get_masked_contributions()
data = space.get_masked_variances()

# If you need the underlying coordinates of the data you can extract them with
grid = space.get_grid()

# You can add two spaces provided they have the same labels and the same
# resolution. The ranges of the spaces can be arbitrary.
total = binoculars.load('test.hdf5') + binoculars.load('test1.hdf5')

# You can view the configuration settings used to create the space.
binoculars.info(space)

# If you want to extract the configuration file use totxtfile()
space.config.totxtfile('config.txt')

# You can save the space in another format by changing the extension,
# currently only .txt and .hdf5 supported, and .edf if PyMca is installed
binoculars.save('test.txt', space)

# You can plot a space in a python script, or in an interactive terminal
# using the binoculars.plotspace function. This function automatically puts
# the right coordinates on the axes. Check the advanced options by typing
# 'help(binoculars.plotspace)'
import matplotlib.pyplot as plt
binoculars.plotspace(specular)
plt.show()
