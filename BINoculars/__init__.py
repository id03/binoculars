import os,sys

# for scripted useage
def run(args):    
    '''Parameters
        args: string
            String as if typed in terminal. The string must consist
            of the location of the configuration file and the command
            for specifying the jobs that need to be processed. 
            All additonal configuration file overides can be included

        Returns
        A BINoculars space

        Examples:
        >>> space = BINoculars.run('config.txt 10')
        >>> space
        Axes (3 dimensions, 2848 points, 33.0 kB) {
            Axis qx (min=-0.01, max=0.0, res=0.01, count=2)
            Axis qy (min=-0.04, max=-0.01, res=0.01, count=4)
            Axis qz (min=0.48, max=4.03, res=0.01, count=356)
        }
    '''

    import BINoculars.main
    BINoculars.util.register_python_executable(__file__)
    main = BINoculars.main.Main.from_args(args.split(' '))
    if isinstance(main.result, BINoculars.space.Space):
        return main.result
    if type(main.result) == bool:
        filename = main.dispatcher.config.destination.final_filename()
        return BINoculars.space.Space.fromfile(filename)

def load(filename):
    ''' Parameters
        filename: string
            Only hdf5 files are acceptable

        Returns
        A BINoculars space

        Examples:
        >>> space = BINoculars.load('test.hdf5')
        >>> space
        Axes (3 dimensions, 2848 points, 33.0 kB) {
            Axis qx (min=-0.01, max=0.0, res=0.01, count=2)
            Axis qy (min=-0.04, max=-0.01, res=0.01, count=4)
            Axis qz (min=0.48, max=4.03, res=0.01, count=356)
        }
    '''
    import BINoculars.space
    if os.path.exists(filename):
        return BINoculars.space.Space.fromfile(filename)
    else:
        raise IOError("File '{0}' does not exist".format(filename))

def save(filename, space):
    '''
        Save a space to file        

        Parameters
        filename: string
            filename to which the data is saved.
        space: BINoculars space
            the space containing the data that needs to be saved

        Examples:
        >>> space
        Axes (3 dimensions, 2848 points, 33.0 kB) {
            Axis qx (min=-0.01, max=0.0, res=0.01, count=2)
            Axis qy (min=-0.04, max=-0.01, res=0.01, count=4)
            Axis qz (min=0.48, max=4.03, res=0.01, count=356)
        }
        >>> BINoculars.save('test.hdf5', space)
    '''

    import BINoculars.space
    if isinstance(space, BINoculars.space.Space):
        space.tofile(filename)
    else:
        raise TypeError("'{0!r}' is not a BINoculars space".format(space))

def plot(space, log=True, clipping=0.0, fit=None, norm=None, colorbar=True, labels=True, **plotopts):
    '''         
        plots a space with the correct axes. The space can be either one or two dimensinal.
        
        Parameters
        space: BINoculars space
            the space containing the data that needs to be plotted
        log: bool
            axes or colorscale logarithmic
        clipping: 0 < float < 1
            cuts a lowest and highst value on the color scale
        fit: numpy.ndarray
            same shape and the space. If one dimensional the fit will be overlayed.
        norm: matplotlib.colors
            object defining the colorscale
        colorbar: bool
            show or not show the colorbar
        labels: bool
            show or not show the labels
        plotopts: keyword arguments
            keywords that will be accepted by matplotlib.pyplot.plot or matplotlib.pyplot.imshow

        Examples:
        >>> space
        Axes (3 dimensions, 2848 points, 33.0 kB) {
            Axis qx (min=-0.01, max=0.0, res=0.01, count=2)
            Axis qy (min=-0.04, max=-0.01, res=0.01, count=4)
            Axis qz (min=0.48, max=4.03, res=0.01, count=356)
        }
        >>> BINoculars.plot('test.hdf5')
    '''

    import matplotlib.pyplot as pyplot
    import BINoculars.plot, BINoculars.space

    if isinstance(space, BINoculars.space.Space):
        BINoculars.plot.plot(space, pyplot.gcf(), pyplot.gca(), log=log, clipping=clipping, fit=fit, norm=norm, colorbar=colorbar, labels=labels, **plotopts)
    else:
        raise TypeError("'{0!r}' is not a BINoculars space".format(space))

def transform(space, labels, resolutions, exprs):
    ''' 
        transformation of the coordinates.
        
        Parameters
        space: BINoculars space
        labels: list
            a list of length N with the labels
        resolutions: list
            a list of length N with the resolution per label
        exprs: list
            a list of length N with strings containing the expressions that will be evaluated.
            all numpy funtions can be called without adding 'numpy.' to the functions.

        Returns
        A BINoculars space of dimension N with labels and resolutions specified in the input

        Examples:
        >>> space = BINoculars.load('test.hdf5')
        >>> space
        Axes (3 dimensions, 2848 points, 33.0 kB) {
            Axis qx (min=-0.01, max=0.0, res=0.01, count=2)
            Axis qy (min=-0.04, max=-0.01, res=0.01, count=4)
            Axis qz (min=0.48, max=4.03, res=0.01, count=356)
        }
        >>> newspace = BINoculars.transform(space, ['twotheta'], [0.003], ['2 * arcsin(0.51 * (sqrt(qx**2 + qy**2 + qz**2) / (4 * pi)) / (pi * 180))'])
        >>> newspace
        Axes (1 dimensions, 152 points, 1.0 kB) {
            Axis twotheta (min=0.066, max=0.519, res=0.003, count=152)
        }
    '''
    import BINoculars.util, BINoculars.space
    if isinstance(space, BINoculars.space.Space):
        transformation = BINoculars.util.transformation_from_expressions(space, exprs)
        newspace = space.transform_coordinates(resolutions, labels, transformation)
    else:
        raise TypeError("'{0!r}' is not a BINoculars space".format(space))
    return newspace


