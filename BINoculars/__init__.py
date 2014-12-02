import os,sys

# for scripted useage
def run(args):# args is string as if typed in terminal
    import BINoculars.main
    BINoculars.util.register_python_executable(__file__)
    main = BINoculars.main.Main.from_args(args.split(' '))
    if isinstance(main.result, BINoculars.space.Space):
        return main.result
    if type(main.result) == bool:
        filename = main.dispatcher.config.destination.final_filename()
        return BINoculars.space.Space.fromfile(filename)

def load(filename):
    import BINoculars.space
    if os.path.exists(filename):
        return BINoculars.space.Space.fromfile(filename)
    else:
        raise IOError("File '{0}' does not exist".format(filename))

def save(filename, space):
    import BINoculars.space
    if isinstance(space, BINoculars.space.Space):
        space.tofile(filename)
    else:
        raise TypeError("'{0!r}' is not a BINoculars space".format(space))

def plot(space, log=True, clipping=0.0, fit=None, norm=None, colorbar=True, labels=True, **plotopts):
    import matplotlib.pyplot as pyplot
    import BINoculars.plot

    if isinstance(space, BINoculars.space.Space):
        BINoculars.plot.plot(space, pyplot.gcf(), pyplot.gca(), log=log, clipping=clipping, fit=fit, norm=norm, colorbar=colorbar, labels=labels, **plotopts)
    else:
        raise TypeError("'{0!r}' is not a BINoculars space".format(space))

def transform(space, labels, resolutions, exprs):
    ''' Parameters
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


