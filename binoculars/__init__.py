from __future__ import print_function, with_statement, division

import os
import sys

# for scripted useage
def run(args):
    '''Parameters
        args: string
            String as if typed in terminal. The string must consist
            of the location of the configuration file and the command
            for specifying the jobs that need to be processed.
            All additonal configuration file overides can be included

        Returns
        A tuple of binoculars spaces

        Examples:
        >>> space = binoculars.run('config.txt 10')
        >>> space[0]
        Axes (3 dimensions, 2848 points, 33.0 kB) {
            Axis qx (min=-0.01, max=0.0, res=0.01, count=2)
            Axis qy (min=-0.04, max=-0.01, res=0.01, count=4)
            Axis qz (min=0.48, max=4.03, res=0.01, count=356)
        }
    '''

    import binoculars.main
    binoculars.util.register_python_executable(__file__)
    main = binoculars.main.Main.from_args(args.split(' '))

    if isinstance(main.result, binoculars.space.Multiverse):
        return main.result.spaces
    if type(main.result) == bool:
        filenames = main.dispatcher.config.destination.final_filenames()
        return tuple(binoculars.space.Space.fromfile(fn) for fn in filenames)


def load(filename, key=None):
    ''' Parameters
        filename: string
            Only hdf5 files are acceptable
        key: a tuple with slices in as much dimensions as the space is

        Returns
        A binoculars space

        Examples:
        >>> space = binoculars.load('test.hdf5')
        >>> space
        Axes (3 dimensions, 2848 points, 33.0 kB) {
            Axis qx (min=-0.01, max=0.0, res=0.01, count=2)
            Axis qy (min=-0.04, max=-0.01, res=0.01, count=4)
            Axis qz (min=0.48, max=4.03, res=0.01, count=356)
        }
    '''
    import binoculars.space
    if os.path.exists(filename):
        return binoculars.space.Space.fromfile(filename, key=key)
    else:
        raise IOError("File '{0}' does not exist".format(filename))


def save(filename, space):
    '''
        Save a space to file

        Parameters
        filename: string
            filename to which the data is saved. '.txt', '.hdf5' are supported.
        space: binoculars space
            the space containing the data that needs to be saved

        Examples:
        >>> space
        Axes (3 dimensions, 2848 points, 33.0 kB) {
            Axis qx (min=-0.01, max=0.0, res=0.01, count=2)
            Axis qy (min=-0.04, max=-0.01, res=0.01, count=4)
            Axis qz (min=0.48, max=4.03, res=0.01, count=356)
        }
        >>> binoculars.save('test.hdf5', space)
    '''

    import binoculars.space
    import binoculars.util
    if isinstance(space, binoculars.space.Space):
        ext = os.path.splitext(filename)[-1]
        if ext == '.txt':
            binoculars.util.space_to_txt(space, filename)
        elif ext == '.edf':
            binoculars.util.space_to_edf(space, filename)
        else:
            space.tofile(filename)
    else:
        raise TypeError("'{0!r}' is not a binoculars space".format(space))


def plotspace(space, log=True, clipping=0.0, fit=None, norm=None, colorbar=True, labels=True, **plotopts):
    '''
        plots a space with the correct axes. The space can be either one or two dimensinal.

        Parameters
        space: binoculars space
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
        >>> binoculars.plotspace('test.hdf5')
    '''

    import matplotlib.pyplot as pyplot
    import binoculars.plot
    import binoculars.space

    if isinstance(space, binoculars.space.Space):
        if space.dimension == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = pyplot.gcf().gca(projection='3d')
            return binoculars.plot.plot(space, pyplot.gcf(), ax, log=log, clipping=clipping, fit=None, norm=norm, colorbar=colorbar, labels=labels, **plotopts)
        if fit is not None and space.dimension == 2:
            ax = pyplot.gcf().add_subplot(121)
            binoculars.plot.plot(space, pyplot.gcf(), ax, log=log, clipping=clipping, fit=None, norm=norm, colorbar=colorbar, labels=labels, **plotopts)
            ax = pyplot.gcf().add_subplot(122)
            return binoculars.plot.plot(space, pyplot.gcf(), ax, log=log, clipping=clipping, fit=fit, norm=norm, colorbar=colorbar, labels=labels, **plotopts)
        else:
            return binoculars.plot.plot(space, pyplot.gcf(), pyplot.gca(), log=log, clipping=clipping, fit=fit, norm=norm, colorbar=colorbar, labels=labels, **plotopts)
    else:
        raise TypeError("'{0!r}' is not a binoculars space".format(space))


def transform(space, labels, resolutions, exprs):
    '''
        transformation of the coordinates.

        Parameters
        space: binoculars space
        labels: list
            a list of length N with the labels
        resolutions: list
            a list of length N with the resolution per label
        exprs: list
            a list of length N with strings containing the expressions that will be evaluated.
            all numpy funtions can be called without adding 'numpy.' to the functions.

        Returns
        A binoculars space of dimension N with labels and resolutions specified in the input

        Examples:
        >>> space = binoculars.load('test.hdf5')
        >>> space
        Axes (3 dimensions, 2848 points, 33.0 kB) {
            Axis qx (min=-0.01, max=0.0, res=0.01, count=2)
            Axis qy (min=-0.04, max=-0.01, res=0.01, count=4)
            Axis qz (min=0.48, max=4.03, res=0.01, count=356)
        }
        >>> newspace = binoculars.transform(space, ['twotheta'], [0.003], ['2 * arcsin(0.51 * (sqrt(qx**2 + qy**2 + qz**2) / (4 * pi)) / (pi * 180))'])
        >>> newspace
        Axes (1 dimensions, 152 points, 1.0 kB) {
            Axis twotheta (min=0.066, max=0.519, res=0.003, count=152)
        }
    '''
    import binoculars.util
    import binoculars.space
    if isinstance(space, binoculars.space.Space):
        transformation = binoculars.util.transformation_from_expressions(space, exprs)
        newspace = space.transform_coordinates(resolutions, labels, transformation)
    else:
        raise TypeError("'{0!r}' is not a binoculars space".format(space))
    return newspace


def fitspace(space, function, guess=None):
    '''
        fit the space data.

        Parameters
        space: binoculars space
        function: list
            a string with the name of the desired function. supported are:
            lorentzian (automatically selects 1d or 2d), gaussian1d and voigt1d
        guess: list
            a list of length N with the resolution per label

        Returns
        A binoculars fit object.

        Examples:
        >>> fit = binoculars.fitspace(space, 'lorentzian')
        >>> print(fit.summary)
            I: 1.081e-07 +/- inf
            loc: 0.3703 +/- inf
            gamma: 0.02383 +/- inf
            slope: 0.004559 +/- inf
            offset: -0.001888 +/- inf
        >>> parameters = fit.parameters
        >>> data = fit.fitdata
        >>> binoculars.plotspace(space, fit = data)
    '''

    import binoculars.fit
    if isinstance(space, binoculars.space.Space):
        fitclass = binoculars.fit.get_class_by_name(function)
        return fitclass(space, guess)
    else:
        raise TypeError("'{0!r}' is not a binoculars space".format(space))
    return newspace


def info(filename):
    '''
        Explore the file without loading the file, or after loading the file

        Parameters
        filename: filename or space

        Examples:
        >>> print binoculars.info('test.hdf5')
        Axes (3 dimensions, 46466628 points, 531.0 MB) {
            Axis H (min=-0.1184, max=0.0632, res=0.0008, count=228)
            Axis K (min=-1.1184, max=-0.9136, res=0.0008, count=257)
            Axis L (min=0.125, max=4.085, res=0.005, count=793)
        }
        ConfigFile{
           [dispatcher]
           [projection]
           [input]
        }
        origin = test.hdf5
        >>> space = binoculars.load('test.hdf5')
        >>> print binoculars.info(space)
        Axes (3 dimensions, 46466628 points, 531.0 MB) {
            Axis H (min=-0.1184, max=0.0632, res=0.0008, count=228)
            Axis K (min=-1.1184, max=-0.9136, res=0.0008, count=257)
            Axis L (min=0.125, max=4.085, res=0.005, count=793)
        }
        ConfigFile{
           [dispatcher]
           [projection]
           [input]
        }
        origin = test.hdf5

    '''

    import binoculars.space
    ret = ''
    if isinstance(filename, binoculars.space.Space):
        ret += '{!r}\n{!r}'.format(filename, filename.config)
    elif type(filename) == str:
        if os.path.exists(filename):
            try:
                axes = binoculars.space.Axes.fromfile(filename)
            except Exception as e:
                raise IOError('{0}: unable to load Space: {1!r}'.format(filename, e))
            ret += '{!r}\n'.format(axes)
            try:
                config = binoculars.util.ConfigFile.fromfile(filename)
            except Exception as e:
                raise IOError('{0}: unable to load util.ConfigFile: {1!r}'.format(filename, e))
            ret += '{!r}'.format(config)
        else:
            raise IOError("File '{0}' does not exist".format(filename))
    return ret
