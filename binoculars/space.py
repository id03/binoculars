from __future__ import unicode_literals

import numbers
import sys
import itertools

import numpy as np

from . import util, errors

#python3 support
PY3 = sys.version_info > (3,)
if PY3:
    from functools import reduce
    basestring = (str, bytes)
else:
    from itertools import izip as zip


def silence_numpy_errors():
    """Silence numpy warnings about zero division. Normal usage of Space()
    will trigger these warnings."""
    np.seterr(divide='ignore', invalid='ignore')


def sum_onto(a, axis):
    """Numpy convenience. Project all dimensions of an array onto an axis,
    i.e. apply sum() to all axes except the one given."""
    return a.sum(tuple(i for i in range(a.ndim) if i != axis))


class Axis(object):
    """Represents a single dimension finite discrete grid centered at 0.

    Important attributes:
    min     lower bound
    max     upper bound
    res     step size / resolution
    label   human-readable identifier

    min, max and res are floats, but internally only integer operations are
    used. In particular    min = imin * res, max = imax * res
    """

    def __init__(self, min, max, res, label=None):
        self.res = float(res)
        if isinstance(min, (int, np.integer)):
            self.imin = min
        else:
            self.imin = int(np.floor(min / self.res))
        if isinstance(max, (int, np.integer)):
            self.imax = max
        else:
            self.imax = int(np.ceil(max / self.res))
        self.label = label

    @property
    def max(self):
        return self.imax * self.res

    @property
    def min(self):
        return self.imin * self.res

    def __len__(self):
        return self.imax - self.imin + 1

    def __iter__(self):
        return iter(self[index] for index in range(len(self)))

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is not None:
                raise IndexError('stride not supported')
            if key.start is None:
                start = 0
            elif key.start < 0:
                raise IndexError('key out of range')
            elif isinstance(key.start, int):
                start = key.start
            else:
                raise IndexError('key start must be integer')
            if key.stop is None:
                stop = len(self)
            elif key.stop > len(self):
                raise IndexError('key out of range')
            elif isinstance(key.stop, int):
                stop = key.stop
            else:
                raise IndexError('slice stop must be integer')
            return self.__class__(self.imin + start, self.imin + stop - 1, self.res, self.label)
        elif isinstance(key, int):
            if key >= len(self):  # to support iteration
                raise IndexError('key out of range')
            return (self.imin + key) * self.res
        else:
            raise IndexError('unknown key {0!r} {1}'.format(key, type(key)))

    def get_index(self, value):
        """ round is always rounding to nearest even integer, as is
        np.around. maybe change to int(value/res + 0.5)

        returns a single index for a number
        returns slice of indices for input slice
        returns array of indices for input array (same shape)"""
        if isinstance(value, numbers.Number):
            intvalue = int(round(value / self.res))
            if self.imin <= intvalue <= self.imax:
                return intvalue - self.imin
            raise ValueError('cannot get index: value {0} not in range [{1}, {2}]'.format(value, self.min, self.max))
        elif isinstance(value, slice):
            if value.step is not None:
                raise IndexError('stride not supported')
            if value.start is None:
                start = None
            else:
                start = self.get_index(value.start)
            if value.stop is None:
                stop = None
            else:
                stop = self.get_index(value.stop)
            if start is not None and stop is not None and start > stop:
                start, stop = stop, start
            return slice(start, stop)
        else:
            intvalue = np.around(value / self.res).astype(int)
            if ((self.imin <= intvalue) & (intvalue <= self.imax)).all():
                return intvalue - self.imin
            raise ValueError('cannot get indices, values from [{0}, {1}], axes range [{2}, {3}]'.format(value.min(), value.max(), self.min, self.max))

    def __or__(self, other):
        """implement union operation used as '|' ('bitwise or')"""
        if not isinstance(other, Axis):
            return NotImplemented
        if not self.is_compatible(other):
            raise ValueError('cannot unite axes with different resolution/label')
        return self.__class__(min(self.imin, other.imin), max(self.imax, other.imax), self.res, self.label)

    def __eq__(self, other):
        if not isinstance(other, Axis):
            return NotImplemented
        return self.res == other.res and self.imin == other.imin and self.imax == other.imax and self.label == other.label

    def __hash__(self):
        return hash(self.imin) ^ hash(self.imax) ^ hash(self.res) ^ hash(self.label)

    def is_compatible(self, other):
        if not isinstance(other, Axis):
            return False
        return self.res == other.res and self.label == other.label

    def __contains__(self, other):
        """Checks if a number is in the range of the axis or of another
        axis is in the axis. Use as:
            if other in self: do something"""
        if isinstance(other, numbers.Number):
            return self.min <= other <= self.max
        if isinstance(other, Axis):
            return self.is_compatible(other) and self.imin <= other.imin and self.imax >= other.imax
        return NotImplemented

    def rebin(self, factor):
        # for integers the following relations hold:
        # a // b == floor(a / b), -(-a // b) == ceil(a / b)
        new = self.__class__(self.imin // factor, -(-self.imax // factor), factor*self.res, self.label)
        return self.imin % factor, -self.imax % factor, new

    def __repr__(self):
        return '{0.__class__.__name__} {0.label} (min={0.min}, max={0.max}, res={0.res}, count={1})'.format(self, len(self))

    def restrict(self, value):  # Useful for plotting
        if isinstance(value, numbers.Number):
            if value < self.min:
                return self.min
            if value > self.max:
                return self.max
            return value
        if isinstance(value, slice):
            if value.step is not None:
                raise IndexError('stride not supported')
            if value.start is None:
                start = None
            else:
                start = self.restrict(value.start)
            if value.stop is None or value.stop == self.max:
                stop = None
            else:
                stop = self.restrict(value.stop)
            if start is not None and stop is not None and start > stop:
                start, stop = stop, start
            return slice(start, stop)
        return NotImplemented


class Axes(object):
    """Luxurious tuple of Axis objects."""

    def __init__(self, axes):
        self.axes = tuple(axes)
        if len(self.axes) > 1 and any(axis.label is None for axis in self.axes):
            raise ValueError('axis label required for multidimensional space')

    def __iter__(self):
        return iter(self.axes)

    @property
    def dimension(self):
        return len(self.axes)

    @property
    def npoints(self):
        return np.array([len(ax) for ax in self.axes]).prod()

    @property
    def memory_size(self):
        # assuming double precision floats for photons,
        # 32 bit integers for contributions and
        # double precision floats for the variances
        return (8+4+8) * self.npoints

    @classmethod
    def fromfile(cls, filename):
        with util.open_h5py(filename, 'r') as fp:
            try:
                if 'axes' in fp and 'axes_labels' in fp:
                    # oldest style, float min/max
                    return cls(tuple(Axis(min, max, res, lbl) for ((min, max, res), lbl) in zip(fp['axes'], fp['axes_labels'])))
                if 'axes' in fp:  # new
                    try:
                        axes = tuple(Axis(int(imin), int(imax), res, lbl) for ((index, fmin, fmax, res, imin, imax), lbl) in zip(fp['axes'].values(), fp['axes'].keys()))
                        return cls(tuple(axes[int(values[0])] for values in fp['axes'].values()))  # reorder the axes to the way in which they were saved
                    except ValueError:
                        return cls(tuple(Axis(int(imin), int(imax), res, lbl) for ((imin, imax, res), lbl) in zip(fp['axes'].values(), fp['axes'].keys())))
                else:
                    # older style, integer min/max
                    return cls(tuple(Axis(imin, imax, res, lbl) for ((imin, imax), res, lbl) in zip(fp['axes_range'], fp['axes_res'], fp['axes_labels'])))
            except (KeyError, TypeError) as e:
                raise errors.HDF5FileError('unable to load axes definition from HDF5 file {0}, is it a valid BINoculars file? (original error: {1!r})'.format(filename, e))

    def tofile(self, filename):
        with util.open_h5py(filename, 'w') as fp:
            axes = fp.create_group('axes')
            for index, ax in enumerate(self.axes):
                axes.create_dataset(ax.label, data=[index, ax.min, ax.max, ax.res, ax.imin, ax.imax])

    def toarray(self):
        return np.vstack([np.hstack([str(ax.imin), str(ax.imax), str(ax.res), ax.label]) for ax in self.axes])

    @classmethod
    def fromarray(cls, arr):
        return cls(tuple(Axis(int(imin), int(imax), float(res), lbl) for (imin, imax, res, lbl) in arr))

    def index(self, obj):
        if isinstance(obj, Axis):
            return self.axes.index(obj)
        if isinstance(obj, int) and 0 <= obj < len(self.axes):
            return obj
        if isinstance(obj, basestring):
            label = obj.lower()
            matches = tuple(i for i, axis in enumerate(self.axes) if axis.label.lower() == label)
            if len(matches) == 0:
                raise ValueError('no matching axis found')
            elif len(matches) == 1:
                return matches[0]
            else:
                raise ValueError('ambiguous axis label {0}'.format(label))
        else:
            raise ValueError('invalid axis identifier {0!r}'.format(obj))

    def __contains__(self, obj):
        if isinstance(obj, Axis):
            return obj in self.axes
        if isinstance(obj, int):
            return 0 <= obj < len(self.axes)
        if isinstance(obj, basestring):
            label = obj.lower()
            return any(axis.label.lower() == label for axis in self.axes)
        raise ValueError('invalid axis identifier {0!r}'.format(obj))

    def __len__(self):
        return len(self.axes)

    def __getitem__(self, key):
        return self.axes[key]

    def __eq__(self, other):
        if not isinstance(other, Axes):
            return NotImplemented
        return self.axes == other.axes

    def __ne__(self, other):
        if not isinstance(other, Axes):
            return NotImplemented
        return self.axes != other.axes

    def __repr__(self):
        return '{0.__class__.__name__} ({0.dimension} dimensions, {0.npoints} points, {1}) {{\n    {2}\n}}'.format(self, util.format_bytes(self.memory_size), '\n    '.join(repr(ax) for ax in self.axes))

    def restricted_key(self, key):
        if len(key) == 0:
            return None
        if len(key) == len(self.axes):
            return tuple(ax.restrict(s) for s, ax in zip(key, self.axes))
        raise IndexError('dimension mismatch')


class EmptySpace(object):
    """Convenience object for sum() and friends. Treated as zero for addition.
    Does not share a base class with Space for simplicity."""
    def __init__(self, config=None, metadata=None):
        self.config = config
        self.metadata = metadata

    def __add__(self, other):
        if not isinstance(other, Space) and not isinstance(other, EmptySpace):
            return NotImplemented
        return other

    def __radd__(self, other):
        if not isinstance(other, Space) and not isinstance(other, EmptySpace):
            return NotImplemented
        return other

    def __iadd__(self, other):
        if not isinstance(other, Space) and not isinstance(other, EmptySpace):
            return NotImplemented
        return other

    def tofile(self, filename):
        """Store EmptySpace in HDF5 file."""
        with util.atomic_write(filename) as tmpname:
            with util.open_h5py(tmpname, 'w') as fp:
                fp.attrs['type'] = 'Empty'

    def __repr__(self):
        return '{0.__class__.__name__}'.format(self)


class Space(object):
    """Main data-storing object in BINoculars.
    Data is represented on an n-dimensional rectangular grid. Per grid point,
    the number of photons (~ intensity), the number of original data points
    (pixels) contribution and the estimated variances (squared standard
    deviation) from a Poisson distribution is stored.

    Important attributes:
        axes             Axes instances describing range and stepsizes of each of the dimensions
        photons          n-dimensional numpy float array, total intensity per grid point
        contributions    n-dimensional numpy integer array, number of original datapoints (pixels) per grid point
        variances        n-dimensional numpy float array, total variance (squared standard deviation) per grid point
        dimension        n"""

    def __init__(self, axes, config=None, metadata=None):
        if not isinstance(axes, Axes):
            self.axes = Axes(axes)
        else:
            self.axes = axes

        self.config = config
        self.metadata = metadata

        self.photons = np.zeros([len(ax) for ax in self.axes], order='C')
        self.contributions = np.zeros_like(self.photons)
        self.variances = np.zeros_like(self.photons)

    @property
    def dimension(self):
        return self.axes.dimension

    @property
    def npoints(self):
        return self.photons.size

    @property
    def memory_size(self):
        """Returns approximate memory consumption of this Space.
        Only considers size of .photons, .contributions and .variances,
        does not take into account the overhead."""
        return self.photons.nbytes + self.contributions.nbytes + self.variances.nbytes

    @property
    def config(self):
        """util.ConfigFile instance describing configuration file used to create this Space instance"""
        return self._config

    @config.setter
    def config(self, conf):
        if isinstance(conf, util.ConfigFile):
            self._config = conf
        elif not conf:
            self._config = util.ConfigFile()
        else:
            raise TypeError("'{0!r}' is not a util.ConfigFile".format(conf))

    @property
    def metadata(self):
        """util.MetaData instance describing metadata used to create
        this Space instance"""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        if isinstance(metadata, util.MetaData):
            self._metadata = metadata
        elif not metadata:
            self._metadata = util.MetaData()
        else:
            raise TypeError("'{0!r}' is not a util.MetaData".format(metadata))

    def copy(self):
        """Returns a copy of self. Numpy data is not shared,
        but the Axes object is."""
        new = self.__class__(self.axes, self.config, self.metadata)
        new.photons[:] = self.photons
        new.contributions[:] = self.contributions
        new.variances[:] = self.variances
        return new

    def __repr__(self):
        return '{0.__class__.__name__} ({0.dimension} dimensions, {0.npoints} points, {1}) {{\n    {2}\n}}'.format(self, util.format_bytes(self.memory_size), '\n    '.join(repr(ax) for ax in self.axes))

    def __getitem__(self, key):
        """Slicing only! space[-0.2:0.2, 0.9:1.1] does exactly what the syntax
        implies. Ellipsis operator '...' is not supported."""

        newkey = self.get_key(key)      # float slice to index slice(integer)
        slicedaxes = tuple(ax[k] for k, ax in zip(newkey, self.axes) if isinstance(ax[k], Axis))
        if not slicedaxes:
            return self.photons[newkey] / self.contributions[newkey]
        newspace = self.__class__(slicedaxes, self.config, self.metadata)
        newspace.photons = self.photons[newkey].copy()
        newspace.contributions = self.contributions[newkey].copy()
        newspace.variances = self.variances[newkey].copy()
        return newspace

    def get_key(self, key):
        """Convert the n-dimensional interval described by key (as used by
        e.g. __getitem__()) from data coordinates to indices. Needed in fitaid
        for visualising interpolated data"""
        if isinstance(key, numbers.Number) or isinstance(key, slice):
            if not len(self.axes) == 1:
                raise IndexError('dimension mismatch')
            else:
                key = [key]
        elif not (isinstance(key, tuple) or isinstance(key, list)) or not len(key) == len(self.axes):
            raise IndexError('dimension mismatch')
        return tuple(ax.get_index(k) for k, ax in zip(key, self.axes))

    def project(self, axis, *more_axes):
        """Reduce dimensionality of Space by projecting onto 'axis'.
        Sum all data (photons, contributions, variances) along this axis.

        axis         the label of the axis or the index
        *more_axis   also project on these axes"""

        index = self.axes.index(axis)
        newaxes = list(self.axes)
        newaxes.pop(index)
        newspace = self.__class__(newaxes, self.config, self.metadata)
        newspace.photons = self.photons.sum(axis=index)
        newspace.contributions = self.contributions.sum(axis=index)
        newspace.variances = self.variances.sum(axis=index)
        if more_axes:
            return newspace.project(more_axes[0], *more_axes[1:])
        return newspace

    def indexedSlice(self, axis, key):
        """ get a single slice along 'axis'
        This reduces the dimensionality of the space by 1

        axis         the label of the axis or the index
        key          the number of the slice to get along the axes 'axis'
        """
        index = self.axes.index(axis)
        newaxes = list(self.axes)
        newaxes.pop(index)
        newspace = self.__class__(newaxes, self.config, self.metadata)
        npindex = [slice(None) for ax in self.axes]
        npindex[index] = key
        newspace.photons = self.photons[tuple(npindex)]
        newspace.contributions = self.contributions[tuple(npindex)]
        newspace.variances = self.variances[tuple(npindex)]
        return newspace

    def slice(self, axis, key):
        """Single-axis slice.

        axis    label or index of axis to slice
        key     something like slice(lower_data_range, upper_data_range)"""
        axindex = self.axes.index(axis)
        newkey = [slice(None) for ax in self.axes]
        newkey[axindex] = key
        return self.__getitem__(tuple(newkey))

    def get_weighted_intensity(self):
        """ Approximates the weighted average of a Poisson distribution with
        a Gaussian distribution and a correction constant A (reasonable values
        are e.g. 1 or 2)"""
        A = 1
        shifted_variances = 1 / (self.variances + A)
        return (self.get_masked_photons() * shifted_variances).sum() / shifted_variances.sum()

    def get_masked_photons(self):
        return np.ma.array(data=self.photons.copy(), mask=(self.contributions == 0))

    def get_masked_contributions(self):
        return np.ma.array(data=self.contributions.copy(), mask=(self.contributions == 0))

    def get_masked_standarddevs(self):
        if hasattr(self, 'standarddevs'):
            return np.ma.array(data=self.standarddevs.copy(), mask=(self.contributions == 0))
        print('{0} has no standdardevs, did you mean to call get_masked_variances()?'.format(self))

    def get_masked_variances(self):
        return np.ma.array(data=self.variances.copy(), mask=(self.contributions == 0))

    def get_norm_intensity(self):
        """Returns normalized photons(intensity) with divide-by-zero's masked out."""
        mask = (self.contributions == 0)
        return np.ma.array(data=self.photons, mask=mask) / np.ma.array(data=self.contributions, mask=mask)

    def get_norm_standarddevs(self):
        """Returns normalized standarddevs with divide-by-zero's masked out."""
        if hasattr(self, 'standarddevs'):
            mask = (self.contributions == 0)
            return np.ma.array(data=self.standarddevs, mask=mask) / np.ma.array(data=self.contributions, mask=mask)
        print('{0} has no standdardevs, did you mean to call get_norm_variances()?'.format(self))

    def get_norm_variances(self):
        """Returns normalized variances with divide-by-zero's masked out.
        If the standard deviations are normalized by x, the variances need to
        be normalized by x**2"""
        mask = (self.contributions == 0)
        return np.ma.array(data=self.variances, mask=mask) / np.ma.array(data=self.contributions, mask=mask)**2

    def get_grid(self):
        """Returns the data coordinates of each grid point, as n-tuple of
        n-dimensinonal arrays. Basically np.mgrid() in data coordinates."""
        igrid = np.mgrid[tuple(slice(0, len(ax)) for ax in self.axes)]
        grid = tuple(np.array((grid + ax.imin) * ax.res) for grid, ax in zip(igrid, self.axes))
        return grid

    def max(self, axis=None):
        """Returns maximum intensity."""
        return self.get_norm_intensity().max(axis=axis)

    def argmax(self):
        """Returns data coordinates of grid point with maximum intensity."""
        array = self.get_norm_intensity()
        return tuple(ax[key] for ax, key in zip(self.axes, np.unravel_index(np.argmax(array), array.shape)))

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            # variances are unchanged when adding a number
            new = self.copy()
            new.photons += other * self.contributions
            return new
        if not isinstance(other, Space):
            return NotImplemented
        if not len(self.axes) == len(other.axes) or not all(a.is_compatible(b) for (a, b) in zip(self.axes, other.axes)):
            raise ValueError('cannot add spaces with different dimensionality or resolution')

        new = self.__class__([a | b for (a, b) in zip(self.axes, other.axes)])
        new += self
        new += other
        return new

    def __iadd__(self, other):
        """ Implementation of 'self += other' """
        if isinstance(other, numbers.Number):
            self.photons += other * self.contributions
            return self
        if not isinstance(other, Space):
            return NotImplemented
        if not len(self.axes) == len(other.axes) or not all(a.is_compatible(b) for (a, b) in zip(self.axes, other.axes)):
            raise ValueError('cannot add spaces with different dimensionality or resolution')

        if not all(b in a for (a, b) in zip(self.axes, other.axes)):
            return self.__add__(other)

        index = tuple(slice(a.get_index(b.min), a.get_index(b.min) + len(b)) for (a, b) in zip(self.axes, other.axes))
        self.photons[index] += other.photons
        self.contributions[index] += other.contributions
        self.variances[index] += other.variances
        self.metadata += other.metadata
        return self

    def __mul__(self, other):
        """Multiplying a space with a factor scales the intensity with this
        factor, leaves contributions unchanged and scales variances
        with the square of this factor (variances > 0)."""
        if isinstance(other, numbers.Number):
            new = self.__class__(self.axes, self.config, self.metadata)
            new.photons = self.photons * other
            new.contributions = self.contributions
            new.variances = self.variances * other**2
            return new
        return NotImplemented

    def trim(self):
        """Reduce total size of Space by trimming zero-contribution
        data points on the boundaries."""
        mask = self.contributions > 0
        lims = [np.flatnonzero(sum_onto(mask, i)) for (i, ax) in enumerate(self.axes)]
        lims = [(min(i), max(i)) for i in lims]
        self.axes = Axes(Axis(axmin + ax.imin, axmax + ax.imin, ax.res, ax.label) for (ax, (axmin, axmax)) in zip(self.axes, lims))
        slices = tuple(slice(axmin, axmax+1) for (axmin, axmax) in lims)
        self.photons = self.photons[slices].copy()
        self.contributions = self.contributions[slices].copy()
        self.variances = self.variances[slices].copy()

    def rebin(self, resolutions):
        """Change bin size.

        resolution    n-tuple of floats, new resolution of each axis"""
        if len(resolutions) != len(self.axes):
            raise ValueError('cannot rebin space with different dimensionality')
        old_resolutions = tuple(ax.res for ax in self.axes)
        if resolutions == old_resolutions:
            return self
        resolutions = list(resolutions)
        for i, axresolution in enumerate(resolutions):
            if axresolution is None:
                resolutions[i] = old_resolutions[i]
        resolutions = tuple(resolutions)

        # gather data and transform
        coords = self.get_grid()
        intensity = self.photons / self.contributions
        weights = self.contributions
        variances = self.variances / self.contributions**2
        labels = tuple(ax.label for ax in self.axes.axes)
        return self.from_image(resolutions, labels, coords, intensity, weights, variances)

    def reorder(self, labels):
        """Change order of axes."""
        if not self.dimension == len(labels):
            raise ValueError('dimension mismatch')
        newindices = list(self.axes.index(label) for label in labels)
        new = self.__class__(tuple(self.axes[index] for index in newindices), self.config, self.metadata)
        new.photons = np.transpose(self.photons, axes=newindices)
        new.contributions = np.transpose(self.contributions, axes=newindices)
        new.variances = np.transpose(self.variances, axes=newindices)
        return new

    def transform_coordinates(self, resolutions, labels, transformation):
        # gather data and transform
        coords = self.get_grid()
        transcoords = transformation(*coords)
        intensity = self.photons / self.contributions
        weights = self.contributions
        variances = self.variances / self.contributions**2

        # get rid of invalid coords
        valid = reduce(np.bitwise_and, itertools.chain((np.isfinite(t) for t in transcoords)), (weights > 0))
        transcoords = tuple(t[valid] for t in transcoords)

        return self.from_image(resolutions, labels, transcoords, intensity[valid], weights[valid], variances[valid])

    def bin_image(self, coordinates, intensity, weights, variances, valids=None):
        """Load image data into Space, do the binning.

        coordinates  n-tuple of data coordinate arrays
        intensity    data intensity array
        weights      weights array, normally this is the contributions array
        variances    variances array
        valids       filter array (0=filter out, 1=keep point) """

        if len(coordinates) != len(self.axes):
            raise ValueError('dimension mismatch between coordinates and axes')

        # bincount needs 1D arrays
        intensity = np.nan_to_num(intensity).flatten()
        weights = weights.flatten()
        variances = np.nan_to_num(variances).flatten()
        if valids is None:
            valids = np.ones_like(intensity)
        else:
            valids = valids.flatten()

        # indices are the indices of the new data points with respect to the
        # axis they are added to. Same index means the points are in the same
        # bin and combined.
        indices = np.array(tuple(ax.get_index(coord) for (ax, coord) in zip(self.axes, coordinates)))
        for i in range(len(self.axes)):
            for j in range(i+1, len(self.axes)):
                indices[i, :] *= len(self.axes[j])
        indices = indices.sum(axis=0).astype(int).flatten()

        photons = np.bincount(indices, weights=intensity*weights*valids)
        contributions = np.bincount(indices, weights=weights*valids)
        variances = np.bincount(indices, weights=variances*weights**2*valids)

        self.photons.ravel()[:photons.size] += photons
        self.contributions.ravel()[:contributions.size] += contributions
        self.variances.ravel()[:variances.size] += variances

    @classmethod
    def from_image(cls, resolutions, labels, coordinates, intensity, weights, variances=None, limits=None):
        """Create Space from image data.

        resolutions   n-tuple of axis resolutions
        labels        n-tuple of axis labels
        coordinates   n-tuple of data coordinate arrays
        intensity     data intensity array
        weights       weights array (contributions count)
        variances     variances array"""
        # filter out invalid points (coordinates outside the limits (if given))
        if limits is not None:
            invalid = np.zeros(intensity.shape).astype(np.bool)
            for coord, sl in zip(coordinates, limits):
                if sl.start is None and sl.stop is not None:
                    invalid += coord > sl.stop
                elif sl.start is not None and sl.stop is None:
                    invalid += coord < sl.start
                elif sl.start is not None and sl.stop is not None:
                    invalid += np.bitwise_or(coord < sl.start, coord > sl.stop)

            if np.all(invalid):
                return EmptySpace()
            coordinates = tuple(coord[~invalid] for coord in coordinates)
            intensity = intensity[~invalid]
            weights = weights[~invalid]
            variances = variances[~invalid]

        if variances is None:
            print('variances not provided')
            variances = np.full_like(intensity, np.nan)

        axes = tuple(Axis(coord.min(), coord.max(), res, label) for res, label, coord in zip(resolutions, labels, coordinates))
        newspace = cls(axes)
        newspace.bin_image(coordinates, intensity, weights, variances)
        return newspace

    def tofile(self, filename):
        """Store Space in HDF5 file."""
        with util.atomic_write(filename) as tmpname:
            with util.open_h5py(tmpname, 'w') as fp:
                fp.attrs['type'] = 'Space'
                self.config.tofile(fp)
                self.axes.tofile(fp)
                self.metadata.tofile(fp)
                fp.create_dataset('counts', self.photons.shape, dtype=self.photons.dtype, compression='gzip').write_direct(self.photons)
                fp.create_dataset('contributions', self.contributions.shape, dtype=self.contributions.dtype, compression='gzip').write_direct(self.contributions)
                fp.create_dataset('variances', self.variances.shape, dtype=self.variances.dtype, compression='gzip').write_direct(self.variances)

    @classmethod
    def fromfile(cls, file, key=None):
        """Load Space from HDF5 file.

        file      filename string or h5py.Group instance
        key       sliced (subset) loading, should be an n-tuple
                  of slice()s in data coordinates"""
        try:
            with util.open_h5py(file, 'r') as fp:
                if 'type' in fp.attrs.keys():
                    if fp.attrs['type'] == 'Empty':
                        return EmptySpace()

                axes = Axes.fromfile(fp)
                config = util.ConfigFile.fromfile(fp)
                metadata = util.MetaData.fromfile(fp)
                if key:
                    if len(axes) != len(key):
                        raise ValueError("dimensionality of 'key' {0} does not match dimensionality of space in HDF5 file {1}".format(key, file))
                    key = tuple(ax.get_index(k) for k, ax in zip(key, axes))
                    for sl in key:
                        if sl.start == sl.stop and sl.start is not None:
                            raise KeyError('key {0} results in empty space'.format(key))
                    axes = tuple(ax[k] for k, ax in zip(key, axes) if isinstance(k, slice))
                else:
                    key = Ellipsis
                space = cls(axes, config, metadata)
                try:
                    fp['counts'].read_direct(space.photons, key)
                    fp['contributions'].read_direct(space.contributions, key)
                except (KeyError, TypeError) as e:
                    raise errors.HDF5FileError('unable to load space from HDF5 file {0}, is it a valid BINoculars file? (original error: {1!r})'.format(file, e))
                try:
                    fp['variances'].read_direct(space.variances, key)
                except (KeyError, TypeError) as e:
                    space.variances = np.zeros_like(space.photons)
                    print('Variances not found in HDF5 file {0}, using 0 as default. You should recreate the space with BINoculars. Original error: {1}'.format(file, e))
        except IOError as e:
            raise errors.HDF5FileError('unable to open {0} as HDF5 file (original error: {1!r})'.format(file, e))
        return space


class Multiverse(object):
    """A collection of spaces with basic support for addition.
       Only to be used when processing data. This makes it possible to
       process multiple limit sets in a combination of scans"""

    def __init__(self, spaces):
        self.spaces = list(spaces)

    @property
    def dimension(self):
        return len(self.spaces)

    def __add__(self, other):
        if not isinstance(other, Multiverse):
            return NotImplemented
        if not self.dimension == other.dimension:
            raise ValueError('cannot add multiverses with different dimensionality')
        return self.__class__(tuple(s + o for s, o in zip(self.spaces, other.spaces)))

    def __iadd__(self, other):
        if not isinstance(other, Multiverse):
            return NotImplemented
        if not self.dimension == other.dimension:
            raise ValueError('cannot add multiverses with different dimensionality')
        for index, o in enumerate(other.spaces):
            self.spaces[index] += o
        return self

    def tofile(self, filename):
        with util.atomic_write(filename) as tmpname:
            with util.open_h5py(tmpname, 'w') as fp:
                fp.attrs['type'] = 'Multiverse'
                for index, sp in enumerate(self.spaces):
                    spacegroup = fp.create_group('space_{0}'.format(index))
                    sp.tofile(spacegroup)

    @classmethod
    def fromfile(cls, file):
        """Load Multiverse from HDF5 file."""
        try:
            with util.open_h5py(file, 'r') as fp:
                if 'type' in fp.attrs:
                    if fp.attrs['type'] == 'Multiverse':
                        return cls(tuple(Space.fromfile(fp[label]) for label in fp))
                    raise TypeError('This is not a multiverse')
                else:
                    raise TypeError('This is not a multiverse')
        except IOError as e:
            raise errors.HDF5FileError("unable to open '{0}' as HDF5 file (original error: {1!r})".format(file, e))

    def __repr__(self):
        return '{0.__class__.__name__}\n{1}'.format(self, self.spaces)


class EmptyVerse(object):
    """Convenience object for sum() and friends. Treated as zero for addition."""

    def __add__(self, other):
        if not isinstance(other, Multiverse):
            return NotImplemented
        return other

    def __radd__(self, other):
        if not isinstance(other, Multiverse):
            return NotImplemented
        return other

    def __iadd__(self, other):
        if not isinstance(other, Multiverse):
            return NotImplemented
        return other


def union_axes(axes):
    axes = tuple(axes)
    if len(axes) == 1:
        return axes[0]
    if not all(isinstance(ax, Axis) for ax in axes):
        raise TypeError('not all objects are Axis instances')
    if len(set(ax.res for ax in axes)) != 1:
        raise ValueError('cannot unite axes with different resolutions')
    if len(set(ax.label for ax in axes)) != 1:
        raise ValueError('cannot unite axes with different labels')
    mi = min(ax.min for ax in axes)
    ma = max(ax.max for ax in axes)
    first = axes[0]
    return first.__class__(mi, ma, first.res, first.label)


def union_unequal_axes(axes):
    axes = tuple(axes)
    if len(axes) == 1:
        return axes[0]
    if not all(isinstance(ax, Axis) for ax in axes):
        raise TypeError('not all objects are Axis instances')
    if len(set(ax.label for ax in axes)) != 1:
        raise ValueError('cannot unite axes with different labels')
    mi = min(ax.min for ax in axes)
    ma = max(ax.max for ax in axes)
    res = min(ax.res for ax in axes)  # making it easier to use the sliderwidget otherwise this has no meaning
    first = axes[0]
    return first.__class__(mi, ma, res, first.label)


def sum(spaces):
    """Calculate sum of iterable of Space instances."""
    spaces = tuple(space for space in spaces if not isinstance(space, EmptySpace))
    if len(spaces) == 0:
        return EmptySpace()
    if len(spaces) == 1:
        return spaces[0]
    if len(set(space.dimension for space in spaces)) != 1:
        raise TypeError('dimension mismatch in spaces')

    first = spaces[0]
    axes = tuple(union_axes(space.axes[i] for space in spaces) for i in range(first.dimension))
    newspace = first.__class__(axes)
    for space in spaces:
        newspace += space
    return newspace


def verse_sum(verses):
    i = iter(M.spaces for M in verses)
    return Multiverse(sum(spaces) for spaces in zip(*i))


def chunked_sum(verses, chunksize=10):
    """Calculate sum of iterable of Multiverse instances. Creates intermediate
    sums to avoid growing a large space at every summation.

    verses     iterable of Multiverse instances
    chunksize  number of Multiverse instances in each intermediate sum"""
    result = EmptyVerse()
    for chunk in util.grouper(verses, chunksize):
        result += verse_sum(M for M in chunk)
    return result


def iterate_over_axis(space, axis, resolution=None):
    ax = space.axes[space.axes.index(axis)]
    if resolution:
        bins = get_bins(ax, resolution)
        for start, stop in zip(bins[:-1], bins[1:]):
            yield space.slice(axis, slice(start, stop))
    else:
        for value in ax:
            yield space.slice(axis, value)


def iterate_over_axis_keys(axes, axis, resolution=None):
    axindex = axes.index(axis)
    ax = axes[axindex]
    k = [slice(None) for _ in axes]
    if resolution:
        bins = get_bins(ax, resolution)
        for start, stop in zip(bins[:-1], bins[1:]):
            k[axindex] = slice(start, stop)
            yield k
    else:
        for value in ax:
            k[axindex] = value
            yield k


def get_bins(ax, resolution):
    """ Divides an axis into bins for a given resolution"""
    if float(resolution) < ax.res:
        raise ValueError('interval {0} is too low, minimum interval is {1}'.format(resolution, ax.res))

    mi, ma, length = ax.min, ax.max, len(ax) + 1
    return np.linspace(mi, ma, length)


def dstack(spaces, dindices, dlabel, dresolution):
    def transform(space, dindex):
        resolutions = list(ax.res for ax in space.axes)
        resolutions.append(dresolution)
        labels = list(ax.label for ax in space.axes)
        labels.append(dlabel)
        exprs = list(ax.label for ax in space.axes)
        exprs.append('ones_like({0}) * {1}'.format(labels[0], dindex))
        transformation = util.transformation_from_expressions(space, exprs)
        return space.transform_coordinates(resolutions, labels, transformation)
    return sum(transform(space, dindex) for space, dindex in zip(spaces, dindices))


def axis_offset(space, label, offset):
    exprs = list(ax.label for ax in space.axes)
    index = space.axes.index(label)
    exprs[index] += '+ {0}'.format(offset)
    transformation = util.transformation_from_expressions(space, exprs)
    return space.transform_coordinates((ax.res for ax in space.axes), (ax.label for ax in space.axes), transformation)


def make_compatible(spaces):
    if not np.alen(np.unique(len(space.axes) for space in spaces)) == 1:
        raise ValueError('cannot make spaces with different dimensionality compatible')
    ax0 = tuple(ax.label for ax in spaces[0].axes)
    resmax = tuple(np.vstack(tuple(ax.res for ax in space.reorder(ax0).axes) for space in spaces).max(axis=0))
    resmin = tuple(np.vstack(tuple(ax.res for ax in space.reorder(ax0).axes) for space in spaces).min(axis=0))
    if resmax != resmin:
        print('Warning: Not all spaces have the same resolution. Resolution will be changed to: {0}'.format(resmax))
    return tuple(space.reorder(ax0).rebin(resmax) for space in spaces)
