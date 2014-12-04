import itertools
import numbers
import __builtin__
import numpy
import h5py

from . import util, errors


def silence_numpy_errors():
    numpy.seterr(divide='ignore', invalid='ignore')


def sum_onto(a, axis):
    for i in reversed(range(len(a.shape))):
        if axis != i:
            a = a.sum(axis=i)
    return a


class Axis(object):
    def __init__(self, min, max, res, label=None):
        self.res = float(res)
        if isinstance(min, int):
            self.imin = min
        else:
            self.imin = int(numpy.floor(min / self.res))
        if isinstance(max, int):
            self.imax = max
        else:
            self.imax = int(numpy.ceil(max / self.res))
        self.label = label

    @property
    def max(self):
        return self.imax * self.res

    @property
    def min(self):
        return self.imin * self.res
    
    def __len__(self):
        return self.imax - self.imin + 1

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is not None:
                raise IndexError('stride not supported')
            if key.start is None:
                start = 0
            elif isinstance(key.start, int):
                start = key.start
            else:
                raise IndexError('key start must be integer')
            if key.stop is None:
                stop = len(self)
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
            raise IndexError('unknown key {0!r}'.format(key))

    def get_index(self, value):
        if isinstance(value, numbers.Number):
            if self.min <= value <= self.max:
                return int(round((value - self.min) / self.res))
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
            if ((self.min <= value) & (value <= self.max)).all():
                return numpy.around((value - self.min) / self.res).astype(int)
            raise ValueError('cannot get indices, values from [{0}, {1}], axes range [{2}, {3}]'.format(value.min(), value.max(), self.min, self.max))

    def __or__(self, other): # union operation
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
        if isinstance(other, numbers.Number):
            return self.min <= other <= self.max
        elif isinstance(other, Axis):
            return self.is_compatible(other) and self.imin <= other.imin and self.imax >= other.imax

    def rebound(self, min, max):
        return self.__class__(min, max, self.res, self.label)

    def rebin(self, factor):
        # for integers the following relations hold: a // b == floor(a / b), -(-a // b) == ceil(a / b)
        new = self.__class__(self.imin // factor, -(-self.imax  // factor), factor*self.res, self.label)
        return self.imin % factor, -self.imax % factor, new

    def __repr__(self):
        return '{0.__class__.__name__} {0.label} (min={0.min}, max={0.max}, res={0.res}, count={1})'.format(self, len(self))


    def restrict(self, value):#Usefull for plotting
       if isinstance(value, numbers.Number):
           if value <= self.min:
               return self.min
           elif value >= self.max:
               return self.max
           else:
               return value
       elif isinstance(value, slice):
           if value.step is not None:
                   raise IndexError('stride not supported')
           if value.start is None:
               start = None
           else:
               start = self.restrict(value.start)
           if value.stop is None:
               stop = None
           else:
               stop = self.restrict(value.stop)
           if start is not None and stop is not None and start > stop:
               start, stop = stop, start
           return slice(start, stop)


class Axes(object):
    def __init__(self, axes):
        self.axes = tuple(axes)
        if len(self.axes) > 1 and any(axis.label is None for axis in self.axes):
            raise ValueError('axis label is required for multidimensional space')

    def __iter__(self):
        return iter(self.axes)

    @property
    def dimension(self):
        return len(self.axes)

    @property
    def npoints(self):
        return numpy.array([len(ax) for ax in self.axes]).prod()

    @property
    def memory_size(self):
        # assuming double precision floats for photons, 32 bit integers for contributions
        return (8+4) * self.npoints

    @classmethod
    def fromfile(cls, filename):
        with util.open_h5py(filename, 'r') as fp:
            try:
                if 'axes' in fp:
                    # old style, float min/max
                    return cls(tuple(Axis(min, max, res, lbl) for ((min, max, res), lbl) in zip(fp['axes'], fp['axes_labels'])))
                else:
                    # new style, integer min/max
                    return cls(tuple(Axis(imin, imax, res, lbl) for ((imin, imax), res, lbl) in zip(fp['axes_range'], fp['axes_res'], fp['axes_labels'])))
            except (KeyError, TypeError) as e:
                raise errors.HDF5FileError('unable to load axes definition from HDF5 file {0}, is it a valid BINoculars file? (original error: {1!r})'.format(filename, e))

    def tofile(self, filename):
        with util.open_h5py(filename, 'w') as fp:
            range = fp.create_dataset('axes_range', [len(self.axes), 2], dtype=int)
            res = fp.create_dataset('axes_res', [len(self.axes)], dtype=float)
            labels = fp.create_dataset('axes_labels', [len(self.axes)], dtype=h5py.special_dtype(vlen=str))
            for i, ax in enumerate(self.axes):
                range[i, :] = ax.imin, ax.imax
                res[i] = ax.res
                labels[i] = ax.label

    def index(self, obj):
        if isinstance(obj, Axis):
            return self.axes.index(obj)
        elif isinstance(obj, int) and 0 <= obj < len(self.axes):
            return obj
        elif isinstance(obj, basestring):
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
        elif isinstance(obj, int):
            return 0 <= obj < len(self.axes)
        elif isinstance(obj, basestring):
            label = obj.lower()
            return any(axis.label.lower() == label for axis in self.axes)
        else:
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
        else:
            raise IndexError('dimension mismatch')


class EmptySpace(object):
    def __add__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        return other

    def __radd__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        return other

    def __iadd__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        return other


class Space(object):
    def __init__(self, axes, config=None):
        if not isinstance(axes, Axes):
            self.axes = Axes(axes)
        else:
            self.axes = axes

        self.config = config
        
        self.photons = numpy.zeros([len(ax) for ax in self.axes], order='C')
        self.contributions = numpy.zeros(self.photons.shape, dtype=numpy.uint32, order='C')

    @property
    def dimension(self):
        return self.axes.dimension

    @property
    def npoints(self):
        return self.photons.size

    @property
    def memory_size(self):
        # approximate! does not take into account all the overhead
        return self.photons.nbytes + self.contributions.nbytes

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, conf):
        if isinstance(conf, util.ConfigFile):
            self._config = conf
        elif not conf:
            self._config = util.ConfigFile()
        else:
            raise TypeError("'{0!r}' is not a util.ConfigFile".format(space))

    def copy(self):
        new = self.__class__(self.axes)
        new.photons[:] = self.photons
        new.contributions[:] = self.contributions
        return new

    def get(self):
        return self.photons/self.contributions

    def __repr__(self):
        return '{0.__class__.__name__} ({0.dimension} dimensions, {0.npoints} points, {1}) {{\n    {2}\n}}'.format(self, util.format_bytes(self.memory_size), '\n    '.join(repr(ax) for ax in self.axes))
    
    def __getitem__(self, key):
        if isinstance(key, numbers.Number) or isinstance(key, slice):
            if not len(self.axes) == 1:
                raise IndexError('dimension mismatch')
            else:
                key = [key]
        elif not isinstance(key, tuple) or not len(key) == len(self.axes):
            raise IndexError('dimension mismatch')

        newkey = tuple(ax.get_index(k) for k, ax in zip(key, self.axes))
        newaxes = tuple(ax[k] for k, ax in zip(newkey, self.axes) if isinstance(ax[k], Axis))
        if not newaxes:
            raise ValueError('zero-dimensional spaces are not supported')
        newspace = self.__class__(newaxes)
        newspace.photons = self.photons[newkey].copy()
        newspace.contributions = self.contributions[newkey].copy()
        return newspace

    def get_key(self, key):#needed in the fitaid for visualising the interpolated data
        if isinstance(key, numbers.Number) or isinstance(key, slice):
            if not len(self.axes) == 1:
                raise IndexError('dimension mismatch')
            else:
                key = [key]
        elif not isinstance(key, tuple) or not len(key) == len(self.axes):
            raise IndexError('dimension mismatch')
        return tuple(ax.get_index(k) for k, ax in zip(key, self.axes))

    def get_value(self, key):
        if isinstance(key, numbers.Number):
            if not len(self.axes) == 1:
                raise IndexError('dimension mismatch')
            newkey = self.axes[0].get_index(key)
            return self.photons[newkey] / self.contributions[newkey]
        elif isinstance(key, tuple) or isinstance(key, list):
            newkey = tuple(ax.get_index(k) for k, ax in zip(key, self.axes))
            return self.photons[newkey] / self.contributions[newkey]
        else:
            raise ValueError('invalid key: {0}'.format(key))


    def project(self, axis, *more_axes):
        index = self.axes.index(axis)
        newaxes = list(self.axes)
        newaxes.pop(index)
        newspace = self.__class__(newaxes)
        newspace.photons = self.photons.sum(axis=index)
        newspace.contributions = self.contributions.sum(axis=index)

        if more_axes:
            return newspace.project(more_axes[0], *more_axes[1:])
        else:
            return newspace

    def slice(self, axis, key):
        axindex = self.axes.index(axis)
        newkey = list(slice(None) for ax in self.axes)
        newkey[axindex] = key
        return self.__getitem__(tuple(newkey))

    def get_masked(self):
        return numpy.ma.array(data=self.get(), mask=(self.contributions == 0))
        
    def get_grid(self):
        igrid = numpy.mgrid[tuple(slice(0, len(ax)) for ax in self.axes)]
        grid = tuple(numpy.array((grid + ax.imin) * ax.res) for grid, ax in zip(igrid, self.axes))
        return grid

    def max(self, axis=None):
        return self.get_masked().max(axis=axis)

    def argmax(self):
        array = self.get_masked()
        return tuple(ax[key] for ax, key in zip(self.axes, numpy.unravel_index(numpy.argmax(array), array.shape)))

    def __add__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        if not len(self.axes) == len(other.axes) or not all(a.is_compatible(b) for (a, b) in zip(self.axes, other.axes)):
            raise ValueError('cannot add spaces with different dimensionality or resolution')

        new = self.__class__([a | b for (a, b) in zip(self.axes, other.axes)])
        new += self
        new += other
        return new

    def __iadd__(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        if not len(self.axes) == len(other.axes) or not all(a.is_compatible(b) for (a, b) in zip(self.axes, other.axes)):
            raise ValueError('cannot add spaces with different dimensionality or resolution')

        if not all(other_ax in self_ax for (self_ax, other_ax) in zip(self.axes, other.axes)):
            return self.__add__(other)

        index = tuple(slice(self_ax.get_index(other_ax.min), self_ax.get_index(other_ax.min) + len(other_ax)) for (self_ax, other_ax) in zip(self.axes, other.axes))
        self.photons[index] += other.photons
        self.contributions[index] += other.contributions
        return self

    def __sub__(self, other):
        if isinstance(other, numbers.Number):#to test more advanced background subtraction routines
            new = self.copy()
            new.photons -= other * self.contributions
            return new 
        elif not isinstance(other, Space):
            return NotImplemented
        if self.axes != other.axes or not (self.contributions == other.contributions).all():
            # TODO: we could be a bit more helpful if all axes are compatible
            raise ValueError('cannot subtract spaces that are not identical (axes + contributions)')
        new = self.copy()
        new.photons -= other.photons # don't call __isub__ here because the compatibility check is labourous
        return new

    def __isub__(self, other):
        if isinstance(other, numbers.Number):#to test more advanced background subtraction routines
            self.photons -= other * self.contributions
            return self
        elif not isinstance(other, Space):
            return NotImplemented
        if self.axes != other.axes or not (self.contributions == other.contributions).all():
             raise ValueError('cannot subtract spaces that are not identical (axes + contributions)')
        self.photons -= other.photons
        return self

    def __mul__(self, other):#to test more advanced background subtraction routines
        if type(other) == float or type(other) == int:
            self.photons *= other
        else:
            return NotImplemented
        return self

    def trim(self):
        mask = self.contributions > 0
        lims = (numpy.flatnonzero(sum_onto(mask, i)) for (i, ax) in enumerate(self.axes))
        lims = tuple((i.min(), i.max()) for i in lims)
        self.axes = Axes(ax.rebound(min + ax.imin, max + ax.imin) for (ax, (min, max)) in zip(self.axes, lims))
        slices = tuple(slice(min, max+1) for (min, max) in lims)
        self.photons = self.photons[slices].copy()
        self.contributions = self.contributions[slices].copy()

    def rebin(self, factors):
        if isinstance(factors, int):
            factors = [factors] * len(self.axes)
        elif len(factors) != len(self.axes):
            raise ValueError('dimension mismatch between factors and axes')
        if not all(isinstance(factor, int) for factor in factors) or not all(factor == 1 or factor % 2 == 0 for factor in factors):
            raise ValueError('binning factors must be even integers')

        lefts, rights, newaxes = zip(*[ax.rebin(factor) for ax, factor in zip(self.axes, factors)])
        tempshape = tuple(size + left + right + factor for size, left, right, factor in zip(self.photons.shape, lefts, rights, factors))

        photons = numpy.zeros(tempshape, order='C')
        contributions = numpy.zeros(tempshape, dtype=numpy.uint32, order='C')
        pad = tuple(slice(left, left+size) for left, factor, size in zip(lefts, factors, self.photons.shape))
        photons[pad] = self.photons
        contributions[pad] = self.contributions

        new = self.__class__(newaxes)
        for offsets in itertools.product(*[range(factor) for factor in factors]):
            stride = tuple(slice(offset, offset + len(ax)*factor, factor) for offset, ax, factor in zip(offsets, newaxes, factors))
            new.photons += photons[stride]
            new.contributions += contributions[stride]

        return new

    def rebin2(self, resolutions):
        if not len(resolutions) == len(self.axes):
            raise ValueError('cannot rebin space with different dimensionality compatible')
        labels = tuple(ax.label for ax in self.axes)
        coordinates = tuple(grid.flatten() for grid in self.get_grid())

        contribution_space = self.from_image(resolutions, labels, coordinates, self.contributions.flatten())
        contributions = contribution_space.photons.astype(int)
        del contribution_space

        new = self.from_image(resolutions, labels, coordinates, self.photons.flatten())
        new.contributions = contributions
        return new

    def make_compatible(self, other):
        if not isinstance(other, Space):
            return NotImplemented
        if not len(self.axes) == len(other.axes):
            raise ValueError('cannot make spaces with different dimensionality compatible')

        other = other.reorder(tuple(ax.label for ax in self.axes)) 

        resolutions = tuple(max(a.res, b.res) for (a, b) in zip(self.axes, other.axes))
        keys = tuple(slice(max(a.min, b.min), min(a.max, b.max)) for (a, b) in zip(self.axes, other.axes))

        for key in keys:
            if key.start > key.stop:
               raise ValueError('spaces to be compared have no overlap')

        newself = self.__getitem__(keys).rebin2(resolutions)
        newother =  other.__getitem__(keys).rebin2(resolutions)

        return newself, newother

    def reorder(self, labels):
        if not self.dimension == len(labels):
            raise ValueError('cannot make spaces with different dimensionality compatible')
        newindices = list(self.axes.index(label) for label in labels)
        new = self.__class__(tuple(self.axes[index] for index in newindices))
        new.photons = numpy.transpose(self.photons, axes = newindices)
        new.contributions = numpy.transpose(self.contributions, axes = newindices)
        return new

                         
    def transform_coordinates(self, resolutions, labels, transformation):
        # gather data and transform
        intensity = self.get_masked()
        coords = self.get_grid()
        transcoords = transformation(*coords)

        # get rid of invalids & masked intensities
        valid = ~__builtin__.sum((~numpy.isfinite(t) for t in transcoords), intensity.mask)
        transcoords = tuple(t[valid] for t in transcoords)

        return self.from_image(resolutions, labels, transcoords, intensity[valid])

    def process_image(self, coordinates, intensity):
        # note: coordinates must be tuple of arrays, not a 2D array
        if len(coordinates) != len(self.axes):
            raise ValueError('dimension mismatch between coordinates and axes')

        valid = numpy.isfinite(intensity)
        intensity = intensity[valid]
        if not intensity.size:
            return
        coordinates = tuple(coord[valid] for coord in coordinates)

        indices = numpy.array(tuple(ax.get_index(coord) for (ax, coord) in zip(self.axes, coordinates)))
        for i in range(0, len(self.axes)):
            for j in range(i+1, len(self.axes)):
                indices[i,:] *= len(self.axes[j])
        indices = indices.sum(axis=0).astype(int)
        photons = numpy.bincount(indices, weights=intensity)
        contributions = numpy.bincount(indices)
    
        self.photons.ravel()[:photons.size] += photons
        self.contributions.ravel()[:contributions.size] += contributions

    @classmethod
    def from_image(cls, resolutions, labels, coordinates, intensity):
        axes = tuple(Axis(coord.min(), coord.max(), res, label) for res, label, coord in zip(resolutions, labels, coordinates))
        newspace = cls(axes)
        newspace.process_image(coordinates, intensity)
        return newspace

    def tofile(self, filename):
        with util.atomic_write(filename) as tmpname:
            with util.open_h5py(tmpname, 'w') as fp:
                self.config.tofile(fp)
                self.axes.tofile(fp)
                fp.create_dataset('counts', self.photons.shape, dtype=self.photons.dtype, compression='gzip').write_direct(self.photons)
                fp.create_dataset('contributions', self.contributions.shape, dtype=self.contributions.dtype, compression='gzip').write_direct(self.contributions)

    @classmethod
    def fromfile(cls, file, key=None):
        try:
            with util.open_h5py(file, 'r') as fp:
                axes = Axes.fromfile(fp)
                config = util.ConfigFile.fromfile(fp)
                if key:
                    if len(axes) != len(key):
                        raise ValueError("dimensionality of 'key' does not match dimensionality of Space in HDF5 file {0}".format(file))
                    key = tuple(ax.get_index(k) for k, ax in zip(key, axes))
                    axes = tuple(ax[k] for k, ax in zip(key, axes) if isinstance(k, slice))
                else:
                    key = Ellipsis
                space = cls(axes, config)
                try:
                    fp['counts'].read_direct(space.photons, key)
                    fp['contributions'].read_direct(space.contributions, key)
                except (KeyError, TypeError) as e:
                    raise errors.HDF5FileError('unable to load Space from HDF5 file {0}, is it a valid BINoculars file? (original error: {1!r})'.format(file, e))
        except IOError as e:
            raise errors.HDF5FileError("unable to open '{0}' as HDF5 file (original error: {1!r})".format(file, e))
        return space


def union_axes(axes):
    axes = tuple(axes)
    if len(axes) == 1:
        return axes[0]
    if not all(isinstance(ax, Axis) for ax in axes):
        raise TypeError('not all objects are Axis instances')
    if len(set(ax.res for ax in axes)) != 1 or len(set(ax.label for ax in axes)) != 1:
        raise ValueError('cannot unite axes with different resolution/label')
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
        raise ValueError('cannot unite axes with different label')
    mi = min(ax.min for ax in axes)
    ma = max(ax.max for ax in axes)
    res = min(ax.res for ax in axes) #making it easier to use the sliderwidget otherwise this hase no meaning
    first = axes[0]
    return first.__class__(mi, ma, res, first.label)

def sum(spaces):
    spaces = tuple(spaces)
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

# hybrid sum() / __iadd__()
def chunked_sum(spaces, chunksize=10):
    result = EmptySpace()
    for chunk in util.grouper(spaces, chunksize):
        result += sum(space for space in chunk)
    return result
