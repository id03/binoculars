import os
import sys
import gzip
import itertools
import random
import cPickle as pickle
import inspect
import time
import copy
import numpy


### ARGUMENT HANDLING

def argparse_common_arguments(parser, *args):
    for arg in args:
        # OPERATIONS
        if arg == 'project':
            parser.add_argument('-p', '--project', metavar='AXIS', action='append', default=[], help='project space on AXIS')
        elif arg == 'slice':
            parser.add_argument('--slice', nargs=2, metavar=('AXIS', 'START:STOP'), action='append', default=[], help="slice AXIS from START to STOP (replace minus signs by 'm')")
        elif arg == 'subtract':
            parser.add_argument('--subtract', metavar='SPACE', help='subtract SPACE from input file')

        # PRESENTATION
        elif arg == 'nolog':
            parser.add_argument('--nolog', action='store_true', help='do not use logarithmic axis')
        elif arg == 'clip':
            parser.add_argument('-c', '--clip', metavar='FRACTION', default=0.00, help='clip color scale to remove FRACTION datapoints')

        # OUTPUT
        elif arg == 'savepdf':
            parser.add_argument('-s', '--savepdf', action='store_true', help='save output as pdf, automatic file naming')
        elif arg == 'savefile':
            parser.add_argument('--savefile', metavar='FILENAME', help='save output as FILENAME, autodetect filetype')

        # ERROR!
        else:
            raise ValueError("unsupported argument '{0}'".format(arg))

def project_and_slice(space, args, auto3to2=False):
    info = []

    # SLICING
    for sl in args.slice:
        ax, key = sl
        axindex = space.get_axindex_by_label(ax)
        axlabel = space.axes[axindex].label
        if ':' in key:
            start, stop = key.split(':')
            if start:
                start = float(start.replace('m', '-'))
            else:
                start = space.axes[axindex].min
            if stop:
                stop = float(stop.replace('m', '-'))
            else:
                stop = space.axes[axindex].max
            key = slice(start, stop)

            info.append('sliced in {0} from {1} to {2}'.format(axlabel, start, stop))
        else:
            key = float(key.replace('m', '-'))
            info.append('sliced in {0} at {1}'.format(axlabel, key))
        olddim = space.dimension
        space = space.slice(axindex, key)
        if space.dimension == olddim:
            space = space.project(axindex)

    # PROJECTION
    for proj in args.project:
        projectaxis = space.get_axindex_by_label(proj)
        info.append('projected on {0}'.format(space.axes[projectaxis].label))
        space = space.project(projectaxis)

    if auto3to2 and space.dimension == 3: # automatic projection on smallest axis
        projectaxis = numpy.argmin(space.photons.shape)
        info.append('projected on {0}'.format(space.axes[projectaxis].label))
        space = space.project(projectaxis)

    return space, info


### STATUS LINES

_status_line_length = 0
def status(line, eol=False):
    """Prints a status line to sys.stdout, overwriting the previous one.
    Set eol to True to append a newline to the end of the line"""

    global _status_line_length
    sys.stdout.write('\r{0}\r{1}'.format(' '*_status_line_length, line))
    if eol:
        sys.stdout.write('\n')
        _status_line_length = 0
    else:
        _status_line_length = len(line)

    sys.stdout.flush()

def statusnl(line):
    """Shortcut for status(..., eol=True)"""
    return status(line, eol=True)

def statuseol():
    """Starts a new status line, keeping the previous one intact"""
    global _status_line_length
    _status_line_length = 0
    sys.stdout.write('\n')
    sys.stdout.flush()

def statuscl():
    """Clears the status line, shortcut for status('')"""
    return status('')


### CONFIGURATION MANAGEMENT

def parse_range(r):
    if '-' in r:
        a, b = r.split('-')
        return range(int(a), int(b)+1)
    elif r:
        return [int(r)]
    else:
        return []

def parse_multi_range(s):
    out = []
    ranges = s.split(',')
    for r in ranges:
        out.extend(parse_range(r))
    return numpy.asarray(out)

def parse_tuple(s, length=None, type=str):
    t = tuple(type(i) for i in s.split(','))
    if length is not None and len(t) != length:
        raise ValueError('invalid tuple length: expected {0} got {0}'.format(length, len(t)))
    return t

def parse_bool(s):
    l = s.lower()
    if l in ('1', 'true', 'yes', 'on'):
        return True
    elif l in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError("invalid input for boolean: '{0}'".format(s))


class Config(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def copy(self):
        return copy.deepcopy(self)


class ConfigurableObject(object):
    def __init__(self, config):
        if isinstance(config, Config):
            self.config = config
        else:
            self.config = Config()
            self.parse_config(config)
            for k in config:
                print 'warning: unrecognized configuration option {0} for {1}'.format(k, self.__class__.__name__)
            self.config.class_ = self.__class__

    def parse_config(self, config):
        # every known option should be pop()'ed from config, converted to a
        # proper type and stored as property in self.config, for example:
        # self.config.foo = int(config.pop('foo', 1))
        pass


### FILES
def best_effort_atomic_rename(src, dest):
    if sys.platform == 'win32' and os.path.exists(dest):
        os.remove(dest)
    os.rename(src, dest)

def filename_enumerator(filename, start=0):
    base,ext = os.path.splitext(filename)
    for count in itertools.count(start):    
        yield '{0}_{2}{1}'.format(base,ext,count)

def find_unused_filename(filename):
    if not os.path.exists(filename):
        return filename
    for f in filename_enumerator(filename, 2):
        if not os.path.exists(f):
            return f

def yield_when_exists(filelist, timeout=None):
    """Wait for files in 'filelist' to appear, for a maximum of 'timeout' seconds,
    yielding them in arbitrary order as soon as they appear.
    If 'filelist' is a set, it will be modified in place, and on timeout it will
    contain the files that have not appeared yet."""
    if not isinstance(filelist, set):
        filelist = set(filelist)
    delay = loop_delayer(5)
    start = time.time()
    while filelist:
        next(delay)
        exists = set(f for f in filelist if os.path.exists(f))
        for e in exists:
            yield e
        filelist -= exists
        if timeout is not None and time.time() - start > timeout:
            break

def wait_for_files(filelist, timeout=None):
    """Wait until the files in 'filelist' have appeared, for a maximum of 'timeout' seconds.
    Returns True on success, False on timeout."""
    filelist = set(filelist)
    for i in yield_when_exists(filelist, timeout):
        pass
    return not filelist

def wait_for_file(file, timeout=None):
    return wait_for_files([file], timeout=timeout)

def space_to_edf(space, filename):
    from PyMca import EdfFile

    header = {}
    for a in space.axes:
        header[str(a.label)] = '{0} {1} {2}'.format(a.min, a.max, a.res)
    edf = EdfFile.EdfFile(filename)
    edf.WriteImage(header, space.get_masked().filled(0), DataType="Float")

def space_to_txt(space, filename):
    data = [coord.flatten() for coord in space.get_grid()]
    data.append(space.get_masked().filled(0).flatten())
    data = numpy.array(data).T

    with open(filename, 'w') as fp:
        fp.write('\t'.join(ax.label for ax in space.axes))
        fp.write('\tintensity\n')
        numpy.savetxt(fp, data, fmt='%.6g', delimiter='\t')


### VARIOUS

def uniqid():
    return '{0:08x}'.format(random.randint(0, 2**32-1))

def grouper(iterable, n):
    while True:
        chunk = list(itertools.islice(iterable, n))
        if not chunk:
            break
        yield chunk

_python_executable = None
def register_python_executable(scriptname):
    global _python_executable
    _python_executable = sys.executable, scriptname

def get_python_executable():
    return _python_executable

def chunk_slicer(count, chunksize):
    """yields slice() objects that split an array of length 'count' into equal sized chunks of at most 'chunksize'"""
    chunkcount = int(numpy.ceil(float(count) / chunksize))
    realchunksize = int(numpy.ceil(float(count) / chunkcount))
    for i in range(chunkcount):
        yield slice(i*realchunksize, min(count, (i+1)*realchunksize))

def cluster_jobs(jobs, target_weight):
    jobs = sorted(jobs, key=lambda job: job.weight)

    # we cannot split jobs here, so just yield away all jobs that are overweight or just right
    while jobs and jobs[-1].weight >= target_weight:
        yield [jobs.pop()]

    while jobs:
        cluster = [jobs.pop()] # take the biggest remaining job
        size = cluster[0].weight
        for i in range(len(jobs)-1, -1, -1): # and exhaustively search for all jobs that can accompany it (biggest first)
            if size + jobs[i].weight <= target_weight:
                size += jobs[i].weight
                cluster.append(jobs.pop(i))
        yield cluster

def loop_delayer(delay):
    """Delay a loop such that it runs at most once every 'delay' seconds. Usage example:
    delay = loop_delayer(5)
    while some_condition:
        next(delay)
        do_other_tasks
    """
    def generator():
        polltime = 0
        while 1:
            diff = time.time() - polltime
            if diff < delay:
                time.sleep(delay - diff)
            polltime = time.time()
            yield
    return generator()


### GZIP PICKLING (zpi)

# handle old zpi's from before ivoxoar's major restructuring
def _pickle_translate(module, name):
    if module == '__main__' and name in ('Space', 'Axis'):
        return 'ivoxoar.space', name
    return module, name

if inspect.isbuiltin(pickle.Unpickler):
    # real cPickle: cannot subclass
    def _find_global(module, name):
        module, name = _pickle_translate(module, name)
        __import__(module)
        return getattr(sys.modules[module], name)

    def pickle_load(fileobj):
        unpickler = pickle.Unpickler(fileobj)
        unpickler.find_global = _find_global
        return unpickler.load()
else:
    # pure python implementation
    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            module, name = _pickle_translate(module, name)
            return pickle.Unpickler.find_class(self, module, name)

    def pickle_load(fileobj):
        unpickler = _Unpickler(fileobj)
        return unpickler.load()

def zpi_save(obj, filename):
    tmpfile = '{0}-{1}.tmp'.format(os.path.splitext(filename)[0], uniqid())
    fp = gzip.open(tmpfile, 'wb')
    try:
        try:
           pickle.dump(obj, fp, pickle.HIGHEST_PROTOCOL)
        finally:
           fp.close()
        best_effort_atomic_rename(tmpfile, filename)
    finally:
        if os.path.exists(tmpfile):
            os.remove(tmpfile)

def zpi_load(filename):
    if hasattr(filename, 'read'):
        fp = gzip.GzipFile(filename.name, fileobj=filename)
    else:
        fp = gzip.open(filename, 'rb')
    try:
        return pickle_load(fp)
    finally:
        fp.close()
