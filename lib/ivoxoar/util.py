import sys
import numpy

def wait_for_files(filelist):
    filelist = filelist[:] # make copy
    i = 0
    while filelist:
        i = i % len(filelist)
        if os.path.exists(filelist[i]):
            yield filelist.pop(i)
        else:
            time.sleep(5)
            i == 1


def wait_for_file(filename):
    return bool(list(wait_for_files([filename])))


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


def space_to_edf(space, filename):
    from PyMca import EdfFile

    header = {}
    for a in space.axes:
        header[str(a.label)] = '{0} {1} {2}'.format(a.min, a.max, a.res)
    edf = EdfFile.EdfFile(filename)
    edf.WriteImage(header, space.get_masked().filled(0), DataType="Float")


def space_to_txt(space, filename):
    data = numpy.mgrid[tuple(slice(0, len(ax)) for ax in space.axes)]
    data = [(coord * ax.res + ax.min).flatten() for coord, ax in zip(data, space.axes)]
    data.append(space.get_masked().filled(0).flatten())
    data = numpy.array(data).T

    with open(filename, 'w') as fp:
        fp.write('\t'.join(ax.label for ax in space.axes))
        fp.write('\tintensity\n')
        numpy.savetxt(fp, data, fmt='%.6g', delimiter='\t')


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
