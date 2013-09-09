#!/usr/bin/python

import sys
import os
import argparse
import numpy
import numbers

import BINoculars.space, BINoculars.util

### INFO
def command_info(args):
    parser = argparse.ArgumentParser(prog='binoculars info')
    parser.add_argument('infile', nargs='+', help='input files, must be .zpi')

    args = parser.parse_args(args)
    
    for f in args.infile:
        try:
            space = BINoculars.space.Space.fromfile(f)
        except Exception as e:
            print '{0}: unable to load Space: {1}'.format(f, e)
        else:
            print '{0}: {1!r}'.format(f, space)


### CONVERT
def parse_transform_args(transform):
    for t in transform:
        lhs, expr = t.split('=')
        ax, res = lhs.split('@')
        yield ax.strip(), float(res), expr.strip()


def command_convert(args):
    parser = argparse.ArgumentParser(prog='binoculars convert')
    parser.add_argument('--wait', action='store_true', help='wait for input files to appear')
    parser.add_argument('--rebin', metavar='N,M,...', default=None, help='reduce binsize by factor N in first dimension, M in second, etc')
    BINoculars.util.argparse_common_arguments(parser, 'project', 'slice', 'pslice')
    parser.add_argument('infile', help='input file, must be a .zpi')
    parser.add_argument('outfile', help='output file, can be .zpi or .edf or .txt')
    parser.add_argument('transform', metavar='VAR@RES=EXPR', nargs='*', default=[], help='perform coordinate transformation, rebinning data on new axis named VAR with resolution RES defined by EXPR, example: Q@0.1=sqrt(H**2+K**2+L**2)')

    args = parser.parse_args(args)
    
    if args.wait:
        BINoculars.util.statusnl('waiting for {0} to appear'.format(args.infile))
        BINoculars.util.wait_for_file(args.infile)
        BINoculars.util.statusnl('processing...')

    space = BINoculars.space.Space.fromfile(args.infile)
    ext = os.path.splitext(args.outfile)[-1]

    if args.transform:
        labels, resolutions, exprs = zip(*parse_transform_args(args.transform))
        transformation = BINoculars.util.transformation_from_expressions(space, exprs)
        space = space.transform_coordinates(resolutions, labels, transformation)

    space, info = BINoculars.util.project_and_slice(space, args)
    
    if args.rebin:
        if ',' in args.rebin:
            factors = tuple(int(i) for i in args.rebin.split(','))
        else:
            factors = (int(args.rebin),)
        space = space.rebin(factors)

    if ext == '.edf':
        BINoculars.util.space_to_edf(space, args.outfile)
        print 'saved at {0}'.format(args.outfile)

    elif ext == '.txt':
        BINoculars.util.space_to_txt(space, args.outfile)
        print 'saved at {0}'.format(args.outfile)

    elif ext == '.zpi' or ext == '.hdf5':
        space.tofile(args.outfile)
        print 'saved at {0}'.format(args.outfile)

    else:
        sys.stderr.write('unknown extension {0}, unable to save!\n'.format(ext))
        sys.exit(1)


### PLOT
def command_plot(args):
    import matplotlib.pyplot as pyplot
    import matplotlib.colors

    import BINoculars.plot

    parser = argparse.ArgumentParser(prog='binoculars plot')
    parser.add_argument('infile', nargs='+')
    BINoculars.util.argparse_common_arguments(parser, 'savepdf', 'savefile', 'clip', 'nolog', 'project', 'slice', 'pslice', 'subtract')
    parser.add_argument('--multi', default=None, choices=('grid', 'stack'))
    parser.add_argument('--fit', default = None)
    parser.add_argument('--guess', default = None)
    args = parser.parse_args(args)

    if args.subtract:
        subtrspace = BINoculars.space.Space.fromfile(args.subtract)
        subtrspace, subtrinfo = BINoculars.util.project_and_slice(subtrspace, args, auto3to2=True)
        args.nolog = True

    guess = []
    if args.guess is not None:
        for n in args.guess.split(','):
            guess.append(float(n.replace('m', '-')))

    # PLOTTING AND SIMPLEFITTING
    pyplot.figure(figsize=(12, 9))
    plotcount = len(args.infile)
    plotcolumns = int(numpy.ceil(numpy.sqrt(plotcount)))
    plotrows = int(numpy.ceil(float(plotcount) / plotcolumns))

    for i, filename in enumerate(args.infile):
        space = BINoculars.space.Space.fromfile(filename)
        space, info = BINoculars.util.project_and_slice(space, args, auto3to2=True)

        fitdata = None
        if args.fit:
            fit = BINoculars.fit.get_class_by_name(args.fit)(space, guess)
            print fit
            if fit.success:
                fitdata = fit.fitdata

        if plotcount > 1:
            if space.dimension == 1 and args.multi is None:
                args.multi = 'stack'
            if space.dimension == 2 and args.multi != 'grid':
                if args.multi is not None:
                    sys.stderr.write('warning: stack display not supported for multi-file-plotting, falling back to grid\n')
                args.multi = 'grid'
            # elif space.dimension == 3:
                # not reached, project_and_slice() guarantees that
            elif space.dimension > 3:
                sys.stderr.write('error: cannot display 4 or higher dimensional data, use --project or --slice to decrease dimensionality\n')
                sys.exit(1)

        if args.subtract:
            space -= subtrspace

        basename = os.path.splitext(os.path.basename(filename))[0]

        if args.multi == 'grid':
            pyplot.subplot(plotrows, plotcolumns, i+1)
        BINoculars.plot.plot(space, pyplot.gcf(), pyplot.gca(), label=basename, log=not args.nolog, clipping=float(args.clip), fit=fitdata)

        if plotcount == 1:
            pyplot.suptitle('{0} {1}'.format(basename, ' '.join(info)))
        elif args.multi == 'grid':
            pyplot.gca().set_title(basename)


    if plotcount > 1:
        pyplot.suptitle('{0} files, {1}'.format(plotcount, ' '.join(info)))
        if args.multi == 'stack':
            pyplot.legend()

    if args.savepdf or args.savefile:
        if args.savefile:
            pyplot.savefig(args.savefile)
            #print 'saved at {0}'.format(args.savefile)
        else:
            filename = '{0}_plot.pdf'.format(os.path.splitext(args.infile[0])[0])
            filename = BINoculars.util.find_unused_filename(filename)
            pyplot.savefig(filename)
            #print 'saved at {0}'.format(filename)
    else:
        pyplot.show()


### FIT
def command_fit(args):
    import matplotlib.pyplot as pyplot
    import matplotlib.colors

    import BINoculars.fit, BINoculars.plot

    parser = argparse.ArgumentParser(prog='binoculars fit')
    parser.add_argument('infile')
    parser.add_argument('axis')
    parser.add_argument('resolution')
    parser.add_argument('func')
    BINoculars.util.argparse_common_arguments(parser, 'savepdf', 'savefile', 'clip', 'nolog', 'project', 'slice', 'pslice', 'subtract')
    args = parser.parse_args(args)

    if args.subtract:
        subtrspace = BINoculars.space.Space.fromfile(args.subtract)
        subtrspace, subtrinfo = project_and_sice(subtrspace, args)
        args.nolog = True

    space = BINoculars.space.Space.fromfile(args.infile)
    space, info = BINoculars.util.project_and_slice(space, args)

    if float(args.resolution) < space.axes[space.get_axindex_by_label(args.axis)].res:
        raise ValueError('interval {0} to low, minimum interval is {1}'.format(args.resolution, space.axes[space.get_axindex_by_label(args.axis)].res))

    axindex = space.get_axindex_by_label(args.axis)
    axlabel = space.axes[axindex].label
    mi, ma = space.axes[axindex].min, space.axes[axindex].max
    bins = numpy.linspace(mi, ma, numpy.ceil(1 / numpy.float(args.resolution) * (ma - mi)) + 1)

    parameters = []
    variance = []
    fitlabel = []

    basename = os.path.splitext(os.path.basename(args.infile))[0]

    if args.savepdf or args.savefile:
        if args.savefile:
            filename = BINoculars.util.filename_enumerator(args.savefile)
        else:
            filename = BINoculars.util.filename_enumerator('{0}_fit.pdf'.format(basename))

    fitclass = BINoculars.fit.get_class_by_name(args.func)
 
    for start, stop in zip(bins[:-1], bins[1:]):
        info = []
        key = slice(start, stop)
        newspace = space.slice(axindex, key)
        left, right = newspace.axes[axindex].min,newspace.axes[axindex].max
        if newspace.dimension == space.dimension:
            newspace = newspace.project(axindex)

        fit = fitclass(newspace)
        print fit
        if fit.success:
            fitlabel.append(numpy.mean([start, stop]))
            parameters.append(fit.results)
            variance.append(fit.variances)
            fit = fit.fitdata
        else:
            fit = None

        if len(newspace.get_masked().compressed()):
            if newspace.dimension == 1:
                pyplot.figure(figsize=(12, 9))
                pyplot.subplot(111)
                BINoculars.plot.plot(newspace, pyplot.gcf(), pyplot.gca(), label=basename, log=not args.nolog, clipping=float(args.clip), fit = fit)
            elif newspace.dimension == 2:
                pyplot.figure(figsize=(12, 9))
                pyplot.subplot(121)
                BINoculars.plot.plot(newspace, pyplot.gcf(), pyplot.gca(), label=basename, log=not args.nolog, clipping=float(args.clip), fit = None)
                pyplot.subplot(122)
                BINoculars.plot.plot(newspace, pyplot.gcf(), pyplot.gca(), label=basename, log=not args.nolog, clipping=float(args.clip), fit = fit)

            info.append('sliced in {0} from {1} to {2}'.format(axlabel, left, right))
            pyplot.suptitle('{0}'.format(' '.join(info)))

            pyplot.savefig(filename.next())
            pyplot.close()
  
    parameters = numpy.vstack(n for n in parameters).T
    variance = numpy.vstack(n for n in variance).T

    pyplot.figure(figsize=(9, 4 * parameters.shape[0] + 2))

    for i in range(parameters.shape[0]):
        pyplot.subplot(parameters.shape[0], 1, i)
        pyplot.plot(fitlabel, parameters[i, :])
        if paramnames[i] in ['I']:
            pyplot.semilogy()
        pyplot.xlabel(paramnames[i])
        
    pyplot.suptitle('fit summary of {0}'.format(args.infile))     
    if args.savepdf or args.savefile:
        if args.savefile:
            root, ext = os.path.split(args.savefile) 
            pyplot.savefig('{0}_summary{1}'.format(root, ext))
            print 'saved at {0}_summary{1}'.format(root, ext)
            filename = '{0}_summary{1}'.format(root, '.txt')
        else:
            pyplot.savefig('{0}_summary.pdf'.format(os.path.splitext(args.infile)[0]))
            print 'saved at {0}_summary.pdf'.format(os.path.splitext(args.infile)[0])
            filename = '{0}_summary.txt'.format(os.path.splitext(args.infile)[0])
          

        file = open(filename, 'w')
        file.write('L\t')
        file.write('\t'.join(paramnames))
        file.write('\n')
        for n in range(parameters.shape[1]):
            file.write('{0}\t'.format(fitlabel[n]))
            file.write('\t'.join(numpy.array(parameters[:, n], dtype = numpy.str)))
            file.write('\n')
        file.close()


### PROCESS
def command_process(args):
    import BINoculars.main

    BINoculars.util.register_python_executable(__file__)
    BINoculars.main.Main.from_args(args)


### SUBCOMMAND ARGUMENT HANDLING
def usage(msg=''):
    print """usage: binoculars COMMAND ...
{1}
available commands:

 convert    mathematical operations & file format conversions
 info       basic information on Space in .zpi file
 fit        crystal truncation rod fitting
 plot       1D & 2D plotting (parts of) Space and basic fitting
 process    data crunching / binning

run binoculars COMMAND --help more info on that command
""".format(sys.argv[0], msg)
    sys.exit(1)


if __name__ == '__main__':
    BINoculars.space.silence_numpy_errors()

    subcommands = {'info': command_info, 'convert': command_convert, 'plot': command_plot, 'fit': command_fit, 'process': command_process}
    
    if len(sys.argv) < 2:
        usage()
    subcommand = sys.argv[1]
    if subcommand in ('-h', '--help'):
        usage()
    if subcommand not in subcommands:
        usage("binoculars error: unknown command '{0}'\n".format(subcommand))

    subcommands[sys.argv[1]](sys.argv[2:])
