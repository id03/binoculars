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



