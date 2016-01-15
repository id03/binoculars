import os
import sys
import argparse

from . import space, backend, util, errors


def parse_args(args):
    parser = argparse.ArgumentParser(prog='binoculars process')
    parser.add_argument('-c', metavar='SECTION:OPTION=VALUE', action='append', type=parse_commandline_config_option, default=[], help='additional configuration option in the form section:option=value')
    parser.add_argument('configfile', help='configuration file')
    parser.add_argument('command', nargs='*', default=[])
    return parser.parse_args(args)


def parse_commandline_config_option(s):
    try:
        key, value = s.split('=', 1)
        section, option = key.split(':')
    except ValueError:
        raise argparse.ArgumentTypeError("configuration specification '{0}' not in the form section:option=value".format(s))
    return section, option, value


def multiprocessing_main(xxx_todo_changeme):  # note the double parenthesis for map() convenience
    (config, command) = xxx_todo_changeme
    Main.from_object(config, command)
    return config.dispatcher.destination.retrieve()


class Main(object):
    def __init__(self, config, command):
        if isinstance(config, util.ConfigSectionGroup):
            self.config = config.configfile.copy()
        elif isinstance(config, util.ConfigFile):
            self.config = config.copy()
        else:
            raise ValueError('Configfile is the wrong type')

        # distribute the configfile to space and to the metadata instance
        spaceconf = self.config.copy()

        #input from either the configfile or the configsectiongroup is valid
        self.dispatcher = backend.get_dispatcher(config.dispatcher, self, default='local')
        self.projection = backend.get_projection(config.projection)
        self.input = backend.get_input(config.input)

        self.dispatcher.config.destination.set_final_options(self.input.get_destination_options(command))
        if 'limits' in self.config.projection:
            self.dispatcher.config.destination.set_limits(self.config.projection['limits'])
        if command:
            self.dispatcher.config.destination.set_config(spaceconf)
        self.run(command)

    @classmethod
    def from_args(cls, args):
        args = parse_args(args)
        if not os.path.exists(args.configfile):
            # wait up to 10 seconds if it is a zpi, it might take a while for the file to appear accross the network
            if not args.configfile.endswith('.zpi') or not util.wait_for_file(args.configfile, 10):
                raise errors.FileError("configuration file '{0}' does not exist".format(args.configfile))
        configobj = False
        with open(args.configfile, 'rb') as fp:
            if fp.read(2) == '\x1f\x8b':  # gzip marker
                fp.seek(0)
                configobj = util.zpi_load(fp)
        if not configobj:
            # reopen args.configfile as text
            configobj = util.ConfigFile.fromtxtfile(args.configfile, command=args.command, overrides=args.c)
        return cls(configobj, args.command)

    @classmethod
    def from_object(cls, config, command):
        config.command = command
        return cls(config, command)

    def run(self, command):
        if self.dispatcher.has_specific_task():
            self.dispatcher.run_specific_task(command)
        else:
            jobs = self.input.generate_jobs(command)
            tokens = self.dispatcher.process_jobs(jobs)
            self.result = self.dispatcher.sum(tokens)
            if self.result is True:
                pass
            elif isinstance(self.result, space.EmptySpace):
                sys.stderr.write('error: output is an empty dataset\n')
            else:
                self.dispatcher.config.destination.store(self.result)

    def process_job(self, job):
        def generator():
            res = self.projection.config.resolution
            labels = self.projection.get_axis_labels()
            for intensity, weights, params in self.input.process_job(job):
                coords = self.projection.project(*params)
                if self.projection.config.limits == None:
                    yield space.Multiverse((space.Space.from_image(res, labels, coords, intensity, weights=weights), ))
                else:
                    yield space.Multiverse(space.Space.from_image(res, labels, coords, intensity, weights=weights, limits=limits) for limits in self.projection.config.limits)
        jobverse = space.chunked_sum(generator(), chunksize=25)
        for sp in jobverse.spaces:
            if isinstance(sp, space.Space):
                sp.metadata.add_dataset(self.input.metadata)
        return jobverse

    def clone_config(self):
        config = util.ConfigSectionGroup()
        config.configfile = self.config
        config.dispatcher = self.dispatcher.config.copy()
        config.projection = self.projection.config.copy()
        config.input = self.input.config.copy()
        return config

    def get_reentrant(self):
        return multiprocessing_main


class Split(Main):  # completely ignores the dispatcher, just yields a space per image
    def __init__(self, config, command):
        self.command = command
        if isinstance(config, util.ConfigSectionGroup):
            self.config = config.configfile.copy()
        elif isinstance(config, util.ConfigFile):
            self.config = config.copy()
        else:
            raise ValueError('Configfile is the wrong type')

        #input from either the configfile or the configsectiongroup is valid
        self.projection = backend.get_projection(config.projection)
        self.input = backend.get_input(config.input)

    def process_job(self, job):
        res = self.projection.config.resolution
        labels = self.projection.get_axis_labels()
        for intensity, weights, params in self.input.process_job(job):
            coords = self.projection.project(*params)
            if self.projection.config.limits == None:
                yield space.Space.from_image(res, labels, coords, intensity, weights=weights)
            else:
                yield space.Multiverse(space.Space.from_image(res, labels, coords, intensity, weights=weights, limits=limits) for limits in self.projection.config.limits)

    def run(self):
        for job in self.input.generate_jobs(self.command):
            for verse in self.process_job(job):
                yield verse
