import os
import sys
import argparse
import ConfigParser

from ivoxoar import space, backend, util

def parse_args():
    parser = argparse.ArgumentParser(prog='ivoxoar')
    parser.add_argument('-c', metavar='SECTION:OPTION=VALUE', action='append', type=parse_commandline_config_option, default=[], help='additional configuration option in the form section:option=value')
    parser.add_argument('configfile', help='configuration file')
    parser.add_argument('command', nargs='*', default=[])
    return parser.parse_args()

def parse_commandline_config_option(s):
    try:
        key, value = s.split('=', 1)
        section, option = key.split(':')
    except ValueError:
        raise argparse.ArgumentTypeError("configuration specification '{0}' not in the form section:option=value".format(s))
    return section, option, value


def multiprocessing_main((config, command)): # note the double parenthesis for map() convenience
    Main.from_object(config, command)
    if isinstance(config.dispatcher.destination, util.Container):
        return config.dispatcher.destination.get()


class Main(object):
    def __init__(self, config, command):
        self.config = config
        self.dispatcher = backend.get_dispatcher(self.config.dispatcher, self, default='local')
        self.projection = backend.get_projection(self.config.projection)
        self.input = backend.get_input(self.config.input)
        self.run(command)

    @classmethod
    def from_args(cls):
        args = parse_args()
        if not os.path.exists(args.configfile):
            # wait up to 10 seconds if it is a zpi, it might take a while for the file to appear accross the network
            if not args.configfile.endswith('.zpi') or not util.wait_for_file(args.configfile, 10):
                raise errors.FileError("configuration file '{0}' does not exists".format(args.configfile))
        configobj = False
        with open(args.configfile, 'rb') as fp:
            if fp.read(2) == '\x1f\x8b': # gzip marker
                fp.seek(0)
                configobj = util.zpi_load(fp)
        if not configobj:
            # reopen args.configfile as text
            config = ConfigParser.RawConfigParser()
            config.read(args.configfile)

            for section, option, value in args.c:
                config.set(section, option, value)

            configobj = util.Config()
            for section in 'dispatcher', 'projection', 'input':
                setattr(configobj, section, dict((k, v.split('#')[0].strip()) for (k, v) in config.items(section)))

        return cls(configobj, args.command)

    @classmethod
    def from_object(cls, config, command):
        return cls(config, command)
        
    def run(self, command):
        if self.dispatcher.has_specific_task():
            self.dispatcher.run_specific_task(command)
        else:
            jobs = self.input.generate_jobs(command)
            tokens = self.dispatcher.process_jobs(jobs)
            result = self.dispatcher.sum(tokens)
            if result is True:
                pass
            elif isinstance(result, space.EmptySpace):
                sys.stderr.write('error: output is an empty dataset\n')
            else:
                result.tofile(self.dispatcher.config.destination)
            
    def process_job(self, job):
        def generator():
            res = self.projection.config.resolution
            labels = self.projection.get_axis_labels()
            for intensity, params in self.input.process_job(job):
                coords = self.projection.project(*params)
                yield space.Space.from_image(res, labels, coords, intensity)
        return space.chunked_sum(generator(), chunksize=25)

    def clone_config(self):
        config = util.Config()
        config.dispatcher = self.dispatcher.config.copy()
        config.projection = self.projection.config.copy()
        config.input = self.input.config.copy()
        return config

    def get_reentrant(self):
        return multiprocessing_main
