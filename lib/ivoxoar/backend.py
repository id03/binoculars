import itertools

from . import space, util, errors, dispatcher


class ProjectionBase(util.ConfigurableObject):
    def parse_config(self, config):
        super(ProjectionBase, self).parse_config(config)
        res = config.pop('resolution')
        labels = self.get_axis_labels()
        if ',' in res:
            self.config.resolution = util.parse_tuple(res, type=float)
            if not len(labels) == len(self.config.resolution):
                raise errors.ConfigError('dimension mismatch between projection axes ({0}) and resolution specification ({1}) in {2}', labels, self.config.resolution, self.__class__.__name__)
        else:
            self.config.resolution = tuple([float(res)] * len(labels))

    def project(self, *args):
        raise NotImplementedError

    def get_axis_labels(self):
        raise NotImplementedError


class Job(object):
    weight = 1. # estimate of job difficulty (arbitrary units)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class InputBase(util.ConfigurableObject):
    """Generate and process Job()s. 

    Note: there is no guarantee that generate_jobs() and process_jobs() will
    be called on the same instance, not even in the same process or on the
    same computer!"""
    def parse_config(self, config):
        super(InputBase, self).parse_config(config)
        self.config.target_weight = int(config.pop('target_weight', 0))

    def generate_jobs(self, command):
        """Receives command from user, yields Job() instances"""
        raise NotImplementedError

    def process_jobs(self, job):
        """Receives a Job() instance, yields (intensity, args_to_be_sent_to_a_Projection_instance)

        Job()s could have been pickle'd and distributed over a cluster"""
        raise NotImplementedError


def get_dispatcher(config, main, default=None):
    return _get_backend(config, 'dispatcher', dispatcher.DispatcherBase, default=default, args=[main])

def get_input(config, default=None):
    return _get_backend(config, 'input', InputBase, default=default)

def get_projection(config, default=None):
    return _get_backend(config, 'projection', ProjectionBase, default=default)

def _get_backend(config, section, basecls, default=None, args=[], kwargs={}):
    if isinstance(config, util.Config):
        return config.class_(config, *args, **kwargs)

    type = config.pop('type', default)
    if type is None:
        raise errors.ConfigError("required option 'type' not given in section '{0}'".format(section))

    if ':' in type:
        try:
            modname, clsname = type.split(':')
        except ValueError:
            raise errors.ConfigError("invalid type '{0}' in section '{1}'".format(type, section))
        try:
            backend = __import__('backends.{0}'.format(modname), globals(), locals(), [], 1)
        except ImportError as e:
            raise errors.ConfigError("unable to import module backends.{0}: {1}".format(modname, e))
        module = getattr(backend, modname)
    elif section == 'dispatcher':
        module = dispatcher
        clsname = type
    else:
        raise errors.ConfigError("invalid type '{0}' in section '{1}'".format(type, section))

    names = dict((name.lower(), name) for name in dir(module))
    if clsname in names:
        cls = getattr(module, names[clsname])
        
        if issubclass(cls, basecls):
            return cls(config, *args, **kwargs)
        else:
            raise errors.ConfigError("type '{0}' not compatible in section '{1}': expected class derived from '{2}', got '{3}'".format(type, section, basecls.__name__, cls.__name__))
    else:
        raise errors.ConfigError("invalid type '{0}' in section '{1}'".format(type, section))
