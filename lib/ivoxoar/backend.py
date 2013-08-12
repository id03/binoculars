import itertools

from . import space, util, errors


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


def get_input(config):
    return _get_backend(config, 'input', InputBase)

def get_projection(config):
    return _get_backend(config, 'projection', ProjectionBase)

def _get_backend(config, section, basecls, default=None):
    if isinstance(config, util.Config):
        return config.class_(config)

    type = config.pop('type', default)
    if type is None:
        raise errors.ConfigError("required option 'type' not given in section '{0}'".format(section))

    # TODO: handle errors
    if ':' in type:
        module, clsname = type.split(':')
        backend = __import__('backends.{0}'.format(module), globals(), locals(), [], 1)
        mod = getattr(backend, module)
        names = dict((name.lower(), name) for name in dir(mod))
        if clsname in names:
            cls = getattr(mod, names[clsname])
            
            if issubclass(cls, basecls):
                return cls(config)

    raise errors.ConfigError("invalid type '{0}' in section '{1}'".format(type, section))
