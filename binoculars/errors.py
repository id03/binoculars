# TODO: present exceptions based on errors.ExceptionBase in a gentle way to the user


class ExceptionBase(Exception):
    pass


class ConfigError(ExceptionBase):
    pass


class FileError(ExceptionBase):
    pass


class HDF5FileError(FileError):
    pass


class SubprocessError(ExceptionBase):
    pass


class BackendError(ExceptionBase):
    pass


class CommunicationError(ExceptionBase):
    pass


def addmessage(args, errormsg):
    if not args:
        arg0 = ''
    else:
        arg0 = args[0]
    arg0 += errormsg
    return (arg0, )
