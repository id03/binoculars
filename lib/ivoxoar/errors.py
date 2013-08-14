# TODO: present exceptions based on errors.ExceptionBase in a gentle way to the user

class ExceptionBase(Exception):
    pass

class ConfigError(ExceptionBase):
    pass

class FileError(ExceptionBase):
    pass

class SubprocessError(ExceptionBase):
    pass
