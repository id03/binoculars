class ExceptionBase(Exception):
    pass

class ConfigError(ExceptionBase):
    pass

class FileError(ExceptionBase):
    pass

class SubprocessError(ExceptionBase):
	pass
