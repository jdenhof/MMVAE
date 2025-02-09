from ._utils import *
from . import h5File

def log_method_decorator(logger: logging.Logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f">{func.__name__}: {args} {kwargs}")
            result = func(*args, **kwargs)
            logger.debug(f"<{func.__name__}: {result}")
            return result
        return wrapper
    return decorator