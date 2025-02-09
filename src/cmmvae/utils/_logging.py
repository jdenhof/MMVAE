import os
import logging


def log_method_decorator(logger: logging.Logger):
    off = os.environ.get("CMMVAE_LOGGER_FINE_GRAIN_OFF", False)
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not off:
                logger.debug(f">{func.__name__}: {args} {kwargs}")
            result = func(*args, **kwargs)
            if not off:
                logger.debug(f"<{func.__name__}: {result}")
            return result
        return wrapper
    return decorator
