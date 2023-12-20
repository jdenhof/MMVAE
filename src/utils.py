import time
import sys
import os

# START_TIME = time.time()
DEBUG = True

# _history = {}

# _logger = print
# def print(_str: str, _key = None):
#     log = DEBUG if _key is not None else False
    
#     _log, _interval = '', ''
#     if log:
#         total_time = time.time() - START_TIME
#         _interval = total_time - _history[_key] if _key in _history else 0
#         _log = f"Total: {total_time:.3f}, Interval: {_interval:.3f}"
#         _history[_key] = total_time
#     _logger(_str, _log, _interval, flush=True)

def assert_fields_exists(self, *__attr: str):
    for attr in __attr:
        try:
            assert(getattr(self, attr, None) is not None)
        except:
            raise TypeError(f"Attribute {attr} does not exist in {self.__class__.__name__}")

def debug_attribute(self, attr, val) -> None:
    print(f"In {self.__class__.__name__}: setting {attr.replace('_', ' ')} to {val}")

def attribute_initilized_error(self, attr) -> None:
    raise ValueError(f'{attr} attribute should not be set after {self.__class__.__name__} is initialized')

def range_error(self, value, __max: int, __min: int = 0):
    if value is None or value < __min or value > __max:
        raise ValueError(f"{self.__class__.__name__} {value.__class__.__name__} range error ({__min}, {__max})")
    return value

def _isinstance_error(__o: object, __type: type):
    if not isinstance(__o, __type):
        raise TypeError(f"The {__o.__class__.__name__} must be an instance of {__type.__name__}")
    
def isinstance_error(*args):
    """
    Args:
     - __o: object, __type: type
     - __o: object, __name: str, __type: type
    """
    if len(args) == 3:
        _isinstance_error(getattr(args[0], args[1]), args[2])
    elif len(args) == 2:
        _isinstance_error(args[0], args[1])
    else:
        raise NotImplementedError("Unkown number of args isinstance_error")
    
def get_model_instance_from_file(file_path: str):
    """
        Args: file_path - points to file that contains Model to run
        Notes:
        - file must contain path Model that has zero arguments
        
        Example:
        >>> class Model(VAE):
        ...     def __init__(self):
        ...         super().__init__(Encoder(60664, 512, 128), Decoder(128, 512, 60664))
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("user_defined_module", os.path.join(file_path))
    module = module_from_spec(spec)
    sys.modules["user_defined_module"] = module
    spec.loader.exec_module(module)
    model = getattr(module, 'Model')()

    from torch.nn import Module
    try:
        assert(isinstance(model, Module))
    except:
        raise Exception(f"File: {file_path} - No class Model of torch.nn.Module found")
    
    return model
    