# Top-level package initialization
__version__ = "1.0.0"
__all__ = ['data', 'models', 'visualization']

# Import subpackages without circular dependencies
from .data import *
from .models import *
from .visualization import *