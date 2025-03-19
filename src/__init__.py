# Top-level package initialization
__version__ = "1.0.0"
__all__ = ['data', 'models', 'visualization']

# Import subpackages without circular dependencies
import src.data
import src.models
import src.visualization