"""clef-flash local server. Importing the package prepares per-platform env vars before torch loads."""

from .backend import prepare_environment as _prepare_environment
from .config import VERSION as __version__

_prepare_environment()

__all__ = ["__version__"]
