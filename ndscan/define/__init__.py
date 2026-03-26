"""Fragment-definition layer.

This package owns the structural experiment-building primitives:

- fragments,
- parameters,
- result channels,
- default analyses and annotations,
- small helper functions tightly coupled to those concepts.
"""

from . import annotations, default_analysis, fragment, parameters, result_channels, utils
from .annotations import *
from .default_analysis import *
from .fragment import *
from .parameters import *
from .result_channels import *
from .utils import *

__all__ = []
__all__.extend(annotations.__all__)
__all__.extend(default_analysis.__all__)
__all__.extend(fragment.__all__)
__all__.extend(parameters.__all__)
__all__.extend(result_channels.__all__)
__all__.extend(utils.__all__)
