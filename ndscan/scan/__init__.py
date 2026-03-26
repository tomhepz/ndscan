"""Scan semantics used by prepared-scan runtime execution.

This package owns scan-time concepts that are shared between code-defined requests,
dashboard-compiled requests, and runtime execution:

- logical scan variables and parameter mappings,
- point-selection policies,
- optional optimisation backends for ask/tell policies.
"""

from . import mapping, point_policy
from .mapping import *
from .point_policy import *

__all__ = []
__all__.extend(mapping.__all__)
__all__.extend(point_policy.__all__)

try:
    from . import optimisation
    from .optimisation import *

    __all__.extend(optimisation.__all__)
except ModuleNotFoundError:
    optimisation = None
