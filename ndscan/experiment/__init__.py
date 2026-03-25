"""Experiment-side ``ndscan`` interface

``ndscan.experiment`` contains the code for implementing the ``ndscan`` primitives from
ARTIQ experiments (as opposed to submitting the experiments with certain parameters, or
later analysing and plotting the generated data).

The top-level module provides a single convenient way to import commonly used symbols
from experiment client code, like ``artiq.experiment`` does for upstream ARTIQ::

    # Import commonly used symbols, including all of artiq.experiment:
    from ndscan.experiment import *
"""

# The wildcard imports below aren't actually unused, as we re-export them.
# ruff: noqa: F401

import artiq.experiment
from artiq.experiment import *

from . import (
    annotations,
    default_analysis,
    entry_point,
    fragment,
    parameters,
    point_policy,
    result_channels,
    scan_mapping,
    scan_generator,
    scan_runner,
    subscan,
)
from ..runtime import api as runtime_api
from ..runtime import persistence as runtime_persistence
from .default_analysis import *
from .entry_point import *
from .fragment import *
from .parameters import *
from .point_policy import *
from .result_channels import *
from .scan_mapping import *
from .scan_generator import *
from .scan_runner import *
from .subscan import *
from ..runtime.api import *
from ..runtime.persistence import *

__all__ = ["annotations"]  # Export annotations as `annotations.curve_1d()`, etc.
__all__.extend(artiq.experiment.__all__)
__all__.extend(default_analysis.__all__)
__all__.extend(entry_point.__all__)
__all__.extend(fragment.__all__)
__all__.extend(parameters.__all__)
__all__.extend(point_policy.__all__)
__all__.extend(result_channels.__all__)
__all__.extend(scan_mapping.__all__)
__all__.extend(scan_generator.__all__)
__all__.extend(scan_runner.__all__)
__all__.extend(subscan.__all__)
__all__.extend(runtime_api.__all__)
__all__.extend(runtime_persistence.__all__)

# The optimiser backends depend on optional third-party libraries. Import them when
# available, but do not make the whole experiment package unavailable otherwise.
try:
    from . import optimisation
    from .optimisation import *

    __all__.extend(optimisation.__all__)
except ModuleNotFoundError:
    optimisation = None
