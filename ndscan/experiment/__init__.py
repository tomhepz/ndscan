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

from ..define import (
    annotations as define_annotations,
    default_analysis as define_default_analysis,
    fragment as define_fragment,
    parameters as define_parameters,
    result_channels as define_result_channels,
)
from ..legacy import (
    entry_point as legacy_entry_point,
    scan_generator as legacy_scan_generator,
    scan_runner as legacy_scan_runner,
    subscan as legacy_subscan,
)
from ..scan import mapping as scan_mapping
from ..scan import point_policy as scan_point_policy
from ..submission import expression as submission_expression
from ..submission import host_scan_schema as submission_host_scan_schema
from ..runtime import api as runtime_api
from ..runtime import persistence as runtime_persistence
from .default_analysis import *
from .entry_point import *
from .fragment import *
from .parameters import *
from .result_channels import *
from .scan_generator import *
from .scan_runner import *
from .subscan import *
from ..scan.mapping import *
from ..scan.point_policy import *
from ..submission.expression import *
from ..submission.host_scan_schema import *
from ..runtime.api import *
from ..runtime.persistence import *

annotations = define_annotations

__all__ = ["annotations"]  # Export annotations as `annotations.curve_1d()`, etc.
__all__.extend(artiq.experiment.__all__)
__all__.extend(define_default_analysis.__all__)
__all__.extend(legacy_entry_point.__all__)
__all__.extend(define_fragment.__all__)
__all__.extend(define_parameters.__all__)
__all__.extend(scan_point_policy.__all__)
__all__.extend(define_result_channels.__all__)
__all__.extend(scan_mapping.__all__)
__all__.extend(submission_expression.__all__)
__all__.extend(submission_host_scan_schema.__all__)
__all__.extend(legacy_scan_generator.__all__)
__all__.extend(legacy_scan_runner.__all__)
__all__.extend(legacy_subscan.__all__)
__all__.extend(runtime_api.__all__)
__all__.extend(runtime_persistence.__all__)

# The optimiser backends depend on optional third-party libraries. Import them when
# available, but do not make the whole experiment package unavailable otherwise.
try:
    from ..scan import optimisation
    from ..scan.optimisation import *

    __all__.extend(optimisation.__all__)
except ModuleNotFoundError:
    optimisation = None
