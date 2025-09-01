import os

import warnings

import inspect

# Suppress polars warnings. Force 1 thread polars shouldn't cause deadlocks
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="joblib.externals.loky.backend.fork_exec"
)

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

# Set environment variables before importing it
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

import threadpoolctl

threadpoolctl.threadpool_limits(1)

# Proceed with the rest of the imports
import numpy as np
from joblib import Parallel, delayed
import polars as pl
import importlib

from typing import TYPE_CHECKING

# Eager modules (safe to import)
from .simulate_discoal import Simulator, DISCOAL, DECODE_MAP, DEMES_EXAMPLES
from .fv import summary_statistics
from .data import Data
from . import balancing

# Version
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    __version__ = "2.0"


# Lazy access to cnn module
# Not importing import .cnn, but expose attributes via __getattr__.
# Avoid loading tensorflow till the fs.CNN is call
class _LazyModule:
    """Proxy for a submodule; loads on first real use."""

    __slots__ = ("_fqname", "_pkg", "_mod")

    def __init__(self, fqname: str, pkg: str):
        self._fqname = fqname
        self._pkg = pkg
        self._mod = None  # real module once loaded

    def _load(self):
        if self._mod is None:
            self._mod = importlib.import_module(self._fqname, self._pkg)
        return self._mod

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __dir__(self):
        # Don’t trigger import during tab; show minimal names
        return [] if self._mod is None else dir(self._mod)

    def __repr__(self):
        suffix = "unloaded" if self._mod is None else "loaded"
        return f"<lazy module {self._fqname!r} ({suffix})>"


class _LazyAttr:
    """Proxy for an attribute in a (lazy) module; loads on first real use."""

    __slots__ = ("_mod_proxy", "_attr")

    def __init__(self, mod_proxy: _LazyModule, attr: str):
        self._mod_proxy = mod_proxy
        self._attr = attr

    def _target(self):
        return getattr(
            self._mod_proxy._load(), self._attr
        )  # Allow calling like fs.CNN(...)

    def __call__(self, *a, **kw):
        return self._target()(*a, **kw)  # Support attribute access like fs.CNN.__name__

    def __getattr__(self, name):
        return getattr(self._target(), name)

    def __repr__(self):
        return f"<lazy attr {self._mod_proxy._fqname}.{self._attr} (unloaded)>"


BUILDING_DOCS = (
    os.environ.get("READTHEDOCS") == "True"
    or os.environ.get("FLEXSWEEP_BUILD_DOCS") == "1"
)

if BUILDING_DOCS:
    from .cnn import CNN, rank_probabilities

    cnn = importlib.import_module(".cnn", __name__)
else:
    _cnn_module_proxy = _LazyModule(".cnn", __name__)
    cnn = _cnn_module_proxy
    CNN = _LazyAttr(_cnn_module_proxy, "CNN")
    rank_probabilities = _LazyAttr(_cnn_module_proxy, "rank_probabilities")


# What the package exports (also helps tab completion)
__all__ = [
    # eager
    "balancing",
    "Simulator",
    "DISCOAL",
    "DEMES_EXAMPLES",
    "summary_statistics",
    "Data",
    "np",
    "pl",
    "Parallel",
    "delayed",
    "os",
    "warnings",
    "importlib",
    "threadpoolctl",
    "multiprocessing",
    "cnn",
    "CNN",
    "rank_probabilities",
    "__version__",
]


def __dir__():
    # Ensure proxies appear in fs.<TAB> without triggering imports
    return sorted(set(list(globals().keys()) + ["cnn", "CNN", "rank_probabilities"]))


if TYPE_CHECKING:
    from .cnn import CNN as _CNNType
    from .cnn import rank_probabilities as _rank_probabilities

# from .simulate_discoal import Simulator, DISCOAL, DEMES_EXAMPLES
# from .fv import summary_statistics
# from .data import Data
# from .cnn import CNN, rank_probabilities

# try:
#     from . import _version

#     __version__ = _version.version
# except ImportError:
#     __version__ = "2.0"
