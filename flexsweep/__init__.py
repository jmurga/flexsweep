import importlib
import os
import warnings
from typing import TYPE_CHECKING

<<<<<<< HEAD
import inspect

=======
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
# Suppress polars warnings. Force 1 thread polars shouldn't cause deadlocks
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="joblib.externals.loky.backend.fork_exec"
)

<<<<<<< HEAD
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

# Set environment variables before importing it
=======
# Force libraries to single thread
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["POLARS_MAX_THREADS"] = "1"
<<<<<<< HEAD
os.environ["POLARS_MAX_THREADS"] = "1"
=======
# os.environ["JOBLIB_TEMP_FOLDER"] = "/labstorage/jmurgamoreno/"
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

_CONFIGURED = False


<<<<<<< HEAD
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

=======
def _configure_runtime():
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set; keep current behavior.
        pass

    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(1)
    except Exception:
        # If threadpoolctl isn't available, just continue.
        pass


>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
# Version
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    __version__ = "2.0"


# Lazy access to cnn module
<<<<<<< HEAD
# Not importing import .cnn, but expose attributes via __getattr__.
=======
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
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
<<<<<<< HEAD
        return getattr(
            self._mod_proxy._load(), self._attr
        )  # Allow calling like fs.CNN(...)

    def __call__(self, *a, **kw):
        return self._target()(*a, **kw)  # Support attribute access like fs.CNN.__name__
=======
        return getattr(self._mod_proxy._load(), self._attr)

    def __call__(self, *a, **kw):
        return self._target()(*a, **kw)
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

    def __getattr__(self, name):
        return getattr(self._target(), name)

    def __repr__(self):
        return f"<lazy attr {self._mod_proxy._fqname}.{self._attr} (unloaded)>"


<<<<<<< HEAD
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
=======
_cnn_module_proxy = _LazyModule(".cnn", __name__)
cnn = _cnn_module_proxy
CNN = _LazyAttr(_cnn_module_proxy, "CNN")


_LAZY_ATTRS = {
    # Heavy numeric libs
    "np": ("numpy", None),
    "pl": ("polars", None),
    "Parallel": ("joblib", "Parallel"),
    "delayed": ("joblib", "delayed"),
    # Project modules
    "fv": (".fv", None),
    "fv_v2": (".fv_v2", None),
    "scan": (".scan", None),
    "polarize": (".polarize", None),
    "simulate_discoal": (".simulate_discoal", None),
    "utils": (".utils", None),
    "Data": (".data", "Data"),
    "summary_statistics": (".fv", "summary_statistics"),
    "Simulator": (".simulate_discoal", "Simulator"),
    "DISCOAL": (".simulate_discoal", "DISCOAL"),
    "DECODE_MAP": (".simulate_discoal", "DECODE_MAP"),
    "DEMES_EXAMPLES": (".simulate_discoal", "DEMES_EXAMPLES"),
    "rank_probabilities": (".utils", "rank_probabilities"),
    "plot_sfs": (".utils", "plot_sfs"),
    "plot_diversity": (".utils", "plot_diversity"),
    # Rust CLI wrappers
    "run_sort_maf": (".polarize", "run_sort_maf"),
    "run_polarize": (".polarize", "run_polarize"),
    "build_rust_polarization": (".polarize", "build_rust_polarization"),
    # Convenience accessors
    "threadpoolctl": ("threadpoolctl", None),
    "multiprocessing": ("multiprocessing", None),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        _configure_runtime()
        module_name, attr_name = _LAZY_ATTRS[name]
        if module_name.startswith("."):
            module = importlib.import_module(module_name, __name__)
        else:
            module = importlib.import_module(module_name)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)


# What the package exports (also helps tab completion)
__all__ = [
<<<<<<< HEAD
    # eager
    "balancing",
    "Simulator",
    "DISCOAL",
=======
    "utils",
    "Simulator",
    "DISCOAL",
    "DECODE_MAP",
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
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
<<<<<<< HEAD
=======
    "plot_sfs",
    "plot_diversity",
    "run_sort_maf",
    "run_polarize",
    "build_rust_polarization",
    "fv",
    "fv_v2",
    "polarize",
    "simulate_discoal",
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
    "__version__",
]


def __dir__():
<<<<<<< HEAD
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
=======
    return sorted(set(list(globals().keys()) + list(__all__) + ["cnn", "CNN"]))


# if TYPE_CHECKING:
#     from .cnn import CNN as _CNNType
#     from .fv import summary_statistics as _summary_statistics


if TYPE_CHECKING:
    from .cnn import CNN as CNN
    from .fv import summary_statistics as summary_statistics
else:
    _cnn_module_proxy = _LazyModule(".cnn", __name__)
    cnn = _cnn_module_proxy
    CNN = _LazyAttr(_cnn_module_proxy, "CNN")
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
