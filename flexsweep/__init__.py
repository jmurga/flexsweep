import os

import warnings

# Suppress polars warnings. Force 1 thread polars shouldn't cause deadlocks

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="joblib.externals.loky.backend.fork_exec"
)

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

# Set Polars environment variables before importing it
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import threadpoolctl

threadpoolctl.threadpool_limits(1)

# Proceed with the rest of the imports
import numpy as np
from joblib import Parallel, delayed
import polars as pl

from .simulate_discoal import Simulator, DISCOAL
from .fv import summary_statistics
from .data import Data
from .cnn import CNN

try:
    from . import _version

    __version__ = _version.version
except ImportError:
    __version__ = "0.2"

# assert (
#     multiprocessing.get_start_method() == "spawn"
# ), "Multiprocessing start method is not 'spawn'"
