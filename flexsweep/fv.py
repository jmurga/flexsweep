import os

import subprocess

from . import np, Parallel, delayed, pl
from .data import Data

from threadpoolctl import threadpool_limits
from multiprocessing.pool import ThreadPool
from math import comb
from functools import partial, reduce
import sys
import gc


# from jax import config

# config.update("jax_enable_x64", True)
# config.update("jax_platform_name", "cpu")

# import jax.numpy as jnp
# from jax import jit
# from jax.lax import fori_loop

from numba import njit, prange, float64, int64, uint64, types
from numba.typed import List, Dict
from typing import Tuple
from allel import (
    HaplotypeArray,
    ihs,
    nsl,
    garud_h,
    standardize_by_allele_count,
    sequence_diversity,
    mean_pairwise_difference,
    haplotype_diversity,
    moving_haplotype_diversity,
    tajima_d,
    sfs,
    read_vcf,
    GenotypeArray,
    index_windows,
)
from allel.compat import memoryview_safe
from allel.opt.stats import ihh01_scan, ihh_scan
from allel.util import asarray_ndim, check_dim0_aligned, check_integer_dtype
from allel.stats.selection import compute_ihh_gaps
from scipy.interpolate import interp1d

from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict, namedtuple, OrderedDict
from itertools import product, chain

from warnings import filterwarnings, warn
import pickle

import gzip
import re
import glob
import shutil

filterwarnings("ignore", message="invalid INFO header", module="allel.io.vcf_read")

# Define the inner namedtuple structure
summaries = namedtuple("summaries", ["stats", "parameters"])
binned_stats = namedtuple("binned_stats", ["mean", "std"])

from contextlib import contextmanager

################## Utils


# @jit
# def jax_dot(hap):
#     _hap = hap.astype(jnp.float32)
#     return hap @ hap.T


@contextmanager
def omp_num_threads(n_threads: int):
    """
    Temporarily set the OMP_NUM_THREADS environment variable.

    Usage:
        with omp_num_threads(10):
            # within this block: OMP_NUM_THREADS == "10"
            heavy_compute()
        # outside: OMP_NUM_THREADS is back to whatever it was
    """
    key = "OMP_NUM_THREADS"
    old_val = os.environ.get(key, None)
    os.environ[key] = str(n_threads)
    try:
        yield
    finally:
        # restore original state
        if old_val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_val


def filter_gt(hap, rec_map, region=None):
    """
    Convert the 2d numpy haplotype matrix to HaplotypeArray from scikit-allel and change type np.int8. It filters and processes the haplotype matrix based on a recombination map and
    returns key information for further analysis such as allele frequencies and physical positions.

    Parameters
    ----------
    hap : array-like, HaplotypeArray
        The input haplotype data which can be in one of the following forms:
        - A `HaplotypeArray` object.
        - A genotype matrix (as a numpy array or similar).

    rec_map : numpy.ndarray
        A 2D numpy array representing the recombination map, where each row corresponds
        to a genomic variant and contains recombination information. The third column (index 2)
        of the recombination map provides the physical positions of the variants.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - hap_01 (numpy.ndarray): The filtered haplotype matrix, with only biallelic variants.
        - ac (AlleleCounts): An object that stores allele counts for the filtered variants.
        - biallelic_mask (numpy.ndarray): A boolean mask indicating which variants are biallelic.
        - hap_int (numpy.ndarray): The haplotype matrix converted to integer format (int8).
        - rec_map_01 (numpy.ndarray): The recombination map filtered to include only biallelic variants.
        - position_masked (numpy.ndarray): The physical positions of the biallelic variants.
        - sequence_length (int): An arbitrary sequence length set to 1.2 million bases (1.2e6).
        - freqs (numpy.ndarray): The frequencies of the alternate alleles for each biallelic variant.
    """
    try:
        # Avoid unnecessary conversion if hap is already a HaplotypeArray
        if not isinstance(hap, HaplotypeArray):
            hap = HaplotypeArray(
                hap if isinstance(hap, np.ndarray) else hap.genotype_matrix()
            )
    except:
        hap = HaplotypeArray(load(hap).genotype_matrix())

    # positions = rec_map[:, -1]
    # physical_position = rec_map[:, -2]

    # HAP matrix centered to analyse whole chromosome
    hap_01, ac, biallelic_mask = filter_biallelics(hap)
    sequence_length = int(1.2e6)

    rec_map_01 = rec_map[biallelic_mask]

    mask_dup = (~pl.DataFrame(rec_map_01)[:, -1].is_duplicated()).to_numpy()

    hap_01 = hap_01[mask_dup]
    rec_map_01 = rec_map_01[mask_dup]
    position_masked = rec_map_01[:, -1].astype(int)
    genetic_position_masked = rec_map_01[:, -2]

    ac = ac[mask_dup]

    return (
        hap_01,
        rec_map_01,
        ac,
        biallelic_mask,
        position_masked,
        genetic_position_masked,
    )


def filter_biallelics(hap: HaplotypeArray) -> tuple:
    """
    Filter out non-biallelic loci from the haplotype data.

    Args:
        hap (allel.HaplotypeArray): Haplotype data represented as a HaplotypeArray.

    Returns:
        tuple: A tuple containing three elements:
            - hap_biallelic (allel.HaplotypeArray): Filtered biallelic haplotype data.
            - ac_biallelic (numpy.ndarray): Allele counts for the biallelic loci.
            - biallelic_mask (numpy.ndarray): Boolean mask indicating biallelic loci.
    """
    ac = hap.count_alleles()
    biallelic_mask = ac.is_biallelic_01()

    # Use a subset to filter directly, minimizing intermediate memory usage
    hap_biallelic = hap.subset(biallelic_mask)

    ac_biallelic = ac[biallelic_mask]

    return (hap_biallelic.values, ac_biallelic.values, biallelic_mask)


def ms_parser(ms_file, param=None, seq_len=1.2e6):
    """Read a ms file and output the positions and the genotypes.
    Genotypes are a numpy array of 0s and 1s with shape (num_segsites, num_samples).
    """

    assert (
        ms_file.endswith(".out")
        or ms_file.endswith(".out.gz")
        or ms_file.endswith(".ms")
        or ms_file.endswith(".ms.gz")
    )

    open_function = gzip.open if ms_file.endswith(".gz") else open

    with open_function(ms_file, "rt") as file:
        file_content = file.read()

    # Step 2: Split by pattern (e.g., `---`)
    pattern = r"//"
    partitions = re.split(pattern, file_content)

    if len(partitions) == 1:
        warn(f"File {ms_file} is malformed.")
        return None
    else:
        positions = []
        haps = []
        rec_map = []
        for r in partitions[1:]:
            # Read in number of segregating sites and positions
            data = []
            for line in r.splitlines()[1:]:
                if line == "":
                    continue
                # if "discoal" in line or "msout" in line:
                # seq_len = int(line.strip().split()[3])
                if line.startswith("segsites"):
                    num_segsites = int(line.strip().split()[1])
                    if num_segsites == 0:
                        continue
                        #     # Shape of data array for 0 segregating sites should be (0, 1)
                        # return np.array([]), np.array([], ndmin=2, dtype=np.uint8).T
                elif line.startswith("positions"):
                    tmp_pos = np.array([float(x) for x in line.strip().split()[1:]])
                    tmp_pos = np.round(tmp_pos * seq_len).astype(int)

                    # Find duplicates in the array
                    duplicates = np.diff(tmp_pos) == 0

                    # While there are any duplicates, increment them by 1
                    for i in np.where(duplicates)[0]:
                        tmp_pos[i + 1] += 1
                    tmp_pos += 1
                    positions.append(tmp_pos)
                    tmp_map = np.column_stack(
                        [
                            np.repeat(1, tmp_pos.size),
                            np.arange(tmp_pos.size),
                            tmp_pos,
                            tmp_pos,
                        ]
                    )
                    rec_map.append(tmp_map)

                else:
                    # Now read in the data
                    data.append(np.array(list(line), dtype=np.int8))
            try:
                data = np.vstack(data).T
            except:
                data = None
                warn(f"File {ms_file} is malformed.")
                return None

            # data = np.vstack(data).T
            haps.append(data)

    if param is None:
        param = np.zeros(4)

    return (haps[0], rec_map[0], param)


def ms_parser_np(ms_file, param=None, seq_len=1.2e6):
    """Read an ms/msms/hudson file and return (haps, rec_map, param) for the FIRST replicate.
    haps: (num_segsites, num_samples) int8 array of 0/1
    rec_map: (num_segsites, 4) int64 array: [1, idx, pos, pos]
    """
    if not ms_file.endswith((".out", ".out.gz", ".ms", ".ms.gz")):
        warn(f"File {ms_file} has an unexpected extension.")

    open_function = gzip.open if ms_file.endswith(".gz") else open

    in_rep = False
    num_segsites = None
    pos_arr = None
    hap_rows = []  # strings of '0'/'1', one per sample

    def finish_and_return(p_in):
        # Validate minimal structure
        if num_segsites is None or pos_arr is None or not hap_rows:
            warn(f"File {ms_file} is malformed.")
            return None

        # Build hap matrix fast: bytes -> uint8 -> 0/1 -> int8, then transpose
        try:
            n_samples = len(hap_rows)
            n_sites = len(hap_rows[0])
            H = np.empty((n_samples, n_sites), dtype=np.int8)
            for i, s in enumerate(hap_rows):
                H[i, :] = (
                    np.frombuffer(s.encode("ascii"), dtype=np.uint8) - 48
                ).astype(np.int8, copy=False)
            H = H.T  # (num_segsites, num_samples)
        except Exception:
            warn(f"File {ms_file} is malformed.")
            return None

        n = pos_arr.size
        rec_map = np.column_stack(
            (np.repeat(1, n), np.arange(n, dtype=np.int64), pos_arr, pos_arr)
        ).astype(np.int64, copy=False)

        p_out = np.zeros(4) if p_in is None else p_in
        return (H, rec_map, p_out)

    with open_function(ms_file, "rt") as fh:
        for raw in fh:
            line = raw.strip()

            # start of a replicate
            if line.startswith("//"):
                if in_rep:
                    out = finish_and_return(param)
                    if out is not None:
                        return out
                    # else, malformed replicate; continue scanning for next
                in_rep = True
                num_segsites = None
                pos_arr = None
                hap_rows.clear()
                continue

            if not in_rep or not line:
                continue

            if line.startswith("segsites"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        num_segsites = int(parts[1])
                    except ValueError:
                        warn(f"File {ms_file} is malformed.")
                        return None
                else:
                    warn(f"File {ms_file} is malformed.")
                    return None
                continue

            if line.startswith("positions"):
                # line like: "positions: 0.123 0.456 ..."
                try:
                    _, values = line.split(":", 1)
                    tmp_pos = np.fromstring(values, sep=" ", dtype=np.float64)
                except Exception:
                    warn(f"File {ms_file} is malformed.")
                    return None

                # EXACTLY match original: np.round then single-pass bump of consecutive duplicates, then +1
                tmp_pos = np.round(tmp_pos * seq_len).astype(np.int64, copy=False)

                dups = np.diff(tmp_pos) == 0
                if dups.any():
                    idxs = np.nonzero(dups)[0]
                    tmp_pos[idxs + 1] += 1  # bump NEXT duplicate once

                tmp_pos += 1
                pos_arr = tmp_pos
                continue

            # haplotype row ('0'/'1' string)
            c0 = line[0]
            if c0 == "0" or c0 == "1":
                hap_rows.append(line)

        # EOF: if a replicate was started, finalize it
        if in_rep:
            return finish_and_return(param)

    warn(f"File {ms_file} is malformed.")
    return None


def check_malformed(ms, _iter):
    try:
        hap, r, p = ms_parser(ms)
        return -1
    except:
        return _iter


def cleaning_summaries(data, summ_stats, params, model):
    # Cleaning params and simulations to remove malformed simulations
    mask = []
    summ_stats_filtered = []
    malformed_files = []
    for i, j in enumerate(summ_stats):
        if j is None:
            mask.append(i)
            malformed_files.append(
                "File " + data + "/" + model + "_" + str(i) + ".ms.gz is malformed."
            )
        else:
            summ_stats_filtered.append(j)

    if len(mask) != 0:
        params = np.delete(params, mask, axis=0)

    return summ_stats_filtered, params, malformed_files


def genome_reader(hap_data, recombination_map=None, region=None, samples=None, _iter=1):
    filterwarnings("ignore", message="invalid INFO header", module="allel.io.vcf_read")

    raw_data = read_vcf(hap_data, region=region, samples=samples)

    try:
        gt = GenotypeArray(raw_data["calldata/GT"])
    except:
        return {region: None}

    pos = raw_data["variants/POS"]
    np_chrom = np.char.replace(raw_data["variants/CHROM"].astype(str), "chr", "")
    try:
        np_chrom = np_chrom.astype(int)
    except:
        pass
    ac = gt.count_alleles()

    # Filtering monomorphic just in case
    biallelic_filter = ac.is_biallelic_01()

    hap = gt.to_haplotypes()
    hap = hap.values[biallelic_filter]
    pos = pos[biallelic_filter]
    np_chrom = np_chrom[biallelic_filter]

    if hap.shape[0] == 0:
        return {region: None}

    if region is None:
        d_pos = dict(zip(np.arange(pos.size + 1), pos))
    else:
        tmp = list(map(int, region.split(":")[-1].split("-")))
        d_pos = dict(zip(np.arange(tmp[0], tmp[1] + 1), np.arange(int(1.2e6)) + 1))

    if recombination_map is None:
        rec_map = pl.DataFrame(
            {"chrom": np_chrom, "idx": np.arange(pos.size), "pos": pos, "cm": pos}
        ).to_numpy()
    else:
        df_recombination_map = (
            pl.read_csv(
                recombination_map,
                separator="\t",
                comment_prefix="#",
                schema=pl.Schema(
                    [
                        ("chr", pl.String),
                        ("start", pl.Int64),
                        ("end", pl.Int64),
                        ("cm_mb", pl.Float64),
                        ("cm", pl.Float64),
                    ]
                ),
            )
            .filter(pl.col("chr") == "chr" + str(np_chrom[0]))
            .sort("start")
        )
        genetic_distance = get_cm(df_recombination_map, pos)

        rec_map = pl.DataFrame(
            [np_chrom, np.arange(pos.size), pos, genetic_distance]
        ).to_numpy()

        if np.all(rec_map[:, -1] == 0):
            rec_map[:, -1] = rec_map[:, -2]

    # for r in rec_map:
    #     r[-1] = d_pos[r[-1]]
    #     r[-2] = d_pos[r[-2]]

    return (hap, rec_map[:, [0, 1, -1, 2]], np.zeros(4))


def get_cm(df_rec_map, positions):
    # Create the interpolating function
    interp_func = interp1d(
        df_rec_map.select(df_rec_map.columns[1]).to_numpy().flatten(),
        df_rec_map.select(df_rec_map.columns[-1]).to_numpy().flatten(),
        kind="linear",
        fill_value="extrapolate",
    )

    # Interpolate the cM values at the interval positions
    rr1 = interp_func(positions)
    # rr2 = interp_func(positions[:, 1])
    rr1[rr1 < 0] = 0
    # Calculate the recombination rate in cM/Mb
    # rate = (rr2 - rr1) / ((positions[:, 1] - positions[:, 0]) / 1e6)

    return rr1


def get_cm_mb(df_rec_map):
    return df_rec_map.with_columns(
        (
            (pl.col("cm").shift(-1) - pl.col("cm"))
            / (pl.col("pos").shift(-1) - pl.col("pos"))
            * 1e6
        ).alias("cm_mb")
    )


def center_window_cols(df, _iter=1):
    if df.is_empty():
        # Return a dataframe with one row of the specified default values
        return pl.concat(
            [pl.DataFrame({"iter": _iter}), df],
            # [pl.DataFrame({"iter": _iter, "center": center, "window": window}), df],
            how="horizontal",
        )

    df = (
        df.with_columns(
            [
                pl.lit(_iter).alias("iter"),
                # pl.lit(center).alias("center"),
                # pl.lit(window).alias("window"),
            ]
        ).with_columns(pl.col(["iter"]).cast(pl.Int64))
        # .with_columns(pl.col(["iter", "center", "window"]).cast(pl.Int64))
        .select(
            pl.col(["iter", "positions"]),
            pl.all().exclude(["iter", "positions"]),
            # pl.col(["iter", "center", "window", "positions"]),
            # pl.all().exclude(["iter", "center", "window", "positions"]),
        )
    )
    return df


def pivot_feature_vectors(df_fv, vcf=False):
    """
    Categorizes genomic sweep data into different models based on timing and fixation status,
    then pivots the data for analysis.

    Parameters
    ----------
    df_fv : polars.DataFrame
        DataFrame containing sweep data with columns including 't', 'f_t', 'f_i', 's', 'iter',
        'window', and 'center'.

    Returns
    -------
    polars.DataFrame
        Pivoted DataFrame with categorized sweep models.
    """
    # Remove delta_ihh column if present
    # df_fv = df_fv.select(pl.exclude("delta_ihh"))

    # Categorize sweeps based on age and completeness
    df_fv = df_fv.with_columns(
        pl.when((pl.col("t") >= 2000) & (pl.col("f_t") >= 0.9))
        .then(pl.lit("hard_old_complete"))
        .when((pl.col("t") >= 2000) & (pl.col("f_t") < 0.9))
        .then(pl.lit("hard_old_incomplete"))
        .when((pl.col("t") < 2000) & (pl.col("f_t") >= 0.9))
        .then(pl.lit("hard_young_complete"))
        .otherwise(pl.lit("hard_young_incomplete"))
        .alias("model")
    )

    # Further categorize as soft or hard sweep based on initial frequency
    df_fv = df_fv.with_columns(
        pl.when(pl.col("f_i") != df_fv["f_i"].min())
        .then(pl.col("model").str.replace("hard", "soft"))
        .otherwise(pl.col("model"))
        .alias("model")
    )

    # Handle the case where all selection coefficients are zero (neutral model)
    if (df_fv["s"] == 0).all():
        df_fv = df_fv.with_columns(pl.lit("neutral").alias("model"))

    # Determine sorting method based on iter column type
    sort_multi = True if df_fv["iter"].dtype == pl.Utf8 else False

    # Pivot the data
    value_columns = df_fv.columns[
        7:-1
    ]  # Assuming columns 7 to end-1 are the values to pivot

    if vcf:
        # remove nchr
        value_columns = value_columns[:-1]
        fv_center = np.linspace(6e5 - 1e5, 6e5 + 1e5, 21).astype(int)
        rows_per_center = df_fv["window"].unique().len()
        n_rows = df_fv.height
        full_center = np.tile(
            np.repeat(fv_center, rows_per_center),
            n_rows // (len(fv_center) * rows_per_center) + 1,
        )[:n_rows]

        df_fv = df_fv.with_columns(
            (
                pl.col("nchr").cast(pl.String)
                + ":"
                + (pl.col("iter").cast(pl.Int64) - int(6e5)).cast(pl.String)
                + "-"
                + (pl.col("iter").cast(pl.Int64) + int(6e5)).cast(pl.String)
            ).alias("iter"),
            pl.lit(full_center).alias("center"),
        ).select(pl.exclude("nchr"))

    df_fv_w = df_fv.pivot(
        values=value_columns,
        index=["iter", "s", "t", "f_i", "f_t", "model"],
        on=["window", "center"],
    )

    # Clean up column names
    df_fv_w = df_fv_w.rename(
        {
            col: col.replace("{", "").replace("}", "").replace(",", "_")
            for col in df_fv_w.columns
        }
    )

    return df_fv_w


def get_closest_snps(position_array, center, N):
    """
    Given a list of SNP positions and a center position, return the indices of the N closest SNPs.

    Parameters:
        position_array (np.ndarray): 1D array of SNP positions (e.g., [100, 500, 1000, ...]).
        center (int or float): The central genomic coordinate to anchor the window around.
        N (int): Number of SNPs to retrieve (must be ≤ len(position_array)).

    Returns:
        np.ndarray: Indices of the N closest SNPs to the center.
    """
    position_array = np.asarray(position_array)
    assert position_array.ndim == 1, "position_array must be a 1D array"
    assert N <= len(position_array), "N exceeds the number of SNPs in the array"

    distances = np.abs(position_array - center)
    closest_indices = np.argsort(distances)[:N]
    return closest_indices  # sorted left to right on genome


################## Summaries


def _process_vcf(data_dir, nthreads, center, windows, step, recombination_map):
    """
    Handle vcf=True: read all VCFs in data_dir/vcfs, compute and normalize stats,
    write out per‐VCF parquet files, and return concatenated DataFrames.
    """

    # Paths and containers
    fvs_file = {}
    sims = {}
    regions = {}
    df_params = []

    vcf_glob = os.path.join(data_dir, "vcfs", "*vcf.gz")
    for vcf_path in sorted(glob.glob(vcf_glob)):
        basename = os.path.basename(vcf_path)
        key = basename.replace(".vcf", "").replace(".bcf", "").replace(".gz", "")
        key = key.replace(".", "_").lower()

        fs_data = Data(vcf_path, nthreads=nthreads)
        sim_dict = fs_data.read_vcf()
        # returns {"sweep": [...], "region": [...]}

        # build parameter DataFrame
        n = len(sim_dict["sweep"])
        df_params.append(
            pl.DataFrame(
                {
                    "model": np.repeat(key, n),
                    "s": np.zeros(n),
                    "t": np.zeros(n),
                    "saf": np.zeros(n),
                    "eaf": np.zeros(n),
                }
            )
        )

        sims[key] = sim_dict["sweep"]
        regions[key] = sim_dict["region"]
        fvs_file[key] = os.path.join(data_dir, "vcfs", f"fvs_{key}.parquet")

    df_params = pl.concat(df_params)

    results = {}
    tmp_bins = []
    d_centers = {}
    malformed_files = {}

    for k, vcf_file in sims.items():
        with Parallel(n_jobs=nthreads, backend="loky", verbose=1) as parallel:
            print(k)
            params = df_params.filter(pl.col("model") == k)[:, 1:].to_numpy()

            # compute center from region strings "chr: start-end"
            center_coords = [
                tuple(map(int, r.split(":")[-1].split("-"))) for r in regions[k]
            ]
            d_centers[k] = np.array([(a + b) // 2 for a, b in center_coords])

            # Open a joblib process by VCF, so not same pool slower but solve RAM issues
            stats = calculate_stats_vcf(
                vcf_file,
                region=regions[k],
                # center=center,
                # step=step,
                recombination_map=recombination_map,
                nthreads=nthreads,
                parallel_manager=parallel,
            )

            stats, params, malformed = cleaning_summaries(data_dir, stats, params, key)
            malformed_files[k] = malformed

            raw_stats, norm_stats = stats
            tmp_bins.append(norm_stats)

            if not np.all(params[:, 3] == 0):
                params[:, 0] = -np.log(params[:, 0])
            results[k] = summaries(raw_stats, params)

    empirical_bins = binned_stats(*normalize_neutral(tmp_bins))
    df_fv_cnn = {}
    df_fv_cnn_raw = {}

    # for k, stats_values in tqdm(results.items()):
    with Parallel(n_jobs=nthreads, backend="loky", verbose=0) as parallel:
        for k, stats_values in results.items():
            print(k)
            df_w, df_w_raw = normalize_stats(
                stats_values,
                empirical_bins,
                center=d_centers[k],
                windows=windows,
                vcf=True,
                parallel_manager=parallel,
            )
            df_w.write_parquet(fvs_file[k])
            df_w_raw.write_parquet(fvs_file[k].replace(".parquet", "_raw.parquet"))
            df_fv_cnn[k] = df_w
            df_fv_cnn_raw[k] = df_w_raw

    df_train = pl.concat(df_fv_cnn.values(), how="vertical")
    df_train_raw = pl.concat(df_fv_cnn_raw.values(), how="vertical")

    with open(os.path.join(data_dir, "empirical_bins.pickle"), "wb") as f:
        pickle.dump(empirical_bins, f)

    return df_train, df_train_raw


def _process_sims_full(data_dir, nthreads, center, windows, step, recombination_map):
    """
    Handle vcf=False: read sweep/neutral sims under data_dir,
    compute stats in a single Parallel pool, normalize sweep vs neutral,
    write out parquet and pickle, and return concatenated DataFrames.
    """

    for folder in ("neutral", "sweep"):
        path = os.path.join(data_dir, folder)
        if not os.path.isdir(path):
            raise ValueError(f"Missing folder: {path}")
        if not glob.glob(os.path.join(path, "*")):
            raise ValueError(f"No files in folder: {path}")

    fs_data = Data(data_dir)
    sims, df_params = fs_data.read_simulations()

    results = {}
    malformed_files = {}
    d_centers = {}
    binned_data = {}
    hap_matrices = {}
    t_m_l = []
    k_neutral = {"neutral": None, "sweep": None}

    with Parallel(n_jobs=nthreads, backend="loky", verbose=2) as parallel:
        for sim_type, sim_list in sims.items():
            print(sim_type)
            mask = np.random.choice(np.arange(0, 100000), 12500)
            params = df_params.filter(pl.col("model") == sim_type)[:, 1:].to_numpy()
            d_centers[sim_type] = center

            # map each sim to calculate_stats
            _tmp_stats = tuple(
                zip(
                    *parallel(
                        delayed(calculate_stats_simplify_full)(
                            hap_data,
                            i,
                            center=center,
                            windows=windows,
                            step=step,
                            region=None,
                        )
                        for i, hap_data in enumerate(sim_list[:], 1)
                    )
                )
            )
            stats, params, malformed = cleaning_summaries(
                data_dir, _tmp_stats, params, sim_type
            )
            malformed_files[sim_type] = malformed

            if sim_type == "neutral":
                raw_stats, norm_stats = stats
                binned_data["neutral"] = binned_stats(
                    *normalize_neutral_full(norm_stats)
                )
            else:
                raw_stats, norm_stats = stats

            if not np.all(params[:, 3] == 0):
                params[:, 0] = -np.log(params[:, 0])
            results[sim_type] = summaries(raw_stats, params)

        # df_t_m = pl.concat(t_m_l)
        df_fv_cnn = {}
        df_fv_cnn_raw = {}

    with Parallel(n_jobs=nthreads, backend="loky", verbose=2) as parallel:
        for sim_type, stats_values in results.items():
            df_w, df_w_raw = normalize_stats_full(
                stats_values,
                binned_data["neutral"],
                center=d_centers[sim_type],
                windows=windows,
                parallel_manager=parallel,
                nthreads=nthreads,
                vcf=False,
            )
            df_fv_cnn[sim_type] = df_w
            df_fv_cnn_raw[sim_type] = df_w_raw

    with open(os.path.join(data_dir, "neutral_bins.pickle"), "wb") as f:
        pickle.dump(binned_data["neutral"], f)

    df_train = pl.concat(df_fv_cnn.values(), how="vertical")
    # df_train = df_train.join(
    #     df_t_m.select(pl.exclude("s", "t", "f_t", "f_i")), on=["iter", "model"]
    # )
    df_train_raw = pl.concat(df_fv_cnn_raw.values(), how="vertical")

    out_base = os.path.join(data_dir, "fvs.parquet")
    df_train.write_parquet(out_base)
    df_train_raw.write_parquet(out_base.replace(".parquet", "_raw.parquet"))

    # np.savez_compressed(
    #     f"{data_dir}/haplotype_matrices.npz",
    #     neutral=hap_matrices["neutral"],
    #     sweep=hap_matrices["sweep"],
    # )

    # return df_train, df_train_raw, hap_matrices
    return df_train, df_train_raw


def _process_sims(data_dir, nthreads, center, windows, step, recombination_map):
    """
    Handle vcf=False: read sweep/neutral sims under data_dir,
    compute stats in a single Parallel pool, normalize sweep vs neutral,
    write out parquet and pickle, and return concatenated DataFrames.
    """

    for folder in ("neutral", "sweep"):
        path = os.path.join(data_dir, folder)
        if not os.path.isdir(path):
            raise ValueError(f"Missing folder: {path}")
        if not glob.glob(os.path.join(path, "*")):
            raise ValueError(f"No files in folder: {path}")

    fs_data = Data(data_dir)
    sims, df_params = fs_data.read_simulations()

    results = {}
    malformed_files = {}
    d_centers = {}
    binned_data = {}
    hap_matrices = {}
    t_m_l = []
    k_neutral = {"neutral": None, "sweep": None}

    with Parallel(n_jobs=nthreads, backend="loky", verbose=2) as parallel:
        for sim_type, sim_list in sims.items():
            print(sim_type)
            mask = np.random.choice(np.arange(0, 100000), 12500)
            params = df_params.filter(pl.col("model") == sim_type)[:, 1:].to_numpy()
            d_centers[sim_type] = center

            # map each sim to calculate_stats
            _tmp_stats = tuple(
                zip(
                    *parallel(
                        delayed(calculate_stats_simplify)(
                            hap_data,
                            i,
                            center=center,
                            step=step,
                            region=None,
                        )
                        for i, hap_data in enumerate(sim_list[:], 1)
                    )
                )
            )
            # stats = (_tmp_stats[0], _tmp_stats[1])
            # hap_matrices[sim_type] = _tmp_stats[-1]
            # lassi need to estimate K_neutral and t/m through each K_counts by simulation
            # _df_t_m, K_neutral = compute_t_m(
            #     sim_list[mask],
            #     K_neutral=k_neutral["neutral"],
            #     params=params,
            #     parallel_manager=parallel,
            # )
            # t_m_l.append(_df_t_m)
            # df_params.filter(pl.col("model") == sim_type)[:1000, :].join(df_t_m)
            # k_neutral[sim_type] = K_neutral

            stats, params, malformed = cleaning_summaries(
                data_dir, _tmp_stats, params, sim_type
            )
            malformed_files[sim_type] = malformed

            if sim_type == "neutral":
                raw_stats, norm_stats = stats
                binned_data["neutral"] = binned_stats(*normalize_neutral(norm_stats))
            else:
                raw_stats, norm_stats = stats

            if not np.all(params[:, 3] == 0):
                params[:, 0] = -np.log(params[:, 0])
            results[sim_type] = summaries(raw_stats, params)

        # df_t_m = pl.concat(t_m_l)
        df_fv_cnn = {}
        df_fv_cnn_raw = {}
    with Parallel(n_jobs=nthreads, backend="loky", verbose=2) as parallel:
        for sim_type, stats_values in results.items():
            df_w, df_w_raw = normalize_stats(
                stats_values,
                binned_data["neutral"],
                center=d_centers[sim_type],
                windows=windows,
                parallel_manager=parallel,
                nthreads=nthreads,
                vcf=False,
            )
            df_fv_cnn[sim_type] = df_w
            df_fv_cnn_raw[sim_type] = df_w_raw

    with open(os.path.join(data_dir, "neutral_bins.pickle"), "wb") as f:
        pickle.dump(binned_data["neutral"], f)

    df_train = pl.concat(df_fv_cnn.values(), how="vertical")
    # df_train = df_train.join(
    #     df_t_m.select(pl.exclude("s", "t", "f_t", "f_i")), on=["iter", "model"]
    # )
    df_train_raw = pl.concat(df_fv_cnn_raw.values(), how="vertical")

    out_base = os.path.join(data_dir, "fvs.parquet")
    df_train.write_parquet(out_base)
    df_train_raw.write_parquet(out_base.replace(".parquet", "_raw.parquet"))

    # np.savez_compressed(
    #     f"{data_dir}/haplotype_matrices.npz",
    #     neutral=hap_matrices["neutral"],
    #     sweep=hap_matrices["sweep"],
    # )

    # return df_train, df_train_raw, hap_matrices
    return df_train, df_train_raw


def summary_statistics(
    data_dir,
    vcf=False,
    nthreads=1,
    center=[500000, 700000],
    windows=[50000, 100000, 200000, 500000, 1000000],
    step=10000,
    recombination_map=None,
    full=False,
):
    """
    Wrapper: dispatch to VCF or simulation pipeline.
    """
    if vcf:
        return _process_vcf(
            data_dir, nthreads, center, windows, step, recombination_map
        )
    else:
        if full:
            return _process_sims_full(
                data_dir, nthreads, center, windows, step, recombination_map
            )
        else:
            return _process_sims(
                data_dir, nthreads, center, windows, step, recombination_map
            )


################## Stats


def calculate_stats_vcf(
    vcf_file,
    region,
    _iter=1,
    # center=[500000, 700000],
    # step=1e4,
    recombination_map=None,
    parallel_manager=None,
    nthreads=1,
):
    filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in scalar divide",
    )
    np.seterr(divide="ignore", invalid="ignore")

    if isinstance(vcf_file, list) or isinstance(vcf_file, tuple):
        hap, rec_map, p = vcf_file
    elif isinstance(vcf_file, str):
        try:
            hap, rec_map, p = genome_reader(
                vcf_file, recombination_map=recombination_map, region=None
            )
        except:
            return None
    else:
        return None

    # Open and filtering data
    (
        hap_int,
        rec_map_01,
        ac,
        biallelic_mask,
        position_masked,
        genetic_position_masked,
    ) = filter_gt(hap, rec_map, region=None)
    freqs = ac[:, 1] / ac.sum(axis=1)

    # If recombination is not provided, then genetic_pos == pos
    if recombination_map is None:
        genetic_position_masked = None

    # Define variables
    windows = [tuple(map(int, r.split(":")[-1].split("-"))) for r in region]

    nchr = region[0].split(":")[0]
    d_stats = defaultdict(dict)

    # Estimate fs stats
    df_dind_high_low, df_s_ratio, df_hapdaf_s, df_hapdaf_o = run_fs_stats(
        hap_int[:, :], ac[:, :], rec_map_01[:, :]
    )

    df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
    df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
    df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
    df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

    # Estimate iHS and nSL
    df_ihs = ihs_ihh(
        hap_int,
        position_masked,
        map_pos=genetic_position_masked,
        min_ehh=0.05,
        min_maf=0.05,
        include_edges=False,
        use_threads=True,
    )
    df_ihs = center_window_cols(df_ihs, _iter=_iter)

    nsl_v = nsl(hap_int[freqs >= 0.05], use_threads=True)
    df_nsl = pl.DataFrame(
        {
            "positions": position_masked[freqs >= 0.05],
            "daf": freqs[freqs >= 0.05],
            "nsl": nsl_v,
        }
    ).fill_nan(None)

    df_ihs = center_window_cols(df_ihs, _iter=_iter)
    df_nsl = center_window_cols(df_nsl, _iter=_iter)

    # Estimate windowed (h12,haf,isafe) stats
    if parallel_manager is None:
        with Parallel(n_jobs=nthreads, backend="loky", verbose=5) as parallel:
            out_h12_haf, out_isafe = zip(
                *parallel(
                    delayed(run_windowed_stats)(
                        hap_int[
                            (position_masked >= window[0])
                            & (position_masked <= window[1])
                        ],
                        position_masked[
                            (position_masked >= window[0])
                            & (position_masked <= window[1])
                        ],
                        window,
                    )
                    for _iter, (window) in enumerate(windows[:], 1)
                )
            )
    else:
        out_h12_haf, out_isafe = zip(
            *parallel_manager(
                delayed(run_windowed_stats)(
                    hap_int[
                        (position_masked >= window[0]) & (position_masked <= window[1])
                    ],
                    position_masked[
                        (position_masked >= window[0]) & (position_masked <= window[1])
                    ],
                    window,
                )
                for _iter, window in enumerate(windows[:], 1)
            )
        )

    df_window = pl.concat(out_h12_haf, how="vertical")

    # Because of random shuffle in iSAFE it will output different values for same snps
    # in contigous windows, get the max value or the mean
    # df_isafe = pl.concat(out_isafe, how="vertical")
    df_isafe = (
        pl.concat(out_isafe, how="vertical")
        .group_by(["iter", "positions", "daf"], maintain_order=True)
        # .group_by(["iter", "center", "window", "positions", "daf"], maintain_order=True)
        .agg(pl.col("isafe").mean().alias("isafe"))
        .sort("positions")
    )

    # Save estimations in dict
    d_stats["dind_high_low"] = df_dind_high_low
    d_stats["s_ratio"] = df_s_ratio
    d_stats["hapdaf_o"] = df_hapdaf_o
    d_stats["hapdaf_s"] = df_hapdaf_s
    d_stats["ihs"] = df_ihs
    d_stats["nsl"] = df_nsl
    d_stats["isafe"] = df_isafe
    d_stats["h12_haf"] = df_window

    if region is not None:
        for k, df in d_stats.items():
            d_stats[k] = df.with_columns([pl.lit(nchr).cast(pl.Utf8).alias("iter")])

    ####### if neutral:

    df_stats_norm = (
        reduce(
            lambda left, right: left.join(
                right,
                # on=["iter", "center", "window", "positions", "daf"],
                on=["iter", "positions", "daf"],
                how="full",
                coalesce=True,
            ),
            [
                df_nsl.lazy(),
                df_ihs.lazy(),
                df_isafe.lazy(),
                df_dind_high_low.lazy(),
                df_s_ratio.lazy(),
                df_hapdaf_o.lazy(),
                df_hapdaf_s.lazy(),
                df_window.lazy(),
            ],
        )
        .sort("positions")
        .collect()
    )

    # df_snps_norm = reduce(
    #     lambda left, right: left.join(
    #         right,
    #         on=["iter", "positions", "daf"],
    #         # on=["iter", "center", "window", "positions", "daf"],
    #         how="full",
    #         coalesce=True,
    #     ),
    #     [df_nsl, df_ihs, df_isafe],
    #     # ).sort(["iter", "center", "window", "positions"])
    # ).sort(["iter", "positions"])

    # df_stats = reduce(
    #     lambda left, right: left.join(
    #         right,
    #         # on=["iter", "center", "window", "positions", "daf"],
    #         on=["iter", "positions", "daf"],
    #         how="full",
    #         coalesce=True,
    #     ),
    #     [df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s, df_window],
    # ).sort(["iter", "positions"])
    # # ).sort(["iter", "center", "window", "positions"])

    # df_stats_norm = df_snps_norm.join(
    #     df_stats.select(
    #         pl.all().exclude(
    #             ["iter", "delta_ihh", "ihs", "isafe", "nsl"]
    #             # ["iter", "center", "window", "delta_ihh", "ihs", "isafe", "nsl"]
    #         )
    #     ),
    #     on=["positions", "daf"],
    #     how="full",
    #     coalesce=True,
    # ).sort(["positions"])

    # df_stats_norm = (
    #     df_stats_norm.with_columns(
    #         [
    #             pl.lit(_iter).alias("iter"),
    #             # pl.lit(600000).alias("center"),
    #             # pl.lit(1200000).alias("window"),
    #             pl.col("positions").cast(pl.Int64),
    #         ]
    #     ).select(
    #         pl.col(["iter", "positions"]),
    #         pl.all().exclude(["iter", "positions"]),
    #         # pl.col(["iter", "center", "window", "positions"]),
    #         # pl.all().exclude(["iter", "center", "window", "positions"]),
    #     )
    #     # .sort(["iter", "center", "window", "positions"])
    #     .sort(["iter", "positions"])
    # )

    if region is not None:
        df_stats_norm = df_stats_norm.with_columns(
            [pl.lit(nchr).cast(pl.Utf8).alias("iter")]
        )

    return d_stats, df_stats_norm


def calculate_stats_simplify(
    hap_data,
    _iter=1,
    center=[5e5, 7e5],
    windows=[50000, 100000, 200000, 500000, 1000000],
    step=1e4,
    region=None,
):
    filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in scalar divide",
    )
    np.seterr(divide="ignore", invalid="ignore")

    if isinstance(hap_data, list) or isinstance(hap_data, tuple):
        hap, rec_map, p = hap_data
    elif isinstance(hap_data, str):
        try:
            # hap, rec_map, p = ms_parser(hap_data)
            hap, rec_map, p = ms_parser_np(hap_data)
            if hap.shape[0] != rec_map.shape[0]:
                return None, None, None
        except:
            try:
                hap, rec_map, p = genome_reader(hap_data, region)
            except:
                return None, None, None
    else:
        return None, None, None

    # Open and filtering data
    (
        hap_int,
        rec_map_01,
        ac,
        biallelic_mask,
        position_masked,
        genetic_position_masked,
    ) = filter_gt(hap, rec_map, region=region)
    freqs = ac[:, 1] / ac.sum(axis=1)

    # if len(center) == 1:
    #     centers = np.arange(center[0], center[0] + step, step).astype(int)
    # else:
    # centers = np.arange(center[0], center[1] + step, step).astype(int)
    # Estimate fs stats
    df_dind_high_low, df_s_ratio, df_hapdaf_s, df_hapdaf_o = run_fs_stats(
        hap_int, ac, rec_map_01
    )

    d_stats = defaultdict(dict)

    df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
    df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
    df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
    df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

    # iSAFE
    df_isafe = run_isafe(hap_int, position_masked)
    df_isafe = center_window_cols(df_isafe, _iter=_iter)

    # iHS and nSL
    df_ihs = ihs_ihh(
        hap_int,
        position_masked,
        map_pos=genetic_position_masked,
        min_ehh=0.05,
        min_maf=0.05,
        include_edges=False,
    )

    df_ihs = center_window_cols(df_ihs, _iter=_iter)

    nsl_v = nsl(hap_int[freqs >= 0.05], use_threads=False)
    df_nsl = pl.DataFrame(
        {
            "positions": position_masked[freqs >= 0.05],
            "daf": freqs[freqs >= 0.05],
            "nsl": nsl_v,
        }
    ).fill_nan(None)
    df_nsl = center_window_cols(df_nsl, _iter=_iter)

    try:
        # h12_v, h2_h1 = h12_enard(
        #     hap_int,
        #     position_masked,
        #     window_size=int(1e5),
        # )
        h12_v, h2_h1, h2_v = h12_enard_numba(
            hap_int,
            position_masked
            # , window_size=int(1e5)
        )
    except:
        h12_v, h2_h1, h2_v = np.nan, np.nan

    haf_v = haf_top(
        hap_int.astype(np.float64),
        position_masked,
        # window_size=int(1e5),
    )

    daf_w = 1.0
    pos_w = int(6e5)
    if 6e5 in position_masked:
        daf_w = freqs[position_masked == 6e5][0]

    df_window = pl.DataFrame(
        {
            "iter": pl.Series([_iter], dtype=pl.Int64),
            # "center": pl.Series([int(6e5)], dtype=pl.Int64),
            # "window": pl.Series([int(1e6)], dtype=pl.Int64),
            "positions": pl.Series([pos_w], dtype=pl.Int64),
            "daf": pl.Series([daf_w], dtype=pl.Float64),
            "h12": pl.Series([h12_v], dtype=pl.Float64),
            "h2_h1": pl.Series([h2_h1], dtype=pl.Float64),
            "haf": pl.Series([haf_v], dtype=pl.Float64),
        }
    )

    d_stats["dind_high_low"] = df_dind_high_low
    d_stats["s_ratio"] = df_s_ratio
    d_stats["hapdaf_o"] = df_hapdaf_o
    d_stats["hapdaf_s"] = df_hapdaf_s
    d_stats["ihs"] = df_ihs
    d_stats["isafe"] = df_isafe
    d_stats["nsl"] = df_nsl
    d_stats["h12_haf"] = df_window

    if region is not None:
        for k, df in d_stats.items():
            d_stats[k] = df.with_columns(pl.lit(region).alias("iter"))

    ####### if neutral:

    df_stats_norm = (
        reduce(
            lambda left, right: left.join(
                right,
                # on=["iter", "center", "window", "positions", "daf"],
                on=["iter", "positions", "daf"],
                how="full",
                coalesce=True,
            ),
            [
                df_nsl.lazy(),
                df_ihs.lazy(),
                df_isafe.lazy(),
                df_dind_high_low.lazy(),
                df_s_ratio.lazy(),
                df_hapdaf_o.lazy(),
                df_hapdaf_s.lazy(),
                df_window.lazy(),
            ],
            # [df_nsl, df_ihs_n, df_isafe],
        )
        .sort("positions")
        .collect()
    )

    # df_snps_norm = reduce(
    #     lambda left, right: left.join(
    #         right,
    #         # on=["iter", "center", "window", "positions", "daf"],
    #         on=["iter", "positions", "daf"],
    #         how="full",
    #         coalesce=True,
    #     ),
    #     [df_nsl, df_ihs, df_isafe],
    #     # [df_nsl, df_ihs_n, df_isafe],
    # )

    # df_stats = reduce(
    #     lambda left, right: left.join(
    #         right,
    #         on=["iter", "positions", "daf"],
    #         how="full",
    #         coalesce=True,
    #     ),
    #     [df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s, df_window],
    # ).sort(["iter", "positions"])
    # df_stats_norm = df_snps_norm.join(
    #     df_stats.select(pl.all().exclude(["iter", "delta_ihh", "ihs", "isafe", "nsl"])),
    #     on=["positions", "daf"],
    #     how="full",
    #     coalesce=True,
    # ).sort(["positions"])

    # df_stats_norm = (
    #     df_stats_norm.with_columns(
    #         [
    #             pl.lit(_iter).alias("iter"),
    #             # pl.lit(600000).alias("center"),
    #             # pl.lit(1200000).alias("window"),
    #             pl.col("positions").cast(pl.Int64),
    #         ]
    #     )
    #     .select(
    #         pl.col(["iter", "positions"]),
    #         pl.all().exclude(["iter", "positions"]),
    #         # pl.col(["iter", "center", "window", "positions"]),
    #         # pl.all().exclude(["iter", "center", "window", "positions"]),
    #     )
    #     .sort(["iter", "positions"])
    #     # .sort(["iter", "center", "window", "positions"])
    # )

    if region is not None:
        df_stats_norm = df_stats_norm.with_columns(pl.lit(region).alias("iter"))

    # sorting 500 SNP
    # mask = get_closest_snps(position_masked, 6e5, 500)
    # hap_sorted_corr_daf = daf_sorting(corr_sorting(hap_int[mask].T))

    del hap, hap_int, rec_map, rec_map_01, ac, biallelic_mask
    del position_masked, genetic_position_masked, freqs
    del df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s
    del df_isafe, df_ihs, df_nsl, df_window, df_stats, df_snps_norm
    gc.collect()
    # sys.stdout.flush()

    # return d_stats, df_stats_norm, hap_sorted_corr_daf
    return d_stats, df_stats_norm


def calculate_stats_simplify_full(
    hap_data,
    _iter=1,
    center=[5e5, 7e5],
    windows=[50000, 100000, 200000, 500000, 1000000],
    step=1e4,
    region=None,
):
    filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in scalar divide",
    )
    np.seterr(divide="ignore", invalid="ignore")

    if isinstance(hap_data, list) or isinstance(hap_data, tuple):
        hap, rec_map, p = hap_data
    elif isinstance(hap_data, str):
        try:
            hap, rec_map, p = ms_parser(hap_data)
            if hap.shape[0] != rec_map.shape[0]:
                return None, None, None
        except:
            try:
                hap, rec_map, p = genome_reader(hap_data, region)
            except:
                return None, None, None
    else:
        return None, None, None

    # Open and filtering data
    (
        hap_int,
        rec_map_01,
        ac,
        biallelic_mask,
        position_masked,
        genetic_position_masked,
    ) = filter_gt(hap, rec_map, region=region)
    freqs = ac[:, 1] / ac.sum(axis=1)

    if len(center) == 1:
        centers = np.arange(center[0], center[0] + step, step).astype(int)
    else:
        centers = np.arange(center[0], center[1] + step, step).astype(int)

    # Estimate fs stats
    df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s = run_fs_stats(
        hap_int[:, :], ac[:, :], rec_map_01[:, :]
    )

    d_stats = defaultdict(dict)

    df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
    df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
    df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
    df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

    # iSAFE
    df_isafe = run_isafe(hap_int, position_masked)
    df_isafe = center_window_cols(df_isafe, _iter=_iter)
    # iHS and nSL
    df_ihs = ihs_ihh(
        hap_int,
        position_masked,
        map_pos=genetic_position_masked,
        min_ehh=0.05,
        min_maf=0.05,
        include_edges=False,
    )

    df_ihs = center_window_cols(df_ihs, _iter=_iter)

    nsl_v = nsl(hap_int[freqs >= 0.05], use_threads=False)
    df_nsl = pl.DataFrame(
        {
            "positions": position_masked[freqs >= 0.05],
            "daf": freqs[freqs >= 0.05],
            "nsl": nsl_v,
        }
    ).fill_nan(None)
    df_nsl = center_window_cols(df_nsl, _iter=_iter)

    # raisd mu
    _r2 = compute_r2_matrix_upper(hap_int)
    # zns_v, omega_max = Ld(_r2)

    df_mu = mu_stat(hap_int, position_masked, _r2)

    # windowed stats
    _tmp_window = []
    for c, w in product(centers, windows):
        lower = c - w // 2
        upper = c + w // 2
        mask = (position_masked >= lower) & (position_masked <= upper)

        _tmp_hap = hap_int[mask]
        _tmp_pos = position_masked[mask]
        _ac = ac[mask]
        if _tmp_hap.size == 0:
            h12_v = (
                h2_h1
            ) = (
                haf_v
            ) = (
                _pos
            ) = (
                mu_var
            ) = (
                mu_sfs
            ) = (
                mu_ld
            ) = (
                mu_total
            ) = (
                zns_v
            ) = (
                omega_max
            ) = (
                tmp_pi
            ) = tmp_fay = tmp_tajd = tmp_y = tmp_e = tmp_fulid = tmp_fulif = np.nan
        else:
            try:
                #     h12_v, h2_h1 = h12_enard(
                #         hap_int[mask], position_masked[mask], window_size=int(5e5)
                #     )

                h12_v, h2_h1, h2_v = h12_enard_numba(
                    _tmp_hap,
                    _tmp_pos
                    # , window_size=int(5e5)
                )
                # h12_v, h2_h1 = garud_h_numba(_tmp_hap)[:2]
            except:
                h12_v, h2_h1, h2_v = np.nan, np.nan, np.nan

            haf_v = haf_top(
                _tmp_hap.astype(float),
                _tmp_pos,
                # window_size=int(5e5),
            )
            _pos, mu_var, mu_sfs, mu_ld, mu_total = (
                df_mu.filter(
                    (pl.col("positions") >= lower) & (pl.col("positions") <= upper)
                )
                .mean()
                .to_numpy()
                .flatten()
            )

            zns_v, omega_max = Ld(_r2, mask)

            tmp_pi = theta_pi(_ac).sum() / (upper - lower + 1)
            tmp_fay = fay_wu_h_norm_si(_ac)[-1]
            tmp_tajd = tajima_d(_ac)
            tmp_y = achaz_y(sfs_nb(_ac[:, -1], _ac.sum(axis=1)[0]))
            tmp_e = zeng_e_si(_ac)
            tmp_fulid = fuli_d_nb(_ac)
            tmp_fulif = fuli_f_nb(_ac)

        _tmp_window.append(
            np.array(
                [
                    int(_iter),
                    int(c),
                    int(w),
                    h12_v,
                    h2_h1,
                    haf_v,
                    tmp_pi,
                    tmp_fay,
                    omega_max,
                    zns_v,
                    mu_var,
                    mu_sfs,
                    mu_ld,
                    mu_total,
                    tmp_tajd,
                    tmp_y,
                    tmp_e,
                    tmp_fulid,
                    tmp_fulif,
                ]
            )
        )

    df_window_new = pl.DataFrame(
        np.vstack(_tmp_window),
        schema=pl.Schema(
            [
                ("iter", pl.Int64),
                ("center", pl.Int64),
                ("window", pl.Int64),
                ("h12", pl.Float64),
                ("h2_h1", pl.Float64),
                ("haf", pl.Float64),
                ("pi", pl.Float64),
                ("fay_wu_h", pl.Float64),
                ("omega_max", pl.Float64),
                ("zns", pl.Float64),
                ("mu_var", pl.Float64),
                ("mu_sfs", pl.Float64),
                ("mu_ld", pl.Float64),
                ("mu_total", pl.Float64),
                ("tajima_d", pl.Float64),
                ("achaz_y", pl.Float64),
                ("zeng_e", pl.Float64),
                ("fuli_d", pl.Float64),
                ("fuli_f", pl.Float64),
            ]
        ),
    )

    d_stats["dind_high_low"] = df_dind_high_low
    d_stats["s_ratio"] = df_s_ratio
    d_stats["hapdaf_o"] = df_hapdaf_o
    d_stats["hapdaf_s"] = df_hapdaf_s
    d_stats["ihs"] = df_ihs
    d_stats["isafe"] = df_isafe
    d_stats["nsl"] = df_nsl
    d_stats["h12_haf"] = df_window_new
    # d_stats["neutral_stats"] = df_window_new

    if region is not None:
        for k, df in d_stats.items():
            d_stats[k] = df.with_columns(pl.lit(region).alias("iter"))

    ####### if neutral:
    df_stats_norm = (
        reduce(
            lambda left, right: left.join(
                right,
                on=["iter", "positions", "daf"],
                how="full",
                coalesce=True,
            ),
            # [df_nsl, df_ihs_n, df_isafe],
            [
                df_nsl,
                df_ihs,
                df_isafe,
                df_dind_high_low,
                df_s_ratio,
                df_hapdaf_o,
                df_hapdaf_s,
            ],
        )
        # .with_columns(pl.lit(1200000).alias("window"))
        .sort(["iter", "positions"])
        # .sort(["iter", "center", "window", "positions"])
    )

    if region is not None:
        df_stats_norm = df_stats_norm.with_columns(pl.lit(region).alias("iter"))

    return d_stats, {"snps": df_stats_norm, "windows": df_window_new}


# def relative_position(positions, window, mask):
@njit
def relative_position(positions, window):
    window_start = window[0]
    # masked_positions = positions[mask]
    positions_relative = np.empty(positions.shape[0], dtype=np.int32)

    for i in range(positions.shape[0]):
        positions_relative[i] = positions[i] - window_start + 1

    return positions_relative


# def run_windowed_stats(hap, positions, window):
def run_windowed_stats(hap, positions, window):
    # mask = (positions >= window[0]) & (positions <= window[1])

    if hap.size != 0:
        # positions_relative = relative_position(positions, window, mask)
        # d_pos = {k: i for i, k in zip(range(1, int(1.2e6) + 1), range(window[0], window[1] + 1))}
        # positions_relative = np.array([d_pos[i] for i in positions[mask]])
        # df_isafe = run_isafe(hap[mask, :].astype(float), positions[mask]).fill_nan(None)

        # with numba_thread_control(nthreads):

        # Use 6e5 as actual center, change windows positions to range 1-1.2e6
        # Concordance with summary statistic simulation estimation
        positions_relative = relative_position(positions, window)
        df_isafe = run_isafe(hap.astype(float), positions).fill_nan(None)

        try:
            # h12_v, h2_h1 = h12_enard(hap[mask, :],positions_relative,window_size=int(5e5))
            h12_v, h2_h1, h2_v = h12_enard_numba(
                hap,
                positions_relative,
                # window_size=int(1e5)
                # hap[mask, :], positions_relative, window_size=int(1e5)
            )
        except:
            h12_v, h2_h1, h2_v = np.nan, np.nan, np.nan

        haf_v = haf_top(
            hap.astype(np.float64),
            # hap.astype(np.float64)[mask, :],
            positions_relative,
            # window_size=int(1e5),
            # window_size=int(5e5),
        )

        df_isafe = center_window_cols(df_isafe)

        # Middle point window as position
        df_h12_haf = pl.DataFrame(
            {
                "iter": pl.Series([1], dtype=pl.Int64),
                # "center": pl.Series([int(6e5)], dtype=pl.Int64),
                # "window": pl.Series([int(1.2e6)], dtype=pl.Int64),
                "positions": pl.Series([(window[0] + window[-1]) // 2], dtype=pl.Int64),
                "daf": pl.Series([1], dtype=pl.Float64),
                "h12": pl.Series([h12_v], dtype=pl.Float64),
                "h2_h1": pl.Series([h2_h1], dtype=pl.Float64),
                "haf": pl.Series([haf_v], dtype=pl.Float64),
            }
        ).fill_nan(None)
    else:
        df_h12_haf = pl.DataFrame(
            schema={
                "iter": pl.Int64,
                # "center": pl.Int64,
                # "window": pl.Int64,
                "positions": pl.Int64,
                "daf": pl.Float64,
                "h12": pl.Float64,
                "h2_h1": pl.Float64,
                "haf": pl.Float64,
            }
        )
        df_isafe = pl.DataFrame(
            schema={
                "iter": pl.Int64,
                # "center": pl.Int64,
                # "window": pl.Int64,
                "positions": pl.Int64,
                "daf": pl.Float64,
                "isafe": pl.Float64,
            }
        )

    return df_h12_haf, df_isafe


################## Normalization


def normalize_stats(
    stats_values,
    bins,
    center=[5e5, 7e5],
    # windows=[10000, 25000, 50000, 100000, 200000],
    windows=[50000, 100000, 200000, 500000, 1000000],
    parallel_manager=None,
    nthreads=1,
    vcf=False,
):
    df_fv, df_fv_raw = normalization_raw(
        deepcopy(stats_values),
        bins,
        center=center,
        windows=windows,
        parallel_manager=parallel_manager,
        nthreads=nthreads,
        vcf=vcf,
    )

    df_fv_w = pivot_feature_vectors(df_fv, vcf=vcf)
    df_fv_w_raw = pivot_feature_vectors(df_fv_raw, vcf=vcf)

    # dump fvs with more than 10% nans
    df_fv_w = df_fv_w.fill_nan(None)
    num_nans = (
        df_fv_w.select(
            pl.exclude(["iter", "s", "t", "f_i", "f_t", "model", "^.*flip.*$"])
        )
        .transpose()
        .null_count()
        .to_numpy()
        .flatten()
    )
    df_fv_w = df_fv_w.filter(
        num_nans
        < int(
            df_fv_w.select(
                pl.exclude(["iter", "s", "t", "f_i", "f_t", "model", "^.*flip.*$"])
            ).shape[1]
            * 0.1
        )
    ).fill_null(0)

    if not vcf:
        df_fv_w.sort(["iter", "model"])

    df_fv_w_raw = df_fv_w_raw.fill_nan(None)

    return df_fv_w, df_fv_w_raw


def normalization_raw(
    stats_values,
    bins,
    center=[5e5, 7e5],
    # windows=[10000, 25000, 50000, 100000, 200000],
    windows=[50000, 100000, 200000, 500000, 1000000],
    vcf=False,
    nthreads=1,
    parallel_manager=None,
):
    """
    Normalizes sweep statistics using neutral expectations or optionally using precomputed neutral normalized values.
    The function applies normalization across different genomic windows and supports multi-threading.

    Parameters
    ----------
    sweeps_stats : namedtuple
        A Namedtuple containing the statistics for genomic sweeps and sweep parameters across
        different genomic windows.

    neutral_stats_norm : namedtuple
        A Namedtuple containing the statistics for neutral region and neutral parameters across
        different genomic windows, used as the baselinefor normalizing the sweep statistics.
        This allows comparison of sweeps against neutral expectations.

    norm_values : dict or None, optional (default=None)
        A dictionary of precomputed neutral normalizated values. If provided, these values are
        used to directly normalize the statistics. If None, the function computes
        normalization values from the neutral statistics.

    center : list of float, optional (default=[5e5, 7e5])
        A list specifying the center positions (in base pairs) for the analysis windows.
        If a single center value is provided, normalization is centered around that value.
        Otherwise, it will calculate normalization for a range of positions between the two provided centers.

    windows : list of int, optional (default=[50000, 100000, 200000, 500000, 1000000])
        A list of window sizes (in base pairs) for which the normalization will be applied.
        The function performs normalization for each of the specified window sizes.

    nthreads : int, optional (default=1)
        The number of threads to use for parallel processing. If set to 1, the function
        runs in single-threaded mode. Higher values enable multi-threaded execution for
        faster computation.

    Returns
    -------
    normalized_stats : pandas.DataFrame
        A DataFrame containing the normalized sweep statistics across the specified
        windows and genomic regions. The sweep statistics are scaled relative to
        neutral expectations.
    """
    df_stats, params = deepcopy(stats_values)

    if vcf:
        df_h12_haf = df_stats.pop("h12_haf")
        snps_genome = reduce(
            lambda left, right: left.join(
                right,
                # on=["iter", "center", "window", "positions", "daf"],
                on=["iter", "positions", "daf"],
                how="full",
                coalesce=True,
            ),
            list(df_stats.values()),
        ).sort(["iter", "positions"])
        # ).sort(["iter", "center", "window", "positions"])
        nchr = snps_genome["iter"].unique().first()

        # Overwrite snps_values
        # snps_genome = snps_genome.select(pl.exclude(["h12", "haf"]))
        snps_values = snps_genome.select(pl.exclude(["h12", "haf"]))

        # stats_names = snps_values.select(snps_values.columns[5:]).columns
        stats_names = snps_values.select(snps_values.columns[3:]).columns

        binned_values = bin_values(snps_values)

        normalized_df = normalize_snps_statistics(
            binned_values, bins, stats_names
        ).sort("positions")

        # tmp_normalized = [normalized_df.filter((pl.col('positions') >= (w-6e5)) & (pl.col('positions') <= (w+6e5))) for w in center]
        # tmp_raw = [binned_values.filter((pl.col('positions') >= (w-6e5)) & (pl.col('positions') <= (w+6e5))) for w in center]
        left_idxs = np.searchsorted(
            normalized_df["positions"].to_numpy(), center - 6e5, side="left"
        )
        right_idxs = np.searchsorted(
            normalized_df["positions"].to_numpy(), center + 6e5, side="right"
        )
        tmp_normalized = [
            normalized_df.slice(start, end - start)
            for start, end in zip(left_idxs, right_idxs)
        ]
        tmp_raw = [
            binned_values.slice(start, end - start)
            for start, end in zip(left_idxs, right_idxs)
        ]

        if parallel_manager is None:
            df_fv_n = pl.concat(
                Parallel(n_jobs=nthreads, verbose=0)(
                    delayed(cut_snps)(
                        df,
                        None,
                        windows,
                        stats_names,
                        fixed_center=coord,
                        iter_value=coord,
                        # vcf=True,
                    )
                    for df, coord in zip(tmp_normalized, center)
                )
            )
            df_fv_n_raw = pl.concat(
                Parallel(n_jobs=nthreads, verbose=1)(
                    delayed(cut_snps)(
                        df,
                        None,
                        windows,
                        stats_names,
                        fixed_center=coord,
                        iter_value=coord,
                        # vcf=True,
                    )
                    for df, coord in zip(tmp_raw, center)
                )
            )
        else:
            df_fv_n = pl.concat(
                parallel_manager(
                    delayed(cut_snps)(
                        df,
                        center,
                        windows,
                        stats_names,
                        fixed_center=coord,
                        iter_value=coord,
                    )
                    for df, coord in zip(tmp_normalized, center)
                )
            )
            df_fv_n_raw = pl.concat(
                parallel_manager(
                    delayed(cut_snps)(
                        df,
                        center,
                        windows,
                        stats_names,
                        fixed_center=coord,
                        iter_value=coord,
                    )
                    for df, coord in zip(tmp_raw, center)
                )
            )

        window_values = (
            df_h12_haf.with_columns(pl.col("positions").alias("iter"))
            .filter(pl.col("positions").is_in(df_fv_n["iter"].unique().to_numpy()))
            .select(pl.exclude(["center", "window", "positions", "daf"]))
        )
        df_fv_n = df_fv_n.join(
            window_values,
            on=["iter"],
            how="full",
            coalesce=True,
        )
        df_fv_n_raw = df_fv_n_raw.join(
            window_values,
            on=["iter"],
            how="full",
            coalesce=True,
        )

        df_params_unpack = pl.DataFrame(
            np.repeat(
                np.zeros((1, 4)),
                df_fv_n.shape[0],
                axis=0,
            ),
            schema=["s", "t", "f_i", "f_t"],
        )
        df_fv_n = df_fv_n.with_columns(
            pl.lit(snps_genome["iter"].unique().first()).alias("nchr")
        )
        df_fv_n_raw = df_fv_n_raw.with_columns(
            pl.lit(snps_genome["iter"].unique().first()).alias("nchr")
        )
    else:
        if parallel_manager is None:
            df_fv_n_l, df_fv_n_l_raw = zip(
                *Parallel(n_jobs=nthreads, verbose=1)(
                    delayed(normalize_cut_raw)(
                        snps_values, bins, center=center, windows=windows
                    )
                    for _iter, snps_values in enumerate(df_stats, 1)
                )
            )
        else:
            df_fv_n_l, df_fv_n_l_raw = zip(
                *parallel_manager(
                    delayed(normalize_cut_raw)(
                        snps_values, bins, center=center, windows=windows
                    )
                    for _iter, snps_values in enumerate(df_stats, 1)
                )
            )

        df_fv_n = pl.concat(df_fv_n_l).with_columns(
            pl.col(["iter", "window", "center"]).cast(pl.Int64)
        )
        df_fv_n_raw = pl.concat(df_fv_n_l_raw).with_columns(
            pl.col(["iter", "window", "center"]).cast(pl.Int64)
        )

        df_window = pl.concat([df["h12_haf"] for df in df_stats]).select(
            pl.exclude(["center", "window", "positions", "daf"])
        )

        df_fv_n = df_fv_n.join(df_window, on=["iter"], how="full", coalesce=True)
        df_fv_n_raw = df_fv_n_raw.join(
            df_window, on=["iter"], how="full", coalesce=True
        )

        # params = params[:, [0, 1, 3, 4, ]]
        df_params_unpack = pl.DataFrame(
            np.repeat(
                params,
                df_fv_n.select(["center", "window"])
                .unique()
                .sort(["center", "window"])
                .shape[0],
                axis=0,
            ),
            schema=["s", "t", "f_i", "f_t"],
        )

    df_fv_n = pl.concat(
        [df_params_unpack, df_fv_n],
        how="horizontal",
    )
    df_fv_n_raw = pl.concat(
        [df_params_unpack, df_fv_n_raw],
        how="horizontal",
    )

    force_order = ["iter"] + [col for col in df_fv_n.columns if col != "iter"]
    df_fv_n = df_fv_n.select(force_order)
    df_fv_n_raw = df_fv_n_raw.select(force_order)

    return df_fv_n, df_fv_n_raw


def normalize_stats_full(
    stats_values,
    bins,
    center=[5e5, 7e5],
    # windows=[10000, 25000, 50000, 100000, 200000],
    windows=[50000, 100000, 200000, 500000, 1000000],
    parallel_manager=None,
    nthreads=1,
    vcf=False,
):
    df_fv, df_fv_raw = normalization_raw_full(
        deepcopy(stats_values),
        bins,
        center=center,
        windows=windows,
        parallel_manager=parallel_manager,
        nthreads=nthreads,
        vcf=vcf,
    )

    df_fv_w = pivot_feature_vectors(df_fv, vcf=vcf)
    df_fv_w_raw = pivot_feature_vectors(df_fv_raw, vcf=vcf)

    # dump fvs with more than 10% nans
    df_fv_w = df_fv_w.fill_nan(None)
    num_nans = (
        df_fv_w.select(
            pl.exclude(["iter", "s", "t", "f_i", "f_t", "model", "^.*flip.*$"])
        )
        .transpose()
        .null_count()
        .to_numpy()
        .flatten()
    )
    df_fv_w = df_fv_w.filter(
        num_nans
        < int(
            df_fv_w.select(
                pl.exclude(["iter", "s", "t", "f_i", "f_t", "model", "^.*flip.*$"])
            ).shape[1]
            * 0.1
        )
    ).fill_null(0)

    if not vcf:
        df_fv_w.sort(["iter", "model"])

    df_fv_w_raw = df_fv_w_raw.fill_nan(None)

    return df_fv_w, df_fv_w_raw


def normalization_raw_full(
    stats_values,
    bins,
    center=[5e5, 7e5],
    # windows=[10000, 25000, 50000, 100000, 200000],
    windows=[50000, 100000, 200000, 500000, 1000000],
    vcf=False,
    nthreads=1,
    parallel_manager=None,
):
    """
    Normalizes sweep statistics using neutral expectations or optionally using precomputed neutral normalized values.
    The function applies normalization across different genomic windows and supports multi-threading.

    Parameters
    ----------
    sweeps_stats : namedtuple
        A Namedtuple containing the statistics for genomic sweeps and sweep parameters across
        different genomic windows.

    neutral_stats_norm : namedtuple
        A Namedtuple containing the statistics for neutral region and neutral parameters across
        different genomic windows, used as the baselinefor normalizing the sweep statistics.
        This allows comparison of sweeps against neutral expectations.

    norm_values : dict or None, optional (default=None)
        A dictionary of precomputed neutral normalizated values. If provided, these values are
        used to directly normalize the statistics. If None, the function computes
        normalization values from the neutral statistics.

    center : list of float, optional (default=[5e5, 7e5])
        A list specifying the center positions (in base pairs) for the analysis windows.
        If a single center value is provided, normalization is centered around that value.
        Otherwise, it will calculate normalization for a range of positions between the two provided centers.

    windows : list of int, optional (default=[50000, 100000, 200000, 500000, 1000000])
        A list of window sizes (in base pairs) for which the normalization will be applied.
        The function performs normalization for each of the specified window sizes.

    nthreads : int, optional (default=1)
        The number of threads to use for parallel processing. If set to 1, the function
        runs in single-threaded mode. Higher values enable multi-threaded execution for
        faster computation.

    Returns
    -------
    normalized_stats : pandas.DataFrame
        A DataFrame containing the normalized sweep statistics across the specified
        windows and genomic regions. The sweep statistics are scaled relative to
        neutral expectations.
    """
    df_stats, params = deepcopy(stats_values)

    if vcf:
        df_h12_haf = df_stats.pop("h12_haf")
        snps_genome = reduce(
            lambda left, right: left.join(
                right,
                on=["iter", "center", "window", "positions", "daf"],
                how="full",
                coalesce=True,
            ),
            list(df_stats.values()),
        ).sort(["iter", "center", "window", "positions"])
        nchr = snps_genome["iter"].unique().first()

        # Overwrite snps_values
        # snps_genome = snps_genome.select(pl.exclude(["h12", "haf"]))
        snps_values = snps_genome.select(pl.exclude(["h12", "haf"]))

        stats_names = snps_values.select(snps_values.columns[5:]).columns

        binned_values = bin_values(snps_values)

        normalized_df = normalize_snps_statistics(
            binned_values, bins, stats_names
        ).sort("positions")

        # tmp_normalized = [normalized_df.filter((pl.col('positions') >= (w-6e5)) & (pl.col('positions') <= (w+6e5))) for w in center]
        # tmp_raw = [binned_values.filter((pl.col('positions') >= (w-6e5)) & (pl.col('positions') <= (w+6e5))) for w in center]
        left_idxs = np.searchsorted(
            normalized_df["positions"].to_numpy(), center - 6e5, side="left"
        )
        right_idxs = np.searchsorted(
            normalized_df["positions"].to_numpy(), center + 6e5, side="right"
        )
        tmp_normalized = [
            normalized_df.slice(start, end - start)
            for start, end in zip(left_idxs, right_idxs)
        ]
        tmp_raw = [
            binned_values.slice(start, end - start)
            for start, end in zip(left_idxs, right_idxs)
        ]

        if parallel_manager is None:
            df_fv_n = pl.concat(
                Parallel(n_jobs=nthreads, verbose=1)(
                    delayed(cut_snps)(
                        df,
                        None,
                        windows,
                        stats_names,
                        fixed_center=coord,
                        iter_value=coord,
                        # vcf=True,
                    )
                    for df, coord in zip(tmp_normalized, center)
                )
            )
            df_fv_n_raw = pl.concat(
                Parallel(n_jobs=nthreads, verbose=1)(
                    delayed(cut_snps)(
                        df,
                        None,
                        windows,
                        stats_names,
                        fixed_center=coord,
                        iter_value=coord,
                        # vcf=True,
                    )
                    for df, coord in zip(tmp_raw, center)
                )
            )
        else:
            df_fv_n = pl.concat(
                parallel_manager(
                    delayed(cut_snps)(
                        df,
                        center,
                        windows,
                        stats_names,
                        fixed_center=coord,
                        iter_value=coord,
                    )
                    for df, coord in zip(tmp_normalized, center_coords)
                )
            )
            df_fv_n_raw = pl.concat(
                parallel_manager(
                    delayed(cut_snps)(
                        df,
                        center,
                        windows,
                        stats_names,
                        fixed_center=coord,
                        iter_value=coord,
                    )
                    for df, coord in zip(tmp_raw, center_coords)
                )
            )
        window_values = (
            df_h12_haf.with_columns(pl.col("positions").alias("iter"))
            .filter(pl.col("positions").is_in(df_fv_n["iter"].unique().to_numpy()))
            .select(pl.exclude(["center", "window", "positions", "daf"]))
        )
        df_fv_n = df_fv_n.join(
            window_values,
            on=["iter"],
            how="full",
            coalesce=True,
        )
        df_fv_n_raw = df_fv_n_raw.join(
            window_values,
            on=["iter"],
            how="full",
            coalesce=True,
        )

        df_params_unpack = pl.DataFrame(
            np.repeat(
                np.zeros((1, 4)),
                df_fv_n.shape[0],
                axis=0,
            ),
            schema=["s", "t", "f_i", "f_t"],
        )
        df_fv_n = df_fv_n.with_columns(
            pl.lit(snps_genome["iter"].unique().first()).alias("nchr")
        )
        df_fv_n_raw = df_fv_n_raw.with_columns(
            pl.lit(snps_genome["iter"].unique().first()).alias("nchr")
        )
    else:
        if parallel_manager is None:
            df_fv_n_l, df_fv_n_l_raw = zip(
                *Parallel(n_jobs=nthreads, verbose=1)(
                    delayed(normalize_cut_raw_full)(
                        snps_values, bins, center=center, windows=windows
                    )
                    for _iter, snps_values in enumerate(df_stats, 1)
                )
            )
        else:
            df_fv_n_l, df_fv_n_l_raw = zip(
                *parallel_manager(
                    delayed(normalize_cut_raw_full)(
                        snps_values, bins, center=center, windows=windows
                    )
                    for _iter, snps_values in enumerate(df_stats, 1)
                )
            )

        df_fv_n = pl.concat(df_fv_n_l).with_columns(
            pl.col(["iter", "window", "center"]).cast(pl.Int64)
        )
        df_fv_n_raw = pl.concat(df_fv_n_l_raw).with_columns(
            pl.col(["iter", "window", "center"]).cast(pl.Int64)
        )

        # params = params[:, [0, 1, 3, 4, ]]
        df_params_unpack = pl.DataFrame(
            np.repeat(
                params,
                df_fv_n.select(["center", "window"])
                .unique()
                .sort(["center", "window"])
                .shape[0],
                axis=0,
            ),
            schema=["s", "t", "f_i", "f_t"],
        )

    df_fv_n = pl.concat(
        [df_params_unpack, df_fv_n],
        how="horizontal",
    )
    df_fv_n_raw = pl.concat(
        [df_params_unpack, df_fv_n_raw],
        how="horizontal",
    )

    force_order = ["iter"] + [col for col in df_fv_n.columns if col != "iter"]
    df_fv_n = df_fv_n.select(force_order)
    df_fv_n_raw = df_fv_n_raw.select(force_order)

    return df_fv_n, df_fv_n_raw


def normalize_cut_raw_full(
    snps_values,
    bins,
    center=[5e5, 7e5],
    # windows=[10000, 25000, 50000, 100000, 200000],
    windows=[50000, 100000, 200000, 500000, 1000000],
    step=int(1e4),
):
    """
    Normalizes SNP-level statistics by comparing them to neutral expectations, and aggregates
    the statistics within sliding windows around specified genomic centers.

    This function takes SNP statistics, normalizes them based on the expected mean and standard
    deviation from neutral simulations, and computes the average values within windows
    centered on specific genomic positions. It returns a DataFrame with the normalized values
    for each window across the genome.

    Parameters
    ----------
    _iter : int
        The iteration or replicate number associated with the current set of SNP statistics.

    snps_values : pandas.DataFrame
        A DataFrame containing SNP-level statistics such as iHS, nSL, and iSAFE. The DataFrame
        should contain derived allele frequencies ("daf") and other statistics to be normalized.

    expected : pandas.DataFrame
        A DataFrame containing the expected mean values of the SNP statistics for each frequency bin,
        computed from neutral simulations.

    stdev : pandas.DataFrame
        A DataFrame containing the standard deviation of the SNP statistics for each frequency bin,
        computed from neutral simulations.

    center : list of float, optional (default=[5e5, 7e5])
        A list specifying the center positions (in base pairs) for the analysis. Normalization is
        performed around these genomic centers using the specified window sizes.

    windows : list of int, optional (default=[50000, 100000, 200000, 500000, 1000000])
        A list of window sizes (in base pairs) over which the SNP statistics will be aggregated
        and normalized. The function performs normalization for each specified window size.

    Returns
    -------
    out : pandas.DataFrame
        A DataFrame containing the normalized SNP statistics for each genomic center and window.
        The columns include the iteration number, center, window size, and the average values
        of normalized statistics iSAFE within the window.

    Notes
    -----
    - The function first bins the SNP statistics based on derived allele frequencies using the
      `bin_values` function. The statistics are then normalized by subtracting the expected mean
      and dividing by the standard deviation for each frequency bin.
    - After normalization, SNPs are aggregated into windows centered on specified genomic positions.
      The average values of the normalized statistics are calculated for each window.
    - The window size determines how far upstream and downstream of the center position the SNPs
      will be aggregated.

    """
    df_out = []
    df_out_raw = []
    _iter = np.unique([i.select("iter").unique() for i in snps_values.values()])

    if len(center) == 2:
        centers = np.arange(center[0], center[1] + step, step).astype(int)
    else:
        centers = center

    df = reduce(
        lambda left, right: left.join(
            right,
            on=["iter", "positions", "daf"],
            # on=["iter", "center", "window", "positions", "daf"],
            how="full",
            coalesce=True,
        ),
        # [v for k, v in snps_values.items()],
        [v for k, v in snps_values.items() if k != "h12_haf" and k != "neutral_stats"],
    )
    df_window = snps_values["h12_haf"].select(pl.exclude("positions"))
    stats_names = df[:, 3:].columns
    # stats_names = df[:, 5:].columns

    binned_values = bin_values(df)
    normalized_df, normalized_window = normalize_snps_statistics_full(
        binned_values, df_window, bins, stats_names
    )

    df_out = cut_snps(
        normalized_df,
        centers,
        windows,
        stats_names,
        fixed_center=None,
        iter_value=_iter,
    )
    df_out_raw = cut_snps(
        df,
        centers,
        windows,
        stats_names,
        fixed_center=None,
        iter_value=_iter,
    )
    df_out = df_out.join(
        normalized_window, on=["iter", "center", "window"], how="full", coalesce=True
    )
    df_out_raw = df_out_raw.join(
        df_window, on=["iter", "center", "window"], how="full", coalesce=True
    )

    return df_out, df_out_raw


def vectorized_cut_vcf(
    normalized_df: pl.DataFrame,
    centers: np.ndarray,
    windows: list[int],
    stats_names: list[str],
) -> pl.DataFrame:
    """
    For each genomic center in `centers`, this function examines
    21 sub-centers evenly spaced ±100kb around it, and for each
    window size in `windows` computes the mean of each column in
    `stats_names` over that sub-interval.

    It does so in O(N) + O(len(centers)*21*len(windows)) time by:

      • building zero-prepended prefix sums (and count arrays)
        for each stat to allow O(1) range-sum queries;
      • using two global searchsorted calls to locate all
        window boundaries at once;
      • performing all mean computations via vectorized NumPy
        arithmetic; and
      • constructing one big Polars DataFrame at the end.

    Parameters
    ----------
    normalized_df
        Polars DataFrame with a sorted "positions" column (i64)
        plus the numeric columns in `stats_names` (f64).
    centers
        1D NumPy array of genomic center positions to query.
    windows
        List of integer window sizes (e.g. [50000,100000,…]).
    stats_names
        List of column names in `normalized_df` to average.

    Returns
    -------
    Polars DataFrame
        Rows = len(centers) * 21 * len(windows). Columns:
        "iter"  (outer center),
        "center" (sub-center = iter + offset),
        "window" (window size),
        plus one column per entry in `stats_names`.
    """

    # Extract positions as a NumPy array for fast searchsorted
    positions = normalized_df["positions"].to_numpy()
    n = positions.size

    # Build prefix-sums and counts for each stat, with a leading zero
    ps0, pc0 = {}, {}
    for stat in stats_names:
        arr = normalized_df[stat].to_numpy()
        valid = ~np.isnan(arr)
        csum = np.cumsum(np.where(valid, arr, 0.0))
        cnt = np.cumsum(valid.astype(np.int64))

        # Prepend zero so that sum over [i:j] is ps0[j] - ps0[i]
        ps0[stat] = np.concatenate(([0.0], csum))
        pc0[stat] = np.concatenate(([0], cnt))

    # Prepare the 21 inner offsets (±100kb) and tile with each window
    inner_offsets = np.linspace(-100_000, +100_000, 21, dtype=np.int64)
    W = len(windows)
    inner_base = np.repeat(inner_offsets, W)  # repeat each offset for all windows
    window_base = np.tile(
        windows, inner_offsets.size
    )  # repeat all windows for each offset
    M = inner_base.size  # total sub-queries per center (21 * W)

    # Tile the base arrays for every center
    C = centers.size
    iter_col = np.repeat(centers, M)  # each center repeated M times
    inner_col = iter_col + np.tile(inner_base, C)  # absolute sub-center positions
    window_col = np.tile(window_base, C)  # window sizes aligned per sub-center

    # Compute lower/upper bounds for each sub-window and lookup indices
    lowers = inner_col - (window_col // 2)
    uppers = inner_col + (window_col // 2)
    start_idx = np.searchsorted(positions, lowers, side="left")
    end_idx = np.searchsorted(positions, uppers, side="right")

    # Allocate output dict with the iter/center/window columns
    out = {
        "iter": iter_col,
        "center": inner_col,
        "window": window_col,
    }

    # Compute means using prefix-sum differences and safe division
    for stat in stats_names:
        sb = ps0[stat]
        cb = pc0[stat]

        # Total sum and count in each interval
        svals = sb[end_idx] - sb[start_idx]
        cvals = cb[end_idx] - cb[start_idx]

        # Initialize result array with NaN for empty windows
        mean_vals = np.full_like(svals, np.nan, dtype=np.float64)
        # Divide only where count>0, avoids divide-by-zero warnings
        np.divide(svals, cvals, out=mean_vals, where=(cvals > 0))

        out[stat] = mean_vals

    # Build and return the final Polars DataFrame in one shot
    return pl.DataFrame(out)


def normalize_cut_raw(
    snps_values,
    bins,
    center=[5e5, 7e5],
    # windows=[10000, 25000, 50000, 100000, 200000],
    windows=[50000, 100000, 200000, 500000, 1000000],
    step=int(1e4),
):
    """
    Normalizes SNP-level statistics by comparing them to neutral expectations, and aggregates
    the statistics within sliding windows around specified genomic centers.

    This function takes SNP statistics, normalizes them based on the expected mean and standard
    deviation from neutral simulations, and computes the average values within windows
    centered on specific genomic positions. It returns a DataFrame with the normalized values
    for each window across the genome.

    Parameters
    ----------
    _iter : int
        The iteration or replicate number associated with the current set of SNP statistics.

    snps_values : pandas.DataFrame
        A DataFrame containing SNP-level statistics such as iHS, nSL, and iSAFE. The DataFrame
        should contain derived allele frequencies ("daf") and other statistics to be normalized.

    expected : pandas.DataFrame
        A DataFrame containing the expected mean values of the SNP statistics for each frequency bin,
        computed from neutral simulations.

    stdev : pandas.DataFrame
        A DataFrame containing the standard deviation of the SNP statistics for each frequency bin,
        computed from neutral simulations.

    center : list of float, optional (default=[5e5, 7e5])
        A list specifying the center positions (in base pairs) for the analysis. Normalization is
        performed around these genomic centers using the specified window sizes.

    windows : list of int, optional (default=[50000, 100000, 200000, 500000, 1000000])
        A list of window sizes (in base pairs) over which the SNP statistics will be aggregated
        and normalized. The function performs normalization for each specified window size.

    Returns
    -------
    out : pandas.DataFrame
        A DataFrame containing the normalized SNP statistics for each genomic center and window.
        The columns include the iteration number, center, window size, and the average values
        of normalized statistics iSAFE within the window.

    Notes
    -----
    - The function first bins the SNP statistics based on derived allele frequencies using the
      `bin_values` function. The statistics are then normalized by subtracting the expected mean
      and dividing by the standard deviation for each frequency bin.
    - After normalization, SNPs are aggregated into windows centered on specified genomic positions.
      The average values of the normalized statistics are calculated for each window.
    - The window size determines how far upstream and downstream of the center position the SNPs
      will be aggregated.

    """
    df_out = []
    df_out_raw = []
    _iter = np.unique([i.select("iter").unique() for i in snps_values.values()])

    if len(center) == 2:
        centers = np.arange(center[0], center[1] + step, step).astype(int)
    else:
        centers = center

    df = reduce(
        lambda left, right: left.join(
            right,
            # on=["iter", "center", "window", "positions", "daf"],
            on=["iter", "positions", "daf"],
            how="full",
            coalesce=True,
        ),
        [v for k, v in snps_values.items() if k != "h12_haf"],
    )

    df_window = snps_values["h12_haf"]

    stats_names = df[:, 3:].columns

    binned_values = bin_values(df)
    normalized_df = normalize_snps_statistics(binned_values, bins, stats_names)

    df_out = cut_snps(
        normalized_df,
        centers,
        windows,
        stats_names,
        fixed_center=None,
        iter_value=_iter,
    )
    df_out_raw = cut_snps(
        df,
        centers,
        windows,
        stats_names,
        fixed_center=None,
        iter_value=_iter,
    )

    return df_out, df_out_raw


def normalize_snps_statistics(df_snps, bins, stats_names, norm_type=1):
    """
    Normalizes SNP-level statistics by comparing them to neutral expectations.

    Parameters
    ----------
    df_snps : polars.DataFrame
        DataFrame containing SNP-level statistics with binned frequency values.
    binned_stats : dict or bins
        Neutral and empirical statistics, either as a dictionary with 'neutral' and 'empirical'
        keys (containing mean and std) or a bins object.
    stats_names : list
        List of statistical measure column names to normalize.

    Returns
    -------
    polars.DataFrame
        DataFrame with normalized statistics.
    """
    if not stats_names:
        raise ValueError("stats_names cannot be empty")

    if norm_type not in [1, 2, 3, 4]:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

    # Handle dictionary or defaultdict case
    if isinstance(bins, (dict, defaultdict)):
        # Extract statistics dataframes
        neutral_means = bins["neutral"].mean.select(["freq_bins"] + stats_names)
        neutral_stds = bins["neutral"].std.select(["freq_bins"] + stats_names)
        empirical_means = bins["empirical"].mean.select(["freq_bins"] + stats_names)
        empirical_stds = bins["empirical"].std.select(["freq_bins"] + stats_names)

        # Join all statistics to the input dataframe
        normalized_df = (
            df_snps.join(
                neutral_means,
                on="freq_bins",
                how="left",
                suffix="_mean_neutral",
                coalesce=True,
            )
            .join(
                neutral_stds,
                on="freq_bins",
                how="left",
                suffix="_std_neutral",
                coalesce=True,
            )
            .join(
                empirical_means,
                on="freq_bins",
                how="left",
                suffix="_mean_empirical",
                coalesce=True,
            )
            .join(
                empirical_stds,
                on="freq_bins",
                how="left",
                suffix="_std_empirical",
                coalesce=True,
            )
        )

        # Apply normalization based on type
        if norm_type == 1:
            normalized_cols = [
                (
                    (pl.col(s) - pl.col(f"{s}_mean_empirical"))
                    / pl.col(f"{s}_std_empirical")
                ).alias(s)
                for s in stats_names
            ]
        elif norm_type == 2:
            normalized_cols = [
                (
                    (
                        (pl.col(s) - pl.col(f"{s}_mean_empirical"))
                        / pl.col(f"{s}_std_empirical")
                    )
                    * (
                        (pl.col(f"{s}_mean_empirical") - pl.col(f"{s}_mean_neutral"))
                        / pl.col(f"{s}_std_neutral")
                    )
                ).alias(s)
                for s in stats_names
            ]
        elif norm_type == 3:
            normalized_cols = [
                (
                    (
                        (pl.col(s) - pl.col(f"{s}_mean_empirical"))
                        / pl.col(f"{s}_std_empirical")
                    )
                    * (
                        1
                        + (
                            (
                                pl.col(f"{s}_mean_empirical")
                                - pl.col(f"{s}_mean_neutral")
                            )
                            / pl.col(f"{s}_std_neutral")
                        ).abs()
                    )
                ).alias(s)
                for s in stats_names
            ]
        elif norm_type == 4:
            normalized_cols = [
                (
                    (
                        (pl.col(s) - pl.col(f"{s}_mean_empirical"))
                        / pl.col(f"{s}_std_empirical")
                    )
                    * (
                        1
                        + (
                            (
                                pl.col(f"{s}_mean_empirical")
                                - pl.col(f"{s}_mean_neutral")
                            )
                            / pl.col(f"{s}_std_neutral")
                        )
                    )
                ).alias(s)
                for s in stats_names
            ]

    elif isinstance(bins, binned_stats):
        neutral_means = bins.mean.select(["freq_bins"] + stats_names)
        neutral_stds = bins.std.select(["freq_bins"] + stats_names)

        normalized_df = (
            df_snps.join(
                neutral_means,
                on="freq_bins",
                how="left",
                coalesce=True,
                suffix="_mean_neutral",
            )
            .join(
                neutral_stds,
                on="freq_bins",
                how="left",
                coalesce=True,
                suffix="_std_neutral",
            )
            .fill_nan(None)
        )

        # Normalize with empirical statistics (assumed provided elsewhere)
        normalized_cols = [
            (
                (pl.col(s) - pl.col(f"{s}_mean_neutral")) / pl.col(f"{s}_std_neutral")
            ).alias(s)
            for s in stats_names
        ]

    # Perform normalization and return the final DataFrame
    return (
        normalized_df.with_columns(normalized_cols)
        .select(["positions"] + stats_names)
        .sort(["positions"])
        # normalized_df.with_columns(normalized_cols).select(["positions", "center", "window"] + stats_names).sort(["positions", "center", "window"])
    )


def normalize_snps_statistics_full(df_snps, df_window, bins, stats_names):
    """
    Normalizes SNP-level statistics by comparing them to neutral expectations.

    Parameters
    ----------
    df_snps : polars.DataFrame
        DataFrame containing SNP-level statistics with binned frequency values.
    binned_stats : dict or bins
        Neutral and empirical statistics, either as a dictionary with 'neutral' and 'empirical'
        keys (containing mean and std) or a bins object.
    stats_names : list
        List of statistical measure column names to normalize.

    Returns
    -------
    polars.DataFrame
        DataFrame with normalized statistics.
    """
    if not stats_names:
        raise ValueError("stats_names cannot be empty")

    neutral_means = bins.mean[0].select(["freq_bins"] + stats_names)
    neutral_stds = bins.std[0].select(["freq_bins"] + stats_names)

    normalized_df = (
        df_snps.join(
            neutral_means,
            on="freq_bins",
            how="left",
            coalesce=True,
            suffix="_mean_neutral",
        )
        .join(
            neutral_stds,
            on="freq_bins",
            how="left",
            coalesce=True,
            suffix="_std_neutral",
        )
        .fill_nan(None)
    )

    # Normalize with empirical statistics (assumed provided elsewhere)
    normalized_cols = [
        ((pl.col(s) - pl.col(f"{s}_mean_neutral")) / pl.col(f"{s}_std_neutral")).alias(
            s
        )
        for s in stats_names
    ]
    normalized_df = normalized_df.with_columns(normalized_cols).select(
        ["positions"]
        + stats_names
        # ["positions", "center", "window"] + stats_names
    )
    df_window_normalized = pl.concat(
        [
            df_window[:, :3],
            (df_window[:, 3:] - bins.mean[1][:, 2:]) / bins.std[1][:, 2:],
        ],
        how="horizontal",
    )
    # Perform normalization and return the final DataFrame
    return (normalized_df, df_window_normalized)


def cut_snps(df, centers, windows, stats_names, fixed_center=None, iter_value=1):
    """
    Processes data within windows across multiple centers and window sizes.

    Parameters
    ----------
    normalized_df : polars.DataFrame
        DataFrame containing the positions and statistics.
    iter_value : int
        Iteration or replicate number.
    centers : list
        List of center positions to analyze.
    windows : list
        List of window sizes to use.
    stats_names : list, optional
        Names of statistical columns to compute means for.
        If None, all columns except position-related ones will be used.
    position_col : str, optional
        Name of the column containing position values.
    center_col : str, optional
        Name of the column containing center values.
    fixed_center : int, optional
        If provided, use this fixed center value instead of the ones in centers list.

    Returns
    -------
    polars.DataFrame
        DataFrame with aggregated statistics for each center and window.
    """
    # If stats_names not provided, use all appropriate columns

    # reset centers

    if centers is None:
        centers = np.arange(5e5, 7e5 + 1e4, 1e4).astype(int)

    results = []
    for c, w in list(product(centers, windows)):
        query = df.lazy()

        # 1.2MB simulations derives into 21 center/windows combinations
        if fixed_center is not None:
            c_fix = fixed_center
        else:
            c_fix = c

        # Filter data by center and window boundaries
        # Define window boundaries
        lower = c_fix - w // 2
        upper = c_fix + w // 2

        window_data = query.filter(
            (pl.col("positions") >= lower) & (pl.col("positions") <= upper)
        )

        # Calculate mean statistics for window
        window_stats = window_data.select(stats_names).fill_nan(None).mean().collect()

        # Add metadata columns
        metadata_cols = [
            pl.lit(iter_value).alias("iter"),
            pl.lit(c).alias("center"),
            pl.lit(w).alias("window"),
        ]

        results.append(window_stats.with_columns(metadata_cols))

    return (
        pl.concat(results, how="vertical").select(
            ["iter", "center", "window"] + stats_names
        )
        if results
        else None
    )


def normalize_neutral(df_stats_neutral, mask=False):
    """
    Calculates the expected mean and standard deviation of summary statistics
    from neutral simulations, used for normalization in downstream analyses.

    This function processes a DataFrame of neutral simulation statistics, bins the
    values based on frequency, and computes the mean (expected) and standard deviation
    for each bin. These statistics are used as a baseline to normalize sweep or neutral simulations

    Parameters
    ----------
    df_stats_neutral : list or pandas.DataFrame
        A list or concatenated pandas DataFrame containing the neutral simulation statistics.
        The DataFrame should contain frequency data and various summary statistics,
        including H12 and HAF, across multiple windows and bins.

    Returns
    -------
    expected : pandas.DataFrame
        A DataFrame containing the mean (expected) values of the summary statistics
        for each frequency bin. The frequency bins are the index, and the columns
        are the summary statistics.

    stdev : pandas.DataFrame
        A DataFrame containing the standard deviation of the summary statistics
        for each frequency bin. The frequency bins are the index, and the columns
        are the summary statistics.

    Notes
    -----
    - The function first concatenates the neutral statistics, if provided as a list,
      and bins the values by frequency using the `bin_values` function.
    - It computes both the mean and standard deviation for each frequency bin, which
      can later be used to normalize observed statistics (e.g., from sweeps).
    - The summary statistics processed exclude window-specific statistics such as "h12" and "haf."

    """
    # df_snps, df_window = df_stats_neutral

    window_stats = ["h12", "haf"]

    if mask:
        tmp = []
        for df in df_stats:
            centromeres = {
                1: 125,
                2: 93.3,
                3: 91,
                4: 50.4,
                5: 48.4,
                6: 61,
                7: 59.9,
                8: 45.6,
                9: 49,
                10: 40.2,
                11: 53.7,
                12: 35.8,
                13: 17.9,
                14: 17.6,
                15: 19,
                16: 36.6,
                17: 24,
                18: 17.2,
                19: 26.5,
                20: 27.5,
                21: 13.2,
                22: 14.7,
            }
            chr_l = {
                1: 247.2,
                2: 242.8,
                3: 199.4,
                4: 191.3,
                5: 180.8,
                6: 170.9,
                7: 158.8,
                8: 146.3,
                9: 140.4,
                10: 135.4,
                11: 134.5,
                12: 132.3,
                13: 114.1,
                14: 106.3,
                15: 100.3,
                16: 88.8,
                17: 78.7,
                18: 76.1,
                19: 63.8,
                20: 62.4,
                21: 46.9,
                22: 49.5,
            }
            nchr = df["iter"].unique().first()

            centromer_window = (2.5e-6 - centromeres[nchr], centromeres[nchr] + 2.5e-6)
            telomer_window = (2.5e-6, chr_l[nchr] - 2.5e-6)
            pos = df["positions"].to_numpy()
            centromer_mask = (pos < centromer_window[0]) & (pos > centromer_window[1])
            telomer_mask = (pos > telomer_window[0]) & (pos < telomer_window[1])
            mask = centromer_mask & telomer_mask

            tmp.append(df.filter(mask))
        tmp_neutral = pl.concat(tmp)
    else:
        # Get std and mean values from dataframe
        tmp_neutral = pl.concat(df_stats_neutral).fill_nan(None)

    df_binned = bin_values(tmp_neutral.select(pl.exclude(window_stats)))

    # get expected value (mean) and standard deviation
    expected = (
        df_binned.select(df_binned.columns[3:])
        .group_by("freq_bins")
        .mean()
        .sort("freq_bins")
        .fill_nan(None)
    )
    stdev = (
        df_binned.select(df_binned.columns[3:])
        .group_by("freq_bins")
        .agg([pl.all().exclude("freq_bins").std()])
        .sort("freq_bins")
        .fill_nan(None)
    )

    # expected.index = expected.index.astype(str)
    # stdev.index = stdev.index.astype(str)

    return expected, stdev


def normalize_neutral_full(d_stats_neutral, mask=False):
    """
    Calculates the expected mean and standard deviation of summary statistics
    from neutral simulations, used for normalization in downstream analyses.

    This function processes a DataFrame of neutral simulation statistics, bins the
    values based on frequency, and computes the mean (expected) and standard deviation
    for each bin. These statistics are used as a baseline to normalize sweep or neutral simulations

    Parameters
    ----------
    df_stats_neutral : list or pandas.DataFrame
        A list or concatenated pandas DataFrame containing the neutral simulation statistics.
        The DataFrame should contain frequency data and various summary statistics,
        including H12 and HAF, across multiple windows and bins.

    Returns
    -------
    expected : pandas.DataFrame
        A DataFrame containing the mean (expected) values of the summary statistics
        for each frequency bin. The frequency bins are the index, and the columns
        are the summary statistics.

    stdev : pandas.DataFrame
        A DataFrame containing the standard deviation of the summary statistics
        for each frequency bin. The frequency bins are the index, and the columns
        are the summary statistics.

    Notes
    -----
    - The function first concatenates the neutral statistics, if provided as a list,
      and bins the values by frequency using the `bin_values` function.
    - It computes both the mean and standard deviation for each frequency bin, which
      can later be used to normalize observed statistics (e.g., from sweeps).
    - The summary statistics processed exclude window-specific statistics such as "h12" and "haf."

    """
    # df_snps, df_window = df_stats_neutral

    if mask:
        tmp = []
        for df in df_stats:
            centromeres = {
                1: 125,
                2: 93.3,
                3: 91,
                4: 50.4,
                5: 48.4,
                6: 61,
                7: 59.9,
                8: 45.6,
                9: 49,
                10: 40.2,
                11: 53.7,
                12: 35.8,
                13: 17.9,
                14: 17.6,
                15: 19,
                16: 36.6,
                17: 24,
                18: 17.2,
                19: 26.5,
                20: 27.5,
                21: 13.2,
                22: 14.7,
            }
            chr_l = {
                1: 247.2,
                2: 242.8,
                3: 199.4,
                4: 191.3,
                5: 180.8,
                6: 170.9,
                7: 158.8,
                8: 146.3,
                9: 140.4,
                10: 135.4,
                11: 134.5,
                12: 132.3,
                13: 114.1,
                14: 106.3,
                15: 100.3,
                16: 88.8,
                17: 78.7,
                18: 76.1,
                19: 63.8,
                20: 62.4,
                21: 46.9,
                22: 49.5,
            }
            nchr = df["iter"].unique().first()

            centromer_window = (2.5e-6 - centromeres[nchr], centromeres[nchr] + 2.5e-6)
            telomer_window = (2.5e-6, chr_l[nchr] - 2.5e-6)
            pos = df["positions"].to_numpy()
            centromer_mask = (pos < centromer_window[0]) & (pos > centromer_window[1])
            telomer_mask = (pos > telomer_window[0]) & (pos < telomer_window[1])
            mask = centromer_mask & telomer_mask

            tmp.append(df.filter(mask))
        tmp_neutral = pl.concat(tmp)
    else:
        snps_list, windows_list = zip(
            *((d["snps"], d["windows"]) for d in d_stats_neutral)
        )
        tmp_neutral_snps = pl.concat(snps_list, how="vertical").fill_nan(None)
        tmp_neutral_window = pl.concat(windows_list, how="vertical").fill_nan(None)

    df_binned = bin_values(
        tmp_neutral_snps.select(
            pl.exclude(
                [
                    "h12",
                    "haf",
                    # "pi",
                    "fay_wu_h",
                    "omega_max",
                    "zns",
                    # "mu_var",
                    # "mu_sfs",
                    # "mu_ld",
                    # "mu_total",
                ]
            )
        )
    )

    df_window = tmp_neutral_window.select(
        [
            "center",
            "window",
            "h12",
            "h2_h1",
            "haf",
            "pi",
            "fay_wu_h",
            "omega_max",
            "zns",
            "mu_var",
            "mu_sfs",
            "mu_ld",
            "mu_total",
            "tajima_d",
            "achaz_y",
            "zeng_e",
            "fuli_d",
            "fuli_f",
        ]
    ).drop_nulls()

    df_window_mean = (
        df_window.group_by("center", "window").mean().sort("center", "window")
    )
    df_window_std = (
        df_window.group_by("center", "window").mean().sort("center", "window")
    )
    # get expected value (mean) and standard deviation
    expected = (
        df_binned.select(df_binned.columns[3:])
        # df_binned.select(df_binned.columns[5:])
        .group_by("freq_bins")
        .mean()
        .sort("freq_bins")
        .fill_nan(None)
    )
    stdev = (
        df_binned.select(df_binned.columns[3:])
        # df_binned.select(df_binned.columns[5:])
        .group_by("freq_bins")
        .agg([pl.all().exclude("freq_bins").std()])
        .sort("freq_bins")
        .fill_nan(None)
    )

    # expected = expected.join(df_window_mean,on='freq_bins', how='full', coalesce=True)
    # stdev = stdev.join(df_window_std,on='freq_bins', how='full', coalesce=True)

    # expected.index = expected.index.astype(str)
    # stdev.index = stdev.index.astype(str)

    return ([expected, df_window_mean], [stdev, df_window_std])


def bin_values(values, freq=0.02):
    """
    Bins allele frequency data into discrete frequency intervals (bins) for further analysis.

    This function takes a DataFrame containing a column of derived allele frequencies ("daf")
    and bins these values into specified frequency intervals. The resulting DataFrame will
    contain a new column, "freq_bins", which indicates the frequency bin for each data point.

    Parameters
    ----------
    values : pandas.DataFrame
        A DataFrame containing at least a column labeled "daf", which represents the derived
        allele frequency for each variant.

    freq : float, optional (default=0.02)
        The width of the frequency bins. This value determines how the frequency range (0, 1)
        is divided into discrete bins. For example, a value of 0.02 will create bins
        such as [0, 0.02], (0.02, 0.04], ..., [0.98, 1.0].

    Returns
    -------
    values_copy : pandas.DataFrame
        A copy of the original DataFrame, with an additional column "freq_bins" that contains
        the frequency bin label for each variant. The "freq_bins" are categorical values based
        on the derived allele frequencies.

    Notes
    -----
    - The `pd.cut` function is used to bin the derived allele frequencies into intervals.
    - The bins are inclusive of the lowest boundary (`include_lowest=True`) to ensure that
      values exactly at the boundary are included in the corresponding bin.
    - The resulting bins are labeled as strings with a precision of two decimal places.
    """
    # Modify the copy
    values_copy = pl.concat(
        [
            values,
            values["daf"]
            .cut(np.arange(0, 1 + freq, freq))
            .to_frame()
            .rename({"daf": "freq_bins"}),
        ],
        how="horizontal",
    )

    return values_copy.sort("iter", "positions")


################## Haplotype length stats


def run_hapbin(
    hap,
    rec_map,
    _iter=1,
    min_ehh=0.05,
    min_maf=0.05,
    gap_scale=20000,
    max_extend=1000000,
    binom=False,
    ihsbin=None,
):
    if ihsbin is None:
        ihsbin = shutil.which("ihsbin")
    df_hap = pl.DataFrame(hap)
    df_rec_map = pl.DataFrame(
        rec_map,
        pl.Schema(
            [
                ("chr", pl.Int64),
                ("location", pl.Int64),
                ("cm_mb", pl.Float64),
                ("cm", pl.Float64),
            ]
        ),
    )
    hap_file = "/tmp/tmp_" + str(_iter) + ".hap"
    map_file = "/tmp/tmp_" + str(_iter) + ".map"

    df_hap.write_csv(hap_file, separator=" ", include_header=False)

    # with open(hap_file, "w") as f:
    #     for row in df_hap.iter_rows():
    #         f.write("".join(str(value) for value in row) + "\n")

    df_rec_map.write_csv(map_file, include_header=False, separator=" ")

    out_file = f"/tmp/tmp_{_iter}.out"
    cmd_hapbin = f"{ihsbin} --hap {hap_file} --map {map_file} --cutoff {min_ehh} --minmaf {min_maf} --out {out_file}"

    if gap_scale > 0:
        cmd_hapbin += f" -s {gap_scale}"

    if max_extend > 0:
        cmd_hapbin += f" -e {max_extend}"

    if binom:
        cmd_hapbin += " --binom"

    ihs_output = subprocess.run(cmd_hapbin, shell=True, capture_output=True, text=True)

    tmp_ihs = pl.read_csv(
        out_file,
        separator="\t",
        schema=pl.Schema(
            [
                ("location", pl.Int64),
                ("ihh_0", pl.Float64),
                ("ihh_1", pl.Float64),
                ("ihs", pl.Float64),
                ("std_ihs", pl.Float64),
            ]
        ),
    )

    r_pos = dict(zip(rec_map[:, 1], rec_map[:, -1]))
    _p = np.array([r_pos[i] for i in tmp_ihs["location"]])
    df_daf = pl.DataFrame(
        {"positions": rec_map[:, -1].astype(int), "daf": hap.sum(axis=1) / hap.shape[1]}
    )
    df_ihs = (
        tmp_ihs.with_columns(pl.lit(_p).cast(pl.Int64).alias("positions"))
        .join(df_daf, on="positions", coalesce=True)
        .with_columns(
            (pl.col("ihh_1") - pl.col("ihh_0"))
            .abs()
            .cast(pl.Float64)
            .alias("delta_ihh")
        )
        .select(["positions", "daf", "ihs", "delta_ihh"])
    ).fill_nan(None)

    os.remove(hap_file)
    os.remove(map_file)
    os.remove(out_file)

    return df_ihs


def ihs_ihh(
    h,
    pos,
    map_pos=None,
    min_ehh=0.05,
    min_maf=0.05,
    include_edges=False,
    gap_scale=20000,
    max_gap=200000,
    is_accessible=None,
    use_threads=True,
):
    """
    Computes iHS (integrated Haplotype Score) and delta iHH (difference in integrated
    haplotype homozygosity) for a given set of haplotypes and positions.
    delta iHH represents the absolute difference in iHH between the
    derived and ancestral alleles.

    Parameters
    ----------
    h : numpy.ndarray
        A 2D array of haplotypes where each row corresponds to a SNP (variant), and each
        column corresponds to a haplotype for an individual. The entries are expected to
        be binary (0 or 1), representing the ancestral and derived alleles.

    pos : numpy.ndarray
        A 1D array of physical positions corresponding to the SNPs in `h`. The length
        of `pos` should match the number of rows in `h`.

    map_pos : numpy.ndarray or None, optional (default=None)
        A 1D array representing the genetic map positions (in centiMorgans or other genetic distance)
        corresponding to the SNPs. If None, physical positions (`pos`) are used instead to compute
        gaps between SNPs for EHH integration.

    min_ehh : float, optional (default=0.05)
        The minimum EHH value required for integration. EHH values below this threshold are ignored
        when calculating iHH.

    min_maf : float, optional (default=0.05)
        The minimum minor allele frequency (MAF) required for computing iHS. Variants with lower MAF
        are excluded from the analysis.

    include_edges : bool, optional (default=False)
        Whether to include SNPs at the edges of the haplotype array when calculating iHH. If False,
        edge SNPs may be excluded if they don't meet the `min_ehh` threshold.

    gap_scale : int, optional (default=20000)
        The scaling factor for gaps between consecutive SNPs, used when computing iHH over physical
        distances. If `map_pos` is provided, this scaling factor is not used.

    max_gap : int, optional (default=200000)
        The maximum allowed gap between SNPs when integrating EHH. Gaps larger than this are capped
        to `max_gap` to avoid overly large contributions from distant SNPs.

    is_accessible : numpy.ndarray or None, optional (default=None)
        A boolean array of the same length as `pos`, indicating whether each SNP is in a genomic region
        accessible for analysis (e.g., non-repetitive or non-masked regions). If None, all SNPs are
        assumed to be accessible.

    Returns
    -------
    df_ihs : pandas.DataFrame
        A DataFrame containing the following columns:
        - "positions": The physical positions of the SNPs.
        - "daf": The derived allele frequency (DAF) at each SNP.
        - "ihs": The iHS value for each SNP.
        - "delta_ihh": The absolute difference in integrated haplotype homozygosity (iHH) between
          the derived and ancestral alleles at each SNP.

    Notes
    -----
    - The function first computes the iHH (integrated haplotype homozygosity) for both the forward
      and reverse scans of the haplotypes. iHH represents the area under the EHH decay curve, which
      measures the extent of haplotype homozygosity extending from the focal SNP.
    - iHS is calculated as the natural logarithm of the ratio of iHH for the ancestral and derived
      alleles at each SNP.
    - SNPs with missing or invalid iHS values (e.g., due to low MAF) are removed from the output DataFrame.

    Example Workflow:
    - Compute iHH for forward and reverse directions using the haplotype data.
    - Calculate iHS as `log(iHH_derived / iHH_ancestral)`.
    - Calculate delta iHH as the absolute difference between the iHH values for derived and ancestral alleles.

    """
    # check inputs
    h = asarray_ndim(h, 2)
    check_integer_dtype(h)
    pos = asarray_ndim(pos, 1)
    check_dim0_aligned(h, pos)
    h = memoryview_safe(h)
    pos = memoryview_safe(pos)

    # compute gaps between variants for integration
    gaps = compute_ihh_gaps(pos, map_pos, gap_scale, max_gap, is_accessible)

    # setup kwargs
    kwargs = dict(min_ehh=min_ehh, min_maf=min_maf, include_edges=include_edges)

    if use_threads:
        # run with threads

        # create pool
        pool = ThreadPool(2)

        # scan forward
        result_fwd = pool.apply_async(ihh01_scan, (h, gaps), kwargs)

        # scan backward
        result_rev = pool.apply_async(ihh01_scan, (h[::-1], gaps[::-1]), kwargs)

        # wait for both to finish
        pool.close()
        pool.join()

        # obtain results
        ihh0_fwd, ihh1_fwd = result_fwd.get()
        ihh0_rev, ihh1_rev = result_rev.get()

        # cleanup
        pool.terminate()

    else:
        # run without threads

        # scan forward
        ihh0_fwd, ihh1_fwd = ihh01_scan(h, gaps, **kwargs)

        # scan backward
        ihh0_rev, ihh1_rev = ihh01_scan(h[::-1], gaps[::-1], **kwargs)

    # compute unstandardized score
    ihh0 = ihh0_fwd + ihh0_rev
    ihh1 = ihh1_fwd + ihh1_rev

    # og estimation
    mask = (ihh1 != 0) & (ihh0 > 0) & (ihh1 > 0)
    ihs = np.full_like(ihh0, np.nan, dtype=float)
    ihs[mask] = np.log(ihh0[mask] / ihh1[mask])

    ihs = np.log(ihh0 / ihh1)

    delta_ihh = np.abs(ihh1 - ihh0)

    df_ihs = (
        pl.DataFrame(
            {
                "positions": pos,
                "daf": h.sum(axis=1) / h.shape[1],
                "ihs": ihs,
                "delta_ihh": delta_ihh,
            }
        )
        .fill_nan(None)
        .drop_nulls()
    )

    df_ihs = df_ihs.filter(~pl.col("ihs").is_infinite())
    return df_ihs


def haf_top(hap, pos, cutoff=0.1, start=None, stop=None, window_size=None, n_snps=1001):
    """
    Calculates the Haplotype Allele Frequency (HAF) for the top proportion of haplotypes,
    which is a measure used to summarize haplotype diversity. The function computes the
    HAF statistic for a filtered set of variants and returns the sum of the top `cutoff`
    proportion of the HAF values.

    Parameters
    ----------
    hap : numpy.ndarray
        A 2D array where each row represents a SNP (variant), and each column represents
        a haplotype for an individual. The entries are expected to be binary (0 or 1),
        indicating the presence of ancestral or derived alleles.

    pos : numpy.ndarray
        A 1D array of physical positions corresponding to the SNPs in the `hap` matrix.
        The length of `pos` should match the number of rows in `hap`.

    cutoff : float, optional (default=0.1)
        The proportion of HAF values to exclude from the top and bottom when calculating the final HAF score.
        For example, a `cutoff` of 0.1 excludes the lowest 10% and highest 10% of HAF values,
        and the function returns the sum of the remaining HAF values.

    start : float or None, optional (default=None)
        The starting physical position (in base pairs) for the genomic region of interest.
        If provided, only SNPs at or after this position are included in the calculation.

    stop : float or None, optional (default=None)
        The ending physical position (in base pairs) for the genomic region of interest.
        If provided, only SNPs at or before this position are included in the calculation.

    Returns
    -------
    haf_top : float
        The sum of the top `cutoff` proportion of HAF values, which represents the
        higher end of the haplotype diversity distribution within the specified region.

    Notes
    -----
    - The function first filters the SNPs by the specified genomic region (using `start` and `stop`).
    - HAF (Haplotype Allele Frequency) is computed by summing the pairwise dot product of
      haplotypes and dividing by the total number of haplotypes.
    - The HAF values are sorted, and the top proportion (based on the `cutoff`) is returned.

    """
    if start is not None or stop is not None:
        loc = (pos >= start) & (pos <= stop)
        pos = pos[loc]
        hap = hap[loc, :]
    elif window_size is not None:
        loc = (pos >= (6e5 - window_size // 2)) & (pos <= (6e5 + window_size // 2))
        hap = hap[loc, :]
    elif n_snps is not None:
        S, N = hap.shape

        # if (N >= 50 and N < 100):
        #     n_snps = 401
        # elif N < 50:
        #     n_snps = 201

        closer_center_snp = np.argmin(np.abs(pos - 6e5))
        loc = np.arange(
            max(closer_center_snp - n_snps // 2, 0),
            min(closer_center_snp + n_snps // 2 + 1, pos.size),
        )
        hap = hap[loc, :]

    freqs = hap.sum(axis=1) / hap.shape[1]
    hap_tmp = hap[(freqs > 0) & (freqs < 1)]
    haf_num = (np.dot(hap_tmp.T, hap_tmp) / hap.shape[1]).sum(axis=1)
    # haf_num = (jax_dot(hap_tmp.T) / hap.shape[1]).sum(axis=1)
    haf_den = hap_tmp.sum(axis=0)
    # haf = np.sort(haf_num / haf_den)

    if 0 in haf_den:
        mask_zeros = haf_den != 0
        haf = np.full_like(haf_num, np.nan, dtype=np.float64)
        haf[mask_zeros] = haf_num[mask_zeros] / haf_den[mask_zeros]
        haf = np.sort(haf)
    else:
        haf = np.sort(haf_num / haf_den)

    if cutoff <= 0 or cutoff >= 1:
        cutoff = 1
    idx_low = int(cutoff * haf.size)
    idx_high = int((1 - cutoff) * haf.size)

    # 10% higher
    return haf[idx_high:].sum()


# def haf_top_jax(hap, pos, cutoff=0.1, start=None, stop=None, window_size=None):
#     """
#     Calculates the Haplotype Allele Frequency (HAF) for the top proportion of haplotypes,
#     which is a measure used to summarize haplotype diversity. The function computes the
#     HAF statistic for a filtered set of variants and returns the sum of the top `cutoff`
#     proportion of the HAF values.

#     Parameters
#     ----------
#     hap : numpy.ndarray
#         A 2D array where each row represents a SNP (variant), and each column represents
#         a haplotype for an individual. The entries are expected to be binary (0 or 1),
#         indicating the presence of ancestral or derived alleles.

#     pos : numpy.ndarray
#         A 1D array of physical positions corresponding to the SNPs in the `hap` matrix.
#         The length of `pos` should match the number of rows in `hap`.

#     cutoff : float, optional (default=0.1)
#         The proportion of HAF values to exclude from the top and bottom when calculating the final HAF score.
#         For example, a `cutoff` of 0.1 excludes the lowest 10% and highest 10% of HAF values,
#         and the function returns the sum of the remaining HAF values.

#     start : float or None, optional (default=None)
#         The starting physical position (in base pairs) for the genomic region of interest.
#         If provided, only SNPs at or after this position are included in the calculation.

#     stop : float or None, optional (default=None)
#         The ending physical position (in base pairs) for the genomic region of interest.
#         If provided, only SNPs at or before this position are included in the calculation.

#     Returns
#     -------
#     haf_top : float
#         The sum of the top `cutoff` proportion of HAF values, which represents the
#         higher end of the haplotype diversity distribution within the specified region.

#     Notes
#     -----
#     - The function first filters the SNPs by the specified genomic region (using `start` and `stop`).
#     - HAF (Haplotype Allele Frequency) is computed by summing the pairwise dot product of
#       haplotypes and dividing by the total number of haplotypes.
#     - The HAF values are sorted, and the top proportion (based on the `cutoff`) is returned.

#     """
#     if start is not None or stop is not None:
#         loc = (pos >= start) & (pos <= stop)
#         pos = pos[loc]
#         hap = hap[loc, :]
#     elif window_size is not None:
#         loc = (pos >= (6e5 - window_size // 2)) & (pos <= (6e5 + window_size // 2))
#         hap = hap[loc, :]

#     freqs = hap.sum(axis=1) / hap.shape[1]
#     hap_tmp = hap[(freqs > 0) & (freqs < 1)]
#     # haf_num = (np.dot(hap_tmp.T, hap_tmp) / hap.shape[1]).sum(axis=1)
#     haf_num = (jax_dot(hap_tmp.T) / hap.shape[1]).sum(axis=1)
#     haf_den = hap_tmp.sum(axis=0)
#     # haf = np.sort(haf_num / haf_den)

#     if 0 in haf_den:
#         mask_zeros = haf_den != 0
#         haf = np.full_like(haf_num, np.nan, dtype=np.float64)
#         haf[mask_zeros] = haf_num[mask_zeros] / haf_den[mask_zeros]
#         haf = jnp.sort(haf)
#     else:
#         haf = jnp.sort(haf_num / haf_den)

#     if cutoff <= 0 or cutoff >= 1:
#         cutoff = 1
#     idx_low = int(cutoff * haf.size)
#     idx_high = int((1 - cutoff) * haf.size)

#     # 10% higher
#     return haf[idx_high:].sum()


def h12_enard(hap, positions, focal_coord=600000, window_size=1200000):
    coords, haplos, true_coords, count_coords = process_hap_map(hap, positions)

    keep_haplo_freq = {}

    key_001 = focal_coord
    coord = key_001
    int_coord = (coord // 100) * 100
    inf = int_coord - window_size // 2
    sup = int_coord + window_size // 2
    hap_line = "1" * hap.shape[1]
    hap = list(hap_line)

    ongoing_haplos = defaultdict(str)

    for i in range(1, (window_size // 200) + 1):
        inf_i = int_coord - i * 100
        low_bound = inf_i

        if inf_i <= 0:
            break

        if inf_i in coords.keys():
            chain = coords[inf_i]
            splitter_chain = chain.split()
            for true_coord in splitter_chain:
                true_coord = int(true_coord)
                if true_coord != coord:
                    haplotype = haplos[true_coord]
                    current_haplo = list(haplotype)
                    for k, h in enumerate(hap):
                        if h == "1":
                            ongoing_haplos[str(k)] += f"{current_haplo[k]} "

        if i * 100 >= window_size // 2:
            break

    for i in range(1, (window_size // 200) + 1):
        sup_i = int_coord + i * 100
        up_bound = sup_i

        if sup_i >= 1200000:
            break

        if sup_i in coords.keys():
            chain = coords[sup_i]
            splitter_chain = chain.split()
            for true_coord in splitter_chain:
                true_coord = int(true_coord)
                if true_coord != coord:
                    haplotype = haplos[true_coord]
                    current_haplo = list(haplotype)
                    for k, h in enumerate(hap):
                        if h == "1":
                            ongoing_haplos[str(k)] += f"{current_haplo[k]} "

        if i * 100 >= window_size // 2:
            break

    haplos_number = defaultdict(int)
    for key_ongo in sorted(ongoing_haplos.keys()):
        haplo = ongoing_haplos[key_ongo]
        haplos_number[haplo] += 1

    # print("ORIG haplos_number:", dict(haplos_number), "toto:", len(ongoing_haplos))
    best_haplos = {}
    revert_number = defaultdict(str)

    # Populate revert_number dictionary
    for key_numb in sorted(haplos_number.keys()):
        number = haplos_number[key_numb]
        # revert_number[number] += f"{key_numb}"
        revert_number[number] += f"{key_numb}_"

    counter_rev = 0
    done_rev = 0

    # Sort revert_number keys in descending order and process
    for key_rev in sorted(revert_number.keys(), reverse=True):
        chain = revert_number[key_rev].rstrip("_")
        splitter_chain = chain.split("_")
        for f, haplo in enumerate(splitter_chain):
            if haplo:  # Check if the haplo is not empty
                done_rev += 1
                best_haplos[done_rev] = haplo
                keep_haplo_freq[done_rev] = key_rev

        counter_rev += done_rev

        if counter_rev >= 10:
            break

    similar_pairs = defaultdict(str)
    done = {}

    # Ensure best_haplos has string keys
    best_haplos = {str(k): v for k, v in best_haplos.items()}

    # Initialize similar_pairs
    for key_compf in sorted(best_haplos.keys(), key=int):
        similar_pairs[key_compf] = ""

    sorted_keys = sorted(best_haplos.keys(), key=int)  # Sort keys only once

    # Pre-split haplotypes to avoid calling split() multiple times
    # split_haplos = {key: best_haplos[key].split() for key in sorted_keys}
    split_haplos = {
        key: np.array(best_haplos[key].split(), dtype=np.uint8) for key in sorted_keys
    }
    for i, key_comp in enumerate(sorted_keys):
        haplo_1 = split_haplos[key_comp]

        for key_comp2 in sorted_keys:
            # Only compare each pair once (key_comp < key_comp2)
            if key_comp != key_comp2:
                pair_key_1_2 = f"{key_comp} {key_comp2}"

                if pair_key_1_2 not in done:
                    # print(pair_key_1_2)
                    haplo_2 = split_haplos[key_comp2]

                    # Compare the two haplotypes using optimized compare_haplos
                    identical, different, total = compare_haplos_optimized(
                        haplo_1, haplo_2
                    )

                    if total > 0 and different / total <= 0.2:
                        similar_pairs[key_comp] += f"{key_comp2} "
                        done[pair_key_1_2] = "yes"
                        done[f"{key_comp2} {key_comp}"] = "yes"

    #  Extend to all haplotypes
    exclude = {}
    counter_rev2 = 0
    # max_haplo = ""
    # second_haplo = ""
    # third_haplo = ""
    # four_haplo=""
    # five_haplo=""
    pairs_keys = sorted(similar_pairs, key=int)

    haplo_counter = dict(zip(pairs_keys, [""] * len(pairs_keys)))

    for key_rev2 in pairs_keys:
        if key_rev2 not in exclude:
            chain = best_haplos[key_rev2]
            similar = similar_pairs[key_rev2]
            if similar != "":
                splitter_similar = similar.split()
                for cur_rev in splitter_similar:
                    exclude[cur_rev] = "yes"
                    chain += "_" + best_haplos[cur_rev]

            counter_rev2 += 1
            haplo_counter[str(counter_rev2)] = chain
            # if counter_rev2 == 1:
            #     max_haplo = chain
            # elif counter_rev2 == 2:
            #     second_haplo = chain
            # elif counter_rev2 == 3:
            #     third_haplo = chain
            # elif counter_rev2 == 4:
            #     four_haplo = chain
            # elif counter_rev2 == 5:
            #     five_haplo = chain

    freq_counts = {i: 0 for i in haplo_counter}  # {1:0, 2:0, 3:0, …}
    toto = 0

    for key_ongo2 in sorted(ongoing_haplos):
        ongoing = ongoing_haplos[key_ongo2]
        toto += 1

        for idx, chain in haplo_counter.items():
            if ongoing in chain:
                freq_counts[str(idx)] += 1
                break

    # print("ORIG freq_counts:", freq_counts, "toto:", toto)

    # freqs = np.sort([count / toto for i, count in freq_counts.items()])[::-1]
    # Probably bugged code on David's version, not ordered properly
    freqs = np.array([count / toto for i, count in freq_counts.items()])

    # David og's H12
    H12 = np.sum(freqs[:2]) ** 2
    H1 = np.sum(freqs[:] ** 2)
    H2 = H1 - freqs[0] ** 2
    H2_H1 = H2 / H1

    return H12, H2_H1, H2


@njit
def process_hap_map(hap, positions):
    derived_freq = hap.sum(1) / hap.shape[1]
    okfreq_indices = np.where((derived_freq >= 0.05) & (derived_freq <= 1))[0] + 1
    coord = positions[okfreq_indices - 1].astype(np.int64)
    int_coord = (coord // 100) * 100

    # numba Dict with explicit types
    coords = Dict.empty(
        key_type=types.int64,
        value_type=types.unicode_type,
    )
    haplos = Dict.empty(
        key_type=types.int64,
        value_type=types.unicode_type,
    )
    true_coords = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    count_coords = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )

    # Initialize coords with empty strings
    for v in int_coord:
        coords[v] = ""

    for i, v in enumerate(int_coord):
        coord_index = okfreq_indices[i]
        coords[v] += f"{coord[i]} "
        true_coords[coord[i]] = coord_index
        count_coords[coord_index] = coord[i]
        haplos[coord[i]] = "".join(map(str, hap[coord_index - 1]))

    return coords, haplos, true_coords, count_coords


@njit
def compare_haplos_optimized(haplo1, haplo2):
    identical = 0
    different = 0
    for i in range(len(haplo1)):
        h1 = haplo1[i]
        h2 = haplo2[i]

        if (h1 == 1) and (h2 == 1):
            identical += 1
        elif h1 != h2:
            different += 1

    total = identical + different
    return identical, different, total


@njit
def garud_h_numba(h):
    """
    Numba-accelerated Garud’s H-stats.
    h: (n_variants, n_haplotypes) uint8 array of 0/1
    """
    L, n = h.shape

    # 1) rolling uint64 hash to count distinct columns
    counts = Dict.empty(key_type=uint64, value_type=int64)
    for j in range(n):
        hsh = np.uint64(146527)
        for i in range(L):
            hsh = (hsh * np.uint64(1000003)) ^ np.uint64(h[i, j])
        counts[hsh] = counts.get(hsh, 0) + 1

    # 2) collect counts into an array
    m = len(counts)
    cnts = np.empty(m, np.int64)
    idx = 0
    for k in counts:
        cnts[idx] = counts[k]
        idx += 1

    # 3) to frequencies & sort descending
    freqs = cnts.astype(np.float64) / n
    freqs = np.sort(freqs)[::-1]

    # 4) pad top‐3
    p1 = freqs[0] if freqs.size > 0 else 0.0
    p2 = freqs[1] if freqs.size > 1 else 0.0
    p3 = freqs[2] if freqs.size > 2 else 0.0

    # 5) compute H1, H12, H123, H2/H1
    H1 = 0.0
    for i in range(freqs.size):
        H1 += freqs[i] * freqs[i]

    H12 = (p1 + p2) ** 2
    for i in range(2, freqs.size):
        H12 += freqs[i] * freqs[i]

    H123 = (p1 + p2 + p3) ** 2
    for i in range(3, freqs.size):
        H123 += freqs[i] * freqs[i]

    H2 = H1 - p1**2
    H2_H1 = H2 / H1

    return H12, H2_H1, H1, H123


@njit(cache=True)
def _compare_hap_cols(H, col_a, col_b):
    """
    identical = #(1,1)
    different = #(mismatch: 1,0 or 0,1)
    total     = identical + different
    """
    L = H.shape[0]
    identical = 0
    different = 0
    for i in range(L):
        a = H[i, col_a]
        b = H[i, col_b]
        if (a == 1) and (b == 1):
            identical += 1
        elif a != b:
            different += 1
    total = identical + different
    return identical, different, total


@njit(cache=True)
def _legacy_row_order_indices(hap, positions, focal_coord, window_size, min_freq):
    """
    Returns row indices exactly in the order legacy visits sites:
      - 100bp bins: left first (-1,-2,...) then right (+1,+2,...)
      - exclude focal bin (step==0)
      - freq filter in [min_freq, 1.0]
      - right cap: sup_i < 1_200_000
    Implementation is O(#rows_in_window): two passes + bucketize by step.
    """
    L_total, n = hap.shape
    int_coord = (focal_coord // 100) * 100
    half = window_size // 2
    low = int_coord - half
    high = int_coord + half
    max_steps = half // 100

    # pass 1: collect "ok" rows and their step
    # store into temporary arrays
    ok_idx = np.empty(L_total, dtype=int64)
    ok_step = np.empty(L_total, dtype=int64)
    k = 0
    for i in range(L_total):
        pos = positions[i]
        if pos == focal_coord:
            continue
        if (pos < low) or (pos > high):
            continue
        # derived freq
        s = 0
        for j in range(n):
            s += hap[i, j]
        f = s / n
        if (f < min_freq) or (f > 1.0):
            continue
        # step in 100bp units
        bin_i = (pos // 100) * 100
        step = (bin_i - int_coord) // 100
        if step == 0:
            continue  # legacy skips focal bin
        ok_idx[k] = i
        ok_step[k] = step
        k += 1

    if k == 0:
        return np.empty(0, dtype=int64)

    # pass 2: bucketize by step with stability
    # LEFT
    left_limit = (int_coord - 1) // 100  # require inf_i > 0
    left_counts = np.zeros(max_steps, dtype=int64)  # index = abs(step)-1
    # RIGHT (with cap sup_i < 1_200_000)
    right_cap = (1_200_000 - int_coord - 1) // 100
    right_counts = np.zeros(max_steps, dtype=int64)  # index = step-1

    # count per step
    for t in range(k):
        step = ok_step[t]
        if step < 0:
            st = -step
            if (st <= max_steps) and (st <= left_limit):
                left_counts[st - 1] += 1
        else:  # step > 0
            st = step
            if (st <= max_steps) and (st <= right_cap):
                right_counts[st - 1] += 1

    left_total = int(left_counts.sum())
    right_total = int(right_counts.sum())
    out = np.empty(left_total + right_total, dtype=int64)

    # prefix sums to place indices in bin order (preserve ok order within step)
    left_off = np.zeros(max_steps, dtype=int64)
    right_off = np.zeros(max_steps, dtype=int64)
    # compute running offsets
    acc = 0
    for s in range(max_steps):
        left_off[s] = acc
        acc += left_counts[s]
    left_end = acc
    acc = 0
    for s in range(max_steps):
        right_off[s] = acc
        acc += right_counts[s]
    # pass 3: fill left, then right
    # LEFT fill in step order 1..max_steps
    for t in range(k):
        step = ok_step[t]
        if step < 0:
            st = -step
            if (st <= max_steps) and (st <= left_limit):
                pos = left_off[st - 1]
                out[pos] = ok_idx[t]
                left_off[st - 1] = pos + 1

    # RIGHT fill appended after left block, in step order 1..max_steps
    base = left_end
    for t in range(k):
        step = ok_step[t]
        if step > 0:
            st = step
            if (st <= max_steps) and (st <= right_cap):
                pos = base + right_off[st - 1]
                out[pos] = ok_idx[t]
                right_off[st - 1] = right_off[st - 1] + 1

    return out


@njit(cache=True)
def _legacy_row_order_indices(hap, positions, focal_coord, num_snps, min_freq):
    """
    Count-based SNP window centered at `focal_coord`:
      - Select variants passing derived-frequency filter in [min_freq, 1.0]
      - Exclude the focal site itself (pos == focal_coord) from the output
      - Choose ~half from the left (pos < focal) and ~half from the right (pos > focal),
        by nearest genomic distance to the focal coord. If one side lacks enough SNPs,
        fill the remainder from the other side.
      - Return indices ordered as: left (nearest -> farther), then right (nearest -> farther)

    Parameters
    ----------
    hap : uint8[:, :]   (L_total, n) 0/1 matrix
    positions : int64[:]  (L_total,) genomic coords
    focal_coord : int64
    num_snps : int       desired total SNPs in the window (including focal if present)
    min_freq : float64   derived allele frequency lower bound (inclusive)

    Returns
    -------
    out : int64[:]   indices of selected SNP rows, excluding the focal site
                     length = min(num_snps - 1 if focal exists else num_snps, available)
    """
    L_total, n = hap.shape

    # Preallocate arrays for left/right candidates
    left_idx = np.empty(L_total, dtype=int64)
    left_dist = np.empty(L_total, dtype=int64)
    right_idx = np.empty(L_total, dtype=int64)
    right_dist = np.empty(L_total, dtype=int64)

    left_k = 0
    right_k = 0
    has_focal = False

    # Pass 1: frequency filter and split by side relative to focal
    for i in range(L_total):
        pos = positions[i]

        # Derived frequency at row i
        s = 0
        for j in range(n):
            s += hap[i, j]
        f = s / n

        if (f < min_freq) or (f > 1.0):
            continue

        if pos == focal_coord:
            has_focal = True
            continue
        elif pos < focal_coord:
            left_idx[left_k] = i
            left_dist[left_k] = focal_coord - pos  # distance to focal
            left_k += 1
        else:  # pos > focal_coord
            right_idx[right_k] = i
            right_dist[right_k] = pos - focal_coord
            right_k += 1

    available = left_k + right_k
    if available == 0 or num_snps <= 0:
        return np.empty(0, dtype=int64)

    # Target total excludes focal if the focal site exists
    target_total = num_snps - 1 if has_focal else num_snps
    if target_total <= 0:
        return np.empty(0, dtype=int64)

    # Order candidates by proximity (ascending distance)
    if left_k > 0:
        left_ord = np.argsort(left_dist[:left_k])
    else:
        left_ord = np.empty(0, dtype=int64)

    if right_k > 0:
        right_ord = np.argsort(right_dist[:right_k])
    else:
        right_ord = np.empty(0, dtype=int64)

    # Split roughly evenly; fill shortfall from the other side
    left_need = target_total // 2
    right_need = target_total - left_need

    left_take = left_need if left_need < left_k else left_k
    right_take = right_need if right_need < right_k else right_k

    remaining = target_total - (left_take + right_take)
    if remaining > 0:
        # Try to take extra from right
        extra_right = right_k - right_take
        take = remaining if remaining < extra_right else extra_right
        right_take += take
        remaining = target_total - (left_take + right_take)

    if remaining > 0:
        # Take any remaining from left
        extra_left = left_k - left_take
        take = remaining if remaining < extra_left else extra_left
        left_take += take
        remaining = target_total - (left_take + right_take)

    take_total = left_take + right_take
    if take_total == 0:
        return np.empty(0, dtype=int64)

    # Build output: left (nearest→farther), then right (nearest→farther)
    out = np.empty(take_total, dtype=int64)
    w = 0
    for t in range(left_take):
        out[w] = left_idx[left_ord[t]]
        w += 1
    for t in range(right_take):
        out[w] = right_idx[right_ord[t]]
        w += 1

    return out


# ----------------------------------------------------
# 2) unique haplotypes and per-sample assignments
# ----------------------------------------------------
@njit(cache=True)
def _unique_hash_counts_reprs_and_assign(H):
    """
    H: (L, n) uint8
    Returns:
      cnts:   (m,) int64        counts per unique haplotype
      reprj:  (m,) int64        representative column index for each unique hap
      assign: (n,) int64        sample -> uid
    """
    L, n = H.shape
    counts = Dict.empty(key_type=uint64, value_type=int64)
    reprs = Dict.empty(key_type=uint64, value_type=int64)
    hashes = np.empty(n, dtype=uint64)

    for j in range(n):
        hsh = np.uint64(146527)
        for i in range(L):
            hsh = (hsh * np.uint64(1000003)) ^ np.uint64(H[i, j])
        hashes[j] = hsh
        counts[hsh] = counts.get(hsh, 0) + 1
        if hsh not in reprs:
            reprs[hsh] = j

    m = len(counts)
    cnts = np.empty(m, dtype=int64)
    reprj = np.empty(m, dtype=int64)
    key2id = Dict.empty(key_type=uint64, value_type=int64)

    k = 0
    for hsh in counts:
        cnts[k] = counts[hsh]
        reprj[k] = reprs[hsh]
        key2id[hsh] = k
        k += 1

    assign = np.empty(n, dtype=int64)
    for j in range(n):
        assign[j] = key2id[hashes[j]]
    return cnts, reprj, assign


# -------------------------------------------------------------
# 3) helpers for lex ordering on columns of H (no strings)
# -------------------------------------------------------------
@njit(cache=True, inline="always")
def _lex_less_cols(H, reprj, uid_a, uid_b):
    """
    True if col(uid_a) < col(uid_b) lexicographically
    Equivalent to comparing legacy "0 1 1 ..." strings (spaces don't affect order).
    """
    L = H.shape[0]
    ca = reprj[uid_a]
    cb = reprj[uid_b]
    for i in range(L):
        va = H[i, ca]
        vb = H[i, cb]
        if va < vb:
            return True
        elif va > vb:
            return False
    return False  # equal


@njit(cache=True)
def _argsort_by_count_then_lex(cnts, H, reprj):
    """
    Return indices 0..m-1 sorted by (count desc, lex asc on H[:, reprj[uid]]).
    Implemented via two-stage: count sort (argsort desc) then in-place
    segment lex insertion-sort per count tier.
    """
    m = cnts.size
    order = np.argsort(cnts)  # ascending
    # reverse to descending
    for i in range(m // 2):
        tmp = order[i]
        order[i] = order[m - 1 - i]
        order[m - 1 - i] = tmp

    # walk segments of equal count and lex-sort within each
    i = 0
    while i < m:
        c = cnts[order[i]]
        j = i + 1
        while j < m and cnts[order[j]] == c:
            j += 1
        # insertion sort order[i:j] by lex
        k = i + 1
        while k < j:
            x = order[k]
            p = k - 1
            while (p >= i) and _lex_less_cols(H, reprj, x, order[p]):
                order[p + 1] = order[p]
                p -= 1
            order[p + 1] = x
            k += 1
        i = j
    return order


@njit(cache=True)
def h12_enard_numba(
    hap,
    positions,
    focal_coord=600000,
    n_snps=1001,
    # window_size=500_000,
    min_derived_freq=0.05,
    similarity_threshold=0.8,
    top_k=10,
):
    """
    Legacy-exact outputs with much lower overhead.
    Returns: (H12, H2_H1, H2)
    """

    L_total, n = hap.shape

    # Change n_snps dinamically maximizing power based on Zhao et al. 2024
    # if (n >= 50 and n < 100):
    #     n_snps = 401
    # elif n < 50:
    #     n_snps = 201

    # 1) legacy row order
    rows = _legacy_row_order_indices(
        hap,
        positions,
        np.int64(focal_coord),
        np.int64(n_snps),
        # np.int64(window_size),
        np.float64(min_derived_freq),
    )
    if rows.size == 0:
        return 0.0, 0.0, 0.0

    # 2) window matrix in exact order
    L = rows.size
    H = np.empty((L, n), dtype=np.uint8)
    for r in range(L):
        i = rows[r]
        for j in range(n):
            H[r, j] = hap[i, j]

    # 3) unique haplotypes + per-sample assignment
    cnts, reprj, assign = _unique_hash_counts_reprs_and_assign(H)
    m = cnts.size
    if m == 0:
        return 0.0, 0.0, 0.0

    # 4) global order by (count desc, lex asc) and apply legacy accumulator
    order = _argsort_by_count_then_lex(cnts, H, reprj)  # length m
    # accumulator quirk
    done_rev = 0
    counter_rev = 0
    # we don't know K a priori due to accumulator → collect into temporary array
    best_mask = np.zeros(m, dtype=np.uint8)
    # iterate by count tiers (segments of same count)
    i = 0
    while i < m:
        c = cnts[order[i]]
        j = i + 1
        while (j < m) and (cnts[order[j]] == c):
            j += 1
        # add whole tier (i..j-1) in lex order
        for t in range(i, j):
            best_mask[order[t]] = 1
            done_rev += 1
        counter_rev += done_rev
        if counter_rev >= top_k:
            break
        i = j

    # collect selected uids in the *selection order* induced by `order`
    keep_m = 0
    for t in range(order.size):
        u = order[t]
        if best_mask[u] == 1:
            keep_m += 1
    sel = np.empty(keep_m, dtype=int64)
    w = 0
    for t in range(order.size):
        u = order[t]
        if best_mask[u] == 1:
            sel[w] = u
            w += 1

    # 5) forward-only similarity on selected; exclude-as-you-go groups
    # group_of[a] = group id; insertion order defines p1, p2,...
    group_of = np.full(keep_m, -1, dtype=int64)
    groups = 0
    thr = similarity_threshold
    for ai in range(keep_m):
        if group_of[ai] != -1:
            continue
        g = groups
        groups += 1
        group_of[ai] = g
        ua_col = reprj[sel[ai]]
        for bi in range(ai + 1, keep_m):
            if group_of[bi] != -1:
                continue
            ub_col = reprj[sel[bi]]
            ident, diff, tot = _compare_hap_cols(H, ua_col, ub_col)
            if tot == 0:
                continue
            if (ident / tot) >= thr:
                group_of[bi] = g

    # 6) uid -> group id map (only for selected)
    uid_to_group = np.full(m, -1, dtype=int64)
    for ai in range(keep_m):
        uid_to_group[sel[ai]] = group_of[ai]

    # 7) count only selected uids; denominator is n
    if groups == 0:
        return 0.0, 0.0, 0.0
    freq_counts = np.zeros(groups, dtype=int64)
    for j in range(n):
        uid = assign[j]
        g = uid_to_group[uid]
        if g != -1:
            freq_counts[g] += 1

    # 8) legacy stats (NO resort)
    toto = float(n)
    # insertion order is 0..groups-1 by construction
    H1 = 0.0
    p1 = freq_counts[0] / toto if groups > 0 else 0.0
    p2 = freq_counts[1] / toto if groups > 1 else 0.0
    # accumulate H1
    for g in range(groups):
        f = freq_counts[g] / toto
        H1 += f * f
    H12 = (p1 + p2) * (p1 + p2)
    H2 = H1 - p1 * p1
    H2_H1 = (H2 / H1) if H1 != 0.0 else 0.0
    return H12, H2_H1, H2


################## FS stats


@njit(parallel=True)
def fast_sq_freq_pairs_old(
    hap,
    ac,
    rec_map,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
    genetic_distance=False,
):
    # Precompute
    hap_derived = hap
    hap_ancestral = np.bitwise_xor(hap_derived, 1)
    derived_count = ac[:, 1]
    ancestral_count = ac[:, 0]
    total_count = ac.sum(axis=1)
    freqs = derived_count / total_count

    if genetic_distance:
        rec_pos = rec_map[:, -2]
    else:
        rec_pos = rec_map[:, -1]

    # Focal SNPs
    focal_filter = (freqs >= min_focal_freq) & (freqs <= max_focal_freq)
    focal_index = np.where(focal_filter)[0]
    focal_derived = hap_derived[focal_filter]
    focal_ancestral = hap_ancestral[focal_filter]
    focal_derived_count = derived_count[focal_filter]
    focal_ancestral_count = ancestral_count[focal_filter]

    n_focal = len(focal_index)
    n_samples = hap.shape[1]

    # Outputs
    sq_out_list = [np.empty((1, 3), dtype=np.float64) for _ in range(n_focal)]
    info = np.zeros((n_focal, 4), dtype=np.float64)

    # Precompute window bounds
    half_window = window_size / 2
    window_bounds = np.zeros((len(rec_pos), 2), dtype=np.int64)
    for i in range(len(rec_pos)):
        center = rec_pos[i]
        window_bounds[i, 0] = np.searchsorted(
            rec_pos, center - half_window, side="left"
        )
        window_bounds[i, 1] = (
            np.searchsorted(rec_pos, center + half_window, side="right") - 1
        )

    for j in prange(n_focal):
        i = focal_index[j]
        center_pos = rec_pos[i]
        x_l, y_l = window_bounds[i, 0], i - 1
        x_r, y_r = i + 1, window_bounds[i, 1]

        len_l = max(0, y_l - x_l + 1)
        len_r = max(0, y_r - x_r + 1)
        total_len = len_l + len_r
        out = np.empty((total_len, 3), dtype=np.float64)

        k_out = 0  # output row counter

        # Left
        for k in range(x_l, y_l + 1):
            overlap_d = 0
            overlap_a = 0
            for m in range(n_samples):
                if focal_derived[j, m] == 1 and hap_derived[k, m] == 1:
                    overlap_d += 1
                if focal_ancestral[j, m] == 1 and hap_derived[k, m] == 1:
                    overlap_a += 1

            f_d = (
                float64(overlap_d) / float64(focal_derived_count[j])
                if focal_derived_count[j] > 0
                else 0.0
            )
            f_a = (
                float64(overlap_a) / float64(focal_ancestral_count[j])
                if focal_ancestral_count[j] > 0
                else 0.0
            )
            total = float64(freqs[k])

            out[len_l - (k - x_l) - 1, 0] = f_d
            out[len_l - (k - x_l) - 1, 1] = f_a
            out[len_l - (k - x_l) - 1, 2] = total
            k_out += 1

        # Right
        for k in range(x_r, y_r + 1):
            overlap_d = 0
            overlap_a = 0
            for m in range(n_samples):
                if focal_derived[j, m] == 1 and hap_derived[k, m] == 1:
                    overlap_d += 1
                if focal_ancestral[j, m] == 1 and hap_derived[k, m] == 1:
                    overlap_a += 1

            f_d = (
                float64(overlap_d) / float64(focal_derived_count[j])
                if focal_derived_count[j] > 0
                else 0.0
            )
            f_a = (
                float64(overlap_a) / float64(focal_ancestral_count[j])
                if focal_ancestral_count[j] > 0
                else 0.0
            )
            total = float64(freqs[k])

            out[k_out, 0] = f_d
            out[k_out, 1] = f_a
            out[k_out, 2] = total
            k_out += 1

        sq_out_list[j] = out[:k_out]  # trim unused if any
        info[j] = np.array(
            [
                rec_map[i, -1],
                freqs[i],
                float64(focal_derived_count[j]),
                float64(focal_ancestral_count[j]),
            ]
        )

    return sq_out_list, info


@njit(parallel=True)
def fast_sq_freq_pairs_noindex(
    hap,
    ac,
    rec_map,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=300000,
    genetic_distance=False,
):
    hap_derived = hap
    hap_ancestral = np.bitwise_xor(hap_derived, 1)

    derived_count = ac[:, 1]
    ancestral_count = ac[:, 0]
    total_count = ac.sum(axis=1)
    freqs = derived_count / total_count

    if genetic_distance:
        rec_pos = rec_map[:, -2]
    else:
        rec_pos = rec_map[:, -1]

    focal_filter = (freqs >= min_focal_freq) & (freqs <= max_focal_freq)
    focal_index = np.where(focal_filter)[0]
    focal_derived = hap_derived[focal_filter]
    focal_ancestral = hap_ancestral[focal_filter]
    focal_derived_count = derived_count[focal_filter]
    focal_ancestral_count = ancestral_count[focal_filter]

    n_focal = len(focal_index)
    n_samples = hap.shape[1]
    n_snps = hap.shape[0]

    half_window = window_size / 2
    window_bounds = np.zeros((n_snps, 2), dtype=np.int64)
    for i in range(n_snps):
        center = rec_pos[i]
        window_bounds[i, 0] = np.searchsorted(
            rec_pos, center - half_window, side="left"
        )
        window_bounds[i, 1] = (
            np.searchsorted(rec_pos, center + half_window, side="right") - 1
        )

    sq_out_list = [np.empty((1, 3), dtype=np.float64) for _ in range(n_focal)]
    snp_indices_list = [np.empty(1, dtype=np.int64) for _ in range(n_focal)]
    info = np.zeros((n_focal, 2), dtype=np.float64)

    for j in prange(n_focal):
        i = focal_index[j]
        x_l, y_l = window_bounds[i, 0], i - 1
        x_r, y_r = i + 1, window_bounds[i, 1]

        len_l = max(0, y_l - x_l + 1)
        len_r = max(0, y_r - x_r + 1)
        total_len = len_l + len_r
        out = np.empty((total_len, 3), dtype=np.float64)
        indices_out = np.empty(total_len, dtype=np.int64)
        k_out = 0

        # LEFT window (reversed)
        for k in range(x_l, y_l + 1):
            overlap_d = 0
            overlap_a = 0
            for m in range(n_samples):
                overlap_d += focal_derived[j, m] * hap_derived[k, m]
                overlap_a += focal_ancestral[j, m] * hap_derived[k, m]

            f_d = (
                overlap_d / focal_derived_count[j]
                if focal_derived_count[j] > 0
                else 0.0
            )
            f_a = (
                overlap_a / focal_ancestral_count[j]
                if focal_ancestral_count[j] > 0
                else 0.0
            )

            out[len_l - (k - x_l) - 1, 0] = f_d
            out[len_l - (k - x_l) - 1, 1] = f_a
            out[len_l - (k - x_l) - 1, 2] = freqs[k]
            indices_out[len_l - (k - x_l) - 1] = k
            k_out += 1

        # RIGHT window (forward)
        for k in range(x_r, y_r + 1):
            overlap_d = 0
            overlap_a = 0
            for m in range(n_samples):
                overlap_d += focal_derived[j, m] * hap_derived[k, m]
                overlap_a += focal_ancestral[j, m] * hap_derived[k, m]

            f_d = (
                overlap_d / focal_derived_count[j]
                if focal_derived_count[j] > 0
                else 0.0
            )
            f_a = (
                overlap_a / focal_ancestral_count[j]
                if focal_ancestral_count[j] > 0
                else 0.0
            )

            out[k_out, 0] = f_d
            out[k_out, 1] = f_a
            out[k_out, 2] = freqs[k]
            indices_out[k_out] = k
            k_out += 1

        sq_out_list[j] = out[:k_out]
        snp_indices_list[j] = np.sort(indices_out[:k_out])
        info[j, 0] = rec_map[i, -1]
        info[j, 1] = freqs[i]

    return sq_out_list, info, snp_indices_list


@njit(parallel=True)
def fast_sq_freq_pairs(
    hap,
    ac,
    rec_map,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=300000,
    genetic_distance=False,
):
    """Optimized version that also returns SNP indices"""

    # Direct calculations - no intermediate arrays
    n_snps, n_samples = hap.shape
    derived_count = ac[:, 1]
    ancestral_count = ac[:, 0]
    total_count = derived_count + ancestral_count

    # Pre-allocate frequency array once
    freqs = np.empty(n_snps, dtype=np.float64)
    for i in range(n_snps):
        freqs[i] = derived_count[i] / total_count[i]

    # Get positions directly
    rec_pos = rec_map[:, -2] if genetic_distance else rec_map[:, -1]
    half_window = window_size * 0.5

    # Find focal SNPs without creating boolean mask
    focal_indices = np.empty(n_snps, dtype=np.int64)
    n_focal = 0
    for i in range(n_snps):
        if min_focal_freq <= freqs[i] <= max_focal_freq:
            focal_indices[n_focal] = i
            n_focal += 1

    focal_indices = focal_indices[:n_focal]

    # Pre-compute window bounds only for focal SNPs
    window_bounds = np.empty((n_focal, 2), dtype=np.int64)
    for j in range(n_focal):
        i = focal_indices[j]
        center = rec_pos[i]
        window_bounds[j, 0] = np.searchsorted(
            rec_pos, center - half_window, side="left"
        )
        window_bounds[j, 1] = (
            np.searchsorted(rec_pos, center + half_window, side="right") - 1
        )

    # Initialize output lists
    sq_out_list = [np.empty((1, 3), dtype=np.float64) for _ in range(n_focal)]
    snp_indices_list = [np.empty(1, dtype=np.int64) for _ in range(n_focal)]
    info = np.zeros((n_focal, 4), dtype=np.float64)

    # Main processing loop
    for j in prange(n_focal):
        focal_idx = focal_indices[j]

        # Get window bounds
        x_l, y_r = window_bounds[j, 0], window_bounds[j, 1]
        y_l = focal_idx - 1
        x_r = focal_idx + 1

        # Calculate lengths
        len_l = max(0, y_l - x_l + 1)
        len_r = max(0, y_r - x_r + 1)
        total_len = len_l + len_r

        if total_len == 0:
            sq_out_list[j] = np.empty((0, 3), dtype=np.float64)
            snp_indices_list[j] = np.empty(0, dtype=np.int64)
            info[j, 0] = rec_map[focal_idx, -1]
            info[j, 1] = freqs[focal_idx]
            continue

        # Allocate output arrays
        out = np.empty((total_len, 3), dtype=np.float64)
        indices_out = np.empty(total_len, dtype=np.int64)

        # Get focal SNP counts once
        focal_d_count = derived_count[focal_idx]
        focal_a_count = ancestral_count[focal_idx]

        # Process LEFT window (reversed order)
        out_idx = 0
        for k in range(y_l, x_l - 1, -1):  # Reverse iteration
            # Compute overlaps
            overlap_d = 0
            overlap_a = 0
            for m in range(n_samples):
                hap_focal = hap[focal_idx, m]
                hap_k = hap[k, m]
                overlap_d += hap_focal * hap_k
                overlap_a += (1 - hap_focal) * hap_k

            # Compute frequencies
            f_d = overlap_d / focal_d_count if focal_d_count > 0 else 0.0
            f_a = overlap_a / focal_a_count if focal_a_count > 0 else 0.0

            out[out_idx, 0] = f_d
            out[out_idx, 1] = f_a
            out[out_idx, 2] = freqs[k]
            indices_out[out_idx] = k
            out_idx += 1

        # Process RIGHT window (forward order)
        for k in range(x_r, y_r + 1):
            # Compute overlaps
            overlap_d = 0
            overlap_a = 0
            for m in range(n_samples):
                hap_focal = hap[focal_idx, m]
                hap_k = hap[k, m]
                overlap_d += hap_focal * hap_k
                overlap_a += (1 - hap_focal) * hap_k

            # Compute frequencies
            f_d = overlap_d / focal_d_count if focal_d_count > 0 else 0.0
            f_a = overlap_a / focal_a_count if focal_a_count > 0 else 0.0

            out[out_idx, 0] = f_d
            out[out_idx, 1] = f_a
            out[out_idx, 2] = freqs[k]
            indices_out[out_idx] = k
            out_idx += 1

        sq_out_list[j] = out
        snp_indices_list[j] = indices_out
        # info[j, 0] = rec_map[focal_idx, -1]
        # info[j, 1] = freqs[focal_idx]
        # info[j] = freqs[focal_idx]
        info[j] = np.array(
            [
                rec_map[focal_idx, -1],
                freqs[focal_idx],
                float64(focal_d_count),
                float64(focal_a_count),
            ]
        )
    return sq_out_list, info, snp_indices_list


def s_ratio(
    hap,
    ac,
    rec_map,
    max_ancest_freq=1,
    min_tot_freq=0,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=300000,
    genetic_distance=False,
):
    sq_freqs, info, snps_indices = fast_sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size, genetic_distance
    )

    n_rows = len(sq_freqs)
    results = np.empty((n_rows, 1), dtype=np.float64)

    for i, v in enumerate(sq_freqs):
        f_d = v[:, 0]
        f_a = v[:, 1]

        f_d2 = np.zeros(f_d.shape)
        f_a2 = np.zeros(f_a.shape)

        f_d2[(f_d > 0.0000001) & (f_d < 1)] = 1
        f_a2[(f_a > 0.0000001) & (f_a < 1)] = 1

        num = (f_d2 - f_d2 + f_a2 + 1).sum()
        den = (f_a2 - f_a2 + f_d2 + 1).sum()
        # redefine to add one to get rid of blowup issue introduced by adding 0.001 to denominator

        s_ratio_v = num / den
        # s_ratio_v_flip = den / num
        # results.append((s_ratio_v, s_ratio_v_flip))
        results[i] = s_ratio_v

    tmp_schema = {
        "positions": pl.Int64,
        "daf": pl.Float64,
        "s_ratio": pl.Float64,
        # "s_ratio_flip": pl.Float64,
    }

    try:
        out = np.hstack([info[:, :2], results])
        # out = np.hstack([info[:,:2], np.array(results)])
        df_out = pl.DataFrame(out, schema=tmp_schema)
        # df_out = pl.DataFrame([out[:, 0], out[:, 1], out[:, 4], out[:, 5]], schema=tmp_schema)
    except:
        df_out = pl.DataFrame([[], [], []], schema=tmp_schema)

    return df_out


def hapdaf_o(
    hap,
    ac,
    rec_map,
    max_ancest_freq=0.25,
    min_tot_freq=0.25,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=300000,
    genetic_distance=False,
):
    sq_freqs, info, snps_indices = fast_sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size, genetic_distance
    )

    n_rows = len(sq_freqs)
    results = np.empty((n_rows, 1), dtype=np.float64)

    nan_index = []

    for i, v in enumerate(sq_freqs):
        f_d = v[:, 0]
        f_a = v[:, 1]
        f_tot = v[:, 2]

        f_d2 = (
            f_d[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )
        f_a2 = (
            f_a[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )

        if len(f_d2) != 0 and len(f_a2) != 0:
            hapdaf = (f_d2 - f_a2).sum() / f_d2.shape[0]
        else:
            hapdaf = np.nan

        # # Flipping derived to ancestral, ancestral to derived
        # f_d2f = (
        #     f_a[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # f_a2f = (
        #     f_d[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # if len(f_d2f) != 0 and len(f_a2f) != 0:
        #     hapdaf_flip = (f_d2f - f_a2f).sum() / f_d2f.shape[0]
        # else:
        #     hapdaf_flip = np.nan

        # results.append((hapdaf, hapdaf_flip))
        results[i] = hapdaf

    tmp_schema = {
        "positions": pl.Int64,
        "daf": pl.Float64,
        "hapdaf_o": pl.Float64,
        # "hapdaf_o_flip": pl.Float64,
    }

    try:
        out = np.hstack(
            [
                info[:, :2],
                # np.array(results),
                results,
            ]
        )
        df_out = pl.DataFrame(out, schema=tmp_schema)
        # df_out = pl.DataFrame([out[:, 0], out[:, 1], out[:, 4], out[:, 5]], schema=tmp_schema)
    except:
        df_out = pl.DataFrame([[], [], []], schema=tmp_schema)

    return df_out


def hapdaf_s(
    hap,
    ac,
    rec_map,
    max_ancest_freq=0.1,
    min_tot_freq=0.1,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=300000,
    genetic_distance=False,
):
    sq_freqs, info, snps_indices = fast_sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size, genetic_distance
    )

    n_rows = len(sq_freqs)
    results = np.empty((n_rows, 1), dtype=np.float64)
    nan_index = []
    for i, v in enumerate(sq_freqs):
        f_d = v[:, 0]
        f_a = v[:, 1]
        f_tot = v[:, 2]

        f_d2 = (
            f_d[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )
        f_a2 = (
            f_a[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )

        if len(f_d2) != 0 and len(f_a2) != 0:
            hapdaf = (f_d2 - f_a2).sum() / f_d2.shape[0]
        else:
            hapdaf = np.nan

        # # Flipping derived to ancestral, ancestral to derived
        # f_d2f = (
        #     f_a[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # f_a2f = (
        #     f_d[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # if len(f_d2f) != 0 and len(f_a2f) != 0:
        #     hapdaf_flip = (f_d2f - f_a2f).sum() / f_d2f.shape[0]
        # else:
        #     hapdaf_flip = np.nan

        # results.append((hapdaf, hapdaf_flip))

        results[i] = hapdaf

    tmp_schema = {
        "positions": pl.Int64,
        "daf": pl.Float64,
        "hapdaf_s": pl.Float64,
        # "hapdaf_s_flip": pl.Float64,
    }

    try:
        out = np.hstack(
            [
                info[:, :2],
                # np.array(results),
                results,
            ]
        )
        df_out = pl.DataFrame(out, schema=tmp_schema)

        # df_out = pl.DataFrame([out[:, 0], out[:, 1], out[:, 4], out[:, 5]], schema=tmp_schema)
    except:
        df_out = pl.DataFrame([[], [], [], []], schema=tmp_schema)

    return df_out


def dind_high_low(
    hap,
    ac,
    rec_map,
    max_ancest_freq=0.25,
    min_tot_freq=0,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=300000,
    genetic_distance=False,
):
    # Extract frequency pairs and info array
    sq_freqs, info, snps_indices = fast_sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size, genetic_distance
    )

    focal_counts = info[:, 2:]

    # Pre-allocate arrays for results to avoid growing lists
    n_rows = len(sq_freqs)
    results_dind = np.empty((n_rows, 1), dtype=np.float64)
    results_high = np.empty((n_rows, 1), dtype=np.float64)
    results_low = np.empty((n_rows, 1), dtype=np.float64)

    # Main computation loop
    for i, v in enumerate(sq_freqs):
        f_d = v[:, 0]
        f_a = v[:, 1]

        focal_derived_count = focal_counts[i][0]
        focal_ancestral_count = focal_counts[i][1]

        # Calculate derived and ancestral components with in-place operations
        f_d2 = f_d * (1 - f_d) * focal_derived_count / (focal_derived_count - 1)
        f_a2 = (
            f_a
            * (1 - f_a)
            * focal_ancestral_count
            / (focal_ancestral_count - 1 + 0.001)
        )

        # Calculate dind values
        num = (f_d2 - f_d2 + f_a2).sum()
        den = (f_a2 - f_a2 + f_d2).sum() + 0.001
        dind_v = num / den if not np.isinf(num / den) else np.nan
        dind_v_flip = den / num if not np.isinf(den / num) else np.nan

        # results_dind[i] = [dind_v, dind_v_flip]
        results_dind[i] = dind_v

        # Calculate high and low frequency values
        hf_v = (f_d[f_d > max_ancest_freq] ** 2).sum() / max(
            len(f_d[f_d > max_ancest_freq]), 1
        )
        # hf_v_flip = (f_a[f_a > max_ancest_freq] ** 2).sum() / max(
        #     len(f_a[f_a > max_ancest_freq]), 1
        # )
        # results_high[i] = [hf_v, hf_v_flip]
        results_high[i] = hf_v

        lf_v = ((1 - f_d[f_d < max_ancest_freq]) ** 2).sum() / max(
            len(f_d[f_d < max_ancest_freq]), 1
        )

        # lf_v_flip = ((1 - f_a[f_a < max_ancest_freq]) ** 2).sum() / max(
        #     len(f_a[f_a < max_ancest_freq]), 1
        # )
        # results_low[i] = [lf_v, lf_v_flip]
        results_low[i] = lf_v

        # Free memory explicitly for large arrays
        del f_d, f_a, f_d2, f_a2

    tmp_schema = {
        "positions": pl.Int64,
        "daf": pl.Float64,
        "dind": pl.Float64,
        # "dind_flip": pl.Float64,
        "high_freq": pl.Float64,
        # "high_freq_flip": pl.Float64,
        "low_freq": pl.Float64,
        # "low_freq_flip": pl.Float64,
    }

    # Final DataFrame creation
    try:
        out = np.hstack([info[:, :2], results_dind, results_high, results_low])
        df_out = pl.DataFrame(out, schema=tmp_schema)
        # df_out = pl.DataFrame([out[:, 0],out[:, 1],out[:, 4],out[:, 5],out[:, 6],out[:, 7],out[:, 8],out[:, 9],],schema=tmp_schema,)

    except:
        df_out = pl.DataFrame([[], [], [], [], [], [], [], []], schema=tmp_schema)

    return df_out


def run_fs_stats(
    hap,
    ac,
    rec_map,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
    genetic_distance=False,
):
    sq_freqs, info, snps_indices = fast_sq_freq_pairs(
        hap,
        ac,
        rec_map,
        min_focal_freq=min_focal_freq,
        max_focal_freq=max_focal_freq,
        window_size=window_size,
        genetic_distance=genetic_distance,
    )

    if info.shape[0] == 0:
        return fs_stats_dataframe(info, [], [], [], [], [], [])

    results_dind, results_high, results_low = dind_high_low_from_pairs(sq_freqs, info)
    results_s_ratio = s_ratio_from_pairs(sq_freqs)
    results_hapdaf_o = hapdaf_o_from_pairs(hap, sq_freqs, snps_indices)
    results_hapdaf_s = hapdaf_s_from_pairs(sq_freqs)

    df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s = fs_stats_dataframe(
        info,
        results_dind,
        results_high,
        results_low,
        results_s_ratio,
        results_hapdaf_o,
        results_hapdaf_s,
    )

    return df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s


# def s_ratio_from_pairs(sq_freqs, max_ancest_freq=1, min_tot_freq=0):
@njit(parallel=True)
def s_ratio_from_pairs(sq_freqs, max_ancest_freq=1, min_tot_freq=0):
    n_rows = len(sq_freqs)
    results = np.zeros((n_rows, 1))

    for i in prange(len(sq_freqs)):
        f_d = sq_freqs[i][:, 0]
        f_a = sq_freqs[i][:, 1]

        f_d2 = np.zeros(f_d.shape)
        f_a2 = np.zeros(f_a.shape)

        f_d2[(f_d > 0.0000001) & (f_d < 1)] = 1
        f_a2[(f_a > 0.0000001) & (f_a < 1)] = 1

        num = (f_d2 - f_d2 + f_a2 + 1).sum()
        den = (f_a2 - f_a2 + f_d2 + 1).sum()
        # redefine to add one to get rid of blowup issue introduced by adding 0.001 to denominator

        # Add error checking before division
        if den == 0:
            s_ratio_v = np.nan
        else:
            s_ratio_v = num / den

        # if num == 0:
        #     s_ratio_v_flip = np.nan
        # else:
        #     s_ratio_v_flip = den / num
        # s_ratio_v = num / den
        # s_ratio_v_flip = den / num
        # results[i] = s_ratio_v, s_ratio_v_flip
        results[i] = s_ratio_v

    return results


# def hapdaf_o_from_pairs(sq_freqs, max_ancest_freq=0.35, min_tot_freq=0.25):
@njit(parallel=True)
def hapdaf_o_from_pairs(
    hap, sq_freqs, snps_indices, max_ancest_freq=0.25, min_tot_freq=0.25
):
    n_rows = len(sq_freqs)
    results = np.zeros((n_rows, 1))
    # results = np.zeros((n_rows, 2))

    for i in prange(len(sq_freqs)):
        f_d = sq_freqs[i][:, 0]
        f_a = sq_freqs[i][:, 1]
        f_tot = sq_freqs[i][:, 2]

        f_d2 = (
            f_d[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )
        f_a2 = (
            f_a[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )

        if len(f_d2) != 0 and len(f_a2) != 0:
            hapdaf = (f_d2 - f_a2).sum() / f_d2.shape[0]
        else:
            hapdaf = np.nan

        # # Flipping derived to ancestral, ancestral to derived
        # f_d2f = (
        #     f_a[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # f_a2f = (
        #     f_d[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # if len(f_d2f) != 0 and len(f_a2f) != 0:
        #     hapdaf_flip = (f_d2f - f_a2f).sum() / f_d2f.shape[0]
        # else:
        #     hapdaf_flip = np.nan

        # # Derived haplotype LD (omega_deriv)
        # omega_mask_deriv = snps_indices[i][
        #     (f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)
        # ]
        # omega_mask_ancest = snps_indices[i][
        #     (f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)
        # ]
        # # need ≥3 SNPs
        # if omega_mask_deriv.shape[0] >= 3:
        #     _r2_deriv = compute_r2_matrix_upper(hap[omega_mask_deriv])
        #     omega_deriv = omega_linear_correct_mask(_r2_deriv)
        # else:
        #     omega_deriv = np.nan

        # # # Ancestral haplotype LD (omega_ancest)
        # if omega_mask_ancest.shape[0] >= 3:
        #     _r2_ancest = compute_r2_matrix_upper(hap[omega_mask_ancest])
        #     omega_ancest = omega_linear_correct_mask(_r2_ancest)
        # else:
        #     omega_ancest = np.nan

        # if not np.isnan(omega_deriv) and not np.isnan(omega_ancest):
        #     omega_diff = omega_deriv - omega_ancest
        # else:
        #     omega_diff = np.nan

        # _r2 = compute_r2_matrix_upper(hap_int[omega_mask])
        # _zns,_omega = Ld(_r2)

        results[i] = hapdaf
        # results[i] = (hapdaf, hapdaf_flip)
        # results[i] = (hapdaf, omega_diff)
    return results


@njit(parallel=True)
def hapdaf_s_from_pairs(sq_freqs, max_ancest_freq=0.1, min_tot_freq=0.1):
    n_rows = len(sq_freqs)
    results = np.zeros((n_rows, 1))
    # results = np.zeros((n_rows, 2))

    for i in prange(len(sq_freqs)):
        f_d = sq_freqs[i][:, 0]
        f_a = sq_freqs[i][:, 1]
        f_tot = sq_freqs[i][:, 2]

        f_d2 = (
            f_d[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )
        f_a2 = (
            f_a[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )

        if len(f_d2) != 0 and len(f_a2) != 0:
            hapdaf = (f_d2 - f_a2).sum() / f_d2.shape[0]
        else:
            hapdaf = np.nan

        # # Flipping derived to ancestral, ancestral to derived
        # f_d2f = (
        #     f_a[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # f_a2f = (
        #     f_d[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # if len(f_d2f) != 0 and len(f_a2f) != 0:
        #     hapdaf_flip = (f_d2f - f_a2f).sum() / f_d2f.shape[0]
        # else:
        #     hapdaf_flip = np.nan

        # results[i] = [hapdaf, hapdaf_flip]
        results[i] = hapdaf

    return results


# def dind_high_low_from_pairs(sq_freqs, info, max_ancest_freq=0.25, min_tot_freq=0.1):
@njit(parallel=True)
def dind_high_low_from_pairs(sq_freqs, info, max_ancest_freq=0.25, min_tot_freq=0):
    # Pre-allocate arrays for results to avoid growing lists
    n_rows = len(sq_freqs)
    results_dind = np.zeros((n_rows, 1))
    results_high = np.zeros((n_rows, 1))
    results_low = np.zeros((n_rows, 1))
    # results_dind = np.zeros((n_rows, 2))
    # results_high = np.zeros((n_rows, 2))
    # results_low = np.zeros((n_rows, 2))

    # Main computation loop
    for i in prange(len(sq_freqs)):
        f_d = sq_freqs[i][:, 0]
        f_a = sq_freqs[i][:, 1]
        f_tot = sq_freqs[i][:, 2]

        focal_derived_count = info[i][-2]
        focal_ancestral_count = info[i][-1]

        # Calculate derived and ancestral components
        f_d2 = f_d * (1 - f_d) * focal_derived_count / (focal_derived_count - 1)
        f_a2 = (
            f_a
            * (1 - f_a)
            * focal_ancestral_count
            / (focal_ancestral_count - 1 + 0.001)
        )

        # Calculate dind values
        num = (f_d2 - f_d2 + f_a2).sum()
        den = (f_a2 - f_a2 + f_d2).sum() + 0.001
        if den != 0.0:
            dind_v = num / den
        else:
            dind_v = np.nan

        # if num != 0.0:
        #     dind_v_flip = den / num
        # else:
        #     dind_v_flip = np.nan

        # results_dind[i] = [dind_v, dind_v_flip]
        results_dind[i] = dind_v

        # Calculate high and low frequency values
        fd_h_mask = (f_d > max_ancest_freq) & (f_tot >= min_tot_freq)
        fa_h_mask = (f_a > max_ancest_freq) & (f_tot >= min_tot_freq)
        fd_l_mask = (f_d < max_ancest_freq) & (f_tot >= min_tot_freq)
        fa_l_mask = (f_a < max_ancest_freq) & (f_tot >= min_tot_freq)

        fd_l_mask = ((f_d > max_ancest_freq) & (f_d < 0.8)) & (f_tot >= min_tot_freq)
        fd_l_mask = (f_d > max_ancest_freq) & (f_tot >= min_tot_freq)
        # fa_l_mask = ((f_a > 0.25) & (f_a < 0.8)) & (f_tot >= min_tot_freq)

        hf_v = (f_d[fd_h_mask] ** 2).sum() / max(len(f_d[fd_h_mask]), 1)
        # hf_v_flip = (f_a[fa_h_mask] ** 2).sum() / max(len(f_a[fa_h_mask]), 1)
        # results_high[i] = [hf_v, hf_v_flip]
        results_high[i] = hf_v

        lf_v = ((1 - f_d[fd_l_mask]) ** 2).sum() / max(len(f_d[fd_l_mask]), 1)
        # lf_v_flip = ((1 - f_a[fa_l_mask]) ** 2).sum() / max(len(f_a[fa_l_mask]), 1)
        # results_low[i] = [lf_v, lf_v_flip]
        results_low[i] = lf_v

    return results_dind, results_high, results_low


def fs_stats_dataframe(
    info,
    results_dind,
    results_high,
    results_low,
    results_s_ratio,
    results_hapdaf_o,
    results_hapdaf_s,
):
    try:
        out_dind_high_low = np.hstack(
            [info[:, :2], results_dind, results_high, results_low]
        )
        df_dind_high_low = pl.DataFrame(
            out_dind_high_low,
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "dind": pl.Float64,
                # "dind_flip": pl.Float64,
                "high_freq": pl.Float64,
                # "high_freq_flip": pl.Float64,
                "low_freq": pl.Float64,
                # "low_freq_flip": pl.Float64,
            },
        )

    except:
        df_dind_high_low = pl.DataFrame(
            # [[], [], [], [], [], [], [], []],
            [[], [], [], [], []],
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "dind": pl.Float64,
                # "dind_flip": pl.Float64,
                "high_freq": pl.Float64,
                # "high_freq_flip": pl.Float64,
                "low_freq": pl.Float64,
                # "low_freq_flip": pl.Float64,
            },
        )

    try:
        out_s_ratio = np.hstack([info[:, :2], results_s_ratio])
        df_s_ratio = pl.DataFrame(
            out_s_ratio,
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "s_ratio": pl.Float64,
                # "s_ratio_flip": pl.Float64,
            },
        )
    except:
        df_s_ratio = pl.DataFrame(
            # [[], [], [], []],
            [[], [], []],
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "s_ratio": pl.Float64,
                # "s_ratio_flip": pl.Float64,
            },
        )

    try:
        out_hapdaf_s = np.hstack([info[:, :2], np.array(results_hapdaf_s)])
        df_hapdaf_s = pl.DataFrame(
            out_hapdaf_s,
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "hapdaf_s": pl.Float64,
                # "hapdaf_s_flip": pl.Float64,
            },
        )
    except:
        df_hapdaf_s = pl.DataFrame(
            [[], [], []],
            # [[], [], [], []],
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "hapdaf_s": pl.Float64,
                # "hapdaf_s_flip": pl.Float64,
            },
        )

    try:
        out_hapdaf_o = np.hstack([info[:, :2], np.array(results_hapdaf_o)])
        df_hapdaf_o = pl.DataFrame(
            out_hapdaf_o,
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "hapdaf_o": pl.Float64,
                # "hapdaf_o_flip": pl.Float64,
                # "omega_diff": pl.Float64,
            },
        )
    except:
        df_hapdaf_o = pl.DataFrame(
            [[], [], []],
            # [[], [], [], []],
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "hapdaf_o": pl.Float64,
                # "hapdaf_o_flip": pl.Float64,
                # "omega_diff": pl.Float64,
            },
        )

    return (
        df_dind_high_low.fill_nan(None),
        df_s_ratio.fill_nan(None),
        df_hapdaf_o.fill_nan(None),
        df_hapdaf_s.fill_nan(None),
    )


################## iSAFE


@njit("int64[:](float64[:])", cache=True)
def rank_with_duplicates(x):
    # sorted_arr = sorted(x, reverse=True)
    sorted_arr = np.sort(x)[::-1]
    rank_dict = {}
    rank = 1
    prev_value = -1

    for value in sorted_arr:
        if value != prev_value:
            rank_dict[value] = rank
        rank += 1
        prev_value = value

    return np.array([rank_dict[value] for value in x])


# @njit("float64[:,:](float64[:,:])", cache=True)
@njit(parallel=False)
def dot_nb(hap):
    return np.dot(hap.T, hap)


@njit
def neutrality_divergence_proxy(kappa, phi, freq, method=3):
    sigma1 = (kappa) * (1 - kappa)
    sigma1[sigma1 == 0] = 1.0
    sigma1 = sigma1**0.5
    p1 = (phi - kappa) / sigma1
    sigma2 = (freq) * (1 - freq)
    sigma2[sigma2 == 0] = 1.0
    sigma2 = sigma2**0.5
    p2 = (phi - kappa) / sigma2
    nu = freq[np.argmax(p1)]
    p = p1 * (1 - nu) + p2 * nu

    if method == 1:
        return p1
    elif method == 2:
        return p2
    elif method == 3:
        return p


@njit
def calc_H_K(hap, haf):
    """
    :param snp_matrix: Binary SNP Matrix
    :return: H: Sum of HAF-score of carriers of each mutation.
    :return: N: Number of distinct carrier haplotypes of each mutation.

    """
    num_snps, num_haplotypes = hap.shape

    haf_matrix = haf * hap

    K = np.zeros((num_snps))

    for j in range(num_snps):
        ar = haf_matrix[j, :]
        K[j] = len(np.unique(ar[ar > 0]))
    H = np.sum(haf_matrix, 1)
    return (H, K)


def safe(hap):
    num_snps, num_haplotypes = hap.shape

    haf = dot_nb(hap.astype(np.float64)).sum(1)
    # haf = np.dot(hap.T, hap).sum(1)
    H, K = calc_H_K(hap, haf)

    phi = 1.0 * H / haf.sum()
    kappa = 1.0 * K / (np.unique(haf).shape[0])
    freq = hap.sum(1) / num_haplotypes
    safe_values = neutrality_divergence_proxy(kappa, phi, freq)

    # rank = np.zeros(safe_values.size)
    # rank = rank_with_duplicates(safe_values)
    # rank = (
    #     pd.DataFrame(safe_values).rank(method="min", ascending=False).values.flatten()
    # )
    rank = (
        pl.DataFrame({"safe": safe_values})
        .select(pl.col("safe").rank(method="min", descending=True))
        .to_numpy()
        .flatten()
    )

    return haf, safe_values, rank, phi, kappa, freq


def creat_windows_summary_stats_nb(hap, pos, w_size=300, w_step=150):
    num_snps, num_haplotypes = hap.shape
    rolling_indices = create_rolling_indices_nb(num_snps, w_size, w_step)
    windows_stats = {}
    windows_haf = []
    snp_summary = []

    for i, I in enumerate(rolling_indices):
        window_i_stats = {}
        haf, safe_values, rank, phi, kappa, freq = safe(hap[I[0] : I[1], :])

        tmp = pl.DataFrame(
            {
                "safe": safe_values,
                "rank": rank,
                "phi": phi,
                "kappa": kappa,
                "freq": freq,
                "pos": pos[I[0] : I[1]],
                "ordinal_pos": np.arange(I[0], I[1]),
                "window": np.repeat(i, I[1] - I[0]),
            }
        )

        window_i_stats["safe"] = tmp
        windows_haf.append(haf)
        windows_stats[i] = window_i_stats
        snp_summary.append(tmp)

    combined_df = pl.concat(snp_summary).with_columns(
        pl.col("ordinal_pos").cast(pl.Float64)
    )
    # combined_df = combined_df.with_row_count(name="index")
    # snps_summary.select(snps_summary.columns[1:])

    return windows_stats, windows_haf, combined_df


@njit
def create_rolling_indices_nb(total_variant_count, w_size, w_step):
    assert total_variant_count < w_size or w_size > 0

    rolling_indices = []
    w_start = 0
    while True:
        w_end = min(w_start + w_size, total_variant_count)
        if w_end >= total_variant_count:
            break
        rolling_indices.append([w_start, w_end])
        # rolling_indices += [range(int(w_start), int(w_end))]
        w_start += w_step

    return rolling_indices


def run_isafe(
    hap,
    positions,
    max_freq=1,
    min_region_size_bp=49000,
    min_region_size_ps=300,
    ignore_gaps=True,
    window=300,
    step=150,
    top_k=1,
    max_rank=15,
):
    """
    Estimate iSAFE or SAFE when not possible using default Flex-Sweep values.

    Args:
     hap (TYPE): Description
     total_window_size (TYPE): Description
     positions (TYPE): Description
     max_freq (int, optional): Description
     min_region_size_bp (int, optional): Description
     min_region_size_ps (int, optional): Description
     ignore_gaps (bool, optional): Description
     window (int, optional): Description
     step (int, optional): Description
     top_k (int, optional): Description
     max_rank (int, optional): Description

    Returns:
     TYPE: Description

    Raises:
     ValueError: Description
    """

    total_window_size = positions.max() - positions.min()

    dp = np.diff(positions)
    num_gaps = sum(dp > 6000000)
    f = hap.mean(1)
    freq_filter = ((1 - f) * f) > 0
    hap_filtered = hap[freq_filter, :]
    positions_filtered = positions[freq_filter]
    num_snps = hap_filtered.shape[0]

    if (num_snps <= min_region_size_ps) | (total_window_size < min_region_size_bp):
        haf, safe_values, rank, phi, kappa, freq = safe(hap_filtered)

        df_safe = pl.DataFrame(
            {
                "isafe": safe_values,
                "rank": rank,
                "phi": phi,
                "kappa": kappa,
                "daf": freq,
                "positions": positions_filtered,
            }
        )

        return df_safe.select(["positions", "daf", "isafe"]).sort("positions")
    else:
        df_isafe = isafe(
            hap_filtered, positions_filtered, window, step, top_k, max_rank
        )
        df_isafe = (
            df_isafe.filter(pl.col("freq") < max_freq)
            .sort("ordinal_pos")
            .rename({"id": "positions", "isafe": "isafe", "freq": "daf"})
            .filter(pl.col("daf") < max_freq)
            .select(["positions", "daf", "isafe"])
        )

    return df_isafe


def isafe(hap, pos, w_size=300, w_step=150, top_k=1, max_rank=15):
    windows_summaries, windows_haf, snps_summary = creat_windows_summary_stats_nb(
        hap, pos, w_size, w_step
    )
    df_top_k1 = get_top_k_snps_in_each_window(snps_summary, k=top_k)

    ordinal_pos_snps_k1 = np.sort(df_top_k1["ordinal_pos"].unique()).astype(np.int64)

    psi_k1 = step_function(creat_matrix_Psi_k_nb(hap, windows_haf, ordinal_pos_snps_k1))

    df_top_k2 = get_top_k_snps_in_each_window(snps_summary, k=max_rank)
    temp = np.sort(df_top_k2["ordinal_pos"].unique())

    ordinal_pos_snps_k2 = np.sort(np.setdiff1d(temp, ordinal_pos_snps_k1)).astype(
        np.int64
    )

    psi_k2 = step_function(creat_matrix_Psi_k_nb(hap, windows_haf, ordinal_pos_snps_k2))

    alpha = psi_k1.sum(0) / psi_k1.sum()

    iSAFE1 = pl.DataFrame(
        data={
            "ordinal_pos": ordinal_pos_snps_k1,
            "isafe": np.dot(psi_k1, alpha),
            "tier": np.repeat(1, ordinal_pos_snps_k1.size),
        }
    )

    iSAFE2 = pl.DataFrame(
        {
            "ordinal_pos": ordinal_pos_snps_k2,
            "isafe": np.dot(psi_k2, alpha),
            "tier": np.repeat(2, ordinal_pos_snps_k2.size),
        }
    )

    # Concatenate the DataFrames and reset the index
    iSAFE = pl.concat([iSAFE1, iSAFE2])

    # Add the "id" column using values from `pos`

    iSAFE = iSAFE.with_columns(
        pl.col("ordinal_pos")
        .map_elements(lambda x: pos[x], return_dtype=pl.Int64)
        .alias("id")
    )
    # Add the "freq" column using values from `freq`
    freq = hap.mean(1)
    iSAFE = iSAFE.with_columns(
        pl.col("ordinal_pos")
        .map_elements(lambda x: freq[x], return_dtype=pl.Float64)
        .alias("freq")
    )

    # Select the required columns
    df_isafe = iSAFE.select(["ordinal_pos", "id", "isafe", "freq", "tier"])

    return df_isafe


@njit
def creat_matrix_Psi_k_nb(hap, hafs, Ifp):
    P = np.zeros((len(Ifp), len(hafs)))
    for i in range(len(Ifp)):
        for j in range(len(hafs)):
            P[i, j] = isafe_kernel_nb(hafs[j], hap[Ifp[i], :])
    return P


@njit
def isafe_kernel_nb(haf, snp):
    phi = haf[snp == 1].sum() * 1.0 / haf.sum()
    kappa = len(np.unique(haf[snp == 1])) / (1.0 * len(np.unique(haf)))
    f = np.mean(snp)
    sigma2 = (f) * (1 - f)
    if sigma2 == 0:
        sigma2 = 1.0
    sigma = sigma2**0.5
    p = (phi - kappa) / sigma
    return p


@njit
def creat_matrix_Psi_k_nb(hap, hafs, Ifp):
    """Further optimized version with pre-computed unique values"""
    P = np.zeros((len(Ifp), len(hafs)))

    # Pre-compute for each haf: sum and unique count
    haf_sums = np.zeros(len(hafs))
    haf_unique_counts = np.zeros(len(hafs))

    for j in range(len(hafs)):
        haf_sums[j] = hafs[j].sum()
        haf_unique_counts[j] = len(np.unique(hafs[j]))

    for i in range(len(Ifp)):
        snp = hap[Ifp[i], :]

        # Pre-compute common values for this row
        f = np.mean(snp)
        sigma2 = f * (1 - f)
        if sigma2 == 0:
            sigma2 = 1.0
        sigma = sigma2**0.5

        snp_ones_idx = np.where(snp == 1)[0]

        for j in range(len(hafs)):
            haf = hafs[j]

            # Use pre-computed values
            phi = haf[snp_ones_idx].sum() / haf_sums[j]
            kappa = len(np.unique(haf[snp_ones_idx])) / haf_unique_counts[j]

            p = (phi - kappa) / sigma
            P[i, j] = p

    return P


def step_function(P0):
    P = P0.copy()
    P[P < 0] = 0
    return P


def get_top_k_snps_in_each_window(df_snps, k=1):
    """
    :param df_snps:  this datafram must have following columns: ["safe","ordinal_pos","window"].
    :param k:
    :return: return top k snps in each window.
    """
    return (
        df_snps.group_by("window")
        .agg(pl.all().sort_by("safe", descending=True).head(k))
        .explode(pl.all().exclude("window"))
        .sort("window")
        .select(pl.all().exclude("window"), pl.col("window"))
    )


################## LD stats


def Ld(r_2, mask=None) -> tuple:
    """
    Compute Kelly Zns statistic (1997) and omega_max. Average r2
    among every pair of loci in the genomic window.

    Args:hap (numpy.ndarray): 2D array representing haplotype data. Rows correspond to different mutations, and columnscorrespond to chromosomes.
    pos (numpy.ndarray): 1D array representing the positions of mutations.
    min_freq (float, optional): Minimum frequency threshold. Default is 0.05.
    max_freq (float, optional): Maximum frequency threshold. Default is 1.
    window (int, optional): Genomic window size. Default is 500000.

    Returns:tuple: A tuple containing two values:
    - kelly_zns (float): Kelly Zns statistic.
    - omega_max (float): Nielsen omega max.
    """

    # r2_matrix = r2_torch(hap_filter)
    if mask is not None:
        idx = np.flatnonzero(mask)
        _r_2 = r_2[np.ix_(idx, idx)]
    else:
        _r_2 = r_2

    S = _r_2.shape[0]
    zns = _r_2.sum() / comb(S, 2)
    # Index combination to iter
    omega_max = omega_linear_correct_mask(np.asarray(_r_2))

    # return zns, 0
    return zns, omega_max


# def Ld_jax(r_2) -> tuple:
#     """
#     Compute Kelly Zns statistic (1997) and omega_max. Average r2
#     among every pair of loci in the genomic window.

#     Args:hap (numpy.ndarray): 2D array representing haplotype data. Rows correspond to different mutations, and columnscorrespond to chromosomes.
#     pos (numpy.ndarray): 1D array representing the positions of mutations.
#     min_freq (float, optional): Minimum frequency threshold. Default is 0.05.
#     max_freq (float, optional): Maximum frequency threshold. Default is 1.
#     window (int, optional): Genomic window size. Default is 500000.

#     Returns:tuple: A tuple containing two values:
#     - kelly_zns (float): Kelly Zns statistic.
#     - omega_max (float): Nielsen omega max.
#     """

#     # r2_matrix = r2_torch(hap_filter)

#     S = r_2.shape[0]
#     zns = r_2.sum() / comb(S, 2)
#     # Index combination to iter
#     omega_max = omega_linear_correct_jax(r_2)

#     # return zns, 0
#     return zns, omega_max


@njit("float64(int8[:], int8[:])", parallel=False)
def r2(locus_A: np.ndarray, locus_B: np.ndarray) -> float:
    """
    Calculate r^2 and D between the two loci A and B.

    Args: locus_A (numpy.ndarray): 1D array representing alleles at locus A.
    locus_B (numpy.ndarray): 1D array representing alleles at locus B.

    Returns:
        float: r^2 value.
    """
    n = locus_A.size
    # Frequency of allele 1 in locus A and locus B
    a1 = 0
    b1 = 0
    count_a1b1 = 0

    for i in range(n):
        a1 += locus_A[i]
        b1 += locus_B[i]
        count_a1b1 += locus_A[i] * locus_B[i]

    a1 /= n
    b1 /= n
    a1b1 = count_a1b1 / n
    D = a1b1 - a1 * b1

    r_squared = (D**2) / (a1 * (1 - a1) * b1 * (1 - b1))
    return r_squared


def compute_r2_matrix_upper_og(hap):
    """
    Computes the pairwise linkage disequilibrium (LD) measure r² between all pairs of SNPs,
    using a vectorized implementation. Only the strict upper triangle of the r² matrix is returned,
    where r²[i, j] is defined only for i < j (i.e., no redundancy).

    This function assumes biallelic haplotype data and computes r² as:

        r²_ij = (P_ij - p_i * p_j)^2 / [p_i(1 - p_i) * p_j(1 - p_j)]

    where:
        - P_ij: empirical joint frequency that SNP i and SNP j are both 1
        - p_i: allele frequency (of 1s) at SNP i
        - p_j: allele frequency (of 1s) at SNP j

    Parameters
    ----------
    hap : ndarray of shape (S, N)
        Integer haplotype matrix with S SNPs (rows) and N haplotypes (columns).
        Values should be 0 or 1. Input dtype may be int8 or bool.

    Returns
    -------
    r2_upper : ndarray of shape (S, S)
        Float64 matrix of r² values with only the upper triangle (i < j) filled in.
        All values where i >= j are zero.
    """
    # Convert haplotype data to float64 for dot product operations
    X = hap.astype(np.float64)
    S, N = X.shape

    # Allele frequency at each SNP
    p = X.mean(axis=1)

    # Joint frequency matrix: E[h_i * h_j]
    P = (X @ X.T) / N

    # Covariance between SNPs: D_ij = P_ij - p_i * p_j
    D = P - p[:, None] * p[None, :]

    # Denominator matrix denom_i * denom_j
    denom = p * (1.0 - p)
    den_outer = denom[:, None] * denom[None, :]

    # Compute r^2 matrix, checking zero division at monomorphic sites
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = (D * D) / den_outer
        r2[den_outer == 0.0] = 0.0

    # Retain only the upper triangle (i < j), zero elsewhere
    r2_upper = np.triu(r2, k=1)

    return r2_upper


@njit(parallel=False)
def compute_r2_matrix_upper(hap):
    S, N = hap.shape
    X = hap.astype(np.float64)

    # Allele frequencies
    p = np.empty(S, dtype=np.float64)
    for i in range(S):
        s = 0.0
        for n in range(N):
            s += X[i, n]
        p[i] = s / N

    denom = p * (1.0 - p)

    # Joint frequency matrix using efficient dot product
    P = (X @ X.T) / N

    r2_upper = np.zeros((S, S), dtype=np.float64)

    for i in range(S):
        for j in range(i + 1, S):
            D = P[i, j] - p[i] * p[j]
            denom_ij = denom[i] * denom[j]
            if denom_ij == 0.0:
                r2_upper[i, j] = 0.0
            else:
                r2_upper[i, j] = (D * D) / denom_ij

    return r2_upper


@njit(parallel=False)
def omega_linear_correct(r2_matrix):
    """
    Compute omega_max with:
      - O(n^2) to build row_sum & col_sum
      - O(n) to sweep over cutpoints _l with no inner loops
    Only reads r2_matrix[i,j] for i<j.
    """
    S = r2_matrix.shape[0]
    if S < 3:
        # return np.array([0.0,0.0])
        return 0.0

    #   Build row_sum[i] = sum_{j>i} r2[i,j]
    #   and       col_sum[j] = sum_{i<j} r2[i,j]
    #   Also accumulate total of all upper‐triangle entries.
    row_sum = np.zeros(S, np.float64)
    col_sum = np.zeros(S, np.float64)
    total = 0.0
    for i in range(S):
        s = 0.0
        for j in range(i + 1, S):
            v = r2_matrix[i, j]
            s += v
            col_sum[j] += v
        row_sum[i] = s
        total += s

    # Build prefix_L[_l] = sum_{i<j<_l} r2[i,j]
    prefix_L = np.zeros(S, np.float64)
    for _l in range(1, S):
        prefix_L[_l] = prefix_L[_l - 1] + col_sum[_l - 1]

    # Build suffix_R[_l] = sum_{_l≤i<j} r2[i,j]
    suffix_R = np.zeros(S + 1, np.float64)
    for _l in range(S - 1, -1, -1):
        suffix_R[_l] = suffix_R[_l + 1] + row_sum[_l]

    # Sweep _l = 3..S-3 in O(S), compute _omega and track maximum
    omega_max = 0.0
    omega_argmax = -1.0
    for _l in range(3, S - 2):
        sum_L = prefix_L[_l]
        sum_R = suffix_R[_l]
        sum_LR = total - sum_L - sum_R
        if sum_LR > 0.0:
            denom_L = (_l * (_l - 1) / 2.0) + ((S - _l) * (S - _l - 1) / 2.0)
            denom_R = _l * (S - _l)
            _omega = ((sum_L + sum_R) / denom_L) / (sum_LR / denom_R)
            if _omega > omega_max:
                omega_max = _omega
                omega_argmax = _l + 2

    # return np.array([omega_max,omega_argmax])
    return omega_max


@njit(parallel=False)
def omega_linear_correct_mask(_r2, mask=None):
    """
    Computes the omega statistic (Kim and Nielsen, 2004) from an r^2 matrix,
    used to detect signatures of recent selective sweeps by comparing LD within
    vs. between partitions of a SNP window.

    This implementation returns the maximum omega value over all valid
    SNP split points (l) within the region.

    Precomputes all LD sums once (O(S²)) and uses cumulative prefix/suffix arrays for fast within/between partitioning avoding redundant recomputation for each l

    Parameters
    ----------
    _r2 : ndarray (S x S)
        Symmetric matrix of pairwise r^2 values between S SNPs.
        Only the upper triangle is assumed to contain valid values.
    mask : ndarray of bool, optional
        Boolean array to restrict computation to a subset of SNPs.
        If None, all SNPs are included.

    Returns
    -------
    omega_max : float
        Maximum omega value observed across all candidate breakpoints.
    """

    # Select the SNP indices to include based on mask
    if mask is None:
        idx = np.arange(_r2.shape[0])
    else:
        idx = np.where(mask)[0]

    S = len(idx)
    if S < 3:
        return 0.0

    # Accumulate sums of r^2 for all pairs (i < j)
    row_sum = np.zeros(S, np.float64)  # ∑_j>i r2[i,j]
    col_sum = np.zeros(S, np.float64)  # ∑_i<j r2[i,j]
    total = 0.0
    for a in range(S):
        i = idx[a]
        s = 0.0
        for b in range(a + 1, S):
            j = idx[b]
            v = _r2[i, j]
            s += v
            col_sum[b] += v
        row_sum[a] = s
        total += s

    # Precompute prefix and suffix sums for L and R partitions
    # ∑_{i<l} ∑_{j<l} r²[i,j]
    prefix_L = np.zeros(S, np.float64)
    for _l in range(1, S):
        prefix_L[_l] = prefix_L[_l - 1] + col_sum[_l - 1]

    # ∑_{i>=l} ∑_{j>i} r²[i,j]
    suffix_R = np.zeros(S + 1, np.float64)
    for _l in range(S - 1, -1, -1):
        suffix_R[_l] = suffix_R[_l + 1] + row_sum[_l]

    # Main omega scan: iterate over candidate split points _l
    omega_max = 0.0
    omega_argmax = -1.0
    for _l in range(3, S - 2):
        # within-left block
        sum_L = prefix_L[_l]
        # within-right block
        sum_R = suffix_R[_l]
        # between-block
        sum_LR = total - sum_L - sum_R
        if sum_LR > 0.0:
            denom_L = (_l * (_l - 1) / 2.0) + ((S - _l) * (S - _l - 1) / 2.0)
            denom_R = _l * (S - _l)
            _omega = ((sum_L + sum_R) / denom_L) / (sum_LR / denom_R)
            if _omega > omega_max:
                omega_max = _omega
                # SNP index offset for compatibility
                omega_argmax = _l + 2

    return omega_max


# @jit
# def compute_r2_matrix_upper_jax(hap):
#     """
#     Vectorized, JAX‐native computation of the strict upper triangle of
#     the r^2 matrix for a (S x N) 0/1 haplotype array `hap`.

#     Returns:
#       r2_upper (S x S) float64 array where
#         r2_upper[i,j] = r^2 between site i and j if i<j, else 0.
#     """
#     # 1) Cast to float64
#     X = hap.astype(jnp.float64)
#     S, N = X.shape

#     # 2) Allele freqs p_i = mean over cols
#     p = jnp.mean(X, axis=1)  # (S,)

#     # 3) Joint freqs P = (X @ X.T) / N
#     P = (X @ X.T) / N  # (S, S)

#     # 4) Covariance D = P - p_i * p_j
#     D = P - p[:, None] * p[None, :]  # (S, S)

#     # 5) Variance denom = p*(1-p), and outer product
#     var = p * (1.0 - p)  # (S,)
#     den = var[:, None] * var[None, :]  # (S, S)

#     # 6) Compute r2 with guard: when den==0, set 0
#     r2 = (D * D) / den
#     r2 = jnp.where(den > 0.0, r2, 0.0)

#     # 7) Strict upper triangle
#     r2_upper = jnp.triu(r2, k=1)

#     return r2_upper


# @jit
# def omega_linear_correct_jax(r2_matrix):
#     S = r2_matrix.shape[0]
#     if S < 3:
#         return 0.0

#     # Build row_sum[i] = sum_{j>i} r2[i,j]
#     #       col_sum[j] = sum_{i<j} r2[i,j]
#     mask = jnp.triu(jnp.ones((S, S)), k=1)
#     row_sum = jnp.sum(r2_matrix * mask, axis=1)
#     col_sum = jnp.sum(r2_matrix * mask, axis=0)
#     total = jnp.sum(row_sum)

#     # prefix_L[_l] = sum_{i<j<_l} r2[i,j] (cumsum over col_sum)
#     prefix_L = jnp.zeros(S)
#     prefix_L = prefix_L.at[1:].set(jnp.cumsum(col_sum[:-1]))

#     # suffix_R[_l] = sum_{_l≤i<j} r2[i,j] (reverse cumsum over row_sum)
#     suffix_R = jnp.zeros(S + 1)
#     suffix_R = suffix_R.at[:-1].set(jnp.cumsum(row_sum[::-1])[::-1])

#     # Sweep _l = 3..S-3
#     def omega_body(_l, state):
#         omega_max, omega_argmax = state
#         sum_L = prefix_L[_l]
#         sum_R = suffix_R[_l]
#         sum_LR = total - sum_L - sum_R
#         cond = sum_LR > 0.0
#         denom_L = (_l * (_l - 1) / 2.0) + ((S - _l) * (S - _l - 1) / 2.0)
#         denom_R = _l * (S - _l)
#         _omega = jnp.where(
#             cond,
#             ((sum_L + sum_R) / denom_L) / (sum_LR / denom_R),
#             -jnp.inf,
#         )
#         update = _omega > omega_max
#         omega_max = jnp.where(update, _omega, omega_max)
#         omega_argmax = jnp.where(update, _l + 2, omega_argmax)
#         return omega_max, omega_argmax

#     omega_max = 0.0
#     omega_argmax = -1.0

#     # Use JAX's fori_loop for efficient iteration
#     (omega_max, omega_argmax) = fori_loop(
#         3, S - 2, omega_body, (omega_max, omega_argmax)
#     )

#     return omega_max


################## Spectrum stats


def fay_wu_h_normalized(hap: np.ndarray, pos, start=None, stop=None) -> tuple:
    """
    Compute Fay-Wu's H test statistic and its normalized version.

    Args:hap (numpy.ndarray): 2D array representing haplotype data. Rows correspond to different mutations, and columns correspond to chromosomes.

    Returns:tuple: A tuple containing two values:
        - h (float): Fay-Wu H test statistic.
        - h_normalized (float): Normalized Fay-Wu H test statistic.
    """

    if start is not None or stop is not None:
        loc = (pos >= start) & (pos <= stop)
        pos = pos[loc]
        hap = hap[loc, :]

    # Count segregating and chromosomes
    S, n = hap.shape
    # Create SFS to count ith mutation in sample
    Si = sfs(hap.sum(axis=1), n)[1:-1]
    # ith mutations
    i = np.arange(1, n)

    # (n-1)th harmonic numbers
    an = np.sum(1 / i)
    bn = np.sum(1 / i**2)
    bn_1 = bn + 1 / (n**2)

    # calculate theta_w absolute value
    theta_w = S / an

    # calculate theta_pi absolute value
    theta_pi = ((2 * Si * i * (n - i)) / (n * (n - 1))).sum()

    # calculate theta_h absolute value
    theta_h = ((2 * Si * np.power(i, 2)) / (n * (n - 1))).sum()

    # calculate theta_l absolute value
    theta_l = (np.arange(1, n) * Si).sum() / (n - 1)

    theta_square = (S * (S - 1)) / (an**2 + bn)

    h = theta_pi - theta_h

    var_1 = (n - 2) / (6 * (n - 1)) * theta_w

    var_2 = (
        (
            (18 * (n**2) * (3 * n + 2) * bn_1)
            - ((88 * (n**3) + 9 * (n**2)) - (13 * n + 6))
        )
        / (9 * n * ((n - 1) ** 2))
    ) * theta_square

    # cov = (((n+1) / (3*(n-1)))*theta_w) + (((7*n*n+3*n-2-4*n*(n+1)*bn_1)/(2*(n-1)**2))*theta_square)

    # var_theta_l = (n * theta_w)/(2.0 * (n - 1.0)) + (2.0 * np.power(n/(n - 1.0), 2.0) * (bn_1 - 1.0) - 1.0) * theta_square;
    # var_theta_pi = (3.0 * n *(n + 1.0) * theta_w + 2.0 * ( n * n + n + 3.0) * theta_square)/ (9 * n * (n -1.0));

    h_normalized = h / np.sqrt(var_1 + var_2)

    # h_prime = h / np.sqrt(var_theta_l+var_theta_pi - 2.0 * cov)

    return (h, h_normalized)


def zeng_e(hap: np.ndarray, pos, start=None, stop=None) -> float:
    """
    Compute Zeng's E test statistic.

    Args:hap (numpy.ndarray): 2D array representing haplotype data. Rows correspond to different mutations, and columnscorrespond to chromosomes.

    Returns:
    float: Zeng's E test statistic.
    """

    if start is not None or stop is not None:
        loc = (pos >= start) & (pos <= stop)
        pos = pos[loc]
        hap = hap[loc, :]

    # Count segregating and chromosomes
    S, n = hap.shape
    # Create SFS to count ith mutation in sample
    Si = sfs(hap.sum(axis=1), n)[1:-1]
    # ith mutations
    i = np.arange(1, n)

    # (n-1)th harmonic numbers
    an = np.sum(1.0 / i)
    bn = np.sum(1.0 / i**2.0)

    # calculate theta_w absolute value
    theta_w = S / an

    # calculate theta_l absolute value
    theta_l = (np.arange(1, n) * Si).sum() / (n - 1)

    theta_square = S * (S - 1.0) / (an**2 + bn)

    # Eq. 14
    var_1 = (n / (2.0 * (n - 1.0)) - 1.0 / an) * theta_w
    var_2 = (
        bn / an**2
        + 2 * (n / (n - 1)) ** 2 * bn
        - 2 * (n * bn - n + 1) / ((n - 1) * an)
        - (3 * n + 1) / (n - 1)
    ) * theta_square

    (
        (bn / an**2)
        + (2 * (n / (n - 1)) ** 2 * bn)
        - (2 * (n * bn - n + 1) / ((n - 1) * an))
        - ((3 * n + 1) / (n - 1)) * theta_square
    )
    e = (theta_l - theta_w) / (var_1 + var_2) ** 0.5
    return e


def fuli_f_star(hap, ac):
    """Calculates Fu and Li's D* statistic"""
    S, n = hap.shape

    an = np.sum(np.divide(1.0, range(1, n)))
    bn = np.sum(np.divide(1.0, np.power(range(1, n), 2)))
    an1 = an + np.true_divide(1, n)

    vfs = (
        (
            (2 * (n**3.0) + 110.0 * (n**2.0) - 255.0 * n + 153)
            / (9 * (n**2.0) * (n - 1.0))
        )
        + ((2 * (n - 1.0) * an) / (n**2.0))
        - ((8.0 * bn) / n)
    ) / ((an**2.0) + bn)
    ufs = (
        (
            n / (n + 1.0)
            + (n + 1.0) / (3 * (n - 1.0))
            - 4.0 / (n * (n - 1.0))
            + ((2 * (n + 1.0)) / ((n - 1.0) ** 2)) * (an1 - ((2.0 * n) / (n + 1.0)))
        )
        / an
    ) - vfs

    pi = mean_pairwise_difference(ac).sum()
    ss = np.sum(np.sum(hap, axis=1) == 1)
    Fstar1 = (pi - (((n - 1.0) / n) * ss)) / ((ufs * S + vfs * (S**2.0)) ** 0.5)
    return Fstar1


def fuli_f(hap, ac):
    an = np.sum(np.divide(1.0, range(1, n)))
    an1 = an + 1.0 / n
    bn = np.sum(np.divide(1.0, np.power(range(1, n), 2)))

    ss = np.sum(np.sum(hap, axis=1) == 1)
    pi = mean_pairwise_difference(ac).sum()

    if n == 2:
        cn = 1
    else:
        cn = 2.0 * (n * an - 2.0 * (n - 1.0)) / ((n - 1.0) * (n - 2.0))

    v = (
        cn + 2.0 * (np.power(n, 2) + n + 3.0) / (9.0 * n * (n - 1.0)) - 2.0 / (n - 1.0)
    ) / (np.power(an, 2) + bn)
    u = (
        1.0
        + (n + 1.0) / (3.0 * (n - 1.0))
        - 4.0 * (n + 1.0) / np.power(n - 1, 2) * (an1 - 2.0 * n / (n + 1.0))
    ) / an - v
    F = (pi - ss) / np.sqrt(u * S + v * np.power(S, 2))

    return F


def fuli_d_star(hap):
    """Calculates Fu and Li's D* statistic"""

    S, n = hap.shape
    an = np.sum(np.divide(1.0, range(1, n)))
    bn = np.sum(np.divide(1.0, np.power(range(1, n), 2)))
    an1 = an + np.true_divide(1, n)

    cn = 2 * (((n * an) - 2 * (n - 1))) / ((n - 1) * (n - 2))
    dn = (
        cn
        + np.true_divide((n - 2), ((n - 1) ** 2))
        + np.true_divide(2, (n - 1)) * (3.0 / 2 - (2 * an1 - 3) / (n - 2) - 1.0 / n)
    )

    vds = (
        ((n / (n - 1.0)) ** 2) * bn
        + (an**2) * dn
        - 2 * (n * an * (an + 1)) / ((n - 1.0) ** 2)
    ) / (an**2 + bn)
    uds = ((n / (n - 1.0)) * (an - n / (n - 1.0))) - vds

    ss = np.sum(np.sum(hap, axis=1) == 1)
    Dstar1 = ((n / (n - 1.0)) * S - (an * ss)) / (uds * S + vds * (S ^ 2)) ** 0.5
    return Dstar1


def fuli_d(hap):
    S, n = hap.shape

    an = np.sum(np.divide(1.0, range(1, n)))
    bn = np.sum(np.divide(1.0, np.power(range(1, n), 2)))

    ss = np.sum(np.sum(hap, axis=1) == 1)

    if n == 2:
        cn = 1
    else:
        cn = 2.0 * (n * an - 2.0 * (n - 1.0)) / ((n - 1.0) * (n - 2.0))

    v = 1.0 + (np.power(an, 2) / (bn + np.power(an, 2))) * (cn - (n + 1.0) / (n - 1.0))
    u = an - 1.0 - v
    D = (S - ss * an) / np.sqrt(u * S + v * np.power(S, 2))
    return D


@njit
def achaz_y(fs):
    n = fs.sum()
    if n < 3:
        return np.nan  # Or raise Exception if you prefer

    a1, a2 = _harmonic_sums(n)
    a1m1 = a1 - 1.0
    ff = (n - 2.0) / (n * a1m1)

    # Precomputed constants
    alpha = (
        ff**2 * a1m1
        + ff
        * (
            a1 * (4.0 * (n + 1.0) / (n - 1.0) ** 2)
            - 2.0 * (n + 1.0) * (n + 2.0) / (n * (n - 1.0))
        )
        - a1 * 8.0 * (n - 1.0) / (n * (n - 1.0) ** 2)
        + (2.0 * n**2 + 60.0 * n + 12.0) / ((3.0 * n**2) * (n - 1.0))
    )

    beta = (
        ff**2 * (a2 + a1 * (4.0 / ((n - 1.0) * (n - 2.0))) - 4.0 / (n - 2.0))
        + ff
        * (
            -a1 * (4.0 * (n + 2.0) / (n * (n - 1.0) * (n - 2.0)))
            - (n**2 - 3.0 * n**2 - 16.0 * n + 20.0) / (n * (n - 1.0) * (n - 2.0))
        )
        + a1 * (8.0 / (n * (n - 1.0) * (n - 2.0)))
        + (2.0 * (n**4 - n**3 - 17.0 * n**2 - 42.0 * n + 72.0))
        / ((9.0 * n**2) * (n - 1.0) * (n - 2.0))
    )

    # Mask singletons and fixed sites
    y = fs.copy()
    y[0] = 0.0
    y[1] = 0.0
    y[n] = 0.0

    # Segregating sites
    S = 0.0
    for i in range(2, n):
        S += y[i]

    # Pairwise π̂
    pi_hat = 0.0
    for i in range(1, n):
        pi_hat += y[i] * i * (n - i)
    pi_hat /= n * (n - 1.0) / 2.0

    that = S / a1m1
    that_sq = S * (S - 1.0) / (a1m1**2)

    num = pi_hat - ff * S
    den = np.sqrt(alpha * that + beta * that_sq)

    return num / den


@njit(float64[:](int64), cache=True)
def _harmonic_sums(n):
    a1 = 0.0
    a2 = 0.0
    for i in range(1, int(n)):
        inv = 1.0 / i
        a1 += inv
        a2 += inv * inv
    return np.array((a1, a2), dtype=np.float64)


@njit
def _harmonic_watterson(n):
    w = 0.0
    for i in range(1, n):
        w += 1 / i
    return w


@njit
def theta_watterson(ac, positions):
    # count segregating variants
    S = ac.shape[0]
    n = ac[0].sum()

    # (n-1)th harmonic number
    a1 = _harmonic_watterson(n)

    # calculate absolute value
    theta_hat_w_abs = S / a1

    # calculate value per base
    n_bases = positions[-1] - (positions[1] + 1)
    theta_hat_w = theta_hat_w_abs / n_bases

    return theta_hat_w


@njit
def sfs_nb(dac, n):
    """
    Compute the site frequency spectrum given derived allele counts.

    Parameters
    ----------
    dac : int64[:]   # array of derived allele counts (0..n)
    n   : int64      # total number of chromosomes; if <= 0, inferred as max(dac)

    Returns
    -------
    sfs : int64[:]   # length n+1, where sfs[k] is count of sites with k derived alleles
    """
    # infer n if not provided or invalid
    if n <= 0:
        maxv = 0
        for i in range(dac.shape[0]):
            if dac[i] > maxv:
                maxv = dac[i]
        n = maxv

    # initialize spectrum
    s = np.zeros(n + 1, dtype=np.int64)

    # counts
    for i in range(dac.shape[0]):
        k = dac[i]
        if 0 <= k <= n:
            s[k] += 1
    return s


@njit
def theta_pi(ac):
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()
    denom_pairs = n * (n - 1.0)

    # pi = np.zeros(S)
    pi = np.zeros(S)
    for j in range(S):
        aj = ac[j, 1]
        pi[j] = 2.0 * aj * (n - aj) / denom_pairs
    return pi


@njit
def tajima_d(ac, min_sites=3):
    # count segregating variants
    S = ac.shape[0]

    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()
    if S < min_sites:
        return np.nan

    # (n-1)th harmonic number
    an, bn = _harmonic_sums(n)

    # calculate Watterson's theta (absolute value)
    theta_hat_w_abs = S / an

    # calculate mean pairwise difference
    mpd = theta_pi(ac)

    # calculate theta_hat pi (sum differences over variants)
    theta_hat_pi_abs = mpd.sum()

    # N.B., both theta estimates are usually divided by the number of
    # (accessible) bases but here we want the absolute difference
    d = theta_hat_pi_abs - theta_hat_w_abs

    # calculate the denominator (standard deviation)
    a2 = np.sum(1 / (np.arange(1, n) ** 2))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - (1 / an)
    c2 = b2 - ((n + 2) / (an * n)) + (a2 / (an**2))
    e1 = c1 / an
    e2 = c2 / (an**2 + a2)
    d_stdev = np.sqrt((e1 * S) + (e2 * S * (S - 1)))

    # finally calculate Tajima's D
    D = d / d_stdev

    return D


@njit(float64(int64[:]), cache=True)
def achaz_y(fs):
    n = fs.shape[0] - 1
    if n < 3:
        return np.nan
    a1, a2 = _harmonic_sums(n)
    a1m1 = a1 - 1.0
    ff = (n - 2.0) / (n * a1m1)
    inv_n = 1.0 / n
    inv_n1 = 1.0 / (n - 1.0)
    inv_n2 = 1.0 / (n - 2.0)
    n2 = n * n
    alpha = (
        ff * ff * a1m1
        + ff
        * (
            a1 * (4.0 * (n + 1.0) * inv_n1 * inv_n1)
            - 2.0 * (n + 1.0) * (n + 2.0) * inv_n * inv_n1
        )
        - a1 * 8.0 * (n - 1.0) * inv_n * inv_n1 * inv_n1
        + (2.0 * n2 + 60.0 * n + 12.0) * (inv_n * inv_n) * (1.0 / 3.0) * inv_n1
    )
    beta = (
        ff * ff * (a2 + a1 * (4.0 * inv_n1 * inv_n2) - 4.0 * inv_n2)
        + ff
        * (
            -a1 * (4.0 * (n + 2.0) * inv_n * inv_n1 * inv_n2)
            - ((n2 - 3.0 * n2 - 16.0 * n + 20.0) * inv_n * inv_n1 * inv_n2)
        )
        + a1 * (8.0 * inv_n * inv_n1 * inv_n2)
        + (2.0 * (n2 * n2 - n2 * n - 17.0 * n2 - 42.0 * n + 72.0))
        * (inv_n * inv_n)
        * (inv_n1 * inv_n2)
        * (1.0 / 9.0)
    )
    y = fs.copy()
    y[0] = y[1] = y[n] = 0.0
    S = 0.0
    pi_sum = 0.0
    for i in range(2, n + 1):
        yi = y[i]
        if i > 1 and i < n:
            S += yi
        if i < n:
            pi_sum += yi * i * (n - i)

    # At least 1 seg site
    if S < 1:
        return np.nan

    pi_hat = pi_sum / (n * (n - 1.0) * 0.5)
    that = S / a1m1
    that_sq = S * (S - 1.0) / (a1m1 * a1m1)
    return (pi_hat - ff * S) / np.sqrt(alpha * that + beta * that_sq)


@njit
def fay_wu_h_norm_si(ac, positions=None):
    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    fs = sfs_nb(ac[:, 1], n)[1:-1]

    i_arr = np.arange(1, int(n))
    a1 = np.sum(1.0 / i_arr)
    bn = np.sum(1.0 / (i_arr * i_arr)) + 1.0 / (n * n)
    theta_w = S / a1
    pi = 0.0
    theta_h = 0.0
    for k in range(1, int(n)):
        si = fs[k - 1]
        pi += (2 * si * k * (n - k)) / (n * (n - 1.0))
        theta_h += (2 * si * k * k) / (n * (n - 1.0))
    tl = 0.0
    for k in range(1, int(n)):
        tl += k * fs[k - 1]
    tl /= n - 1.0
    var1 = (n - 2.0) / (6.0 * (n - 1.0)) * theta_w
    theta_sq = S * (S - 1.0) / (a1 * a1 + bn)

    var2 = (
        ((18 * n * n * (3 * n + 2) * bn) - (88 * n * n * n + 9 * n * n - 13 * n - 6))
        / (9.0 * n * (n - 1.0) * (n - 1.0))
    ) * theta_sq

    h = pi - theta_h

    if positions is not None:
        theta_h = theta_h / (positions[-1] - (positions[0] - 1))
    return theta_h, h, h / np.sqrt(var1 + var2)


@njit
def zeng_e_si(ac):
    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    fs = sfs_nb(ac[:, 1], n)[1:-1]

    i_arr = np.arange(1, int(n))
    a1 = np.sum(1.0 / i_arr)
    bn = np.sum(1.0 / (i_arr * i_arr))
    theta_w = S / a1
    tl = 0.0
    for k in range(1, int(n)):
        tl += k * fs[k - 1]
    tl /= n - 1.0
    theta_sq = S * (S - 1.0) / (a1 * a1 + bn)
    var1 = (n / (2.0 * (n - 1.0)) - 1.0 / a1) * theta_w
    var2 = (
        bn / a1 / a1
        + 2 * (n / (n - 1.0)) * (n / (n - 1.0)) * bn
        - 2 * (n * bn - n + 1) / ((n - 1.0) * a1)
        - (3 * n + 1) / (n - 1.0)
    ) * theta_sq
    return (tl - theta_w) / np.sqrt(var1 + var2)


@njit
def fuli_f_star_nb(ac):
    """Calculates Fu and Li's D* statistic"""
    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    an, bn = _harmonic_sums(n)
    an1 = an + np.true_divide(1, n)

    denom_pairs = n * (n - 1.0)
    pi = 0.0
    for j in range(S):
        aj = ac[j, 1]
        pi += 2.0 * aj * (n - aj) / denom_pairs

    ss = (ac[:, 1] == 1).sum()

    vfs = (
        (
            (2 * (n**3.0) + 110.0 * (n**2.0) - 255.0 * n + 153)
            / (9 * (n**2.0) * (n - 1.0))
        )
        + ((2 * (n - 1.0) * an) / (n**2.0))
        - ((8.0 * bn) / n)
    ) / ((an**2.0) + bn)
    ufs = (
        (
            n / (n + 1.0)
            + (n + 1.0) / (3 * (n - 1.0))
            - 4.0 / (n * (n - 1.0))
            + ((2 * (n + 1.0)) / ((n - 1.0) ** 2)) * (an1 - ((2.0 * n) / (n + 1.0)))
        )
        / an
    ) - vfs

    num = pi - ((n - 1.0) / n) * ss
    den = np.sqrt(ufs * S + vfs * (S * S))
    return num / den


@njit
def fuli_f_nb(ac):
    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()
    an, bn = _harmonic_sums(n)
    an1 = an + 1.0 / n

    ss = (ac[:, 1] == 1).sum()

    denom_pairs = n * (n - 1.0)
    pi = 0.0
    for j in range(S):
        aj = ac[j, 1]
        pi += 2.0 * aj * (n - aj) / denom_pairs

    if n == 2:
        cn = 1
    else:
        cn = 2.0 * (n * an - 2.0 * (n - 1.0)) / ((n - 1.0) * (n - 2.0))

    v = (
        cn + 2.0 * (np.power(n, 2) + n + 3.0) / (9.0 * n * (n - 1.0)) - 2.0 / (n - 1.0)
    ) / (np.power(an, 2) + bn)
    u = (
        1.0
        + (n + 1.0) / (3.0 * (n - 1.0))
        - 4.0 * (n + 1.0) / np.power(n - 1, 2) * (an1 - 2.0 * n / (n + 1.0))
    ) / an - v
    F = (pi - ss) / np.sqrt(u * S + v * np.power(S, 2))

    return F


@njit
def fuli_d_star_nb(ac):
    """Calculates Fu and Li's D* statistic"""

    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    an, bn = _harmonic_sums(n)
    an1 = an + np.true_divide(1, n)

    cn = 2 * (((n * an) - 2 * (n - 1))) / ((n - 1) * (n - 2))
    dn = (
        cn
        + np.true_divide((n - 2), ((n - 1) ** 2))
        + np.true_divide(2, (n - 1)) * (3.0 / 2 - (2 * an1 - 3) / (n - 2) - 1.0 / n)
    )

    vds = (
        ((n / (n - 1.0)) ** 2) * bn
        + (an**2) * dn
        - 2 * (n * an * (an + 1)) / ((n - 1.0) ** 2)
    ) / (an**2 + bn)
    uds = ((n / (n - 1.0)) * (an - n / (n - 1.0))) - vds

    ss = (ac[:, 1] == 1).sum()
    Dstar1 = ((n / (n - 1.0)) * S - (an * ss)) / (uds * S + vds * (S**2)) ** 0.5
    return Dstar1


@njit
def fuli_d_nb(ac):
    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    an, bn = _harmonic_sums(n)

    ss = (ac[:, 1] == 1).sum()

    if n == 2:
        cn = 1
    else:
        cn = 2.0 * (n * an - 2.0 * (n - 1.0)) / ((n - 1.0) * (n - 2.0))

    v = 1.0 + (np.power(an, 2) / (bn + np.power(an, 2))) * (cn - (n + 1.0) / (n - 1.0))
    u = an - 1.0 - v
    D = (S - ss * an) / np.sqrt(u * S + v * np.power(S, 2))
    return D


################## LASSI


def get_empir_freqs_np_fast(hap):
    """
    Optimized version to calculate the empirical frequencies of haplotypes.

    Parameters:
    - hap (numpy.ndarray): Shape (S, n), where S = SNPs, n = individuals.

    Returns:
    - k_counts (numpy.ndarray): Counts of each unique haplotype.
    - h_f (numpy.ndarray): Frequencies of each unique haplotype.
    """
    # Transpose so each haplotype is a row
    hap_t = hap.T  # shape (n, S)

    # Hash each haplotype row into a unique identifier
    hashes = np.ascontiguousarray(hap_t).view(
        np.dtype((np.void, hap_t.dtype.itemsize * hap_t.shape[1]))
    )

    # Use np.unique on 1D hashes
    _, unique_counts = np.unique(hashes, return_counts=True)

    # Sort counts in descending order
    k_counts = np.sort(unique_counts)[::-1]
    h_f = k_counts / hap_t.shape[0]

    return k_counts, h_f


def process_spectra(
    k: np.ndarray, h_f: np.ndarray, K_truncation: int, n_ind: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process haplotype count and frequency spectra.

    Parameters:
    - k (numpy.ndarray): Counts of each unique haplotype.
    - h_f (numpy.ndarray): Empirical frequencies of each unique haplotype.
    - K_truncation (int): Number of haplotypes to consider.
    - n_ind (int): Number of individuals.

    Returns:
    - Kcount (numpy.ndarray): Processed haplotype count spectrum.
    - Kspect (numpy.ndarray): Processed haplotype frequency spectrum.
    """
    # Truncate count and frequency spectrum
    Kcount = k[:K_truncation]
    Kspect = h_f[:K_truncation]

    # Normalize count and frequency spectra
    Kcount = Kcount / Kcount.sum() * n_ind
    Kspect = Kspect / Kspect.sum()

    # Pad with zeros if necessary
    if Kcount.size < K_truncation:
        Kcount = np.concatenate([Kcount, np.zeros(K_truncation - Kcount.size)])
        Kspect = np.concatenate([Kspect, np.zeros(K_truncation - Kspect.size)])

    return Kcount, Kspect


def LASSI_spectrum_and_Kspectrum(
    hap_data, K_truncation=10, window=110, step=5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute haplotype count and frequency spectra within sliding windows.

    Parameters:
    - hap (numpy.ndarray): Array of haplotypes where each column represents an individual and each row represents a SNP.
    - pos (numpy.ndarray): Array of SNP positions.
    - K_truncation (int): Number of haplotypes to consider.
    - window (int): Size of the sliding window.
    - step (int): Step size for sliding the window.

    Returns:
    - K_count (numpy.ndarray): Haplotype count spectra for each window.
    - K_spectrum (numpy.ndarray): Haplotype frequency spectra for each window.
    - windows_centers (numpy.ndarray): Centers of the sliding windows.
    """
    filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in scalar divide",
    )
    np.seterr(divide="ignore", invalid="ignore")

    if isinstance(hap_data, list) or isinstance(hap_data, tuple):
        hap, rec_map, p = hap_data
    elif isinstance(hap_data, str):
        try:
            hap, rec_map, p = ms_parser(hap_data)
        except:
            try:
                hap, rec_map, p = genome_reader(hap_data, region)
            except:
                return None
    else:
        return None

    (
        hap_int,
        rec_map_01,
        ac,
        biallelic_mask,
        position_masked,
        genetic_position_masked,
    ) = filter_gt(hap, rec_map, region=None)
    freqs = ac[:, 1] / ac.sum(axis=1)

    K_count = []
    K_spectrum = []
    windows_centers = []
    S, n = hap_int.shape
    for i in range(0, S, step):
        hap_subset = hap_int[i : i + window, :]

        # Calculate window center based on median SNP position
        windows_centers.append(np.median(position_masked[i : i + window]))

        # Compute empirical frequencies and process spectra for the window
        k, h_f = get_empir_freqs_np_fast(hap_subset)
        K_count_subset, K_spectrum_subset = process_spectra(k, h_f, K_truncation, n)

        K_count.append(K_count_subset)
        K_spectrum.append(K_spectrum_subset)
        if hap_subset.shape[0] < window:
            break

    return np.array(K_count), np.array(K_spectrum), np.array(windows_centers)


def neut_average(K_spectrum: np.ndarray) -> np.ndarray:
    """
    Compute the neutral average of haplotype frequency spectra.

    Parameters:
    - K_spectrum (numpy.ndarray): Haplotype frequency spectra.

    Returns:
    - out (numpy.ndarray): Neutral average haplotype frequency spectrum.
    """
    weights = []
    S, n = K_spectrum.shape
    # Compute mean spectrum
    gwide_K = np.mean(K_spectrum, axis=0)

    # Calculate weights for averaging
    if S % 5e4 == 0:
        weights.append(5e4)
    else:
        small_weight = S % 5e4
        weights.append(small_weight)

    # Compute weighted average
    out = np.average([gwide_K], axis=0, weights=weights)

    return out


@njit("float64(float64[:],float64[:],int64)", cache=True)
def easy_likelihood(K_neutral, K_count, K_truncation):
    """
    Basic computation of the likelihood function; runs as-is for neutrality, but called as part of a larger process for sweep model
    """

    likelihood_list = []

    for i in range(K_truncation):
        likelihood_list.append(K_count[i] * np.log(K_neutral[i]))

    likelihood = sum(likelihood_list)

    return likelihood


@njit("float64(float64[:],float64[:],int64,int64,float64,float64)", cache=True)
def sweep_likelihood(K_neutral, K_count, K_truncation, m_val, epsilon, epsilon_max):
    """
    Computes the likelihood of a sweep under optimized parameters
    """

    if m_val != K_truncation:
        altspect = np.zeros(K_truncation)
        tailclasses = np.zeros(K_truncation - m_val)
        neutdiff = np.zeros(K_truncation - m_val)
        tailinds = np.arange(m_val + 1, K_truncation + 1)

        for i in range(len(tailinds)):
            ti = tailinds[i]
            denom = K_truncation - m_val - 1
            if denom != 0:
                the_ns = epsilon_max - ((ti - m_val - 1) / denom) * (
                    epsilon_max - epsilon
                )
            else:
                the_ns = epsilon
            tailclasses[i] = the_ns
            neutdiff[i] = K_neutral[ti - 1] - the_ns

        headinds = np.arange(1, m_val + 1)

        for hd in headinds:
            altspect[hd - 1] = K_neutral[hd - 1]

        neutdiff_all = np.sum(neutdiff)

        for ival in headinds:
            # class 3
            # total_exp = np.sum(np.exp(-headinds))
            # theadd = (np.exp(-ival) / total_exp) * neutdiff_all
            # class 5
            theadd = (1 / float(m_val)) * neutdiff_all
            altspect[ival - 1] += theadd

        altspect[m_val:] = tailclasses

        output = easy_likelihood(altspect, K_count, K_truncation)
    else:
        output = easy_likelihood(K_neutral, K_count, K_truncation)

    return output


@njit
def compute_epsilon_values(K_truncation, K_neutral_last):
    epsilon_min = 1 / (K_truncation * 100)
    values = []
    for i in range(1, 101):
        val = i * epsilon_min
        if val <= K_neutral_last:
            values.append(val)
    return np.array(values)


@njit
def T_m_statistic_core(K_counts, K_neutral, windows, K_truncation, sweep_mode=5):
    num_windows = len(windows)
    m_vals = K_truncation + 1
    epsilon_values = compute_epsilon_values(K_truncation, K_neutral[-1])

    # Estimate max rows possible: 1 row per window
    output = np.zeros(
        (num_windows, 6 + len(K_counts[0]))
    )  # 6 meta values + K_iter size

    for j in range(num_windows):
        w = windows[j]
        K_iter = K_counts[j]

        null_likelihood = easy_likelihood(K_neutral, K_iter, K_truncation)

        best_likelihood = -np.inf
        best_m = 0
        best_e = 0.0

        for e in epsilon_values:
            for m in range(1, m_vals):
                alt_like = sweep_likelihood(
                    K_neutral, K_iter, K_truncation, m, e, K_neutral[-1]
                )
                likelihood_diff = 2 * (alt_like - null_likelihood)
                if likelihood_diff > best_likelihood:
                    best_likelihood = likelihood_diff
                    best_m = m
                    best_e = e

        # Build the output row
        output[j, 0] = best_likelihood
        output[j, 1] = best_m
        output[j, 2] = best_e
        output[j, 3] = K_neutral[-1]
        output[j, 4] = sweep_mode
        output[j, 5] = w
        output[j, 6:] = K_iter

    return output


def T_m_statistic_fast(
    K_counts, K_neutral, windows, K_truncation, sweep_mode=5, _iter=0
):
    t_m = T_m_statistic_core(K_counts, K_neutral, windows, K_truncation, sweep_mode)
    stats_schema = {
        "t_statistic": pl.Float64,
        "m": pl.Float64,
        "frequency": pl.Float64,
        "e": pl.Float64,
        "model": pl.Float64,
        "window_lassi": pl.Float64,
    }
    k_schema = {"Kcounts_" + str(i): pl.Float64 for i in range(1, K_truncation + 1)}
    output = pl.DataFrame(
        t_m, schema=pl.Schema({**stats_schema, **k_schema})
    ).with_columns(pl.lit(_iter).cast(pl.Int64).alias("iter"))

    return output


def compute_t_m(
    sim_list,
    K_truncation=5,
    w_size=110,
    step=5,
    K_neutral=None,
    # windows=[10000, 25000, 50000, 100000, 200000],
    windows=[50000, 100000, 200000, 500000, 1000000],
    center=[5e5, 7e5],
    nthreads=1,
    params=None,
    parallel_manager=None,
):
    if parallel_manager is None:
        parallel_manager = Parallel(n_jobs=nthreads, verbose=5)

    hfs_stats = parallel_manager(
        delayed(LASSI_spectrum_and_Kspectrum)(hap_data, K_truncation, w_size, step)
        for _index, (hap_data) in enumerate(sim_list[:], 1)
    )

    K_counts, K_spectrum, windows_lassi = zip(*hfs_stats)

    if K_neutral is None:
        K_neutral = neut_average(np.vstack(K_spectrum))

    t_m = parallel_manager(
        delayed(T_m_statistic_fast)(
            kc, K_neutral, windows_lassi[_iter - 1], K_truncation, _iter=_iter
        )
        for _iter, (kc) in enumerate(K_counts, 1)
    )

    t_m_cut = parallel_manager(
        delayed(cut_t_m_argmax)(t, windows=windows, center=center, _iter=_iter)
        for _iter, t in enumerate(t_m, 1)
    )

    t_m_cut = pl.concat(t_m_cut)
    t_m_cut = t_m_cut.select(
        [
            "iter",
            "window",
            "center",
            *[
                col
                for col in t_m_cut.columns
                if col not in ("iter", "window", "center")
            ],
        ]
    )
    if params is not None:
        t_m_cut = pivot_feature_vectors(
            pl.concat(
                [
                    pl.DataFrame(
                        np.repeat(
                            params,
                            t_m_cut.select(["center", "window"]).unique().shape[0],
                            axis=0,
                        ),
                        schema=["s", "t", "f_i", "f_t"],
                    ),
                    t_m_cut,
                ],
                how="horizontal",
            )
        )

    return t_m_cut, K_neutral


def cut_t_m(df_t_m, windows=[10000, 25000, 50000, 100000, 200000], center=6e5):
    out = []
    for w in windows:
        # for w in [1000000]:
        lower = center - w / 2
        upper = center + w / 2

        df_t_m_subset = df_t_m[
            (df_t_m.iloc[:, 5] > lower) & (df_t_m.iloc[:, 5] < upper)
        ]
        try:
            # max_t = df_t_m_subset.iloc[:, 0].argmax()
            max_t = df_t_m_subset.iloc[:, 1].argmin()
            # df_t_m_subset = df_t_m_subset.iloc[max_t:max_t+1, [0,1,-1]]
            # df_t_m_subset.insert(0,'window',w*2)
            m = df_t_m_subset.m.mode()

            if m.size > 1:
                m = df_t_m_subset.iloc[max_t : max_t + 1, 1]

            out.append(
                pd.DataFrame(
                    {
                        "iter": df_t_m_subset["iter"].unique(),
                        "window": w,
                        "t_statistic": df_t_m_subset.t.mean(),
                        "m": m,
                    }
                )
            )
        except:
            out.append(
                pd.DataFrame(
                    {
                        "iter": df_t_m["iter"].unique(),
                        "window": w,
                        "t_statistic": 0,
                        "m": 0,
                    }
                )
            )

    out = pd.concat(out).reset_index(drop=True)

    return out


def cut_t_m_argmax(
    df_t_m,
    windows=[50000, 100000, 200000, 500000, 1000000],
    # windows=[10000, 25000, 50000, 100000, 200000],
    center=[5e5, 7e5],
    step=1e4,
    _iter=0,
):
    K_names_c = df_t_m.select("^Kcounts_.*$").schema
    t_schema = OrderedDict(
        {
            "t_statistic": pl.Float64,
            "m": pl.Float64,
            **K_names_c,
            "iter": pl.Int64,
            "window": pl.Int64,
            "center": pl.Int64,
        }
    )
    out = []
    centers = np.arange(center[0], center[1] + step, step).astype(int)
    iter_c_w = list(product(centers, windows))
    for c, w in iter_c_w:
        # for w in [1000000]:
        lower = c - w / 2
        upper = c + w / 2

        df_t_m_subset = df_t_m.filter((df_t_m[:, 5] > lower) & (df_t_m[:, 5] < upper))

        try:
            max_t = df_t_m_subset["t_statistic"].arg_max()

            # df_t_m_subset = df_t_m_subset[df_t_m_subset.m > 0]
            # max_t = df_t_m_subset[df_t_m_subset.m > 0].m.argmin()
            df_t_m_subset = df_t_m_subset[max_t : max_t + 1, :]

            df_t_m_subset = df_t_m_subset.select(
                pl.exclude(["frequency", "e", "model", "window_lassi"])
            ).with_columns(
                pl.lit(w).cast(pl.Int64).alias("window"),
                pl.lit(c).cast(pl.Int64).alias("center"),
            )
            out.append(df_t_m_subset)
        except:
            tmp = pl.DataFrame(
                {
                    col: [
                        None
                        if col not in ["iter", "center", "window"]
                        else _iter
                        if col == "iter"
                        else c
                        if col == "center"
                        else w
                    ]
                    for col in t_schema.keys()
                },
                schema=t_schema,
            )
            out.append(tmp)

    out = pl.concat(out)

    return out


def open_pickle(f):
    with open(f, "rb") as handle:
        return pickle.load(handle)


def save_pickle(f, data):
    with open(f, "wb") as handle:
        pickle.dump(data, handle)


##########


# Updated μ_VAR
@njit
def compute_mu_var(start_idx, end_idx, snp_positions, D_ln, W_sz):
    l_start = snp_positions[start_idx]
    l_end = snp_positions[end_idx - 1]
    return ((l_end - l_start) / (D_ln * W_sz)) * snp_positions.shape[0]


@njit
def compute_mu_sfs(window, n, theta_W):
    if window.shape[0] == 0:
        return np.nan
    derived_counts = np.sum(window, axis=1)
    edge_mask = (derived_counts == 1) | (derived_counts == n - 1)
    n_edges = np.sum(edge_mask)
    W_sz = window.shape[0]
    return (n_edges / W_sz) * theta_W


@njit
def compute_mu_ld(start_idx: int, end_idx: int, r2: np.ndarray) -> float:
    """
    Numba‐accelerated version of mu_ld_from_r2matrix_lr.
    Operates entirely via loops over the relevant sub‐blocks.
    """
    length = end_idx - start_idx
    mid = length // 2
    L = mid
    R = length - mid
    base = start_idx

    # 1) within‐left block
    if L > 1:
        sumL = 0.0
        # sum strictly upper‐triangle
        for i in range(L):
            row_i = base + i
            for j in range(i + 1, L):
                col_j = base + j
                sumL += r2[row_i, col_j]
        mean_ld_left = 2.0 * sumL / (L * (L - 1))
    else:
        mean_ld_left = 0.0

    # 2) within‐right block
    if R > 1:
        sumR = 0.0
        for i in range(R):
            row_i = base + mid + i
            for j in range(i + 1, R):
                col_j = base + mid + j
                sumR += r2[row_i, col_j]
        mean_ld_right = 2.0 * sumR / (R * (R - 1))
    else:
        mean_ld_right = 0.0

    # 3) cross‐block
    if L > 0 and R > 0:
        sumC = 0.0
        for i in range(L):
            row_i = base + i
            for j in range(R):
                col_j = base + mid + j
                sumC += r2[row_i, col_j]
        mean_ld_cross = sumC / (L * R)
    else:
        mean_ld_cross = 1e-6

    # 4) final ratio, guarding zero‐division
    num = 0.5 * (mean_ld_left + mean_ld_right)
    den = mean_ld_cross if mean_ld_cross > 0.0 else 1e-6
    return num / den


@njit
def _harmonic_watterson(n):
    w = 0.0
    for i in range(1, n):
        w += 1 / i
    return w


# @njit
def mu_stat(hap, snp_positions, r2_matrix, window_size=50):
    # full chromosome/region length
    D_ln = (snp_positions[-1] + 1) - snp_positions[0]
    S, n = hap.shape

    theta_w_correction = _harmonic_watterson(n)
    # Match RAiSD -w option (default: 50)
    _window_size = window_size

    _iter_windows = list(range(S - _window_size + 1))
    mu_var_np = np.zeros(len(_iter_windows))
    mu_sfs_np = np.zeros(len(_iter_windows))
    mu_ld_np = np.zeros(len(_iter_windows))
    mu_total_np = np.zeros(len(_iter_windows))
    center_np = np.zeros(len(_iter_windows))

    for i in _iter_windows:
        # Indices for this window
        start_idx = i
        end_idx = i + _window_size  # exclusive in Python slicing

        # Window SNP positions (for plotting)
        center_pos = (snp_positions[start_idx] + snp_positions[end_idx - 1]) / 2
        window_positions = snp_positions[start_idx:end_idx]
        window = hap[start_idx:end_idx, :]

        if end_idx <= start_idx or end_idx > hap.shape[0]:
            mu_var_corr[i] = np.nan
            mu_sfs_corr[i] = np.nan
            mu_ld_corr[i] = np.nan
            mu_total_corr[i] = np.nan
            continue

        window = hap[start_idx:end_idx]
        mu_var = compute_mu_var(
            start_idx, end_idx, snp_positions, D_ln, end_idx - start_idx
        )
        mu_sfs = compute_mu_sfs(window, n, theta_w_correction)
        mu_ld = compute_mu_ld(start_idx, end_idx, r2_matrix)
        mu_total = mu_var * mu_sfs * mu_ld

        mu_var_np[i] = mu_var
        mu_sfs_np[i] = mu_sfs
        mu_ld_np[i] = mu_ld
        mu_total_np[i] = mu_total
        center_np[i] = center_pos

    df_mu = pl.DataFrame(
        {
            "positions": center_np.astype(int),
            "mu_var": mu_var_np,
            "mu_sfs": mu_sfs_np,
            "mu_ld": mu_ld_np,
            "mu_total": mu_total_np,
        }
    )

    # return mu_var_np,mu_sfs_np,mu_ld_np,mu_total_np
    return df_mu


################## Sorting


@njit(parallel=False)
def corr_sorting(matrix):
    samples, sites = matrix.shape

    # Step 1: Compute PCC matrix between rows
    PCC = np.zeros((samples, samples), dtype=np.float64)
    sum_pcc = np.zeros(samples, dtype=np.float64)
    P_A = np.zeros(samples, dtype=np.int32)

    for i in range(samples):
        for k in range(sites):
            P_A[i] += matrix[i, k]

    for i in range(samples):
        for k in range(samples):
            if i == k:
                PCC[i, k] = 1.000001
            else:
                P_AB = 0
                for m in range(sites):
                    if matrix[i, m] == 1 and matrix[k, m] == 1:
                        P_AB += 1
                num = (P_AB / sites - (P_A[i] / sites) * (P_A[k] / sites)) ** 2
                den = (
                    (P_A[i] / sites)
                    * (1 - P_A[i] / sites)
                    * (P_A[k] / sites)
                    * (1 - P_A[k] / sites)
                )
                PCC[i, k] = num / den if den != 0 else 0.0
    for i in range(samples):
        for k in range(samples):
            sum_pcc[i] += PCC[i, k]

    # Step 2: Find max PCC sum index
    max_idx = 0
    for i in range(1, samples):
        if sum_pcc[i] > sum_pcc[max_idx]:
            max_idx = i

    # Step 3: Sort rows based on PCC[max_idx] in descending order
    indices = np.arange(samples)
    for m in range(samples):
        for n in range(m + 1, samples):
            if PCC[max_idx, indices[m]] < PCC[max_idx, indices[n]]:
                indices[m], indices[n] = indices[n], indices[m]

    # Step 4: Reorder matrix
    sorted_matrix = np.empty_like(matrix)
    for i in range(samples):
        for j in range(sites):
            sorted_matrix[i, j] = matrix[indices[i], j]

    return sorted_matrix


@njit
def daf_sorting(matrix):
    samples, sites = matrix.shape
    count = np.zeros(sites, dtype=int64)

    # Count number of 1s per column (DAF)
    for m in range(sites):
        for n in range(samples):
            if matrix[n, m] == 1:
                count[m] += 1

    # Bubble sort columns by descending count
    for m in range(sites):
        for n in range(m + 1, sites):
            if count[m] < count[n]:
                # Swap columns m and n
                for k in range(samples):
                    tmp = matrix[k, m]
                    matrix[k, m] = matrix[k, n]
                    matrix[k, n] = tmp
                tmpc = count[m]
                count[m] = count[n]
                count[n] = tmpc

    return matrix


@njit
def freq_sorting(matrix):
    samples, sites = matrix.shape
    weights = np.zeros(samples, dtype=np.int32)

    # Step 1: Count the number of 1s (Hamming weight) per row
    for i in range(samples):
        for j in range(sites):
            if matrix[i, j] == 1:
                weights[i] += 1

    # Step 2: Bubble sort rows by descending Hamming weight
    for i in range(samples - 1):
        for j in range(i + 1, samples):
            if weights[i] < weights[j]:
                # Swap weights
                tmp_w = weights[i]
                weights[i] = weights[j]
                weights[j] = tmp_w

                # Swap rows in matrix
                for k in range(sites):
                    tmp = matrix[i, k]
                    matrix[i, k] = matrix[j, k]
                    matrix[j, k] = tmp

    return matrix


@njit
def pcc_column_sort_numba(matrix):
    n, m = matrix.shape
    PCC_matrix = np.zeros((m, m), dtype=np.float64)
    scores = np.zeros(m, dtype=np.float64)

    # Step 1: Compute PCC_matrix between columns
    for i in range(m):
        for j in range(m):
            if i == j:
                PCC_matrix[i, j] = 1.000001
            else:
                PA_i = 0
                PA_j = 0
                PAB = 0
                for k in range(n):
                    PA_i += matrix[k, i]
                    PA_j += matrix[k, j]
                    PAB += matrix[k, i] * matrix[k, j]

                num = (PAB / n - (PA_i * PA_j) / (n * n)) ** 2
                den = (PA_i / n) * (1 - PA_i / n) * (PA_j / n) * (1 - PA_j / n)
                PCC_matrix[i, j] = num / den if den != 0 else 0.0
    # Step 2: Compute total PCC for each SNP (column)
    for i in range(m):
        for j in range(m):
            scores[i] += PCC_matrix[i, j]

    # Step 3: Bubble sort columns by score (descending)
    for i in range(m - 1):
        for j in range(i + 1, m):
            if scores[i] < scores[j]:
                # Swap scores
                tmp_score = scores[i]
                scores[i] = scores[j]
                scores[j] = tmp_score
                # Swap columns i and j in matrix
                for k in range(n):
                    tmp = matrix[k, i]
                    matrix[k, i] = matrix[k, j]
                    matrix[k, j] = tmp

    return matrix


# def calculate_stats(
#     hap_data,
#     _iter=1,
#     center=[5e5, 7e5],
#     windows=[1000000],
#     step=1e4,
#     neutral=False,
#     region=None,
# ):
#     filterwarnings(
#         "ignore",
#         category=RuntimeWarning,
#         message="invalid value encountered in scalar divide",
#     )
#     np.seterr(divide="ignore", invalid="ignore")

#     if isinstance(hap_data, list) or isinstance(hap_data, tuple):
#         hap, rec_map, p = hap_data
#     elif isinstance(hap_data, str):
#         try:
#             hap, rec_map, p = ms_parser(hap_data)
#         except:
#             try:
#                 hap, rec_map, p = genome_reader(hap_data, region)
#             except:
#                 return None
#     else:
#         return None

#     # Open and filtering data
#     (
#         hap_int,
#         rec_map_01,
#         ac,
#         biallelic_mask,
#         position_masked,
#         genetic_position_masked,
#     ) = filter_gt(hap, rec_map, region=region)
#     freqs = ac[:, 1] / ac.sum(axis=1)

#     if len(center) == 1:
#         centers = np.arange(center[0], center[0] + step, step).astype(int)
#     else:
#         centers = np.arange(center[0], center[1] + step, step).astype(int)

#     df_dind_high_low, df_s_ratio, df_hapdaf_s, df_hapdaf_o = run_fs_stats(
#         hap_int[:, :], ac[:, :], rec_map_01[:, :]
#     )

#     d_stats = defaultdict(dict)

#     df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
#     df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
#     df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
#     df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

#     d_stats = defaultdict(dict)

#     df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
#     df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
#     df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
#     df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

#     d_stats["dind_high_low"] = df_dind_high_low
#     d_stats["s_ratio"] = df_s_ratio
#     d_stats["hapdaf_o"] = df_hapdaf_o
#     d_stats["hapdaf_s"] = df_hapdaf_s

#     try:
#         h12_v = h12_enard(
#             hap_int,
#             position_masked,
#             window_size=int(1e6),
#         )
#         # window_size=int(5e5) if neutral else int(1.2e6)
#     except:
#         h12_v = np.nan

#     haf_v = haf_top(
#         hap_int.astype(np.float64),
#         position_masked,
#         window_size=int(1e6),
#         # window_size=int(5e5) if neutral else int(1.2e6),
#     )

#     daf_w = 1.0
#     pos_w = int(6e5)
#     if 6e5 in position_masked:
#         daf_w = freqs[position_masked == 6e5][0]

#     df_window = pl.DataFrame(
#         {
#             "iter": pl.Series([_iter], dtype=pl.Int64),
#             "center": pl.Series([int(6e5)], dtype=pl.Int64),
#             "window": pl.Series([int(1e6)], dtype=pl.Int64),
#             "positions": pl.Series([pos_w], dtype=pl.Int64),
#             "daf": pl.Series([daf_w], dtype=pl.Float64),
#             "h12": pl.Series([h12_v], dtype=pl.Float64),
#             "haf": pl.Series([haf_v], dtype=pl.Float64),
#         }
#     )

#     d_stats["h12_haf"] = df_window

#     d_centers_stats = defaultdict(dict)
#     schema_center = {
#         "iter": pl.Int64,
#         "center": pl.Int64,
#         "window": pl.Int64,
#         "positions": pl.Int64,
#         "daf": pl.Float64,
#         "ihs": pl.Float64,
#         "delta_ihh": pl.Float64,
#         "isafe": pl.Float64,
#         "nsl": pl.Float64,
#     }

#     for c, w in product(centers, windows):
#         lower = c - w / 2
#         upper = c + w / 2

#         p_mask = (position_masked >= lower) & (position_masked <= upper)
#         p_mask
#         f_mask = freqs >= 0.05

#         # Check whether the hap subset is empty or not
#         if hap_int[p_mask].shape[0] == 0:
#             # df_centers_stats = pl.DataFrame({"iter": _iter,"center": c,"window": w,"positions": np.nan,"daf": np.nan,"isafe": np.nan,"ihs": np.nan,"nsl": np.nan,})
#             d_empty = pl.DataFrame(
#                 [
#                     [_iter],
#                     [c],
#                     [w],
#                     [None],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                 ],
#                 schema=schema_center,
#             )

#             d_centers_stats["ihs"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "ihs", "delta_ihh"]
#             )
#             d_centers_stats["isafe"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "isafe"]
#             )
#             d_centers_stats["nsl"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "nsl"]
#             )
#         else:
#             df_isafe = run_isafe(hap_int[p_mask], position_masked[p_mask])

#             # iHS and nSL
#             df_ihs = run_hapbin(
#                 hap_int[p_mask],
#                 rec_map_01[p_mask],
#                 min_ehh=0.05,
#                 gap_scale=0,
#                 max_extend=0,
#                 _iter=_iter,
#             )
#             # df_ihs = ihs_ihh(hap_int[p_mask],position_masked[p_mask],map_pos=genetic_position_masked[p_mask],min_ehh=0.05,min_maf=0.05,include_edges=False,)

#             nsl_v = nsl(hap_int[(p_mask) & (f_mask)], use_threads=False)
#             df_nsl = pl.DataFrame(
#                 {
#                     "positions": position_masked[(p_mask) & (f_mask)],
#                     "daf": freqs[(p_mask) & (f_mask)],
#                     "nsl": nsl_v,
#                 }
#             ).fill_nan(None)

#             df_isafe = center_window_cols(df_isafe, _iter=_iter, center=c, window=w)
#             df_ihs = center_window_cols(df_ihs, _iter=_iter, center=c, window=w)
#             df_nsl = center_window_cols(df_nsl, _iter=_iter, center=c, window=w)

#             d_centers_stats["ihs"][c] = df_ihs
#             d_centers_stats["isafe"][c] = df_isafe
#             d_centers_stats["nsl"][c] = df_nsl

#     d_stats["ihs"] = pl.concat(d_centers_stats["ihs"].values())
#     d_stats["isafe"] = pl.concat(d_centers_stats["isafe"].values())
#     d_stats["nsl"] = pl.concat(d_centers_stats["nsl"].values())

#     if region is not None:
#         for k, df in d_stats.items():
#             d_stats[k] = df.with_columns(pl.lit(region).alias("iter"))

#     ####### if neutral:
#     # Whole chromosome statistic to normalize

#     df_ihs = run_hapbin(
#         hap_int, rec_map_01, min_ehh=0.1, gap_scale=0, max_extend=0, _iter=_iter
#     )

#     nsl_v = nsl(hap_int[freqs >= 0.05], use_threads=False)

#     df_nsl = pl.DataFrame(
#         {
#             "positions": position_masked[freqs >= 0.05],
#             "daf": freqs[freqs >= 0.05],
#             "nsl": nsl_v,
#         }
#     ).fill_nan(None)

#     df_snps_norm = reduce(
#         lambda left, right: left.join(
#             right, on=["positions", "daf"], how="full", coalesce=True
#         ),
#         [df_nsl, df_ihs, df_isafe],
#     )

#     df_stats = reduce(
#         lambda left, right: left.join(
#             right,
#             on=["iter", "center", "window", "positions", "daf"],
#             how="full",
#             coalesce=True,
#         ),
#         [df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s, df_window],
#     ).sort(["iter", "center", "window", "positions"])
#     df_stats_norm = df_snps_norm.join(
#         df_stats.select(
#             pl.all().exclude(
#                 ["iter", "center", "window", "delta_ihh", "ihs", "isafe", "nsl"]
#             )
#         ),
#         on=["positions", "daf"],
#         how="full",
#         coalesce=True,
#     ).sort(["positions"])

#     df_stats_norm = (
#         df_stats_norm.with_columns(
#             [
#                 pl.lit(_iter).alias("iter"),
#                 pl.lit(600000).alias("center"),
#                 pl.lit(1200000).alias("window"),
#                 pl.col("positions").cast(pl.Int64),
#             ]
#         )
#         .select(
#             pl.col(["iter", "center", "window", "positions"]),
#             pl.all().exclude(["iter", "center", "window", "positions"]),
#         )
#         .sort(["iter", "center", "window", "positions"])
#     )

#     if region is not None:
#         df_stats_norm = df_stats_norm.with_columns(pl.lit(region).alias("iter"))

#     return d_stats, df_stats_norm


# Normalization 270525
# old
# def normalization_raw(
#     stats_values,
#     bins,
#     center=[5e5, 7e5],
#     windows=[50000, 100000, 200000, 500000, 1000000],
#     vcf=False,
#     nthreads=1,
#     parallel_manager=None,
# ):
#     """
#     Normalizes sweep statistics using neutral expectations or optionally using precomputed neutral normalized values.
#     The function applies normalization across different genomic windows and supports multi-threading.

#     Parameters
#     ----------
#     sweeps_stats : namedtuple
#         A Namedtuple containing the statistics for genomic sweeps and sweep parameters across
#         different genomic windows.

#     neutral_stats_norm : namedtuple
#         A Namedtuple containing the statistics for neutral region and neutral parameters across
#         different genomic windows, used as the baselinefor normalizing the sweep statistics.
#         This allows comparison of sweeps against neutral expectations.

#     norm_values : dict or None, optional (default=None)
#         A dictionary of precomputed neutral normalizated values. If provided, these values are
#         used to directly normalize the statistics. If None, the function computes
#         normalization values from the neutral statistics.

#     center : list of float, optional (default=[5e5, 7e5])
#         A list specifying the center positions (in base pairs) for the analysis windows.
#         If a single center value is provided, normalization is centered around that value.
#         Otherwise, it will calculate normalization for a range of positions between the two provided centers.

#     windows : list of int, optional (default=[50000, 100000, 200000, 500000, 1000000])
#         A list of window sizes (in base pairs) for which the normalization will be applied.
#         The function performs normalization for each of the specified window sizes.

#     nthreads : int, optional (default=1)
#         The number of threads to use for parallel processing. If set to 1, the function
#         runs in single-threaded mode. Higher values enable multi-threaded execution for
#         faster computation.

#     Returns
#     -------
#     normalized_stats : pandas.DataFrame
#         A DataFrame containing the normalized sweep statistics across the specified
#         windows and genomic regions. The sweep statistics are scaled relative to
#         neutral expectations.
#     """
#     df_stats, params = stats_values

#     if vcf:
#         df_h12_haf = df_stats.pop("h12_haf")
#         snps_genome = reduce(
#             lambda left, right: left.join(
#                 right,
#                 on=["iter", "center", "window", "positions", "daf"],
#                 how="full",
#                 coalesce=True,
#             ),
#             list(df_stats.values()),
#         ).sort(["iter", "center", "window", "positions"])
#         nchr = snps_genome["iter"].unique().first()

#         # Overwrite snps_values
#         # snps_genome = snps_genome.select(pl.exclude(["h12", "haf"]))
#         snps_values = snps_genome.select(pl.exclude(["h12", "haf"]))

#         stats_names = snps_values.select(snps_values.columns[5:]).columns

#         binned_values = bin_values(snps_values)

#         normalized_df = normalize_snps_statistics(
#             binned_values, bins, stats_names
#         ).sort("positions")

#         df_fv_n = vectorized_cut_vcf(
#             normalized_df, center, windows, stats_names
#         ).fill_nan(None)
#         df_fv_n_raw = vectorized_cut_vcf(
#             snps_genome, center, windows, stats_names
#         ).fill_nan(None)

#         window_values = (
#             df_h12_haf.with_columns(pl.col("positions").alias("iter"))
#             .filter(pl.col("positions").is_in(df_fv_n["iter"].unique().to_numpy()))
#             .select(pl.exclude(["center", "window", "positions", "daf"]))
#         )
#         df_fv_n = df_fv_n.join(
#             window_values,
#             on=["iter"],
#             how="full",
#             coalesce=True,
#         )
#         df_fv_n_raw = df_fv_n_raw.join(
#             window_values,
#             on=["iter"],
#             how="full",
#             coalesce=True,
#         )

#         df_params_unpack = pl.DataFrame(
#             np.repeat(
#                 np.zeros((1, 4)),
#                 df_fv_n.shape[0],
#                 axis=0,
#             ),
#             schema=["s", "t", "f_i", "f_t"],
#         )
#         df_fv_n = df_fv_n.with_columns(
#             pl.lit(snps_genome["iter"].unique().first()).alias("nchr")
#         )
#         df_fv_n_raw = df_fv_n_raw.with_columns(
#             pl.lit(snps_genome["iter"].unique().first()).alias("nchr")
#         )
#     else:
#         # df_stats_unzipped = dict(zip(
#         #     df_stats[0].keys(),
#         #     zip(*(d.values() for d in df_stats))
#         # ))

#         # df_stats_unzipped = {i:pl.concat(v) for i,v in df_stats_unzipped.items()}

#         # df_fv_n_l = []
#         # df_fv_n_raw_l = []
#         # for k,snps_genome in tqdm(df_stats_unzipped.items()):
#         #     if k == 'h12_haf':
#         #         continue
#         #     snps_values = snps_genome.select(pl.exclude(["h12", "haf"]))
#         #     binned_values = bin_values(snps_values)
#         #     stats_names = snps_genome.select(snps_genome.columns[5:]).columns

#         #     normalized_df = normalize_snps_statistics(binned_values, bins, stats_names)
#         #     normalized_df = normalized_df.with_columns(pl.lit(snps_genome['iter']))

#         #     df_fv_n.append(vectorized_cut(normalized_df, center, windows, stats_names).fill_nan(None))
#         #     df_fv_n_raw_l.append(vectorized_cut(snps_genome, center, windows, stats_names).fill_nan(None))

#         # df_fv_n = reduce(
#         #     lambda left, right: left.join(
#         #         right,
#         #         on=["iter", "center", "window"],
#         #         how="full",
#         #         coalesce=True,
#         #     ),
#         #     df_fv_n_l,
#         # ).sort(["iter", "center", "window"])

#         # df_fv_n_raw = reduce(
#         #     lambda left, right: left.join(
#         #         right,
#         #         on=["iter", "center", "window"],
#         #         how="full",
#         #         coalesce=True,
#         #     ),
#         #     df_fv_n_raw_l,
#         # ).sort(["iter", "center", "window"])

#         if parallel_manager is None:
#             df_fv_n_l, df_fv_n_l_raw = zip(
#                 *Parallel(n_jobs=nthreads, verbose=1)(
#                     delayed(normalize_cut_raw)(
#                         snps_values, bins, center=center, windows=windows
#                     )
#                     for _iter, snps_values in enumerate(df_stats, 1)
#                 )
#             )
#         else:
#             df_fv_n_l, df_fv_n_l_raw = zip(
#                 *parallel_manager(
#                     delayed(normalize_cut_raw)(
#                         snps_values, bins, center=center, windows=windows
#                     )
#                     for _iter, snps_values in enumerate(df_stats, 1)
#                 )
#             )

#         df_fv_n = pl.concat(df_fv_n_l).with_columns(
#             pl.col(["iter", "window", "center"]).cast(pl.Int64)
#         )
#         df_fv_n_raw = pl.concat(df_fv_n_l_raw).with_columns(
#             pl.col(["iter", "window", "center"]).cast(pl.Int64)
#         )

#         df_window = pl.concat([df["h12_haf"] for df in df_stats]).select(
#             pl.exclude(["center", "window", "positions", "daf"])
#         )
#         df_fv_n = df_fv_n.join(df_window, on=["iter"], how="full", coalesce=True)
#         df_fv_n_raw = df_fv_n_raw.join(
#             df_window, on=["iter"], how="full", coalesce=True
#         )

#         # params = params[:, [0, 1, 3, 4, ]]
#         df_params_unpack = pl.DataFrame(
#             np.repeat(
#                 params,
#                 df_fv_n.select(["center", "window"])
#                 .unique()
#                 .sort(["center", "window"])
#                 .shape[0],
#                 axis=0,
#             ),
#             schema=["s", "t", "f_i", "f_t"],
#         )

#     df_fv_n = pl.concat(
#         [df_params_unpack, df_fv_n],
#         how="horizontal",
#     )
#     df_fv_n_raw = pl.concat(
#         [df_params_unpack, df_fv_n_raw],
#         how="horizontal",
#     )

#     force_order = ["iter"] + [col for col in df_fv_n.columns if col != "iter"]
#     df_fv_n = df_fv_n.select(force_order)
#     df_fv_n_raw = df_fv_n_raw.select(force_order)

#     return df_fv_n, df_fv_n_raw


# # old
# def normalize_cut_raw(
#     snps_values,
#     bins,
#     center=[5e5, 7e5],
#     windows=[50000, 100000, 200000, 500000, 1000000],
#     step=int(1e4),
# ):
#     """
#     Normalizes SNP-level statistics by comparing them to neutral expectations, and aggregates
#     the statistics within sliding windows around specified genomic centers.

#     This function takes SNP statistics, normalizes them based on the expected mean and standard
#     deviation from neutral simulations, and computes the average values within windows
#     centered on specific genomic positions. It returns a DataFrame with the normalized values
#     for each window across the genome.

#     Parameters
#     ----------
#     _iter : int
#         The iteration or replicate number associated with the current set of SNP statistics.

#     snps_values : pandas.DataFrame
#         A DataFrame containing SNP-level statistics such as iHS, nSL, and iSAFE. The DataFrame
#         should contain derived allele frequencies ("daf") and other statistics to be normalized.

#     expected : pandas.DataFrame
#         A DataFrame containing the expected mean values of the SNP statistics for each frequency bin,
#         computed from neutral simulations.

#     stdev : pandas.DataFrame
#         A DataFrame containing the standard deviation of the SNP statistics for each frequency bin,
#         computed from neutral simulations.

#     center : list of float, optional (default=[5e5, 7e5])
#         A list specifying the center positions (in base pairs) for the analysis. Normalization is
#         performed around these genomic centers using the specified window sizes.

#     windows : list of int, optional (default=[50000, 100000, 200000, 500000, 1000000])
#         A list of window sizes (in base pairs) over which the SNP statistics will be aggregated
#         and normalized. The function performs normalization for each specified window size.

#     Returns
#     -------
#     out : pandas.DataFrame
#         A DataFrame containing the normalized SNP statistics for each genomic center and window.
#         The columns include the iteration number, center, window size, and the average values
#         of normalized statistics iSAFE within the window.

#     Notes
#     -----
#     - The function first bins the SNP statistics based on derived allele frequencies using the
#       `bin_values` function. The statistics are then normalized by subtracting the expected mean
#       and dividing by the standard deviation for each frequency bin.
#     - After normalization, SNPs are aggregated into windows centered on specified genomic positions.
#       The average values of the normalized statistics are calculated for each window.
#     - The window size determines how far upstream and downstream of the center position the SNPs
#       will be aggregated.

#     """
#     df_out = []
#     df_out_raw = []
#     _iter = np.unique([i.select("iter").unique() for i in snps_values.values()])

#     if len(center) == 2:
#         centers = np.arange(center[0], center[1] + step, step).astype(int)
#     else:
#         centers = center

#     for k, df in snps_values.items():
#         if k == "h12_haf":
#             continue

#         stats_names = df.select(df.columns[5:]).columns

#         binned_values = bin_values(df)
#         normalized_df = normalize_snps_statistics(binned_values, bins, stats_names)

#         fixed_center = int(6e5) if k not in ["isafe", "ihs", "nsl"] else None

#         df_out.append(
#             cut_snps(
#                 normalized_df,
#                 centers,
#                 windows,
#                 stats_names,
#                 fixed_center=None,
#                 iter_value=_iter,
#             )
#         )
#         df_out_raw.append(
#             cut_snps(
#                 df,
#                 centers,
#                 windows,
#                 stats_names,
#                 fixed_center=None,
#                 iter_value=_iter,
#             )
#         )

#     df_out = reduce(
#         lambda left, right: left.join(
#             right,
#             on=["iter", "center", "window"],
#             how="full",
#             coalesce=True,
#         ),
#         df_out,
#     ).sort(["iter", "center", "window"])

#     df_out_raw = reduce(
#         lambda left, right: left.join(
#             right,
#             on=["iter", "center", "window"],
#             how="full",
#             coalesce=True,
#         ),
#         df_out_raw,
#     ).sort(["iter", "center", "window"])

#     return df_out, df_out_raw


# # old
# def cut_snps(
#     normalized_df, centers, windows, stats_names, fixed_center=None, iter_value=1
# ):
#     """
#     Processes data within windows across multiple centers and window sizes.

#     Parameters
#     ----------
#     normalized_df : polars.DataFrame
#         DataFrame containing the positions and statistics.
#     iter_value : int
#         Iteration or replicate number.
#     centers : list
#         List of center positions to analyze.
#     windows : list
#         List of window sizes to use.
#     stats_names : list, optional
#         Names of statistical columns to compute means for.
#         If None, all columns except position-related ones will be used.
#     position_col : str, optional
#         Name of the column containing position values.
#     center_col : str, optional
#         Name of the column containing center values.
#     fixed_center : int, optional
#         If provided, use this fixed center value instead of the ones in centers list.

#     Returns
#     -------
#     polars.DataFrame
#         DataFrame with aggregated statistics for each center and window.
#     """
#     # If stats_names not provided, use all appropriate columns

#     results = []
#     for c, w in list(product(centers, windows)):
#         # Use fixed center if provided
#         c_fix = fixed_center if fixed_center is not None else c

#         # Define window boundaries
#         lower = c - w // 2
#         upper = c + w // 2

#         # Filter data by center and window boundaries
#         query = normalized_df.lazy()

#         # 1.2MB simulations derives into 21 center/windows combinations
#         if centers.size > 21:
#             query = query.filter(
#                 (pl.col("positions") >= c - int(6e5))
#                 & (pl.col("positions") <= c + int(6e5))
#             )
#         else:
#             query = query.filter(pl.col("center") == c_fix)

#         window_data = query.filter(
#             (pl.col("positions") >= lower) & (pl.col("positions") <= upper)
#         ).collect()

#         window_data = normalized_df.filter(
#             (pl.col("positions") >= lower) & (pl.col("positions") <= upper)
#         )

#         # Skip if no data in this window
#         # if window_data.height == 0:
#         #     continue

#         # Calculate mean statistics for window
#         window_stats = window_data.select(stats_names).fill_nan(None).mean()

#         # Add metadata columns
#         metadata_cols = [
#             pl.lit(iter_value).alias("iter"),
#             pl.lit(c).alias("center"),
#             pl.lit(w).alias("window"),
#         ]

#         results.append(window_stats.with_columns(metadata_cols))

#     return (
#         pl.concat(results, how="vertical").select(
#             ["iter", "center", "window"] + stats_names
#         )
#         if results
#         else None
#     )


#########################
# def calculate_stats(
#     hap_data,
#     _iter=1,
#     center=[5e5, 7e5],
#     windows=[1000000],
#     step=1e4,
#     neutral=False,
#     region=None,
# ):
#     filterwarnings(
#         "ignore",
#         category=RuntimeWarning,
#         message="invalid value encountered in scalar divide",
#     )
#     np.seterr(divide="ignore", invalid="ignore")

#     if isinstance(hap_data, list) or isinstance(hap_data, tuple):
#         hap, rec_map, p = hap_data
#     elif isinstance(hap_data, str):
#         try:
#             hap, rec_map, p = ms_parser(hap_data)
#         except:
#             try:
#                 hap, rec_map, p = genome_reader(hap_data, region)
#             except:
#                 return None
#     else:
#         return None

#     # Open and filtering data
#     (
#         hap_int,
#         rec_map_01,
#         ac,
#         biallelic_mask,
#         position_masked,
#         genetic_position_masked,
#     ) = filter_gt(hap, rec_map, region=region)
#     freqs = ac[:, 1] / ac.sum(axis=1)

#     if len(center) == 1:
#         centers = np.arange(center[0], center[0] + step, step).astype(int)
#     else:
#         centers = np.arange(center[0], center[1] + step, step).astype(int)

#     # df_dind_high_low = dind_high_low(hap_int, ac, rec_map_01)
#     # df_s_ratio = s_ratio(hap_int, ac, rec_map_01)
#     # df_hapdaf_o = hapdaf_o(hap_int, ac, rec_map_01)
#     # df_hapdaf_s = hapdaf_s(hap_int, ac, rec_map_01)
#     df_dind_high_low, df_s_ratio, df_hapdaf_s, df_hapdaf_o = run_fs_stats(
#         hap_int[:, :], ac[:, :], rec_map_01[:, :]
#     )

#     d_stats = defaultdict(dict)

#     df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
#     df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
#     df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
#     df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

#     d_stats = defaultdict(dict)

#     df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
#     df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
#     df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
#     df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

#     d_stats["dind_high_low"] = df_dind_high_low
#     d_stats["s_ratio"] = df_s_ratio
#     d_stats["hapdaf_o"] = df_hapdaf_o
#     d_stats["hapdaf_s"] = df_hapdaf_s

#     try:
#         h12_v = h12_enard(
#             hap_int, position_masked, window_size=int(5e5) if neutral else int(1.2e6)
#         )
#     except:
#         h12_v = np.nan

#     haf_v = haf_top(
#         hap_int.astype(np.float64),
#         position_masked,
#         window_size=int(5e5) if neutral else int(1.2e6),
#     )

#     daf_w = 1.0
#     pos_w = int(6e5)
#     if 6e5 in position_masked:
#         daf_w = freqs[position_masked == 6e5][0]

#     df_window = pl.DataFrame(
#         {
#             "iter": pl.Series([_iter], dtype=pl.Int64),
#             "center": pl.Series([int(6e5)], dtype=pl.Int64),
#             "window": pl.Series([int(1e6)], dtype=pl.Int64),
#             "positions": pl.Series([pos_w], dtype=pl.Int64),
#             "daf": pl.Series([daf_w], dtype=pl.Float64),
#             "h12": pl.Series([h12_v], dtype=pl.Float64),
#             "haf": pl.Series([haf_v], dtype=pl.Float64),
#         }
#     )

#     d_stats["h12_haf"] = df_window

#     d_centers_stats = defaultdict(dict)
#     schema_center = {
#         "iter": pl.Int64,
#         "center": pl.Int64,
#         "window": pl.Int64,
#         "positions": pl.Int64,
#         "daf": pl.Float64,
#         "ihs": pl.Float64,
#         "delta_ihh": pl.Float64,
#         "isafe": pl.Float64,
#         "nsl": pl.Float64,
#     }

#     for c, w in product(centers, windows):
#         lower = c - w / 2
#         upper = c + w / 2

#         p_mask = (position_masked >= lower) & (position_masked <= upper)
#         p_mask
#         f_mask = freqs >= 0.05

#         # Check whether the hap subset is empty or not
#         if hap_int[p_mask].shape[0] == 0:
#             # df_centers_stats = pl.DataFrame({"iter": _iter,"center": c,"window": w,"positions": np.nan,"daf": np.nan,"isafe": np.nan,"ihs": np.nan,"nsl": np.nan,})
#             d_empty = pl.DataFrame(
#                 [
#                     [_iter],
#                     [c],
#                     [w],
#                     [None],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                 ],
#                 schema=schema_center,
#             )

#             d_centers_stats["ihs"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "ihs", "delta_ihh"]
#             )
#             d_centers_stats["isafe"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "isafe"]
#             )
#             d_centers_stats["nsl"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "nsl"]
#             )
#         else:
#             df_isafe = run_isafe(hap_int[p_mask], position_masked[p_mask])

#             # iHS and nSL
#             df_ihs = run_hapbin(
#                 hap_int[p_mask], rec_map_01[p_mask], gap_scale=0, max_extend=0
#             )
#             # df_ihs = ihs_ihh(hap_int[p_mask],position_masked[p_mask],map_pos=genetic_position_masked[p_mask],min_ehh=0.05,min_maf=0.05,include_edges=False,)

#             nsl_v = nsl(hap_int[(p_mask) & (f_mask)], use_threads=False)
#             df_nsl = pl.DataFrame(
#                 {
#                     "positions": position_masked[(p_mask) & (f_mask)],
#                     "daf": freqs[(p_mask) & (f_mask)],
#                     "nsl": nsl_v,
#                 }
#             ).fill_nan(None)

#             df_isafe = center_window_cols(df_isafe, _iter=_iter, center=c, window=w)
#             df_ihs = center_window_cols(df_ihs, _iter=_iter, center=c, window=w)
#             df_nsl = center_window_cols(df_nsl, _iter=_iter, center=c, window=w)

#             d_centers_stats["ihs"][c] = df_ihs
#             d_centers_stats["isafe"][c] = df_isafe
#             d_centers_stats["nsl"][c] = df_nsl

#     d_stats["ihs"] = pl.concat(d_centers_stats["ihs"].values())
#     d_stats["isafe"] = pl.concat(d_centers_stats["isafe"].values())
#     d_stats["nsl"] = pl.concat(d_centers_stats["nsl"].values())

#     if region is not None:
#         for k, df in d_stats.items():
#             d_stats[k] = df.with_columns(pl.lit(region).alias("iter"))

#     ####### if neutral:
#     # Whole chromosome statistic to normalize
#     df_isafe = run_isafe(hap_int, position_masked)
#     df_ihs = ihs_ihh(hap_int, position_masked, min_ehh=0.1, include_edges=True)

#     nsl_v = nsl(hap_int[freqs >= 0.05], use_threads=False)

#     df_nsl = pl.DataFrame(
#         {
#             "positions": position_masked[freqs >= 0.05],
#             "daf": freqs[freqs >= 0.05],
#             "nsl": nsl_v,
#         }
#     ).fill_nan(None)

#     df_snps_norm = reduce(
#         lambda left, right: left.join(
#             right, on=["positions", "daf"], how="full", coalesce=True
#         ),
#         [df_nsl, df_ihs, df_isafe],
#     )

#     df_stats = reduce(
#         lambda left, right: left.join(
#             right,
#             on=["iter", "center", "window", "positions", "daf"],
#             how="full",
#             coalesce=True,
#         ),
#         [df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s, df_window],
#     ).sort(["iter", "center", "window", "positions"])
#     df_stats_norm = df_snps_norm.join(
#         df_stats.select(
#             pl.all().exclude(
#                 ["iter", "center", "window", "delta_ihh", "ihs", "isafe", "nsl"]
#             )
#         ),
#         on=["positions", "daf"],
#         how="full",
#         coalesce=True,
#     ).sort(["positions"])

#     df_stats_norm = (
#         df_stats_norm.with_columns(
#             [
#                 pl.lit(_iter).alias("iter"),
#                 pl.lit(600000).alias("center"),
#                 pl.lit(1200000).alias("window"),
#                 pl.col("positions").cast(pl.Int64),
#             ]
#         )
#         .select(
#             pl.col(["iter", "center", "window", "positions"]),
#             pl.all().exclude(["iter", "center", "window", "positions"]),
#         )
#         .sort(["iter", "center", "window", "positions"])
#     )

#     if region is not None:
#         df_stats_norm = df_stats_norm.with_columns(pl.lit(region).alias("iter"))

#     return d_stats, df_stats_norm


# def run_haplosweep(
#     hap,
#     rec_map,
#     _iter=1,
#     min_ehh=0.05,
#     min_maf=0.05,
#     gap_scale=20000,
#     max_extend=1000000,
#     haplosweep="/home/jmurgamoreno/software/HaploSweep/bin/HaploSweep calc",
# ):
#     df_hap = pl.DataFrame(hap).transpose()
#     df_rec_map = pl.DataFrame(
#         rec_map,
#         pl.Schema(
#             [
#                 ("chr", pl.Int64),
#                 ("location", pl.Int64),
#                 ("cm_mb", pl.Float64),
#                 ("cm", pl.Float64),
#             ]
#         ),
#     )
#     hap_file = "/tmp/tmp_" + str(_iter) + ".hap"
#     map_file = "/tmp/tmp_" + str(_iter) + ".map"

#     df_hap.write_csv(hap_file, separator=" ", include_header=False)

#     df_rec_map.write_csv(map_file, include_header=False, separator=" ")

#     out_file = f"/tmp/tmp_{_iter}.out"
#     cmd_hapbin = f"{haplosweep} -i {hap_file} -m {map_file} -t 1 -cutoff {min_ehh} -maf {min_maf} -o1 {out_file}"

#     if gap_scale > 0:
#         cmd_hapbin += f" -gap-scale {gap_scale}"

#     if max_extend > 0:
#         cmd_hapbin += f" -e {max_extend}"

#     ihs_output = subprocess.run(cmd_hapbin, shell=True, capture_output=True, text=True)

#     df_ihs = pl.read_csv(
#         out_file,
#         separator="\t",
#         schema=pl.Schema(
#             [
#                 ("locusid", pl.Int64),
#                 ("chr", pl.Int64),
#                 ("positions", pl.Int64),
#                 ("daf", pl.Float64),
#                 ("ihh1", pl.Float64),
#                 ("ihh0", pl.Float64),
#                 ("ihhl1", pl.Float64),
#                 ("ihhl0", pl.Float64),
#                 ("ihs", pl.Float64),
#                 ("urihs1", pl.Float64),
#                 ("urihs0", pl.Float64),
#                 ("ihsl", pl.Float64),
#             ]
#         ),
#     ).select("positions", "daf", "ihs", "urihs1", "urihs0", "ihsl")

#     os.remove(hap_file)
#     os.remove(map_file)
#     os.remove(out_file)

#     # normalize and estimate rihsl
#     # rihsl = ihsl**2 + (urihs1 if ihsl > 0 else urihs0)**2

#     return df_ihs


# def mispolarize(hap, proportion=0.1):
#     """
#     Allele mispolarization by randomly flipping the alleles of a haplotype matrix (i.e., switching between 0 and 1). The proportion of rows to be flipped is determined by the `proportion` parameter.

#     Parameters
#     ----------
#     hap : numpy.ndarray
#         A 2D numpy array representing the haplotype matrix of shape (S, n),
#         where S is the number of variants (rows), and n is the number of samples (columns).
#         Each element is expected to be binary (0 or 1), representing the alleles.

#     proportion : float, optional (default=0.1)
#         A float between 0 and 1 specifying the proportion of rows (loci) in the haplotype
#         matrix to randomly flip. For example, if proportion=0.1, 10% of the rows in the
#         haplotype matrix will have their allele values flipped.

#     Returns
#     -------
#     hap_copy : numpy.ndarray
#         A new 2D numpy array of the same shape as `hap`, with a proportion of rows
#         randomly flipped (alleles inverted). The original matrix `hap` is not modified
#         in-place.

#     Notes
#     -----
#     The flipping operation is done using a bitwise XOR operation (`^= 1`), which
#     efficiently flips 0 to 1 and 1 to 0 for the selected rows.

#     """
#     # Get shape of haplotype matrix
#     S, n = hap.shape

#     # Select the column indices to flip based on the given proportion
#     to_flip = np.random.choice(np.arange(S), int(S * proportion), replace=False)

#     # Create a copy of the original hap matrix to avoid in-place modification
#     hap_copy = hap.copy()
#     hap_copy[to_flip, :] ^= 1
#     return hap_copy


# def filter_gt(hap, rec_map, region=None):
#     """
#     Convert the 2d numpy haplotype matrix to HaplotypeArray from scikit-allel and change type np.int8. It filters and processes the haplotype matrix based on a recombination map and
#     returns key information for further analysis such as allele frequencies and physical positions.

#     Parameters
#     ----------
#     hap : array-like, HaplotypeArray
#         The input haplotype data which can be in one of the following forms:
#         - A `HaplotypeArray` object.
#         - A genotype matrix (as a numpy array or similar).

#     rec_map : numpy.ndarray
#         A 2D numpy array representing the recombination map, where each row corresponds
#         to a genomic variant and contains recombination information. The third column (index 2)
#         of the recombination map provides the physical positions of the variants.

#     Returns
#     -------
#     tuple
#         A tuple containing the following elements:
#         - hap_01 (numpy.ndarray): The filtered haplotype matrix, with only biallelic variants.
#         - ac (AlleleCounts): An object that stores allele counts for the filtered variants.
#         - biallelic_mask (numpy.ndarray): A boolean mask indicating which variants are biallelic.
#         - hap_int (numpy.ndarray): The haplotype matrix converted to integer format (int8).
#         - rec_map_01 (numpy.ndarray): The recombination map filtered to include only biallelic variants.
#         - position_masked (numpy.ndarray): The physical positions of the biallelic variants.
#         - sequence_length (int): An arbitrary sequence length set to 1.2 million bases (1.2e6).
#         - freqs (numpy.ndarray): The frequencies of the alternate alleles for each biallelic variant.
#     """
#     try:
#         hap = HaplotypeArray(hap.genotype_matrix())
#     except:
#         try:
#             hap = HaplotypeArray(hap)
#         except:
#             hap = HaplotypeArray(load(hap).genotype_matrix())

#     # positions = rec_map[:, -1]
#     # physical_position = rec_map[:, -2]

#     # HAP matrix centered to analyse whole chromosome
#     hap_01, ac, biallelic_mask = filter_biallelics(hap)
#     hap_int = hap_01.astype(np.int8)
#     rec_map_01 = rec_map[biallelic_mask]
#     sequence_length = int(1.2e6)
#     freqs = ac.to_frequencies()[:, 1]

#     if region is not None:
#         tmp = list(map(int, region.split(":")[-1].split("-")))
#         d_pos = dict(
#             zip(np.arange(tmp[0], tmp[1] + 1), np.arange(1, sequence_length + 1))
#         )
#         for r in rec_map_01:
#             r[-1] = d_pos[r[-1]]

#     position_masked = rec_map_01[:, -1]
#     genetic_position_masked = rec_map_01[:, -2]

#     return (
#         hap_01,
#         ac,
#         biallelic_mask,
#         hap_int,
#         rec_map_01,
#         position_masked,
#         genetic_position_masked,
#         sequence_length,
#         freqs,
#     )


# def filter_biallelics(hap: HaplotypeArray) -> tuple:
#     """
#     Filter out non-biallelic loci from the haplotype data.

#     Args: hap (allel.HaplotypeArray): Haplotype data represented as a HaplotypeArray.

#     Returns:tuple: A tuple containing three elements:
#         - hap_biallelic (allel.HaplotypeArray): Filtered biallelic haplotype data.
#         - ac_biallelic (numpy.ndarray): Allele counts for the biallelic loci.
#         - biallelic_mask (numpy.ndarray): Boolean mask indicating biallelic loci.
#     """
#     ac = hap.count_alleles()
#     biallelic_mask = ac.is_biallelic_01()
#     return (hap.subset(biallelic_mask), ac[biallelic_mask, :], biallelic_mask)


# def normalize_cut_vcf(
#     snps_values,
#     bins,
#     center,
#     windows=[50000, 100000, 200000, 500000, 1000000],
#     nthreads=1,
#     parallel_manager=None,
# ):
#     """
#     Normalizes SNP-level statistics by comparing them to neutral expectations, and aggregates
#     the statistics within sliding windows around specified genomic centers.

#     This function takes SNP statistics, normalizes them based on the expected mean and standard
#     deviation from neutral simulations, and computes the average values within windows
#     centered on specific genomic positions. It returns a DataFrame with the normalized values
#     for each window across the genome.

#     Parameters
#     ----------
#     _iter : int
#         The iteration or replicate number associated with the current set of SNP statistics.

#     snps_values : pandas.DataFrame
#         A DataFrame containing SNP-level statistics such as iHS, nSL, and iSAFE. The DataFrame
#         should contain derived allele frequencies ("daf") and other statistics to be normalized.

#     expected : pandas.DataFrame
#         A DataFrame containing the expected mean values of the SNP statistics for each frequency bin,
#         computed from neutral simulations.

#     stdev : pandas.DataFrame
#         A DataFrame containing the standard deviation of the SNP statistics for each frequency bin,
#         computed from neutral simulations.

#     center : list of float, optional (default=[5e5, 7e5])
#         A list specifying the center positions (in base pairs) for the analysis. Normalization is
#         performed around these genomic centers using the specified window sizes.

#     windows : list of int, optional (default=[50000, 100000, 200000, 500000, 1000000])
#         A list of window sizes (in base pairs) over which the SNP statistics will be aggregated
#         and normalized. The function performs normalization for each specified window size.

#     Returns
#     -------
#     out : pandas.DataFrame
#         A DataFrame containing the normalized SNP statistics for each genomic center and window.
#         The columns include the iteration number, center, window size, and the average values
#         of normalized statistics iSAFE within the window.

#     Notes
#     -----
#     - The function first bins the SNP statistics based on derived allele frequencies using the
#       `bin_values` function. The statistics are then normalized by subtracting the expected mean
#       and dividing by the standard deviation for each frequency bin.
#     - After normalization, SNPs are aggregated into windows centered on specified genomic positions.
#       The average values of the normalized statistics are calculated for each window.
#     - The window size determines how far upstream and downstream of the center position the SNPs
#       will be aggregated.

#     """

#     # Overwrite snps_values
#     snps_values = snps_values.select(pl.exclude(["h12", "haf"]))

#     stats_names = snps_values.select(snps_values.columns[5:]).columns

#     binned_values = bin_values(snps_values)

#     normalized_df = normalize_snps_statistics(binned_values, bins, stats_names).sort(
#         "positions"
#     )

#     df_out = cut_vcf(normalized_df, center, windows, stats_names)
#     df_out_raw = cut_vcf(snps_values, center, windows, stats_names)

#     df_out = df_out.with_columns(
#         (
#             pl.lit(snps_values["iter"].unique())
#             + ":"
#             + (pl.col("iter") - int(6e5)).cast(pl.String)
#             + "-"
#             + (pl.col("iter") + int(6e5)).cast(pl.String)
#         ).alias("iter")
#     )
#     df_out_raw = df_out_raw.with_columns(df_out["iter"].alias("iter"))
#     return df_out, df_out_raw


# def slice_window(c, df):
#     positions = df["positions"].to_numpy()

#     lower = c - int(6e5)
#     upper = c + int(6e5)

#     start_idx = np.searchsorted(positions, lower, side="left")
#     end_idx = np.searchsorted(positions, upper, side="right")

#     return df.slice(start_idx, end_idx - start_idx)


# def vectorized_cut(normalized_df, center, windows, stats_names):
#     grouped = normalized_df.group_by("iter", maintain_order=True)

#     results = []

#     for group in grouped:
#         results.append(
#             vectorized_cut_vcf(group[1], center, windows, stats_names).with_columns(
#                 pl.lit(group[0][0]).alias("iter")
#             )
#         )

#     return pl.concat(results)


# def cut_vcf(normalized_df, center, windows, stats_names):
#     positions = normalized_df["positions"].to_numpy()

#     lower = center - int(6e5)
#     upper = center + int(6e5)

#     start_idx = np.searchsorted(positions, lower, side="left")
#     end_idx = np.searchsorted(positions, upper, side="right")

#     normalized_df = normalized_df.slice(start_idx, end_idx - start_idx)

#     results = []

#     # Slice the data around the center position using slice_window
#     inner_center = np.linspace(center - 1e5, center + 1e5, 21).astype(int)
#     results = []
#     for c, w in list(product(inner_center, windows)):
#         # Define window boundaries
#         lower = c - w // 2
#         upper = c + w // 2

#         # Filter data by center and window boundaries
#         window_data = normalized_df.filter(
#             (pl.col("positions") >= lower) & (pl.col("positions") <= upper)
#         )

#         # Calculate mean statistics for window
#         window_stats = window_data.select(stats_names).fill_nan(None).mean()

#         # Add metadata columns
#         metadata_cols = [
#             pl.lit(center).alias("iter").cast(pl.Int64),
#             pl.lit(c).alias("center").cast(pl.Int64),
#             pl.lit(w).alias("window").cast(pl.Int64),
#         ]

#         results.append(window_stats.with_columns(metadata_cols))

#     return (
#         pl.concat(results, how="vertical").select(
#             ["iter", "center", "window"] + stats_names
#         )
#         if results
#         else None
#     )


# def summary_statistics_v2(
#     data_dir,
#     vcf=False,
#     nthreads=1,
#     center=[500000, 700000],
#     windows=[1000000],
#     step=10000,
#     recombination_map=None,
# ):
#     """
#     Computes summary statistics across multiple simulations or empirical data, potentially using
#     multiple threads for parallel computation. The statistics are calculated over
#     defined genomic windows, with optional mispolarization applied to the haplotype data.
#     Save the dataframe to a parquet file


#     Only iHS, nSL and iSAFE are estimated across all windows/center combination. The other
#     statistics used the actual center (1.2e6 / 2) extended to 500Kb each flank.

#     Parameters
#     ----------
#     sims : str,
#         Discoal simulation path or VCF file. If VCF file ensure you're use `vcf=True` argument.
#     nthreads : int, optional (default=1)
#         The number of threads to use for parallel computation. If set to 1,
#         the function runs in single-threaded mode. Higher values will enable
#         multi-threaded processing to speed up calculations.
#     center : list of int, optional (default=[500000, 700000])
#         A list specifying the center positions (in base pairs) for the analysis windows.
#         If one center is provided, it will use that as a single point; otherwise,
#         the analysis will cover the range between the two provided centers.

#     windows : list of int, optional (default=[1000000])
#         A list of window sizes (in base pairs) over which summary statistics will be computed.

#     step : int, optional (default=10000)
#         The step size (in base pairs) for sliding windows in the analysis. This determines
#         how much the analysis window moves along the genome for each iteration.
#     vcf : bool,
#         If true parse vcf

#     Returns
#     -------
#     summary_stats : pandas.DataFrame
#         A DataFrame containing the computed summary statistics for each simulation and
#         for each genomic window.

#     """
#     # Validate data directories

#     fvs_file = defaultdict(str)
#     regions = defaultdict()

#     if vcf:
#         required_folders = ["vcfs"]
#         sims = defaultdict()
#         neutral_save = f"{data_dir}/empirical_bins.pickle"
#         df_params = []
#         # Process VCF files
#         vcf_files = np.sort(glob.glob(f"{data_dir}/vcfs/*vcf.gz"))

#         for vcf_path in vcf_files:
#             # Process each VCF file
#             fs_data = Data(vcf_path, nthreads=nthreads)
#             _sims = fs_data.read_vcf()

#             # Same folder custom fvs name based on input VCF.
#             f_name = os.path.basename(vcf_path)
#             for ext in [".vcf", ".bcf", ".gz"]:
#                 f_name = f_name.replace(ext, "")
#             f_name = f_name.replace(".", "_").lower()

#             # Extract key information
#             _df_params = pl.DataFrame(
#                 {
#                     "model": np.repeat(f_name, len(_sims["sweep"])),
#                     "s": np.zeros(len(_sims["sweep"])),
#                     "t": np.zeros(len(_sims["sweep"])),
#                     "saf": np.zeros(len(_sims["sweep"])),
#                     "eaf": np.zeros(len(_sims["sweep"])),
#                 }
#             )
#             df_params.append(_df_params)

#             fvs_file[f_name] = f"{data_dir}/vcfs/fvs_{f_name}.parquet"
#             sims[f_name] = _sims["sweep"]
#             regions[f_name] = _sims["region"]

#         df_params = pl.concat(df_params)

#         # Opening neutral expectations
#         try:
#             with open(f"{data_dir}/neutral_bins.pickle", "rb") as handle:
#                 neutral_stats_norm = pickle.load(handle)
#         except:
#             print(f"Please estimate fvs on simulations before continue")
#             return None
#     else:
#         required_folders = ["sweep", "neutral"]

#         for folder in required_folders:
#             folder_path = os.path.join(data_dir, folder)
#             if not os.path.exists(folder_path):
#                 raise ValueError(f"Required directory not found: {folder_path}")
#             if not glob.glob(os.path.join(folder_path, "*")):
#                 raise ValueError(f"Directory is empty: {folder_path}")

#         # Read simulation data
#         fs_data = Data(data_dir)
#         sims, df_params = fs_data.read_simulations()

#         # Define file paths
#         neutral_save = f"{data_dir}/neutral_bins.pickle"
#         fvs_file["sims"] = f"{data_dir}/fvs.parquet"

#         # Initialize regions dictionary
#         regions = {k: [None] * len(sims[k]) for k in ["neutral", "sweep"]}

#         # Validate simulation data
#         if not (
#             len(sims["sweep"]) > 0
#             and (len(sims["neutral"]) > 0 or neutral_save is not None)
#         ):
#             raise ValueError("Please input neutral and sweep simulations")

#     ########################
#     # Not using same parallel pool for simulation type and normalize to avoid RAM issues with multiprocessing. loky show errors increasing memory in some function unexpectedly
#     # with Parallel(n_jobs=nthreads, backend="loky", verbose=5) as parallel:

#     # Saving malformed simulations
#     results = defaultdict(lambda: None)
#     binned_data = defaultdict()
#     malformed_files = defaultdict()
#     d_centers = defaultdict()
#     tmp_bins = []

#     for sim_type, sim_data in sims.items():
#         print(sim_type)

#         params = df_params.filter(pl.col("model") == sim_type)[:, 1:].to_numpy()[:, :]

#         if vcf:
#             tmp_center = [
#                 tuple(map(int, r.split(":")[-1].split("-"))) for r in regions[sim_type]
#             ]
#             d_centers[sim_type] = np.array([(w[0] + w[1]) // 2 for w in tmp_center])

#             # Opening and closing parallel by each case
#             with Parallel(
#                 n_jobs=nthreads, backend="multiprocessing", verbose=5
#             ) as parallel:
#                 # joblib working inside, isafe/h12/haf will run by sliding windows using joblib configuration
#                 # hapdaf/sratio/freq and dind will use parallel numba
#                 stats = calculate_stats_vcf(
#                     sim_data,
#                     region=regions[sim_type],
#                     nthreads=nthreads,
#                     parallel_manager=parallel,
#                 )
#         else:
#             d_centers[sim_type] = center
#             # Limit to first 100 simulations for processing (as in original code)
#             paired_data = list(zip(sim_data, regions[sim_type]))[:]

#             # Opening and closing parallel by each case
#             with Parallel(
#                 n_jobs=nthreads, backend="multiprocessing", verbose=5
#             ) as parallel:
#                 stats = parallel(
#                     delayed(calculate_stats)(
#                         hap_data,
#                         _iter,
#                         center=center,
#                         step=step,
#                         neutral=True if sim_type == "neutral" else False,
#                         region=region,
#                     )
#                     for _iter, (hap_data, region) in enumerate(paired_data[:], 1)
#                 )

#         # Clean up results and handle malformed simulations
#         stats, params, malformed = cleaning_summaries(data_dir, stats, params, sim_type)
#         malformed_files[sim_type] = malformed

#         # Store binned data for specific simulation types
#         if sim_type == "neutral":
#             raw_stats, norm_stats = zip(*stats)
#             binned_data[sim_type] = binned_stats(*normalize_neutral(norm_stats))
#         elif sim_type == "sweep":
#             raw_stats, norm_stats = zip(*stats)
#         else:
#             raw_stats, norm_stats = stats
#             # Saving all available chr to normalize after stat estimations
#             tmp_bins.append(norm_stats)

#         # Create summary results

#         if ~np.all(params[:, 3] == 0):
#             params[:, 0] = -np.log(params[:, 0])

#         results[sim_type] = summaries(raw_stats, params)

#     df_fv_cnn = defaultdict()
#     df_fv_cnn_raw = defaultdict()

#     if vcf:
#         # Join all chromsomes windows and estimate expected and std values
#         binned_data["empirical"] = binned_stats(*normalize_neutral(tmp_bins))
#         binned_name = "empirical"

#         for k, stats_values in results.items():
#             print(k)
#             df_fv_w, df_fv_w_raw = normalize_stats(
#                 stats_values,
#                 binned_data[binned_name],
#                 center=d_centers[sim_type],
#                 parallel_manager=parallel,
#                 vcf=vcf,
#             )
#             df_fv_cnn[k] = df_fv_w
#             df_fv_cnn_raw[k] = df_fv_w_raw

#             df_fv_w.write_parquet(fvs_file[k])
#             df_fv_w_raw.write_parquet(fvs_file[k].replace(".parquet", "_raw.parquet"))

#         df_fv_training = pl.concat(df_fv_cnn.values(), how="vertical")
#         df_fv_training_raw = pl.concat(df_fv_cnn_raw.values(), how="vertical")

#     else:
#         binned_name = "neutral"
#         # Normalization is fast and does not consume high RAM. Opening same pool
#         with Parallel(n_jobs=nthreads, backend="loky", verbose=5) as parallel:
#             for k, stats_values in results.items():
#                 df_fv_w, df_fv_w_raw = normalize_stats(
#                     stats_values,
#                     binned_data[binned_name],
#                     center=d_centers[sim_type],
#                     parallel_manager=parallel,
#                     vcf=vcf,
#                 )
#                 df_fv_cnn[k] = df_fv_w
#                 df_fv_cnn_raw[k] = df_fv_w_raw

#         df_fv_training = pl.concat(df_fv_cnn.values(), how="vertical")
#         df_fv_training_raw = pl.concat(df_fv_cnn_raw.values(), how="vertical")

#         df_fv_w.write_parquet(fvs_file["sims"])
#         df_fv_w_raw.write_parquet(fvs_file["sims"].replace(".parquet", "_raw.parquet"))

#     # Save neutral_bins
#     with open(neutral_save, "wb") as handle:
#         pickle.dump(binned_data[binned_name], handle)

#     return df_fv_training, df_fv_training_raw


# def get_empir_freqs_np(hap):
#     """
#     Calculate the empirical frequencies of haplotypes.

#     Parameters:
#     - hap (numpy.ndarray): Array of haplotypes where each column represents an individual and each row represents a SNP.

#     Returns:
#     - k_counts (numpy.ndarray): Counts of each unique haplotype.
#     - h_f (numpy.ndarray): Empirical frequencies of each unique haplotype.
#     """
#     S, n = hap.shape

#     # Count occurrences of each unique haplotype
#     hap_f, k_counts = np.unique(hap, axis=1, return_counts=True)

#     # Sort counts in descending order
#     k_counts = np.sort(k_counts)[::-1]

#     # Calculate empirical frequencies
#     h_f = k_counts / n
#     return k_counts, h_f


# def T_m_statistic(K_counts, K_neutral, windows, K_truncation, sweep_mode=5, i=0):
#     output = []
#     m_vals = K_truncation + 1
#     epsilon_min = 1 / (K_truncation * 100)

#     _epsilon_values = list(map(lambda x: x * epsilon_min, range(1, 101)))
#     epsilon_max = K_neutral[-1]
#     epsilon_values = []

#     for ev in _epsilon_values:
#         # ev = e * epsilon_min
#         if ev <= epsilon_max:
#             epsilon_values.append(ev)
#     epsilon_values = np.array(epsilon_values)

#     for j, w in enumerate(windows):
#         # if(i==132):
#         # break
#         K_iter = K_counts[j]

#         null_likelihood = easy_likelihood(K_neutral, K_iter, K_truncation)

#         alt_likelihoods_by_e = []

#         for e in epsilon_values:
#             alt_likelihoods_by_m = []
#             for m in range(1, m_vals):
#                 alt_like = sweep_likelihood(
#                     K_neutral, K_iter, K_truncation, m, e, epsilon_max
#                 )
#                 alt_likelihoods_by_m.append(alt_like)

#             alt_likelihoods_by_m = np.array(alt_likelihoods_by_m)
#             likelihood_best_m = 2 * (alt_likelihoods_by_m.max() - null_likelihood)

#             if likelihood_best_m > 0:
#                 ml_max_m = (alt_likelihoods_by_m.argmax()) + 1
#             else:
#                 ml_max_m = 0

#             alt_likelihoods_by_e.append([likelihood_best_m, ml_max_m, e])

#         alt_likelihoods_by_e = np.array(alt_likelihoods_by_e)

#         likelihood_real = max(alt_likelihoods_by_e[:, 0])

#         out_index = np.flatnonzero(alt_likelihoods_by_e[:, 0] == likelihood_real)

#         out_intermediate = alt_likelihoods_by_e[out_index]

#         if out_intermediate.shape[0] > 1:
#             constarg = min(out_intermediate[:, 1])

#             outcons = np.flatnonzero(out_intermediate[:, 1] == constarg)

#             out_cons_intermediate = out_intermediate[outcons]

#             if out_cons_intermediate.shape[0] > 1:
#                 out_cons_intermediate = out_cons_intermediate[0]

#             out_intermediate = out_cons_intermediate

#         outshape = out_intermediate.shape

#         if len(outshape) != 1:
#             out_intermediate = out_intermediate[0]

#         out_intermediate = np.concatenate(
#             [out_intermediate, np.array([K_neutral[-1], sweep_mode, w]), K_iter]
#         )

#         output.append(out_intermediate)

#     # output = np.array(output)
#     # return output[output[:, 0].argmax(), :]

#     K_names = ["Kcounts_" + str(i) for i in range(1, K_iter.size + 1)]
#     output = pd.DataFrame(output)
#     output.insert(output.shape[1], "iter", i)

#     output.columns = (
#         [
#             "t_statistic",
#             "m",
#             "frequency",
#             "e",
#             "model",
#             "window_lassi",
#         ]
#         + K_names
#         + ["iter"]
#     )
#     return output


# def Ld(
#     hap: np.ndarray, pos: np.ndarray, min_freq=0.05, max_freq=1, start=None, stop=None
# ) -> tuple:
#     """
#     Compute Kelly Zns statistic (1997) and omega_max. Average r2
#     among every pair of loci in the genomic window.

#     Args:hap (numpy.ndarray): 2D array representing haplotype data. Rows correspond to different mutations, and columnscorrespond to chromosomes.
#     pos (numpy.ndarray): 1D array representing the positions of mutations.
#     min_freq (float, optional): Minimum frequency threshold. Default is 0.05.
#     max_freq (float, optional): Maximum frequency threshold. Default is 1.
#     window (int, optional): Genomic window size. Default is 500000.

#     Returns:tuple: A tuple containing two values:
#     - kelly_zns (float): Kelly Zns statistic.
#     - omega_max (float): Nielsen omega max.
#     """

#     if start is not None or stop is not None:
#         loc = (pos >= start) & (pos <= stop)
#         pos = pos[loc]
#         hap = hap[loc, :]

#     freqs = hap.sum(axis=1) / hap.shape[1]

#     hap_filter = hap[(freqs >= min_freq) & (freqs <= max_freq)]

#     r2_matrix = compute_r2_matrix_upper(hap_filter)
#     # r2_matrix = r2_torch(hap_filter)
#     S = hap_filter.shape[0]
#     zns = r2_matrix.sum() / comb(S, 2)
#     # Index combination to iter
#     omega_max = omega_linear_correct(r2_matrix)
#     # omega_max2 = dps.omega(r2_matrix)[0]

#     # return zns, 0
#     return zns, omega_max


# def r2_matrix(
#     hap: np.ndarray, pos: np.ndarray, min_freq=0.05, max_freq=1, start=None, stop=None
# ):
#     """
#     Compute Kelly Zns statistic (1997) and omega_max. Average r2
#     among every pair of loci in the genomic window.

#     Args:
#         hap (numpy.ndarray): 2D array representing haplotype data. Rows correspond to different mutations, and columns correspond to chromosomes.
#         pos (numpy.ndarray): 1D array representing the positions of mutations.
#         min_freq (float, optional): Minimum frequency threshold. Default is 0.05.
#         max_freq (float, optional): Maximum frequency threshold. Default is 1.
#         window (int, optional): Genomic window size. Default is 500000.

#     Returns: tuple: A tuple containing two values:
#         - kelly_zns (float): Kelly Zns statistic.
#         - omega_max (float): Nielsen omega max.
#     """

#     # if start is not None or stop is not None:
#     #     loc = (pos >= start) & (pos <= stop)
#     #     pos = pos[loc]
#     #     hap = hap[loc, :]

#     freqs = hap.sum(axis=1) / hap.shape[1]
#     freq_filter = (freqs >= min_freq) & (freqs <= max_freq)
#     hap_filter = hap[freq_filter]

#     r2_matrix = compute_r2_matrix(hap_filter)
#     # r2_matrix = r2_torch(hap_filter)
#     # S = hap_filter.shape[0]
#     # zns = r2_matrix.sum() / comb(S, 2)
#     # Index combination to iter
#     # omega_max = omega(r2_matrix)
#     # omega_max = dps.omega(r2_matrix)[0]

#     return r2_matrix, freq_filter
#     # return zns, omega_max


# def Ld(
#     r2_subset,
#     freq_filter,
#     pos: np.ndarray,
#     min_freq=0.05,
#     max_freq=1,
#     start=None,
#     stop=None,
# ):
#     pos_filter = pos[freq_filter]
#     if start is not None or stop is not None:
#         loc = (pos_filter >= start) & (pos_filter <= stop)
#         pos_filter = pos_filter[loc]
#         r2_subset = r2_subset[loc, :][:, loc]

#     # r2_subset_matrix = compute_r2_subset_matrix(hap_filter)
#     # r2_subset_matrix = r2_subset_torch(hap_filter)
#     S = r2_subset.shape[0]
#     kelly_zns = r2_subset.sum() / comb(S, 2)
#     # omega_max = omega(r2_subset)

#     return kelly_zns, 0


# @njit("float64[:,:](int8[:,:])", cache=True)
# def compute_r2_matrix(hap):
#     num_sites = hap.shape[0]

#     # r2_matrix = OrderedDict()
#     sum_r_squared = 0
#     r2_matrix = np.zeros((num_sites, num_sites))
#     # Avoid itertool.combination, not working on numba
#     # for pair in combinations(range(num_sites), 2):

#     # Check index from triangular matrix of size num_sites x num_sites. Each indices correspond to one one dimension of the array. Same as combinations(range(num_sites), 2)
#     c_1, c_2 = np.triu_indices(num_sites, 1)

#     for i, j in zip(c_1, c_2):
#         r2_matrix[i, j] = r2(hap[i, :], hap[j, :])
#         # r2_matrix[pair[0], pair[1]] = r2(hap[pair[0], :], hap[pair[1], :])

#     return r2_matrix


# @njit("float64(float64[:,:])", cache=True)
# def omega(r2_matrix):
#     """
#     Calculates Kim and Nielsen's (2004, Genetics 167:1513) omega_max statistic. Adapted from PG-Alignments-GAN

#     Args:r2_matrix (numpy.ndarray): 2D array representing r2 values.

#     Returns:
#         float: Kim and Nielsen's omega max.
#     """

#     omega_max = 0
#     S_ = r2_matrix.shape[1]

#     if S_ < 3:
#         omega_max = 0
#     else:
#         for l_ in range(3, S_ - 2):
#             sum_r2_L = 0
#             sum_r2_R = 0
#             sum_r2_LR = 0

#             for i in range(S_):
#                 for j in range(i + 1, S_):
#                     ld_calc = r2_matrix[i, j]
#                     if i < l_ and j < l_:
#                         sum_r2_L += ld_calc

#                     elif i >= l_ and j >= l_:
#                         sum_r2_R += ld_calc

#                     elif i < l_ and j >= l_:
#                         sum_r2_LR += ld_calc

#             # l_ ## to keep the math right outside of indexing
#             omega_numerator = (
#                 1 / ((l_ * (l_ - 1) / 2) + ((S_ - l_) * (S_ - l_ - 1) / 2))
#             ) * (sum_r2_L + sum_r2_R)
#             omega_denominator = (1 / (l_ * (S_ - l_))) * sum_r2_LR

#             if omega_denominator == 0:
#                 omega = 0
#             else:
#                 omega = np.divide(omega_numerator, omega_denominator)

#             if omega > omega_max:
#                 omega_max = omega

#     return omega_max


# @njit
# def sq_freq_pairs(hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size):
#     # Compute counts and frequencies
#     hap_derived = hap
#     hap_ancestral = np.bitwise_xor(hap_derived, 1)
#     derived_count = ac[:, 1]
#     ancestral_count = ac[:, 0]
#     freqs = ac[:, 1] / ac.sum(axis=1)

#     # Focal filter
#     focal_filter = (freqs >= min_focal_freq) & (freqs <= max_focal_freq)
#     focal_derived = hap_derived[focal_filter, :]
#     focal_derived_count = derived_count[focal_filter]
#     focal_ancestral = hap_ancestral[focal_filter, :]
#     focal_ancestral_count = ancestral_count[focal_filter]
#     focal_index = focal_filter.nonzero()[0]

#     # Allocate fixed-size lists to avoid growing lists
#     sq_out = [np.zeros((0, 3))] * len(focal_index)
#     # info = [None] * len(focal_index)
#     info = np.zeros((len(focal_index), 4))
#     # Main loop to calculate frequencies
#     for j in range(len(focal_index)):
#         i = focal_index[j]
#         size = window_size / 2

#         # Find indices within the window
#         # z = np.flatnonzero(np.abs(rec_map[i, -1] - rec_map[:, -1]) <= size)
#         mask = np.abs(rec_map[i, -1] - rec_map[:, -1]) <= size
#         z = np.where(mask)[0]

#         # Index range
#         x_r, y_r = i + 1, z[-1]
#         x_l, y_l = z[0], i - 1

#         # Calculate derived and ancestral frequencies
#         f_d_l = (
#             np.sum(focal_derived[j] & hap_derived[x_l : y_l + 1], axis=1)
#             / focal_derived_count[j]
#         )
#         f_a_l = (
#             np.sum(focal_ancestral[j] & hap_derived[x_l : y_l + 1], axis=1)
#             / focal_ancestral_count[j]
#         )
#         f_tot_l = freqs[x_l : y_l + 1]

#         f_d_r = (
#             np.sum(focal_derived[j] & hap_derived[x_r : y_r + 1], axis=1)
#             / focal_derived_count[j]
#         )
#         f_a_r = (
#             np.sum(focal_ancestral[j] & hap_derived[x_r : y_r + 1], axis=1)
#             / focal_ancestral_count[j]
#         )
#         f_tot_r = freqs[x_r : y_r + 1]

#         # Concatenate frequencies into a single array
#         sq_freqs = np.empty((f_d_l.size + f_d_r.size, 3))
#         sq_freqs[: f_d_l.size, 0] = f_d_l[::-1]
#         sq_freqs[: f_d_l.size, 1] = f_a_l[::-1]
#         sq_freqs[: f_d_l.size, 2] = f_tot_l[::-1]
#         sq_freqs[f_d_l.size :, 0] = f_d_r
#         sq_freqs[f_d_l.size :, 1] = f_a_r
#         sq_freqs[f_d_l.size :, 2] = f_tot_r

#         sq_out[j] = sq_freqs
#         info[j] = np.array(
#             [rec_map[i, -1], freqs[i], focal_derived_count[j], focal_ancestral_count[j]]
#         )

#     return sq_out, info


# def compare_haplos(haplo_1, haplo_2):
#     identical = haplo_1.count("1")  # Count "1"s in haplo_1
#     different = sum(1 for h1, h2 in zip(haplo_1, haplo_2) if h1 != h2)
#     total = identical + different  # Total equals identical + different

#     return identical, different, total


# def run_h12(
#     hap,
#     rec_map,
#     _iter=1,
#     neutral=True,
#     script="/home/jmurgamoreno/software/calculate_H12_modified.pl",
# ):
#     df_hap = pl.DataFrame(hap)
#     df_rec_map = pl.DataFrame(rec_map)
#     hap_file = "/tmp/tmp_" + str(_iter) + ".hap"
#     map_file = "/tmp/tmp_" + str(_iter) + ".map"
#     with open(hap_file, "w") as f:
#         for row in df_hap.iter_rows():
#             f.write("".join(map(str, row)) + "\n")

#     df_rec_map.write_csv(map_file, include_header=False, separator=" ")

#     h12_pl = "perl " + script + " " + hap_file + " " + map_file + " out "
#     h12_pl += "500000 " if neutral else "1200000"

#     with subprocess.Popen(h12_pl.split(), stdout=subprocess.PIPE) as process:
#         h12_v = float(process.stdout.read())

#     os.remove(hap_file)
#     os.remove(map_file)

#     return h12_v


# def run_haf(
#     hap,
#     rec_map,
#     _iter=1,
#     neutral=True,
#     script="/home/jmurgamoreno/software/calculate_HAF.pl",
# ):
#     df_hap = pl.DataFrame(hap)
#     df_rec_map = pl.DataFrame(rec_map)
#     hap_file = "/tmp/tmp_" + str(_iter) + ".hap"
#     map_file = "/tmp/tmp_" + str(_iter) + ".map"
#     with open(hap_file, "w") as f:
#         for row in df_hap.iter_rows():
#             f.write("".join(map(str, row)) + "\n")

#     df_rec_map.write_csv(map_file, include_header=False, separator=" ")

#     haf_pl = "perl " + script + " " + hap_file + " " + map_file + " out "
#     haf_pl += "500000 " if neutral else "1200000"

#     with subprocess.Popen(haf_pl.split(), stdout=subprocess.PIPE) as process:
#         haf_v = float(process.stdout.read())

#     os.remove(hap_file)
#     os.remove(map_file)

#     return haf_v


# def run_hapbin_og(
#     hap,
#     rec_map,
#     _iter=1,
#     min_ehh=0.05,
#     min_maf=0.05,
#     ihsbin=None,
# ):
#     if ihsbin is None:
#         ihsbin = shutil.which("ihsbin")
#     df_hap = pl.DataFrame(hap)
#     df_rec_map = pl.DataFrame(
#         rec_map,
#         pl.Schema(
#             [
#                 ("chr", pl.Int64),
#                 ("location", pl.Int64),
#                 ("cm_mb", pl.Float64),
#                 ("cm", pl.Float64),
#             ]
#         ),
#     )
#     hap_file = "/tmp/tmp_" + str(_iter) + ".hap"
#     map_file = "/tmp/tmp_" + str(_iter) + ".map"

#     with open(hap_file, "w") as f:
#         for row in df_hap.iter_rows():
#             f.write("".join(str(value) for value in row) + "\n")

#     df_rec_map.write_csv(map_file, include_header=False, separator=" ")

#     out_file = f"/tmp/tmp_{_iter}.out"
#     cmd_hapbin = f"{ihsbin} --hap {hap_file} --map {map_file} --cutoff {min_ehh} --minmaf {min_maf} --out {out_file}"

#     ihs_output = subprocess.run(cmd_hapbin, shell=True, capture_output=True, text=True)

#     tmp_ihs = pl.read_csv(
#         out_file,
#         separator="\t",
#         schema=pl.Schema(
#             [
#                 ("location", pl.Int64),
#                 ("ihh_0", pl.Float64),
#                 ("ihh_1", pl.Float64),
#                 ("ihs", pl.Float64),
#                 ("std_ihs", pl.Float64),
#             ]
#         ),
#     )

#     r_pos = dict(zip(rec_map[:, 1], rec_map[:, -1]))
#     _p = np.array([r_pos[i] for i in tmp_ihs["location"]])
#     df_daf = pl.DataFrame(
#         {"positions": rec_map[:, -1].astype(int), "daf": (hap.sum(axis=1) / hap.shape[1]) * 2}
#     )
#     df_ihs = (
#         tmp_ihs.with_columns(pl.lit(_p).cast(pl.Int64).alias("positions"))
#         .join(df_daf, on="positions", coalesce=True)
#         .with_columns(
#             (pl.col("ihh_1") - pl.col("ihh_0"))
#             .abs()
#             .cast(pl.Float64)
#             .alias("delta_ihh")
#         )
#         .select(["positions", "daf", "ihs", "delta_ihh"])
#     ).fill_nan(None)

#     os.remove(hap_file)
#     os.remove(map_file)
#     os.remove(out_file)

#     return df_ihs
