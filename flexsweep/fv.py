import os

import subprocess

from . import np, Parallel, delayed, pl
from .data import Data

# from joblib import Parallel, delayed
# import numpy as np
# import polars as pl

from math import comb
from functools import partial, reduce
from numba import njit
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
)
from allel.compat import memoryview_safe
from allel.opt.stats import ihh01_scan, ihh_scan
from allel.util import asarray_ndim, check_dim0_aligned, check_integer_dtype
from allel.stats.selection import compute_ihh_gaps

from copy import deepcopy
from collections import defaultdict, namedtuple
from itertools import product, chain

from warnings import filterwarnings
import pickle

import gzip
import re


# Define the inner namedtuple structure
summaries = namedtuple("summaries", ["stats", "parameters"])


################## Utils
def mispolarize(hap, proportion=0.1):
    """
    Allele mispolarization by randomly flipping the alleles of a haplotype matrix (i.e., switching between 0 and 1). The proportion of rows to be flipped is determined by the `proportion` parameter.

    Parameters
    ----------
    hap : numpy.ndarray
        A 2D numpy array representing the haplotype matrix of shape (S, n),
        where S is the number of variants (rows), and n is the number of samples (columns).
        Each element is expected to be binary (0 or 1), representing the alleles.

    proportion : float, optional (default=0.1)
        A float between 0 and 1 specifying the proportion of rows (loci) in the haplotype
        matrix to randomly flip. For example, if proportion=0.1, 10% of the rows in the
        haplotype matrix will have their allele values flipped.

    Returns
    -------
    hap_copy : numpy.ndarray
        A new 2D numpy array of the same shape as `hap`, with a proportion of rows
        randomly flipped (alleles inverted). The original matrix `hap` is not modified
        in-place.

    Notes
    -----
    The flipping operation is done using a bitwise XOR operation (`^= 1`), which
    efficiently flips 0 to 1 and 1 to 0 for the selected rows.

    """
    # Get shape of haplotype matrix
    S, n = hap.shape

    # Select the column indices to flip based on the given proportion
    to_flip = np.random.choice(np.arange(S), int(S * proportion), replace=False)

    # Create a copy of the original hap matrix to avoid in-place modification
    hap_copy = hap.copy()
    hap_copy[to_flip, :] ^= 1
    return hap_copy


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
        hap = HaplotypeArray(hap.genotype_matrix())
    except:
        try:
            hap = HaplotypeArray(hap)
        except:
            hap = HaplotypeArray(load(hap).genotype_matrix())

    # positions = rec_map[:, -1]
    # physical_position = rec_map[:, -2]

    # HAP matrix centered to analyse whole chromosome
    hap_01, ac, biallelic_mask = filter_biallelics(hap)
    hap_int = hap_01.astype(np.int8)
    rec_map_01 = rec_map[biallelic_mask]
    sequence_length = int(1.2e6)
    freqs = ac.to_frequencies()[:, 1]

    if region is not None:
        tmp = list(map(int, region.split(":")[-1].split("-")))
        d_pos = dict(
            zip(np.arange(tmp[0], tmp[1] + 1), np.arange(1, sequence_length + 1))
        )
        for r in rec_map_01:
            r[-1] = d_pos[r[-1]]

    position_masked = rec_map_01[:, -1]
    physical_position_masked = rec_map_01[:, -2]

    return (
        hap_01,
        ac,
        biallelic_mask,
        hap_int,
        rec_map_01,
        position_masked,
        physical_position_masked,
        sequence_length,
        freqs,
    )


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
    hap_01, ac, biallelic_mask = filter_biallelics2(hap)
    sequence_length = int(1.2e6)

    rec_map_01 = rec_map[biallelic_mask]
    position_masked = rec_map_01[:, -1]
    physical_position_masked = rec_map_01[:, -2]

    return (
        hap_01,
        rec_map_01,
        ac,
        biallelic_mask,
        position_masked,
        physical_position_masked,
    )


def filter_biallelics(hap: HaplotypeArray) -> tuple:
    """
    Filter out non-biallelic loci from the haplotype data.

    Args: hap (allel.HaplotypeArray): Haplotype data represented as a HaplotypeArray.

    Returns:tuple: A tuple containing three elements:
        - hap_biallelic (allel.HaplotypeArray): Filtered biallelic haplotype data.
        - ac_biallelic (numpy.ndarray): Allele counts for the biallelic loci.
        - biallelic_mask (numpy.ndarray): Boolean mask indicating biallelic loci.
    """
    ac = hap.count_alleles()
    biallelic_mask = ac.is_biallelic_01()
    return (hap.subset(biallelic_mask), ac[biallelic_mask, :], biallelic_mask)


def filter_biallelics2(hap: HaplotypeArray) -> tuple:
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


################## Summaries


def worker(args):
    ts, rec_map, index, center, step, neutral, mispolarize_ratio = args
    return calculate_stats(
        ts,
        rec_map,
        index,
        center=center,
        step=step,
        neutral=neutral,
        mispolarize_ratio=mispolarize_ratio,
    )


def center_window_cols(df, _iter=1, center=int(6e5), window=int(1e6)):
    if df.is_empty():
        # Return a dataframe with one row of the specified default values
        return pl.concat(
            [pl.DataFrame({"iter": _iter, "center": center, "window": window}), df],
            how="horizontal",
        )
    df = (
        df.with_columns(
            [
                pl.lit(_iter).alias("iter"),
                pl.lit(center).alias("center"),
                pl.lit(window).alias("window"),
            ]
        )
        .with_columns(pl.col(["iter", "center", "window"]).cast(pl.Int64))
        .select(
            pl.col(["iter", "center", "window", "positions"]),
            pl.all().exclude(["iter", "center", "window", "positions"]),
        )
    )
    return df


def calculate_stats(
    hap_data,
    _iter=1,
    center=[5e5, 7e5],
    windows=[1000000],
    step=1e4,
    neutral=False,
    region=None,
):
    filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in scalar divide",
    )
    np.seterr(divide="ignore", invalid="ignore")

    try:
        hap, rec_map, p = ms_parser(hap_data)
    except:
        hap, rec_map, p = hap_data

    # Open and filtering data
    (
        hap_int,
        rec_map_01,
        ac,
        biallelic_mask,
        position_masked,
        physical_position_masked,
    ) = filter_gt(hap, rec_map, region=region)
    freqs = ac[:, 1] / ac.sum(axis=1)

    if len(center) == 1:
        centers = np.arange(center[0], center[0] + step, step).astype(int)
    else:
        centers = np.arange(center[0], center[1] + step, step).astype(int)

    df_dind_high_low = dind_high_low(hap_int, ac, rec_map_01)
    df_s_ratio = s_ratio(hap_int, ac, rec_map_01)
    df_hapdaf_o = hapdaf_o(hap_int, ac, rec_map_01)
    df_hapdaf_s = hapdaf_s(hap_int, ac, rec_map_01)

    d_stats = defaultdict(dict)

    df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
    df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
    df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
    df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

    d_stats["dind_high_low"] = df_dind_high_low
    d_stats["s_ratio"] = df_s_ratio
    d_stats["hapdaf_o"] = df_hapdaf_o
    d_stats["hapdaf_s"] = df_hapdaf_s
    try:
        h12_v = h12_enard(
            hap_int, rec_map_01, window_size=int(5e5) if neutral else int(1.2e6)
        )
        # h12_v = run_h12(hap, rec_map, _iter=_iter, neutral=neutral)
    except:
        h12_v = np.nan

    haf_v = haf_top(hap_int.astype(np.float64), position_masked)

    daf_w = 1.0
    pos_w = int(6e5)
    if 6e5 in position_masked:
        daf_w = freqs[position_masked == 6e5][0]

    df_window = pl.DataFrame(
        {
            "iter": pl.Series([_iter], dtype=pl.Int64),
            "center": pl.Series([int(6e5)], dtype=pl.Int64),
            "window": pl.Series([int(1e6)], dtype=pl.Int64),
            "positions": pl.Series([pos_w], dtype=pl.Int64),
            "daf": pl.Series([daf_w], dtype=pl.Float64),
            "h12": pl.Series([h12_v], dtype=pl.Float64),
            "haf": pl.Series([haf_v], dtype=pl.Float64),
        }
    )

    d_stats["h12_haf"] = df_window

    d_centers_stats = defaultdict(dict)
    schema_center = {
        "iter": pl.Int64,
        "center": pl.Int64,
        "window": pl.Int64,
        "positions": pl.Int64,
        "daf": pl.Float64,
        "ihs": pl.Float64,
        "delta_ihh": pl.Float64,
        "isafe": pl.Float64,
        "nsl": pl.Float64,
    }
    for c, w in product(centers, windows):
        lower = c - w / 2
        upper = c + w / 2

        p_mask = (position_masked >= lower) & (position_masked <= upper)
        p_mask
        f_mask = freqs >= 0.05

        # Check whether the hap subset is empty or not
        if hap_int[p_mask].shape[0] == 0:
            # df_centers_stats = pl.DataFrame({"iter": _iter,"center": c,"window": w,"positions": np.nan,"daf": np.nan,"isafe": np.nan,"ihs": np.nan,"nsl": np.nan,})
            d_empty = pl.DataFrame(
                [
                    [_iter],
                    [c],
                    [w],
                    [None],
                    [np.nan],
                    [np.nan],
                    [np.nan],
                    [np.nan],
                    [np.nan],
                ],
                schema=schema_center,
            )

            d_centers_stats["ihs"][c] = d_empty.select(
                ["iter", "center", "window", "positions", "daf", "ihs", "delta_ihh"]
            )
            d_centers_stats["isafe"][c] = d_empty.select(
                ["iter", "center", "window", "positions", "daf", "isafe"]
            )
            d_centers_stats["nsl"][c] = d_empty.select(
                ["iter", "center", "window", "positions", "daf", "nsl"]
            )
        else:
            df_isafe = run_isafe(hap_int[p_mask], position_masked[p_mask])

            # iHS and nSL
            df_ihs = ihs_ihh(
                hap_int[p_mask],
                position_masked[p_mask],
                map_pos=physical_position_masked[p_mask],
                min_ehh=0.05,
                min_maf=0.05,
                include_edges=False,
            )

            nsl_v = nsl(hap_int[(p_mask) & (f_mask)], use_threads=False)
            df_nsl = pl.DataFrame(
                {
                    "positions": position_masked[(p_mask) & (f_mask)],
                    "daf": freqs[(p_mask) & (f_mask)],
                    "nsl": nsl_v,
                }
            ).fill_nan(None)

            df_isafe = center_window_cols(df_isafe, _iter=_iter, center=c, window=w)
            df_ihs = center_window_cols(df_ihs, _iter=_iter, center=c, window=w)
            df_nsl = center_window_cols(df_nsl, _iter=_iter, center=c, window=w)

            d_centers_stats["ihs"][c] = df_ihs
            d_centers_stats["isafe"][c] = df_isafe
            d_centers_stats["nsl"][c] = df_nsl

    d_stats["ihs"] = pl.concat(d_centers_stats["ihs"].values())
    d_stats["isafe"] = pl.concat(d_centers_stats["isafe"].values())
    d_stats["nsl"] = pl.concat(d_centers_stats["nsl"].values())

    if region is not None:
        for k, df in d_stats.items():
            d_stats[k] = df.with_columns(pl.lit(region).alias("iter"))

    if neutral:
        # Whole chromosome statistic to normalize
        df_isafe = run_isafe(hap_int, position_masked)
        df_ihs = ihs_ihh(hap_int, position_masked, min_ehh=0.1, include_edges=True)

        nsl_v = nsl(hap_int[freqs >= 0.05], use_threads=False)

        df_nsl = pl.DataFrame(
            {
                "positions": position_masked[freqs >= 0.05],
                "daf": freqs[freqs >= 0.05],
                "nsl": nsl_v,
            }
        ).fill_nan(None)

        df_snps_norm = reduce(
            lambda left, right: left.join(
                right, on=["positions", "daf"], how="full", coalesce=True
            ),
            [df_nsl, df_ihs, df_isafe],
        )

        df_stats = reduce(
            lambda left, right: left.join(
                right,
                on=["iter", "center", "window", "positions", "daf"],
                how="full",
                coalesce=True,
            ),
            [df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s, df_window],
        ).sort(["iter", "center", "window", "positions"])
        df_stats_norm = df_snps_norm.join(
            df_stats.select(
                pl.all().exclude(
                    ["iter", "center", "window", "delta_ihh", "ihs", "isafe", "nsl"]
                )
            ),
            on=["positions", "daf"],
            how="full",
            coalesce=True,
        ).sort(["positions"])

        df_stats_norm = (
            df_stats_norm.with_columns(
                [
                    pl.lit(_iter).alias("iter"),
                    pl.lit(600000).alias("center"),
                    pl.lit(1200000).alias("window"),
                    pl.col("positions").cast(pl.Int64),
                ]
            )
            .select(
                pl.col(["iter", "center", "window", "positions"]),
                pl.all().exclude(["iter", "center", "window", "positions"]),
            )
            .sort(["iter", "center", "window", "positions"])
        )

        return d_stats, df_stats_norm
    else:
        return d_stats


def summary_statistics(
    data,
    nthreads=1,
    center=[500000, 700000],
    windows=[1000000],
    step=10000,
    neutral_save=None,
    vcf=False,
    recombination_map=None,
):
    """
    Computes summary statistics across multiple simulations or empirical data, potentially using
    multiple threads for parallel computation. The statistics are calculated over
    defined genomic windows, with optional mispolarization applied to the haplotype data.
    Save the dataframe to a parquet file


    Only iHS, nSL and iSAFE are estimated across all windows/center combination. The other
    statistics used the actual center (1.2e6 / 2) extended to 500Kb each flank.

    Parameters
    ----------
    sims : str,
        Discoal simulation path or VCF file. If VCF file ensure you're use `vcf=True` argument.
    nthreads : int, optional (default=1)
        The number of threads to use for parallel computation. If set to 1,
        the function runs in single-threaded mode. Higher values will enable
        multi-threaded processing to speed up calculations.
    center : list of int, optional (default=[500000, 700000])
        A list specifying the center positions (in base pairs) for the analysis windows.
        If one center is provided, it will use that as a single point; otherwise,
        the analysis will cover the range between the two provided centers.

    windows : list of int, optional (default=[1000000])
        A list of window sizes (in base pairs) over which summary statistics will be computed.

    step : int, optional (default=10000)
        The step size (in base pairs) for sliding windows in the analysis. This determines
        how much the analysis window moves along the genome for each iteration.
    vcf : bool,
        If true parse vcf

    Returns
    -------
    summary_stats : pandas.DataFrame
        A DataFrame containing the computed summary statistics for each simulation and
        for each genomic window.

    """

    vcf = True if "vcf" in data else False

    if not vcf:
        # Reading simulations
        fs_data = Data(data, nthreads=nthreads)
        sims, df_params = fs_data.read_simulations()

        # Default file names form simulations
        neutral_save = data + "/neutral_bins.pickle"
        fvs_file = data + "/fvs.parquet"

        # Opening neutral expectations
        if os.path.exists(neutral_save):
            with open(neutral_save, "rb") as handle:
                neutral_stats_norm = pickle.load(handle)
        else:
            neutral_stats_norm = None

        # Region as large as possible, later zip do the proper combination in case simulation number differs
        n_sims = (
            len(sims["sweep"])
            if len(sims["sweep"]) > len(sims["neutral"])
            else len(sims["neutral"])
        )
        regions = [None] * n_sims

        # assert len(sims["sweep"]) > 0 and (
        #     len(sims["neutral"]) > 0 or neutral_save is not None
        # ), "Please input neutral and sweep simulations"
        if not (
            len(sims["sweep"]) > 0
            and (len(sims["neutral"]) > 0 or neutral_save is not None)
        ):
            raise ValueError("Please input neutral and sweep simulations")

    else:
        # elif isinstance(data, dict) and "region" in data.keys():

        # Force opening bins
        # assert neutral_save is not None, "Input neutral bins"
        if neutral_save is None:
            raise ValueError(
                "Neutral bins are required. Please provide the path for neutral_save."
            )

        # Reading VCF
        fs_data = Data(data, recombination_map=recombination_map, nthreads=nthreads)
        sims = fs_data.read_vcf()

        # Save region and remove from dict to iter only genotype data on summary_statistics.
        regions = sims["region"]
        sims.pop("region", None)

        with open(neutral_save, "rb") as handle:
            neutral_stats_norm = pickle.load(handle)

        if (
            neutral_stats_norm["expected"].is_empty()
            or neutral_stats_norm["expected"].to_pandas().isnull().all().all()
            or neutral_stats_norm["stdev"].is_empty()
            or neutral_stats_norm["stdev"].to_pandas().isnull().all().all()
        ):
            raise ValueError(
                "Check expected and stdev neutral data before continue. Some pl.DataFrame is empty"
            )

        # Same folder custom fvs name based on input VCF.
        f_name = os.path.basename(data)
        for ext in [".vcf", ".bcf", ".gz"]:
            f_name = f_name.replace(ext, "")
        f_name = f_name.replace(".", "_").lower()

        fvs_file = os.path.dirname(data) + "/fvs_" + f_name + ".parquet"

        # Empty params dataframe to process empirical data
        df_params = pl.DataFrame(
            {
                "model": np.repeat("sweep", len(sims["sweep"])),
                "s": np.zeros(len(sims["sweep"])),
                "t": np.zeros(len(sims["sweep"])),
                "saf": np.zeros(len(sims["sweep"])),
                "eaf": np.zeros(len(sims["sweep"])),
            }
        )

    for k, s in sims.items():
        pars = [i for i in zip(s, regions)]

        # Use joblib to parallelize the execution
        # summ_stats = Parallel(n_jobs=nthreads, backend="loky", verbose=5)(
        p_backend = "multiprocessing" if len(pars) < 5e4 else "loky"
        summ_stats = Parallel(n_jobs=nthreads, backend="loky", verbose=5)(
            delayed(calculate_stats)(
                hap_data,
                _iter,
                center=center,
                step=step,
                neutral=True if k == "neutral" else False,
                region=region,
            )
            # for _iter, (hap,rec_map,region) in enumerate(s, 1)
            for _iter, (hap_data, region) in enumerate(pars, 1)
        )

        # Ensure params order
        # return params from summ_stats if reading
        # params = df_params[df_params.model == k].iloc[:, 1:].values
        params = df_params.filter(model=k).select(df_params.columns[1:]).to_numpy()

        if k == "neutral":
            summ_stats, summ_stats_norm = zip(*summ_stats)
            neutral_stats = summaries(
                stats=summ_stats,
                parameters=params,
            )
            neutral_stats_norm = summaries(
                stats=summ_stats_norm,
                parameters=params,
            )
        else:
            if ~np.all(params[:, 3] == 0):
                params[:, 0] = -np.log(params[:, 0])
            # summ_stats, summ_nsl = zip(*summ_stats)
            sweeps_stats = summaries(
                stats=summ_stats,
                parameters=params,
            )

    df_fv_sweep, neutral_stats_norm = normalization(
        sweeps_stats, neutral_stats_norm, nthreads=nthreads
    )

    df_fv_sweep = df_fv_sweep.with_columns(pl.lit("sweep").alias("model")).select(
        pl.exclude("delta_ihh")
    )

    df_fv_sweep = df_fv_sweep.with_columns(
        pl.when((pl.col("t") >= 2000) & (pl.col("f_t") >= 0.9))
        .then(pl.lit("hard_old_complete"))
        .when((pl.col("t") >= 2000) & (pl.col("f_t") < 0.9))
        .then(pl.lit("hard_old_incomplete"))
        .when((pl.col("t") < 2000) & (pl.col("f_t") >= 0.9))
        .then(pl.lit("hard_young_complete"))
        .otherwise(pl.lit("hard_young_incomplete"))
        .alias("model")
    )

    df_fv_sweep = df_fv_sweep.with_columns(
        pl.when(pl.col("f_i") != df_fv_sweep["f_i"].min())
        .then(pl.col("model").str.replace("hard", "soft"))
        .otherwise(pl.col("model"))
        .alias("model")
    )

    if (df_fv_sweep["s"] == 0).all():
        df_fv_sweep = df_fv_sweep.with_columns(pl.lit("neutral").alias("model"))

    # Do not sort iter if vcf, sort lexicographically
    sort_multi = True if df_fv_sweep["iter"].dtype == pl.Utf8 else False

    df_fv_sweep_w = df_fv_sweep.pivot(
        values=df_fv_sweep.columns[7:-1],
        index=["iter", "s", "t", "f_i", "f_t", "model"],
        on=["window", "center"],
    ).sort(by="iter")

    df_fv_sweep_w = df_fv_sweep_w.rename(
        {
            col: col.replace("{", "").replace("}", "").replace(",", "_")
            for col in df_fv_sweep_w.columns
        }
    )

    if "neutral" in sims.keys():
        # Save neutral expectations
        if os.path.exists(neutral_save) is False:
            with open(neutral_save, "wb") as handle:
                pickle.dump(neutral_stats_norm, handle)

        # Normalizing neutral simulations
        df_fv_neutral, neutral_stats_norm = normalization(
            neutral_stats,
            neutral_stats_norm,
            nthreads=nthreads,
        )
        df_fv_neutral = df_fv_neutral.with_columns(
            pl.lit("neutral").alias("model")
        ).select(pl.exclude("delta_ihh"))

        # Unstack instead pivot since we only need to reshape based on window and center values
        df_fv_neutral_w = df_fv_neutral.pivot(
            values=df_fv_neutral.columns[7:-1],
            index=["iter", "s", "t", "f_i", "f_t", "model"],
            on=["window", "center"],
        ).sort(by="iter")

        df_fv_neutral_w = df_fv_neutral_w.rename(
            {
                col: col.replace("{", "").replace("}", "").replace(",", "_")
                for col in df_fv_neutral_w.columns
            }
        )
        df_fv_w = pl.concat([df_fv_sweep_w, df_fv_neutral_w], how="vertical")
    else:
        df_fv_w = df_fv_sweep_w

    # dump fvs with more than 10% nans
    df_fv_w = df_fv_w.fill_nan(None)
    num_nans = df_fv_w.transpose().null_count().to_numpy().flatten()
    df_fv_w = (
        df_fv_w.filter(
            num_nans < int(df_fv_w.select(df_fv_w.columns[6:]).shape[1] * 0.1)
        )
        .sort(["model", "iter"])
        .fill_null(0)
    )

    df_fv_w.write_parquet(fvs_file)

    return df_fv_w


################## Normalization


def normalization(
    sweeps_stats,
    neutral_stats_norm,
    center=[5e5, 7e5],
    windows=[50000, 100000, 200000, 500000, 1000000],
    nthreads=1,
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

    df_stats, params = sweeps_stats

    if isinstance(neutral_stats_norm, dict):
        expected, stdev = neutral_stats_norm.values()
    else:
        df_stats_neutral, params_neutral = neutral_stats_norm
        expected, stdev = normalize_neutral(df_stats_neutral)

    # Tried different nthreads/batch_size combinations for 100k sims, 200 threads
    # p_backend = "multiprocessing" if len(df_stats) < 5e4 else "loky"
    p_backend = "loky"
    df_fv_n_l = Parallel(n_jobs=nthreads, backend=p_backend, verbose=5)(
        delayed(normalize_cut)(
            _iter, v, expected=expected, stdev=stdev, center=center, windows=windows
        )
        for _iter, v in enumerate(df_stats, 1)
    )

    # Ensure dtypes
    df_fv_n = pl.concat(df_fv_n_l).with_columns(
        pl.col(["iter", "window", "center"]).cast(pl.Int64)
    )

    # Save region instead of iter if vcf
    try:
        df_window = pl.concat([df["h12_haf"] for df in df_stats]).select(
            pl.exclude(["center", "window", "positions", "daf"])
        )
        df_fv_n = df_fv_n.join(df_window, on=["iter"], how="full", coalesce=True)
    except:
        df_window = pl.concat([df["h12_haf"] for df in df_stats]).select(
            pl.exclude(["center", "window", "positions", "daf"])
        )

        df_window.insert_column(
            0, pl.Series("_true_iter", np.arange(1, df_window.shape[0] + 1))
        )
        tmp_names = list(df_fv_n.columns)
        tmp_names[0] = "_true_iter"
        df_fv_n.columns = tmp_names
        df_fv_n = df_fv_n.join(df_window, on=["_true_iter"], how="full", coalesce=True)

        df_fv_n = df_fv_n.drop("_true_iter")

    # params = params[:, [0, 1, 3, 4, ]]
    df_fv_n = pl.concat(
        [
            pl.DataFrame(
                np.repeat(
                    params.copy(),
                    df_fv_n.select(["center", "window"])
                    .unique()
                    .sort(["center", "window"])
                    .shape[0],
                    axis=0,
                ),
                schema=["s", "t", "f_i", "f_t"],
            ),
            df_fv_n,
        ],
        how="horizontal",
    )

    force_order = ["iter"] + [col for col in df_fv_n.columns if col != "iter"]
    df_fv_n = df_fv_n.select(force_order)
    return df_fv_n, {"expected": expected, "stdev": stdev}


def normalize_neutral(df_stats_neutral):
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

    # Get std and mean values from dataframe
    tmp_neutral = pl.concat(df_stats_neutral).fill_nan(None)
    df_binned = bin_values(tmp_neutral.select(pl.exclude(window_stats)))

    # get expected value (mean) and standard deviation
    expected = (
        df_binned.select(df_binned.columns[5:])
        .group_by("freq_bins")
        .mean()
        .sort("freq_bins")
        .fill_nan(None)
    )
    stdev = (
        df_binned.select(df_binned.columns[5:])
        .group_by("freq_bins")
        .agg([pl.all().exclude("freq_bins").std()])
    )

    # expected.index = expected.index.astype(str)
    # stdev.index = stdev.index.astype(str)

    return expected, stdev


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

    return values_copy


def normalize_cut(
    _iter,
    snps_values,
    expected,
    stdev,
    center=[5e5, 7e5],
    windows=[50000, 100000, 200000, 500000, 1000000],
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

    centers = np.arange(center[0], center[1] + 1e4, 1e4).astype(int)
    iter_c_w = list(product(centers, windows))

    df_out = []
    for k, df in snps_values.items():
        if k == "h12_haf":
            continue

        stats_names = df.select(df.columns[5:]).columns

        binned_values = bin_values(df)
        binned_values = (
            binned_values.join(
                expected.select(["freq_bins"] + stats_names),
                on="freq_bins",
                how="left",
                coalesce=True,
                suffix="_mean",
            )
            .join(
                stdev.select(["freq_bins"] + stats_names),
                on="freq_bins",
                how="left",
                coalesce=True,
                suffix="_std",
            )
            .fill_nan(None)
        )

        out = []
        for s in stats_names:
            out.append(
                binned_values.with_columns(
                    (pl.col(s) - pl.col(s + "_mean")) / pl.col(s + "_std")
                ).select(s)
            )
        binned_values = pl.concat(
            [
                binned_values.select(["iter", "center", "window", "positions"]),
                pl.concat(out, how="horizontal"),
            ],
            how="horizontal",
        )

        out_c_w = []
        for c, w in iter_c_w:
            lower = c - w / 2
            upper = c + w / 2

            c_fix = int(6e5) if k not in ["isafe", "ihs", "nsl"] else c

            cut_values = (
                binned_values.filter(
                    (pl.col("center") == c_fix)
                    & (pl.col("positions") >= lower)
                    & (pl.col("positions") <= upper)
                )
                .select(pl.exclude("positions"))
                .fill_nan(None)
                .mean()
            )

            cut_values = cut_values.with_columns(
                [
                    pl.lit(_iter).alias("iter"),
                    pl.lit(c).alias("center"),
                    pl.lit(w).alias("window"),
                ]
            )

            out_c_w.append(cut_values)
        df_out.append(pl.concat(out_c_w, how="vertical"))

    df_out = reduce(
        lambda left, right: left.join(
            right,
            on=["iter", "center", "window"],
            how="full",
            coalesce=True,
        ),
        df_out,
    ).sort(["iter", "center", "window"])

    return df_out


################## Haplotype length stats


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

    # scan forward
    ihh0_fwd, ihh1_fwd = ihh01_scan(h, gaps, **kwargs)

    # scan backward
    ihh0_rev, ihh1_rev = ihh01_scan(h[::-1], gaps[::-1], **kwargs)

    # handle reverse scan
    ihh0_rev = ihh0_rev[::-1]
    ihh1_rev = ihh1_rev[::-1]

    # compute unstandardized score
    ihh0 = ihh0_fwd + ihh0_rev
    ihh1 = ihh1_fwd + ihh1_rev

    # og estimation
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

    return df_ihs


def haf_top(hap, pos, cutoff=0.1, start=None, stop=None):
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

    freqs = hap.sum(axis=1) / hap.shape[1]
    hap_tmp = hap[(freqs > 0) & (freqs < 1)]
    haf_num = (np.dot(hap_tmp.T, hap_tmp) / hap.shape[1]).sum(axis=1)
    haf_den = hap_tmp.sum(axis=0)

    haf = np.sort(haf_num / haf_den)

    idx_low = int(cutoff * haf.size)
    idx_high = int((1 - cutoff) * haf.size)

    # 10% higher
    return haf[idx_high:].sum()


@njit
def process_hap_map(ts, rec_map):
    derived_freq = ts.sum(1) / ts.shape[1]
    okfreq_indices = np.where((derived_freq >= 0.05) & (derived_freq <= 1))[0] + 1

    # okfreq = {i: "yes" for i in okfreq_indices}

    coord = rec_map[okfreq_indices - 1, -1].astype(np.int64)
    int_coord = (coord // 100) * 100
    coords = {}
    haplos = {}
    true_coords = {}
    count_coords = {}

    coords = {v: "" for v in int_coord}
    for i, v in enumerate(int_coord):
        coord_index = okfreq_indices[i]
        coords[v] += f"{coord[i]} "

        true_coords[coord[i]] = coord_index
        count_coords[coord_index] = coord[i]

        haplos[coord[i]] = "".join(map(str, ts[coord_index - 1]))

    return coords, haplos, true_coords, count_coords


def h12_enard(ts, rec_map, window_size=500000):
    coords, haplos, true_coords, count_coords = process_hap_map(ts, rec_map)

    maxhaplos = {}
    secondhaplos = {}
    thirdhaplos = {}
    keep_haplo_freq = {}

    key_001 = 600000
    coord = key_001
    int_coord = (coord // 100) * 100
    inf = int_coord - window_size // 2
    sup = int_coord + window_size // 2
    hap_line = "1" * ts.shape[1]
    hap = list(hap_line)

    ongoing_haplos = defaultdict(str)

    for i in range(1, window_size // 200):
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

    for i in range(1, window_size // 200):
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

    max_haplo = ""
    second_haplo = ""
    third_haplo = ""

    best_haplos = {}
    revert_number = defaultdict(str)

    # Populate revert_number dictionary
    for key_numb in sorted(haplos_number.keys()):
        number = haplos_number[key_numb]
        revert_number[number] += f"{key_numb}_"

    counter_rev = 0
    done_rev = 0

    # Sort revert_number keys in descending order and process
    for key_rev in sorted(revert_number.keys(), reverse=True):
        chain = revert_number[key_rev]
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
    split_haplos = {key: best_haplos[key].split() for key in sorted_keys}

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
                    identical, different, total = compare_haplos(haplo_1, haplo_2)

                    if total > 0 and different / total <= 0.2:
                        similar_pairs[key_comp] += f"{key_comp2} "
                        done[pair_key_1_2] = "yes"
                        done[f"{key_comp2} {key_comp}"] = "yes"

    exclude = {}
    counter_rev2 = 0
    max_haplo = ""
    second_haplo = ""
    third_haplo = ""

    for key_rev2 in sorted(similar_pairs, key=int):
        if key_rev2 not in exclude:
            chain = best_haplos[key_rev2]
            similar = similar_pairs[key_rev2]
            if similar != "":
                splitter_similar = similar.split()
                for cur_rev in splitter_similar:
                    exclude[cur_rev] = "yes"
                    chain += "_" + best_haplos[cur_rev]

            counter_rev2 += 1

            if counter_rev2 == 1:
                max_haplo = chain
            elif counter_rev2 == 2:
                second_haplo = chain
            elif counter_rev2 == 3:
                third_haplo = chain
                break

    freq_1 = 0
    freq_2 = 0
    freq_3 = 0
    toto = 0

    for key_ongo2 in sorted(ongoing_haplos.keys()):
        ongoing = ongoing_haplos[key_ongo2]
        toto += 1

        if ongoing in max_haplo:
            freq_1 += 1
        elif ongoing in second_haplo:
            freq_2 += 1
        elif ongoing in third_haplo:
            freq_3 += 1

    H12 = ((freq_1 / toto) + (freq_2 / toto)) ** 2

    return H12


def compare_haplos(haplo_1, haplo_2):
    identical = haplo_1.count("1")  # Count "1"s in haplo_1
    different = sum(1 for h1, h2 in zip(haplo_1, haplo_2) if h1 != h2)
    total = identical + different  # Total equals identical + different

    return identical, different, total


def run_h12(
    hap,
    rec_map,
    _iter=1,
    neutral=True,
    script="/home/jmurgamoreno/software/calculate_H12_modified.pl",
):
    df_hap = pd.DataFrame(hap)
    df_rec_map = pd.DataFrame(rec_map)
    hap_file = "/tmp/tmp_" + str(_iter) + ".hap"
    map_file = "/tmp/tmp_" + str(_iter) + ".map"
    with open(hap_file, "w") as f:
        for row in df_hap.itertuples(index=False, name=None):
            f.write("".join(map(str, row)) + "\n")

    df_rec_map.to_csv(map_file, index=False, header=None, sep=" ")

    h12_enard = "perl " + script + " " + hap_file + " " + map_file + " out "
    h12_enard += "500000 " if neutral else "1200000"

    with subprocess.Popen(h12_enard.split(), stdout=subprocess.PIPE) as process:
        h12_v = float(process.stdout.read())

    os.remove(hap_file)
    os.remove(map_file)

    return h12_v


################## FS stats


@njit
def sq_freq_pairs(hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size):
    # Compute counts and frequencies
    hap_derived = hap
    hap_ancestral = np.bitwise_xor(hap_derived, 1)
    derived_count = ac[:, 1]
    ancestral_count = ac[:, 0]
    freqs = ac[:, 1] / ac.sum(axis=1)

    # Focal filter
    focal_filter = (freqs >= min_focal_freq) & (freqs <= max_focal_freq)
    focal_derived = hap_derived[focal_filter, :]
    focal_derived_count = derived_count[focal_filter]
    focal_ancestral = hap_ancestral[focal_filter, :]
    focal_ancestral_count = ancestral_count[focal_filter]
    focal_index = focal_filter.nonzero()[0]

    # Allocate fixed-size lists to avoid growing lists
    sq_out = [np.zeros((0, 3))] * len(focal_index)
    # info = [None] * len(focal_index)
    info = np.zeros((len(focal_index), 4))
    # Main loop to calculate frequencies
    for j in range(len(focal_index)):
        i = focal_index[j]
        size = window_size / 2

        # Find indices within the window
        # z = np.flatnonzero(np.abs(rec_map[i, -1] - rec_map[:, -1]) <= size)
        mask = np.abs(rec_map[i, -1] - rec_map[:, -1]) <= size
        z = np.where(mask)[0]

        # Index range
        x_r, y_r = i + 1, z[-1]
        x_l, y_l = z[0], i - 1

        # Calculate derived and ancestral frequencies
        f_d_l = (
            np.sum(focal_derived[j] & hap_derived[x_l : y_l + 1], axis=1)
            / focal_derived_count[j]
        )
        f_a_l = (
            np.sum(focal_ancestral[j] & hap_derived[x_l : y_l + 1], axis=1)
            / focal_ancestral_count[j]
        )
        f_tot_l = freqs[x_l : y_l + 1]

        f_d_r = (
            np.sum(focal_derived[j] & hap_derived[x_r : y_r + 1], axis=1)
            / focal_derived_count[j]
        )
        f_a_r = (
            np.sum(focal_ancestral[j] & hap_derived[x_r : y_r + 1], axis=1)
            / focal_ancestral_count[j]
        )
        f_tot_r = freqs[x_r : y_r + 1]

        # Concatenate frequencies into a single array
        sq_freqs = np.empty((f_d_l.size + f_d_r.size, 3))
        sq_freqs[: f_d_l.size, 0] = f_d_l[::-1]
        sq_freqs[: f_d_l.size, 1] = f_a_l[::-1]
        sq_freqs[: f_d_l.size, 2] = f_tot_l[::-1]
        sq_freqs[f_d_l.size :, 0] = f_d_r
        sq_freqs[f_d_l.size :, 1] = f_a_r
        sq_freqs[f_d_l.size :, 2] = f_tot_r

        sq_out[j] = sq_freqs
        info[j] = np.array(
            [rec_map[i, -1], freqs[i], focal_derived_count[j], focal_ancestral_count[j]]
        )

    return sq_out, info


def s_ratio(
    hap,
    ac,
    rec_map,
    max_ancest_freq=1,
    min_tot_freq=0,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
):
    sq_freqs, info = sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size
    )
    results = []

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
        s_ratio_v_flip = den / num
        results.append((s_ratio_v, s_ratio_v_flip))

    try:
        # out = np.hstack([info, np.array(results).reshape(len(results), 1)])
        out = np.hstack([info, np.array(results)])
        df_out = pl.DataFrame(
            {
                "positions": pl.Series(out[:, 0], dtype=pl.Int64),
                "daf": out[:, 1],
                "s_ratio": out[:, 4],
                "s_ratio_flip": out[:, 5],
            }
        )
    except:
        df_out = pl.DataFrame(
            {"positions": [], "daf": [], "s_ratio": [], "s_ratio_flip": []}
        )

    return df_out


def hapdaf_o(
    hap,
    ac,
    rec_map,
    max_ancest_freq=0.25,
    min_tot_freq=0.25,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
):
    sq_freqs, info = sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size
    )

    results = []
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

        # Flipping derived to ancestral, ancestral to derived
        f_d2f = (
            f_a[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )
        f_a2f = (
            f_d[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )

        if len(f_d2) != 0 and len(f_a2) != 0:
            hapdaf = (f_d2 - f_a2).sum() / f_d2.shape[0]
        else:
            hapdaf = np.nan

        if len(f_d2f) != 0 and len(f_a2f) != 0:
            hapdaf_flip = (f_d2f - f_a2f).sum() / f_d2f.shape[0]
        else:
            hapdaf_flip = np.nan

        results.append((hapdaf, hapdaf_flip))

    try:
        out = np.hstack(
            [
                info,
                np.array(results),
                # np.array(results).reshape(len(results), 1),
            ]
        )
        df_out = pl.DataFrame(
            {
                "positions": pl.Series(out[:, 0], dtype=pl.Int64),
                "daf": out[:, 1],
                "hapdaf_o": out[:, 4],
                "hapdaf_o_flip": out[:, 5],
            }
        )
    except:
        df_out = pl.DataFrame(
            {"positions": [], "daf": [], "hapdaf_o": [], "hapdaf_o_flip": []}
        )

    return df_out


def hapdaf_s(
    hap,
    ac,
    rec_map,
    max_ancest_freq=0.1,
    min_tot_freq=0.1,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
):
    sq_freqs, info = sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size
    )
    results = []
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

        # Flipping derived to ancestral, ancestral to derived
        f_d2f = (
            f_a[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )
        f_a2f = (
            f_d[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )

        if len(f_d2) != 0 and len(f_a2) != 0:
            hapdaf = (f_d2 - f_a2).sum() / f_d2.shape[0]
        else:
            hapdaf = np.nan

        if len(f_d2f) != 0 and len(f_a2f) != 0:
            hapdaf_flip = (f_d2f - f_a2f).sum() / f_d2f.shape[0]
        else:
            hapdaf_flip = np.nan

        results.append((hapdaf, hapdaf_flip))

    try:
        out = np.hstack(
            [
                info,
                np.array(results),
            ]
        )
        df_out = pl.DataFrame(
            {
                "positions": pl.Series(out[:, 0], dtype=pl.Int64),
                "daf": out[:, 1],
                "hapdaf_s": out[:, 4],
                "hapdaf_s_flip": out[:, 5],
            }
        )
    except:
        df_out = pl.DataFrame(
            {"positions": [], "daf": [], "hapdaf_s": [], "hapdaf_s_flip": []}
        )

    return df_out


def dind_high_low(
    hap,
    ac,
    rec_map,
    max_ancest_freq=0.25,
    min_tot_freq=0,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
):
    # Extract frequency pairs and info array
    sq_freqs, info = sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size
    )

    # Pre-allocate arrays for results to avoid growing lists
    n_rows = len(sq_freqs)
    results_dind = np.empty((n_rows, 2), dtype=np.float64)
    results_high = np.empty((n_rows, 2), dtype=np.float64)
    results_low = np.empty((n_rows, 2), dtype=np.float64)

    # Main computation loop
    for i, v in enumerate(sq_freqs):
        f_d = v[:, 0]
        f_a = v[:, 1]

        focal_derived_count = info[i][-2]
        focal_ancestral_count = info[i][-1]

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

        results_dind[i] = [dind_v, dind_v_flip]

        # Calculate high and low frequency values
        hf_v = (f_d[f_d > max_ancest_freq] ** 2).sum() / max(
            len(f_d[f_d > max_ancest_freq]), 1
        )
        hf_v_flip = (f_a[f_a > max_ancest_freq] ** 2).sum() / max(
            len(f_a[f_a > max_ancest_freq]), 1
        )
        results_high[i] = [hf_v, hf_v_flip]

        lf_v = ((1 - f_d[f_d < max_ancest_freq]) ** 2).sum() / max(
            len(f_d[f_d < max_ancest_freq]), 1
        )
        lf_v_flip = ((1 - f_a[f_a < max_ancest_freq]) ** 2).sum() / max(
            len(f_a[f_a < max_ancest_freq]), 1
        )
        results_low[i] = [lf_v, lf_v_flip]

        # Free memory explicitly for large arrays
        del f_d, f_a, f_d2, f_a2

    # Final DataFrame creation
    try:
        out = np.hstack([info, results_dind, results_high, results_low])
        df_out = pl.DataFrame(
            {
                "positions": pl.Series(out[:, 0], dtype=pl.Int64),
                "daf": out[:, 1],
                "dind": out[:, 4],
                "dind_flip": out[:, 5],
                "high_freq": out[:, 6],
                "high_freq_flip": out[:, 7],
                "low_freq": out[:, 8],
                "low_freq_flip": out[:, 9],
            }
        )

    except:
        df_out = pl.DataFrame(
            {
                "positions": [],
                "daf": [],
                "dind": [],
                "dind_flip": [],
                "high_freq": [],
                "high_freq_flip": [],
                "low_freq": [],
                "low_freq_flip": [],
            }
        )

    return df_out


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
def dot_two_nb(x, y):
    return np.dot(x, y)


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


def Ld(
    hap: np.ndarray, pos: np.ndarray, min_freq=0.05, max_freq=1, start=None, stop=None
) -> tuple:
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

    if start is not None or stop is not None:
        loc = (pos >= start) & (pos <= stop)
        pos = pos[loc]
        hap = hap[loc, :]

    freqs = hap.sum(axis=1) / hap.shape[1]

    hap_filter = hap[(freqs >= min_freq) & (freqs <= max_freq)]

    r2_matrix = compute_r2_matrix(hap_filter)
    # r2_matrix = r2_torch(hap_filter)
    S = hap_filter.shape[0]
    zns = r2_matrix.sum() / comb(S, 2)
    # Index combination to iter
    # omega_max = omega(r2_matrix)
    # omega_max = dps.omega(r2_matrix)[0]

    return zns, 0
    # return zns, omega_max


def r2_matrix(
    hap: np.ndarray, pos: np.ndarray, min_freq=0.05, max_freq=1, start=None, stop=None
):
    """
    Compute Kelly Zns statistic (1997) and omega_max. Average r2
    among every pair of loci in the genomic window.

    Args:
        hap (numpy.ndarray): 2D array representing haplotype data. Rows correspond to different mutations, and columns correspond to chromosomes.
        pos (numpy.ndarray): 1D array representing the positions of mutations.
        min_freq (float, optional): Minimum frequency threshold. Default is 0.05.
        max_freq (float, optional): Maximum frequency threshold. Default is 1.
        window (int, optional): Genomic window size. Default is 500000.

    Returns: tuple: A tuple containing two values:
        - kelly_zns (float): Kelly Zns statistic.
        - omega_max (float): Nielsen omega max.
    """

    # if start is not None or stop is not None:
    #     loc = (pos >= start) & (pos <= stop)
    #     pos = pos[loc]
    #     hap = hap[loc, :]

    freqs = hap.sum(axis=1) / hap.shape[1]
    freq_filter = (freqs >= min_freq) & (freqs <= max_freq)
    hap_filter = hap[freq_filter]

    r2_matrix = compute_r2_matrix(hap_filter)
    # r2_matrix = r2_torch(hap_filter)
    # S = hap_filter.shape[0]
    # zns = r2_matrix.sum() / comb(S, 2)
    # Index combination to iter
    # omega_max = omega(r2_matrix)
    # omega_max = dps.omega(r2_matrix)[0]

    return r2_matrix, freq_filter
    # return zns, omega_max


def Ld(
    r2_subset,
    freq_filter,
    pos: np.ndarray,
    min_freq=0.05,
    max_freq=1,
    start=None,
    stop=None,
):
    pos_filter = pos[freq_filter]
    if start is not None or stop is not None:
        loc = (pos_filter >= start) & (pos_filter <= stop)
        pos_filter = pos_filter[loc]
        r2_subset = r2_subset[loc, :][:, loc]

    # r2_subset_matrix = compute_r2_subset_matrix(hap_filter)
    # r2_subset_matrix = r2_subset_torch(hap_filter)
    S = r2_subset.shape[0]
    kelly_zns = r2_subset.sum() / comb(S, 2)
    # omega_max = omega(r2_subset)

    return kelly_zns, 0


@njit("float64(int8[:], int8[:])", cache=True)
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


@njit("float64[:,:](int8[:,:])", cache=True)
def compute_r2_matrix(hap):
    num_sites = hap.shape[0]

    # r2_matrix = OrderedDict()
    sum_r_squared = 0
    r2_matrix = np.zeros((num_sites, num_sites))
    # Avoid itertool.combination, not working on numba
    # for pair in combinations(range(num_sites), 2):

    # Check index from triangular matrix of size num_sites x num_sites. Each indices correspond to one one dimension of the array. Same as combinations(range(num_sites), 2)
    c_1, c_2 = np.triu_indices(num_sites, 1)

    for i, j in zip(c_1, c_2):
        r2_matrix[i, j] = r2(hap[i, :], hap[j, :])
        # r2_matrix[pair[0], pair[1]] = r2(hap[pair[0], :], hap[pair[1], :])

    return r2_matrix


@njit("float64(float64[:,:])", cache=True)
def omega(r2_matrix):
    """
    Calculates Kim and Nielsen's (2004, Genetics 167:1513) omega_max statistic. Adapted from PG-Alignments-GAN

    Args:r2_matrix (numpy.ndarray): 2D array representing r2 values.

    Returns:
        float: Kim and Nielsen's omega max.
    """

    omega_max = 0
    S_ = r2_matrix.shape[1]

    if S_ < 3:
        omega_max = 0
    else:
        for l_ in range(3, S_ - 2):
            sum_r2_L = 0
            sum_r2_R = 0
            sum_r2_LR = 0

            for i in range(S_):
                for j in range(i + 1, S_):
                    ld_calc = r2_matrix[i, j]
                    if i < l_ and j < l_:
                        sum_r2_L += ld_calc

                    elif i >= l_ and j >= l_:
                        sum_r2_R += ld_calc

                    elif i < l_ and j >= l_:
                        sum_r2_LR += ld_calc

            # l_ ## to keep the math right outside of indexing
            omega_numerator = (
                1 / ((l_ * (l_ - 1) / 2) + ((S_ - l_) * (S_ - l_ - 1) / 2))
            ) * (sum_r2_L + sum_r2_R)
            omega_denominator = (1 / (l_ * (S_ - l_))) * sum_r2_LR

            if omega_denominator == 0:
                omega = 0
            else:
                omega = np.divide(omega_numerator, omega_denominator)

            if omega > omega_max:
                omega_max = omega

    return omega_max


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
    F = (pi - ss) / sqrt(u * S + v * np.power(S, 2))

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
    D = (S - ss * an) / sqrt(u * S + v * np.power(S, 2))
    return D


################## LASSI
def get_empir_freqs_np(hap):
    """
    Calculate the empirical frequencies of haplotypes.

    Parameters:
    - hap (numpy.ndarray): Array of haplotypes where each column represents an individual and each row represents a SNP.

    Returns:
    - k_counts (numpy.ndarray): Counts of each unique haplotype.
    - h_f (numpy.ndarray): Empirical frequencies of each unique haplotype.
    """
    S, n = hap.shape

    # Count occurrences of each unique haplotype
    hap_f, k_counts = np.unique(hap, axis=1, return_counts=True)

    # Sort counts in descending order
    k_counts = np.sort(k_counts)[::-1]

    # Calculate empirical frequencies
    h_f = k_counts / n
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
    ts, rec_map, K_truncation: int, window: int, step: int
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
    (
        hap_01,
        ac,
        biallelic_mask,
        hap_int,
        rec_map_01,
        position_masked,
        sequence_length,
        freqs,
    ) = filter_gt(ts, rec_map)

    K_count = []
    K_spectrum = []
    windows_centers = []
    S, n = hap_int.shape
    for i in range(0, S, step):
        hap_subset = hap_int[i : i + window, :]

        # Calculate window center based on median SNP position
        windows_centers.append(np.median(position_masked[i : i + window]))

        # Compute empirical frequencies and process spectra for the window
        k, h_f = get_empir_freqs_np(hap_subset)
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


def T_m_statistic(K_counts, K_neutral, windows, K_truncation, sweep_mode=5, i=0):
    output = []
    m_vals = K_truncation + 1
    epsilon_min = 1 / (K_truncation * 100)

    _epsilon_values = list(map(lambda x: x * epsilon_min, range(1, 101)))
    epsilon_max = K_neutral[-1]
    epsilon_values = []

    for ev in _epsilon_values:
        # ev = e * epsilon_min
        if ev <= epsilon_max:
            epsilon_values.append(ev)
    epsilon_values = np.array(epsilon_values)

    for j, w in enumerate(windows):
        # if(i==132):
        # break
        K_iter = K_counts[j]

        null_likelihood = easy_likelihood(K_neutral, K_iter, K_truncation)

        alt_likelihoods_by_e = []

        for e in epsilon_values:
            alt_likelihoods_by_m = []
            for m in range(1, m_vals):
                alt_like = sweep_likelihood(
                    K_neutral, K_iter, K_truncation, m, e, epsilon_max
                )
                alt_likelihoods_by_m.append(alt_like)

            alt_likelihoods_by_m = np.array(alt_likelihoods_by_m)
            likelihood_best_m = 2 * (alt_likelihoods_by_m.max() - null_likelihood)

            if likelihood_best_m > 0:
                ml_max_m = (alt_likelihoods_by_m.argmax()) + 1
            else:
                ml_max_m = 0

            alt_likelihoods_by_e.append([likelihood_best_m, ml_max_m, e])

        alt_likelihoods_by_e = np.array(alt_likelihoods_by_e)

        likelihood_real = max(alt_likelihoods_by_e[:, 0])

        out_index = np.flatnonzero(alt_likelihoods_by_e[:, 0] == likelihood_real)

        out_intermediate = alt_likelihoods_by_e[out_index]

        if out_intermediate.shape[0] > 1:
            constarg = min(out_intermediate[:, 1])

            outcons = np.flatnonzero(out_intermediate[:, 1] == constarg)

            out_cons_intermediate = out_intermediate[outcons]

            if out_cons_intermediate.shape[0] > 1:
                out_cons_intermediate = out_cons_intermediate[0]

            out_intermediate = out_cons_intermediate

        outshape = out_intermediate.shape

        if len(outshape) != 1:
            out_intermediate = out_intermediate[0]

        out_intermediate = np.concatenate(
            [out_intermediate, np.array([K_neutral[-1], sweep_mode, w]), K_iter]
        )

        output.append(out_intermediate)

    # output = np.array(output)
    # return output[output[:, 0].argmax(), :]

    K_names = ["Kcounts_" + str(i) for i in range(1, K_iter.size + 1)]
    output = pd.DataFrame(output)
    output.insert(output.shape[1], "iter", i)

    output.columns = (
        [
            "t_statistic",
            "m",
            "frequency",
            "e",
            "model",
            "window_lassi",
        ]
        + K_names
        + ["iter"]
    )
    return output


def neutral_hfs(sims, K_truncation, w_size, step, nthreads=1):
    pars = [(i[0], i[1]) for i in sims]

    # Use joblib to parallelize the execution
    hfs_stats = Parallel(n_jobs=nthreads, verbose=5)(
        delayed(LASSI_spectrum_and_Kspectrum)(ts, rec_map, K_truncation, w_size, step)
        for index, (ts, rec_map) in enumerate(pars, 1)
    )

    K_counts, K_spectrum, windows = zip(*hfs_stats)

    return neut_average(np.vstack(K_spectrum))

    # t_m = Parallel(n_jobs=nthreads, verbose=5)(
    #     delayed(T_m_statistic)(kc, K_neutral, windows[index], K_truncation)
    #     for index, (kc) in enumerate(K_counts)
    # )
    # return (
    #     pd.DataFrame(t_m, columns=["t", "m", "frequency", "e", "model", "window"]),
    #     K_neutral,
    # )


def compute_t_m(
    sims,
    K_truncation,
    w_size,
    step,
    K_neutral=None,
    windows=[50000, 100000, 200000, 500000, 1000000],
    center=[5e5, 7e5],
    nthreads=1,
):
    pars = [(i[0], i[1]) for i in sims]

    # Log the start of the scheduling

    # Use joblib to parallelize the execution
    hfs_stats = Parallel(n_jobs=nthreads, verbose=5)(
        delayed(LASSI_spectrum_and_Kspectrum)(ts, rec_map, K_truncation, w_size, step)
        for index, (ts, rec_map) in enumerate(pars, 1)
    )

    K_counts, K_spectrum, windows_lassi = zip(*hfs_stats)

    if K_neutral is None:
        K_neutral = neut_average(np.vstack(K_spectrum))

    t_m = Parallel(n_jobs=nthreads, verbose=5)(
        delayed(T_m_statistic)(
            kc, K_neutral, windows_lassi[index - 1], K_truncation, i=index
        )
        for index, (kc) in enumerate(K_counts, 1)
    )
    t_m_cut = Parallel(n_jobs=nthreads, verbose=0)(
        delayed(cut_t_m_argmax)(t, windows=windows, center=center) for t in t_m
    )

    return pd.concat(t_m_cut)


def cut_t_m(df_t_m, windows=[50000, 100000, 200000, 500000, 1000000], center=6e5):
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
    center=[5e5, 7e5],
    step=1e4,
):
    out = []
    centers = np.arange(center[0], center[1] + step, step).astype(int)
    iter_c_w = list(product(centers, windows))
    for c, w in iter_c_w:
        # for w in [1000000]:
        lower = c - w / 2
        upper = c + w / 2

        df_t_m_subset = df_t_m[
            (df_t_m.iloc[:, 5] > lower) & (df_t_m.iloc[:, 5] < upper)
        ]
        try:
            max_t = df_t_m_subset.iloc[:, 0].argmax()

            # df_t_m_subset = df_t_m_subset[df_t_m_subset.m > 0]
            # max_t = df_t_m_subset[df_t_m_subset.m > 0].m.argmin()
            df_t_m_subset = df_t_m_subset.iloc[max_t : max_t + 1, :]

            df_t_m_subset = df_t_m_subset.loc[
                :,
                ~df_t_m_subset.columns.isin(
                    ["iter", "frequency", "e", "model", "window_lassi"]
                ),
            ]
            df_t_m_subset.insert(0, "window", w)
            df_t_m_subset.insert(0, "center", c)
            df_t_m_subset.insert(0, "iter", df_t_m.iter.unique())

            out.append(df_t_m_subset)

        except:
            K_names = pd.DataFrame(
                {
                    k: 0
                    for k in df_t_m.columns[
                        df_t_m.columns.str.contains("Kcount")
                    ].values
                },
                index=[0],
            )

            out.append(
                pd.concat(
                    [
                        pd.DataFrame(
                            {
                                "iter": df_t_m["iter"].unique(),
                                "center": c,
                                "window": w,
                                "t_statistic": 0,
                                "m": 0,
                            }
                        ),
                        K_names,
                    ],
                    axis=1,
                )
            )

    out = pd.concat(out).reset_index(drop=True)

    return out


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
        raise IOError(f"File {ms_file} is malformed.")

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
            raise IOError(f"File {ms_file} is malformed.")

        # data = np.vstack(data).T
        haps.append(data)

    if param is None:
        param = np.zeros(4)

    return (haps[0], rec_map[0], param)
