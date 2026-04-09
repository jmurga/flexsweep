"""
flexsweep/cms.py
================
Composite of Multiple Signals (CMS) — standalone per-SNP selection scan module.

Two input modes:

  Option A  (fast): pre-computed save_stats pickles from fvs-discoal/fvs-vcf
      ``--save_stats`` flag.  No extra computation.

  Option A2 (fallback): raw simulation directory or VCF directory.  Per-SNP
      stats are computed from scratch using the same functions as fv.py, but
      without any window aggregation or CNN feature-vector collapsing.  The
      resulting pickle is cached to disk for faster reruns.

References
----------
Grossman et al. 2010 *Science* 327:883  (original CMS method)
Grossman et al. 2013 *Cell* 152:703     (empirical application; canonical reference)
broadinstitute/cms GitHub               (histogram construction, selDAF logic)
Ma et al. 2015 *Heredity* 115:426      (DCMS; de-correlation extension)
"""

import glob
import logging
import os
import warnings
from functools import reduce

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from .fv import (
    center_window_cols,
    genome_reader,
    get_cm,
    ihs_ihh,
    load_pickle,
    parse_ms_numpy,
    rank_with_duplicates,
    run_fs_stats,
    run_isafe,
    save_pickle,
    snps_to_r_bins,
    summaries,
)

try:
    from allel import nsl as _allel_nsl
except ImportError:  # pragma: no cover
    _allel_nsl = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: All per-SNP stats that CMS can incorporate (Option A2 computes all of these).
#: Stats from the save_stats pickle (Option A) depend on what fvs-discoal/fvs-vcf computed.
SUPPORTED_STATS: list[str] = [
    "ihs",  # log(iHH_A/iHH_D) — haplotype homozygosity ratio
    "delta_ihh",  # |iHH_A − iHH_D| — absolute iHH difference
    "nsl",  # number of segregating sites by length
    "isafe",  # integrated selection of allele favoured by environment
    "dind",  # derived/ancestral inter-SNP distance ratio
    "high_freq",  # high-frequency derived allele excess
    "low_freq",  # low-frequency derived allele deficit
    "s_ratio",  # singleton ratio
    "hapdaf_o",  # haplotype allele frequency
    "hapdaf_s",  # haplotype allele frequency
]

#: Default stat set — single-population, low pairwise correlation
DEFAULT_STATS = ["ihs", "delta_ihh", "dind", "hapdaf_o"]

#: selDAF thresholds matching broadinstitute/cms C++ ``combine_scores_gw.c``
_DAF_THRESHOLDS: tuple[float, float] = (0.35, 0.65)

#: Stats folded (|value|) before histogram lookup; iHS is symmetric w.r.t. direction
_FOLD_STATS: frozenset[str] = frozenset({"ihs", "nsl"})

#: Interior fraction of the locus used for histogram building (avoids edge effects)
#: Matches CMS C++ ``edge = 250000`` on a 1.5 Mb chromosome → 250/1500 ≈ 0.167
#: We use 0.25 (conservative) → interior = [locus*0.25, locus*0.75]
_INTERIOR_FRAC: float = 0.25


# ---------------------------------------------------------------------------
# Recombination bin helpers
# ---------------------------------------------------------------------------


def _assign_r_bin_scalar(r_val: float, breaks: list) -> str | None:
    """
    Return the polars-cut interval label for a single recombination rate value.

    Uses ``pl.Series.cut`` so the label format is identical to the one produced
    by the VCF path (which also uses ``pl.cut`` internally via ``snps_to_r_bins``),
    ensuring dict-key consistency between the two paths.
    """
    if not np.isfinite(r_val):
        return None
    result = pl.Series("r", [r_val]).cut(breaks=list(breaks))
    s = str(result[0])
    return None if s in ("null", "None") else s


def _load_recombination_map(path: str) -> pl.DataFrame:
    """
    Load a recombination map file (TSV, optionally gzipped).

    Expected columns: chr, end (physical position), cm (cumulative cM).
    Additional columns are ignored.
    """
    df = pl.read_csv(path, separator="\t", infer_schema_length=10_000)
    required = {"chr", "end", "cm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Recombination map '{path}' is missing columns: {missing}. "
            f"Required: chr, end, cm."
        )
    return df


def _compute_vcf_snp_r_bins(
    snps_df: pl.DataFrame,
    region_key: str,
    df_rec_map: pl.DataFrame,
    r_bins_breaks: list,
) -> pl.Series:
    """
    Assign recombination bins to VCF SNPs using a genome-wide recombination map.

    Creates a 100 kb window grid spanning the region, computes cM/Mb per window
    via ``get_cm``, then assigns each SNP to the nearest window's r_bin via
    ``snps_to_r_bins``.

    Returns a Polars Series of categorical r_bin strings (same format as the
    sims path), with ``null`` for SNPs that could not be assigned.
    """
    chrom = str(region_key).split(":")[0]
    df_rec_chrom = df_rec_map.filter(pl.col("chr") == chrom)
    if len(df_rec_chrom) == 0:
        logger.warning(
            "No recombination map entries for chromosome '%s' — r_bins will be null.",
            chrom,
        )
        return pl.Series("r_bins", [None] * len(snps_df), dtype=pl.Categorical)

    positions = snps_df["positions"].to_numpy()
    window_size = 100_000
    pos_min = max(0, int(positions.min()) - window_size)
    pos_max = int(positions.max()) + window_size

    starts = np.arange(pos_min, pos_max, window_size, dtype=np.int64)
    ends = starts + window_size
    center_coords = np.column_stack([starts, ends])

    try:
        cm_mb_df = get_cm(df_rec_chrom, center_coords, cm_mb=True)
    except Exception as exc:
        logger.warning(
            "get_cm failed for %s: %s — r_bins will be null.", region_key, exc
        )
        return pl.Series("r_bins", [None] * len(snps_df), dtype=pl.Categorical)

    if len(cm_mb_df) == 0:
        return pl.Series("r_bins", [None] * len(snps_df), dtype=pl.Categorical)

    df_windows = cm_mb_df.with_columns(
        [
            pl.col("cm_mb").cast(pl.Float32),
            pl.col("cm_mb").cut(breaks=list(r_bins_breaks)).alias("r_bins"),
        ]
    )

    snps_with_chr = snps_df.with_columns(pl.lit(chrom).alias("chr"))
    try:
        snps_rbinned = snps_to_r_bins(snps_with_chr, df_windows)
        return snps_rbinned["r_bins"]
    except Exception as exc:
        logger.warning(
            "snps_to_r_bins failed for %s: %s — r_bins will be null.", region_key, exc
        )
        return pl.Series("r_bins", [None] * len(snps_df), dtype=pl.Categorical)


# ---------------------------------------------------------------------------
# Internal: build histograms from an already-loaded sims data dict
# ---------------------------------------------------------------------------


def _build_tables_from_data(
    data: dict,
    stats: list[str],
    n_bins: int,
    daf_thresholds: tuple[float, float],
    locus_length: int,
    fold_stats: frozenset[str],
    r_bins_breaks: list | None = None,
) -> dict:
    """
    Build per-stat PMF histograms from a loaded sims data dict.

    *data* has the same structure as a save_stats pickle:
    ``{sim_type: summaries(stats=list_of_{"snps":df,...}, parameters=arr)}``

    When *r_bins_breaks* is ``None`` the returned dict has flat structure::

        {stat: {"bin_edges": …, "neut": …, "sweep_low": …, "sweep_mid": …, "sweep_high": …}}

    When *r_bins_breaks* is provided the returned dict has one entry per r-bin
    plus an ``"all"`` aggregate fallback::

        {"all":        {stat: {…}},
         "(lo, hi]":   {stat: {…}},
         …}

    Bin edges are computed from the global distribution (all r-bins combined)
    so that the edge breaks are identical across r-bins — this makes the bin
    indices pre-computable once per stat in :func:`cms_score`.
    """
    interior_lo = locus_length * _INTERIOR_FRAC
    interior_hi = locus_length * (1.0 - _INTERIOR_FRAC)
    lo_thr, hi_thr = daf_thresholds

    # Sentinel for "aggregate over all r-bins"
    _ALL = "all"

    # Storage: {r_bin_key: {class_key: {stat: [val_arrays]}}}
    # class_key ∈ {"neut", "sweep_low", "sweep_mid", "sweep_high"}
    # r_bin_key ∈ {_ALL, "(lo, hi]", ...}
    r_bin_keys = [_ALL] if r_bins_breaks is None else [_ALL]  # always include _ALL

    def _empty_class_dict():
        return {
            "neut": {s: [] for s in stats},
            "sweep_low": {s: [] for s in stats},
            "sweep_mid": {s: [] for s in stats},
            "sweep_high": {s: [] for s in stats},
        }

    storage: dict = {_ALL: _empty_class_dict()}

    for sim_type, summary in data.items():
        rep_list = summary.stats  # list of {"snps": df, ...}
        params = summary.parameters  # shape (n_reps, ≥4); col 3 = f_t; col -1 = r

        for i, rep in enumerate(rep_list):
            snps_df = rep["snps"] if isinstance(rep, dict) else rep
            if snps_df is None or len(snps_df) == 0:
                continue

            # Interior filter — avoid EHH edge effects
            snps_df = snps_df.filter(
                (pl.col("positions") >= interior_lo)
                & (pl.col("positions") <= interior_hi)
            )
            if len(snps_df) == 0:
                continue

            # Determine selDAF class key
            if sim_type == "neutral":
                class_key = "neut"
            elif sim_type == "sweep":
                f_t = (
                    float(params[i, 3])
                    if (params is not None and params.ndim > 1)
                    else 0.0
                )
                if f_t <= lo_thr:
                    class_key = "sweep_low"
                elif f_t <= hi_thr:
                    class_key = "sweep_mid"
                else:
                    class_key = "sweep_high"
            else:
                continue

            # Determine r-bin key for this replicate
            if r_bins_breaks is not None and params is not None and params.ndim > 1:
                r_val = (
                    float(params[i, -1]) * 1e8
                )  # convert to cM/Mb (matches _process_sims)
                r_bin_key = _assign_r_bin_scalar(r_val, r_bins_breaks)
                if r_bin_key is not None and r_bin_key not in storage:
                    storage[r_bin_key] = _empty_class_dict()
                    r_bin_keys.append(r_bin_key)
            else:
                r_bin_key = None

            for s in stats:
                if s not in snps_df.columns:
                    continue
                vals = snps_df[s].drop_nulls().to_numpy().astype(np.float64)
                if s in fold_stats:
                    vals = np.abs(vals)
                vals = vals[np.isfinite(vals)]
                if not len(vals):
                    continue
                # Always contribute to the aggregate "all" bucket
                storage[_ALL][class_key][s].append(vals)
                # Also contribute to the r-bin-specific bucket
                if r_bin_key is not None:
                    storage[r_bin_key][class_key][s].append(vals)

    # ------------------------------------------------------------------ #
    # Build PMF histograms.  Bin edges are derived from the global _ALL   #
    # distribution so they are consistent across r-bins.                  #
    # ------------------------------------------------------------------ #

    def _pmf(arrs: list, edges: np.ndarray) -> np.ndarray:
        if not arrs:
            return np.full(n_bins, 1.0 / n_bins)
        combined = np.concatenate(arrs)
        h, _ = np.histogram(combined, bins=edges, density=False)
        total = h.sum()
        if total == 0:
            return np.full(n_bins, 1.0 / n_bins)
        pmf = h.astype(np.float64) / total
        pmf[pmf == 0] = np.finfo(float).tiny  # Laplace pseudocount
        return pmf

    # First pass: compute global edges from _ALL storage
    global_edges: dict[str, np.ndarray] = {}
    for s in stats:
        all_arrs = (
            storage[_ALL]["neut"][s]
            + storage[_ALL]["sweep_low"][s]
            + storage[_ALL]["sweep_mid"][s]
            + storage[_ALL]["sweep_high"][s]
        )
        if not all_arrs:
            logger.warning("No values found for stat '%s' — skipping.", s)
            continue
        all_vals = np.concatenate(all_arrs)
        lo_pct, hi_pct = np.nanpercentile(all_vals, [0.5, 99.5])
        global_edges[s] = np.linspace(lo_pct, hi_pct, n_bins + 1)

    # Second pass: build per-r_bin (or aggregate) stat tables
    def _build_stat_tables(rbin_storage: dict) -> dict:
        stat_tables: dict = {}
        for s in stats:
            if s not in global_edges:
                continue
            edges = global_edges[s]
            stat_tables[s] = {
                "bin_edges": edges,
                "neut": _pmf(rbin_storage["neut"][s], edges),
                "sweep_low": _pmf(rbin_storage["sweep_low"][s], edges),
                "sweep_mid": _pmf(rbin_storage["sweep_mid"][s], edges),
                "sweep_high": _pmf(rbin_storage["sweep_high"][s], edges),
            }
        return stat_tables

    if r_bins_breaks is None:
        # Return flat {stat: {...}} — backward compatible
        return _build_stat_tables(storage[_ALL])

    # Return nested {"all": {stat: {...}}, r_bin: {stat: {...}}, ...}
    result: dict = {}
    for key in [_ALL] + [k for k in storage if k != _ALL]:
        stat_tbl = _build_stat_tables(storage[key])
        if stat_tbl:
            result[key] = stat_tbl
    return result


# ---------------------------------------------------------------------------
# Public API: Step 1 — build_cms_tables
# ---------------------------------------------------------------------------


def build_cms_tables(
    sims_pickle_or_data,
    stats: list[str] | None = None,
    n_bins: int = 60,
    daf_thresholds: tuple[float, float] = _DAF_THRESHOLDS,
    locus_length: int = int(1.2e6),
    fold_stats: frozenset[str] | None = None,
    r_bins: list | None = None,
) -> dict:
    """
    Build per-stat likelihood histograms from simulation data.

    Parameters
    ----------
    sims_pickle_or_data : str or dict
        Either a path to ``raw_statistics.pickle`` from ``fvs-discoal --save_stats``,
        or an already-loaded dict in the same format (returned by
        :func:`_compute_snps_from_sims`).
    stats : list[str] or None
        Stat column names. Default: ``["ihs", "delta_ihh", "dind", "hapdaf_o"]``.
    n_bins : int
        Number of histogram bins per stat per class. Default 60 (matches CMS C++).
    daf_thresholds : tuple[float, float]
        selDAF thresholds splitting low / mid / high sweep distributions.
        Default ``(0.35, 0.65)`` from broadinstitute/cms ``combine_scores_gw.c``.
    locus_length : int
        Simulation locus length in bp.  Interior SNPs (25%–75%) are used to
        avoid haplotype-statistic edge effects.  Default 1,200,000.
    fold_stats : frozenset[str] or None
        Stats to fold (|value|) before binning.  Default: ``{"ihs"}``.
    r_bins : list of float or None
        Recombination-rate bin break points (cM/Mb) — same convention as
        ``_process_sims`` in ``fv.py``.  When provided, separate likelihood
        tables are built for each recombination bin in addition to the
        aggregate ``"all"`` table used as fallback.  The recombination rate
        for each simulation replicate is taken from ``params[:, -1] * 1e8``
        (last column of the params array, scaled to cM/Mb).

    Returns
    -------
    dict
        Without *r_bins*: ``{stat: {"bin_edges": …, "neut": …, "sweep_low": …,
        "sweep_mid": …, "sweep_high": …}}``

        With *r_bins*: ``{"all": {stat: {…}}, "(lo, hi]": {stat: {…}}, …}``
        Each histogram is a proper PMF (sums to 1, all bins > 0).
    """
    if stats is None:
        stats = DEFAULT_STATS
    if fold_stats is None:
        fold_stats = _FOLD_STATS

    if isinstance(sims_pickle_or_data, str):
        data = load_pickle(sims_pickle_or_data)
    else:
        data = sims_pickle_or_data

    return _build_tables_from_data(
        data,
        stats,
        n_bins,
        daf_thresholds,
        locus_length,
        fold_stats,
        r_bins_breaks=r_bins,
    )


# ---------------------------------------------------------------------------
# Public API: Step 2 — cms_score
# ---------------------------------------------------------------------------


def cms_score(
    snps_df: pl.DataFrame,
    tables: dict,
    stats: list[str] | None = None,
    daf_thresholds: tuple[float, float] = _DAF_THRESHOLDS,
    prior_p: float = 0.5,
    fold_stats: frozenset[str] | None = None,
    snps_r_bins: pl.Series | None = None,
) -> pl.DataFrame:
    """
    Compute per-SNP log composite Bayes factor and CMS posterior from observed per-SNP statistics.

    The log composite Bayes factor is::

        log_BF_composite(i) = Σ_k  log[ P(s_k(i) | selected, selDAF_bin[, r_bin]) /
                                         P(s_k(i) | neutral[, r_bin]) ]

    The CMS posterior (Grossman 2010, Eq. 1) with prior π::

        CMS(i) = BF_composite(i)·π / [BF_composite(i)·π + (1−π)]

    Statistics missing at a SNP (NaN) contribute BF = 1 (log_BF = 0) and are
    silently skipped.

    Parameters
    ----------
    snps_df : pl.DataFrame
        Per-SNP DataFrame with at minimum ``positions``, ``daf``, and any stat
        columns listed in *stats*.
    tables : dict
        Output of :func:`build_cms_tables`.

        - **Without r_bins** (``build_cms_tables(r_bins=None)``):
          ``{stat: {"bin_edges": …, "neut": …, "sweep_low": …, …}}``
        - **With r_bins** (``build_cms_tables(r_bins=[…])``):
          ``{"all": {stat: {…}}, "(lo, hi]": {stat: {…}}, …}``
    stats : list[str] or None
        Stat columns to score. Inferred from *tables* when ``None``.
    daf_thresholds : tuple[float, float]
        selDAF stratification thresholds (must match those used in
        :func:`build_cms_tables`).
    prior_p : float
        Prior P(selection). Default 0.5 (genome-wide scanning).
    fold_stats : frozenset[str] or None
        Stats to fold (|value|) before bin lookup. Default: ``{"ihs"}``.
    snps_r_bins : pl.Series or None
        Per-SNP recombination bin labels (same categorical format as produced
        by ``pl.Series.cut``).  Required when *tables* was built with *r_bins*.
        When ``None`` the flat (non-r_bin-stratified) tables are used.

    Returns
    -------
    pl.DataFrame
        Columns: ``positions``, ``daf``, ``log_bf``, ``cms``.
    """
    if fold_stats is None:
        fold_stats = _FOLD_STATS

    # Detect table format: r-binned vs flat
    _r_binned = "all" in tables and not any(k in SUPPORTED_STATS for k in tables)

    if stats is None:
        if _r_binned:
            stats = list(tables["all"].keys())
        else:
            stats = [k for k in tables if k in SUPPORTED_STATS]

    n = len(snps_df)
    log_bf = np.zeros(n, dtype=np.float64)

    daf_arr = snps_df["daf"].to_numpy().astype(np.float64)
    lo_thr, hi_thr = daf_thresholds
    sweep_key = np.where(
        daf_arr <= lo_thr,
        "sweep_low",
        np.where(daf_arr <= hi_thr, "sweep_mid", "sweep_high"),
    )

    def _accumulate_bf(
        sub_tables: dict,
        row_mask: np.ndarray,
    ) -> None:
        """Add log-BF contributions to log_bf[row_mask] using sub_tables."""
        for s in stats:
            if s not in sub_tables or s not in snps_df.columns:
                continue
            t = sub_tables[s]
            edges = t["bin_edges"]
            neut_hist = t["neut"]
            n_bins = len(neut_hist)

            vals = snps_df[s].to_numpy().astype(np.float64)
            if s in fold_stats:
                vals = np.abs(vals)

            bin_idx = np.clip(
                np.searchsorted(edges, vals, side="right") - 1,
                0,
                n_bins - 1,
            )
            valid = row_mask & np.isfinite(vals)
            log_neut = np.log(neut_hist)

            for group in ("sweep_low", "sweep_mid", "sweep_high"):
                mask = valid & (sweep_key == group)
                if mask.sum() == 0:
                    continue
                log_bf[mask] += (
                    np.log(t[group][bin_idx[mask]]) - log_neut[bin_idx[mask]]
                )

    if not _r_binned or snps_r_bins is None:
        # Flat tables — use directly for all SNPs
        sub = tables.get("all", tables) if _r_binned else tables
        _accumulate_bf(sub, np.ones(n, dtype=bool))
    else:
        # r-bin-stratified: iterate over unique r-bins in this region
        r_bins_arr = snps_r_bins.cast(pl.Utf8).to_numpy().astype(str)
        for r_bin in np.unique(r_bins_arr):
            row_mask = r_bins_arr == r_bin
            if not row_mask.any():
                continue
            # Use r_bin-specific tables; fall back to "all" if not available
            sub = tables.get(str(r_bin)) or tables.get("all", {})
            if sub:
                _accumulate_bf(sub, row_mask)

    # CMS posterior — overflow-safe
    ratio = (1.0 - prior_p) / prior_p
    cms = np.where(
        log_bf > 500,
        1.0,
        np.where(
            log_bf < -500,
            0.0,
            np.exp(log_bf) / (np.exp(log_bf) + ratio),
        ),
    )

    return snps_df.select(["positions", "daf"]).with_columns(
        [
            pl.Series("log_bf", log_bf),
            pl.Series("cms", cms),
        ]
    )


# ---------------------------------------------------------------------------
# Option A2: compute per-SNP stats from raw haplotype data
# ---------------------------------------------------------------------------


def _snps_from_hap(
    hap_int: np.ndarray,
    ac: np.ndarray,
    rec_map: np.ndarray,
    position_masked: np.ndarray,
    genetic_position_masked: np.ndarray,
    stats: list[str],
    locus_length: int = int(1.2e6),
    _iter: int = 1,
) -> pl.DataFrame | None:
    """
    Compute per-SNP stats needed for CMS from a parsed haplotype matrix.
    No window aggregation — only the stat columns listed in *stats* are computed.
    """
    freqs = np.ascontiguousarray(
        ac[:, 1] / np.maximum(ac.sum(axis=1), 1), dtype=np.float64
    )
    stat_set = set(stats)
    frames: list = []

    # iHS / delta_ihh ---------------------------------------------------
    if stat_set & {"ihs", "delta_ihh"}:
        try:
            df_ihs = ihs_ihh(
                hap_int,
                position_masked,
                map_pos=genetic_position_masked,
                min_ehh=0.05 if locus_length > 1e6 else 0.1,
                min_maf=0.05,
                include_edges=locus_length <= 1e6,
            )
            frames.append(center_window_cols(df_ihs, _iter=_iter).lazy())
        except Exception as exc:
            warnings.warn(f"ihs_ihh failed (iter={_iter}): {exc}")

    # nSL ---------------------------------------------------------------
    if "nsl" in stat_set and _allel_nsl is not None:
        try:
            maf_mask = freqs >= 0.05
            nsl_v = _allel_nsl(hap_int[maf_mask], use_threads=False)
            df_nsl = pl.DataFrame(
                {
                    "positions": position_masked[maf_mask],
                    "daf": freqs[maf_mask],
                    "nsl": nsl_v,
                }
            ).fill_nan(None)
            frames.append(center_window_cols(df_nsl, _iter=_iter).lazy())
        except Exception as exc:
            warnings.warn(f"nsl failed (iter={_iter}): {exc}")

    # iSAFE -------------------------------------------------------------
    if "isafe" in stat_set:
        try:
            df_isafe = run_isafe(hap_int, position_masked)
            frames.append(center_window_cols(df_isafe, _iter=_iter).lazy())
        except Exception as exc:
            warnings.warn(f"run_isafe failed (iter={_iter}): {exc}")

    # DIND / high_freq / low_freq / s_ratio / hapdaf_o / hapdaf_s ------
    fs_needed = stat_set & {
        "dind",
        "high_freq",
        "low_freq",
        "s_ratio",
        "hapdaf_o",
        "hapdaf_s",
    }
    if fs_needed:
        try:
            df_dind, df_sratio, df_hapdaf_o, df_hapdaf_s = run_fs_stats(
                hap_int, ac, rec_map
            )
            for df in (df_dind, df_sratio, df_hapdaf_o, df_hapdaf_s):
                df2 = center_window_cols(df, _iter=_iter)
                keep = [
                    c
                    for c in df2.columns
                    if c in {"iter", "positions", "daf"} or c in stat_set
                ]
                frames.append(df2.select(keep).lazy())
        except Exception as exc:
            warnings.warn(f"run_fs_stats failed (iter={_iter}): {exc}")

    if not frames:
        return None

    return (
        reduce(
            lambda l, r: l.join(
                r, on=["iter", "positions", "daf"], how="full", coalesce=True
            ),
            frames,
        )
        .sort("positions")
        .collect()
    )


def _read_discoal_params(ms_file: str) -> np.ndarray:
    """
    Extract simulation parameters from the discoal ms header line.
    Returns array ``[s, t, f_i, f_t, mu, r]`` with ``f_t = 0`` for neutral sims.
    """
    import gzip as _gzip

    open_fn = _gzip.open if ms_file.endswith(".gz") else open
    params = np.zeros(6, dtype=np.float64)
    try:
        with open_fn(ms_file, "rt") as fh:
            header = fh.readline().strip()
        parts = header.split()
        _flag_col = {
            "-a": 0,  # selection coefficient s
            "-x": 1,  # time of sweep t
            "-ws": 2,  # initial sweep frequency f_i
            "-we": 3,  # final sweep frequency f_t
        }
        for j, token in enumerate(parts):
            if token in _flag_col and j + 1 < len(parts):
                try:
                    params[_flag_col[token]] = float(parts[j + 1])
                except ValueError:
                    pass
    except Exception:
        pass
    return params


def _compute_snps_from_sims(
    sims_dir: str,
    stats: list[str],
    locus_length: int = int(1.2e6),
    nthreads: int = 1,
) -> dict:
    """
    Compute per-SNP stats from raw ``.ms.gz`` simulation files in *sims_dir*.

    Returns a dict in the same format as a save_stats sims pickle::

        {sim_type: summaries(stats=[{"snps": df, "windows": None}, ...],
                             parameters=params_arr)}

    No windowing, no CNN feature vectors — only per-SNP stat columns needed for CMS.
    """
    results: dict = {}

    for sim_type in ("neutral", "sweep"):
        sub_dir = os.path.join(sims_dir, sim_type)
        if not os.path.isdir(sub_dir):
            continue

        ms_files = sorted(
            glob.glob(os.path.join(sub_dir, "*.ms.gz"))
            + glob.glob(os.path.join(sub_dir, "*.ms"))
        )
        if not ms_files:
            logger.warning("No .ms/.ms.gz files in %s — skipping.", sub_dir)
            continue

        def _process_one(ms_file: str, idx: int):
            try:
                (hap_int, rec_map, ac, _, pos_masked, gpos_masked) = parse_ms_numpy(
                    ms_file, seq_len=locus_length
                )
                if hap_int.shape[0] == 0:
                    return None, None
                snps = _snps_from_hap(
                    hap_int,
                    ac,
                    rec_map,
                    pos_masked,
                    gpos_masked,
                    stats,
                    locus_length=locus_length,
                    _iter=idx + 1,
                )
                p = _read_discoal_params(ms_file)
                return snps, p
            except Exception as exc:
                warnings.warn(f"Failed to process {ms_file}: {exc}")
                return None, None

        job_out = Parallel(n_jobs=nthreads, prefer="threads")(
            delayed(_process_one)(f, i) for i, f in enumerate(ms_files)
        )

        rep_list: list = []
        param_rows: list = []
        for snps, p in job_out:
            if snps is not None:
                rep_list.append({"snps": snps, "windows": None})
                param_rows.append(p if p is not None else np.zeros(6))

        if not rep_list:
            continue

        params_arr = np.vstack(param_rows)
        results[sim_type] = summaries(rep_list, params_arr)

    return results


def _compute_snps_from_vcf(
    vcf_dir: str,
    stats: list[str],
    locus_length: int = int(1.2e6),
    nthreads: int = 1,
) -> dict:
    """
    Compute per-SNP stats from ``*.vcf.gz`` files in *vcf_dir*.

    Returns a dict in the same format as a save_stats VCF pickle::

        {"chrN:start-end": summaries(stats={"snps": df, "window": None},
                                     parameters=None)}

    No window aggregation — only per-SNP stat columns needed for CMS.
    """
    vcf_files = sorted(
        glob.glob(os.path.join(vcf_dir, "*.vcf.gz"))
        + glob.glob(os.path.join(vcf_dir, "*.bcf.gz"))
    )
    if not vcf_files:
        raise FileNotFoundError(f"No *.vcf.gz or *.bcf.gz files found in {vcf_dir}")

    results: dict = {}
    for vcf_file in vcf_files:
        regions_data = genome_reader(vcf_file)
        for region, payload in regions_data.items():
            if payload is None:
                continue
            hap_int, rec_map, ac, _, pos_masked, gpos_masked = payload
            if hap_int.shape[0] == 0:
                continue
            snps = _snps_from_hap(
                hap_int,
                ac,
                rec_map,
                pos_masked,
                gpos_masked,
                stats,
                locus_length=locus_length,
                _iter=1,
            )
            if snps is not None:
                results[str(region)] = summaries({"snps": snps, "window": None}, None)
    return results


# ---------------------------------------------------------------------------
# Input resolution: Option A (pickle) vs Option A2 (from scratch)
# ---------------------------------------------------------------------------


def _resolve_sims(
    sims: str | None,
    sims_dir: str | None,
    stats: list[str],
    locus_length: int,
    nthreads: int,
) -> dict:
    """Load or compute sims data. Returns a loaded dict (never a path)."""
    if sims and os.path.isfile(sims):
        return load_pickle(sims)

    if sims_dir and os.path.isdir(sims_dir):
        cache = os.path.join(sims_dir, "raw_statistics_cms.pickle")
        if os.path.isfile(cache):
            logger.info("Loading cached CMS sims pickle: %s", cache)
            return load_pickle(cache)
        logger.info("Computing per-SNP stats from %s (Option A2) …", sims_dir)
        data = _compute_snps_from_sims(sims_dir, stats, locus_length, nthreads)
        save_pickle(cache, data)
        logger.info("Wrote %s — pass --sims %s for faster reruns.", cache, cache)
        return data

    raise ValueError(
        "Provide either --sims <pickle> or --sims-dir <raw sim directory>."
    )


def _resolve_vcf(
    vcf: str | None,
    vcf_dir: str | None,
    stats: list[str],
    locus_length: int,
    nthreads: int,
) -> dict:
    """Load or compute VCF data. Returns a loaded dict (never a path)."""
    if vcf and os.path.isfile(vcf):
        return load_pickle(vcf)

    if vcf_dir and os.path.isdir(vcf_dir):
        cache = os.path.join(vcf_dir, "raw_statistics_cms.pickle")
        if os.path.isfile(cache):
            logger.info("Loading cached CMS VCF pickle: %s", cache)
            return load_pickle(cache)
        logger.info(
            "Computing per-SNP stats from VCF files in %s (Option A2) …",
            vcf_dir,
        )
        data = _compute_snps_from_vcf(vcf_dir, stats, locus_length, nthreads)
        save_pickle(cache, data)
        logger.info("Wrote %s — pass --vcf %s for faster reruns.", cache, cache)
        return data

    raise ValueError("Provide either --vcf <pickle> or --vcf-dir <raw VCF directory>.")


# ---------------------------------------------------------------------------
# Public API: Step 3 — run_cms
# ---------------------------------------------------------------------------


def run_cms(
    sims_pickle: str | None = None,
    vcf_pickle: str | None = None,
    sims_dir: str | None = None,
    vcf_dir: str | None = None,
    stats: list[str] | None = None,
    n_bins: int = 60,
    daf_thresholds: tuple[float, float] = _DAF_THRESHOLDS,
    prior_p: float = 0.5,
    fold_stats: frozenset[str] | None = None,
    cms_tables_path: str | None = None,
    locus_length: int = int(1.2e6),
    recombination_map: str | None = None,
    r_bins: list | None = None,
    out_prefix: str | None = None,
    nthreads: int = 1,
) -> pl.DataFrame:
    """
    Compute Composite of Multiple Signals (CMS) posterior per SNP.

    Accepts pre-computed save_stats pickles (Option A — fast path) or raw
    directories (Option A2 — computes stats from scratch on the fly).
    At least one of *sims_pickle* / *sims_dir* and one of *vcf_pickle* /
    *vcf_dir* must be supplied.

    Parameters
    ----------
    sims_pickle : str or None
        Path to ``raw_statistics.pickle`` from ``fvs-discoal --save_stats``.
    vcf_pickle : str or None
        Path to ``raw_statistics.pickle`` from ``fvs-vcf --save_stats``.
    sims_dir : str or None
        Raw simulation directory (Option A2 fallback).
    vcf_dir : str or None
        Raw VCF directory (Option A2 fallback).
    stats : list[str] or None
        Stat column names. Default: ``["ihs", "delta_ihh", "dind", "hapdaf_o"]``.
    n_bins : int
        Histogram bins per stat. Default 60.
    daf_thresholds : tuple[float, float]
        selDAF stratification thresholds. Default ``(0.35, 0.65)``.
    prior_p : float
        Prior P(selection). Default 0.5 (Grossman 2010 genome-wide mode).
    fold_stats : frozenset[str] or None
        Stats to fold (|value|) before histogram lookup. Default ``{"ihs"}``.
    cms_tables_path : str or None
        Path to save / load prebuilt likelihood tables.  If the file exists it
        is loaded (skipping ``build_cms_tables``); otherwise tables are built and
        saved there for reuse.
    locus_length : int
        Simulation locus length in bp for interior-position filtering.
        Default 1,200,000.
    recombination_map : str or None
        Path to a recombination map TSV file (optionally gzipped).  Required
        columns: ``chr``, ``end`` (physical position), ``cm`` (cumulative cM).
        When provided together with *r_bins*, each SNP is assigned to a
        recombination-rate bin and scored against the corresponding stratified
        likelihood tables — identical to the ``_process_sims`` / ``_process_vcf``
        treatment in ``fv.py``.  Ignored if *r_bins* is ``None``.
    r_bins : list of float or None
        Recombination-rate bin break points in cM/Mb.  When provided, the
        simulation likelihood tables are stratified by recombination rate
        (using ``params[:, -1] * 1e8`` per replicate) and VCF SNPs are
        assigned to bins from the recombination map.  Typical values from
        ``fv.py`` default usage: ``[0.5, 1, 2, 3, 5]``.
    out_prefix : str or None
        Output file prefix.  Writes ``{out_prefix}.cms.txt`` (TSV) and
        ``{out_prefix}.cms.parquet``.
    nthreads : int
        Threads for Option A2 stat computation. Default 1.

    Returns
    -------
    pl.DataFrame
        Columns: ``region``, ``positions``, ``daf``, ``log_bf``, ``cms``,
        ``cms_rank``.

        - ``log_bf``   — log composite Bayes factor (Σ log BF_k)
        - ``cms``      — posterior P(selected) in [0, 1]
        - ``cms_rank`` — empirical genome-wide percentile rank in [0, 1]
    """
    if stats is None:
        stats = list(DEFAULT_STATS)
    if fold_stats is None:
        fold_stats = _FOLD_STATS

    # Validate requested stats against the supported set
    unknown = [s for s in stats if s not in SUPPORTED_STATS]
    if unknown:
        raise ValueError(
            f"Unknown stat(s) for CMS: {unknown}.\nSupported stats: {SUPPORTED_STATS}"
        )

    # Validate recombination map / r_bins consistency
    if r_bins is not None and recombination_map is None:
        raise ValueError(
            "--r_bins requires --recombination_map.  Provide a recombination "
            "map file so that per-SNP recombination rates can be computed."
        )

    # Load recombination map once (shared across all VCF regions)
    df_rec_map: pl.DataFrame | None = None
    if recombination_map is not None and r_bins is not None:
        df_rec_map = _load_recombination_map(recombination_map)
        logger.info(
            "Loaded recombination map: %s (%d rows)", recombination_map, len(df_rec_map)
        )

    # ----- likelihood tables -----
    if cms_tables_path and os.path.isfile(cms_tables_path):
        tables = load_pickle(cms_tables_path)
        logger.info("Loaded prebuilt CMS tables from %s", cms_tables_path)
    else:
        sims_data = _resolve_sims(sims_pickle, sims_dir, stats, locus_length, nthreads)
        tables = build_cms_tables(
            sims_data,
            stats=stats,
            n_bins=n_bins,
            daf_thresholds=daf_thresholds,
            locus_length=locus_length,
            fold_stats=fold_stats,
            r_bins=r_bins,
        )
        if cms_tables_path:
            save_pickle(cms_tables_path, tables)
            logger.info("Saved CMS tables to %s", cms_tables_path)

    # ----- score VCF regions -----
    vcf_data = _resolve_vcf(vcf_pickle, vcf_dir, stats, locus_length, nthreads)

    all_results: list[pl.DataFrame] = []
    for key, summary in vcf_data.items():
        stats_obj = summary.stats
        # VCF pickle: stats is a dict {"snps": df, "window": df}
        # Option A2:  same structure {"snps": df, "window": None}
        if isinstance(stats_obj, dict):
            snps_df = stats_obj.get("snps")
        elif isinstance(stats_obj, list) and stats_obj:
            snps_df = (
                stats_obj[0].get("snps") if isinstance(stats_obj[0], dict) else None
            )
        else:
            snps_df = None

        if snps_df is None or len(snps_df) == 0:
            continue

        # Compute per-SNP recombination bins for this VCF region
        snps_r_bins: pl.Series | None = None
        if df_rec_map is not None and r_bins is not None:
            snps_r_bins = _compute_vcf_snp_r_bins(snps_df, key, df_rec_map, r_bins)

        scored = cms_score(
            snps_df,
            tables,
            stats,
            daf_thresholds,
            prior_p,
            fold_stats,
            snps_r_bins=snps_r_bins,
        )
        all_results.append(scored.with_columns(pl.lit(str(key)).alias("region")))

    if not all_results:
        logger.warning(
            "No VCF regions produced CMS scores — returning empty DataFrame."
        )
        return pl.DataFrame(
            schema={
                "region": pl.Utf8,
                "positions": pl.Int64,
                "daf": pl.Float64,
                "log_bf": pl.Float64,
                "cms": pl.Float64,
                "cms_rank": pl.Float64,
            }
        )

    df_out = pl.concat(all_results).sort(["region", "positions"])

    # Genome-wide empirical percentile rank of log_bf
    log_bf_arr = df_out["log_bf"].to_numpy().astype(np.float64)
    ranks = rank_with_duplicates(log_bf_arr)
    df_out = df_out.with_columns(
        pl.Series("cms_rank", (ranks.astype(np.float64) - 0.5) / len(ranks))
    )

    df_out = df_out.select(["region", "positions", "daf", "log_bf", "cms", "cms_rank"])

    if out_prefix:
        df_out.write_csv(f"{out_prefix}.cms.txt", separator="\t")
        df_out.write_parquet(f"{out_prefix}.cms.parquet")
        logger.info("Wrote %s.cms.txt and %s.cms.parquet", out_prefix, out_prefix)

    return df_out


# ---------------------------------------------------------------------------
# Public API: CMSlocal — fine-map causal variant within candidate regions
# ---------------------------------------------------------------------------


def run_cms_local(
    candidate_regions: list[str] | None = None,
    vcf_pickle: str | None = None,
    vcf_dir: str | None = None,
    cms_tables_path: str | None = None,
    sims_pickle: str | None = None,
    sims_dir: str | None = None,
    stats: list[str] | None = None,
    n_bins: int = 60,
    daf_thresholds: tuple[float, float] = _DAF_THRESHOLDS,
    fold_stats: frozenset[str] | None = None,
    locus_length: int = int(1.2e6),
    recombination_map: str | None = None,
    r_bins: list | None = None,
    out_prefix: str | None = None,
    nthreads: int = 1,
) -> pl.DataFrame:
    """
    CMSlocal: fine-map the causal variant within each candidate sweep region.

    Implements the within-region fine-mapping mode from Grossman et al. (2010).
    For each candidate region the prior is set to ``π = 1 / N_SNP`` (uniform over
    the SNPs in that region), so the ``cms`` column approximates the posterior
    probability that *this specific SNP* is the selected variant — and the values
    across all SNPs in the region sum to ≈ 1.

    Uses the same simulation-derived likelihood tables as :func:`run_cms`.
    The tables can be prebuilt (pass *cms_tables_path*) or built on the fly
    (pass *sims_pickle* or *sims_dir*).

    Parameters
    ----------
    candidate_regions : list[str] or None
        Region keys to fine-map, e.g. ``["chr22:17000000-18200000", ...]``.
        These must match keys in the VCF pickle / VCF directory.
        Typical sources:

        * ``df.filter(pl.col("cms_rank") > 0.999)["region"].unique().to_list()``
          from a prior :func:`run_cms` (CMS_GW) result.
        * Top CNN prediction windows from ``cnn.predict()``.

        When ``None``, all regions in the VCF data are fine-mapped.
    vcf_pickle : str or None
        Path to ``raw_statistics.pickle`` from ``fvs-vcf --save_stats``.
    vcf_dir : str or None
        Raw VCF directory (Option A2 fallback — computes stats on the fly).
    cms_tables_path : str or None
        Path to prebuilt CMS likelihood tables (``cms_tables.pickle``).
        If the file exists it is loaded directly; otherwise tables are built
        from *sims_pickle* / *sims_dir* and saved here for reuse.
    sims_pickle : str or None
        Path to ``raw_statistics.pickle`` from ``fvs-discoal --save_stats``.
        Required only when *cms_tables_path* does not exist yet.
    sims_dir : str or None
        Raw simulation directory (Option A2 fallback).
    stats : list[str] or None
        Stat columns to score. Default: ``["ihs", "delta_ihh", "dind", "hapdaf_o"]``.
    n_bins : int
        Histogram bins per stat. Default 60.
    daf_thresholds : tuple[float, float]
        selDAF stratification thresholds. Default ``(0.35, 0.65)``.
    fold_stats : frozenset[str] or None
        Stats to fold (|value|) before histogram lookup. Default ``{"ihs"}``.
    locus_length : int
        Simulation locus length in bp for interior-position filtering. Default 1,200,000.
    recombination_map : str or None
        Recombination map TSV. Required when *r_bins* is provided.
    r_bins : list of float or None
        Recombination-rate bin breaks in cM/Mb.
    out_prefix : str or None
        Output file prefix. Writes ``{out_prefix}.cms_local.txt`` and
        ``{out_prefix}.cms_local.parquet``.
    nthreads : int
        Threads for Option A2 stat computation. Default 1.

    Returns
    -------
    pl.DataFrame
        Columns: ``region``, ``positions``, ``daf``, ``log_bf``, ``cms``,
        ``n_snps_region``, ``prior_p``.

        - ``log_bf``        — log composite Bayes factor (Σ log BF_k)
        - ``cms``           — P(this SNP is selected | data, region) in [0, 1]
        - ``n_snps_region`` — number of SNPs in the region (denominator of prior)
        - ``prior_p``       — per-region prior used (= 1 / n_snps_region)

        ``cms`` values within each region sum to ≈ 1 (fine-mapping posterior).
        To find the most likely causal variant: ``df.sort("cms", descending=True)``.
    """
    if stats is None:
        stats = list(DEFAULT_STATS)
    if fold_stats is None:
        fold_stats = _FOLD_STATS

    unknown = [s for s in stats if s not in SUPPORTED_STATS]
    if unknown:
        raise ValueError(
            f"Unknown stat(s) for CMS: {unknown}.\nSupported stats: {SUPPORTED_STATS}"
        )

    if r_bins is not None and recombination_map is None:
        raise ValueError(
            "--r_bins requires --recombination_map."
        )

    # Load recombination map once
    df_rec_map: pl.DataFrame | None = None
    if recombination_map is not None and r_bins is not None:
        df_rec_map = _load_recombination_map(recombination_map)

    # Load or build likelihood tables
    if cms_tables_path and os.path.isfile(cms_tables_path):
        tables = load_pickle(cms_tables_path)
        logger.info("Loaded prebuilt CMS tables from %s", cms_tables_path)
    else:
        if sims_pickle is None and sims_dir is None:
            raise ValueError(
                "Provide --tables (prebuilt) or --sims / --sims-dir to build tables."
            )
        sims_data = _resolve_sims(sims_pickle, sims_dir, stats, locus_length, nthreads)
        tables = build_cms_tables(
            sims_data,
            stats=stats,
            n_bins=n_bins,
            daf_thresholds=daf_thresholds,
            locus_length=locus_length,
            fold_stats=fold_stats,
            r_bins=r_bins,
        )
        if cms_tables_path:
            save_pickle(cms_tables_path, tables)
            logger.info("Saved CMS tables to %s", cms_tables_path)

    # Load VCF data
    vcf_data = _resolve_vcf(vcf_pickle, vcf_dir, stats, locus_length, nthreads)

    # Filter to requested candidate regions
    all_keys = list(vcf_data.keys())
    if candidate_regions is not None:
        missing = [r for r in candidate_regions if r not in vcf_data]
        if missing:
            logger.warning(
                "%d candidate region(s) not found in VCF data: %s",
                len(missing), missing[:5],
            )
        keys_to_score = [r for r in candidate_regions if r in vcf_data]
    else:
        keys_to_score = all_keys

    if not keys_to_score:
        logger.warning("No matching regions found — returning empty DataFrame.")
        return pl.DataFrame(
            schema={
                "region": pl.Utf8,
                "positions": pl.Int64,
                "daf": pl.Float64,
                "log_bf": pl.Float64,
                "cms": pl.Float64,
                "n_snps_region": pl.Int64,
                "prior_p": pl.Float64,
            }
        )

    all_results: list[pl.DataFrame] = []
    for key in keys_to_score:
        summary = vcf_data[key]
        stats_obj = summary.stats
        if isinstance(stats_obj, dict):
            snps_df = stats_obj.get("snps")
        elif isinstance(stats_obj, list) and stats_obj:
            snps_df = (
                stats_obj[0].get("snps") if isinstance(stats_obj[0], dict) else None
            )
        else:
            snps_df = None

        if snps_df is None or len(snps_df) == 0:
            logger.debug("Region %s: no SNPs, skipping.", key)
            continue

        n_snps = len(snps_df)
        region_prior = 1.0 / n_snps

        snps_r_bins: pl.Series | None = None
        if df_rec_map is not None and r_bins is not None:
            snps_r_bins = _compute_vcf_snp_r_bins(snps_df, key, df_rec_map, r_bins)

        scored = cms_score(
            snps_df,
            tables,
            stats,
            daf_thresholds,
            region_prior,
            fold_stats,
            snps_r_bins=snps_r_bins,
        )
        scored = scored.with_columns(
            [
                pl.lit(str(key)).alias("region"),
                pl.lit(n_snps).cast(pl.Int64).alias("n_snps_region"),
                pl.lit(region_prior).alias("prior_p"),
            ]
        )
        all_results.append(scored)

    if not all_results:
        logger.warning("No regions produced CMSlocal scores — returning empty DataFrame.")
        return pl.DataFrame(
            schema={
                "region": pl.Utf8,
                "positions": pl.Int64,
                "daf": pl.Float64,
                "log_bf": pl.Float64,
                "cms": pl.Float64,
                "n_snps_region": pl.Int64,
                "prior_p": pl.Float64,
            }
        )

    df_out = (
        pl.concat(all_results)
        .select(["region", "positions", "daf", "log_bf", "cms",
                 "n_snps_region", "prior_p"])
        .sort(["region", "cms"], descending=[False, True])
    )

    if out_prefix:
        df_out.write_csv(f"{out_prefix}.cms_local.txt", separator="\t")
        df_out.write_parquet(f"{out_prefix}.cms_local.parquet")
        logger.info(
            "Wrote %s.cms_local.txt and %s.cms_local.parquet", out_prefix, out_prefix
        )

    return df_out
