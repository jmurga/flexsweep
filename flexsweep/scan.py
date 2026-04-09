"""
flexsweep/scan.py — Standalone outlier scan.

Completely separate from the CNN pipeline (fvs-vcf / fvs*.parquet).
No neutral simulations, no fixed window grid. Each stat runs at its natural resolution.
Outlier ranking uses the genome-wide empirical distribution.

API usage::

    from flexsweep.scan import scan

    results = scan(
        "vcf_folder/",          # directory of *.vcf.gz, one per chromosome
        "YRI",
        stats=["ihs", "nsl", "h12", "lassip"],
        nthreads=8,
        config={"lassip": {"max_extend": 1e5}, "raisd": {"window_size": 100}},
    )
    # results["ihs"]    → Polars DataFrame at SNP resolution, ihs_pvalue column included
    # results["lassip"] → Polars DataFrame at window resolution, lassip_pvalue column included

CLI usage::

    flexsweep scan --vcf_path vcf_folder/ --out YRI \\
        --stats ihs,nsl,h12,lassip --w_size 201 --min_maf 0.05 --nthreads 8

Available stats
---------------
Per-SNP (output at SNP resolution):
    ihs, nsl, isafe, dind, high_freq, low_freq, s_ratio, hapdaf_o, hapdaf_s, haf

Sliding SNP-window (output at window resolution):
    h12, garud, lassi, lassip, raisd

Sliding bp-window (output at window resolution):
    tajima_d, pi, theta_w, fay_wu_h, zeng_e, achaz_y,
    fuli_f, fuli_f_star, fuli_d, fuli_d_star, neutrality, omega, zns, beta, ncd

Notes
-----
- iHS and nSL are z-scored within genome-wide DAF bins (100 bins by default).
  Add ``--recombination_map`` to normalize by (DAF × recomb_bin) jointly.
- dind, s_ratio, hapdaf_o, hapdaf_s use REF/ALT as ancestral/derived (same as fvs-vcf).
  Pass ``recombination_map`` to use genetic distances for window boundaries.
- lassip and lassi derive their neutral spectrum from the VCF itself (average
  haplotype frequency spectrum); no simulations are needed.
- delta_ihh is intentionally excluded.
- window_mode="auto" uses per-stat defaults (h12/garud/lassi/lassip/raisd=snp;
  tajima_d/pi/theta_w/fay_wru_h/zeng_e/achaz_y/fuli_*/neutrality/omega/zns/beta=bp).
  Pass window_mode="snp" or window_mode="bp" to override for all window stats.
  Default "auto" is correct when mixing stats with different natural modes.
"""

import glob
import os
import warnings
from collections import namedtuple
from math import ceil

from allel import nsl

from . import Parallel, delayed, np, pl
from .fv_v2 import (
    Lambda_statistic_fast,
    LASSI_spectrum_and_Kspectrum,
    Ld,
    T_m_statistic_fast,
    achaz_y,
    compute_r2_matrix_upper,
    dind_high_low_from_pairs,
    fast_sq_freq_pairs,
    fay_wu_h_norm,
    fs_stats_dataframe,
    fuli_d,
    fuli_d_star,
    fuli_f,
    fuli_f_star,
    garud_h,
    genome_reader,
    haf_top,
    hapdaf_from_pairs,
    hscan,
    ihs_ihh,
    mu_stat,
    ncd1,
    neut_average,
    neutrality_stats,
    omega_linear_correct,
    run_beta_window,
    run_isafe,
    s_ratio_from_pairs,
    tajima_d,
    theta_pi,
    theta_watterson,
    zeng_e,
)

# Stat definition

# resolution: "snp" | "window"
# tier: 1=hap+pos only, 2=+allele counts, 3=+rec_map/polarization
# rank_col: primary column used for ranking (higher = more sweepy)
# default_params: defaults merged with shared params at call time

StatDef = namedtuple("StatDef", ["resolution", "tier", "rank_col", "default_params"])

STAT_REGISTRY: dict[str, StatDef] = {
    # -- Per-SNP stats --
    "ihs": StatDef(
        "snp",
        1,
        "ihs",
        {
            "min_maf": 0.05,
            "include_edges": False,
            "gap_scale": 20000,
            "max_gap": 200000,
        },
    ),
    "nsl": StatDef("snp", 1, "nsl", {"min_maf": 0.05}),
    "isafe": StatDef(
        "snp",
        1,
        "isafe",
        {
            "region_size_bp": 1_000_000,
            "isafe_window": 300,
            "isafe_step": 150,
            "top_k": 1,
            "max_rank": 15,
        },
    ),
    "dind": StatDef(
        "snp",
        3,
        "dind",
        {"window_size": 50000, "min_focal_freq": 0.25, "max_focal_freq": 0.95},
    ),
    "high_freq": StatDef(
        "snp",
        3,
        "high_freq",
        {"window_size": 50000, "min_focal_freq": 0.25, "max_focal_freq": 0.95},
    ),
    "low_freq": StatDef(
        "snp",
        3,
        "low_freq",
        {"window_size": 50000, "min_focal_freq": 0.25, "max_focal_freq": 0.95},
    ),
    "s_ratio": StatDef(
        "snp",
        3,
        "s_ratio",
        {"window_size": 50000, "min_focal_freq": 0.25, "max_focal_freq": 0.95},
    ),
    "hapdaf_o": StatDef(
        "snp",
        3,
        "hapdaf_o",
        {
            "window_size": 50000,
            "min_focal_freq": 0.25,
            "max_focal_freq": 0.95,
            "max_ancest_freq": 0.25,
            "min_tot_freq": 0.25,
        },
    ),
    "hapdaf_s": StatDef(
        "snp",
        3,
        "hapdaf_s",
        {
            "window_size": 50000,
            "min_focal_freq": 0.25,
            "max_focal_freq": 0.95,
            "max_ancest_freq": 0.10,
            "min_tot_freq": 0.10,
        },
    ),
    "haf": StatDef("window", 1, "haf", {"window_mode": "snp", "w_size": 201, "step": 10}),
    "hscan": StatDef(
        "snp", 1, "hscan", {"max_gap": 200_000, "dist_mode": 0, "hscan_step": 1}
    ),
    # -- Sliding-window stats (SNP-count mode) --
    "h12": StatDef(
        "window", 1, "h12", {"window_mode": "snp", "w_size": 200, "step": 10}
    ),
    "garud": StatDef(
        "window", 1, "h12", {"window_mode": "snp", "w_size": 200, "step": 10}
    ),
    "lassi": StatDef(
        "window",
        1,
        "T_m",
        {
            "window_mode": "snp",
            "w_size": 201,
            "step": 10,
            "K_truncation": 10,
            "sweep_mode": 4,
        },
    ),
    "lassip": StatDef(
        "window",
        1,
        "Lambda",
        {
            "window_mode": "snp",
            "w_size": 201,
            "step": 10,
            "K_truncation": 10,
            "sweep_mode": 4,
            "max_extend": 1e5,
            "n_A": 100,
        },
    ),
    "raisd": StatDef(
        "window", 1, "mu_total", {"window_mode": "snp", "window_size": 50}
    ),
    # -- Sliding-window stats (physical bp mode) --
    "tajima_d": StatDef(
        "window",
        2,
        "tajima_d",
        {"window_mode": "bp", "w_size_bp": 1_000_000, "step_bp": 10_000},
    ),
    "pi": StatDef(
        "window",
        2,
        "pi",
        {"window_mode": "bp", "w_size_bp": 1_000_000, "step_bp": 10_000},
    ),
    "theta_w": StatDef(
        "window",
        2,
        "theta_w",
        {"window_mode": "bp", "w_size_bp": 1_000_000, "step_bp": 10_000},
    ),
    "fay_wu_h": StatDef(
        "window",
        2,
        "fay_wu_h",
        {"window_mode": "bp", "w_size_bp": 1_000_000, "step_bp": 10_000},
    ),
    "zeng_e": StatDef(
        "window",
        2,
        "zeng_e",
        {"window_mode": "bp", "w_size_bp": 1_000_000, "step_bp": 10_000},
    ),
    "achaz_y": StatDef(
        "window",
        2,
        "achaz_y",
        {"window_mode": "bp", "w_size_bp": 1_000_000, "step_bp": 10_000},
    ),
    "fuli_f": StatDef(
        "window",
        2,
        "fuli_f",
        {"window_mode": "bp", "w_size_bp": 1_000_000, "step_bp": 10_000},
    ),
    "fuli_f_star": StatDef(
        "window",
        2,
        "fuli_f_star",
        {"window_mode": "bp", "w_size_bp": 1_000_000, "step_bp": 10_000},
    ),
    "fuli_d": StatDef(
        "window",
        2,
        "fuli_d",
        {"window_mode": "bp", "w_size_bp": 1_000_000, "step_bp": 10_000},
    ),
    "fuli_d_star": StatDef(
        "window",
        2,
        "fuli_d_star",
        {"window_mode": "bp", "w_size_bp": 1_000_000, "step_bp": 10_000},
    ),
    "neutrality": StatDef(
        "window",
        2,
        "tajima_d",
        {"window_mode": "bp", "w_size_bp": 1_000_000, "step_bp": 10_000},
    ),
    "omega": StatDef(
        "window",
        1,
        "omega_max",
        {"window_mode": "bp", "w_size_bp": 100_000, "step_bp": 10_000},
    ),
    "zns": StatDef(
        "window",
        1,
        "zns",
        {"window_mode": "bp", "w_size_bp": 100_000, "step_bp": 10_000},
    ),
    "beta": StatDef(
        "window",
        2,
        "beta1",
        {"window_mode": "bp", "w_size_bp": 50_000, "step_bp": 5_000, "m": 0.1},
    ),
    "ncd": StatDef("window", 3, "ncd1", {"tf": 0.5, "w": 3000, "minIS": 2}),
}


def available_stats() -> list[str]:
    """Return list of all available stat keys."""
    return list(STAT_REGISTRY.keys())


def stat_params(stat_key: str | None = None) -> dict:
    """Return parameter documentation for scan stats.

    Parameters
    ----------
    stat_key : str, optional
        A specific stat key (e.g. ``"ihs"``, ``"hscan"``). If None, returns
        the full table for all stats.

    Returns
    -------
    dict
        Keys are stat names. Each value is a dict with keys ``rank_col``,
        ``resolution``, ``window_mode``, ``default_window``, ``default_step``,
        ``shared_params``, and ``stat_params``.

    Notes
    -----
    Shared params always injected by ``scan()`` for every stat:
    ``w_size`` (201), ``step`` (10), ``w_size_bp`` (1000000),
    ``step_bp`` (10000), ``min_maf`` (0.05), ``window_mode`` ("auto").

    Per-stat params are passed via ``config={"stat": {"param": value}}``.

    Examples
    --------
    >>> from flexsweep.scan import stat_params
    >>> stat_params("hscan")
    >>> stat_params()   # full table
    """
    _SHARED = {
        "w_size": 201,
        "step": 10,
        "w_size_bp": 1_000_000,
        "step_bp": 10_000,
        "min_maf": 0.05,
        "window_mode": "auto",
    }

    # Stat-specific params: registry defaults merged with runner-read kwargs.
    # These are the keys a user can pass via config={"stat": {key: val}}.
    _STAT_SPECIFIC: dict[str, dict] = {
        "ihs":       {"include_edges": False, "gap_scale": 20000, "max_gap": 200000},
        "nsl":       {},
        "isafe":     {"region_size_bp": 1_000_000, "isafe_window": 300, "isafe_step": 150,
                      "top_k": 1, "max_rank": 15},
        "dind":      {"window_size": 50000, "min_focal_freq": 0.25, "max_focal_freq": 0.95},
        "high_freq": {"window_size": 50000, "min_focal_freq": 0.25, "max_focal_freq": 0.95},
        "low_freq":  {"window_size": 50000, "min_focal_freq": 0.25, "max_focal_freq": 0.95},
        "s_ratio":   {"window_size": 50000, "min_focal_freq": 0.25, "max_focal_freq": 0.95},
        "hapdaf_o":  {"window_size": 50000, "min_focal_freq": 0.25, "max_focal_freq": 0.95,
                      "max_ancest_freq": 0.25, "min_tot_freq": 0.25},
        "hapdaf_s":  {"window_size": 50000, "min_focal_freq": 0.25, "max_focal_freq": 0.95,
                      "max_ancest_freq": 0.10, "min_tot_freq": 0.10},
        "haf":       {},
        "hscan":     {"max_gap": 200_000, "dist_mode": 0,
                      "hscan_step": 1},  # note: NOT "step" — avoids shared override
        "h12":       {},
        "garud":     {},
        "neutrality": {},
        "omega":     {},
        "zns":       {},
        "tajima_d":  {},
        "pi":        {},
        "theta_w":   {},
        "fay_wu_h":  {},
        "zeng_e":    {},
        "achaz_y":   {},
        "fuli_f":    {},
        "fuli_f_star": {},
        "fuli_d":    {},
        "fuli_d_star": {},
        "lassi":     {"K_truncation": 10, "sweep_mode": 4},
        "lassip":    {"K_truncation": 10, "sweep_mode": 4, "max_extend": 1e5, "n_A": 100},
        "raisd":     {"window_size": 50},
        "beta":      {"m": 0.1},
        "ncd":       {"tf": 0.5, "w": 3000, "minIS": 2},
    }

    def _entry(key):
        defn = STAT_REGISTRY[key]
        dp = defn.default_params
        # window_mode: per-stat default from registry, or "snp"/"bp" if set there
        wm = dp.get("window_mode", "n/a (per-SNP stat)")
        # effective window size and step shown for window stats
        if wm == "snp":
            win_default = f"{dp.get('w_size', _SHARED['w_size'])} SNPs"
            step_default = f"{dp.get('step', _SHARED['step'])} SNPs"
        elif wm == "bp":
            win_default = f"{dp.get('w_size_bp', _SHARED['w_size_bp']):,} bp"
            step_default = f"{dp.get('step_bp', _SHARED['step_bp']):,} bp"
        else:
            win_default = "n/a"
            step_default = "n/a"
        return {
            "rank_col": defn.rank_col,
            "resolution": defn.resolution,
            "window_mode": wm,
            "default_window": win_default,
            "default_step": step_default,
            "shared_params": _SHARED,
            "stat_params": _STAT_SPECIFIC.get(key, {}),
        }

    if stat_key is not None:
        if stat_key not in STAT_REGISTRY:
            raise ValueError(f"Unknown stat: {stat_key!r}. Available: {available_stats()}")
        return {stat_key: _entry(stat_key)}

    return {k: _entry(k) for k in STAT_REGISTRY}


# Utilities


def _sliding_windows(positions, w_size: int, step: int):
    """Yield (start_idx, end_idx, center_pos) for SNP-count sliding windows."""
    n = len(positions)
    for i in range(0, n - w_size + 1, step):
        center = int(positions[i + w_size // 2])
        yield i, i + w_size, center


def _bp_windows(positions, w_size_bp: int, step_bp: int):
    """Yield (start_idx, end_idx, center_pos) for physical bp sliding windows."""
    max_pos = int(positions[-1])
    win_start = int(positions[0])
    while win_start <= max_pos:
        win_end = win_start + w_size_bp
        i = int(np.searchsorted(positions, win_start))
        j = int(np.searchsorted(positions, win_end, side="right"))
        if j - i >= 2:
            yield i, j, win_start + w_size_bp // 2
        win_start += step_bp
        if win_start >= max_pos:
            break


def _get_windows(positions, params):
    """Dispatch to SNP-count or bp sliding windows based on window_mode in params."""
    mode = params.get("window_mode", "snp")
    if mode == "bp":
        w_size_bp = int(params.get("w_size_bp", 1_000_000))
        step_bp = int(params.get("step_bp", 10_000))
        return _bp_windows(positions, w_size_bp, step_bp)
    else:
        w_size = int(params.get("w_size", 201))
        step = int(params.get("step", 10))
        return _sliding_windows(positions, w_size, step)


def _snp_cm_mb(positions: np.ndarray, rec_map: np.ndarray) -> np.ndarray:
    """Local recombination rate (cM/Mb) at each SNP position.

    Assigns each SNP to the rec_map segment it falls in and returns that
    segment's rate (Δ cM / (Δ bp / 1e6)). No arbitrary window size needed.

    Parameters
    ----------
    positions : 1D int64 array of SNP physical positions (bp)
    rec_map   : numpy array from genome_reader; col 0 = bp, last col = cumulative cM

    Returns
    -------
    1D float64 array, length == len(positions), values in cM/Mb (>= 0).
    """
    map_pos = rec_map[:, 0].astype(np.float64)
    map_cm = rec_map[:, -1].astype(np.float64)
    delta_bp = np.diff(map_pos)
    delta_cm = np.diff(map_cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        seg_rates = np.where(delta_bp > 0, delta_cm / (delta_bp / 1e6), 0.0)
    seg_rates = np.maximum(0.0, seg_rates)
    # Assign each SNP to the leftmost segment that starts at or before its position
    idx = np.clip(
        np.searchsorted(map_pos, positions, side="right") - 1,
        0,
        len(seg_rates) - 1,
    )
    return seg_rates[idx]


def _normalize_daf_bins(
    values: np.ndarray,
    daf: np.ndarray,
    recomb: np.ndarray | None = None,
    n_daf_bins: int = 50,
    n_r_bins: int | None = None,
) -> np.ndarray:
    """Z-score values within genome-wide DAF bins (+ recomb bins if provided).

    When ``recomb`` is provided, creates a joint (DAF × recomb_rate) grid:
    ``n_daf_bins`` equal-frequency DAF bins × ``n_r_bins`` equal-frequency
    recombination rate bins (Johnson et al. approach: 10 r_bins default).
    """
    if recomb is not None:
        daf_edges = np.nanpercentile(daf, np.linspace(0, 100, n_daf_bins + 1))
        daf_edges[0] -= 1e-10
        r_edges = np.nanpercentile(recomb, np.linspace(0, 100, n_r_bins + 1))
        r_edges[0] -= 1e-10
        daf_bin = np.clip(np.digitize(daf, daf_edges) - 1, 0, n_daf_bins - 1)
        r_bin = np.clip(np.digitize(recomb, r_edges) - 1, 0, n_r_bins - 1)
        bin_key = daf_bin * n_r_bins + r_bin
    else:
        edges = np.nanpercentile(daf, np.linspace(0, 100, n_daf_bins + 1))
        edges[0] -= 1e-10
        bin_key = np.clip(np.digitize(daf, edges) - 1, 0, n_daf_bins - 1)

    normalized = np.full_like(values, np.nan, dtype=np.float64)
    for b in np.unique(bin_key):
        mask = bin_key == b
        if mask.sum() < 2:
            continue
        v = values[mask].astype(np.float64)
        mu, std = np.nanmean(v), np.nanstd(v)
        if std > 0:
            normalized[mask] = (v - mu) / std
    return normalized


def empirical_pvalues(
    df: pl.DataFrame, stat_col: str, abs_rank: bool = False
) -> pl.DataFrame:
    """Empirical p-value following the empirical outlier approach (Akey 2009).

    p = rank(−value, na.last=keep) / N_valid
    Small p → outlier/candidate (locus in extreme upper tail of the empirical distribution).
    NaN → null in output (excluded from ranking and from N_valid denominator).

    abs_rank=True: rank by ``abs(value)`` before negating — for signed stats where
    large magnitude in either direction signals selection (iHS, nSL, Tajima's D).
    """
    s = df[stat_col]
    if abs_rank:
        s = s.abs()
    n_valid = int(s.is_not_null().sum() - s.is_nan().sum())
    # Negate: largest original value → most negative → rank 1 → p = 1/N_valid ≈ 0 (outlier)
    p_emp = (-s).fill_nan(None).rank(method="average") / n_valid
    return df.with_columns(p_emp.alias(f"{stat_col}_pvalue"))


# Per-SNP stat runners


def _run_ihs(hap, positions, ac, rec_map, genetic_pos, **params):
    min_maf = params.get("min_maf", 0.05)
    include_edges = params.get("include_edges", False)
    gap_scale = params.get("gap_scale", 20000)
    max_gap = params.get("max_gap", 200000)
    map_pos = genetic_pos if genetic_pos is not None else None
    df = ihs_ihh(
        hap,
        positions,
        map_pos=map_pos,
        min_maf=min_maf,
        min_ehh=0.05,
        include_edges=include_edges,
        gap_scale=gap_scale,
        max_gap=max_gap,
        use_threads=False,
    )
    if df is None or len(df) == 0:
        return None
    return df.select(["positions", "daf", "ihs"]).rename({"positions": "pos"})


def _run_nsl(hap, positions, ac, rec_map, genetic_pos, **params):
    min_maf = params.get("min_maf", 0.05)
    freqs = ac[:, 1] / ac.sum(axis=1)
    mask = (freqs >= min_maf) & (freqs <= 1 - min_maf)
    if mask.sum() < 2:
        return None
    nsl_vals = nsl(hap[mask], use_threads=False)
    return pl.DataFrame(
        {
            "pos": positions[mask].tolist(),
            "daf": freqs[mask].tolist(),
            "nsl": nsl_vals.tolist(),
        }
    )


def _run_isafe(hap, positions, ac, rec_map, genetic_pos, **params):
    region_size_bp = int(params.get("region_size_bp", 1_000_000))
    isafe_window = params.get("isafe_window", 300)
    isafe_step = params.get("isafe_step", 150)
    top_k = params.get("top_k", 1)
    max_rank = params.get("max_rank", 15)
    pos = positions
    results = []
    region_start = int(pos[0])
    max_pos = int(pos[-1])
    while region_start <= max_pos:
        region_end = region_start + region_size_bp
        mask = (pos >= region_start) & (pos < region_end)
        if mask.sum() >= 300:
            df_r = run_isafe(
                hap[mask],
                pos[mask],
                window=isafe_window,
                step=isafe_step,
                top_k=top_k,
                max_rank=max_rank,
            )
            if df_r is not None and len(df_r) > 0:
                results.append(
                    df_r.select(["positions", "daf", "isafe"]).rename(
                        {"positions": "pos"}
                    )
                )
        region_start = region_end
    return pl.concat(results) if results else None


def _run_dind(hap, positions, ac, rec_map, genetic_pos, **params):
    window_size = params.get("window_size", 50000)
    min_focal_freq = params.get("min_focal_freq", 0.25)
    max_focal_freq = params.get("max_focal_freq", 0.95)
    sq_freqs, info, _ = fast_sq_freq_pairs(
        hap,
        ac,
        rec_map,
        min_focal_freq=min_focal_freq,
        max_focal_freq=max_focal_freq,
        window_size=window_size,
    )
    if info.shape[0] == 0:
        return None
    r_dind, r_high, r_low = dind_high_low_from_pairs(sq_freqs, info)
    df, _, _, _ = fs_stats_dataframe(info, r_dind, r_high, r_low, [], [], [])
    if df is None or len(df) == 0:
        return None
    return df.select(["positions", "daf", "dind", "high_freq", "low_freq"]).rename(
        {"positions": "pos"}
    )


def _run_high_freq(hap, positions, ac, rec_map, genetic_pos, **params):
    window_size = params.get("window_size", 50000)
    min_focal_freq = params.get("min_focal_freq", 0.25)
    max_focal_freq = params.get("max_focal_freq", 0.95)
    sq_freqs, info, _ = fast_sq_freq_pairs(
        hap,
        ac,
        rec_map,
        min_focal_freq=min_focal_freq,
        max_focal_freq=max_focal_freq,
        window_size=window_size,
    )
    if info.shape[0] == 0:
        return None
    r_dind, r_high, r_low = dind_high_low_from_pairs(sq_freqs, info)
    df, _, _, _ = fs_stats_dataframe(info, r_dind, r_high, r_low, [], [], [])
    if df is None or len(df) == 0:
        return None
    return df.select(["positions", "daf", "high_freq"]).rename({"positions": "pos"})


def _run_low_freq(hap, positions, ac, rec_map, genetic_pos, **params):
    window_size = params.get("window_size", 50000)
    min_focal_freq = params.get("min_focal_freq", 0.25)
    max_focal_freq = params.get("max_focal_freq", 0.95)
    sq_freqs, info, _ = fast_sq_freq_pairs(
        hap,
        ac,
        rec_map,
        min_focal_freq=min_focal_freq,
        max_focal_freq=max_focal_freq,
        window_size=window_size,
    )
    if info.shape[0] == 0:
        return None
    r_dind, r_high, r_low = dind_high_low_from_pairs(sq_freqs, info)
    df, _, _, _ = fs_stats_dataframe(info, r_dind, r_high, r_low, [], [], [])
    if df is None or len(df) == 0:
        return None
    return df.select(["positions", "daf", "low_freq"]).rename({"positions": "pos"})


def _run_s_ratio(hap, positions, ac, rec_map, genetic_pos, **params):
    window_size = params.get("window_size", 50000)
    min_focal_freq = params.get("min_focal_freq", 0.25)
    max_focal_freq = params.get("max_focal_freq", 0.95)
    sq_freqs, info, _ = fast_sq_freq_pairs(
        hap,
        ac,
        rec_map,
        min_focal_freq=min_focal_freq,
        max_focal_freq=max_focal_freq,
        window_size=window_size,
    )
    if info.shape[0] == 0:
        return None
    r_s = s_ratio_from_pairs(sq_freqs)
    _, df, _, _ = fs_stats_dataframe(info, [], [], [], r_s, [], [])
    if df is None or len(df) == 0:
        return None
    return df.select(["positions", "daf", "s_ratio"]).rename({"positions": "pos"})


def _run_hapdaf_o(hap, positions, ac, rec_map, genetic_pos, **params):
    window_size = params.get("window_size", 50000)
    min_focal_freq = params.get("min_focal_freq", 0.25)
    max_focal_freq = params.get("max_focal_freq", 0.95)
    max_ancest_freq = params.get("max_ancest_freq", 0.25)
    min_tot_freq = params.get("min_tot_freq", 0.25)
    sq_freqs, info, _ = fast_sq_freq_pairs(
        hap,
        ac,
        rec_map,
        min_focal_freq=min_focal_freq,
        max_focal_freq=max_focal_freq,
        window_size=window_size,
    )
    if info.shape[0] == 0:
        return None
    r_hdo = hapdaf_from_pairs(sq_freqs, max_ancest_freq, min_tot_freq)
    _, _, df_o, _ = fs_stats_dataframe(info, [], [], [], [], r_hdo, [])
    if df_o is None or len(df_o) == 0:
        return None
    return df_o.select(["positions", "daf", "hapdaf_o"]).rename({"positions": "pos"})


def _run_hapdaf_s(hap, positions, ac, rec_map, genetic_pos, **params):
    window_size = params.get("window_size", 50000)
    min_focal_freq = params.get("min_focal_freq", 0.25)
    max_focal_freq = params.get("max_focal_freq", 0.95)
    max_ancest_freq = params.get("max_ancest_freq", 0.10)
    min_tot_freq = params.get("min_tot_freq", 0.10)
    sq_freqs, info, _ = fast_sq_freq_pairs(
        hap,
        ac,
        rec_map,
        min_focal_freq=min_focal_freq,
        max_focal_freq=max_focal_freq,
        window_size=window_size,
    )
    if info.shape[0] == 0:
        return None
    r_hds = hapdaf_from_pairs(sq_freqs, max_ancest_freq, min_tot_freq)
    _, _, _, df_s = fs_stats_dataframe(info, [], [], [], [], [], r_hds)
    if df_s is None or len(df_s) == 0:
        return None
    return df_s.select(["positions", "daf", "hapdaf_s"]).rename({"positions": "pos"})


def _run_haf(hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params):
    if _single_window is not None:
        i, j, center = _single_window
        val = haf_top(hap[i:j], positions[i:j])
        return pl.DataFrame({"pos": [center], "haf": [float(val)]})
    rows = []
    for i, j, center in _get_windows(positions, params):
        val = haf_top(hap[i:j], positions[i:j])
        rows.append({"pos": center, "haf": float(val)})
    if not rows:
        return None
    return pl.DataFrame(rows)


def _run_hscan(hap, positions, ac, rec_map, genetic_pos, **params):
    max_gap = params.get("max_gap", 200_000)
    dist_mode = params.get("dist_mode", 0)
    step = params.get("hscan_step", 1)
    pos_out, h_out = hscan(
        hap, positions, max_gap=max_gap, dist_mode=dist_mode, step=step
    )
    if len(pos_out) == 0:
        return None
    return pl.DataFrame({"pos": pos_out.astype(int).tolist(), "hscan": h_out.tolist()})


# Sliding-window stat runners


def _run_h12(hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params):
    if _single_window is not None:
        h12_val, h2_h1, h1, h123, _n = garud_h(hap)
        return pl.DataFrame(
            [
                {
                    "pos": _single_window,
                    "n_snps": len(positions),
                    "h12": float(h12_val),
                    "h2_h1": float(h2_h1),
                }
            ]
        )
    rows = []
    for i, j, center in _get_windows(positions, params):
        h12_val, h2_h1, h1, h123, _n = garud_h(hap[i:j])
        rows.append(
            {
                "pos": center,
                "n_snps": j - i,
                "h12": float(h12_val),
                "h2_h1": float(h2_h1),
            }
        )
    return pl.DataFrame(rows) if rows else None


def _run_garud(hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params):
    if _single_window is not None:
        h12_val, h2_h1, h1, h123, _n = garud_h(hap)
        return pl.DataFrame(
            [
                {
                    "pos": _single_window,
                    "n_snps": len(positions),
                    "h1": float(h1),
                    "h12": float(h12_val),
                    "h2_h1": float(h2_h1),
                }
            ]
        )
    rows = []
    for i, j, center in _get_windows(positions, params):
        h12_val, h2_h1, h1, h123, _n = garud_h(hap[i:j])
        rows.append(
            {
                "pos": center,
                "n_snps": j - i,
                "h1": float(h1),
                "h12": float(h12_val),
                "h2_h1": float(h2_h1),
            }
        )
    return pl.DataFrame(rows) if rows else None


def _run_neutrality(
    hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params
):
    if _single_window is not None:
        res = neutrality_stats(ac.astype(np.int32), positions)
        return pl.DataFrame(
            [
                {
                    "pos": _single_window,
                    "n_snps": len(positions),
                    "tajima_d": float(res[0]),
                    "fay_wu_h_norm": float(res[3]),
                    "pi": float(res[4]),
                    "theta_w": float(res[5]),
                }
            ]
        )
    rows = []
    for i, j, center in _get_windows(positions, params):
        res = neutrality_stats(ac[i:j].astype(np.int32), positions[i:j])
        rows.append(
            {
                "pos": center,
                "n_snps": j - i,
                "tajima_d": float(res[0]),
                "fay_wu_h_norm": float(res[3]),
                "pi": float(res[4]),
                "theta_w": float(res[5]),
            }
        )
    return pl.DataFrame(rows) if rows else None


def _run_omega(hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params):
    if _single_window is not None:
        hap_f = np.ascontiguousarray(hap.astype(np.float64))
        r2 = compute_r2_matrix_upper(hap_f)
        omega = omega_linear_correct(r2)
        return pl.DataFrame(
            [
                {
                    "pos": _single_window,
                    "n_snps": len(positions),
                    "omega_max": float(omega),
                }
            ]
        )
    rows = []
    for i, j, center in _get_windows(positions, params):
        hap_f = np.ascontiguousarray(hap[i:j].astype(np.float64))
        r2 = compute_r2_matrix_upper(hap_f)
        omega = omega_linear_correct(r2)
        rows.append({"pos": center, "n_snps": j - i, "omega_max": float(omega)})
    return pl.DataFrame(rows) if rows else None


def _run_zns(hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params):
    if _single_window is not None:
        hap_f = np.ascontiguousarray(hap.astype(np.float64))
        zns_val, _ = Ld(hap_f)
        return pl.DataFrame(
            [{"pos": _single_window, "n_snps": len(positions), "zns": float(zns_val)}]
        )
    rows = []
    for i, j, center in _get_windows(positions, params):
        hap_f = np.ascontiguousarray(hap[i:j].astype(np.float64))
        zns_val, _ = Ld(hap_f)
        rows.append({"pos": center, "n_snps": j - i, "zns": float(zns_val)})
    return pl.DataFrame(rows) if rows else None


def _run_lassi_scan(hap, positions, ac, rec_map, genetic_pos, **params):
    K_truncation = params.get("K_truncation", 10)
    w_size = params.get("w_size", 201)
    step = params.get("step", 10)
    sweep_mode = params.get("sweep_mode", 4)
    hap_data = [hap, positions]
    K_counts, K_spectrum, windows_lassi = LASSI_spectrum_and_Kspectrum(
        hap_data, K_truncation, w_size, int(step)
    )
    K_neutral = neut_average(np.vstack(K_spectrum))

    t_m = T_m_statistic_fast(
        K_counts, K_neutral, windows_lassi, K_truncation, sweep_mode=sweep_mode
    )
    return t_m.select([
        pl.col("window_lassi").cast(pl.Int64).alias("pos"),
        pl.col("T").alias("T_m"),
        pl.col("m"),
        pl.col("frequency").alias("epsilon"),
    ])


def _run_lassip_scan(hap, positions, ac, rec_map, genetic_pos, nthreads=1, **params):
    K_truncation = params.get("K_truncation", 10)
    w_size = params.get("w_size", 201)
    step = params.get("step", 10)
    sweep_mode = params.get("sweep_mode", 4)
    max_extend = params.get("max_extend", 1e5)
    n_A = params.get("n_A", 100)
    K_counts, K_spectrum, windows_centers = LASSI_spectrum_and_Kspectrum(
        [hap, positions], K_truncation, w_size, int(step)
    )
    K_neutral = neut_average(np.vstack(K_spectrum))
    return (
        Lambda_statistic_fast(
            K_counts,
            K_neutral,
            windows_centers,
            K_truncation,
            n_A=n_A,
            sweep_mode=sweep_mode,
            nthreads=nthreads,
            max_extend=max_extend,
        )
        .rename({"window_lassip": "pos"})
        .select(pl.exclude("iter"))
    )


def _run_raisd(hap, positions, ac, rec_map, genetic_pos, **params):
    window_size = params.get("window_size", 50)
    return mu_stat(hap, positions, window_size).rename({"positions": "pos"})


def _run_beta(hap, positions, ac, rec_map, genetic_pos, **params):
    m = params.get("m", 0.1)
    # use run_beta_window which takes ac and positions directly
    out = run_beta_window(ac, positions, m=m)
    if out is None or out.shape[0] == 0:
        return None
    return pl.DataFrame(
        {
            "pos": out[:, 0].astype(int).tolist(),
            "beta1": out[:, 1].tolist(),
            "beta2": out[:, 2].tolist(),
        }
    )


def _run_ncd(hap, positions, ac, rec_map, genetic_pos, **params):
    tf = params.get("tf", 0.5)
    w = params.get("w", 3000)
    minIS = params.get("minIS", 2)
    n_hap = ac.sum(axis=1)
    freqs = ac[:, 1] / n_hap
    # ncd1 returns results[valid_mask] with no positions — recompute valid window centers
    w1 = w / 2.0
    start_positions = np.arange(positions[0], positions[-1], w1)
    n_snps = len(positions)
    valid_centers = []
    j_start = 0
    j_end = 0
    for widx in range(len(start_positions)):
        start = start_positions[widx]
        end = start + w
        while j_start < n_snps and positions[j_start] < start:
            j_start += 1
        while j_end < n_snps and positions[j_end] <= end:
            j_end += 1
        if j_end - j_start >= minIS:
            valid_centers.append(int(start + w / 2.0))
    ncd_vals = ncd1(positions, freqs, tf=tf, w=w, minIS=minIS)
    if len(valid_centers) == 0 or len(ncd_vals) == 0:
        return None
    n_valid = min(len(valid_centers), len(ncd_vals))
    return pl.DataFrame(
        {"pos": valid_centers[:n_valid], "ncd1": ncd_vals[:n_valid].tolist()}
    )


# Individual neutrality runners (bp-window mode)


def _run_single_neutrality_array(
    hap, positions, ac, params, col_name, arr_idx, _single_window=None
):
    """Run per-window, extract from neutrality_stats() array by index."""
    if _single_window is not None:
        res = neutrality_stats(ac.astype(np.int32), positions)
        return pl.DataFrame(
            [
                {
                    "pos": _single_window,
                    "n_snps": len(positions),
                    col_name: float(res[arr_idx]),
                }
            ]
        )
    rows = []
    for i, j, center in _get_windows(positions, params):
        res = neutrality_stats(ac[i:j].astype(np.int32), positions[i:j])
        rows.append({"pos": center, "n_snps": j - i, col_name: float(res[arr_idx])})
    return pl.DataFrame(rows) if rows else None


def _run_single_neutrality_fn(
    hap, positions, ac, params, col_name, fn, _single_window=None
):
    """Run per-window, call individual function fn(ac_win[, pos_win])."""
    import inspect
    _fn_nparams = len(inspect.signature(fn).parameters)

    def _call(ac_win, pos_win):
        return float(fn(ac_win, pos_win) if _fn_nparams >= 2 else fn(ac_win))

    if _single_window is not None:
        val = _call(ac, positions)
        return pl.DataFrame(
            [{"pos": _single_window, "n_snps": len(positions), col_name: val}]
        )
    rows = []
    for i, j, center in _get_windows(positions, params):
        val = _call(ac[i:j], positions[i:j])
        rows.append({"pos": center, "n_snps": j - i, col_name: val})
    return pl.DataFrame(rows) if rows else None


def _run_tajima_d(
    hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params
):
    return _run_single_neutrality_array(
        hap, positions, ac, params, "tajima_d", 0, _single_window=_single_window
    )


def _run_pi(hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params):
    return _run_single_neutrality_array(
        hap, positions, ac, params, "pi", 4, _single_window=_single_window
    )


def _run_theta_w(
    hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params
):
    return _run_single_neutrality_array(
        hap, positions, ac, params, "theta_w", 5, _single_window=_single_window
    )


def _run_fay_wu_h(
    hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params
):
    return _run_single_neutrality_fn(
        hap,
        positions,
        ac,
        params,
        "fay_wu_h",
        fay_wu_h_norm,
        _single_window=_single_window,
    )


def _run_zeng_e(
    hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params
):
    return _run_single_neutrality_fn(
        hap, positions, ac, params, "zeng_e", zeng_e, _single_window=_single_window
    )


def _run_achaz_y(
    hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params
):
    return _run_single_neutrality_fn(
        hap, positions, ac, params, "achaz_y", achaz_y, _single_window=_single_window
    )


def _run_fuli_f(
    hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params
):
    return _run_single_neutrality_fn(
        hap, positions, ac, params, "fuli_f", fuli_f, _single_window=_single_window
    )


def _run_fuli_f_star(
    hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params
):
    return _run_single_neutrality_fn(
        hap,
        positions,
        ac,
        params,
        "fuli_f_star",
        fuli_f_star,
        _single_window=_single_window,
    )


def _run_fuli_d(
    hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params
):
    return _run_single_neutrality_fn(
        hap, positions, ac, params, "fuli_d", fuli_d, _single_window=_single_window
    )


def _run_fuli_d_star(
    hap, positions, ac, rec_map, genetic_pos, _single_window=None, **params
):
    return _run_single_neutrality_fn(
        hap,
        positions,
        ac,
        params,
        "fuli_d_star",
        fuli_d_star,
        _single_window=_single_window,
    )


# Map from stat key to runner function
_RUNNERS = {
    "ihs": _run_ihs,
    "nsl": _run_nsl,
    "isafe": _run_isafe,
    "dind": _run_dind,
    "high_freq": _run_high_freq,
    "low_freq": _run_low_freq,
    "s_ratio": _run_s_ratio,
    "hapdaf_o": _run_hapdaf_o,
    "hapdaf_s": _run_hapdaf_s,
    "haf": _run_haf,
    "hscan": _run_hscan,
    "h12": _run_h12,
    "garud": _run_garud,
    "tajima_d": _run_tajima_d,
    "pi": _run_pi,
    "theta_w": _run_theta_w,
    "fay_wu_h": _run_fay_wu_h,
    "zeng_e": _run_zeng_e,
    "achaz_y": _run_achaz_y,
    "fuli_f": _run_fuli_f,
    "fuli_f_star": _run_fuli_f_star,
    "fuli_d": _run_fuli_d,
    "fuli_d_star": _run_fuli_d_star,
    "neutrality": _run_neutrality,
    "omega": _run_omega,
    "zns": _run_zns,
    "lassi": _run_lassi_scan,
    "lassip": _run_lassip_scan,
    "raisd": _run_raisd,
    "beta": _run_beta,
    "ncd": _run_ncd,
}

# Stats that need DAF-bin normalization after computation
_NORMALIZE_BY_DAF = {
    "ihs",
    "nsl",
    "dind",
    "high_freq",
    "low_freq",
    "s_ratio",
    "hapdaf_o",
    "hapdaf_s",
}
# Stats where rank is by absolute value (signed stats)
_ABS_RANK = {
    "ihs",
    "nsl",
    "tajima_d",
    "zeng_e",
    "fay_wu_h",
    "neutrality",
}

# Stat category sets for global task-pool parallelism
# Per-SNP flat: one task per stat per chromosome (whole chromosome in one shot)
_SNP_FLAT = {
    "ihs",
    "nsl",
    "dind",
    "high_freq",
    "low_freq",
    "s_ratio",
    "hapdaf_o",
    "hapdaf_s",
    "hscan",
}
# Per-SNP regional: isafe runs as non-overlapping 1–5 Mb chunks, one task per region
_SNP_REGIONAL = {"isafe"}
# Window batchable: windows are independent → split into nthreads batches per chromosome
_WINDOW_BATCHABLE = {
    "h12",
    "garud",
    "haf",
    "neutrality",
    "omega",
    "zns",
    "tajima_d",
    "pi",
    "theta_w",
    "fay_wu_h",
    "zeng_e",
    "achaz_y",
    "fuli_f",
    "fuli_f_star",
    "fuli_d",
    "fuli_d_star",
}
# Window whole: stat needs all windows at once (LASSI spatial kernel, RAiSD, beta/ncd)
_WINDOW_WHOLE = {"lassi", "lassip", "raisd", "beta", "ncd"}


# Parallelism helpers (called from global task pool in scan())


def _region_bounds(positions, region_size_bp: int):
    """Yield non-overlapping (lo, hi) bp bounds covering all positions."""
    lo = int(positions[0])
    max_pos = int(positions[-1])
    while lo <= max_pos:
        yield lo, lo + region_size_bp
        lo += region_size_bp


def _run_isafe_region(hap_slice, pos_slice, ac_slice, params):
    """Run isafe on a pre-sliced non-overlapping chromosome region (one task)."""
    isafe_window = params.get("isafe_window", 300)
    isafe_step = params.get("isafe_step", 150)
    top_k = params.get("top_k", 1)
    max_rank = params.get("max_rank", 15)
    if len(pos_slice) < 300:
        return None
    df_r = run_isafe(
        hap_slice,
        pos_slice,
        window=isafe_window,
        step=isafe_step,
        top_k=top_k,
        max_rank=max_rank,
    )
    if df_r is None or len(df_r) == 0:
        return None
    return df_r.select(["positions", "daf", "isafe"]).rename({"positions": "pos"})


def _run_window_batch(
    stat_key, hap, positions, ac, rec_map, genetic_pos, window_batch, **params
):
    """Run a batchable window stat on a pre-enumerated list of (i, j, center_pos) windows.

    Called from the global joblib pool. Each window is computed independently.
    The runner is called with pre-sliced arrays and _single_window=center_pos so
    it skips its internal _get_windows loop and computes directly on the slice.
    """
    runner = _RUNNERS[stat_key]
    parts = []
    for i, j, center_pos in window_batch:
        try:
            df = runner(
                hap[i:j],
                positions[i:j],
                ac[i:j],
                rec_map,
                genetic_pos,
                _single_window=center_pos,
                **params,
            )
        except Exception:
            continue
        if df is not None and len(df) > 0:
            parts.append(df)
    return pl.concat(parts) if parts else None


# Main scan function
def scan(
    vcf_path,
    out_prefix,
    stats,
    config=None,
    w_size=201,
    step=10,
    w_size_bp=1_000_000,
    step_bp=10_000,
    min_maf=0.05,
    recombination_map=None,
    n_daf_bins=50,
    n_r_bins=None,
    nthreads=1,
    window_mode="auto",
    **kwargs,
) -> dict[str, pl.DataFrame]:
    """Standalone outlier scan from a directory of per-chromosome VCF files.

    Uses a global task pool across ALL VCF files simultaneously: all chromosomes
    are pre-loaded, then one ``Parallel(n_jobs=nthreads)`` call processes every
    stat × chromosome combination. This fully exploits nthreads regardless of how
    many chromosomes are present.

    Parameters
    ----------
    vcf_path:
        Directory containing ``*.vcf.gz`` (or ``*.bcf.gz``) files, one per
        chromosome/contig. Must be a directory; single-file input is not supported.
    out_prefix:
        Output file prefix. Writes ``{out_prefix}.{stat}.txt`` for each stat.
    stats:
        List of stat keys to compute. See ``available_stats()`` for options.
        Per-SNP: ihs, nsl, isafe, dind, high_freq, low_freq, s_ratio,
        hapdaf_o, hapdaf_s.
        SNP-window: h12, garud, lassi, lassip, raisd.
        bp-window: tajima_d, pi, theta_w, fay_wu_h, zeng_e, achaz_y,
        fuli_f, fuli_f_star, fuli_d, fuli_d_star, neutrality, omega, beta, ncd.
    config:
        Per-stat parameter overrides, e.g.
        ``{"raisd": {"window_size": 100}, "lassip": {"max_extend": 5e4}}``.
        Overrides ``w_size``, ``step``, and any kwargs for that stat only.
    w_size:
        SNP-count window size for SNP-mode sliding-window stats (default 201).
    step:
        SNP step for SNP-mode sliding-window stats (default 10).
    w_size_bp:
        Physical window size in bp for bp-mode stats (default 1 Mb).
    step_bp:
        Physical step size in bp for bp-mode stats (default 10 kb).
    min_maf:
        Minimum minor allele frequency for iHS and nSL (default 0.05).
    recombination_map:
        Path to recombination map TSV (chr, start, end, cm_mb, cm).
        If provided: genetic distances used for T3 stat windows, and
        frequency-sensitive stats (iHS, nSL, dind, …) are normalized
        by joint (DAF × recomb_rate) bins (Johnson et al. approach).
    n_daf_bins:
        Number of equal-frequency DAF bins for normalization (default 50).
    n_r_bins:
        Number of equal-frequency recombination rate bins for joint
        (DAF × r_bins) normalization. ``None`` (default) → DAF-only normalization.
        Set to 10 (Johnson et al.) to enable joint normalization when
        ``recombination_map`` is provided.
    nthreads:
        Total worker threads for the global task pool (default 1).
        Window-batchable stats split each chromosome into ``nthreads`` window
        batches so every thread stays busy.
    window_mode:
        Override window mode for all sliding-window stats.
        "auto" (default) uses per-stat defaults from STAT_REGISTRY.
        "snp" forces SNP-count windows for all window stats.
        "bp" forces physical bp windows for all window stats.
    kwargs:
        Shared overrides forwarded to all stats: max_extend, K_truncation,
        sweep_mode, raisd_window, tf, etc.

    Returns
    -------
    dict[str, polars.DataFrame]
        Keys are stat names; each DataFrame has a ``{rank_col}_pvalue`` column.
        Files written to ``{out_prefix}.{stat}.txt`` (tab-separated).
    """

    unknown = [s for s in stats if s not in STAT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown stats: {unknown}. Available: {available_stats()}")

    config = config or {}

    # vcf_path must be a directory
    if not os.path.isdir(vcf_path):
        raise ValueError(
            f"vcf_path must be a directory of *.vcf.gz files, got: {vcf_path!r}. "
            "Use --vcf_path pointing to a directory."
        )
    vcf_files = sorted(
        glob.glob(os.path.join(vcf_path, "*.vcf.gz"))
        + glob.glob(os.path.join(vcf_path, "*.bcf.gz"))
    )
    if not vcf_files:
        raise FileNotFoundError(f"No *.vcf.gz or *.bcf.gz files found in {vcf_path}")

    def _make_params(stat_key):
        """Merge params: registry defaults < shared < per-stat config."""
        p = {**STAT_REGISTRY[stat_key].default_params}
        if window_mode != "auto":
            p["window_mode"] = window_mode
        p.update(
            {
                "w_size": w_size,
                "step": step,
                "w_size_bp": w_size_bp,
                "step_bp": step_bp,
                "min_maf": min_maf,
            }
        )
        p.update(config.get(stat_key, {}))
        return p

    # ------------------------------------------------------------------
    # Phase 1: Pre-load all chromosomes sequentially (genome_reader uses
    # pysam which is not thread-safe, so this must stay sequential).
    # ------------------------------------------------------------------
    chrom_data: dict = {}  # chrom → (hap_int, rec_map, ac, positions, genetic_pos, recomb_vals)
    for vcf_file in vcf_files:
        hap_int, rec_map, ac, _, position_masked, genetic_pos = genome_reader(
            vcf_file, recombination_map=recombination_map
        )
        chrom = str(int(rec_map[0, 0]))
        recomb_vals = _snp_cm_mb(position_masked, rec_map) if recombination_map is not None else None
        chrom_data[chrom] = (
            hap_int,
            rec_map,
            ac,
            position_masked,
            genetic_pos,
            recomb_vals,
        )

    # ------------------------------------------------------------------
    # Phase 2: Build one global flat task list across ALL chromosomes
    # and ALL stats. Three task categories:
    #   snp       — 1 task per stat per chromosome (whole-chromosome runner)
    #   isafe     — 1 task per non-overlapping 1-5 Mb region per chromosome
    #   win_batch — window batches (nthreads tasks per chrom per batchable stat)
    #   win_whole — 1 whole-chromosome task per chromosome (lassi/lassip/raisd/…)
    # ------------------------------------------------------------------
    all_tasks: list = []
    all_labels: list = []

    for chrom, (
        hap_int,
        rec_map,
        ac,
        position_masked,
        genetic_pos,
        _,
    ) in chrom_data.items():
        for stat_key in stats:
            params = _make_params(stat_key)

            if stat_key in _SNP_REGIONAL:
                # isafe: non-overlapping region tasks
                if "region_size_bp" not in params:
                    raise KeyError(
                        f"Stat '{stat_key}' (snp-regional) requires 'region_size_bp' "
                        f"in STAT_REGISTRY default_params or config override."
                    )
                region_size_bp = int(params.get("region_size_bp", 1_000_000))
                for region_idx, (lo, hi) in enumerate(
                    _region_bounds(position_masked, region_size_bp)
                ):
                    mask = (position_masked >= lo) & (position_masked < hi)
                    if mask.sum() >= 300:
                        all_tasks.append(
                            delayed(_run_isafe_region)(
                                hap_int[mask], position_masked[mask], ac[mask], params
                            )
                        )
                        all_labels.append(("isafe", chrom, region_idx))

            elif stat_key in _SNP_FLAT:
                # Per-SNP flat: one whole-chromosome task per stat
                all_tasks.append(
                    delayed(_RUNNERS[stat_key])(
                        hap_int, position_masked, ac, rec_map, genetic_pos, **params
                    )
                )
                all_labels.append(("snp", chrom, stat_key))

            elif stat_key in _WINDOW_BATCHABLE:
                # Split windows into up to nthreads batches per chromosome
                wm = params.get("window_mode", "snp")
                if wm == "bp" and (
                    "w_size_bp" not in params or "step_bp" not in params
                ):
                    raise KeyError(
                        f"Stat '{stat_key}' uses window_mode='bp' but 'w_size_bp' or "
                        f"'step_bp' are missing from STAT_REGISTRY default_params."
                    )
                if wm != "bp" and ("w_size" not in params or "step" not in params):
                    raise KeyError(
                        f"Stat '{stat_key}' uses window_mode='{wm}' but 'w_size' or "
                        f"'step' are missing from STAT_REGISTRY default_params."
                    )
                all_windows = list(_get_windows(position_masked, params))
                if not all_windows:
                    continue
                chunk_size = max(1, ceil(len(all_windows) / nthreads))
                for batch_idx, start in enumerate(
                    range(0, len(all_windows), chunk_size)
                ):
                    batch = all_windows[start : start + chunk_size]
                    all_tasks.append(
                        delayed(_run_window_batch)(
                            stat_key,
                            hap_int,
                            position_masked,
                            ac,
                            rec_map,
                            genetic_pos,
                            batch,
                            **params,
                        )
                    )
                    all_labels.append(("win_batch", chrom, stat_key, batch_idx))

            else:
                # _WINDOW_WHOLE: whole-chromosome task (lassi, lassip, raisd, beta, ncd)
                # Pass nthreads=1 to avoid nested parallelism inside the pool.
                all_tasks.append(
                    delayed(_RUNNERS[stat_key])(
                        hap_int,
                        position_masked,
                        ac,
                        rec_map,
                        genetic_pos,
                        nthreads=1,
                        **params,
                    )
                )
                all_labels.append(("win_whole", chrom, stat_key))

    # ------------------------------------------------------------------
    # Phase 3: Single global Parallel call — exploits all nthreads across
    # every chromosome and stat simultaneously.
    # ------------------------------------------------------------------

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        task_results = Parallel(n_jobs=nthreads, backend="loky", verbose=2)(all_tasks)

    # ------------------------------------------------------------------
    # Phase 4: Collect results and reassemble into raw_per_stat dict.
    # ------------------------------------------------------------------
    raw_per_stat: dict[str, list] = {s: [] for s in stats}
    isafe_parts: dict[str, list] = {}  # chrom → list of region DataFrames
    win_batch_parts: dict[tuple, list] = {}  # (chrom, stat) → list of batch DataFrames

    for label, result in zip(all_labels, task_results):
        if result is None or (hasattr(result, "__len__") and len(result) == 0):
            continue
        kind = label[0]
        chrom = label[1]

        if kind == "snp":
            stat_key = label[2]
            df = result.with_columns(pl.lit(chrom).alias("chrom")).select(
                ["chrom", "pos"]
                + [c for c in result.columns if c not in ("chrom", "pos")]
            )
            pos_masked = chrom_data[chrom][3]
            recomb_vals = chrom_data[chrom][5]
            raw_per_stat[stat_key].append((df, pos_masked, recomb_vals))

        elif kind == "isafe":
            df = result.with_columns(pl.lit(chrom).alias("chrom"))
            isafe_parts.setdefault(chrom, []).append(df)

        elif kind == "win_batch":
            stat_key = label[2]
            df = result.with_columns(pl.lit(chrom).alias("chrom")).select(
                ["chrom", "pos"]
                + [c for c in result.columns if c not in ("chrom", "pos")]
            )
            win_batch_parts.setdefault((chrom, stat_key), []).append(df)

        elif kind == "win_whole":
            stat_key = label[2]
            df = result.with_columns(pl.lit(chrom).alias("chrom")).select(
                ["chrom", "pos"]
                + [c for c in result.columns if c not in ("chrom", "pos")]
            )
            pos_masked = chrom_data[chrom][3]
            recomb_vals = chrom_data[chrom][5]
            raw_per_stat[stat_key].append((df, pos_masked, recomb_vals))

    # Consolidate isafe: regions are non-overlapping, just concat per chromosome
    if "isafe" in stats:
        for chrom, parts in isafe_parts.items():
            df_iso = pl.concat(parts).select(
                ["chrom", "pos"]
                + [c for c in parts[0].columns if c not in ("chrom", "pos")]
            )
            pos_masked = chrom_data[chrom][3]
            recomb_vals = chrom_data[chrom][5]
            raw_per_stat["isafe"].append((df_iso, pos_masked, recomb_vals))

    # Consolidate window batches: concat all batches per (chrom, stat)
    for (chrom, stat_key), parts in win_batch_parts.items():
        df_win = pl.concat(parts)
        pos_masked = chrom_data[chrom][3]
        recomb_vals = chrom_data[chrom][5]
        raw_per_stat[stat_key].append((df_win, pos_masked, recomb_vals))

    # ------------------------------------------------------------------
    # Phase 5: Genome-wide DAF normalization, ranking, and output writing.
    # ------------------------------------------------------------------
    results: dict[str, pl.DataFrame] = {}

    for stat_key in stats:
        if not raw_per_stat[stat_key]:
            continue

        defn = STAT_REGISTRY[stat_key]
        rank_col = defn.rank_col

        df_all = pl.concat([t[0] for t in raw_per_stat[stat_key]])

        # Genome-wide DAF-bin normalization for frequency-sensitive per-SNP stats
        if stat_key in _NORMALIZE_BY_DAF and "daf" in df_all.columns:
            daf = df_all["daf"].to_numpy()
            recomb_for_norm = None
            if recombination_map is not None and n_r_bins is not None:
                aligned_parts = []
                for df_contig, pos_masked, rec_vals in raw_per_stat[stat_key]:
                    if rec_vals is None:
                        aligned_parts.append(np.full(len(df_contig), np.nan))
                    else:
                        pos_arr = df_contig["pos"].to_numpy()
                        idx = np.clip(
                            np.searchsorted(pos_masked, pos_arr),
                            0,
                            len(rec_vals) - 1,
                        )
                        aligned_parts.append(rec_vals[idx])
                recomb_for_norm = np.concatenate(aligned_parts)
            if rank_col in df_all.columns:
                vals = df_all[rank_col].to_numpy().astype(np.float64)
                normalized = _normalize_daf_bins(vals, daf, recomb_for_norm, n_daf_bins, n_r_bins)
                df_all = df_all.with_columns(pl.Series(rank_col, normalized))

        if rank_col in df_all.columns:
            df_all = empirical_pvalues(
                df_all, rank_col, abs_rank=(stat_key in _ABS_RANK)
            )

        df_all.write_csv(f"{out_prefix}.{stat_key}.txt", separator="\t")
        results[stat_key] = df_all.sort("chrom", "pos")

    return results
