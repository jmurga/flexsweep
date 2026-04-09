import glob
import gzip
import math
import re
from collections import defaultdict

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from allel import (
    GenotypeArray,
    read_vcf,
    windowed_count,
    windowed_diversity,
    windowed_watterson_theta,
)
from numba import int64, njit

# from pybedtools import BedTool
from polars_bio import merge, nearest, overlap

from . import Parallel, delayed, np, pl
from .fv import get_cm


################## Plotting


def plot_diversity(data_dir, nthreads=1):
    vcf_files = glob.glob(f"{data_dir}/*vcf.gz")
    vcf_files = [p for p in vcf_files if "masked" not in p]
    with Parallel(n_jobs=nthreads, verbose=2) as parallel:
        vcf_data = parallel(delayed(read_vcf)(i) for i in vcf_files)

    out = []
    for v in vcf_data:
        hap = GenotypeArray(v["calldata/GT"]).to_haplotypes()
        ac = hap.count_alleles()
        _pos = v["variants/POS"]
        pi_w, _w_pi, _n_pi, _c_pi = windowed_diversity(
            _pos, ac, size=int(1e5), step=int(5e4)
        )
        theta_w, _w_theta, _n_theta, _c_theta = windowed_watterson_theta(
            _pos, ac, size=int(1e5), step=int(5e4)
        )

        _counts, _windows = windowed_count(_pos, size=int(1e5), step=int(5e4))

        nchr = np.unique(v["variants/CHROM"])[0]

        tmp_1 = pl.DataFrame(
            {
                "contig": nchr,
                "start": _w_pi[:, 0],
                "end": _w_pi[:, 1],
                "pi": pi_w,
                "theta_w": theta_w,
                "s": _counts,
                "window_size": int(1e5),
            }
        )

        #######

        pi_w, _w_pi, _n_pi, _c_pi = windowed_diversity(
            _pos, ac, size=int(1.2e6), step=int(1e5)
        )
        theta_w, _w_theta, _n_theta, _c_theta = windowed_watterson_theta(
            _pos, ac, size=int(1.2e6), step=int(1e5)
        )
        _counts, _windows = windowed_count(_pos, size=int(1.2e6), step=int(1e5))

        tmp_2 = pl.DataFrame(
            {
                "contig": nchr,
                "start": _w_pi[:, 0],
                "end": _w_pi[:, 1],
                "pi": pi_w,
                "theta_w": theta_w,
                "s": _counts,
                "window_size": int(1.2e6),
            }
        )

        out.append(pl.concat([tmp_1, tmp_2]))

    df = pl.concat(out).sort("contig")

    df_plot = df.with_columns(
        ((pl.col("start") + pl.col("end")) / 2).alias("mid")
    ).filter(pl.col("window_size") == int(1.2e6))

    # Preserve original contig order
    contig_order = df_plot.select("contig").unique(maintain_order=True)

    # Compute contig lengths
    contig_lengths = (
        df_plot.group_by("contig")
        .agg(pl.max("end").alias("length"))
        .join(contig_order, on="contig")
    )

    # Cumulative offsets
    contig_offsets = contig_lengths.with_columns(
        pl.col("length").cum_sum().shift(1).fill_null(0).alias("offset")
    )

    # Genome-wide positions
    df_plot = df_plot.join(
        contig_offsets.select(["contig", "offset"]), on="contig"
    ).with_columns((pl.col("mid") + pl.col("offset")).alias("genome_pos"))

    pdf = df_plot.sort("genome_pos").to_pandas()
    boundaries = contig_offsets.to_pandas()

    # -----------------------
    # 2. Compute mean per contig
    # -----------------------
    stats = ["pi", "theta_w", "s"]
    means = df_plot.group_by("contig").agg([pl.mean(s).alias(s) for s in stats])
    means_pdf = means.join(contig_offsets, on="contig").to_pandas()
    means_pdf["start_pos"] = means_pdf["offset"]
    means_pdf["end_pos"] = means_pdf["offset"] + means_pdf["length"]

    # -----------------------
    # 3. Set publication-ready colors
    # -----------------------
    colors = {
        "pi": "#1f77b4",  # blue
        "theta_w": "#ff7f0e",  # orange
        "s": "#2ca02c",  # green
    }

    # -----------------------
    # 4. Plot genome tracks
    # -----------------------

    ylabels = ["π", "θ watterson", "S"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    for ax, stat, ylabel in zip(axes, stats, ylabels):
        ax.plot(pdf["genome_pos"], pdf[stat], color=colors[stat], linewidth=1)

        # Vertical lines between contigs (skip first contig)
        if len(boundaries) > 1:
            for offset in boundaries["offset"].iloc[1:]:
                ax.axvline(offset, color="black", linestyle="-.", alpha=0.5)

        # Mean lines per contig
        for _, row in means_pdf.iterrows():
            ax.hlines(
                y=row[stat],
                xmin=row["start_pos"],
                xmax=row["end_pos"],
                colors="gray",
                linestyles="dashed",
                alpha=0.7,
            )

        # Clean axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(ylabel, fontsize=12)

    axes[-1].set_xlabel("genome position")

    # -----------------------
    # 5. Chromosome labels
    # -----------------------
    if len(boundaries) > 1:
        centers = boundaries.copy()
        centers["center"] = centers["offset"] + centers["length"] / 2
        axes[-1].set_xticks(centers["center"])
        axes[-1].set_xticklabels(centers["contig"], rotation=90)

    plt.tight_layout()
    plt.show()

    return df


def _lncomb(N, k):
    """Log of N choose k, vectorized over k. Returns -inf for out-of-range."""
    from scipy.special import gammaln

    with np.errstate(invalid="ignore"):
        result = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
    return np.where(np.isfinite(result), result, -np.inf)


def _project_sfs(sfs, n_proj):
    """
    Project a 1-D SFS (length proj_from+1) to n_proj chromosomes.

    Uses hypergeometric weights following dadi/moments convention
    (_cached_projection). Fixed sites (index 0 and proj_from) are not
    projected. Returns array of length n_proj+1.
    """
    proj_from = len(sfs) - 1
    p_sfs = np.zeros(n_proj + 1)
    proj_hits = np.arange(n_proj + 1)
    lnc_to = _lncomb(n_proj, proj_hits)

    for hits in range(1, proj_from):
        if sfs[hits] == 0:
            continue
        lncontrib = (
            lnc_to
            + _lncomb(proj_from - n_proj, hits - proj_hits)
            - float(_lncomb(proj_from, hits))
        )
        contrib = np.exp(np.where(np.isfinite(lncontrib), lncontrib, -np.inf))
        least = max(n_proj - (proj_from - hits), 0)
        most = min(hits, n_proj)
        p_sfs[least : most + 1] += sfs[hits] * contrib[least : most + 1]

    return p_sfs


def plot_sfs(
    vcf_path,
    n_proj=None,
    fold=False,
    merge=False,
    nthreads=1,
    figsize=None,
    title=None,
    out=None,
):
    """
    Plot the Site Frequency Spectrum (SFS) from one or more VCF files.

    Each bar shows the percentage of segregating sites at that frequency class
    (singletons, doubletons, …). For multiple files the bars are plotted
    side-by-side within each frequency class, one bar per file.

    Parameters
    ----------
    vcf_path : str
        Path to a single VCF/VCF.gz file *or* a directory of ``*.vcf.gz``
        files (files containing "masked" in the name are skipped).
    n_proj : int, optional
        Diploid sample size to project down to (e.g. 50 → 100 haplotypes,
        SFS runs 1–99).  Must be smaller than the actual diploid sample size.
    fold : bool, default False
        If True, use minor-allele count (folded SFS).  If False, treat the
        ALT allele as derived (unfolded SFS).
    merge : bool, default True
        If True, pool all VCF files into a single SFS labelled with
        ``vcf_path``.  If False, plot one bar group per file.
    nthreads : int, default 1
        Parallel workers for reading multiple VCF files.
    figsize : tuple, optional
        Matplotlib figure size.  Defaults to ``(max(8, n_classes * 0.6 + 2), 4)``.
    title : str, optional
        Figure title.
    out : str, optional
        If given, save the figure to this path instead of displaying it.

    Returns
    -------
    polars.DataFrame
        Columns: ``dataset``, ``dac``, ``count``, ``pct``.
    """
    import os

    if os.path.isdir(vcf_path):
        vcf_files = sorted(glob.glob(f"{vcf_path}/*.vcf.gz"))
        vcf_files = [p for p in vcf_files if "masked" not in p]
    else:
        vcf_files = [vcf_path]

    if not vcf_files:
        raise FileNotFoundError(f"No VCF files found at {vcf_path}")

    def _read_ac(vcf_file):
        """Return (dac array, n_max) for one VCF file, or None on failure."""
        raw = read_vcf(vcf_file)
        if raw is None:
            return None
        ac = GenotypeArray(raw["calldata/GT"]).count_alleles()
        bial = ac.is_biallelic_01()
        ac = ac[bial]
        if len(ac) == 0:
            return None
        n_called = ac.sum(axis=1)
        n_max = int(n_called.max())
        dac = np.minimum(ac[:, 0], ac[:, 1]) if fold else ac[:, 1]
        mask = n_called == n_max
        return dac[mask].astype(int), n_max

    def _sfs_frame(dac_full, n_max, label):
        """Build a pl.DataFrame from raw dac counts."""
        sfs = np.bincount(dac_full, minlength=n_max + 1).astype(float)

        if n_proj is not None:
            n_hap = n_proj * 2
            if n_hap >= n_max:
                raise ValueError(
                    f"n_proj={n_proj} diploid ({n_hap} haplotypes) must be "
                    f"< actual diploid sample size {n_max // 2} ({n_max} haplotypes)."
                )
            sfs = _project_sfs(sfs, n_hap)
            n_total = n_hap
        else:
            n_total = n_max

        sfs_trim = sfs[1 : n_total // 2 + 1] if fold else sfs[1:-1]
        total = sfs_trim.sum()
        sfs_pct = sfs_trim / total * 100 if total > 0 else sfs_trim
        dac_idx = np.arange(1, len(sfs_trim) + 1)
        return pl.DataFrame(
            {"dataset": label, "dac": dac_idx, "count": sfs_trim, "pct": sfs_pct}
        )

    with Parallel(n_jobs=nthreads, verbose=2) as parallel:
        ac_results = parallel(delayed(_read_ac)(f) for f in vcf_files)

    ac_results = [r for r in ac_results if r is not None]
    if not ac_results:
        raise ValueError("No valid SFS could be computed from the input files.")

    if merge:
        # Pool all dac arrays; require same n_max across files
        n_values = [n for _, n in ac_results]
        n_max = min(n_values)
        dac_all = np.concatenate([d[d <= n_max] for d, _ in ac_results])
        df = _sfs_frame(dac_all, n_max, os.path.basename(vcf_path.rstrip("/")))
    else:
        frames = []
        for (dac_full, n_max), vcf_file in zip(ac_results, vcf_files):
            label = (
                os.path.basename(vcf_file).replace(".vcf.gz", "").replace(".vcf", "")
            )
            frames.append(_sfs_frame(dac_full, n_max, label))
        df = pl.concat(frames)

    # Align to minimum shared dac range across datasets
    max_shared = df.group_by("dataset").agg(pl.max("dac")).select(pl.min("dac")).item()
    df = df.filter(pl.col("dac") <= max_shared)

    # Plot

    labels = df.select("dataset").unique(maintain_order=True).to_series().to_list()
    n_files = len(labels)
    n_classes = max_shared

    x = np.arange(1, n_classes + 1)
    bar_width = 0.8 / n_files
    offsets = (np.arange(n_files) - n_files / 2 + 0.5) * bar_width
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_files))

    if figsize is None:
        figsize = (max(8, n_classes * 0.6 + 2), 4)

    fig, ax = plt.subplots(figsize=figsize)

    for i, label in enumerate(labels):
        pct = df.filter(pl.col("dataset") == label).sort("dac")["pct"].to_numpy()
        ax.bar(
            x + offsets[i],
            pct,
            width=bar_width,
            label=label,
            color=colors[i],
            alpha=0.85,
            edgecolor="none",
        )

    xlabel = "Minor allele count" if fold else "Derived allele count"
    if n_proj is not None:
        xlabel += f" (projected n={n_proj} diploid)"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Segregating sites (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if n_files > 1:
        ax.legend(fontsize=9, frameon=False, bbox_to_anchor=(0.75, 1), loc="upper left")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if out is not None:
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return df


def plot_manhattan(
    input,
    eps: float = 1e-10,
    chr_col: str | None = None,
    pos_col: str | None = None,
    p_col: str | None = None,
    log_transform: bool = True,
    threshold_lines: list | None = None,
    figsize: tuple = (14, 5),
    out: str | None = None,
    title: str | None = None,
):
    """Genome-wide Manhattan plot.

    Works for both CNN output (``prob_sweep``) and scan.py output (``{stat}_pvalue``).
    All new parameters default to None/True to preserve backward-compatible CNN behaviour.

    Parameters
    ----------
    input : str | polars.DataFrame
        CSV/DataFrame with genomic positions and p-values.
    eps : float
        Floor applied to p-values before -log10 to avoid inf.
    chr_col : str, optional
        Chromosome column name. None → ``"chr"`` (CNN default).
    pos_col : str, optional
        Position column name. None → ``"start"`` (CNN default).
    p_col : str, optional
        Column to use as p-value directly. None → compute ``1 - prob_sweep`` (CNN default).
    log_transform : bool
        If True (default), plot ``-log10(P)`` on y-axis. If False, plot raw values.
    threshold_lines : list of (y_value, linestyle, label), optional
        Horizontal threshold lines. None → CNN defaults (y=3 solid, y=2 dashed).
        Pass ``[]`` for no lines.
    figsize : tuple
        Figure size in inches.
    out : str, optional
        Save path. If None, shows interactively.
    title : str, optional
        Plot title.
    """
    # Load
    if isinstance(input, str):
        try:
            df = pl.read_csv(input, separator=",")
        except Exception:
            df = None
    else:
        df = input

    # CNN-specific stats (only when predicted_model column present)
    if "predicted_model" in df.columns:
        pred_counts = df["predicted_model"].value_counts()
        total = len(df)
        count_1 = (
            pred_counts.filter(pl.col("predicted_model") == 1)["count"][0]
            if 1 in pred_counts["predicted_model"].to_list()
            else 0
        )

    # Resolve column names
    _chr_col = chr_col or "chr"
    _pos_col = pos_col or "start"

    # Build CHR, BP, P columns
    if p_col is not None:
        df = df.with_columns(
            [
                pl.col(_chr_col)
                .cast(pl.Utf8)
                .str.replace("^chr", "")
                .cast(pl.Float64)
                .alias("CHR"),
                pl.col(p_col).cast(pl.Float64).clip(lower_bound=eps).alias("P"),
            ]
        ).select(["CHR", pl.col(_pos_col).cast(pl.Float64).alias("BP"), "P"])
    else:
        # CNN default: P = 1 - prob_sweep
        df = df.with_columns(
            [
                pl.col(_chr_col)
                .cast(pl.Utf8)
                .str.replace("^chr", "")
                .cast(pl.Float64)
                .alias("CHR"),
                (1 - pl.col("prob_sweep")).clip(lower_bound=eps).alias("P"),
            ]
        ).select(["CHR", pl.col(_pos_col).cast(pl.Float64).alias("BP"), "P"])

    # Compute cumulative chromosome offsets
    chr_lens = (
        df.group_by("CHR")
        .agg(pl.col("BP").max().alias("chr_len"))
        .sort("CHR")
        .with_columns((pl.col("chr_len").cum_sum() - pl.col("chr_len")).alias("tot"))
        .select(["CHR", "tot"])
    )

    df = (
        df.join(chr_lens, on="CHR", how="left")
        .sort(["CHR", "BP"])
        .with_columns((pl.col("BP") + pl.col("tot")).alias("BPcum"))
    )

    # Axis tick positions (center of each chromosome)
    axisdf = (
        df.group_by("CHR")
        .agg(((pl.col("BPcum").max() + pl.col("BPcum").min()) / 2).alias("center"))
        .sort("CHR")
    )

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    chromosomes = df["CHR"].unique().sort().to_list()
    colors = ["black", "0.8"]
    for i, chrom in enumerate(chromosomes):
        sub = df.filter(pl.col("CHR") == chrom)
        p_vals = sub["P"].to_numpy()
        y_vals = -np.log10(p_vals) if log_transform else p_vals
        ax.scatter(
            sub["BPcum"].to_numpy(),
            y_vals,
            color=colors[i % 2],
            alpha=0.8,
            s=8,
            linewidths=0,
        )

    # Threshold lines
    if threshold_lines is None:
        # CNN defaults
        ax.axhline(
            y=3,
            color="black",
            linestyle="-",
            linewidth=1.2,
            label=r"$p_{sweep} > 0.999$",
        )
        ax.axhline(
            y=2,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=r"$p_{sweep} > 0.99$",
        )
    else:
        for y_val, ls, label in threshold_lines:
            ax.axhline(y_val, color="black", linestyle=ls, linewidth=1.0, label=label)

    # Axis formatting
    ax.set_xticks(axisdf["center"].to_list())
    ax.set_xticklabels([str(int(c)) for c in axisdf["CHR"].to_list()], fontsize=7)
    if log_transform:
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_ylim(0, 10)
        ax.set_ylabel(
            r"$-\log_{10}(1 - p_{sweep})$"
            if p_col is None
            else rf"$-\log_{{10}}({p_col})$"
        )
    else:
        ax.set_ylabel(p_col or "stat")
    ax.set_xlabel("")
    if title:
        ax.set_title(title)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.grid(axis="y", color="lightgray", linewidth=0.5)
    ax.grid(axis="x", visible=False)

    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_sweep_density(prediction, output_path=None):
    """
    Histogram of per-window sweep probability split by chromosome.

    Parameters
    ----------
    prediction : pl.DataFrame | str
        Output of CNN.predict() or path to parquet/CSV with columns
        chr, start, end, prob_sweep.
    output_path : str, optional
        If given, saves the figure as SVG.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if isinstance(prediction, str):
        df = (
            pl.read_parquet(prediction)
            if prediction.endswith(".parquet")
            else pl.read_csv(prediction)
        )
    else:
        df = prediction

    def _chr_key(x):
        s = str(x).replace("chr", "")
        return int(s) if s.isdigit() else ord(s[0])

    chroms = sorted(df["chr"].unique().to_list(), key=_chr_key)
    ncols = 4
    nrows = math.ceil(len(chroms) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes = np.array(axes).flatten()

    for ax, chrom in zip(axes, chroms):
        vals = df.filter(pl.col("chr") == chrom)["prob_sweep"].to_numpy()
        ax.hist(
            vals, bins=30, range=(0, 1), color="steelblue", edgecolor="none", alpha=0.8
        )
        pct = (vals > 0.5).mean() * 100
        ax.set_title(f"{chrom}  ({pct:.1f}% > 0.5)", fontsize=9)
        ax.set_xlabel("P(sweep)", fontsize=8)
        ax.axvline(0.5, color="tomato", linewidth=0.9, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(chroms) :]:
        ax.set_visible(False)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_fv_pca(train_data, empirical_data, subsample=5000, output_path=None):
    """
    PCA of the feature vector matrix colored by neutral/sweep label.

    Parameters
    ----------
    train_data : str | pl.DataFrame
        Path to fvs*.parquet or already-loaded DataFrame. Must have a 'model' column.
    subsample : int
        Max rows to use (avoids slow PCA on very large datasets). Default 5000.
    output_path : str, optional
        If given, saves SVG.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from sklearn.decomposition import PCA

    if isinstance(train_data, str):
        df = pl.read_parquet(train_data)
    else:
        df = train_data

    if df.shape[0] > subsample:
        df = df.sample(n=subsample, seed=42)

    df = df.with_columns(
        pl.when(pl.col("model") != "neutral")
        .then(pl.lit("sweep"))
        .otherwise(pl.lit("neutral"))
        .alias("model")
    )
    meta_cols = {"iter", "s", "t", "f_i", "f_t", "mu", "r", "model"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df.select(feat_cols).fill_null(0).to_numpy()
    labels = df["model"].to_numpy()

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, color in [("neutral", "steelblue"), ("sweep", "tomato")]:
        mask = labels == cls
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            c=color,
            label=cls,
            alpha=0.4,
            s=8,
            edgecolors="none",
        )

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    ax.legend(markerscale=3, frameon=False)
    ax.set_title("Feature vector PCA — neutral vs sweep")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_stat_distributions(
    train_data,
    empirical_data=None,
    stats=[
        "dind",
        "dist_kurtosis",
        "dist_skew",
        "dist_var",
        "h1",
        "h12",
        "h2_h1",
        "haf",
        "hapdaf_o",
        "hapdaf_s",
        "high_freq",
        "ihs",
        "isafe",
        "k_counts",
        "low_freq",
        "max_fda",
        "nsl",
        "omega_max",
        "pi",
        "s_ratio",
        "tajima_d",
        "theta_h",
        "theta_w",
        "zns",
    ],
    output_path=None,
):
    """
    Violin plots of feature stats split by neutral/sweep (and optionally empirical).

    Parameters
    ----------
    train_data : str | pl.DataFrame
        Training fvs*.parquet — must have a 'model' column.
    empirical_data : str | pl.DataFrame, optional
        Empirical fvs*.parquet — no 'model' column; plotted as a third distribution.
    stats : list[str], optional
        Stat base names to plot (e.g. ['pi', 'h12', 'ihs']).
        Default: one representative column per unique stat base name.
    output_path : str, optional
        Save path for SVG.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if isinstance(train_data, str):
        df = pl.read_parquet(train_data)
    else:
        df = train_data

    df = df.with_columns(
        pl.when(pl.col("model") != "neutral")
        .then(pl.lit("sweep"))
        .otherwise(pl.lit("neutral"))
        .alias("model")
    )

    if empirical_data is not None:
        if isinstance(empirical_data, str):
            df_emp = pl.read_parquet(empirical_data)
        else:
            df_emp = empirical_data
    else:
        df_emp = None

    meta_cols = {"iter", "s", "t", "f_i", "f_t", "mu", "r", "model"}
    feat_cols = [c for c in df.columns if c not in meta_cols]

    # Pick one representative column per stat base name
    if stats is None:
        seen = {}
        for c in feat_cols:
            parts = c.rsplit("_", 2)
            base = parts[0] if len(parts) == 3 else c
            if base not in seen:
                seen[base] = c
        plot_cols = list(seen.values())
        stat_labels = list(seen.keys())
    else:
        plot_cols, stat_labels = [], []
        for s in stats:
            col = next((c for c in feat_cols if c.startswith(s + "_")), None)
            if col is not None:
                plot_cols.append(col)
                stat_labels.append(s)

    ncols = 4
    nrows = math.ceil(len(plot_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.array(axes).flatten()

    for ax, col, label in zip(axes, plot_cols, stat_labels):
        data_neutral = (
            df.filter(pl.col("model") == "neutral")[col].drop_nulls().to_numpy()
        )
        data_sweep = df.filter(pl.col("model") == "sweep")[col].drop_nulls().to_numpy()

        datasets = [data_neutral, data_sweep]
        positions = [0, 1]
        face_colors = ["steelblue", "tomato"]

        if df_emp is not None and col in df_emp.columns:
            datasets.append(df_emp[col].drop_nulls().to_numpy())
            positions.append(2)
            face_colors.append("goldenrod")

        parts = ax.violinplot(
            datasets, positions=positions, showmedians=True, widths=0.6
        )
        for body, color in zip(parts["bodies"], face_colors):
            body.set_facecolor(color)
            body.set_alpha(0.7)

        tick_labels = (
            ["neutral", "sweep"]
            if df_emp is None
            else ["neutral", "sweep", "empirical"]
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(plot_cols) :]:
        ax.set_visible(False)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def _load_scan(stat_file: str, stat_col: str) -> pl.DataFrame:
    df = pl.read_csv(stat_file, separator="\t")
    if stat_col not in df.columns:
        raise ValueError(
            f"Column '{stat_col}' not found in {stat_file}. Available: {df.columns}"
        )
    return df


def _resolve_scan_inputs(stats, stat_cols):
    """Normalise flexible scan inputs to (list[pl.DataFrame], list[str]).

    Accepts:
    - ``dict`` from ``scan()`` — keys are stat names, values are DataFrames.
    - ``str`` — path to a single TSV file; ``stat_cols`` must be provided.
    - ``list[str]`` — paths to multiple TSV files; ``stat_cols`` must match length.
    """
    if isinstance(stats, dict):
        if stat_cols is None:
            from .scan import STAT_REGISTRY

            stat_cols = [
                STAT_REGISTRY[s].rank_col if s in STAT_REGISTRY else s
                for s in stats.keys()
            ]
        elif isinstance(stat_cols, str):
            stat_cols = [stat_cols]
        dfs = [stats[s] for s in stats.keys()]
    elif isinstance(stats, str):
        if stat_cols is None:
            raise ValueError("stat_cols required when stats is a file path")
        if isinstance(stat_cols, str):
            stat_cols = [stat_cols]
        dfs = [_load_scan(stats, s) for s in stat_cols]
    elif isinstance(stats, (list, tuple)):
        if isinstance(stat_cols, str):
            stat_cols = [stat_cols] * len(stats)
        elif stat_cols is None:
            raise ValueError("stat_cols required when stats is a list of file paths")
        dfs = [_load_scan(f, s) for f, s in zip(stats, stat_cols)]
    else:
        raise TypeError(
            "stats must be a dict (scan() output), a file path str, or a list of paths"
        )
    return dfs, stat_cols


def _chr_to_float(col_expr):
    """Polars expression: strip 'chr' prefix and cast to Float64 for sorting."""
    return col_expr.cast(pl.Utf8).str.replace("^chr", "").cast(pl.Float64)


def plot_scan(
    stats,
    stat_cols=None,
    pvalue: bool = False,
    top_pct: float = 0.01,
    threshold_lines: list | None = None,
    out: str | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    sharey: bool = False,
    chrom: str | None = None,
    center: int | None = None,
    window_bp: int = 500_000,
) -> plt.Figure:
    """Genome-wide or regional scan plot — single stat or stacked multi-stat.

    Accepts output from ``scan()`` (dict) or paths to ``{prefix}.{stat}.txt`` files.

    **Genome-wide mode** (default, ``chrom=None``):
    Single stat → 1-panel Manhattan. Multiple stats → stacked panels, shared x-axis.
    ``pvalue=False`` plots raw values with bipolar colouring for signed stats.
    ``pvalue=True`` plots ``-log10(p_emp)`` with threshold lines at p = 0.01 / 0.001.

    **Zoom mode** (``chrom`` + ``center`` provided):
    n rows × 2 columns: raw stat (left) | ``-log10(p_emp)`` (right).
    Filtered to ``[center ± window_bp]``. Orange dashed line marks the centre.

    Parameters
    ----------
    stats : dict | str | list[str]
        - ``dict`` from ``scan()`` — keys are stat names, values are DataFrames.
        - ``str`` — path to a single ``{stat}.txt`` scan output file.
        - ``list[str]`` — paths to multiple scan output files.
    stat_cols : str | list[str], optional
        Stat column name(s). Required when ``stats`` is a file path / list of paths.
        When ``stats`` is a dict, defaults to all keys.
    pvalue : bool
        Genome-wide mode only. If True, plot ``-log10(p_emp)`` with y=[0,10] and
        threshold lines. If False, plot raw stat values with bipolar colouring.
    top_pct : float
        Fraction highlighted as outliers in genome-wide raw mode.
    threshold_lines : list of (y_value, linestyle, label), optional
        Horizontal lines. ``pvalue=True`` default: y=2 dashed (p=0.01), y=3 solid.
        Pass ``[]`` to suppress.
    out : str, optional
        Save path. If None, shows interactively.
    figsize : tuple, optional
        Genome-wide default: (14, 4) / (14, 3×n). Zoom default: (10, 2.5×n).
    title : str, optional
        Title for single-stat genome-wide plots.
    sharey : bool
        Share y-axis across panels in genome-wide mode (default False).
    chrom : str, optional
        Chromosome for zoom mode (e.g. ``"22"`` or ``"chr22"``).
    center : int, optional
        Centre position (bp) for zoom mode.
    window_bp : int
        Half-window size in bp for zoom mode (default 500 000 = ±500 kb).
    """
    dfs, stat_cols = _resolve_scan_inputs(stats, stat_cols)
    n = len(dfs)

    # ── Zoom mode ────────────────────────────────────────────────────────────
    if chrom is not None and center is not None:
        lo, hi = center - window_bp, center + window_bp
        chrom_str = str(chrom).lstrip("chr")
        if figsize is None:
            figsize = (10, 4) if n == 1 else (10, 3.0 * n)
        fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, sharey=sharey)
        axes = [axes] if n == 1 else list(axes)
        for ax, df, stat_col in zip(axes, dfs, stat_cols):
            pval_col = f"{stat_col}_pvalue"
            df = df.filter(
                pl.col("chrom").cast(pl.Utf8).str.replace("^chr", "") == chrom_str
            ).filter((pl.col("pos") >= lo) & (pl.col("pos") <= hi))
            pos = df["pos"].to_numpy()

            if pvalue and pval_col in df.columns:
                p_raw = df[pval_col].to_numpy(allow_copy=True).astype(np.float64)
                y = -np.log10(np.clip(p_raw, 1e-10, None))
                y_label = (
                    rf"$\mathrm{{{stat_col}}}:\ -\log_{{10}}(p_{{\mathrm{{emp}}}})$"
                )
                ax.scatter(
                    pos,
                    y,
                    s=4,
                    color="#333333",
                    alpha=0.7,
                    linewidths=0,
                    rasterized=True,
                )
                lines = (
                    threshold_lines
                    if threshold_lines is not None
                    else [
                        (2, "--", "p = 0.01"),
                        (3, "-", "p = 0.001"),
                    ]
                )
                for y_val, ls, label in lines:
                    ax.axhline(
                        y_val, color="black", linestyle=ls, linewidth=1.0, label=label
                    )
                ax.set_ylim(0, 10)
                ax.set_yticks([2, 4, 6, 8, 10])
            else:
                y = df[stat_col].to_numpy(allow_copy=True).astype(np.float64)
                y_label = stat_col
                has_neg = np.nanmin(y) < 0
                if has_neg:
                    abs_thresh = np.nanpercentile(np.abs(y), (1 - top_pct) * 100)
                    pos_out = y >= abs_thresh
                    neg_out = y <= -abs_thresh
                    background = ~pos_out & ~neg_out
                else:
                    thresh = np.nanpercentile(y, (1 - top_pct) * 100)
                    pos_out = y >= thresh
                    neg_out = np.zeros(len(y), dtype=bool)
                    background = ~pos_out
                ax.scatter(
                    pos[background],
                    y[background],
                    s=3,
                    color="#333333",
                    alpha=0.4,
                    linewidths=0,
                    rasterized=True,
                )
                if pos_out.any():
                    ax.scatter(
                        pos[pos_out],
                        y[pos_out],
                        s=8,
                        color="#d62728",
                        alpha=0.8,
                        linewidths=0,
                        rasterized=True,
                        label=f"Top {top_pct * 100:g}%{' (+)' if has_neg else ''}",
                    )
                if has_neg and neg_out.any():
                    ax.scatter(
                        pos[neg_out],
                        y[neg_out],
                        s=8,
                        color="#1f77b4",
                        alpha=0.8,
                        linewidths=0,
                        rasterized=True,
                        label=f"Top {top_pct * 100:g}% (−)",
                    )
                if threshold_lines:
                    for y_val, ls, label in threshold_lines:
                        ax.axhline(
                            y_val,
                            color="black",
                            linestyle=ls,
                            linewidth=1.0,
                            label=label,
                        )

            ax.axvline(center, color="black", lw=1, ls="--", alpha=1)
            ax.set_ylabel(y_label, fontsize=8)
            ax.legend(fontsize=8, frameon=False, markerscale=3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", color="lightgray", linewidth=0.5)
            ax.grid(axis="x", visible=False)

        if n == 1 and title:
            axes[0].set_title(title)
        axes[-1].set_xlabel("Position")
        axes[-1].xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.2f} Mb")
        )
        fig.tight_layout()
        if out:
            fig.savefig(out, dpi=150)
            plt.close(fig)
        else:
            plt.show()
        return fig

    # ── Genome-wide mode ─────────────────────────────────────────────────────
    if figsize is None:
        figsize = (14, 4) if n == 1 else (14, 3.0 * n)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, sharey=sharey)
    axes = [axes] if n == 1 else list(axes)

    # Chr offsets: computed once from the first DataFrame
    df0 = dfs[0].with_columns(
        [
            _chr_to_float(pl.col("chrom")).alias("CHR"),
            pl.col("pos").cast(pl.Float64).alias("BP"),
        ]
    )
    chr_lens = (
        df0.group_by("CHR")
        .agg(pl.col("BP").max().alias("chr_len"))
        .sort("CHR")
        .with_columns((pl.col("chr_len").cum_sum() - pl.col("chr_len")).alias("tot"))
        .select(["CHR", "tot"])
    )
    axisdf = None
    alt_colors = ["#333333", "#aaaaaa"]

    for ax, df, stat_col in zip(axes, dfs, stat_cols):
        pval_col = f"{stat_col}_pvalue"

        df_plot = (
            df.with_columns(
                [
                    _chr_to_float(pl.col("chrom")).alias("CHR"),
                    pl.col("pos").cast(pl.Float64).alias("BP"),
                ]
            )
            .join(chr_lens, on="CHR", how="left")
            .sort(["CHR", "BP"])
            .with_columns((pl.col("BP") + pl.col("tot")).alias("BPcum"))
        )
        if axisdf is None:
            axisdf = (
                df_plot.group_by("CHR")
                .agg(
                    ((pl.col("BPcum").max() + pl.col("BPcum").min()) / 2).alias(
                        "center"
                    )
                )
                .sort("CHR")
            )

        bpcum = df_plot["BPcum"].to_numpy()
        chr_vals = df_plot["CHR"].to_numpy()

        if pvalue and pval_col in df_plot.columns:
            p_raw = df_plot[pval_col].to_numpy(allow_copy=True).astype(np.float64)
            y = -np.log10(np.clip(p_raw, 1e-10, None))
            y_label = rf"$\mathrm{{{stat_col}}}:\ -\log_{{10}}(p_{{\mathrm{{emp}}}})$"

            chromosomes = sorted(df_plot["CHR"].unique().to_list())
            for i, chrom in enumerate(chromosomes):
                mask = chr_vals == chrom
                ax.scatter(
                    bpcum[mask],
                    y[mask],
                    s=4,
                    color=alt_colors[i % 2],
                    alpha=0.7,
                    linewidths=0,
                    rasterized=True,
                )

            lines = (
                threshold_lines
                if threshold_lines is not None
                else [
                    (2, "--", "p = 0.01"),
                    (3, "-", "p = 0.001"),
                ]
            )
            for y_val, ls, label in lines:
                ax.axhline(
                    y_val, color="black", linestyle=ls, linewidth=1.0, label=label
                )

            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])

        else:
            y = df_plot[stat_col].to_numpy(allow_copy=True).astype(np.float64)
            y_label = stat_col
            has_neg = np.nanmin(y) < 0

            if has_neg:
                abs_thresh = np.nanpercentile(np.abs(y), (1 - top_pct) * 100)
                pos_out = y >= abs_thresh
                neg_out = y <= -abs_thresh
                background = ~pos_out & ~neg_out
            else:
                thresh = np.nanpercentile(y, (1 - top_pct) * 100)
                pos_out = y >= thresh
                neg_out = np.zeros(len(y), dtype=bool)
                background = ~pos_out

            chromosomes = sorted(df_plot["CHR"].unique().to_list())
            for i, chrom in enumerate(chromosomes):
                mask = (chr_vals == chrom) & background
                ax.scatter(
                    bpcum[mask],
                    y[mask],
                    s=2,
                    color=alt_colors[i % 2],
                    alpha=0.4,
                    linewidths=0,
                    rasterized=True,
                )

            if pos_out.any():
                ax.scatter(
                    bpcum[pos_out],
                    y[pos_out],
                    s=6,
                    color="#d62728",
                    alpha=0.8,
                    linewidths=0,
                    rasterized=True,
                    label=f"Top {top_pct * 100:g}%{' (+)' if has_neg else ''}",
                )
            if has_neg and neg_out.any():
                ax.scatter(
                    bpcum[neg_out],
                    y[neg_out],
                    s=6,
                    color="#1f77b4",
                    alpha=0.8,
                    linewidths=0,
                    rasterized=True,
                    label=f"Top {top_pct * 100:g}% (−)",
                )

            if threshold_lines:
                for y_val, ls, label in threshold_lines:
                    ax.axhline(
                        y_val, color="black", linestyle=ls, linewidth=1.0, label=label
                    )

        ax.set_ylabel(y_label, fontsize=8)
        ax.legend(fontsize=8, frameon=False, markerscale=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="lightgray", linewidth=0.5)
        ax.grid(axis="x", visible=False)

    if n == 1 and title:
        axes[0].set_title(title)

    axes[-1].set_xticks(axisdf["center"].to_list())
    axes[-1].set_xticklabels([str(int(c)) for c in axisdf["CHR"].to_list()], fontsize=7)
    axes[-1].set_xlabel("Chromosome")

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_scan_zoom(
    stats, stat_cols, chrom, center, window_bp=500_000, **kwargs
) -> plt.Figure:
    """Deprecated alias — use ``plot_scan(..., chrom=chrom, center=center)``."""
    return plot_scan(
        stats,
        stat_cols=stat_cols,
        chrom=chrom,
        center=center,
        window_bp=window_bp,
        **kwargs,
    )


################## Sorting methods


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


def haplotype_freq_sorting_hamming(matrix):
    # Based on scikit haplotype counts

    # setup collection
    d = defaultdict(list)

    S, n = matrix.shape
    # iterate over haplotypes
    for i in range(n):
        # hash the haplotype
        k = hash(matrix[:, i].tobytes())

        # collect
        d[k].append(i)

    # extract sets, sorted by most common
    counts = sorted(d.values(), key=len, reverse=True)
    f = np.array([len(g) / n for g in counts], dtype=float)

    # Representative column index for each group (you said groups are equal, so first is fine)
    reps = np.array([g[0] for g in counts], dtype=int)

    # Choose reference haplotype
    ref = matrix[:, reps[np.argmax(f)]]

    # Compute Hamming distance of each group's representative to ref
    # Vectorized across all representatives
    reps_mat = matrix[:, reps]  # shape (S, k)
    distances = (reps_mat != ref[:, None]).sum(0)  # shape (k,)

    # Sort groups by (-frequency, distance, representative index as a stable tie-breaker)
    # np.lexsort uses last key as primary, so order keys accordingly
    # primary: -f, then distance, then reps
    group_order = np.lexsort((reps, distances, -f))

    # final column order by concatenating columns from each group in the sorted group order
    col_order = np.concatenate([np.asarray(counts[g], dtype=int) for g in group_order])
    matrix_reordered = matrix[:, col_order]

    return matrix_reordered, f


def haplotype_freq_sorting(matrix):
    """
    Reorder matrix columns by haplotype frequency (descending), grouping columns
    exactly as in the hashing/defaultdict approach.

    Returns
    -------
    matrix_reordered : (S, n) ndarray
    col_order : (n,) ndarray of int
    groups_sorted : list[list[int]]
    f : (k,) ndarray of float
    """
    matrix = np.asarray(matrix)
    S, n = matrix.shape

    # --- same grouping logic as in haplotype_freq_sorting_hamming ---
    d = defaultdict(list)
    for i in range(n):
        k = hash(matrix[:, i].tobytes())
        d[k].append(i)

    # groups sorted by most common (stable for ties)
    groups_sorted = sorted(d.values(), key=len, reverse=True)

    # frequencies identical to the first function
    f = np.array([len(g) / n for g in groups_sorted], dtype=float)

    # column permutation and reordered matrix
    if groups_sorted:
        col_order = np.concatenate([np.asarray(g, dtype=int) for g in groups_sorted])
    else:
        col_order = np.array([], dtype=int)

    matrix_reordered = matrix[:, col_order] if n > 0 else matrix

    return matrix_reordered, col_order, groups_sorted, f


def disrupt_genomic_positions(matrix):
    """
    Randomly permute (shuffle) the columns of `matrix`.

    Parameters
    ----------
    matrix : array-like
        2D data whose columns will be permuted.

    Returns
    -------
    permuted : np.ndarray
        Matrix with columns permuted.
    """
    arr = np.array(matrix, copy=True)

    if arr.ndim != 2:
        raise ValueError("`matrix` must be 2D.")

    rng = np.random.default_rng()

    n_cols = arr.shape[1]
    perm = rng.permutation(n_cols)
    permuted = arr[:, perm]

    return permuted


def disrupt_ld(matrix):
    arr = matrix.T
    np.random.shuffle(arr)
    for j in range(arr.shape[0]):
        # shuffle within each rows, which are snp columns
        np.random.shuffle(arr[j])
    return arr


def disrupt_af(matrix):
    rows, cols = matrix.shape
    arr = matrix.copy().reshape(rows * cols)
    np.random.shuffle(arr)
    # reshape back to 2D
    arr = arr.reshape(rows, cols)
    # plot
    return arr


def mediant_af(matrix):
    rows, cols = matrix.shape
    num_ones = np.count_nonzero(matrix.T)
    # make new 1D arr with the counts of two entry groups: zeros and ones
    # zeros = # total entries - ones
    arr = np.concatenate([np.zeros(rows * cols - num_ones), np.ones(num_ones)])
    arr = arr.reshape((cols, rows)).T

    return arr


def mediant_af_left(matrix):
    rows, cols = matrix.T.shape

    num_ones = np.count_nonzero(matrix.T)
    arr = np.concatenate([np.zeros(rows * cols - num_ones), np.ones(num_ones)])
    arr = arr.reshape((cols, rows)).T
    return arr


################## Ranking


def merge_regions(prediction, p):
    """
    Merge genomic regions where prob_sweep > p

    Parameters:
    -----------
    prediction : str or pl.DataFrame
        File path to CSV or Polars DataFrame
    p : float
        Probability threshold for filtering

    Returns:
    --------
    tuple: (df_merged, summary_stats)
        - df_merged: LazyFrame with merged intervals
        - summary_stats: DataFrame with chr, merged_span, total_span, pct
    """
    # Load or clone the data
    if isinstance(prediction, str):
        df_pred = (
            pl.read_csv(prediction, has_header=True, separator=",")
            .select("chr", "start", "end", "prob_sweep")
            .filter(
                pl.col("chr")
                .str.replace("chr", "")
                .is_in([str(i) for i in range(1, 23)])
            )
            .with_columns(
                (pl.lit("chr") + pl.col("chr").str.replace("chr", "")).alias("chr")
            )
            .sort(["chr", "start"])
        )
    elif isinstance(prediction, pl.DataFrame):
        df_pred = (
            prediction.clone()
            .select("chr", "start", "end", "prob_sweep")
            .sort(["chr", "start"])
        )
    else:
        raise ValueError("prediction must be a file path (str) or a Polars DataFrame")

    # Calculate total genomic span analyzed per chromosome
    total_span_analyzed = (
        df_pred.group_by("chr")
        .agg(
            [
                pl.col("start").min().alias("min_start"),
                pl.col("end").max().alias("max_end"),
            ]
        )
        .with_columns((pl.col("max_end") - pl.col("min_start")).alias("total_span"))
        .select(["chr", "total_span"])
    )

    # Filter for prob_sweep > p
    filtered = df_pred.filter(pl.col("prob_sweep") > p)

    # Merge consecutive/overlapping windows
    df_merged = (
        merge(
            filtered,
            min_dist=0,
            cols=["chr", "start", "end"],
            on_cols=None,
            output_type="polars.LazyFrame",
            projection_pushdown=True,
        )
        .with_columns((pl.col("end") - pl.col("start")).alias("d"))
        .collect()
    )

    # Calculate merged span per chromosome
    merged_span = df_merged.group_by("chr").agg(pl.col("d").sum().alias("merged_span"))

    # Join and calculate percentage
    summary_stats = (
        total_span_analyzed.join(merged_span, on="chr", how="left")
        .with_columns(
            [
                pl.col("merged_span").fill_null(0),
                (pl.col("merged_span") / pl.col("total_span") * 100).alias("pct"),
            ]
        )
        .sort("chr")
    )

    return df_merged, summary_stats


def rank_probabilities(prediction, feature_coordinates, rank_distance=False, k=111):
    """
    Goal: match the original pybedtools/bedtools output *exactly*, including tie order.

    Strategy for exact tie-order:
      - Create an explicit, deterministic per-gene input order (`gene_order`) from the
        sorted gene table (chr,start). This mirrors the order bedtools processes A.
      - Carry `gene_order` through the pipeline and use it as the final tie-breaker
        in the final sort (stable + deterministic).

    Strategy for bedtools `closest -k` parity:
      - We emulate `-k K` by sorting hits per gene by (d, chrom_pred, start_pred, end_pred)
        and taking the first K.
      - This gives deterministic selection when multiple windows share the same distance.
    """

    def _to_point_1based_closed(df: pl.DataFrame, chr_col: str) -> pl.DataFrame:
        # midpoint -> 1-based coordinate; represent as point interval [pos1, pos1]
        return (
            df.with_columns(
                (((pl.col("start") + pl.col("end")) // 2) + 1).alias("_pos1")
            )
            .with_columns(pl.col("_pos1").alias("start"), pl.col("_pos1").alias("end"))
            .drop("_pos1")
            .rename({chr_col: "chrom"})
        )

    if isinstance(feature_coordinates, str):
        df_genes = (
            pl.read_csv(
                feature_coordinates,
                has_header=False,
                separator="\t",
                schema={
                    "chr": pl.Utf8,
                    "start": pl.Int64,
                    "end": pl.Int64,
                    "gene_id": pl.Utf8,
                    "strand": pl.Utf8,
                },
            )
            .select("chr", "start", "end", "strand", "gene_id")
            .filter(pl.col("chr").is_in([str(i) for i in range(1, 23)]))
            .with_columns((pl.lit("chr") + pl.col("chr")).alias("chr"))
            .sort(["chr", "start"])
            # explicit deterministic order to break full ties exactly like bedtools A-stream
            .with_row_index("gene_order", offset=0)
        )
    elif isinstance(feature_coordinates, pl.DataFrame):
        df_genes = feature_coordinates.clone()
        if "gene_id" not in df_genes.columns and "feature_id" in df_genes.columns:
            df_genes = df_genes.rename({"feature_id": "gene_id"})
        if "gene_order" not in df_genes.columns:
            # ensure deterministic gene order if caller didn't provide it
            df_genes = df_genes.sort(["chr", "start"]).with_row_index(
                "gene_order", offset=0
            )
    else:
        raise ValueError(
            "feature_coordinates must be a file path (str) or a Polars DataFrame"
        )

    if isinstance(prediction, str):
        df_pred = (
            pl.read_csv(prediction, has_header=True, separator=",")  # adjust if TSV
            .select("chr", "start", "end", "prob_sweep")
            .filter(
                pl.col("chr")
                .str.replace("chr", "")
                .is_in([str(i) for i in range(1, 23)])
            )
            .with_columns(
                (pl.lit("chr") + pl.col("chr").str.replace("chr", "")).alias("chr")
            )
            .sort(["chr", "start"])
        )
    elif isinstance(prediction, pl.DataFrame):
        df_pred = (
            prediction.clone()
            .select("chr", "start", "end", "prob_sweep")
            .sort(["chr", "start"])
        )
    else:
        raise ValueError("prediction must be a file path (str) or a Polars DataFrame")

    genes = _to_point_1based_closed(df_genes.rename({"chr": "chr"}), chr_col="chr")
    preds = _to_point_1based_closed(df_pred.rename({"chr": "chr"}), chr_col="chr")

    genes_lf = genes.lazy()
    preds_lf = preds.lazy()

    nearest_raw = nearest(
        genes_lf,
        preds_lf,
        suffixes=("_gene", "_pred"),
        cols1=["chrom", "start", "end"],
        cols2=["chrom", "start", "end"],
        output_type="polars.LazyFrame",
    )

    nearest_raw = nearest_raw.rename(
        {
            "gene_id_gene": "gene_id",
            "strand_gene": "strand",
            "gene_order_gene": "gene_order",
            "chrom_gene": "chrom_gene",
            "start_gene": "start_gene",
            "end_gene": "end_gene",
            "distance": "d_min",
        }
    ).select(
        "gene_id",
        "strand",
        "gene_order",
        "chrom_gene",
        "start_gene",
        "end_gene",
        "d_min",
    )

    gene_windows = (
        nearest_raw.with_columns(
            (pl.col("d_min") + 500_000).alias("_radius"),
            pl.col("start_gene").alias("gene_pos"),
        )
        .with_columns(
            (pl.col("gene_pos") - pl.col("_radius")).clip(lower_bound=1).alias("start"),
            (pl.col("gene_pos") + pl.col("_radius")).alias("end"),
            pl.col("chrom_gene").alias("chrom"),
        )
        .select(
            "chrom",
            "start",
            "end",
            "gene_id",
            "strand",
            "gene_order",
            "gene_pos",
            "d_min",
        )
    )

    hits_raw = overlap(
        gene_windows,
        preds_lf,
        suffixes=("_win", "_pred"),
        cols1=["chrom", "start", "end"],
        cols2=["chrom", "start", "end"],
        output_type="polars.LazyFrame",
    )

    hits = hits_raw.rename(
        {
            "gene_id_win": "gene_id",
            "strand_win": "strand",
            "gene_order_win": "gene_order",
            "gene_pos_win": "gene_pos",
            "d_min_win": "d_min",
            "chrom_win": "chrom",
            "start_win": "start",
            "end_win": "end",
        }
    ).with_columns((pl.col("gene_pos") - pl.col("start_pred")).abs().alias("d"))

    # k elements like in bedtools
    if k is not None:
        k = int(k)
        hits = (
            hits.sort(["gene_id", "d", "chrom_pred", "start_pred", "end_pred"])
            .with_columns(pl.int_range(0, pl.len()).over("gene_id").alias("_k"))
            .filter(pl.col("_k") < k)
            .drop("_k")
        )

    if rank_distance:
        # Distance-aware composite: sort by prob_sweep sum desc, d_sum asc,
        # and finally gene_order asc to match original tie order.
        w_rank = (
            hits.filter((pl.col("d") - pl.col("d_min")).abs() <= 500_000)
            .group_by("gene_id")
            .agg(
                pl.col("prob_sweep_pred").sum().alias("prob_sweep"),
                pl.col("d").sum().alias("d_sum"),
                pl.col("gene_order").min().alias("gene_order"),
            )
            .sort(
                ["prob_sweep", "d_sum", "gene_order"], descending=[True, False, False]
            )
            .with_row_index("rank", offset=1)
            .select("gene_id", "rank", "prob_sweep")
        ).collect()

        n_rank_max = (
            w_rank.filter(pl.col("prob_sweep") == w_rank["prob_sweep"].max()).height
            if w_rank.height
            else 0
        )
        return w_rank, int(n_rank_max)

    rank_unique = (
        hits.filter(pl.col("d") == pl.col("d_min"))
        .group_by("gene_id")
        .agg(
            pl.col("prob_sweep_pred").max().alias("prob_sweep"),
            pl.col("gene_order").min().alias("gene_order"),
        )
    )

    rank_rep = (
        hits.with_columns((pl.col("d") - pl.col("d_min")).abs().alias("_abs"))
        .filter(pl.col("_abs") <= 500_000)
        .group_by("gene_id")
        .agg(
            pl.col("prob_sweep_pred").sum().alias("win_prob_sum"),
            pl.col("d").sum().alias("win_d_sum"),
        )
    )

    ranked = (
        rank_unique.join(rank_rep, on="gene_id", how="left")
        .with_columns(
            pl.col("win_prob_sum").fill_null(0),
            pl.col("win_d_sum").fill_null(0),
        )
        # FINAL KEY: gene_order ensures exact tie ordering consistent with the original A-stream
        .sort(
            ["prob_sweep", "win_prob_sum", "win_d_sum", "gene_order"],
            descending=[True, True, False, False],
        )
        .with_row_index("rank", offset=1)
        .select("gene_id", "rank", "prob_sweep")
    ).collect()

    n_rank_max = (
        ranked.filter(pl.col("prob_sweep") == ranked["prob_sweep"].max()).height
        if ranked.height
        else 0
    )
    return ranked, int(n_rank_max)


def interpolate_rates(
    prediction, recombination_map, prediction_lr=None, corr=False, bins=10
):
    """
    Interpolate recombination rates (cM/Mb) onto prediction windows and optionally
    replace low-recombination probabilities using an alternate predictions file.

    Given a prediction output file and a recombination map, this function computes
    recombination rates for each unique prediction window per chromosome (via
    ``get_cm(..., cm_mb=True)``) and joins the resulting ``cm_mb`` column back into
    the prediction table using the join keys ``chr``, ``start``, and ``end``.

    If ``prediction_lr`` is provided, the function performs an additional join for
    that alternate prediction table and selectively replaces ``prob_sweep`` and
    ``prob_neutral`` in the main predictions for windows with ``cm_mb < 0.5`` using
    the corresponding values from ``prediction_lr``.

    If ``corr=True``, the function computes correlations between ``prob_sweep`` and
    ``cm_mb`` per chromosome, bins windows by ``cm_mb`` within each chromosome
    (via ``_bin_group``), plots the binned trend per chromosome, and returns both
    the augmented predictions and the per-chromosome binned trend table.

    Input expectations:
      - The **prediction** file must contain at least ``chr``, ``start``, and ``end``,
        and typically contains ``prob_sweep`` / ``prob_neutral`` if replacement or
        correlation is used. Additional columns are preserved.
      - The **recombination_map** must be a tab-separated file (comment lines allowed,
        prefixed with ``#``) with columns ``chr``, ``start``, ``end``, ``cm_mb``, ``cm``.
      - The **prediction_lr** file (if provided) must contain ``chr``, ``start``, ``end``,
        ``prob_sweep``, and ``prob_neutral``.

    :param str prediction: Path to a CSV file containing prediction windows and
        associated probabilities. Must include ``chr``, ``start``, ``end``.
    :param str recombination_map: Path to a tab-separated recombination map with
        columns ``chr``, ``start``, ``end``, ``cm_mb``, ``cm``; may include ``#``
        comment lines.
    :param str | None prediction_lr: Optional path to an alternate predictions CSV.
        If provided, ``prob_sweep`` and ``prob_neutral`` are replaced for windows
        where ``cm_mb < 0.5`` using values from this file.
    :param bool corr: If ``True``, compute and plot per-chromosome binned trends
        and return ``(df_pred_rate, df_trend)``; if ``False`` (default), return only
        the augmented predictions.
    :returns: If ``corr=False``, returns **df_pred_rate**, a DataFrame containing
        all original prediction columns plus ``cm_mb``. If ``corr=True``, returns a
        pair ``(df_pred_rate, df_trend)``, where **df_trend** is a per-chromosome
        binned table suitable for plotting (sorted by ``chr`` and ``bin``).
    :rtype: polars.DataFrame | tuple[polars.DataFrame, polars.DataFrame]

    :notes:
      - Recombination rate interpolation is delegated to ``get_cm`` and is performed
        independently per chromosome. ``get_cm`` is assumed to return a Polars
        DataFrame that includes ``chr``, ``start``, ``end``, and ``cm_mb``.
      - The join key for all merges is ``(chr, start, end)``. Missing keys will
        result in null ``cm_mb`` (and will prevent probability replacement for
        those rows).
      - When ``prediction_lr`` is provided, only windows with ``cm_mb < 0.5`` from
        the LR-joined table are eligible for overwriting probabilities in the base
        predictions.
      - When ``corr=True``, this function produces matplotlib output (calls
        ``plt.show()``) as a side effect.
    """

    def _chr_key(s):
        s = str(s)
        m = re.match(r"^chr(\d+)$", s)
        if m:
            return (0, int(m.group(1)))
        m = re.match(r"^chr([XYM]|MT)$", s)
        if m:
            order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
            return (1, order[m.group(1)])
        return (2, s)

    def _bin_group(g, n_bins=bins):
        g = g.sort("cm_mb")
        n = g.height

        g = g.with_columns(pl.arange(0, n).alias("i"))
        g = g.with_columns(
            ((pl.col("i") * n_bins) / n).floor().cast(pl.Int64).alias("bin")
        )

        out = (
            g.group_by("bin")
            .agg(
                pl.mean("cm_mb").alias("cm_mb"),
                pl.mean("prob_sweep").alias("prob_sweep"),
                pl.len().alias("n_in_bin"),
                pl.min("cm_mb").alias("cm_mb_min"),
                pl.max("cm_mb").alias("cm_mb_max"),
            )
            .sort("cm_mb_min")
            .with_columns(
                pl.lit(g["chr"][0]).alias("chr"),
                # optional: readable range label
                pl.format(
                    "[{}, {}]",
                    pl.col("cm_mb_min").round(4),
                    pl.col("cm_mb_max").round(4),
                ).alias("cm_mb_range"),
            )
        )
        return out

    if isinstance(prediction, pl.DataFrame):
        df_pred = prediction
    else:
        df_pred = pl.read_csv(prediction)

    df_rec = pl.read_csv(
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
    ).sort(["chr", "start"])

    # Compute recombination rates per chr
    rate_frames = [
        get_cm(
            df_rec.filter(pl.col("chr") == chr_),  # assumes get_cm returns a Polars df
            df_pred.filter(pl.col("chr") == chr_)
            .select("start", "end")
            .unique()
            .to_numpy(),
            cm_mb=True,
        )
        for chr_ in df_pred["chr"].unique()
    ]

    df_rate = (
        pl.concat(rate_frames)
        .with_columns(pl.col("chr").str.replace("chr", "").cast(pl.Int32).alias("nchr"))
        .sort(["nchr", "start"])
        .select(["chr", "start", "end", "cm_mb"])
    )

    # Merge recombination rate into both predictions
    df_pred_rate = df_pred.join(df_rate, on=["chr", "start", "end"])

    if prediction_lr is not None:
        df_pred_lr = pl.read_csv(prediction_lr)

        df_pred_rate_lr = df_pred_lr.join(df_rate, on=["chr", "start", "end"])

        # Replace probabilities where cm_mb < 0.5
        df_lr_filtered = df_pred_rate_lr.filter(pl.col("cm_mb") < 0.5).select(
            ["chr", "start", "end", "prob_sweep", "prob_neutral"]
        )

        df_pred_rate = (
            df_pred_rate.join(df_lr_filtered, on=["chr", "start", "end"], how="left")
            .with_columns(
                prob_sweep=pl.when(pl.col("prob_sweep_right").is_not_null())
                .then(pl.col("prob_sweep_right"))
                .otherwise(pl.col("prob_sweep")),
                prob_neutral=pl.when(pl.col("prob_neutral_right").is_not_null())
                .then(pl.col("prob_neutral_right"))
                .otherwise(pl.col("prob_neutral")),
            )
            .select(df_pred.columns + ["cm_mb"])  # restore original column order
        )

    else:
        df_pred_rate = df_pred_rate.select(df_pred.columns + ["cm_mb"])

    if corr:
        df_corr = df_pred_rate.group_by("chr").agg(
            pl.len().alias("n"),
            pl.corr("prob_sweep", "cm_mb").alias("corr"),
        )

        df_bins = df_pred_rate.group_by("chr").map_groups(_bin_group)

        df_corr_binned = df_bins.group_by("chr").agg(
            pl.corr("prob_sweep", "cm_mb").alias("corr"), pl.len().alias("n")
        )

        df_trend = df_corr_binned.join(df_bins, on="chr", how="left")

        chrs = sorted(df_trend["chr"].unique().to_list(), key=_chr_key)
        ncols = 4
        nrows = math.ceil(len(chrs) / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 3.0 * nrows), sharey=True
        )
        axes = np.array(axes).reshape(-1)

        for ax, c in zip(axes, chrs):
            t = df_trend.filter(pl.col("chr") == c)
            x = t.get_column("cm_mb").to_numpy()
            y = t.get_column("prob_sweep").to_numpy()
            ax.plot(x, y, linewidth=1.2, color="#2166ac")
            r = t.get_column("corr")[0]
            r_str = f"r={r:.2f}" if r is not None else "r=NA"
            ax.set_title(f"{c}  ({r_str})", fontsize=8, pad=3)
            ax.axhline(0.5, linewidth=0.8, linestyle="--", color="gray", alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_xlabel("cM/Mb", fontsize=8, labelpad=2)
            ax.set_ylabel(
                "P(sweep)" if ax.get_subplotspec().is_first_col() else "", fontsize=8
            )
            ax.tick_params(labelsize=7)
            ax.tick_params(axis="x", pad=2)
            ax.spines[["top", "right"]].set_visible(False)

        for ax in axes[len(chrs) :]:
            ax.axis("off")

        plt.tight_layout(h_pad=3.5, w_pad=1.5)  # ← h_pad is the main lever here
        # plt.savefig("fig1.pdf", dpi=300, bbox_inches="tight")
        plt.show()

        return (
            df_pred_rate,
            df_trend.sort("chr", "bin").select(pl.exclude("n", "n_in_bin")),
            df_corr,
        )

    else:
        return df_pred_rate
