"""scan-plot: reads scan() output columns, highlights both tails of signed stats,
splits chromosomes on request, and writes SVG by default."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
from click.testing import CliRunner

from flexsweep.main import cli
from flexsweep.utils import _scan_track, plot_scan


def _scan_tsv(path, chroms=("chr22",), legacy=False, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    df = pl.concat(
        pl.DataFrame(
            {
                "chr": [c] * n,
                "positions": np.sort(rng.integers(20_000_000, 25_000_000, n)),
                "ihs": rng.normal(-1.0, 1.0, n),
            }
        )
        for c in chroms
    ).with_columns(pl.col("ihs").rank().truediv(n * len(chroms)).alias("ihs_pvalue"))
    if legacy:
        df = df.rename({"chr": "chrom", "positions": "pos"})
    df.write_csv(path, separator="\t")
    return df


def test_signed_stat_highlights_both_tails(tmp_path):
    df = _scan_tsv(tmp_path / "x.ihs.txt")
    # Skewed negative: an abs-value threshold would flag only the lower tail.
    track, _, signed = _scan_track(df, "ihs", pvalue=False, top_pct=0.01)
    assert signed
    n = track.height
    n_hi = int((track["_cls"] == 1).sum())
    n_lo = int((track["_cls"] == -1).sum())
    assert abs(n_hi - 0.01 * n) <= 2 and abs(n_lo - 0.01 * n) <= 2


def test_all_negative_stat_gets_lower_tail(tmp_path):
    # Fay-Wu H can be negative in every window; the sweep signal is the lower tail.
    df = _scan_tsv(tmp_path / "x.ihs.txt").with_columns(-pl.col("ihs").abs() - 0.1)
    track, _, signed = _scan_track(df, "ihs", pvalue=False, top_pct=0.01)
    assert signed and int((track["_cls"] == -1).sum()) > 0


def _run(args):
    res = CliRunner().invoke(cli, ["scan-plot", *args])
    assert res.exit_code == 0, res.output
    return res


def _two_stats(tmp_path, chroms):
    for st, seed in (("ihs", 0), ("nsl", 1)):
        df = _scan_tsv(tmp_path / f"yri.{st}.txt", chroms=chroms, seed=seed)
        df.rename({"ihs": st, "ihs_pvalue": f"{st}_pvalue"}).write_csv(
            tmp_path / f"yri.{st}.txt", separator="\t"
        )
    return [str(tmp_path / "yri.ihs.txt"), str(tmp_path / "yri.nsl.txt")]


def _svgs(tmp_path):
    return sorted(p.name for p in tmp_path.glob("p*.svg"))


def test_cli_four_modes(tmp_path):
    files = _two_stats(tmp_path, ("chr7", "chr22"))
    stats = [a for f in files for a in ("--stat", f)]  # --col inferred from name
    out = ["--out", str(tmp_path / "p.svg")]

    # per stat, all chromosomes
    _run(stats + out)
    assert _svgs(tmp_path) == ["p.ihs.svg", "p.nsl.svg"]
    # per stat, per chromosome
    for f in tmp_path.glob("p*.svg"):
        f.unlink()
    _run(stats + out + ["--split_chr"])
    assert _svgs(tmp_path) == [
        "p.ihs.chr22.svg",
        "p.ihs.chr7.svg",
        "p.nsl.chr22.svg",
        "p.nsl.chr7.svg",
    ]
    # all stats stacked, all chromosomes
    for f in tmp_path.glob("p*.svg"):
        f.unlink()
    _run(stats + out + ["--multi"])
    assert _svgs(tmp_path) == ["p.svg"]
    # all stats stacked, per chromosome
    for f in tmp_path.glob("p*.svg"):
        f.unlink()
    _run(stats + out + ["--multi", "--split_chr"])
    assert _svgs(tmp_path) == ["p.chr22.svg", "p.chr7.svg"]


def test_cli_rejects_pvalue_without_column_and_uninferable_col(tmp_path):
    f = tmp_path / "yri.ihs.txt"
    _scan_tsv(f).drop("ihs_pvalue").write_csv(f, separator="\t")
    res = CliRunner().invoke(cli, ["scan-plot", "--stat", str(f), "--pvalue"])
    assert res.exit_code != 0 and "ihs_pvalue" in res.output
    g = tmp_path / "custom.txt"
    _scan_tsv(g)
    res = CliRunner().invoke(cli, ["scan-plot", "--stat", str(g)])
    assert res.exit_code != 0 and "--col" in res.output


def test_legacy_columns_and_split_chr(tmp_path):
    f = tmp_path / "x.ihs.txt"
    # Numeric labels first: the chromosome column must still be read as text.
    _scan_tsv(f, chroms=("7", "chr22"), legacy=True)
    figs = plot_scan(str(f), "ihs", split_chr=True, out=str(tmp_path / "s.svg"))
    assert len(figs) == 2
    assert (tmp_path / "s.chr7.svg").exists()
    assert (tmp_path / "s.chr22.svg").exists()


# -- x-axis alignment between SNP and window statistics ----------------------
#
# Window stats (raisd, lassi, haf, fay_wu_h, ...) report window centres, so their
# positions start later and end earlier than per-SNP stats (ihs, isafe). The
# oracle is the rendered figure itself: the Mb tick labels fix the bp -> x map of
# each chromosome block, and every input position must land exactly where that
# map puts it, in every panel and every figure.


def _mixed_tracks(tmp_path):
    rng = np.random.default_rng(3)
    snp, win = [], []
    for c, lo in (("chr7", 100_000_000), ("chr22", 20_000_000)):
        pos = np.sort(rng.choice(np.arange(lo, lo + 5_000_000), 3000, replace=False))
        snp.append(
            pl.DataFrame({"chr": c, "positions": pos, "ihs": rng.normal(size=pos.size)})
        )
        # RAiSD-style centres: midpoint of first/last SNP of 50-SNP windows
        cen = ((pos[:-49] + pos[49:]) // 2)[::10]
        win.append(
            pl.DataFrame(
                {"chr": c, "positions": cen, "mu_total": rng.gamma(2, size=cen.size)}
            )
        )
    f_snp, f_win = tmp_path / "s.ihs.txt", tmp_path / "s.raisd.txt"
    pl.concat(snp).write_csv(f_snp, separator="\t")
    pl.concat(win).write_csv(f_win, separator="\t")
    return [str(f_snp), str(f_win)], ["ihs", "mu_total"], pl.concat(snp), pl.concat(win)


def _block_offsets(ax):
    """{x - bp} constant per chromosome block, read from the Mb tick labels."""
    xs = np.asarray(ax.get_xticks())
    mb = np.array([float(t.get_text()) for t in ax.get_xticklabels()])
    return np.unique(np.round(xs - mb * 1e6, 3))


def _assert_points_on_ticks(ax, df, tick_ax=None):
    pts = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections])
    offs = _block_offsets(tick_ax or ax)
    bp = df["positions"].to_numpy().astype(float)
    # each block's offset recovers bp exactly for every rendered point
    recovered = set()
    for o in offs:
        recovered |= set(np.round(pts - o).astype(np.int64)) & set(bp.astype(np.int64))
    assert recovered == set(bp.astype(np.int64))
    assert pts.size == bp.size


def test_window_and_snp_stats_share_x_scale_multi(tmp_path):
    files, cols, snp, win = _mixed_tracks(tmp_path)
    fig = plot_scan(files, cols, multi=True, out=str(tmp_path / "m.svg"))
    ax_snp, ax_win = fig.axes[:2]
    assert ax_snp.get_xlim() == ax_win.get_xlim()
    # ticks live on the bottom axis; sharex gives the top one the same map
    _assert_points_on_ticks(ax_win, win)
    _assert_points_on_ticks(ax_snp, snp, tick_ax=ax_win)


# -- output format -------------------------------------------------------------


def test_svg_is_real_svg_with_vector_points_and_text(tmp_path):
    import xml.etree.ElementTree as ET

    f = tmp_path / "x.ihs.txt"
    _scan_tsv(f)
    out = tmp_path / "plot.svg"
    plot_scan(str(f), "ihs", out=str(out))
    raw = out.read_bytes()
    assert not raw.startswith(b"\x89PNG")
    root = ET.fromstring(raw)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    tags = {el.tag.split("}")[-1] for el in root.iter()}
    assert "text" in tags  # svg.fonttype="none": labels stay text
    assert "image" not in tags  # small track: no embedded raster



def test_raisd_label_is_mu():
    df = pl.DataFrame(
        {"chr": ["chr22"] * 3, "positions": [1, 2, 3], "mu_total": [1.0, 2.0, 3.0],
         "mu_total_pvalue": [0.5, 0.1, 0.01]}
    )
    _, label, _ = _scan_track(df, "mu_total", pvalue=False, top_pct=0.01)
    assert label == "mu"
    _, label, _ = _scan_track(df, "mu_total", pvalue=True, top_pct=0.01)
    assert label.startswith(r"$\mathrm{mu}:") and "total" not in label
