"""NaN filter of the feature vectors (fv.filter_missing, max_missing option).

Oracles: a brute-force numpy count of missing cells per row with the documented rule
(kept when count < int(n_features * max_missing)), and the inline filter fv.py used
before max_missing existed (copied verbatim below) for the default 0.1.
"""

import numpy as np
import polars as pl
import pytest

from flexsweep.fv import filter_missing

META = ["iter", "s", "t", "f_i", "f_t", "mu", "r", "model"]


def _frame(n_rows=400, n_feat=60, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, n_feat))
    # per-row missing share from 0 to 1, mixed NaN / +inf / -inf / null
    share = rng.uniform(0, 1, n_rows)
    miss = rng.uniform(size=x.shape) < share[:, None]
    kind = rng.integers(0, 4, size=x.shape)
    x[miss & (kind == 0)] = np.nan
    x[miss & (kind == 1)] = np.inf
    x[miss & (kind == 2)] = -np.inf
    df = pl.DataFrame(x, schema=[f"stat_{j}" for j in range(n_feat)])
    null_mask = miss & (kind == 3)
    df = df.with_columns(
        pl.when(pl.Series(null_mask[:, j])).then(None).otherwise(pl.col(c)).alias(c)
        for j, c in enumerate(df.columns)
    )
    meta = pl.DataFrame(
        {
            "iter": [f"chr1:{i}-{i + 1}" for i in range(n_rows)],
            "s": np.zeros(n_rows),
            "t": np.zeros(n_rows),
            "f_i": np.zeros(n_rows),
            "f_t": np.zeros(n_rows),
            "mu": np.zeros(n_rows),
            "r": np.full(n_rows, np.nan),  # meta NaN must not count
            "model": ["sweep"] * n_rows,
        }
    )
    return pl.concat([meta, df], how="horizontal"), miss


def _old_inline(df_fv_w):
    """fv.normalize_stats NaN filter before the max_missing option (verbatim)."""
    df_fv_w = df_fv_w.with_columns(
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
        for c, dtype in df_fv_w.schema.items()
        if dtype.is_float()
    )
    df_fv_w = df_fv_w.fill_nan(None)
    num_nans = (
        df_fv_w.select(pl.exclude(["iter", "s", "t", "f_i", "f_t", "mu", "r", "model"]))
        .transpose()
        .null_count()
        .to_numpy()
        .flatten()
    )
    return df_fv_w.filter(
        num_nans
        < int(
            df_fv_w.select(
                pl.exclude(["iter", "s", "t", "f_i", "f_t", "r", "mu", "model"])
            ).shape[1]
            * 0.1
        )
    ).fill_null(0)


def test_default_matches_previous_inline_filter():
    df, _ = _frame()
    new = filter_missing(df)
    old = _old_inline(df)
    assert new.equals(old)
    assert 0 < new.height < df.height


@pytest.mark.parametrize("max_missing", [0.1, 0.25, 0.5, 0.7, 1.0])
def test_kept_rows_match_bruteforce_count(max_missing):
    df, miss = _frame()
    n_feat = miss.shape[1]
    expected = df["iter"].to_numpy()[miss.sum(1) < int(n_feat * max_missing)]
    out = filter_missing(df, max_missing=max_missing)
    assert out["iter"].to_list() == list(expected)
    feats = out.select(pl.exclude(META)).to_numpy()
    assert np.isfinite(feats).all()
    # every missing cell of a kept row became 0, every other cell is unchanged
    keep = miss.sum(1) < int(n_feat * max_missing)
    src = df.filter(pl.Series(keep)).select(pl.exclude(META)).to_numpy()
    m = miss[keep]
    assert (feats[m] == 0).all()
    assert np.array_equal(feats[~m], src[~m])


def test_boundary_strictly_below_limit():
    n_feat = 20
    cols = {f"stat_{j}": [] for j in range(n_feat)}
    # limit int(20 * 0.5) = 10: 9 missing kept, 10 missing dropped
    for n_miss in (9, 10):
        for j in range(n_feat):
            cols[f"stat_{j}"].append(np.nan if j < n_miss else 1.0)
    df = pl.DataFrame({"iter": ["a", "b"], **cols})
    out = filter_missing(df, max_missing=0.5)
    assert out["iter"].to_list() == ["a"]


def test_larger_value_keeps_more_and_is_monotone():
    df, _ = _frame()
    heights = [filter_missing(df, max_missing=v).height for v in (0.1, 0.3, 0.5, 0.7, 1.0)]
    assert heights == sorted(heights)
    assert heights[0] < heights[-1]


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_invalid_value_raises(bad):
    df, _ = _frame(n_rows=5)
    with pytest.raises(ValueError):
        filter_missing(df, max_missing=bad)


# CLI wiring: fvs-discoal and fvs-vcf pass --max_missing to summary_statistics.


def _capture(monkeypatch):
    import flexsweep as fs
    import flexsweep.data

    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(fs, "summary_statistics", fake)
    monkeypatch.setattr(fs.fv, "summary_statistics", fake)
    monkeypatch.setattr(flexsweep.data, "check_genome_build", lambda *a, **k: None)
    return seen


@pytest.mark.parametrize("cmd", ["fvs-discoal", "fvs-vcf"])
@pytest.mark.parametrize("value,expected", [(None, 0.1), ("0.7", 0.7)])
def test_cli_passes_max_missing(monkeypatch, tmp_path, cmd, value, expected):
    from click.testing import CliRunner

    from flexsweep.main import cli

    seen = _capture(monkeypatch)
    if cmd == "fvs-discoal":
        argv = [cmd, "--simulations_path", str(tmp_path), "--nthreads", "1"]
    else:
        argv = [cmd, "--vcf_path", str(tmp_path), "--suffix", "x"]
    if value is not None:
        argv += ["--max_missing", value]
    res = CliRunner().invoke(cli, argv)
    assert res.exit_code == 0, res.output
    assert seen["max_missing"] == expected


@pytest.mark.parametrize("cmd", ["fvs-discoal", "fvs-vcf"])
@pytest.mark.parametrize("bad", ["0", "1.5", "-0.2"])
def test_cli_rejects_invalid_max_missing(monkeypatch, tmp_path, cmd, bad):
    from click.testing import CliRunner

    from flexsweep.main import cli

    _capture(monkeypatch)
    path_opt = "--simulations_path" if cmd == "fvs-discoal" else "--vcf_path"
    argv = [cmd, path_opt, str(tmp_path), "--suffix", "x", "--max_missing", bad]
    if cmd == "fvs-discoal":
        argv = [a for a in argv if a not in ("--suffix", "x")]
    res = CliRunner().invoke(cli, argv)
    assert res.exit_code == 2
    assert "max_missing" in res.output
