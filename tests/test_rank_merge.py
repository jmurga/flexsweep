"""Multi-population ranking (`flexsweep rank --prediction_dir`) and the case-ID gene set.

Oracle for the merge: `rank_probabilities` run on each population file on its own; the
merged matrix must carry exactly those ranks, in `populations` column order, sorted by
gene_id. Oracle for the gene set: hand-computed case/control sets on a 6-gene universe.
"""

import os

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from flexsweep.enrichment import load_case_control
from flexsweep.main import cli
from flexsweep.utils import find_prediction_files, rank_merge, rank_probabilities

POPS = ["yri", "ceu", "chb"]
GENES = [
    ("1", 100_000, 110_000, "ENSG03"),
    ("1", 900_000, 910_000, "ENSG01"),
    ("1", 2_000_000, 2_050_000, "ENSG05"),
    ("2", 300_000, 320_000, "ENSG02"),
    ("2", 1_500_000, 1_510_000, "ENSG06"),
    ("2", 3_000_000, 3_005_000, "ENSG04"),
]


def _write_bed(path):
    with open(path, "w") as fh:
        for c, s, e, g in GENES:
            fh.write(f"{c}\t{s}\t{e}\t{g}\t+\n")


def _write_predictions(path, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for c in ("chr1", "chr2"):
        for start in range(0, 3_500_000, 100_000):
            rows.append((c, start + 1, start + 1_200_000, float(rng.random())))
    pl.DataFrame(
        rows, schema=["chr", "start", "end", "prob_sweep"], orient="row"
    ).with_columns(pl.lit("neutral").alias("predicted_model")).write_csv(path)


@pytest.fixture
def pred_dir(tmp_path):
    _write_bed(tmp_path / "genes.bed")
    for i, pop in enumerate(POPS):
        _write_predictions(tmp_path / f"{pop}_v3_predictions.txt", seed=i)
    # A stale single-pop rank output must not be picked up by the default pattern.
    (tmp_path / "yri_v3_predictions_rank_1.txt").write_text("ENSG01\t1\n")
    return tmp_path


def test_find_prediction_files_order_and_errors(pred_dir):
    files = find_prediction_files(str(pred_dir), ["chb", "yri"])
    assert [os.path.basename(f) for f in files] == [
        "chb_v3_predictions.txt",
        "yri_v3_predictions.txt",
    ]
    with pytest.raises(ValueError, match="gbr"):
        find_prediction_files(str(pred_dir), ["gbr"])
    (pred_dir / "yri_other_predictions.txt").write_text("x\n")
    with pytest.raises(ValueError, match="yri"):
        find_prediction_files(str(pred_dir), ["yri"])


def test_rank_merge_matches_single_population_ranking(pred_dir):
    bed = str(pred_dir / "genes.bed")
    files = find_prediction_files(str(pred_dir), POPS)
    out = str(pred_dir / "all_ranks_test")
    df, ties = rank_merge(files, POPS, bed, output=out)

    assert df.columns == ["gene_id", *POPS]
    assert df["gene_id"].to_list() == sorted(g for *_, g in GENES)
    for pop, f in zip(POPS, files):
        ref, n_tied = rank_probabilities(f, bed)
        expected = dict(zip(ref["gene_id"], ref["rank"]))
        assert dict(zip(df["gene_id"], df[pop])) == expected
        assert ties[pop] == n_tied

    lines = open(out).read().splitlines()
    assert len(lines) == len(GENES)
    assert all(len(line.split(" ")) == 1 + len(POPS) for line in lines)
    assert lines[0].split(" ")[0] == "ENSG01"


def test_rank_merge_rejects_different_gene_sets(pred_dir):
    bed = str(pred_dir / "genes.bed")
    files = find_prediction_files(str(pred_dir), POPS)
    # Drop chr2 windows from one population: its chr2 genes cannot be ranked.
    df = pl.read_csv(files[1]).filter(pl.col("chr") == "chr1")
    df.write_csv(files[1])
    with pytest.raises(ValueError, match="ceu"):
        rank_merge(files, POPS, bed)


def test_rank_cli_dir_mode(pred_dir):
    out = str(pred_dir / "all_ranks_cli")
    result = CliRunner().invoke(
        cli,
        [
            "rank",
            "--prediction_dir", str(pred_dir),
            "--populations", ",".join(POPS),
            "--feature_coordinates", str(pred_dir / "genes.bed"),
            "--output", out,
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "chb_v3_predictions.txt" in result.output
    merged = pl.read_csv(out, separator=" ", has_header=False)
    assert merged.shape == (len(GENES), 1 + len(POPS))


def test_load_case_control(tmp_path):
    gs = tmp_path / "vips.txt"
    gs.write_text("G1\nG2\n\nG9\nHLA1 extra\n")
    universe = ["G1", "G2", "G3", "G4", "HLA1", "HLA2"]
    case, control, n_dropped = load_case_control(str(gs), universe, {"HLA1", "HLA2"})
    assert case == ["G1", "G2"]
    assert control == ["G3", "G4"]
    assert n_dropped == 2  # G9 outside the universe, HLA1 excluded


def _enrichment_inputs(tmp_path, n_rank_cols):
    sweep = tmp_path / "all_ranks_x"
    sweep.write_text("G1 " + " ".join(["1"] * n_rank_cols) + "\n")
    for name in ("gs.txt", "factors.txt", "genes.bed"):
        (tmp_path / name).write_text("G1\n")
    return str(sweep)


@pytest.mark.parametrize(
    "pops,groups,n_cols,match",
    [
        (["a", "b"], ["1000"], 2, "groups"),
        (["a", "b"], ["x", "y"], 3, "columns"),
    ],
)
def test_run_enrichment_rejects_bad_inputs_before_bootstrap(
    tmp_path, pops, groups, n_cols, match
):
    from flexsweep.enrichment import run_enrichment

    sweep = _enrichment_inputs(tmp_path, n_cols)
    with pytest.raises(ValueError, match=match):
        run_enrichment(
            sweep_files=[sweep],
            gene_set=str(tmp_path / "gs.txt"),
            factors_file=str(tmp_path / "factors.txt"),
            annotation_file=str(tmp_path / "genes.bed"),
            populations=pops,
            groups=groups,
            thresholds=[10],
        )


def _write_curves(path):
    rows = []
    for ds in ("/d/all_ranks_a", "/d/all_ranks_b"):
        for scope in ("yri", "ceu", "afr", "All"):
            for i, thr in enumerate((20, 50, 100)):
                rows.append((scope, thr, 3, 1.0, 0.5, 1.5, 1.0 + i, 0.8, 1.2, 0.1, ds))
    pl.DataFrame(
        rows,
        schema=["scope", "threshold", "vip_count", "ctrl_mean", "ctrl_ci_lo",
                "ctrl_ci_hi", "ratio", "ci_lo_ratio", "ci_hi_ratio", "p_value",
                "dataset"],
        orient="row",
    ).write_csv(path)


def test_plot_enrichment_from_file(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from flexsweep.enrichment import plot_enrichment

    curves = tmp_path / "enrichment_curves.txt"
    _write_curves(curves)
    out = tmp_path / "plots"
    figs = plot_enrichment(str(curves), scopes=["yri", "ceu"], output_folder=str(out))
    assert sorted(figs) == ["/d/all_ranks_a", "/d/all_ranks_b"]
    labels = sorted(t.get_text() for t in figs["/d/all_ranks_a"].axes[0].get_legend().get_texts())
    assert labels == ["ceu", "yri"]
    assert (out / "enrichment_curves_all_ranks_a.svg").is_file()
    assert (out / "enrichment_curves_all_ranks_b.svg").is_file()

    facet = plot_enrichment(pl.read_csv(curves), facet=True)
    visible = [ax for ax in facet["/d/all_ranks_a"].axes if ax.get_visible()]
    assert len(visible) == 4  # one panel per scope: yri, ceu, afr, All


def test_ratio_ci_brackets_ratio():
    from flexsweep.enrichment import _ratio_ci

    df = pl.DataFrame(
        {
            "vip_count": [14, 0],
            "ctrl_mean": [0.261, 1.0],
            "ctrl_ci_lo": [0.0, 0.0],
            "ctrl_ci_hi": [2.0, 3.0],
            "ratio": [(14 + 0.1) / (0.261 + 0.1), (0 + 0.1) / (1.0 + 0.1)],
        }
    )
    out = _ratio_ci(df)
    # Hand-computed: case count fixed, divided by the control percentiles (bounds swap).
    assert out["CI_low_ratio"].to_list() == pytest.approx([14.1 / 2.1, 0.1 / 3.1])
    assert out["CI_up_ratio"].to_list() == pytest.approx([14.1 / 0.1, 0.1 / 0.1])
    assert (out["CI_low_ratio"] <= out["ratio"]).all()
    assert (out["ratio"] <= out["CI_up_ratio"]).all()
