"""Enrichment counting, shuffling and FDR against frozen outputs of the Perl pipeline.

Oracle: tests/data/enrichment_perl_oracle.json, the outputs of the Enard-lab Perl scripts
(count_sweeps_script.pl with simplify_zipped_sweeps.pl, estimate_FPR.pl ... All:, and
shuffle_sweep_script_nocut.pl with mirroring disabled) on the synthetic inputs built
below; provenance is stored in the file. No Perl is run here. If the input builders
change, the fixture has to be regenerated from the Perl scripts.
"""

import json
import os

import numpy as np
import polars as pl
import pytest

from flexsweep import enrichment as E

ORACLE = json.load(
    open(os.path.join(os.path.dirname(__file__), "data", "enrichment_perl_oracle.json"))
)
POPS = ["p1", "p2", "p3", "p4", "p5"]
GROUPS = ["ga", "ga", "gb", "gb", "gc"]
THR = [60, 40, 20, 10, 5]  # descending, as in the Perl Threshold_list
N_GENES = 160


def _synthetic(seed=0, n_sets=150):
    """Rank table with case genes enriched in p1/p2 only; control sets with duplicates."""
    rng = np.random.default_rng(seed)
    genes = [f"ENSG{i:05d}" for i in range(N_GENES)]
    vip = genes[:20]
    ranks = np.stack([rng.permutation(N_GENES) + 1 for _ in POPS], axis=1)
    for j in (0, 1):  # push case genes to the top in p1, p2
        top = rng.permutation(np.arange(1, 25))[:20]
        others = [r for r in ranks[:, j] if r not in set(top)]
        ranks[:20, j] = top
        ranks[20:, j] = rng.permutation(others)[: N_GENES - 20]
    pool = genes[20:]
    sets = np.array([rng.choice(pool, len(vip)) for _ in range(n_sets)])
    table = pl.DataFrame(
        {"column_1": genes}
        | {f"column_{j + 2}": ranks[:, j].astype(str) for j in range(len(POPS))}
    )
    return table, vip, sets


def _coords_bed(tmp):
    """Gene BED (0-based): bins with several genes, a 1-based centre exactly on a 10 kb
    edge, chromosomes 1, 2, 10 and X, file order different from position order."""
    rng = np.random.default_rng(5)
    chroms = ["1"] * 70 + ["2"] * 50 + ["10"] * 38 + ["X"] * 2
    rows = []
    for i, c in enumerate(chroms):
        start1 = 1 + int(rng.integers(0, 60)) * 5_000 + int(rng.integers(0, 3)) * 7
        rows.append((f"ENSG{i:05d}", c, start1, start1 + int(rng.integers(500, 4000))))
    rows[3] = (rows[3][0], "1", 9_001, 10_999)
    with open(tmp / "coords.bed", "w") as fb:
        for k in rng.permutation(len(rows)):
            g, c, s1, e = rows[k]
            fb.write(f"{c}\t{s1 - 1}\t{e}\t{g}\t+\n")
    return str(tmp / "coords.bed")


@pytest.mark.parametrize("setpop", ["p1", "p3"])  # full depth / stops at 20 sets
def test_count_matches_perl(setpop):
    table, vip, sets = _synthetic()
    perl = {(t, s): v for t, s, *v in ORACLE["count"][setpop]}
    py = E.count_sweeps_fast(table, vip, sets, THR, POPS, GROUPS, pop_interest=setpop)

    assert len(perl) == py.height == len(THR) * (len(POPS) + 3 + 1)
    for r in py.iter_rows(named=True):
        ratio, score, av, lo, hi, pval = perl[(r["threshold"], r["scope"])]
        got = [r["ratio"], r["vip_count"], r["ctrl_mean"], r["ctrl_ci_lo"], r["ctrl_ci_hi"], r["p_value"]]
        assert got == pytest.approx([ratio, score, av, lo, hi, pval], rel=1e-9, abs=1e-12), (
            r["scope"], r["threshold"],
        )


def test_shuffle_segments_match_perl(tmp_path):
    table, vip, sets = _synthetic()
    used = sorted(set(vip) | set(np.unique(sets)[:80]))  # universe = bootstrap genes
    ordered = E._ordered_genes(_coords_bed(tmp_path), used)
    strong = E._strong_signal_genes(table, max(THR))
    py_segs = [[ordered[i] for i in seg] for seg in E._shuffle_segments(ordered, strong, 2)]
    perl_segs = ORACLE["segments"]
    assert [g for s in perl_segs for g in s] == ordered  # same universe and gene order
    assert py_segs == perl_segs  # same segment boundaries


def test_fdr_matches_estimate_fpr(tmp_path):
    ref = ORACLE["fdr"]
    table, vip, sets = _synthetic()
    used = sorted(set(vip) | set(np.unique(sets)))
    fdr = E.estimate_fdr_fast(
        table, _coords_bed(tmp_path), used, vip, sets, THR, POPS, GROUPS,
        n_shuffles=ref["n_shuffles"], shuffling_segs=2, seed=ref["seed"], nthreads=2,
        pop_interest=ref["pop_interest"],
    ).row(0, named=True)
    assert fdr["n_replicates"] == ref["n_replicates"]
    assert fdr["real_stat"] == pytest.approx(ref["real_stat"], rel=1e-9)
    assert fdr["max_null_stat"] == pytest.approx(ref["max_null_stat"], rel=1e-9)
    assert fdr["p_value"] == pytest.approx(ref["p_value"], abs=1e-12)


# --- Count_sweeps yes and block shuffles -------------------------------------------
# Oracle: tests/data/enrichment_perl_oracle_clust.json, made by
# tests/data/make_perl_oracle_clust.py from count_sweeps_script.pl (Count_sweeps yes,
# get_all_neighbors.pl) and shuffle_sweep_script_nocut.pl with the block line restored.

ORACLE_CLUST = json.load(
    open(os.path.join(os.path.dirname(__file__), "data", "enrichment_perl_oracle_clust.json"))
)


def _clust_table(name):
    table, vip, sets = _synthetic()
    if name != "real":
        perm = ORACLE_CLUST["permutations"][name]
        table = table.with_columns([table[c].gather(perm) for c in table.columns[1:]])
    return table, vip, sets


# both distances, real and one shuffled table, enriched and non-enriched Pop_interest
CLUST_KEYS = [k for k in sorted(ORACLE_CLUST["count_clust"])
              if k.split("|")[1] in ("real", "perm0") and k.split("|")[2] in ("p1", "p3")]


@pytest.mark.parametrize("key", CLUST_KEYS)
def test_count_sweeps_yes_matches_perl(key, tmp_path):
    dist, name, setpop = key.split("|")
    table, vip, sets = _clust_table(name)
    genes = table[:, 0].to_list()
    nb = E._gene_neighbors(_coords_bed(tmp_path), genes, genes, int(dist))
    perl = {(t, s): v for t, s, *v in ORACLE_CLUST["count_clust"][key]}
    # case genes given in reverse: counting must still follow Perl's sorted VIPs/file_1
    py = E.count_sweeps_fast(table, vip[::-1], sets, THR, POPS, GROUPS, pop_interest=setpop,
                             neighbors=nb)

    assert len(perl) == py.height == len(THR) * (len(POPS) + 3 + 1)
    for r in py.iter_rows(named=True):
        ratio, score, av, lo, hi, pval = perl[(r["threshold"], r["scope"])]
        got = [r["ratio"], r["vip_count"], r["ctrl_mean"], r["ctrl_ci_lo"], r["ctrl_ci_hi"], r["p_value"]]
        assert got == pytest.approx([ratio, score, av, lo, hi, pval], rel=1e-9, abs=1e-12), (
            r["scope"], r["threshold"],
        )


@pytest.mark.parametrize("n_blocks", sorted(ORACLE_CLUST["segments_blocks"]))
def test_block_segments_match_perl(n_blocks, tmp_path):
    table, vip, sets = _synthetic()
    used = sorted(set(vip) | set(np.unique(sets)[:80]))
    ordered = E._ordered_genes(_coords_bed(tmp_path), used)
    strong = E._strong_signal_genes(table, max(THR))
    segs = E._shuffle_segments(ordered, strong, len(ordered) // int(n_blocks))
    assert [[int(i) + 1 for i in s] for s in segs] == ORACLE_CLUST["segments_blocks"][n_blocks]


def test_fdr_null_counts_like_real_data_with_clusters(tmp_path):
    """An identity shuffle must give the real statistic, so the null uses the same
    Count_sweeps yes counting and case order as the real data."""
    table, vip, sets = _synthetic()
    genes = table[:, 0].to_list()
    bed = _coords_bed(tmp_path)
    nb = E._gene_neighbors(bed, genes, genes, 60_000)
    th, bins, V, M, _, _ = E._pop_counts(table, vip, sets, THR, POPS, nb)
    ordered = E._ordered_genes(bed, genes)  # autosomes only: chrX genes are never shuffled
    g2i = {g: i for i, g in enumerate(genes)}
    ord_rows = np.array([g2i[g] for g in ordered], dtype=np.int64)
    ref_bins = np.full((len(genes) + 1, len(POPS)), len(THR), dtype=np.int16)
    ref_bins[ord_rows] = bins[ord_rows]
    vip_c, ctrl_c = E._count_bins_clust(ref_bins, M, V, nb[0], nb[1], len(THR))
    vs, cs, names = E._add_scopes(vip_c, ctrl_c, POPS, GROUPS)
    n = E._adaptive_depth(vs, cs, names, "All")
    filt = (1.0, 1, 0, max(THR))
    real = E._all_stat(vs[-1], cs[-1], n, th, *filt)
    ident = np.arange(len(ordered), dtype=np.int32)[None, :]
    null = E._fdr_null_chunk(bins[ord_rows], ord_rows, len(genes) + 1, ident, M, V, th,
                             POPS, GROUPS, "All", True, filt, nb)
    assert null[0] == pytest.approx(real, rel=1e-12)
    plain = E._fdr_null_chunk(bins[ord_rows], ord_rows, len(genes) + 1, ident, M,
                              np.sort(V), th, POPS, GROUPS, "All", True, filt)
    assert plain[0] != pytest.approx(real, rel=1e-6)  # without neighbours it differs
