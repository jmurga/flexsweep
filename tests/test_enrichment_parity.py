"""Enrichment counting, shuffling and FDR against frozen outputs of the Perl pipeline.

Oracle: tests/data/enrichment_perl_oracle.json, the outputs of the Enard-lab Perl scripts
(count_sweeps_script.pl with simplify_zipped_sweeps.pl, estimate_FPR.pl ... All:, and
shuffle_sweep_script_nocut.pl without mirroring) on the synthetic inputs built
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


def _curves(table, vip, sets, neighbors=None):
    return E.enrichment_curves(
        E.count_sweeps_fast(table, vip, sets, THR, POPS, GROUPS, neighbors)
    )


# Perl stops at 20 or 100 of the 150 sets when the Pop_interest p-values are not extreme;
# the package always uses the sets it is given, so it gets the sets Perl ended up using.
@pytest.mark.parametrize("setpop, n_sets", [("p1", 150), ("p3", 20)])
def test_count_matches_perl(setpop, n_sets):
    table, vip, sets = _synthetic()
    perl = {(t, s): v for t, s, *v in ORACLE["count"][setpop]}
    py = _curves(table, vip, sets[:n_sets])

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


def _fdr_problem(tmp_path, rule, n_sets=20):
    """Counts, annotation and shuffled genes; ``rule``: Count_sweeps no / yes."""
    table, vip, sets = _synthetic()
    sets, genes, bed = sets[:n_sets], table[:, 0].to_list(), _coords_bed(tmp_path)
    nb = E._gene_neighbors(bed, genes, genes, ORACLE["fdr"]["distance"]) if rule == "yes" else None
    counts = E.count_sweeps_fast(table, vip, sets, THR, POPS, GROUPS, nb)
    return counts, bed, sorted(set(vip) | set(np.unique(sets)))


@pytest.mark.parametrize("rule", ["no", "yes"])
def test_fdr_matches_estimate_fpr(tmp_path, rule):
    """One row per scope, as estimate_FPR.pl run once per scope on the same shuffles."""
    ref = ORACLE["fdr"]
    args = _fdr_problem(tmp_path, rule, ref["n_sets"])
    kw = dict(n_shuffles=ref["n_shuffles"], shuffling_segs=2, seed=ref["seed"])
    asked = list(ref[rule])
    fdr = E.estimate_fdr_fast(*args, **kw, pop_interest=asked, nthreads=2)
    assert fdr["scope"].to_list() == asked
    assert fdr["n_replicates"].unique().to_list() == [ref["n_shuffles"]]
    for r in fdr.iter_rows(named=True):
        assert (r["p_value"], r["real_stat"], r["max_null_stat"]) == pytest.approx(
            ref[rule][r["scope"]], rel=1e-9, abs=1e-9
        ), r["scope"]
        k = round(r["p_value"] * ref["n_shuffles"])
        assert r["p_value_perm"] == pytest.approx((k + 1) / (ref["n_shuffles"] + 1))
    # asking for more scopes, in any order or case, or other threads, changes no row
    alone = [E.estimate_fdr_fast(*args, **kw, pop_interest=s) for s in ("ga", "All")]
    both = E.estimate_fdr_fast(*args, **kw, pop_interest="GA, all", nthreads=3)
    assert both.equals(pl.concat(alone))


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
CLUST_DEPTH = {  # control sets Perl used, of 150
    "20000|perm0|p1": 150, "20000|perm0|p3": 20, "20000|real|p1": 150, "20000|real|p3": 20,
    "60000|perm0|p1": 150, "60000|perm0|p3": 100, "60000|real|p1": 150, "60000|real|p3": 20,
}
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
    py = _curves(table, vip[::-1], sets[: CLUST_DEPTH[key]], nb)

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
    counts, bed, _ = _fdr_problem(tmp_path, "yes", 150)
    genes = counts.rank_table[:, 0].to_list()
    ordered = E._ordered_genes(bed, genes)  # autosomes only: chrX genes are never shuffled
    g2i = {g: i for i, g in enumerate(genes)}
    ord_rows = np.array([g2i[g] for g in ordered], dtype=np.int64)
    ref_bins = np.full((len(genes) + 1, len(POPS)), len(THR), dtype=np.int16)
    ref_bins[ord_rows] = counts.bins[ord_rows]
    M, V, nb = counts.M, counts.V, counts.neighbors
    buf = E._clust_buffers(*ref_bins.shape, len(M), len(THR))
    E._count_bins_clust(ref_bins, M, V, nb[0], nb[1], len(THR), *buf)
    vs, cs, names = E._add_scopes(buf[0], buf[1], POPS, GROUPS)
    real = E._excess_stat(vs[-1], cs[-1].mean(axis=1))
    ident = [np.arange(len(ordered), dtype=np.int32)], np.zeros((1, 1), np.int32)
    f = lambda v, n: E._fdr_null_clusters(
        counts.bins[ord_rows], ord_rows, len(genes) + 1, *ident, M, v, len(THR), POPS, GROUPS,
        [len(names) - 1], n,
    )[0, 0]
    assert f(V, nb) == pytest.approx(real, rel=1e-12)
    none = np.zeros(len(genes) + 1, dtype=nb[0].dtype), nb[1][:0]
    assert f(V, none) != pytest.approx(real, rel=1e-6)  # without neighbours it differs


def test_shuffles_are_permutations_of_whole_segments():
    """shuffle_sweep_script_nocut.pl without the mirror: segments move, every gene gives
    its ranks exactly once and a segment keeps its gene order."""
    sizes = [2, 2, 4, 2, 6, 2, 2, 2, 8, 2]
    ends = np.cumsum(sizes)
    segments = [np.arange(e - k, e, dtype=np.int32) for k, e in zip(sizes, ends)]
    orders = E._draw_orders(len(segments), np.random.default_rng(3), 400)
    shuffles = [np.concatenate([segments[k] for k in o]) for o in orders]
    n = int(ends[-1])
    first = np.zeros(len(segments))
    for idx in shuffles:
        assert sorted(idx.tolist()) == list(range(n))
        pos = np.empty(n, dtype=int)
        pos[idx] = np.arange(n)
        for s in segments:
            assert (np.diff(pos[s]) == 1).all()
        first[[k for k, s in enumerate(segments) if s[0] == idx[0]]] += 1
    assert abs(first / 400 - 0.1).max() < 4 * np.sqrt(0.1 * 0.9 / 400)


@pytest.mark.parametrize("dup", [False, True])
def test_fdr_totals_equal_brute_force(tmp_path, dup):
    """The null from gene multiplicities over all sets against shuffling a rank table and
    counting every set. ``dup``: genes listed twice in the annotation."""
    table, vip, sets = _synthetic(n_sets=30)
    bed = _coords_bed(tmp_path)
    if dup:
        with open(bed, "a") as fh:
            fh.write("1\t5000000\t5001000\tENSG00004\t+\n2\t10\t900\tENSG00050\t+\n")
    used = sorted(set(vip) | set(np.unique(sets)))
    counts = E.count_sweeps_fast(table, vip, sets, THR, POPS, GROUPS)
    scopes = ["All", "p2", "ga", "gc"]
    members = {"All": POPS, "p2": ["p2"], "ga": ["p1", "p2"], "gc": ["p5"]}
    seen = []
    real = E._fdr_null_totals
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(E, "_fdr_null_totals", lambda *a: (seen.append(real(*a)), seen[-1])[1])
        E.estimate_fdr_fast(counts, bed, used, n_shuffles=24, shuffling_segs=2, seed=9,
                            pop_interest=scopes)
    ordered, segments = E._fdr_segments(table, bed, used, THR, 2)
    orders = E._draw_orders(len(segments), np.random.default_rng(9), 24)
    ranks = {r[0]: [int(x) for x in r[1:]] for r in table.iter_rows()}
    want = np.empty((24, len(scopes)))
    for r, order in enumerate(orders):
        src = [ordered[i] for k in order for i in segments[k]]
        got = {g: ranks[x] for g, x in zip(ordered, src)}  # a repeated gene keeps the last
        for c, scope in enumerate(scopes):
            pops = [POPS.index(x) for x in members[scope]]
            n = lambda gs, t: sum(got[g][p] <= t for g in gs if g in got for p in pops)
            want[r, c] = sum(
                n(vip, t) - np.mean([n(row, t) for row in sets]) for t in THR if n(vip, t) >= 1
            )
    assert np.allclose(np.concatenate(seen), want, rtol=0, atol=1e-9)
    assert len(np.unique(want[:, 0])) > 10


def test_gene_index_from_table_equals_arrays():
    table, vip, sets = _synthetic()
    genes = table[:, 0].to_list()
    frame = pl.DataFrame({"sample_id": [f"s{i}" for i in range(len(sets))]}).hstack(
        pl.DataFrame(sets.astype(str))
    )
    V, M = E._gene_index(genes, vip, sets)
    V2, M2 = E._gene_index(genes, vip, frame)
    assert np.array_equal(V, V2) and np.array_equal(M, M2) and M.dtype == M2.dtype == np.int32
    assert [[genes[i] for i in row] for row in M[:3]] == sets[:3].tolist()
    with pytest.raises(ValueError, match="1 control genes are not in the rank table"):
        E._gene_index([g for g in genes if g != sets[0][0]], vip, frame)


def test_reused_buffers_do_not_leak_between_shuffles(tmp_path):
    """One block of many shuffles (reused arrays) equals one block per shuffle (fresh)."""
    counts, bed, _ = _fdr_problem(tmp_path, "yes", 30)
    genes = counts.rank_table[:, 0].to_list()
    ordered = E._ordered_genes(bed, genes)
    g2i = {g: i for i, g in enumerate(genes)}
    ord_rows = np.array([g2i[g] for g in ordered], dtype=np.int64)
    segments = E._shuffle_segments(ordered, set(), 2)
    orders = E._draw_orders(len(segments), np.random.default_rng(4), 40)
    f = lambda o: E._fdr_null_clusters(
        counts.bins[ord_rows], ord_rows, len(genes) + 1, segments, o, counts.M, counts.V,
        len(THR), POPS, GROUPS, [0, 5, 8], counts.neighbors,
    )
    block = f(orders)
    assert len(np.unique(block[:, 2])) > 10
    assert np.array_equal(block, np.concatenate([f(orders[r : r + 1]) for r in range(40)]))


@pytest.mark.parametrize("rule", ["no", "yes"])
def test_curves_and_fdr_only_read_the_counts(tmp_path, rule):
    counts, bed, used = _fdr_problem(tmp_path, rule)
    for x in counts:
        if isinstance(x, np.ndarray):
            x.setflags(write=False)
    kw = dict(n_shuffles=16, shuffling_segs=2, seed=1)
    a, b = E.enrichment_curves(counts), E.estimate_fdr_fast(counts, bed, used, **kw)
    assert a.equals(E.enrichment_curves(counts))
    assert b.equals(E.estimate_fdr_fast(counts, bed, used, **kw))
