import os
import re
import time

from numba import njit

from . import Parallel, delayed, np, pl


_NO_CASE_DISTANCE = 100_000_000_000  # Perl distance_file value off case-gene chromosomes


def compute_distances(vip_gene_ids: list, annotation_file: str) -> pl.DataFrame:
    """
    Distance from each gene to the nearest case (VIP) gene, as the Perl
    ``distance_file.txt``: absolute difference of integer 1-based gene centres
    ``(start + 1 + end) // 2`` (BED 0-based start) on the same chromosome, and
    1e11 for genes on chromosomes without case genes.

    Parameters
    ----------
    vip_gene_ids    : case gene IDs (ids absent from the BED are ignored)
    annotation_file : BED (chrom start end gene_id strand), tab-separated, no header

    Returns
    -------
    pl.DataFrame with columns gene_id, distance (Int64), one row per BED gene, sorted.
    """
    df = pl.read_csv(
        annotation_file,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "gene_id", "strand"],
        schema_overrides={"chrom": pl.Utf8},
    ).with_columns(((pl.col("start") + 1 + pl.col("end")) // 2).alias("centre"))
    case = set(vip_gene_ids)
    genes, dists = [], []
    for (_chrom,), sub in df.group_by("chrom"):
        centre = sub["centre"].to_numpy()
        cv = np.sort(sub.filter(pl.col("gene_id").is_in(case))["centre"].to_numpy())
        if cv.size == 0:
            d = np.full(centre.size, _NO_CASE_DISTANCE, dtype=np.int64)
        else:
            k = np.searchsorted(cv, centre)
            left = np.abs(centre - cv[np.clip(k - 1, 0, cv.size - 1)])
            right = np.abs(cv[np.clip(k, 0, cv.size - 1)] - centre)
            d = np.minimum(left, right).astype(np.int64)
        genes.extend(sub["gene_id"].to_list())
        dists.append(d)
    return pl.DataFrame(
        {"gene_id": genes, "distance": np.concatenate(dists) if dists else []},
        schema={"gene_id": pl.Utf8, "distance": pl.Int64},
    ).sort("gene_id")


@njit(cache=True)
def iterative_control_set_fast(
    vip_X: np.ndarray,
    nonvip_X: np.ndarray,
    vip_number: int,
    tolerance: float,
    max_rep: int,
    rng: np.random.Generator,
    skip_factor_idx: int = 8,
    n_batches: int = 10,
    max_tries: int = 100_000_000,
) -> np.ndarray:
    """
    JIT-compiled bootstrap builder for the nomatchomega_fast variant.

    Runs n_batches sequential internal batches with **continuous state**,
    matching Perl's within-process behaviour exactly:
      - used_counts, current_means, fake_seed, sc_gn persist across batches.
      - Target for batch r: 100 + 11 * vip_number * r  (grows each batch).
      - 10 sets sliced from the end of the list after each batch.

    Additional semantic differences vs iterative_control_set (sweep_count.py):
      1. Factor skip_factor_idx (default 8, dN/dS) is unconstrained (±1e18).
      2. The +2 offset on factor 27 has already been applied by the caller.

    Parameters
    ----------
    vip_X           : (vip_number, n_factors) float64 — offset pre-applied
    nonvip_X        : (n_nonvips,  n_factors) float64 — offset pre-applied
    vip_number      : number of VIP genes
    tolerance       : ± fraction for confounder matching (e.g. 0.05)
    max_rep         : dynamic rep-limit numerator
    rng             : numpy Generator (PCG64) for this chain; gene indices are drawn
                      from it in blocks of 4096, uniform on [0, n_nonvips)
    skip_factor_idx : 0-based factor index to leave unconstrained
    n_batches       : number of sequential internal batches (≡ Perl Iterations_number/10)
    max_tries       : consecutive rejected proposals after which the chain gives up
                      (the running means can reach a state no pair can satisfy)

    Returns
    -------
    (sets, used_counts, stuck_at): sets is an int64 array (n_batches * 10, vip_number),
    each row one control set of nonvip_X row indices; used_counts (n_nonvips,) counts how
    often each control gene was accepted (Perl %used_genes = genes with count > 0);
    stuck_at is -1, or the number of accepted control genes when max_tries consecutive
    proposals were rejected (sets is then empty).
    """
    n_nonvips = nonvip_X.shape[0]
    n_factors = nonvip_X.shape[1]

    # Compute VIP mean vector.
    vip_vec = np.zeros(n_factors)
    for i in range(vip_X.shape[0]):
        for f in range(n_factors):
            vip_vec[f] += vip_X[i, f]
    for f in range(n_factors):
        vip_vec[f] /= vip_X.shape[0]

    # Tolerance bounds; skip_factor_idx is unconstrained.
    inf_vec = (1.0 - tolerance) * vip_vec
    sup_vec = (1.0 + tolerance) * vip_vec
    inf_vec[skip_factor_idx] = -1e18
    sup_vec[skip_factor_idx] = 1e18

    init_fake = 100

    # Preallocate for the full continuous list across all batches.
    capacity = (init_fake + 11 * vip_number * n_batches) * 2
    good_idx = np.empty(capacity, dtype=np.int64)

    # Continuous state — persists across all n_batches (matches Perl).
    used_counts = np.zeros(n_nonvips, dtype=np.int32)
    current_means = vip_vec.copy()
    fake_seed = 100
    sc_gn = 0

    # Factors checked, most-rejecting first. Acceptance is an AND over factors, so the
    # order never changes a decision; skip_factor_idx (+-1e18 bounds) never rejects.
    constrained = np.empty(n_factors, dtype=np.int64)
    n_con = 0
    for f in range(n_factors):
        if f != skip_factor_idx:
            constrained[n_con] = f
            n_con += 1
    constrained = constrained[:n_con]
    order = constrained.copy()
    # Factor-major copy: a check reads one factor of two random genes, and a factor
    # column (n_nonvips * 8 bytes) stays in L1 where a gene row would not.
    X_t = np.ascontiguousarray(nonvip_X.T)
    rej_counts = np.zeros(n_factors, dtype=np.int64)
    n_accepted = 0

    # Quantities that only change on acceptance: the repetition limit and the
    # numerator of the new running mean without the candidate pair.
    rep_limit = max_rep * (sc_gn + 1) / vip_number
    denom = sc_gn + 2 + fake_seed
    base = np.empty(n_factors)
    for f in range(n_factors):
        base[f] = fake_seed * vip_vec[f] + current_means[f] * sc_gn

    block = 4096
    draws = rng.integers(0, n_nonvips, size=block)
    k_draw = 0
    n_tries = 0

    out = np.empty((n_batches * 10, vip_number), dtype=np.int64)
    out_row = 0

    for r in range(1, n_batches + 1):
        target_len = init_fake + 11 * vip_number * r  # grows each batch

        while sc_gn < target_len:
            n_tries += 1
            if n_tries > max_tries:
                return out[:0], used_counts, sc_gn
            # Next two values of the i.i.d. uniform stream; i2 == i1 takes the next one.
            if k_draw == block:
                draws = rng.integers(0, n_nonvips, size=block)
                k_draw = 0
            i1 = draws[k_draw]
            k_draw += 1
            i2 = i1
            while i2 == i1:
                if k_draw == block:
                    draws = rng.integers(0, n_nonvips, size=block)
                    k_draw = 0
                i2 = draws[k_draw]
                k_draw += 1

            if used_counts[i1] >= rep_limit or used_counts[i2] >= rep_limit:
                continue

            # Perl: mean of the pair, weighted by 2 (= the pair sum, exactly).
            ok = True
            for k in range(n_con):
                f = order[k]
                new_m = (base[f] + (X_t[f, i1] + X_t[f, i2])) / denom
                if new_m < inf_vec[f] or new_m > sup_vec[f]:
                    rej_counts[f] += 1
                    ok = False
                    break

            if ok:
                if sc_gn + 2 > capacity:
                    new_cap = capacity * 2
                    new_buf = np.empty(new_cap, dtype=np.int64)
                    new_buf[:sc_gn] = good_idx[:sc_gn]
                    good_idx = new_buf
                    capacity = new_cap

                good_idx[sc_gn] = i1
                good_idx[sc_gn + 1] = i2
                used_counts[i1] += 1
                used_counts[i2] += 1

                for f in range(n_factors):
                    current_means[f] = (
                        base[f] + (nonvip_X[i1, f] + nonvip_X[i2, f])
                    ) / denom

                sc_gn += 2
                if fake_seed > 0:
                    fake_seed -= 1

                rep_limit = max_rep * (sc_gn + 1) / vip_number
                denom = sc_gn + 2 + fake_seed
                for f in range(n_factors):
                    base[f] = fake_seed * vip_vec[f] + current_means[f] * sc_gn

                n_accepted += 1
                n_tries = 0
                if n_accepted % 256 == 0:
                    keys = np.empty(n_con, dtype=np.int64)
                    for k in range(n_con):
                        keys[k] = -rej_counts[constrained[k]]
                    order = constrained[np.argsort(keys, kind="mergesort")]

        # Slice 10 sets from the end of good_idx for batch r
        total = sc_gn
        n_sets = 0
        for p in range(10):
            if total - (p + 1) * vip_number < 0:
                break
            n_sets += 1

        for p in range(n_sets):
            sup = total - p * vip_number
            inf_i = sup - vip_number
            for j in range(vip_number):
                out[out_row + (n_sets - 1 - p), j] = good_idx[inf_i + j]
        out_row += n_sets

    return out[:out_row], used_counts, -1


def _chain_layout(total_batches: int, nthreads: int) -> list:
    """
    Batches (of 10 control sets) per bootstrap chain: one chain per thread, batches
    split evenly. A chain's cost is about linear in its number of batches, so this
    gives every thread the same work. The control sets depend on (seed, nthreads).
    """
    n_chains = max(1, min(max(1, nthreads), total_batches))
    base, extra = divmod(total_batches, n_chains)
    return [base + 1] * extra + [base] * (n_chains - extra)


def run_bootstrap_nomatchomega(
    case_genes: list,
    factors_file: str,
    annotation_file: str,
    runs_number: int,
    tolerance: float,
    min_dist: float,
    flip: bool,
    max_rep: int,
    seed: int = None,
    nthreads: int = 1,
    control_genes: list = None,
    distance_file: str = None,
    skip_factor_idx: int = 8,
    offset_factor_idx: int = 27,
    offset_value: float = 2.0,
    n_batches: int = 10,
    return_used: bool = False,
    distance_genes: list | None = None,
    max_tries: int = 100_000_000,
    max_restarts: int = 10,
):
    """
    Step 6: nomatchomega_fast variant of the matched bootstrap control set
    generator.

    Implements the behaviour of bootstrap_test_script_nomatchomega_fast.pl:
      - Factor skip_factor_idx (default 8, omega/dN/dS) is unconstrained.
      - Factor offset_factor_idx (default 27, Perl column 28) gets +offset_value.
      - n_batches sequential internal batches per parallel call, with continuous
        state (used_counts, current_means, fake_seed persist across batches),
        matching Perl's within-process continuous state.
      - Target for batch r: 100 + 11 * vip_number * r (grows each batch).
      - Total control sets = runs_number * n_batches * 10 (Perl Runs_number x
        Iterations_number x 10). The batches are spread over independent chains sized
        to ``nthreads`` (see :func:`_chain_layout`); Perl ran 10 chains of 10 batches.
        Each chain gets its own PCG64 generator from ``SeedSequence(seed).spawn``, so
        the sets depend on (seed, nthreads).
      - ``return_used=True`` also returns the Perl ``used_genes``: case genes plus every
        control gene accepted by a chain (the genome-shuffle universe).

    Parameters
    ----------
    case_genes        : list of VIP gene IDs (HLA/histone excluded by caller)
    factors_file      : path to confounding factors table (TSV, first col gene_id)
    annotation_file   : path to BED gene coordinates file (chrom start end gene_id strand). Distances to nearest VIP are computed automatically via compute_distances.
    runs_number       : number of independent parallel calls (≡ Perl Runs_number)
    tolerance         : allowed ± fraction for confounder matching (e.g. 0.05)
    min_dist          : minimum distance (bp) from VIPs for control eligibility
    flip              : if True, swap VIP and control pools
    max_rep           : max times a control gene may be sampled on average
    nthreads          : number of parallel workers
    control_genes     : optional explicit candidate control gene IDs. If None, all genes in factors_file that are not case genes are used. run_enrichment passes the universe minus case and excluded genes.
    distance_genes    : genes the control-eligibility distance is measured from
                        (default case_genes). The Perl distance_file.txt uses every
                        listed case gene, also those dropped from the universe.
    distance_file     : optional path to pre-computed distance TSV (gene_id \\t distance, no header). When provided, distances are read directly instead of calling compute_distances(). Use Perl's distance_genes_set_file.txt for exact control pool matching.
    skip_factor_idx   : 0-based factor index to leave unconstrained (default 8 = Perl factor 9, dN/dS / omega).
    offset_factor_idx : 0-based factor index to apply +offset_value to (default 27 = Perl column 28).
    offset_value      : value added to offset_factor_idx column (default 2.0).
    n_batches         : sequential internal batches per parallel call (≡ Perl Iterations_number/10; default 10).
    max_tries         : consecutive rejected proposals after which a chain counts as stuck (default 1e8).
    max_restarts      : restarts of a stuck chain on a new seed-derived stream before a RuntimeError (default 10).

    Returns
    -------
    df_set               : pl.DataFrame — VIP genes (gene_id column)
    df_bootstrap_control : pl.DataFrame — rows = control sets, cols = gene positions (sample_id + column_0..column_N)
    """
    case_ids = set(case_genes)

    df_factors = pl.read_csv(factors_file, separator=" ", has_header=False).rename(
        {"column_1": "gene_id"}
    )

    # Perl checks index 19 of the raw line. In Polars, this is column_20.
    df_factors = (
        pl.read_csv(factors_file, separator=" ", has_header=False)
        .rename({"column_1": "gene_id"})
        .filter(pl.col("column_20") >= -1e21)
    )

    # Background pool
    if control_genes is not None:
        background_ids = set(control_genes) - case_ids
    else:
        background_ids = set(df_factors["gene_id"].to_list()) - case_ids

    if distance_file is not None:
        distance_df = pl.read_csv(
            distance_file,
            separator="\t",
            has_header=False,
            new_columns=["gene_id", "distance"],
        )
    else:
        distance_df = compute_distances(
            case_genes if distance_genes is None else distance_genes, annotation_file
        )
    set_by_dist = set(
        distance_df.filter(pl.col("distance") >= min_dist)["gene_id"].to_list()
    )

    df_set = df_factors.filter(pl.col("gene_id").is_in(case_ids))
    df_control_pool = df_factors.filter(
        pl.col("gene_id").is_in(background_ids) & pl.col("gene_id").is_in(set_by_dist)
    )

    if flip:
        df_set, df_control_pool = df_control_pool, df_set

    factor_cols = [c for c in df_factors.columns if c != "gene_id"]
    control_factors = df_control_pool.select(["gene_id"] + factor_cols)

    vip_X = df_set.select(factor_cols).to_numpy().astype(np.float64)
    nonvip_X = control_factors.select(factor_cols).to_numpy().astype(np.float64)
    nonvip_genes = control_factors["gene_id"].to_numpy().astype(str)
    vip_number = df_set.height

    # Apply offset once in Python — shared read-only across all parallel workers.
    if 0 <= offset_factor_idx < vip_X.shape[1]:
        vip_X_off = vip_X.copy()
        nonvip_X_off = nonvip_X.copy()
        vip_X_off[:, offset_factor_idx] += offset_value
        nonvip_X_off[:, offset_factor_idx] += offset_value
    else:
        vip_X_off = vip_X
        nonvip_X_off = nonvip_X

    print(
        f"Bootstrap: {runs_number * n_batches * 10} control sets for {df_set.height} "
        f"case genes from {df_control_pool.height} candidates >= {min_dist:,} bp away",
        flush=True,
    )
    if df_control_pool.height <= 1.5 * df_set.height:
        print(
            "Warning: fewer than 1.5x as many control candidates as case genes; "
            "the FDR may be high.",
            flush=True,
        )

    chain_batches = _chain_layout(runs_number * n_batches, nthreads)
    target = 100 + 11 * vip_number * max(chain_batches)

    # One independent PCG64 stream per chain (seed None -> fresh OS entropy).
    chain_seeds = np.random.SeedSequence(seed).spawn(len(chain_batches))

    def _run_one_chain(run_idx: int, chain_n_batches: int):
        # A chain can reach a running-mean state no pair satisfies, mostly in its first
        # few hundred genes. Such a chain restarts from scratch on a new stream derived
        # from the seed (attempt 0 is the chain's own stream), so runs stay reproducible.
        stuck = []
        for attempt in range(max_restarts + 1):
            ss = chain_seeds[run_idx]
            if attempt > 0:
                ss = np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (attempt,))
            idx, used_counts, stuck_at = iterative_control_set_fast(
                vip_X_off,
                nonvip_X_off,
                vip_number,
                tolerance,
                max_rep,
                np.random.Generator(np.random.PCG64(ss)),
                skip_factor_idx,
                chain_n_batches,
                max_tries,
            )
            if stuck_at < 0:
                return nonvip_genes[idx], used_counts > 0, stuck
            stuck.append(int(stuck_at))
        raise RuntimeError(
            f"Bootstrap stuck: chain {run_idx} found no matched pair in {max_tries:,} "
            f"consecutive proposals in all {max_restarts + 1} attempts (stuck after "
            f"{', '.join(f'{n:,}' for n in stuck)} of up to {target:,} control genes). "
            f"The {len(nonvip_genes)} control candidates cannot keep the factor means "
            f"within +-{tolerance:.0%} of the case means; lower --min_distance or raise "
            "--tolerance."
        )

    batch_results = Parallel(n_jobs=nthreads, backend="loky", verbose=0)(
        delayed(_run_one_chain)(i, nb) for i, nb in enumerate(chain_batches)
    )
    for i, r in enumerate(batch_results):
        if r[2]:
            print(
                f"Warning: bootstrap chain {i} got stuck {len(r[2])} time(s) (after "
                f"{', '.join(f'{n:,}' for n in r[2])} control genes) and was restarted "
                "with a new seed",
                flush=True,
            )

    np_control = np.concatenate([r[0] for r in batch_results], axis=0)
    used_mask = np.logical_or.reduce([r[1] for r in batch_results])
    used_genes = sorted(set(df_set["gene_id"].to_list()) | set(nonvip_genes[used_mask]))
    df_bootstrap_control = pl.DataFrame(np_control)
    df_bootstrap_control = df_bootstrap_control.with_columns(
        (
            pl.lit("sample_")
            + pl.arange(1, df_bootstrap_control.height + 1).cast(pl.Utf8)
        ).alias("sample_id")
    ).select("sample_id", *df_bootstrap_control.columns)

    if return_used:
        return df_set.select("gene_id"), df_bootstrap_control, used_genes
    return df_set.select("gene_id"), df_bootstrap_control


def _sniff_sep(path: str) -> str:
    """Tab if the first line of a (optionally gzipped) table has a tab, else space."""
    import gzip as _gz

    _open = _gz.open if path.endswith(".gz") else open
    with _open(path, "rt") as fh:
        return "\t" if "\t" in fh.readline() else " "


def _ordered_genes(coord_file: str, valid_genes) -> list:
    """
    Valid autosomal genes in chromosomal order, as shuffle_sweep_script_nocut.pl:
    chromosome 1..22, then 10 kb bin of the gene centre (computed on 1-based
    coordinates, ``(start + 1 + end) / 2`` from the 0-based BED), then file order.
    """
    valid_genes = set(valid_genes)
    df_coords = (
        pl.read_csv(
            coord_file,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "gene_id", "strand"],
            schema_overrides={"chrom": pl.Utf8},
        )
        .with_row_index("_line")
        .filter(pl.col("gene_id").is_in(valid_genes))
        .filter(pl.col("chrom").is_in([str(i) for i in range(1, 23)]))
        .with_columns(
            ((pl.col("start") + 1 + pl.col("end")) / 2 / 10000).floor().cast(pl.Int64).alias("bin"),
            pl.col("chrom").cast(pl.Int32).alias("_chrom_int"),
        )
    )
    ordered = df_coords.sort(["_chrom_int", "bin", "_line"])["gene_id"].to_list()
    if not ordered:
        raise ValueError("No valid genes found in coord_file")
    return ordered


def _shuffle_segments(ordered_genes: list, strong_signal_genes: set, seg_size: int) -> list:
    """
    Cut the ordered genes every ``seg_size`` genes, skipping cuts flanked on both sides
    by strong-signal genes (Perl shuffle_sweep_script_nocut.pl).
    """
    n_genes = len(ordered_genes)
    segments = []
    seg_start = 0
    for i in range(seg_size, n_genes, seg_size):
        both_strong = (
            ordered_genes[i - 1] in strong_signal_genes
            and ordered_genes[min(i, n_genes - 1)] in strong_signal_genes
        )
        if not both_strong:
            segments.append(np.arange(seg_start, i, dtype=np.int32))
            seg_start = i
    if seg_start < n_genes:
        segments.append(np.arange(seg_start, n_genes, dtype=np.int32))
    return segments


_NO_SWEEP_RANK = 10_000_000  # rank given to genes absent from a (shuffled) table

# count_sweeps_script.pl: control sets used per round, and the p-value bounds on the
# Pop_interest lines that send the count to the next round (20 -> 100 -> 5000 sets).
_PERL_DEPTH = ((20, 0.05, 0.95), (100, 0.01, 0.99), (5000, None, None))


def _read_rank_table(path: str) -> pl.DataFrame:
    """Merged rank table (gene_id + one rank column per population), all as strings."""
    return pl.read_csv(
        path, has_header=False, separator=_sniff_sep(path), infer_schema_length=0
    )


def _rank_bins(rank_table: pl.DataFrame, n_pops: int, th: np.ndarray) -> np.ndarray:
    """
    Threshold bin of every gene in the first ``n_pops`` rank columns: index of the
    smallest threshold >= rank (``len(th)`` = passes none), as simplify_zipped_sweeps.pl.
    """
    ranks = (
        rank_table.select(rank_table.columns[1 : n_pops + 1])
        .with_columns(pl.all().cast(pl.Int64).fill_null(_NO_SWEEP_RANK))
        .to_numpy()
    )
    return np.searchsorted(th, ranks, side="left").astype(np.int16)


def _strong_signal_genes(rank_table: pl.DataFrame, max_rank: int) -> set:
    """Genes ranked <= max_rank in any column (shuffle_sweep_script_nocut.pl 'bad' genes)."""
    min_rank = rank_table.select(
        pl.min_horizontal(
            [pl.col(c).cast(pl.Int64, strict=False) for c in rank_table.columns[1:]]
        )
    ).to_series()
    return set(rank_table.filter(min_rank <= max_rank)[:, 0].to_list())


def _gene_index(gene_ids: list, vip_genes, control_sets) -> tuple:
    """
    Map case genes and control sets onto row indices of ``gene_ids``.

    Case genes absent from the table are ignored (they can never be in a sweep);
    control genes absent from it raise, since that means the control sets were built
    for another gene universe.
    """
    g2i = {g: i for i, g in enumerate(gene_ids)}
    V = np.array(sorted({g2i[g] for g in vip_genes if g in g2i}), dtype=np.int32)
    flat = np.asarray(control_sets, dtype=object)
    if flat.ndim != 2:
        raise ValueError("control sets must all have the same length")
    missing = {g for g in flat.ravel() if g not in g2i}
    if missing:
        raise ValueError(
            f"{len(missing)} control genes are not in the rank table, e.g. "
            f"{sorted(missing)[:5]}"
        )
    M = np.array([[g2i[g] for g in row] for row in flat], dtype=np.int32)
    return V, M


def _count_bins(bins: np.ndarray, M: np.ndarray, V: np.ndarray, n_thr: int) -> tuple:
    """
    Sweep counts per population column of ``bins`` and threshold.

    :returns: ``vip`` (P, T) case-gene counts and ``ctrl`` (P, T, S) counts per control
        set; a control gene present k times in a set counts k times (Perl
        ``$iterated{$gene}``).
    """
    n_sets = M.shape[0]
    offs = (np.arange(n_sets, dtype=np.int64) * (n_thr + 1))[:, None]
    n_pops = bins.shape[1]
    vip = np.empty((n_pops, n_thr), dtype=np.int64)
    ctrl = np.empty((n_pops, n_thr, n_sets), dtype=np.int32)
    for k in range(n_pops):
        b = bins[:, k].astype(np.int64)
        h = np.bincount((b[M] + offs).ravel(), minlength=n_sets * (n_thr + 1))
        ctrl[k] = np.cumsum(h.reshape(n_sets, n_thr + 1)[:, :n_thr], axis=1).T
        vip[k] = np.cumsum(np.bincount(b[V], minlength=n_thr + 1)[:n_thr])
    return vip, ctrl


def _gene_neighbors(coord_file: str, gene_ids: list, valid_genes, cluster_distance: int) -> tuple:
    """
    Neighbour lists as get_all_neighbors.pl: valid genes on the same chromosome whose
    10 kb bins of the gene centre (``int((start1 + end) / 2)``, 1-based) are at most
    ``ceil(cluster_distance / 10 kb)`` bins apart, the gene itself excluded.

    :returns: CSR arrays ``(ptr, idx)`` over the rows of ``gene_ids``; ``ptr`` has one
        extra empty row, so an extra never-in-sweep row (genome shuffles) has no neighbours.
    """
    valid_genes = set(valid_genes)
    g2i = {g: i for i, g in enumerate(gene_ids)}
    df = (
        pl.read_csv(
            coord_file,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "gene_id", "strand"],
            schema_overrides={"chrom": pl.Utf8},
        )
        .filter(pl.col("gene_id").is_in(valid_genes))
        .unique("gene_id", keep="first", maintain_order=True)
        .with_columns((((pl.col("start") + 1 + pl.col("end")) // 2) // 10_000).alias("bin"))
    )
    k_max = -(-int(cluster_distance) // 10_000)
    by_bin: dict = {}
    for c, b, g in df.select("chrom", "bin", "gene_id").iter_rows():
        by_bin.setdefault((c, b), []).append(g)
    nbrs = [[] for _ in range(len(gene_ids) + 1)]
    for c, b, g in df.select("chrom", "bin", "gene_id").iter_rows():
        if g not in g2i:
            continue
        lst = nbrs[g2i[g]]
        for d in range(-k_max, k_max + 1):
            for h in by_bin.get((c, b + d), ()):
                if h != g and h in g2i:
                    lst.append(g2i[h])
    ptr = np.zeros(len(nbrs) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(x) for x in nbrs])
    idx = np.array([i for x in nbrs for i in x], dtype=np.int32)
    return ptr, idx


@njit(cache=True)
def _count_bins_clust(bins, M, V_order, nb_ptr, nb_idx, n_thr):
    """
    Sweep counts as count_sweeps_script.pl with ``Count_sweeps yes``, per population
    column of ``bins`` (threshold bin per gene, ``n_thr`` = in no sweep).

    Case genes, in ``V_order``: a gene in a sweep that is not yet done counts 1 at every
    threshold it passes and marks its in-sweep neighbours done, so they are never
    counted. Control set s, genes in set order: a gene in a sweep that is not yet done,
    together with its not-yet-done in-sweep neighbours present in the set, adds at each
    threshold the mean multiplicity in the set of those of them that pass it; all of
    them are then done. ``done`` is per population.

    :returns: ``vip`` (P, T) and ``ctrl`` (P, T, S), float64.
    """
    n_genes, n_pops = bins.shape
    n_sets, n_len = M.shape
    vip = np.zeros((n_pops, n_thr))
    ctrl = np.zeros((n_pops, n_thr, n_sets))
    done = np.zeros((n_genes, n_pops), dtype=np.bool_)
    for gi in range(V_order.shape[0]):
        g = V_order[gi]
        for p in range(n_pops):
            b = bins[g, p]
            if b >= n_thr or done[g, p]:
                continue
            done[g, p] = True
            for t in range(b, n_thr):
                vip[p, t] += 1.0
            for k in range(nb_ptr[g], nb_ptr[g + 1]):
                w = nb_idx[k]
                if bins[w, p] < n_thr:
                    done[w, p] = True
    stamp = np.zeros((n_genes, n_pops), dtype=np.int32)  # == s + 1: done in set s
    in_set = np.zeros(n_genes, dtype=np.int32)  # == s + 1: gene present in set s
    mult = np.zeros(n_genes, dtype=np.int64)
    num = np.zeros(n_thr)
    den = np.zeros(n_thr)
    for s in range(n_sets):
        tag = s + 1
        for j in range(n_len):
            g = M[s, j]
            if in_set[g] != tag:
                in_set[g] = tag
                mult[g] = 0
            mult[g] += 1
        for j in range(n_len):
            g = M[s, j]
            for p in range(n_pops):
                b = bins[g, p]
                if b >= n_thr or stamp[g, p] == tag:
                    continue
                stamp[g, p] = tag
                num[:] = 0.0
                den[:] = 0.0
                for t in range(b, n_thr):
                    num[t] += mult[g]
                    den[t] += 1.0
                for k in range(nb_ptr[g], nb_ptr[g + 1]):
                    w = nb_idx[k]
                    if in_set[w] != tag:
                        continue
                    bw = bins[w, p]
                    if bw >= n_thr or stamp[w, p] == tag:
                        continue
                    stamp[w, p] = tag
                    for t in range(bw, n_thr):
                        num[t] += mult[w]
                        den[t] += 1.0
                for t in range(n_thr):
                    if den[t] > 0.0:
                        ctrl[p, t, s] += num[t] / den[t]
    return vip, ctrl


def _case_order(gene_ids: list, vip_genes) -> np.ndarray:
    """
    Row indices of the case genes sorted by gene id: the order of Perl VIPs/file_1
    (bootstrap_test_script_nomatchomega_fast.pl writes ``sort keys %vips``), which
    decides which gene of a cluster is counted with Count_sweeps yes.
    """
    g2i = {g: i for i, g in enumerate(gene_ids)}
    return np.array([g2i[g] for g in sorted(set(vip_genes)) if g in g2i], dtype=np.int64)


def _add_scopes(vip: np.ndarray, ctrl: np.ndarray, populations: list, groups: list) -> tuple:
    """
    Append one scope per group and ``All``. As count_sweeps_script.pl
    (``set_recurrent_score``, ``control_all_scores``), a group or All count is the SUM
    of its populations' counts: a gene in a sweep in k member populations counts k times.
    Every group label gets a scope, also single-population groups.
    """
    names = list(populations)
    vips, ctrls = [vip], [ctrl]
    members: dict = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    for g, idx in members.items():
        names.append(g)
        vips.append(vip[idx].sum(axis=0, keepdims=True))
        ctrls.append(ctrl[idx].sum(axis=0, keepdims=True))
    names.append("All")
    vips.append(vip.sum(axis=0, keepdims=True))
    ctrls.append(ctrl.sum(axis=0, keepdims=True))
    return np.concatenate(vips), np.concatenate(ctrls), names


def _adaptive_depth(vip, ctrl, names: list, pop_interest: str, adaptive: bool = True) -> int:
    """
    Number of control sets count_sweeps_script.pl ends up using: 20, then 100 if a
    Pop_interest line has p <= 0.05 or >= 0.95, then up to 5000 if one has p <= 0.01 or
    >= 0.99. Pop_interest is a regular expression matched against ``"<scope>:"``.
    """
    n_total = ctrl.shape[2]
    if not adaptive:
        return n_total
    sel = [k for k, nm in enumerate(names) if re.search(pop_interest, f"{nm}:")]
    for n, lo, hi in _PERL_DEPTH:
        n_eff = min(n, n_total)
        if lo is None or not sel:
            return n_eff
        p = (ctrl[sel, :, :n_eff] > vip[sel][:, :, None]).mean(axis=2)
        if not np.any((p <= lo) | (p >= hi)):
            return n_eff
    return n_total


def _pop_counts(rank_table, vip_genes, control_sets, thresholds, populations, neighbors=None) -> tuple:
    """
    Per-population counts; ``neighbors`` = CSR from :func:`_gene_neighbors` switches to
    Count_sweeps yes (:func:`_count_bins_clust`), and ``V`` is then the case genes in
    Perl order (:func:`_case_order`).
    """
    th = np.array(sorted(int(x) for x in thresholds), dtype=np.int64)
    bins = _rank_bins(rank_table, len(populations), th)
    gene_ids = rank_table[:, 0].to_list()
    V, M = _gene_index(gene_ids, vip_genes, control_sets)
    if neighbors is None:
        vip, ctrl = _count_bins(bins, M, V, len(th))
    else:
        V = _case_order(gene_ids, vip_genes)
        vip, ctrl = _count_bins_clust(bins, M, V, neighbors[0], neighbors[1], len(th))
    return th, bins, V, M, vip, ctrl


def count_sweeps_fast(
    rank_table: pl.DataFrame,
    vip_genes: list,
    control_sets,
    thresholds: list,
    populations: list,
    groups: list,
    pop_interest: str = "All",
    adaptive: bool = True,
    neighbors: tuple | None = None,
) -> pl.DataFrame:
    """
    Enrichment curves for every population, group and ``All``, as count_sweeps_script.pl
    with ``Count_sweeps no`` (genes counted individually), or ``Count_sweeps yes`` when
    ``neighbors`` (:func:`_gene_neighbors`) is given (neighbouring genes in a sweep count
    once; see :func:`_count_bins_clust`).

    - population: case genes in a sweep (rank <= threshold) vs the same count in each
      control set (genes counted with their multiplicity in the set);
    - group / All: sums of their populations' counts;
    - control sets used: first 20, 100 or all, decided by the p-values of the
      ``pop_interest`` scope (see :func:`_adaptive_depth`); ``adaptive=False`` uses all;
    - ``ctrl_ci_lo``/``ctrl_ci_hi``: sorted control counts at ``int(0.025 n)`` and
      ``int(0.975 n)``; ``p_value``: fraction of control sets with a larger count.

    :param rank_table: Merged rank table (:func:`_read_rank_table`); rank columns in
        ``populations`` order.
    :param control_sets: Control sets, (n_sets, n_case) gene ids.
    :returns: One row per scope and threshold; ``n_sets`` is the number of control sets used.
    """
    th, _, _, _, vip, ctrl = _pop_counts(
        rank_table, vip_genes, control_sets, thresholds, populations, neighbors
    )
    vip, ctrl, names = _add_scopes(vip, ctrl, populations, groups)
    n = _adaptive_depth(vip, ctrl, names, pop_interest, adaptive)
    return _curve_frame(names, th, vip, ctrl[:, :, :n])


def _curve_frame(names: list, th: np.ndarray, vip: np.ndarray, ctrl: np.ndarray) -> pl.DataFrame:
    n = ctrl.shape[2]
    ctrl_mean = ctrl.mean(axis=2)
    srt = np.sort(ctrl, axis=2)
    ci_lo = srt[:, :, int(0.025 * n)].astype(np.float64)
    ci_hi = srt[:, :, min(int(0.975 * n), n - 1)].astype(np.float64)
    denom = ctrl_mean + 0.1
    n_thr = len(th)
    return pl.DataFrame(
        {
            "scope": np.repeat(names, n_thr),
            "threshold": np.tile(th, len(names)),
            "vip_count": vip.ravel(),
            "ctrl_mean": ctrl_mean.ravel(),
            "ctrl_ci_lo": ci_lo.ravel(),
            "ctrl_ci_hi": ci_hi.ravel(),
            "ratio": ((vip + 0.1) / denom).ravel(),
            "ci_lo_ratio": ((ci_lo + 0.1) / denom).ravel(),
            "ci_hi_ratio": ((ci_hi + 0.1) / denom).ravel(),
            "p_value": (ctrl > vip[:, :, None]).mean(axis=2).ravel(),
            "n_sets": np.full(len(names) * n_thr, n, dtype=np.int64),
        }
    )


def _all_stat(vip_all, ctrl_all, n, th, p_cutoff, min_vip_count, min_threshold, max_threshold) -> float:
    """
    estimate_FPR.pl statistic on the ``All:`` lines: sum over thresholds of
    (All score - All control mean), keeping lines with p <= p_cutoff, score >= enough
    and min_threshold <= threshold <= max_threshold.
    """
    c = ctrl_all[:, :n]
    av = c.mean(axis=1)
    p = (c > vip_all[:, None]).mean(axis=1)
    keep = (p <= p_cutoff) & (vip_all >= min_vip_count) & (th >= min_threshold) & (th <= max_threshold)
    return float(vip_all[keep].sum() - av[keep].sum())


def _draw_shuffle_block(
    segments: list, n_genes: int, rng: np.random.Generator, n: int = 8
) -> list:
    """
    One shuffle_sweep_script_nocut.pl call: every segment is mirrored to the opposite
    end of the genome with probability 1/2 (``rand(10) > 5``) once, then ``n`` shuffles
    reorder those same segments. Position i of a shuffle takes the ranks of gene ``out[i]``.
    """
    mirror = rng.random(len(segments)) * 10 > 5
    segs = [np.int32(n_genes - 1) - s if m else s for s, m in zip(segments, mirror)]
    return [np.concatenate([segs[k] for k in rng.permutation(len(segs))]) for _ in range(n)]


def _fdr_shuffles(
    rank_table: pl.DataFrame,
    coord_file: str,
    universe,
    thresholds: list,
    shuffling_segs: int,
    n_shuffles: int,
    seed: int | None,
    shuffle_blocks: int | None = None,
) -> tuple:
    """
    Genome shuffles as exdef_pipeline.pl: genes = ``universe`` (Perl: valid genes used by
    the bootstrap) in chromosomal order; a cut every ``shuffling_segs`` genes unless both
    flanking genes rank <= max(thresholds) (Perl maxi_rank, the first threshold of the
    descending list) in some population; ``n_shuffles / 8`` blocks of 8.
    ``shuffle_blocks`` = B replaces the segment size by ``int(n_genes / B)``, the
    commented-out line of shuffle_sweep_script_nocut.pl (B blocks, Enard & Petrov 2020).

    :returns: ``(ordered_genes, shuffles)`` with ``shuffles`` (n_shuffles, n_genes).
    """
    if n_shuffles % 8 != 0:
        raise ValueError("n_shuffles must be a multiple of 8")
    ordered = _ordered_genes(coord_file, universe)
    strong = _strong_signal_genes(rank_table, max(int(x) for x in thresholds))
    seg_size = len(ordered) // shuffle_blocks if shuffle_blocks else shuffling_segs
    segments = _shuffle_segments(ordered, strong, max(1, seg_size))
    rng = np.random.default_rng(seed)
    shuffles = [
        idx
        for _ in range(n_shuffles // 8)
        for idx in _draw_shuffle_block(segments, len(ordered), rng)
    ]
    return ordered, np.stack(shuffles).astype(np.int32)


def _fdr_null_chunk(
    ord_bins, ord_rows, n_rows, shuffles, M, V, th, populations, groups,
    pop_interest, adaptive, filt, neighbors=None,
) -> np.ndarray:
    """FDR null statistics for a block of shuffles (one joblib task)."""
    out = np.empty(len(shuffles))
    bins = np.full((n_rows, ord_bins.shape[1]), len(th), dtype=np.int16)
    for r, idx in enumerate(shuffles):
        bins[ord_rows] = ord_bins[idx]
        if neighbors is None:
            vip, ctrl = _count_bins(bins, M, V, len(th))
        else:  # neighbours stay genomic; only the ranks move
            vip, ctrl = _count_bins_clust(bins, M, V, neighbors[0], neighbors[1], len(th))
        vip, ctrl, names = _add_scopes(vip, ctrl, populations, groups)
        n = _adaptive_depth(vip, ctrl, names, pop_interest, adaptive)
        out[r] = _all_stat(vip[-1], ctrl[-1], n, th, *filt)
    return out


def estimate_fdr_fast(
    rank_table: pl.DataFrame,
    coord_file: str,
    shuffle_genes,
    vip_genes: list,
    control_sets,
    thresholds: list,
    populations: list,
    groups: list,
    n_shuffles: int,
    shuffling_segs: int,
    seed: int | None = None,
    nthreads: int = 1,
    pop_interest: str = "All",
    adaptive: bool = True,
    p_cutoff: float = 1.0,
    max_threshold: int | None = None,
    min_threshold: int = 0,
    min_vip_count: int = 1,
    neighbors: tuple | None = None,
    shuffle_blocks: int | None = None,
) -> pl.DataFrame:
    """
    Whole-curve FDR p-value from genome shuffles, as shuffle_sweep_script_nocut.pl +
    count_sweeps_script.pl + ``estimate_FPR.pl ... All:``.
    ``neighbors`` switches the real and shuffled counts to Count_sweeps yes;
    ``shuffle_blocks`` cuts the genome into that many blocks (see :func:`_fdr_shuffles`).

    Each shuffle is counted like the real data (same control sets, adaptive depth keyed
    on ``pop_interest``) and scored with the All-line statistic of :func:`_all_stat`.
    Shuffles are drawn in the main process and counted in ``nthreads`` blocks.

    :param shuffle_genes: Genes that are shuffled (Perl: valid genes used by the
        bootstrap, i.e. case genes and every accepted control gene); others count as
        not in a sweep in the shuffles.
    :returns: One row: ``scope`` ("All"), ``p_value`` (#null >= real / n, as Perl),
        ``n_replicates``, ``real_stat``, ``max_null_stat`` (max(0, nulls), as Perl
        ``rand_max``), ``p_value_perm`` ((#null >= real + 1) / (n + 1)).
    """
    import math

    th, bins, V, M, vip, ctrl = _pop_counts(
        rank_table, vip_genes, control_sets, thresholds, populations, neighbors
    )
    if max_threshold is None:
        max_threshold = int(th.max())
    filt = (p_cutoff, min_vip_count, min_threshold, max_threshold)
    vip_s, ctrl_s, names = _add_scopes(vip, ctrl, populations, groups)
    n_real = _adaptive_depth(vip_s, ctrl_s, names, pop_interest, adaptive)
    real_stat = _all_stat(vip_s[-1], ctrl_s[-1], n_real, th, *filt)

    ordered, shuffles = _fdr_shuffles(
        rank_table, coord_file, shuffle_genes, thresholds, shuffling_segs, n_shuffles, seed,
        shuffle_blocks,
    )
    gene_ids = rank_table[:, 0].to_list()
    g2i = {g: i for i, g in enumerate(gene_ids)}
    ord_rows = np.array([g2i.get(g, -1) for g in ordered], dtype=np.int64)
    ord_bins = np.full((len(ordered), len(populations)), len(th), dtype=np.int16)
    ord_bins[ord_rows >= 0] = bins[ord_rows[ord_rows >= 0]]
    # Shuffled genes absent from the rank table map to an extra, never-counted row.
    n_rows = len(gene_ids) + 1
    ord_rows[ord_rows < 0] = len(gene_ids)

    chunk = max(1, math.ceil(n_shuffles / max(1, nthreads)))
    null = Parallel(n_jobs=nthreads, backend="loky")(
        delayed(_fdr_null_chunk)(
            ord_bins, ord_rows, n_rows, shuffles[i : i + chunk], M, V, th,
            populations, groups, pop_interest, adaptive, filt, neighbors,
        )
        for i in range(0, n_shuffles, chunk)
    )
    null = np.concatenate(null)
    k = int(np.sum(null >= real_stat))
    return pl.DataFrame(
        {
            "scope": ["All"],
            "p_value": [k / n_shuffles],
            "n_replicates": [n_shuffles],
            "real_stat": [real_stat],
            "max_null_stat": [max(0.0, float(null.max()))],
            "p_value_perm": [(k + 1) / (n_shuffles + 1)],
        }
    )


def load_case_control(
    gene_set: str, valid_genes: list, exclude: set
) -> tuple[list, list, int]:
    """
    Split the gene universe into case and control-candidate genes.

    :param gene_set: Text file with one case gene id per line (no header). Blank lines
        are skipped; only the first whitespace-separated field of a line is used.
    :param valid_genes: Gene universe (factors ∩ sweep file genes).
    :param exclude: Gene ids removed from both sets (HLA, histones).
    :returns: ``(case, control, n_dropped)``: case = listed ids in the universe and not
        excluded; control = every other universe gene not excluded (both in
        ``valid_genes`` order); ``n_dropped`` = listed ids not kept as cases.
    """
    with open(gene_set) as fh:
        listed = {line.split()[0] for line in fh if line.strip()}
    case_set = {g for g in listed if g not in exclude} & set(valid_genes)
    case = [g for g in valid_genes if g in case_set]
    control = [g for g in valid_genes if g not in case_set and g not in exclude]
    return case, control, len(listed) - len(case)


def _check_enrichment_inputs(
    sweep_files: list, populations: list, groups: list, thresholds: list
) -> None:
    """Fail before the bootstrap if populations, groups and rank columns disagree."""
    import gzip as _gz

    if len(groups) != len(populations):
        raise ValueError(
            f"groups has {len(groups)} labels for {len(populations)} populations; "
            "give one group label per population, in the same order"
        )
    if not thresholds:
        raise ValueError("thresholds is empty")
    for f in sweep_files:
        _open_fn = _gz.open if f.endswith(".gz") else open
        with _open_fn(f, "rt") as fh:
            n_rank_cols = len(fh.readline().split()) - 1
        if n_rank_cols != len(populations):
            raise ValueError(
                f"{f} has {n_rank_cols} rank columns but {len(populations)} "
                "populations were given; rank columns are read by position"
            )


def _load_bootstrap(folder: str, case_genes: list, universe: set) -> tuple:
    """Load case_set.txt / control_set.txt written by a previous enrichment run."""
    df_case = pl.read_csv(os.path.join(folder, "case_set.txt"))
    df_control = pl.read_csv(os.path.join(folder, "control_set.txt"), infer_schema_length=0)
    case = df_case["gene_id"].to_list()
    extra = sorted(set(case) - set(case_genes))
    if extra:
        raise ValueError(
            f"{folder}/case_set.txt has {len(extra)} genes that are not case genes of "
            f"this gene set (e.g. {extra[:5]}); rerun the bootstrap"
        )
    width = df_control.width - 1
    if width != len(case):
        raise ValueError(
            f"control sets have {width} genes but case_set.txt has {len(case)}"
        )
    ctrl_genes = set(df_control.drop("sample_id").to_numpy().ravel())
    outside = sorted(ctrl_genes - universe)
    if outside:
        raise ValueError(
            f"{len(outside)} control genes are outside this gene universe "
            f"(e.g. {outside[:5]}); rerun the bootstrap"
        )
    used_path = os.path.join(folder, "used_genes.txt")
    if os.path.isfile(used_path):
        with open(used_path) as fh:
            used = sorted({ln.split()[0] for ln in fh if ln.strip()})
    else:
        used = sorted(set(case) | ctrl_genes)
        print(
            f"{used_path} not found: shuffling case + control-set genes ({len(used)}); "
            "genes accepted by the sampler but absent from every set are missed",
            flush=True,
        )
    print(
        f"Reusing bootstrap from {folder}: {len(case)} case genes, "
        f"{df_control.height} control sets, {len(used)} used genes",
        flush=True,
    )
    return df_case, df_control, used


def run_enrichment(
    sweep_files: list,
    gene_set: str,
    factors_file: str,
    annotation_file: str,
    populations: list,
    groups: list,
    thresholds: list,
    pop_interest: str = "All",
    cluster_distance: int = 500_000,
    n_control_sets: int = 1000,
    tolerance: float = 0.05,
    min_distance: int = 1_250_000,
    flip: bool = False,
    max_rep: int = 25,
    nthreads: int = 1,
    n_shuffles: int = 8,
    shuffling_segs: int = 2,
    bootstrap_dir: str = None,
    distance_file: str = None,
    seed: int | None = None,
    reuse_bootstrap: str | None = None,
    output_folder: str | None = None,
    count_sweeps: bool = False,
    shuffle_blocks: int | None = None,
) -> tuple:
    """
    Run the full gene-set sweep enrichment pipeline.

    Steps:

    1. Derive the gene universe (factors ∩ sweep_files[0]) and split it into
       case genes (listed in ``gene_set``) and control candidates (the rest).
    2. Bootstrap control sets matched on confounding factors
       (or load pre-computed sets from ``bootstrap_dir``).
    3. For each sweep file: count sweep overlaps, shuffle genome for null
       distribution, and estimate FDR.

    :param sweep_files: Paths to sweep rank files (gene_id + per-population
        rank columns, tab- or space-separated, optionally gzipped).
    :param gene_set: Text file with one case (VIP) gene id per line, no header.
        Control candidates are all other genes of the universe
        (see :func:`load_case_control`).
    :param factors_file: TSV confounding factors file (gene_id + factor
        columns, no header).
    :param annotation_file: BED gene coordinates file (0-based, no header).
    :param populations: Population codes matching sweep file column order.
    :param groups: Group label per population (same length as populations).
    :param thresholds: Rank cutoffs for enrichment curve (e.g. [6000, ..., 20]).
    :param pop_interest: Perl ``Pop_interest``: regular expression on the scope labels
        (population, group or ``All``) whose p-values decide how many control sets are
        used, for the real data and for every shuffle (20, 100 or all; see
        :func:`_adaptive_depth`). The FDR statistic always uses the ``All`` scope
        (``estimate_FPR.pl ... All:``).
    :param cluster_distance: Neighbour distance (bp) for ``count_sweeps``
        (get_all_neighbors.pl, 10 kb bins of the gene centre); unused otherwise.
    :param n_control_sets: Number of matched control sets (multiple of 10; Perl
        Runs_number x Iterations_number x 10 = 1000). The sampling is split into one
        chain per thread, so the sets depend on ``seed`` and ``nthreads``.
    :param tolerance: Allowed ± fraction deviation in factor averages for
        control gene matching.
    :param min_distance: Minimum bp distance from VIPs for control eligibility.
    :param flip: Flip test direction when ``len(VIPs) > len(controls)`` (increases power).
    :param max_rep: Max average resamples per control gene across bootstrap sets.
    :param nthreads: Parallel workers (joblib).
    :param n_shuffles: FDR shuffle replicates (must be a multiple of 8).
    :param shuffling_segs: Genes per genomic shuffle segment.
    :param shuffle_blocks: If given, cut the shuffled genome into this many blocks of
        ``int(n_genes / shuffle_blocks)`` genes instead of ``shuffling_segs``-gene
        segments (the commented-out Perl line; 100 blocks in Enard & Petrov 2020).
    :param count_sweeps: Perl ``Count_sweeps yes``: neighbouring genes in the same sweep
        (within ``cluster_distance``) count once, in the real data and in every shuffle
        (see :func:`_count_bins_clust`). Default: every gene counts (``Count_sweeps no``).
    :param bootstrap_dir: Folder with pre-computed ``VIPs/`` and ``nonVIPs/``
        sub-dirs. When non-empty, the bootstrap step is skipped.
    :param distance_file: Optional pre-computed distance TSV (gene_id \\t distance,
        no header). Passed to ``run_bootstrap_nomatchomega`` to bypass internal
        distance computation and match Perl's control pool exactly.
    :param seed: Seeds the bootstrap and the genome shuffles. If None, one is drawn
        and printed so the run can be repeated.
    :param reuse_bootstrap: Output folder of a previous run; its ``case_set.txt`` and
        ``control_set.txt`` are loaded instead of running the bootstrap (same gene set
        and universe required).
    :param output_folder: If given, write ``case_set.txt``, ``control_set.txt``,
        ``used_genes.txt`` (genome-shuffle universe), ``enrichment_curves.txt`` and
        ``fdr.txt`` there.
    :returns: ``(df_case, df_control, df_curves, df_fdr)``: case genes, control sets,
        enrichment curves (every population, group and ``All``) and one FDR row per
        sweep file.
    """
    import gzip as _gz

    _check_enrichment_inputs(sweep_files, populations, groups, thresholds)
    if n_shuffles % 8 != 0:
        raise ValueError("n_shuffles must be a multiple of 8")
    if n_control_sets < 10 or n_control_sets % 10 != 0:
        raise ValueError("n_control_sets must be a positive multiple of 10")
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    print(f"Seed: {seed}", flush=True)
    _t_start = time.time()

    # --- Derive valid universe from factors ∩ sweep_files[0] ---
    _exclude = set(hla_genes + hist_genes)

    with open(factors_file, "rt") as _fh:
        _fsep = "\t" if "\t" in _fh.readline() else " "
    _factors_ids = set(
        pl.read_csv(
            factors_file, has_header=False, separator=_fsep, infer_schema_length=0
        )[:, 0].to_list()
    )
    _sf0 = sweep_files[0]
    _open_fn = _gz.open if _sf0.endswith(".gz") else open
    with _open_fn(_sf0, "rt") as _fh:
        _sep0 = "\t" if "\t" in _fh.readline() else " "
    _sweep_ids = set(
        pl.read_csv(_sf0, has_header=False, separator=_sep0, infer_schema_length=0)[
            :, 0
        ].to_list()
    )
    _valid_genes_list = sorted(_factors_ids & _sweep_ids)

    _case_genes, _control_genes, _n_dropped = load_case_control(
        gene_set, _valid_genes_list, _exclude
    )
    with open(gene_set) as _fh:  # distances are measured from every listed case gene
        _listed_genes = sorted({_l.split()[0] for _l in _fh if _l.strip()})
    print(
        f"Gene set: {len(_case_genes)} case genes, {len(_control_genes)} control "
        f"candidates ({_n_dropped} listed ids outside the universe or excluded)",
        flush=True,
    )
    # --- Step 6: Bootstrap ---
    if bootstrap_dir is not None:
        _vip_file = os.path.join(bootstrap_dir, "VIPs", "file_1")
        _ctrl_file = os.path.join(bootstrap_dir, "nonVIPs", "file_1")
        with open(_vip_file) as _fh:
            _vip_list_raw = [l.strip() for l in _fh if l.strip()]
        df_case = pl.DataFrame({"gene_id": _vip_list_raw})
        _ctrl_sets_raw = []
        with open(_ctrl_file) as _fh:
            for _line in _fh:
                _parts = _line.strip().split()
                if _parts:
                    _ctrl_sets_raw.append(_parts[1:])  # drop sample_N prefix
        _n_ctrl_genes = max(len(r) for r in _ctrl_sets_raw) if _ctrl_sets_raw else 0
        df_control = pl.DataFrame(
            {"sample_id": [f"sample_{i + 1}" for i in range(len(_ctrl_sets_raw))]}
            | {
                f"gene_{j}": [(r[j] if j < len(r) else None) for r in _ctrl_sets_raw]
                for j in range(_n_ctrl_genes)
            }
        )
        _used_path = os.path.join(bootstrap_dir, "used_genes.txt")
        if os.path.isfile(_used_path):
            with open(_used_path) as _fh:
                _used_genes = sorted({l.split()[0] for l in _fh if l.strip()})
        else:
            _used_genes = sorted(
                set(_vip_list_raw) | {g for r in _ctrl_sets_raw for g in r}
            )
        print(f"Bootstrap: loaded from {bootstrap_dir}", flush=True)
    elif reuse_bootstrap is not None:
        df_case, df_control, _used_genes = _load_bootstrap(
            reuse_bootstrap, _case_genes, set(_valid_genes_list)
        )

    else:
        df_case, df_control, _used_genes = run_bootstrap_nomatchomega(
            case_genes=_case_genes,
            factors_file=factors_file,
            annotation_file=annotation_file,
            runs_number=n_control_sets // 10,
            n_batches=1,  # chains of 10-set batches are laid out over nthreads
            tolerance=tolerance,
            min_dist=min_distance,
            flip=flip,
            max_rep=max_rep,
            nthreads=nthreads,
            control_genes=_control_genes,
            distance_file=distance_file,
            distance_genes=_listed_genes,
            seed=seed,
            return_used=True,
        )


    _t_post_bootstrap = time.time()

    # --- Steps 7-9: curves, genome shuffles and FDR (vectorized) ---
    _vip_list = df_case["gene_id"].to_list()
    _ctrl_sets = df_control.drop("sample_id").to_numpy()
    # Perl shuffles the valid genes listed in used_genes.txt (case + accepted controls).
    _shuffle_genes = sorted(set(_used_genes) & set(_valid_genes_list))

    fdr_results = []
    curve_results = []
    for _rank_file in sweep_files:
        _label = os.path.basename(_rank_file).rsplit("_", 1)[-1]
        _rank_table = _read_rank_table(_rank_file)
        _neighbors = None
        if count_sweeps:
            _neighbors = _gene_neighbors(
                annotation_file, _rank_table[:, 0].to_list(), _valid_genes_list,
                cluster_distance,
            )
            print(
                f"Count_sweeps yes: neighbours within {cluster_distance:,} bp "
                f"(mean {len(_neighbors[1]) / max(1, len(_valid_genes_list)):.1f} per gene)",
                flush=True,
            )
        results = count_sweeps_fast(
            _rank_table, _vip_list, _ctrl_sets, thresholds, populations, groups,
            pop_interest=pop_interest, neighbors=_neighbors,
        )
        fdr = estimate_fdr_fast(
            _rank_table,
            annotation_file,
            _shuffle_genes,
            _vip_list,
            _ctrl_sets,
            thresholds,
            populations,
            groups,
            n_shuffles=n_shuffles,
            shuffling_segs=shuffling_segs,
            seed=seed,
            nthreads=nthreads,
            pop_interest=pop_interest,
            p_cutoff=1.0,  # matches Perl estimate_FPR.pl call in the pipeline
            max_threshold=max(thresholds),
            min_threshold=0,
            min_vip_count=1,
            neighbors=_neighbors,
            shuffle_blocks=shuffle_blocks,
        )
        row = fdr.row(0, named=True)
        print(
            f"FDR {_label}: p={row['p_value']:.4f} (perm {row['p_value_perm']:.4f}), "
            f"{row['n_replicates']} shuffles of {len(_shuffle_genes)} genes, "
            f"real={row['real_stat']:.2f} max_null={row['max_null_stat']:.2f}",
            flush=True,
        )
        curve_results.append(results.with_columns(pl.lit(_rank_file).alias("dataset")))
        fdr_results.append(fdr.with_columns(pl.lit(_rank_file).alias("dataset")))

    df_curves, df_fdr = pl.concat(curve_results), pl.concat(fdr_results)
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        df_case.write_csv(os.path.join(output_folder, "case_set.txt"))
        df_control.write_csv(os.path.join(output_folder, "control_set.txt"))
        with open(os.path.join(output_folder, "used_genes.txt"), "w") as _fh:
            _fh.write("\n".join(_used_genes) + "\n")
        df_curves.write_csv(os.path.join(output_folder, "enrichment_curves.txt"))
        df_fdr.write_csv(os.path.join(output_folder, "fdr.txt"))

    print(
        f"Time: bootstrap {_t_post_bootstrap - _t_start:.1f} s, curves + FDR "
        f"{time.time() - _t_post_bootstrap:.1f} s, total {time.time() - _t_start:.1f} s",
        flush=True,
    )
    return df_case, df_control, df_curves, df_fdr



def _ratio_ci(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add the 95% interval of the enrichment ratio as ``CI_low_ratio``/``CI_up_ratio``.

    The case count is held fixed and divided by the 2.5/97.5% percentiles of the
    control-set counts, with the same +0.1 smoothing as ``ratio``:
    ``(vip + 0.1) / (ctrl_ci_hi + 0.1)`` to ``(vip + 0.1) / (ctrl_ci_lo + 0.1)``
    (the high control percentile gives the low ratio bound). The interval reflects
    variation across matched control sets, not resampling of the case genes.
    """
    num = pl.col("vip_count") + 0.1
    return df.with_columns(
        (num / (pl.col("ctrl_ci_hi") + 0.1)).alias("CI_low_ratio"),
        (num / (pl.col("ctrl_ci_lo") + 0.1)).alias("CI_up_ratio"),
    )


def _curve_svg_path(output_folder: str, dataset: str) -> str:
    """Enrichment-curve SVG path for one sweep file (``dataset`` is its full path)."""
    os.makedirs(output_folder, exist_ok=True)
    stem = os.path.basename(dataset)
    for ext in (".gz", ".txt", ".tsv"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return os.path.join(output_folder, f"enrichment_curves_{stem}.svg")


def plot_enrichment(
    curves: "str | pl.DataFrame",
    scopes: list[str] | None = None,
    facet: bool = False,
    ci: bool = False,
    title: str = "",
    output_folder: str | None = None,
    show: bool = False,
) -> dict:
    """
    Plot enrichment curves (case/control ratio against rank threshold).

    :param curves: ``enrichment_curves.txt`` written by ``flexsweep enrichment``, or the
        curves DataFrame returned by :func:`run_enrichment`.
    :param scopes: Scopes to draw (population codes, group labels or ``All``).
        Default: every scope in the table.
    :param facet: One panel per scope instead of one panel with all lines.
    :param ci: Shade the 95% interval of the ratio across matched control sets
        (see :func:`_ratio_ci`).
    :param title: Figure title.
    :param output_folder: If given, save one SVG per sweep file as
        ``enrichment_curves_<sweep file name>.svg``.
    :param show: Call ``matplotlib.pyplot.show()`` after plotting.
    :returns: ``{dataset: matplotlib Figure}``, one figure per sweep file.
    :raises ValueError: If a requested scope is not in the table.
    """
    import matplotlib.pyplot as plt

    df_curves = curves if isinstance(curves, pl.DataFrame) else pl.read_csv(curves)
    if scopes is not None:
        available = set(df_curves["scope"].cast(pl.Utf8).unique().to_list())
        unknown = [sc for sc in scopes if sc not in available]
        if unknown:
            raise ValueError(
                f"scopes {unknown} not in the curves table; available: {sorted(available)}"
            )

    figs = {}
    for dataset in sorted(df_curves["dataset"].unique().to_list()):
        figs[dataset] = _plot_curves(
            df_curves.filter(pl.col("dataset") == dataset),
            plot_title=title,
            ci=ci,
            population=scopes,
            facet=facet,
        )
        if output_folder is not None:
            figs[dataset].savefig(_curve_svg_path(output_folder, dataset))
    if show:
        plt.show()
    return figs


_NPG_COLORS = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
]

def _plot_curves(
    df_curves,
    plot_title: str = "",
    ci: bool = False,
    population: list[str] | None = None,
    target_groups: list[str] | None = None,
    facet: bool = False,
):
    import matplotlib.pyplot as plt

    df = df_curves.with_columns(
        [
            pl.col("ratio").cast(pl.Float64, strict=False),
            pl.col("threshold").cast(pl.Float64, strict=False),
            pl.col("ctrl_ci_lo").cast(pl.Float64, strict=False),
            pl.col("ctrl_ci_hi").cast(pl.Float64, strict=False),
            pl.col("vip_count").cast(pl.Float64, strict=False),
        ]
    )

    filters = [
        pl.col("threshold").is_not_null(),
        pl.col("ratio") >= 0,
    ]

    if population is not None:
        filters.append(pl.col("scope").is_in(population))

    if target_groups is not None:
        filters.append(pl.col("scope").is_in(target_groups))

    df = df.filter(pl.all_horizontal(filters))

    df = _ratio_ci(df)

    df = df.sort("threshold")

    threshold_levels = sorted(df["threshold"].unique().to_list())
    threshold_labels = [
        str(int(t)) if t == int(t) else str(t) for t in threshold_levels
    ]
    label_to_x = {lbl: i for i, lbl in enumerate(threshold_labels)}

    df = df.with_columns(
        pl.col("threshold")
        .map_elements(
            lambda v: label_to_x[str(int(v)) if v == int(v) else str(v)],
            return_dtype=pl.Int32,
        )
        .alias("x_pos")
    )

    scopes = df["scope"].unique().sort().to_list()
    color_map = {
        scope: _NPG_COLORS[i % len(_NPG_COLORS)] for i, scope in enumerate(scopes)
    }

    if facet:
        pops = scopes
        ncols = min(3, len(pops))
        nrows = int(np.ceil(len(pops) / ncols))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
        )
        ax_flat = axes.flatten()

        for idx, pop in enumerate(pops):
            ax = ax_flat[idx]
            pop_df = df.filter(pl.col("scope") == pop)

            _draw_lines(ax, pop_df, scopes, color_map, ci)
            _style_ax(ax, threshold_labels, subtitle=pop)

        for idx in range(len(pops), len(ax_flat)):
            ax_flat[idx].set_visible(False)

        handles = [
            plt.Line2D(
                [0], [0], color=color_map[s], linewidth=2, marker="o", label=s
            )
            for s in scopes
        ]
        # one row below the panels, so it never covers a curve
        fig.legend(
            handles=handles,
            title="Population",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(handles),
            frameon=False,
        )

    else:
        fig, ax = plt.subplots(figsize=(9, 5))

        _draw_lines(ax, df, scopes, color_map, ci)
        _style_ax(ax, threshold_labels)

        handles = [
            plt.Line2D(
                [0], [0], color=color_map[s], linewidth=2, marker="o", label=s
            )
            for s in scopes
        ]
        ax.legend(handles=handles, title="Population", frameon=False)

    fig.suptitle(plot_title, fontsize=14, fontweight="bold", y=1.01)
    fig.text(
        0.5,
        0.98,
        "Rank Threshold vs. Enrichment Ratio",
        ha="center",
        fontsize=10,
        color="grey",
        transform=fig.transFigure,
    )

    # facet: keep ~0.8 in at the bottom for the legend row
    fig.tight_layout(rect=(0, 0.8 / fig.get_figheight() if facet else 0, 1, 1))
    return fig

def _draw_lines(ax, df, scopes, color_map, ci):
    """Plot one line per scope onto *ax*."""
    ax.set_yscale("log")
    ax.axhline(y=1, linestyle="--", color="grey", linewidth=0.8, zorder=0)

    for scope in scopes:
        sub = df.filter(pl.col("scope") == scope).sort("x_pos")
        if sub.is_empty():
            continue
        x = sub["x_pos"].to_list()
        y = sub["ratio"].to_list()
        color = color_map[scope]

        ax.plot(
            x, y, color=color, linewidth=1.5, marker="o", markersize=4, label=scope
        )

        if ci:
            y_lo = sub["CI_low_ratio"].to_list()
            y_hi = sub["CI_up_ratio"].to_list()
            ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.15, linewidth=0)

def _style_ax(ax, threshold_labels, subtitle=None):
    """Apply common axis styling."""
    import matplotlib.ticker as ticker

    ax.set_xticks(range(len(threshold_labels)))
    ax.set_xticklabels(threshold_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Rank Threshold")
    ax.set_ylabel("Enrichment Ratio (Obs/Exp)")

    ax.yaxis.set_major_locator(
        ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=20)
    )
    # ScalarFormatter: shows "10" instead of "10^1"
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # # Optional: Minor ticks for visual density without labels
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs="auto", numticks=20)
    )
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    ax.spines[["top", "right"]].set_visible(False)
    if subtitle:
        ax.set_title(subtitle, fontsize=10)


thresholds = [
    6000,
    5000,
    4000,
    3000,
    2500,
    2000,
    1500,
    1000,
    900,
    800,
    700,
    600,
    500,
    450,
    400,
    350,
    300,
    250,
    200,
    150,
    100,
    50,
    20,
]


# hla_genes = [  # original expanded list — DO NOT restore
#     "ENSG00000233095", "ENSG00000223980", "ENSG00000230254", "ENSG00000204642",
#     "ENSG00000204632", "ENSG00000206503", "ENSG00000204592", "ENSG00000114455",
#     "ENSG00000235220", "ENSG00000235346", "ENSG00000235657", "ENSG00000229252",
#     "ENSG00000228299", "ENSG00000228987", "ENSG00000236884", "ENSG00000236418",
#     "ENSG00000233209", "ENSG00000223793", "ENSG00000228813", "ENSG00000243496",
#     "ENSG00000226264", "ENSG00000241394", "ENSG00000235744", "ENSG00000229685",
#     "ENSG00000237710", "ENSG00000206435", "ENSG00000228964", "ENSG00000234794",
#     "ENSG00000227357", "ENSG00000228080", "ENSG00000232062", "ENSG00000231939",
#     "ENSG00000231526", "ENSG00000226165", "ENSG00000239457", "ENSG00000241296",
#     "ENSG00000243215", "ENSG00000232957", "ENSG00000236177", "ENSG00000226826",
#     "ENSG00000237508", "ENSG00000226260", "ENSG00000227826", "ENSG00000229074",
#     "ENSG00000225890", "ENSG00000235680", "ENSG00000225824", "ENSG00000231834",
#     "ENSG00000231823", "ENSG00000229493", "ENSG00000239329", "ENSG00000243189",
#     "ENSG00000231558", "ENSG00000228163", "ENSG00000236632", "ENSG00000229295",
#     "ENSG00000237022", "ENSG00000224608", "ENSG00000229698", "ENSG00000132297",
#     "ENSG00000206509", "ENSG00000206506", "ENSG00000206505", "ENSG00000206493",
#     "ENSG00000206452", "ENSG00000206450", "ENSG00000204525", "ENSG00000234745",
#     "ENSG00000232962", "ENSG00000224103", "ENSG00000230708", "ENSG00000206308",
#     "ENSG00000196101", "ENSG00000206306", "ENSG00000206305", "ENSG00000206302",
#     "ENSG00000225103", "ENSG00000196610", "ENSG00000241674", "ENSG00000242685",
#     "ENSG00000206292", "ENSG00000206291", "ENSG00000215048", "ENSG00000233841",
#     "ENSG00000223532", "ENSG00000204287", "ENSG00000198502", "ENSG00000196126",
#     "ENSG00000196735", "ENSG00000179344", "ENSG00000237541", "ENSG00000232629",
#     "ENSG00000241106", "ENSG00000137403", "ENSG00000230413", "ENSG00000224320",
#     "ENSG00000197568", "ENSG00000242574", "ENSG00000204257", "ENSG00000204252",
#     "ENSG00000231389", "ENSG00000223865", "ENSG00000233904", "ENSG00000234487",
#     "ENSG00000237216", "ENSG00000229215", "ENSG00000227715", "ENSG00000230726",
#     "ENSG00000231021", "ENSG00000228284", "ENSG00000231286", "ENSG00000225201",
#     "ENSG00000206301", "ENSG00000228254", "ENSG00000241910", "ENSG00000242386",
#     "ENSG00000239463", "ENSG00000230141", "ENSG00000168384", "ENSG00000230763",
#     "ENSG00000230463", "ENSG00000233192", "ENSG00000230675", "ENSG00000227993",
#     "ENSG00000243612", "ENSG00000231679", "ENSG00000206240", "ENSG00000206237",
#     "ENSG00000204276", "ENSG00000224305", "ENSG00000241386", "ENSG00000225691",
#     "ENSG00000242092", "ENSG00000242361", "ENSG00000235844", "ENSG00000236693",
#     "ENSG00000232126", "ENSG00000234154", "ENSG00000243719",
# ]
#
# hist_genes = [  # original histone list — DO NOT restore
#     "ENSG00000164508", "ENSG00000146047", "ENSG00000124610", "ENSG00000198366",
#     "ENSG00000196176", "ENSG00000124529", "ENSG00000124693", "ENSG00000137259",
#     "ENSG00000196226", "ENSG00000196532", "ENSG00000187837", "ENSG00000197061",
#     "ENSG00000187475", "ENSG00000180596", "ENSG00000180573", "ENSG00000168298",
#     "ENSG00000158373", "ENSG00000197697", "ENSG00000188987", "ENSG00000197409",
#     "ENSG00000196866", "ENSG00000197846", "ENSG00000198518", "ENSG00000187990",
#     "ENSG00000168274", "ENSG00000196966", "ENSG00000124575", "ENSG00000198327",
#     "ENSG00000124578", "ENSG00000256316", "ENSG00000197459", "ENSG00000256018",
#     "ENSG00000168242", "ENSG00000158406", "ENSG00000124635", "ENSG00000196787",
#     "ENSG00000197903", "ENSG00000198339", "ENSG00000184825", "ENSG00000185130",
#     "ENSG00000196747", "ENSG00000203813", "ENSG00000182611", "ENSG00000196374",
#     "ENSG00000197238", "ENSG00000197914", "ENSG00000184348", "ENSG00000233822",
#     "ENSG00000198374", "ENSG00000184357", "ENSG00000182572", "ENSG00000198558",
#     "ENSG00000197153", "ENSG00000233224", "ENSG00000196331", "ENSG00000168148",
#     "ENSG00000181218", "ENSG00000196890", "ENSG00000203818", "ENSG00000203814",
#     "ENSG00000183598", "ENSG00000183941", "ENSG00000203811", "ENSG00000183558",
#     "ENSG00000203812", "ENSG00000203852", "ENSG00000182217", "ENSG00000184678",
#     "ENSG00000184260", "ENSG00000184270", "ENSG00000197837", "ENSG00000265232",
#     "ENSG00000263376", "ENSG00000265133", "ENSG00000265198", "ENSG00000266225",
#     "ENSG00000266725", "ENSG00000264719", "ENSG00000263521", "ENSG00000263965",
# ]

hla_genes = [
    "ENSG00000179344",
    "ENSG00000196126",
    "ENSG00000196735",
    "ENSG00000198502",
    "ENSG00000204252",
    "ENSG00000204257",
    "ENSG00000204287",
    "ENSG00000204525",
    "ENSG00000204592",
    "ENSG00000204632",
    "ENSG00000204642",
    "ENSG00000206503",
    "ENSG00000223865",
    "ENSG00000231389",
    "ENSG00000232629",
    "ENSG00000234745",
    "ENSG00000237541",
    "ENSG00000241106",
    "ENSG00000242574",
]

hist_genes = []
