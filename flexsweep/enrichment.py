import math
import os
import time
from typing import NamedTuple

from numba import njit

from . import Parallel, delayed, np, pl


_NO_CASE_DISTANCE = 100_000_000_000  # Perl distance_file value off case-gene chromosomes


def _log(msg: str) -> None:
    """One timestamped line of the enrichment log, written at once."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


# ---- sampler ---------------------------------------------------------------------
# Search limits, in proposals (blocks of 3 genes). First genes come from the whole pool until
# proposals x genes accepted once > pool_stage, then for reuse_stage proposals from the genes
# already accepted (after list_switch: accepted three times), and so on. Target factors are
# learned over ``learn`` proposals (0: all are used) and relearned every ``relearn``.
_LIMITS = (10_000_000, 150_000, 200_000, 200_000, 20_000_000)
_MAX_PROPOSALS = 180_000_000  # per block, before the chain tries larger blocks
_N_LEVELS = 64  # value levels per constraint in the gene bitsets
_N_SCREEN = 8  # constraints screened with bitsets before the exact check
_TOP_FRAC = 10  # genes 2 and 3 come from the best 1 / _TOP_FRAC of the pool for a factor
_N_PILOT = 128  # pairs that set M before a block is accepted from a screened pair
_MAX_BLOCK = 5


def _pack_bits(mask: np.ndarray) -> np.ndarray:
    """Boolean (..., genes) to uint64 words: gene g is bit g % 64 of word g // 64."""
    pad = np.zeros(mask.shape[:-1] + (-mask.shape[-1] % 64,), dtype=bool)
    packed = np.packbits(np.concatenate([mask, pad], axis=-1), axis=-1, bitorder="little")
    return np.ascontiguousarray(packed).view(np.uint64)


def _match_tables(vip_X, nonvip_X, tolerance: float) -> tuple:
    """
    Matching rule as arrays. Every constraint is a factor, limited on the side of the pool
    mean, in units of the factor's pool sd and signed so that larger is safer. With D[g, c] = value of gene g minus
    the limit, a block fits a chain when, for every constraint c,

        slack[c] + sum over the block of D[g, c] >= 0
        slack = n (chain mean - limit) + fake (case mean - limit)

    which is the running-mean test with its ``fake`` case-mean pseudo-genes, rearranged.

    :returns: D (genes, constraints); seed = case mean - limit; order (factors, top genes),
        best genes of each factor, ties in pool order; their bitsets; bitsets of
        D >= level, per constraint; first level and 1 / step; ra, rb with
        need * ra / r + rb = how far a need is, as a share of the r best single genes;
        target factors (those a gene can help).
    """
    avg, pool_avg = vip_X.mean(axis=0), nonvip_X.mean(axis=0)
    sgn = np.where(avg > pool_avg, 1.0, -1.0)
    sd = nonvip_X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    low, high = (1.0 - tolerance) * avg, (1.0 + tolerance) * avg
    lim = np.where(sgn > 0, low, high)
    D = sgn * (nonvip_X - lim) / sd
    seed = sgn * (avg - lim) / sd

    signed = sgn * nonvip_X
    n_top = max(1, len(nonvip_X) // _TOP_FRAC)
    order = np.argsort(-signed, axis=0, kind="stable").T[:, :n_top].astype(np.int32)
    in_top = np.zeros(signed.T.shape, dtype=bool)
    np.put_along_axis(in_top, order.astype(np.int64), True, axis=1)

    best = signed.max(axis=0)
    unit = np.where(best > 0, best, 1.0)
    d_min, d_max = D.min(axis=0), D.max(axis=0)
    step = np.where(d_max > d_min, (d_max - d_min) / _N_LEVELS, 1.0)
    levels = d_min[:, None] + step[:, None] * (np.arange(_N_LEVELS) - 1e-9)  # supersets
    level_bits = np.stack(  # one constraint at a time: no (constraints, levels, genes) mask
        [_pack_bits(D[:, c] >= levels[c][:, None]) for c in range(D.shape[1])]
    )
    return (
        np.ascontiguousarray(D), seed, np.ascontiguousarray(order), _pack_bits(in_top),
        level_bits, d_min, 1.0 / step, sd / unit, sgn * lim / unit, np.flatnonzero(best > 0),
    )


@njit(cache=True)
def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(cache=True)
def _furthest(Dc, i, j, need_c, scale, ra_c, rb_c, n_cand):
    """Target factor (position) furthest from its need after genes i and j (j < 0: only i)."""
    pos, worst = n_cand, -1e99
    for c in range(n_cand):
        d = Dc[i, c] + Dc[j, c] if j >= 0 else Dc[i, c]
        z = scale * (need_c[c] - d) * ra_c[c] + rb_c[c]
        up = z > worst
        worst = z if up else worst
        pos = c if up else pos
    return pos


@njit(cache=True)
def _propose(
    m, src, n_src, rng, order, Dc, cand, n_cand, need_c, ra_c, rb_c, D, need, check, rejects,
    used, cap, memo, stamp, window, hits, out,
):
    """
    Up to ``m`` random blocks of 3: gene 1 from ``src``, genes 2 and 3 from the best genes of
    the factor furthest off. Stops at the first one that fits (left in ``out``).
    Returns (blocks tried, fitted).
    """
    I = rng.integers(0, n_src, m)
    J = rng.integers(0, order.shape[1], m)
    K = rng.integers(0, order.shape[1], m)
    for p in range(m):
        i = src[I[p]]
        if stamp[i] != window:  # target after gene i: same until the chain changes
            memo[i] = cand[_furthest(Dc, i, -1, need_c, 0.5, ra_c, rb_c, n_cand)]
            stamp[i] = window
        j = order[memo[i], J[p]]
        k3 = cand[_furthest(Dc, i, j, need_c, 1.0, ra_c, rb_c, n_cand)]
        hits[memo[i]] += 1
        hits[k3] += 1
        g = order[k3, K[p]]
        ok = True
        for c in check:
            if D[i, c] + D[j, c] + D[g, c] < need[c]:
                rejects[c] += 1
                ok = False
                break
        if ok and used[i] <= cap and used[j] <= cap and used[g] <= cap:
            out[0], out[1], out[2] = i, j, g
            return p + 1, True
    return m, False


@njit(cache=True)
def batch_control_chain(
    D, seed, order, top_bits, level_bits, d_min, inv_step, ra, rb, targets,
    vip_number, max_rep, rng, n_batches, max_proposals, limits=_LIMITS,
):
    """
    One chain of matched control genes (tables from :func:`_match_tables`).

    Blocks of 3 genes are added while the chain mean stays inside the limits and no gene was
    used more than max_rep * (n + 1) / vip_number times. A block is drawn as in
    bootstrap_match_b3_repair.c (its proposal law and stages), without simulating proposals
    one by one once they get rare: for a pair (gene 1, gene 2), ALL candidates for the last
    gene are screened at once with bitset ANDs, and the pair is accepted with probability
    (number that fit) / M, which is the law of one-by-one rejection as long as no pair has
    more than M fits. M is twice the largest number among _N_PILOT discarded pairs (at least
    16) and doubles if a later pair exceeds it. A pair counts as (candidates / M) proposals,
    so the stage limits are unchanged.

    The limits are never loosened: after max_proposals the block grows (4, then 5 genes), and
    the chain gives up if that does not help.

    :returns: (chain, ends, used, stuck_at): accepted pool indices; chain length at the end of
        each batch; accepted count per gene; -1, or the length where the chain gave up.
    """
    pool_limit, reuse_limit, list_switch, learn, relearn = limits
    n_pool, n_con = D.shape
    n_fac, n_top = order.shape
    n_words = top_bits.shape[1]
    one = np.uint64(1)
    chain = np.empty(100 + 11 * vip_number * n_batches + 2 * _MAX_BLOCK, np.int32)
    ends = np.zeros(n_batches, np.int64)
    used = np.zeros(n_pool, np.int32)

    everyone = np.arange(n_pool).astype(np.int32)
    n_acc = np.zeros(n_pool, np.int32)
    once = np.empty(n_pool, np.int32)
    thrice = np.empty(_MAX_BLOCK * n_pool, np.int32)  # a gene can enter once per copy in a block
    n_once = n_thrice = 0

    slack = 100.0 * seed
    n, fake, n_blocks, window = 0, 100, 0, 0
    need = np.empty(n_con)
    check = np.arange(n_con)  # constraints, most rejecting first (order is free: an AND)
    rejects = np.zeros(n_con, np.int64)
    memo = np.zeros(n_pool, np.int64)
    stamp = np.zeros(n_pool, np.int64)
    avail = np.empty(n_words, np.uint64)
    w = np.empty(n_words, np.uint64)
    out = np.zeros(_MAX_BLOCK, np.int64)

    cand = np.zeros(n_fac + 1, np.int64)  # cand[n_cand] = 0: target when no factor counts
    cand[: targets.shape[0]] = targets
    n_cand = targets.shape[0]
    hits = np.zeros(n_fac, np.int64)
    learning, phase, rebuild = learn > 0, 0.0, True
    Dc = np.empty((n_pool, 1))
    need_c, ra_c, rb_c = np.empty(n_fac), np.empty(n_fac), np.empty(n_fac)

    for r in range(n_batches):
        while n < 100 + 11 * vip_number * (r + 1):
            cap = max_rep * (n + 1) / vip_number
            need[:] = -slack
            window += 1
            reload, have_avail, found = True, False, False
            size, M, pilot = 3, 16.0, _N_PILOT
            tried = reuse_tried = 0.0
            pool_tried = 0
            pool_stage = pool_limit // n_once + 1 if n_once > 0 else (2**60 if pool_limit else 0)
            while not found:
                if tried > max_proposals:
                    if size == _MAX_BLOCK:
                        return chain[:n], ends, used, n
                    size += 1
                    tried, M, pilot = 0.0, 16.0, _N_PILOT
                if learning and phase >= learn:
                    phase, k = 0.0, 0
                    for f in targets:
                        if hits[f] > 0:
                            cand[k] = f
                            k += 1
                    if k > 0:
                        n_cand, learning, rebuild = k, False, True
                        cand[k] = 0
                elif not learning and learn > 0 and phase >= relearn:
                    phase, learning, rebuild = 0.0, True, True
                    n_cand = targets.shape[0]
                    cand[:n_cand] = targets
                    cand[n_cand] = 0
                    hits[:] = 0
                if rebuild:
                    Dc = np.empty((n_pool, max(n_cand, 1)))
                    for g in range(n_pool):
                        for c in range(n_cand):
                            Dc[g, c] = D[g, cand[c]]
                    rebuild, reload = False, True
                    window += 1
                if reload:
                    for c in range(n_cand):
                        need_c[c], ra_c[c], rb_c[c] = need[cand[c]], ra[cand[c]], rb[cand[c]]
                    reload = False

                in_pool_stage = pool_tried < pool_stage
                src, n_src = everyone, n_pool
                left = pool_stage - pool_tried
                if not in_pool_stage:
                    if reuse_tried >= reuse_limit:
                        reuse_tried, pool_tried = 0.0, 0
                        continue
                    left = int(reuse_limit - reuse_tried) + 1
                    if tried <= list_switch and size == 3:
                        if n_once >= 10:
                            src, n_src = once, n_once
                    elif n_thrice >= 2:
                        src, n_src = thrice, n_thrice

                if size == 3 and (in_pool_stage or learning):
                    m = min(64, left)
                    if learning:
                        m = min(m, int(learn - phase) + 1)
                    m, found = _propose(
                        m, src, n_src, rng, order, Dc, cand, n_cand, need_c, ra_c, rb_c, D, need,
                        check, rejects, used, cap, memo, stamp, window, hits, out,
                    )
                    tried += m
                    phase += m
                    if in_pool_stage:
                        pool_tried += m
                    else:
                        reuse_tried += m
                    continue

                # ---- one pair (or triple, ...) against every candidate for the last gene
                if not have_avail:
                    avail[:] = np.uint64(0)
                    for g in range(n_pool):
                        if used[g] <= cap:
                            avail[g >> 6] |= one << np.uint64(g & 63)
                    have_avail = True
                pilot -= 1
                cost = n_top / M if pilot < 0 else 0.0
                tried += cost
                phase += cost
                reuse_tried += cost
                out[0] = src[rng.integers(0, n_src)]
                ok = used[out[0]] <= cap
                for a in range(1, size - 1):  # targeted genes, all but the last
                    if not ok:
                        break
                    if a == 1 and size == 3:
                        i = out[0]
                        if stamp[i] != window:
                            memo[i] = cand[_furthest(Dc, i, -1, need_c, 0.5, ra_c, rb_c, n_cand)]
                            stamp[i] = window
                        f = memo[i]
                    else:
                        worst, f = -1e99, 0
                        for c in range(n_cand):
                            d = 0.0
                            for b in range(a):
                                d += Dc[out[b], c]
                            z = (need_c[c] - d) * ra_c[c] / (size - a) + rb_c[c]
                            if z > worst:
                                worst, f = z, cand[c]
                    out[a] = order[f, rng.integers(0, n_top)]
                    ok = used[out[a]] <= cap
                if not ok:
                    continue
                if size == 3:
                    f = cand[_furthest(Dc, out[0], out[1], need_c, 1.0, ra_c, rb_c, n_cand)]
                else:
                    worst, f = -1e99, 0
                    for c in range(n_cand):
                        d = 0.0
                        for b in range(size - 1):
                            d += Dc[out[b], c]
                        z = (need_c[c] - d) * ra_c[c] + rb_c[c]
                        if z > worst:
                            worst, f = z, cand[c]
                for x in range(n_words):
                    w[x] = avail[x] & top_bits[f, x]
                for c in check[:_N_SCREEN]:
                    q = need[c]
                    for b in range(size - 1):
                        q -= D[out[b], c]
                    lev = int((q - d_min[c]) * inv_step[c])
                    if lev > _N_LEVELS:  # above every gene of the pool
                        ok = False
                        break
                    if lev > 0:
                        lev = min(lev, _N_LEVELS - 1)
                        for x in range(n_words):
                            w[x] &= level_bits[c, lev, x]
                if not ok:
                    continue
                n_fit, last = 0, -1
                for x in range(n_words):
                    v = w[x]
                    while v:
                        bit = v & (~v + one)
                        v ^= bit
                        g = x * 64 + int(_popcount(bit - one))
                        ok = True
                        for c in check:
                            d = D[g, c]
                            for b in range(size - 1):
                                d += D[out[b], c]
                            if d < need[c]:
                                rejects[c] += 1
                                ok = False
                                break
                        if ok:
                            n_fit += 1
                            if rng.integers(0, n_fit) == 0:  # uniform among the fits
                                last = g
                if n_fit > 0:
                    if pilot >= 0:  # only sets M, so M does not depend on what is accepted
                        M = max(M, 2.0 * n_fit)
                        continue
                    while M < n_fit:
                        M *= 2.0
                    if rng.random() * M < n_fit:
                        out[size - 1] = last
                        found = True

            for b in range(size):
                g = out[b]
                chain[n + b] = g
                used[g] += 1
                slack += D[g]
            n += size
            if n >= 200:
                for b in range(size):
                    n_acc[out[b]] += 1
                for b in range(size):
                    g = out[b]
                    if n_acc[g] == 1:
                        once[n_once] = g
                        n_once += 1
                    if n_acc[g] == 3:
                        thrice[n_thrice] = g
                        n_thrice += 1
            # chain mean minus limit, fake genes included; then one fake gene less
            dev = slack / (n + fake)
            fake = max(fake - 1, 0)
            slack = n * dev + fake * seed
            n_blocks += 1
            if n_blocks == 32 or n_blocks % 256 == 0:
                check = np.argsort(-rejects)
        ends[r] = n
    return chain[:n], ends, used, -1


def _chain_layout(total_batches: int, nthreads: int) -> list:
    """
    Batches (of 10 control sets) per bootstrap chain: one chain per thread, batches
    split evenly. A chain's cost is about linear in its number of batches, so this
    gives every thread the same work. The control sets depend on (seed, nthreads).
    """
    n_chains = max(1, min(max(1, nthreads), total_batches))
    base, extra = divmod(total_batches, n_chains)
    return [base + 1] * extra + [base] * (n_chains - extra)


def _skip_mask(skip_factors, n_factors: int) -> np.ndarray:
    """Boolean mask of unmatched factors from None, one factor number (from 1) or a list."""
    if skip_factors is None:
        numbers = []
    elif isinstance(skip_factors, (int, np.integer)):
        numbers = [int(skip_factors)]
    else:
        numbers = [int(i) for i in skip_factors]
    bad = [i for i in numbers if not 1 <= i <= n_factors]
    if bad:
        raise ValueError(f"skip_factors {bad} outside 1..{n_factors} ({n_factors} factors)")
    mask = np.zeros(n_factors, dtype=np.bool_)
    mask[[i - 1 for i in numbers]] = True
    if mask.all():
        raise ValueError("skip_factors leaves no factor to match")
    return mask


def _shift_nonpositive(vip_X: np.ndarray, nonvip_X: np.ndarray, skip: np.ndarray) -> tuple:
    """
    Shift factors whose case mean is <= 0, as bootstrap_match_b3_repair.c.

    The bounds are relative to the case mean, so a mean <= 0 gives an empty window that
    no control set can enter. A constant added to every gene moves the case mean to one
    standard deviation of the factor over cases and candidates (to 1 if the factor is
    constant). Skipped factors are left as they are.

    :returns: ``(vip_X, nonvip_X, shifts)``, copies if anything was shifted; ``shifts``
        is a list of ``(0-based factor, case mean, constant added)``.
    """
    mean = vip_X.mean(axis=0)
    shifts = []
    for f in np.flatnonzero((mean <= 0) & ~skip):
        sd = float(np.concatenate([vip_X[:, f], nonvip_X[:, f]]).std())
        shifts.append((int(f), float(mean[f]), (sd if sd > 0 else 1.0) - float(mean[f])))
    if shifts:
        vip_X, nonvip_X = vip_X.copy(), nonvip_X.copy()
        for f, _, c in shifts:
            vip_X[:, f] += c
            nonvip_X[:, f] += c
    return vip_X, nonvip_X, shifts


def _run_one_chain(
    tables: tuple,
    chain_seed: np.random.SeedSequence,
    run_idx: int,
    n_batches: int,
    vip_number: int,
    max_rep: int,
    max_tries: int,
    max_restarts: int,
    tolerance: float,
    n_candidates: int,
    target: int,
) -> tuple:
    """
    One bootstrap chain (one joblib task): ``(pool row indices, used pool mask, stuck)``.

    At module level so that a worker imports it, and with it the compiled kernels from
    the numba cache: a nested function is pickled by value with the kernels it uses, and
    every worker would then compile them again.

    A chain can reach a running-mean state no pair satisfies, mostly in its first few
    hundred genes. Such a chain restarts from scratch on a new stream derived from the
    seed (attempt 0 is the chain's own stream), so runs stay reproducible.
    """
    stuck = []
    for attempt in range(max_restarts + 1):
        ss = chain_seed
        if attempt > 0:
            ss = np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (attempt,))
        rng = np.random.Generator(np.random.PCG64(ss))
        chain, ends, used_counts, stuck_at = batch_control_chain(
            *tables, vip_number, max_rep, rng, n_batches, max_tries
        )
        if stuck_at < 0:
            stops = (ends[:, None] - vip_number * np.arange(10)).ravel()  # newest first
            return chain[stops[:, None] + np.arange(-vip_number, 0)], used_counts > 0, stuck
        stuck.append(int(stuck_at))
        if attempt < max_restarts:
            _log(
                f"Bootstrap: chain {run_idx} stuck after {stuck_at:,} of up to {target:,} "
                f"control genes, restarting (attempt {attempt + 2} of {max_restarts + 1})"
            )
    raise RuntimeError(
        f"Bootstrap stuck: chain {run_idx} found no matched block in {max_tries:,} "
        f"consecutive proposals in all {max_restarts + 1} attempts (stuck after "
        f"{', '.join(f'{n:,}' for n in stuck)} of up to {target:,} control genes). "
        f"The {n_candidates} control candidates cannot keep the factor means "
        f"within +-{tolerance:.0%} of the case means; lower --min_distance or raise "
        "--tolerance."
    )


def run_bootstrap(
    case_genes: list,
    factors_file: str,
    annotation_file: str,
    n_control_sets: int = 100,
    tolerance: float = 0.05,
    min_dist: float = 200_000,
    flip: bool = False,
    max_rep: int = 25,
    seed: int = None,
    nthreads: int = 1,
    control_genes: list = None,
    distance_file: str = None,
    skip_factors: int | list | None = None,
    return_used: bool = False,
    distance_genes: list | None = None,
    max_tries: int = _MAX_PROPOSALS,
    max_restarts: int = 10,
):
    """
    Step 6: matched bootstrap control set generator.

    The defaults are those of bootstrap_match_b3_repair.c run with
    exdef_input_parameters_file_hassan.txt: the same draw law, one-sided bounds, every
    factor matched, tolerance 0.05, max_rep 25, min_dist 200 kb, flip no; except the
    number of control sets, 100 here and 10 there (Runs_number 1, Iterations_number 10).

      - Blocks of 3 genes with the draw law of the C; candidates for the last gene are
        screened all at once and the limits are never loosened
        (:func:`batch_control_chain`). Factor means are limited only on the side of the
        pool mean.
      - Factors in skip_factors are not matched; every other factor with a case mean <= 0
        is shifted (see :func:`_shift_nonpositive`).
      - Sets are made in batches of 10; the batches of one chain run in sequence with
        continuous state (used_counts, current_means, fake_seed persist across batches),
        matching Perl's within-process continuous state.
      - Target for batch r: 100 + 11 * vip_number * r (grows each batch).
      - The n_control_sets / 10 batches are spread over independent chains sized to
        ``nthreads`` (see :func:`_chain_layout`); Perl runs Runs_number chains of
        Iterations_number / 10 batches (total Runs_number x Iterations_number sets).
        Each chain gets its own stream from ``SeedSequence(seed).spawn`` (PCG64), so the
        sets depend on (seed, nthreads).
      - ``return_used=True`` also returns the Perl ``used_genes``: case genes plus every
        control gene accepted by a chain (the genome-shuffle universe).

    Parameters
    ----------
    case_genes        : list of VIP gene IDs (HLA/histone excluded by caller)
    factors_file      : path to confounding factors table (TSV, first col gene_id)
    annotation_file   : path to BED gene coordinates file (chrom start end gene_id strand). Distances to nearest VIP are computed automatically via compute_distances.
    n_control_sets    : number of matched control sets, a multiple of 10 (default 100, Perl Runs_number 10 x Iterations_number 10)
    tolerance         : allowed ± fraction for confounder matching (default 0.05)
    min_dist          : minimum distance (bp) from VIPs for control eligibility (default 200 kb)
    flip              : if True, swap VIP and control pools (default False)
    max_rep           : max times a control gene may be sampled on average (default 25)
    nthreads          : number of parallel workers
    control_genes     : optional explicit candidate control gene IDs. If None, all genes in factors_file that are not case genes are used. run_enrichment passes the universe minus case and excluded genes.
    distance_genes    : genes the control-eligibility distance is measured from
                        (default case_genes). The Perl distance_file.txt uses every
                        listed case gene, also those dropped from the universe.
    distance_file     : optional path to pre-computed distance TSV (gene_id \\t distance, no header). When provided, distances are read directly instead of calling compute_distances(). Use Perl's distance_genes_set_file.txt for exact control pool matching.
    skip_factors      : factor number, or list of numbers, left out of the matching, from 1 in the order of factors_file (default None: match every factor).
    max_tries         : consecutive rejected proposals, per block size, after which a chain counts as stuck (default 1.8e8).
    max_restarts      : restarts of a stuck chain on a new seed-derived stream before a RuntimeError (default 10).

    Returns
    -------
    df_set               : pl.DataFrame — VIP genes (gene_id column)
    df_bootstrap_control : pl.DataFrame — rows = control sets, cols = gene positions (sample_id + column_0..column_N)
    """
    if n_control_sets < 10 or n_control_sets % 10 != 0:
        raise ValueError("n_control_sets must be a positive multiple of 10")
    case_ids = set(case_genes)

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

    skip = _skip_mask(skip_factors, vip_X.shape[1])

    vip_X, nonvip_X, shifts = _shift_nonpositive(vip_X, nonvip_X, skip)
    for f, m, c in shifts:
        _log(f"Factor {f + 1}: case mean {m:.6g} <= 0, {c:.6g} added to every gene")

    chain_batches = _chain_layout(n_control_sets // 10, nthreads)
    _log(
        f"Bootstrap: {n_control_sets} control sets for {df_set.height} "
        f"case genes from {df_control_pool.height} candidates >= {min_dist:,} bp away "
        f"({len(chain_batches)} chain(s), {nthreads} thread(s))"
    )
    if df_control_pool.height <= 1.5 * df_set.height:
        _log(
            "Warning: fewer than 1.5x as many control candidates as case genes; "
            "the FDR may be high.",
        )

    tables = _match_tables(vip_X[:, ~skip], nonvip_X[:, ~skip], tolerance)
    target = 100 + 11 * vip_number * max(chain_batches)

    # One independent PCG64 stream per chain (seed None -> fresh OS entropy).
    chain_seeds = np.random.SeedSequence(seed).spawn(len(chain_batches))

    # No more worker processes than chains; a single chain runs in this process.
    n_workers = min(max(1, nthreads), len(chain_batches))
    batch_results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
        delayed(_run_one_chain)(
            tables, chain_seeds[i], i, nb, vip_number, max_rep, max_tries,
            max_restarts, tolerance, len(nonvip_genes), target,
        )
        for i, nb in enumerate(chain_batches)
    )
    for i, r in enumerate(batch_results):
        if r[2]:
            _log(
                f"Warning: bootstrap chain {i} got stuck {len(r[2])} time(s) (after "
                f"{', '.join(f'{n:,}' for n in r[2])} control genes) and was restarted "
                "with a new seed",
            )

    sets = np.concatenate([r[0] for r in batch_results], axis=0)
    used_mask = np.logical_or.reduce([r[1] for r in batch_results])
    used_genes = sorted(set(df_set["gene_id"].to_list()) | set(nonvip_genes[used_mask]))
    pool_ids = pl.Series(nonvip_genes)  # columns share its strings
    df_bootstrap_control = pl.DataFrame(
        [pool_ids.gather(sets[:, j]).alias(f"column_{j}") for j in range(sets.shape[1])]
    )
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
    if isinstance(control_sets, pl.DataFrame):  # column by column: no array of str objects
        columns = [c for c in control_sets.columns if c != "sample_id"]
        shape = (control_sets.height, len(columns))
        lines = (control_sets[c].to_list() for c in columns)
    else:
        flat = np.asarray(control_sets, dtype=object)
        if flat.ndim != 2:
            raise ValueError("control sets must all have the same length")
        shape, lines = flat.shape, flat.T
    M = np.empty(shape, dtype=np.int32)
    missing = set()
    for j, line in enumerate(lines):
        row = [g2i.get(g, -1) for g in line]
        M[:, j] = row
        if -1 in row:
            missing.update(g for g, i in zip(line, row) if i < 0)
    if missing:
        raise ValueError(
            f"{len(missing)} control genes are not in the rank table, e.g. "
            f"{sorted(missing)[:5]}"
        )
    return V, M


@njit(cache=True)
def _count_bins(bins, M, V, n_thr):
    """
    Sweep counts per population column of ``bins`` and threshold.

    :returns: ``vip`` (P, T) case-gene counts and ``ctrl`` (P, T, S) counts per control
        set; a control gene present k times in a set counts k times (Perl
        ``$iterated{$gene}``).
    """
    n_pops = bins.shape[1]
    vip = np.empty((n_pops, n_thr), dtype=np.int64)
    ctrl = np.empty((n_pops, n_thr, M.shape[0]), dtype=np.int32)
    hist = np.zeros((n_pops, n_thr + 1), dtype=np.int64)
    for k in range(V.shape[0]):
        for p in range(n_pops):
            hist[p, bins[V[k], p]] += 1
    for p in range(n_pops):
        c = 0
        for t in range(n_thr):
            c += hist[p, t]
            vip[p, t] = c
    for s in range(M.shape[0]):
        hist[:] = 0
        for j in range(M.shape[1]):
            g = M[s, j]
            for p in range(n_pops):
                hist[p, bins[g, p]] += 1
        for p in range(n_pops):
            c = 0
            for t in range(n_thr):
                c += hist[p, t]
                ctrl[p, t, s] = c
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


def _clust_buffers(n_genes: int, n_pops: int, n_sets: int, n_thr: int) -> tuple:
    """Arrays :func:`_count_bins_clust` writes: counts (first two) and scratch."""
    return (
        np.empty((n_pops, n_thr)),
        np.empty((n_pops, n_thr, n_sets)),
        np.empty((n_genes, n_pops), dtype=np.bool_),
        np.empty((n_genes, n_pops), dtype=np.int32),
        np.empty(n_genes, dtype=np.int32),
        np.empty(n_genes, dtype=np.int64),
        np.empty(n_thr),
        np.empty(n_thr),
    )


@njit(cache=True)
def _count_bins_clust(
    bins, M, V_order, nb_ptr, nb_idx, n_thr, vip, ctrl, done, stamp, in_set, mult, num, den
):
    """
    Sweep counts as count_sweeps_script.pl with ``Count_sweeps yes``, per population
    column of ``bins`` (threshold bin per gene, ``n_thr`` = in no sweep).

    Case genes, in ``V_order``: a gene in a sweep that is not yet done counts 1 at every
    threshold it passes and marks its in-sweep neighbours done, so they are never
    counted. Control set s, genes in set order: a gene in a sweep that is not yet done,
    together with its not-yet-done in-sweep neighbours present in the set, adds at each
    threshold the mean multiplicity in the set of those of them that pass it; all of
    them are then done. ``done`` is per population.

    Writes ``vip`` (P, T) and ``ctrl`` (P, T, S), float64; the other arrays are scratch
    (:func:`_clust_buffers`), reset here.
    """
    n_pops = bins.shape[1]
    n_sets, n_len = M.shape
    vip[:] = 0.0
    ctrl[:] = 0.0
    done[:] = False
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
    stamp[:] = 0  # == s + 1: done in set s
    in_set[:] = 0  # == s + 1: gene present in set s
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


def _scope_names(populations: list, groups: list) -> list:
    """Scope labels in the order of :func:`_add_scopes`: populations, groups, ``All``."""
    return list(populations) + list(dict.fromkeys(groups)) + ["All"]


def _fdr_scopes(pop_interest, populations: list, groups: list) -> list:
    """
    ``(label, row)`` of every scope asked for, ``row`` indexing :func:`_add_scopes`.

    ``pop_interest`` is a comma-separated string or a list of population codes, group
    labels and ``All``. Labels match exactly, or ignoring case if no label matches
    exactly; ``all`` is the aggregate. Repeated scopes are dropped.
    """
    asked = pop_interest.split(",") if isinstance(pop_interest, str) else list(pop_interest)
    asked = [str(x).strip() for x in asked]
    if not asked or not all(asked):
        raise ValueError(f"pop_interest has an empty scope: {pop_interest!r}")
    names = _scope_names(populations, groups)
    out = []
    for x in asked:
        rows = [k for k, nm in enumerate(names) if nm == x]
        if not rows:
            rows = [k for k, nm in enumerate(names) if nm.casefold() == x.casefold()]
        if not rows:
            raise ValueError(
                f"pop_interest: unknown scope {x!r}; choose from {', '.join(names)}"
            )
        if len(rows) > 1:
            raise ValueError(
                f"pop_interest: {x!r} fits {len(rows)} scopes ("
                f"{', '.join(names[k] for k in rows)}); rename a population or group"
            )
        if rows[0] not in [k for _, k in out]:
            out.append((names[rows[0]], rows[0]))
    return out


class SweepCounts(NamedTuple):
    """Sweep counts of one rank table (:func:`count_sweeps_fast`)."""

    rank_table: pl.DataFrame
    th: np.ndarray  # thresholds, increasing
    bins: np.ndarray  # threshold bin, gene x population
    V: np.ndarray  # case rows
    M: np.ndarray  # control rows, set x gene
    vip: np.ndarray  # case counts, population x threshold
    ctrl: np.ndarray  # control counts, population x threshold x set
    populations: list
    groups: list
    neighbors: tuple | None


def count_sweeps_fast(
    rank_table: pl.DataFrame,
    vip_genes: list,
    control_sets,
    thresholds: list,
    populations: list,
    groups: list,
    neighbors: tuple | None = None,
) -> SweepCounts:
    """
    Sweep counts of the case genes and of every control set, per population and
    threshold, as count_sweeps_script.pl with ``Count_sweeps no`` (genes counted
    individually), or ``Count_sweeps yes`` when ``neighbors`` (:func:`_gene_neighbors`) is
    given (neighbouring genes in a sweep count once; see :func:`_count_bins_clust`).

    :param rank_table: Merged rank table (:func:`_read_rank_table`); rank columns in
        ``populations`` order.
    :param control_sets: Control sets, (n_sets, n_case) gene ids or the table returned by
        :func:`run_bootstrap`.
    :returns: Input of :func:`enrichment_curves` and :func:`estimate_fdr_fast`.
    """
    th = np.array(sorted(int(x) for x in thresholds), dtype=np.int64)
    bins = _rank_bins(rank_table, len(populations), th)
    gene_ids = rank_table[:, 0].to_list()
    V, M = _gene_index(gene_ids, vip_genes, control_sets)
    if neighbors is None:
        vip, ctrl = _count_bins(bins, M, V, len(th))
    else:
        V = _case_order(gene_ids, vip_genes)
        buf = _clust_buffers(*bins.shape, M.shape[0], len(th))
        _count_bins_clust(bins, M, V, neighbors[0], neighbors[1], len(th), *buf)
        vip, ctrl = buf[0], buf[1]
    return SweepCounts(
        rank_table, th, bins, V, M, vip, ctrl, list(populations), list(groups), neighbors
    )


def enrichment_curves(counts: SweepCounts) -> pl.DataFrame:
    """
    Enrichment curves for every population, group and ``All``, from all control sets.

    - population: case genes in a sweep (rank <= threshold) vs the same count in each
      control set (genes counted with their multiplicity in the set);
    - group / All: sums of their populations' counts;
    - ``ctrl_ci_lo``/``ctrl_ci_hi``: sorted control counts at ``int(0.025 n)`` and
      ``int(0.975 n)``; ``p_value``: fraction of control sets with a larger count.

    :returns: One row per scope and threshold; ``n_sets`` is the number of control sets.
    """
    vip, ctrl, names = _add_scopes(counts.vip, counts.ctrl, counts.populations, counts.groups)
    return _curve_frame(names, counts.th, vip, ctrl)


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


def _excess_stat(vip: np.ndarray, ctrl_mean: np.ndarray) -> float:
    """
    estimate_FPR.pl statistic of one scope, with the arguments of the pipeline (p-value
    cutoff 1, at least 1 case gene, every threshold): sum over the thresholds with a case
    gene of (case count - control mean).
    """
    keep = vip >= 1
    return float(vip[keep].sum() - ctrl_mean[keep].sum())


_PROGRESS_SECONDS = 30  # least time between two progress lines of the shuffles


def _duration(seconds: float) -> str:
    """``45 s``, ``3 min 20 s`` or ``1 h 05 min``."""
    sec = int(round(seconds))
    if sec < 60:
        return f"{sec} s"
    if sec < 3600:
        return f"{sec // 60} min {sec % 60:02d} s"
    return f"{sec // 3600} h {sec % 3600 // 60:02d} min"


def _draw_orders(n_segments: int, rng: np.random.Generator, n: int) -> np.ndarray:
    """
    ``n`` random orders of the segments, one per row (shuffle_sweep_script_nocut.pl): a
    segment keeps its gene order, so every shuffle is a permutation of the genes.
    """
    orders = np.empty((n, n_segments), dtype=np.int32)
    for r in range(n):
        orders[r] = rng.permutation(n_segments)
    return orders


def _fdr_segments(
    rank_table: pl.DataFrame,
    coord_file: str,
    universe,
    thresholds: list,
    shuffling_segs: int,
    shuffle_blocks: int | None = None,
) -> tuple:
    """
    Genes and segments of the genome shuffles of exdef_pipeline.pl: genes = ``universe`` (Perl: valid genes used by
    the bootstrap) in chromosomal order; a cut every ``shuffling_segs`` genes unless both
    flanking genes rank <= max(thresholds) (Perl maxi_rank, the first threshold of the
    descending list) in some population. The segments are reordered, never mirrored.
    ``shuffle_blocks`` = B replaces the segment size by ``int(n_genes / B)``, the
    commented-out line of shuffle_sweep_script_nocut.pl (B blocks, Enard & Petrov 2020).

    :returns: ``(ordered_genes, segments)``.
    """
    ordered = _ordered_genes(coord_file, universe)
    strong = _strong_signal_genes(rank_table, max(int(x) for x in thresholds))
    seg_size = len(ordered) // shuffle_blocks if shuffle_blocks else shuffling_segs
    return ordered, _shuffle_segments(ordered, strong, max(1, seg_size))


def _fdr_null_clusters(
    ord_bins, ord_rows, n_rows, segments, orders, M, V, n_thr, populations, groups,
    scope_rows, neighbors,
) -> np.ndarray:
    """
    FDR null statistics (shuffles x scopes) for a block of segment orders, with
    ``Count_sweeps yes``: what a gene counts depends on the rest of its set, so every set
    is counted. Neighbours stay genomic; only the ranks move.
    """
    out = np.empty((len(orders), len(scope_rows)))
    bins = np.full((n_rows, ord_bins.shape[1]), n_thr, dtype=np.int16)
    idx = np.empty(len(ord_rows), dtype=np.int32)
    buf = _clust_buffers(n_rows, ord_bins.shape[1], M.shape[0], n_thr)
    for r, order in enumerate(orders):
        np.concatenate([segments[k] for k in order], out=idx)
        bins[ord_rows] = ord_bins[idx]
        _count_bins_clust(bins, M, V, neighbors[0], neighbors[1], n_thr, *buf)
        vip, ctrl, _ = _add_scopes(buf[0], buf[1], populations, groups)
        out[r] = [_excess_stat(vip[k], ctrl[k].mean(axis=1)) for k in scope_rows]
    return out


@njit(cache=True, nogil=True)
def _shuffle_totals(orders, starts, ends, ord_bins, w_ctrl, w_case, members, n_thr):
    """
    Cumulative counts of each shuffle, per scope and threshold: case genes, and control
    genes added over all control sets. ``w_ctrl``, ``w_case``: weight of the gene at each
    genome position; ``members`` (scopes, populations).
    """
    n_scopes, n_pops = members.shape
    n_shuf = orders.shape[0]
    case = np.zeros((n_shuf, n_scopes, n_thr), np.int64)
    ctrl = np.zeros((n_shuf, n_scopes, n_thr), np.int64)
    h_case = np.empty((n_pops, n_thr + 1), np.int64)
    h_ctrl = np.empty((n_pops, n_thr + 1), np.int64)
    for r in range(n_shuf):
        h_case[:] = 0
        h_ctrl[:] = 0
        pos = 0
        for j in range(orders.shape[1]):
            k = orders[r, j]
            for i in range(starts[k], ends[k]):
                a, b = w_ctrl[pos], w_case[pos]
                pos += 1
                if a == 0 and b == 0:
                    continue
                for p in range(n_pops):
                    t = ord_bins[i, p]
                    h_ctrl[p, t] += a
                    h_case[p, t] += b
        for p in range(n_pops):
            a = b = 0
            for t in range(n_thr):
                a += h_ctrl[p, t]
                b += h_case[p, t]
                for s in range(n_scopes):
                    if members[s, p]:
                        ctrl[r, s, t] += a
                        case[r, s, t] += b
    return case, ctrl


def _fdr_null_totals(
    ord_bins, starts, ends, orders, w_ctrl, w_case, members, n_sets, n_thr
) -> np.ndarray:
    """
    FDR null statistics (shuffles x scopes) for a block of segment orders, with
    ``Count_sweeps no``. Only the mean over the control sets is needed, so the control
    genes are counted once with their multiplicity over all sets.
    """
    case, ctrl = _shuffle_totals(
        orders, starts, ends, ord_bins, w_ctrl, w_case, members, n_thr
    )
    return np.array(
        [[_excess_stat(v, c / n_sets) for v, c in zip(vs, cs)] for vs, cs in zip(case, ctrl)]
    )


def estimate_fdr_fast(
    counts: SweepCounts,
    coord_file: str,
    shuffle_genes,
    n_shuffles: int,
    shuffling_segs: int,
    seed: int | None = None,
    nthreads: int = 1,
    pop_interest: str | list = "All",
    shuffle_blocks: int | None = None,
    progress: bool = False,
) -> pl.DataFrame:
    """
    Whole-curve FDR p-values from genome shuffles, as shuffle_sweep_script_nocut.pl +
    count_sweeps_script.pl + ``estimate_FPR.pl ... <scope>:``, one per scope of
    ``pop_interest`` (same shuffles for every scope).

    Each shuffle is counted like the real data (same control sets and counting rule) and
    scored with :func:`_excess_stat`. Shuffles are drawn in the main process and counted
    in blocks over ``nthreads`` workers, so ``nthreads`` does not change the result.

    :param counts: Counts of the real data (:func:`count_sweeps_fast`).
    :param shuffle_genes: Genes that are shuffled (Perl: valid genes used by the
        bootstrap, i.e. case genes and every accepted control gene); others count as
        not in a sweep in the shuffles.
    :param pop_interest: Scopes to test (see :func:`_fdr_scopes`).
    :param shuffle_blocks: Cut the genome into this many blocks (:func:`_fdr_segments`).
    :param progress: Log the progress of the shuffles.
    :returns: One row per scope: ``scope``, ``p_value`` (#null >= real / n, as Perl),
        ``n_replicates``, ``real_stat``, ``max_null_stat`` (max(0, nulls), as Perl
        ``rand_max``), ``p_value_perm`` ((#null >= real + 1) / (n + 1)).
    """
    if n_shuffles % 8 != 0:
        raise ValueError("n_shuffles must be a multiple of 8")
    populations, groups, n_thr = counts.populations, counts.groups, len(counts.th)
    scopes = _fdr_scopes(pop_interest, populations, groups)
    rows = [k for _, k in scopes]
    vip, ctrl, names = _add_scopes(counts.vip, counts.ctrl, populations, groups)
    real_stat = np.array([_excess_stat(vip[k], ctrl[k].mean(axis=1)) for k in rows])

    ordered, segments = _fdr_segments(
        counts.rank_table, coord_file, shuffle_genes, counts.th, shuffling_segs, shuffle_blocks
    )
    gene_ids = counts.rank_table[:, 0].to_list()
    g2i = {g: i for i, g in enumerate(gene_ids)}
    ord_rows = np.array([g2i.get(g, -1) for g in ordered], dtype=np.int64)
    ord_bins = np.full((len(ordered), len(populations)), n_thr, dtype=np.int16)
    ord_bins[ord_rows >= 0] = counts.bins[ord_rows[ord_rows >= 0]]
    # Shuffled genes absent from the rank table map to an extra, never-counted row.
    n_rows = len(gene_ids) + 1
    ord_rows[ord_rows < 0] = len(gene_ids)

    # Ten blocks per worker: small enough to log progress, large enough to stay cheap.
    # The segment orders are drawn block by block as the workers take them.
    chunk = max(1, math.ceil(n_shuffles / (10 * max(1, nthreads))))
    rng = np.random.default_rng(seed)
    orders = (
        _draw_orders(len(segments), rng, min(chunk, n_shuffles - i))
        for i in range(0, n_shuffles, chunk)
    )
    M, V = counts.M, counts.V
    if counts.neighbors is None:
        starts = np.array([seg[0] for seg in segments], dtype=np.int64)
        ends = np.array([seg[-1] + 1 for seg in segments], dtype=np.int64)
        # a gene listed twice in the annotation keeps the ranks of its last position
        last = np.zeros(len(ord_rows), dtype=bool)
        last[len(ord_rows) - 1 - np.unique(ord_rows[::-1], return_index=True)[1]] = True
        w_ctrl = np.where(last, np.bincount(M.ravel(), minlength=n_rows)[ord_rows], 0)
        w_case = np.where(last, np.bincount(V, minlength=n_rows)[ord_rows], 0)
        members = np.zeros((len(rows), len(populations)), dtype=bool)
        for s, k in enumerate(rows):
            if k < len(populations):
                members[s, k] = True
            else:
                members[s] = [names[k] == g or k == len(names) - 1 for g in groups]
        backend = "threading"  # the kernel releases the GIL: no copies, no processes
        tasks = (
            delayed(_fdr_null_totals)(
                ord_bins, starts, ends, o, w_ctrl, w_case, members, M.shape[0], n_thr
            )
            for o in orders
        )
    else:
        backend = "loky"
        tasks = (
            delayed(_fdr_null_clusters)(
                ord_bins, ord_rows, n_rows, segments, o, M, V, n_thr, populations, groups,
                rows, counts.neighbors,
            )
            for o in orders
        )
    blocks = Parallel(n_jobs=nthreads, backend=backend, return_as="generator")(tasks)
    null, done, t0, t_log = [], 0, time.time(), time.time()
    for block in blocks:
        null.append(block)
        done += len(block)
        if progress and done < n_shuffles and time.time() - t_log >= _PROGRESS_SECONDS:
            t_log = time.time()
            left = (t_log - t0) * (n_shuffles - done) / done
            _log(f"FDR: {done:,} of {n_shuffles:,} shuffles, about {_duration(left)} left")
    null = np.concatenate(null)
    k = (null >= real_stat).sum(axis=0)
    return pl.DataFrame(
        {
            "scope": [label for label, _ in scopes],
            "p_value": k / n_shuffles,
            "n_replicates": [n_shuffles] * len(scopes),
            "real_stat": real_stat,
            "max_null_stat": np.maximum(0.0, null.max(axis=0)),
            "p_value_perm": (k + 1) / (n_shuffles + 1),
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
    ctrl_genes = set()
    for c in df_control.columns:
        if c != "sample_id":
            ctrl_genes.update(df_control[c].unique().to_list())
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
        _log(
            f"{used_path} not found: shuffling case + control-set genes ({len(used)}); "
            "genes accepted by the sampler but absent from every set are missed",
        )
    _log(
        f"Reusing bootstrap from {folder}: {len(case)} case genes, "
        f"{df_control.height} control sets, {len(used)} used genes",
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
    pop_interest: str | list = "All",
    cluster_distance: int = 500_000,
    n_control_sets: int = 100,
    tolerance: float = 0.05,
    min_distance: int = 200_000,
    flip: bool = False,
    max_rep: int = 25,
    nthreads: int = 1,
    n_shuffles: int = 10_000,
    shuffling_segs: int = 2,
    bootstrap_dir: str = None,
    distance_file: str = None,
    seed: int | None = None,
    reuse_bootstrap: str | None = None,
    output_folder: str | None = None,
    count_sweeps: bool = False,
    shuffle_blocks: int | None = None,
    skip_factors: int | list | None = None,
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
    :param pop_interest: Scopes to run the FDR for: population codes, group labels and
        ``All``, as a comma-separated string (``"All,YRI"``) or a list; case is ignored.
        One FDR row per scope, from the same shuffles and not corrected for the number
        of scopes. Every control set is used for the curves and for every shuffle (Perl
        stops at 20 or 100 sets when the p-values of its ``Pop_interest`` are not extreme).
    :param cluster_distance: Neighbour distance (bp) for ``count_sweeps``
        (get_all_neighbors.pl, 10 kb bins of the gene centre); unused otherwise.
    :param n_control_sets: Number of matched control sets (multiple of 10; default
        100, Perl Runs_number 10 x Iterations_number 10). The sampling is split into one
        chain per thread, so the sets depend on ``seed`` and ``nthreads``.
    :param tolerance: Allowed fraction deviation in factor averages for control gene
        matching, on the side of the pool mean.
    :param min_distance: Minimum bp distance from VIPs for control eligibility.
    :param flip: Flip test direction when ``len(VIPs) > len(controls)`` (increases power).
    :param max_rep: Max average resamples per control gene across bootstrap sets.
    :param nthreads: Parallel workers (joblib).
    :param n_shuffles: FDR shuffle replicates (must be a multiple of 8; default 10,000, Perl
        ``FDR_number``).
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
        no header). Passed to ``run_bootstrap`` to bypass internal
        distance computation and match Perl's control pool exactly.
    :param seed: Seeds the bootstrap and the genome shuffles. If None, one is drawn
        and printed so the run can be repeated.
    :param reuse_bootstrap: Output folder of a previous run; its ``case_set.txt`` and
        ``control_set.txt`` are loaded instead of running the bootstrap (same gene set
        and universe required).
    :param output_folder: If given, write ``case_set.txt``, ``control_set.txt``,
        ``used_genes.txt`` (genome-shuffle universe), ``enrichment_curves.txt`` and
        ``fdr.txt`` there.
    :param skip_factors: Factor number, or list of numbers, left out of the matching, from
        1 in the order of ``factors_file``. Default ``None``: every factor is matched.
    :returns: ``(df_case, df_control, df_curves, df_fdr)``: case genes, control sets,
        enrichment curves (every population, group and ``All``) and one FDR row per
        sweep file and scope of ``pop_interest``.
    """
    import gzip as _gz

    _check_enrichment_inputs(sweep_files, populations, groups, thresholds)
    if n_shuffles % 8 != 0:
        raise ValueError("n_shuffles must be a multiple of 8")
    if n_control_sets < 10 or n_control_sets % 10 != 0:
        raise ValueError("n_control_sets must be a positive multiple of 10")
    _scopes = [label for label, _ in _fdr_scopes(pop_interest, populations, groups)]
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    _t_start = time.time()
    _log(f"Seed: {seed}")
    _log(
        f"Input: {len(sweep_files)} sweep file(s), {len(populations)} populations, "
        f"{len(set(groups))} groups, {len(thresholds)} thresholds "
        f"({min(thresholds)}-{max(thresholds)})"
    )
    _log(
        f"FDR: {', '.join(_scopes)} from {n_shuffles:,} shuffles, {nthreads} thread(s)"
    )
    if bootstrap_dir is None and reuse_bootstrap is None:
        _log(
            f"Matching: {n_control_sets} control sets, tolerance {tolerance}, "
            f"max_rep {max_rep}, flip {'yes' if flip else 'no'}"
        )

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
    _log(
        f"Gene set: {len(_case_genes)} case genes, {len(_control_genes)} control "
        f"candidates ({_n_dropped} listed ids outside the universe or excluded)"
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
        _log(f"Bootstrap: loaded from {bootstrap_dir}, {len(_ctrl_sets_raw)} control sets")
    elif reuse_bootstrap is not None:
        df_case, df_control, _used_genes = _load_bootstrap(
            reuse_bootstrap, _case_genes, set(_valid_genes_list)
        )

    else:
        df_case, df_control, _used_genes = run_bootstrap(
            case_genes=_case_genes,
            factors_file=factors_file,
            annotation_file=annotation_file,
            n_control_sets=n_control_sets,
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
            skip_factors=skip_factors,
        )


    _t_post_bootstrap = time.time()

    # --- Steps 7-9: curves, genome shuffles and FDR (vectorized) ---
    _vip_list = df_case["gene_id"].to_list()
    # Perl shuffles the valid genes listed in used_genes.txt (case + accepted controls).
    _shuffle_genes = sorted(set(_used_genes) & set(_valid_genes_list))
    _log(
        f"Bootstrap: {df_control.height} control sets ready in "
        f"{_duration(_t_post_bootstrap - _t_start)}, {len(_shuffle_genes):,} genes to shuffle"
    )

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
            _log(
                f"Count_sweeps yes: neighbours within {cluster_distance:,} bp "
                f"(mean {len(_neighbors[1]) / max(1, len(_valid_genes_list)):.1f} per gene)",
            )
        _counts = count_sweeps_fast(
            _rank_table, _vip_list, df_control, thresholds, populations, groups, _neighbors
        )
        results = enrichment_curves(_counts)
        _log(
            f"Curves {_label}: {results['scope'].n_unique()} scopes x {len(thresholds)} "
            f"thresholds, {results['n_sets'][0]} control sets"
        )
        _t_fdr = time.time()
        _log(f"FDR {_label}: counting {n_shuffles:,} shuffles on {nthreads} thread(s)")
        fdr = estimate_fdr_fast(
            _counts,
            annotation_file,
            _shuffle_genes,
            n_shuffles=n_shuffles,
            shuffling_segs=shuffling_segs,
            seed=seed,
            nthreads=nthreads,
            pop_interest=pop_interest,
            shuffle_blocks=shuffle_blocks,
            progress=True,
        )
        _log(f"FDR {_label}: shuffles done in {_duration(time.time() - _t_fdr)}")
        for row in fdr.iter_rows(named=True):
            _log(
                f"FDR {_label} {row['scope']}: p={row['p_value']:.4f} "
                f"(perm {row['p_value_perm']:.4f}), real={row['real_stat']:.2f} "
                f"max_null={row['max_null_stat']:.2f}"
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
        _log(f"Output: {output_folder}")

    _log(
        f"Time: bootstrap {_t_post_bootstrap - _t_start:.1f} s, curves + FDR "
        f"{time.time() - _t_post_bootstrap:.1f} s, total {time.time() - _t_start:.1f} s"
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
