"""Haplotype matrices and haplotype-matrix images for image-based CNNs.

Counterpart of :func:`flexsweep.fv.summary_statistics` for raw haplotype matrices: reads
discoal simulations (``neutral/``, ``sweep/``, ``params.txt.gz``) or a directory of
VCF/BCF files and returns one ``(haplotypes, K)`` 0/1 image per locus, built from the K
contiguous biallelic SNPs centred on the SNP closest to the locus center (the sweep
site in simulations). Loci with fewer than K SNPs on either side are zero-padded, so
the central SNP always sits in column ``K // 2``; ``n_snps`` records the real count.

This module also holds the row/column orderings (:func:`sort_haplotypes` and the
functions it dispatches to), the disruption utilities used to test which signal a trained
CNN relies on, and the TrIdent representation. They lived in :mod:`flexsweep.utils` up to
version 2.0.

Image sizes in recent methods: TrIdent (central 499 SNPs), Harris et al. 2025 DANN
sweep classifier (201 SNPs, sorted by haplotype frequency), SKINET (400 SNPs).
"""

import glob
import os
from collections import defaultdict

import numpy as np
import polars as pl
from numba import int64, njit

#: Named row (haplotype) orderings. ``sort`` also accepts any callable taking and
#: returning a ``(haplotypes, sites)`` array, so users can plug in their own.
SORTS = ("haplotype_freq", "haplotype_freq_hamming", "row_dist", "center_window_freq",
         "center_window_dist", "freq", "corr", "daf", "pcc_cols", None)
#: Allele coding of the image pixels. ``derived``: 0/1 uint8 (ancestral/derived, the
#: Flex-sweep default). ``major_minor``: int8 with major -1, minor +1 and missing 0, the
#: coding of Harris et al. 2025 (bioRxiv 10.1101/2025.10.14.682443).
ENCODINGS = ("derived", "major_minor")
#: Sub-window used by the ``center_window_*`` orderings (Harris et al. 2025 use 51 SNPs).
CENTER_SUBWINDOW = 51
# TrIdent images hold sorted minor-allele counts (0-25), stored as uint8 x TRIDENT_SCALE
TRIDENT_SCALE = 10


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

    # same grouping logic as in haplotype_freq_sorting_hamming
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


def trident_matrix(
    h,
    positions,
    num_strands,
    window_length=25,
    num_stride=2,
    K=499,
    center=6e5,
):
    """TrIdent-style feature matrix of one locus, before resizing.

    Follows Arnab et al. 2025 (TrIdent): keeps biallelic sites with a minor allele count
    of at least 3 (no singletons or doubletons), takes the K contiguous sites centred on
    the site closest to `center`, codes the minor allele as 1 over that block, then for
    each window of `window_length` sites (starting every `num_stride` sites) counts minor
    alleles per haplotype and sorts the counts in increasing order. The defaults give the
    published n x 237 matrix from 499 SNPs, 25-SNP windows and a stride of 2.

    Args:
        h (np.ndarray): (sites, haplotypes) 0/1 matrix.
        positions (np.ndarray): (sites,) physical positions.
        num_strands (int): Expected number of haplotypes.
        window_length (int): Sliding window size in SNPs.
        num_stride (int): Stride between window starts.
        K (int): Number of central SNPs (odd K puts the closest SNP in the middle).
        center (float): Center coordinate.

    Returns:
        np.ndarray | None: (num_strands, n_windows) int array, or None if the locus has
        the wrong number of haplotypes or too few SNPs around `center`.
    """
    from .fv import get_closest_snps

    h = np.asarray(h)
    if h.ndim != 2 or h.shape[0] == 0 or h.shape[1] != num_strands:
        return None
    counts = h.sum(axis=1)
    keep = (counts > 2) & (counts < num_strands - 2)
    h, positions = h[keep], np.asarray(positions)[keep]
    if positions.size < K:
        return None
    try:
        idx = get_closest_snps(positions, center=center, N=K, symmetric=True)
    except ValueError:  # not enough SNPs on one side of the center
        return None
    data = h[idx, :].T.astype(np.int64)  # (haplotypes, K)

    # Minor-allele coding of the whole K-SNP block (major 0, minor 1), as in the paper
    flip = data.sum(axis=0) > (num_strands // 2)
    data[:, flip] = 1 - data[:, flip]

    # Windows start at column 2j-1 for j = 1..(K - window_length) // stride, so the final
    # partial-stride window is not used (499 SNPs, 25-SNP windows, stride 2 -> 237 columns)
    cols = [
        np.sort(data[:, start : start + window_length].sum(axis=1))
        for start in range(0, K - window_length, num_stride)
    ]
    if not cols:
        return None
    return np.column_stack(cols)


def trident_parse(
    filename,
    num_strands,
    dim,
    window_length=25,
    num_stride=2,
    K=499,
    center=6e5,
    seq_len=1.2e6,
):
    """TrIdent-style haplotype image of one ms/ms.gz replicate, resized to (dim, dim).

    Parses the file with :func:`flexsweep.fv.parse_ms_numpy` and builds the matrix with
    :func:`trident_matrix`. Resizing uses OpenCV (``cv2``, bilinear), which is an optional
    dependency.

    Returns:
        np.ndarray | None: (dim, dim) float64 image, or None if the file cannot be parsed
        or the locus is unusable (see :func:`trident_matrix`).
    """
    from .fv import parse_ms_numpy

    try:
        import cv2
    except ImportError as e:
        raise ImportError("trident_parse needs OpenCV: pip install opencv-python-headless") from e

    parsed = parse_ms_numpy(filename, seq_len=seq_len)
    if parsed is None:
        return None
    h, _, _, _, positions, _ = parsed
    img = trident_matrix(h, positions, num_strands, window_length, num_stride, K, center)
    if img is None:
        return None
    return cv2.resize(img.astype(np.float64), (dim, dim), interpolation=cv2.INTER_LINEAR)


def _hap_groups(block: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by identical haplotype, most frequent group first."""
    _, inv, counts = np.unique(block, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    order = np.argsort(-counts, kind="stable")  # ties keep first-appearance order
    return [np.flatnonzero(inv == g) for g in order]


def _row_dist_order(block: np.ndarray) -> np.ndarray:
    """Rows ordered by Hamming distance to the most frequent haplotype (Harris et al. 2025).

    Groups of identical haplotypes stay together; within a distance, the more frequent
    group comes first.
    """
    groups = _hap_groups(block)
    if not groups:
        return np.arange(block.shape[0])
    ref = block[groups[0][0]]
    dist = [int((block[g[0]] != ref).sum()) for g in groups]
    return np.concatenate([groups[i] for i in np.argsort(dist, kind="stable")])


def _center_window_order(block: np.ndarray, sub_length: int, inner: str) -> np.ndarray:
    """Group rows by their central sub-window haplotype, then order within each group.

    Harris et al. 2025 (DANNsweeps ``sort_centerWindow``): the central ``sub_length`` SNPs
    define the groups, ordered by frequency; rows inside a group are then ordered over the
    full window by distance to the most frequent haplotype (``inner="row_dist"``) or by
    haplotype frequency (``inner="haplotype_freq"``).
    """
    n_sites = block.shape[1]
    start = max(0, n_sites // 2 - sub_length // 2)
    centre = block[:, start : start + sub_length]
    out = []
    for g in _hap_groups(centre):
        sub = block[g]
        inner_order = (_row_dist_order(sub) if inner == "row_dist"
                       else np.concatenate(_hap_groups(sub)))
        out.append(g[inner_order])
    return np.concatenate(out) if out else np.arange(block.shape[0])


def encode_alleles(block: np.ndarray, encoding: str = "derived") -> np.ndarray:
    """Code the alleles of a ``(haplotypes, sites)`` matrix.

    ``derived`` keeps 0/1 (missing -> 0) as uint8. ``major_minor`` polarizes each site so
    the more common allele is -1 and the rarer +1, missing 0 (int8), as in Harris et al.
    2025. Row orderings that group identical haplotypes are unaffected by this choice,
    but the pixel values a CNN sees are not.
    """
    if encoding == "derived":
        return np.clip(block, 0, 1).astype(np.uint8)
    if encoding != "major_minor":
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
    b = np.asarray(block)
    missing = b < 0
    derived = (b > 0).astype(np.int8)
    # derived allele is the major one wherever it is carried by more than half the rows
    derived_is_major = derived.sum(axis=0) * 2 > derived.shape[0]
    out = np.where(derived.astype(bool) ^ derived_is_major, np.int8(1), np.int8(-1))
    out[missing] = 0
    return out.astype(np.int8)


def sort_haplotypes(block: np.ndarray, sort="haplotype_freq") -> np.ndarray:
    """Rearrange a ``(haplotypes, sites)`` matrix.

    ``sort`` is a name from :data:`SORTS`, **any callable** taking and returning a
    ``(haplotypes, sites)`` array, or a **sequence of these applied in order** (as the
    two-stage ``-a``/``-b`` rearrangement of Zhao and Alachiotis 2025, e.g.
    ``("corr", "daf")``). The SNP block itself is always the central K SNPs chosen by
    :func:`locus_image`.

    Named orderings: ``haplotype_freq`` groups identical haplotypes, most frequent first
    (:func:`haplotype_freq_sorting`); ``haplotype_freq_hamming`` orders
    those groups by Hamming distance to the most frequent haplotype within a frequency
    (:func:`haplotype_freq_sorting_hamming`); ``row_dist`` orders groups by
    Hamming distance to the most frequent haplotype (RowDist of Harris et al. 2025);
    ``center_window_freq`` and ``center_window_dist`` group rows by their central
    :data:`CENTER_SUBWINDOW`-SNP haplotype and order within each group over the full
    window (CenterWindowFreq and CenterWindowDist of the same work); ``freq`` sorts by
    number of derived alleles (:func:`freq_sorting`); ``corr`` by
    correlation with the most connected haplotype (:func:`corr_sorting`);
    ``daf`` and ``pcc_cols`` reorder COLUMNS (SNPs) instead of rows, by descending
    derived allele frequency (:func:`daf_sorting`) and by total correlation
    (:func:`pcc_column_sort_numba`); Zhao and Alachiotis 2025 find column orderings to
    matter more than row orderings. None leaves the matrix unchanged.
    """
    if sort is None or block.shape[1] == 0:
        return block
    if isinstance(sort, (tuple, list)):  # stages applied in order
        for stage in sort:
            block = sort_haplotypes(block, stage)
        return block
    if callable(sort):
        out = np.asarray(sort(block))
        if out.shape != block.shape:
            raise ValueError(
                f"custom sort returned shape {out.shape}, expected {block.shape} "
                "(a (haplotypes, sites) array with the same rows reordered)"
            )
        return out
    if sort == "haplotype_freq":
        return haplotype_freq_sorting(block.T)[0].T
    if sort == "haplotype_freq_hamming":
        return haplotype_freq_sorting_hamming(block.T)[0].T
    if sort == "row_dist":
        return block[_row_dist_order(block)]
    if sort in ("center_window_freq", "center_window_dist"):
        inner = "row_dist" if sort == "center_window_freq" else "haplotype_freq"
        return block[_center_window_order(block, CENTER_SUBWINDOW, inner)]
    if sort == "freq":
        return freq_sorting(block.astype(np.int64)).astype(block.dtype)
    if sort == "corr":
        return corr_sorting(block.astype(np.int64)).astype(block.dtype)
    if sort == "daf":
        return daf_sorting(block.astype(np.int64)).astype(block.dtype)
    if sort == "pcc_cols":
        return pcc_column_sort_numba(block.astype(np.int64)).astype(block.dtype)
    raise ValueError(f"sort must be a callable or one of {SORTS}, got {sort!r}")


def locus_rng(pos: np.ndarray, seed: int = 0) -> np.random.Generator:
    """Generator seeded by ``seed`` and the locus itself.

    Draws differ from locus to locus but are reproducible: a plain ``default_rng(seed)``
    per call would give every locus the same offset, and passing one generator through
    joblib workers would make the result depend on how the loci were batched.
    """
    return np.random.default_rng(
        np.array([seed, pos.size, int(pos[0]), int(pos[-1])], dtype=np.int64)
    )


def thin_indices(n: int, K: int, thin: str = "even", rng=None) -> np.ndarray:
    """Indices of ``K`` of ``n`` sites (``n >= K``), evenly spaced or drawn at random."""
    if thin == "random":
        return np.sort((rng or np.random.default_rng(0)).choice(n, K, replace=False))
    if thin != "even":
        raise ValueError(f"thin must be 'even' or 'random', got {thin!r}")
    return np.round(np.linspace(0, n - 1, K)).astype(np.int64)


def locus_image(
    hap: np.ndarray,
    pos: np.ndarray,
    center: float,
    K: int = 201,
    sort="haplotype_freq",
    encoding: str = "derived",
    span: float | None = None,
    thin: str = "even",
    offset: float = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """K-SNP image of one locus, around ``center``.

    Parameters
    ----------
    hap : (sites, haplotypes) 0/1 array, as returned by the fv.py readers.
    pos : (sites,) increasing physical positions.
    center : position the block is centred on (nearest SNP goes to column ``K // 2``).
    span : width in bp the image covers. ``None`` (default) takes the ``K`` CONTIGUOUS
        SNPs around ``center``, so the width is whatever they happen to span and shrinks
        where SNP density is high. A number instead keeps the SNPs within
        ``center +/- span / 2`` and subsamples them to ``K``, so every image covers the
        same sequence; this is how Harris et al. (2025) build theirs ("we randomly
        sampled 201 SNPs in the simulated window"). Use the simulated locus length to
        match the span of window-based feature vectors.
    thin : with ``span``, how to pick ``K`` of the sites in the window: ``"even"``
        (default, deterministic) or ``"random"``.
    offset : if non-zero, the window centre is drawn uniformly from ``center +/- offset``
        instead of sitting on ``center``, so the feature of interest is not always in the
        middle of the image. In simulations the sweep sits at the locus centre, so this
        is what a sliding-window scan actually sees; Harris et al. (2025) do not offset
        their training windows, but they do scan with windows advancing 10 SNPs at a time.
    seed : with ``offset`` or ``thin="random"``, the draw is seeded by this and by the
        locus itself (:func:`locus_rng`), so it is reproducible and differs per locus.

    Returns
    -------
    img : (haplotypes, K); uint8 0/1 or, with ``encoding="major_minor"``, int8 -1/+1
        (missing 0). Padded columns are 0 in both codings.
    img_pos : (K,) int64 SNP positions, -1 in padded columns.
    n_snps : number of real SNPs in the block.

    Notes
    -----
    A window holding fewer than ``K`` SNPs is zero-padded on both sides, so the image
    stays centred; ``n_snps`` records how many columns are real.
    """
    n_sites, n_hap = hap.shape
    dtype = np.uint8 if encoding == "derived" else np.int8
    img = np.zeros((n_hap, K), dtype=dtype)
    img_pos = np.full(K, -1, dtype=np.int64)
    if n_sites == 0:
        return img, img_pos, 0
    pos = np.asarray(pos)
    rng = locus_rng(pos, seed) if (offset or thin == "random") else None
    if offset:
        center = float(center) + float(rng.uniform(-offset, offset))
    if span is None:
        i0 = int(np.argmin(np.abs(pos.astype(np.float64) - center)))
        lo, hi = i0 - K // 2, i0 - K // 2 + K  # site range [lo, hi)
        a, b = max(lo, 0), min(hi, n_sites)
        block = np.asarray(hap[a:b]).T
        col, n_real = a - lo, b - a
        img_pos[col : col + n_real] = pos[a:b]
    else:
        a = int(np.searchsorted(pos, center - span / 2, side="left"))
        b = int(np.searchsorted(pos, center + span / 2, side="right"))
        idx = np.arange(a, b)  # every SNP within center +/- span / 2
        if idx.size > K:
            idx = idx[thin_indices(idx.size, K, thin, rng)]
        block = np.asarray(hap[idx]).T
        col, n_real = (K - idx.size) // 2, idx.size
        img_pos[col : col + n_real] = pos[idx]
    img[:, col : col + n_real] = sort_haplotypes(encode_alleles(block, encoding), sort)
    return img, img_pos, n_real


def trident_image(
    hap: np.ndarray,
    pos: np.ndarray,
    center: float,
    K: int = 499,
    dim: int | None = 299,
    window_length: int = 25,
    num_stride: int = 2,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """TrIdent-style image of one locus (Arnab et al. 2025).

    Sorted minor-allele counts of 25-SNP windows over the central K SNPs
    (:func:`trident_matrix`), resized to ``dim`` x ``dim`` with linear
    interpolation as in the paper (299 for InceptionResNetV2, 224 for VGG16/MobileNetV2);
    ``dim=None`` keeps the native ``haplotypes x n_windows`` matrix. Counts (0-25) are
    stored as uint8 multiplied by :data:`TRIDENT_SCALE`. Returns None when the locus has
    too few SNPs, as in TrIdent.
    """
    n_hap = hap.shape[1]
    img = trident_matrix(hap, pos, n_hap, window_length, num_stride, K, center)
    if img is None:
        return None
    if dim is not None:
        import cv2

        img = cv2.resize(img.astype(np.float64), (dim, dim), interpolation=cv2.INTER_LINEAR)
    img = np.clip(np.rint(np.asarray(img) * TRIDENT_SCALE), 0, 255).astype(np.uint8)
    return img, np.full(img.shape[1], -1, dtype=np.int64), K


def ms_image(ms_file: str, K: int = 201, sort="haplotype_freq",
             locus_length: float = 1.2e6, representation: str = "sorted",
             dim: int | None = 299, encoding: str = "derived",
             span: float | None = None, thin: str = "even", offset: float = 0,
             seed: int = 0
             ) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Image of one discoal/ms replicate centred on ``locus_length / 2``; None if malformed."""
    from .fv import parse_and_filter_ms

    parsed = parse_and_filter_ms(ms_file, seq_len=locus_length)
    if parsed is None:
        return None
    hap, _, _, _, pos, _ = parsed
    if hap.shape[0] == 0:  # segsites: 0, dropped by haplotype_images
        return np.zeros((0, K), np.uint8), np.full(K, -1, np.int64), 0
    if representation == "trident":
        return trident_image(hap, pos, locus_length / 2, K, dim)
    return locus_image(hap, pos, locus_length / 2, K, sort, encoding, span, thin,
                       offset, seed)


def _batch_ms(files, K, sort, locus_length, representation="sorted", dim=299,
              encoding="derived", span=None, thin="even", offset=0, seed=0):
    return [ms_image(f, K, sort, locus_length, representation, dim, encoding, span, thin,
                     offset, seed) for f in files]


def _batch_vcf(vcf_file, regions, K, sort, locus_length, representation="sorted", dim=299,
               encoding="derived", span=None, thin="even", offset=0, seed=0):
    from .fv import genome_reader

    read = genome_reader(vcf_file, region=None)
    if isinstance(read, dict):  # no biallelic sites
        return [None] * len(regions)
    hap, _, _, _, pos, _ = read
    out = []
    for r in regions:
        start = int(r.split(":")[-1].split("-")[0])
        lo, hi = np.searchsorted(pos, [start, start + locus_length])
        mid = start + locus_length / 2
        out.append(trident_image(hap[lo:hi], pos[lo:hi], mid, K, dim) if representation == "trident"
                   else locus_image(hap[lo:hi], pos[lo:hi], mid, K, sort, encoding,
                                   span, thin, offset, seed))
    return out


def _sweep_subclass(df: pl.DataFrame) -> pl.DataFrame:
    """Subclass label as in fv.pivot_feature_vectors (age 2000 gens, f_t 0.9, soft = f_i above min)."""
    sw = df["model"] != "neutral"
    f_min = df.filter(sw)["f_i"].min() if sw.any() else None
    label = (
        pl.when(pl.col("t") >= 2000).then(pl.lit("old")).otherwise(pl.lit("young"))
    )
    comp = pl.when(pl.col("f_t") >= 0.9).then(pl.lit("complete")).otherwise(pl.lit("incomplete"))
    kind = pl.when(pl.col("f_i") != f_min).then(pl.lit("soft")).otherwise(pl.lit("hard"))
    return df.with_columns(
        pl.when(pl.col("model") == "neutral")
        .then(pl.lit("neutral"))
        .otherwise(pl.concat_str([kind, label, comp], separator="_"))
        .alias("model")
    )


def haplotype_images(
    data_dir: str,
    vcf: bool = False,
    K: int = 201,
    sort: str | None = "haplotype_freq",
    nthreads: int = 1,
    locus_length: int = int(1.2e6),
    step_vcf: int = int(1e4),
    iters: dict | None = None,
    suffix: str | None = None,
    batch_size: int = 50,
    representation: str = "sorted",
    dim: int | None = 299,
    encoding: str = "derived",
    span: float | None = None,
    thin: str = "even",
    offset: float = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Haplotype images for every simulation or every VCF locus window.

    Parameters
    ----------
    data_dir : simulation folder (``neutral/``, ``sweep/``, ``params.txt.gz``) or, with
        ``vcf=True``, a directory of bgzipped, indexed single-contig VCF/BCF files.
    K : central SNPs per locus (image columns when ``representation="sorted"``).
    sort : haplotype (row) ordering: a name from :data:`SORTS` or a callable taking and
        returning a ``(haplotypes, sites)`` array. Ignored for TrIdent images. A callable
        must be importable in the worker processes (a module-level function, not a
        lambda), since images are built with joblib.
    representation : ``"sorted"`` (haplotype matrix, rows reordered by ``sort``) or
        ``"trident"`` (:func:`trident_image`; use ``K=499``, ``dim=299``).
    dim : TrIdent output size; None keeps the native matrix.
    encoding : pixel coding, one of :data:`ENCODINGS`; ignored for TrIdent images.
    span : bp the image covers; ``None`` takes the ``K`` contiguous central SNPs, a number
        subsamples the SNPs within ``+/- span / 2`` of the locus centre to ``K`` (see
        :func:`locus_image`). Pass ``locus_length`` to cover the whole simulated region.
    thin : ``"even"`` or ``"random"``, how ``span`` picks ``K`` SNPs.
    offset : bp; if non-zero the window centre of each simulated locus is drawn uniformly
        from ``+/- offset`` around the centre, so the sweep is not always in the middle of
        the image (see :func:`locus_image`). ``seed`` makes the draws reproducible.
    locus_length : simulated locus length; VCF windows have the same length and slide
        by ``step_vcf`` exactly as in ``fvs-vcf``.
    iters : simulations only, optional ``{"neutral": [...], "sweep": [...]}`` of 1-based
        replicate indices (the ``iter`` of the feature vectors) to process.
    suffix : if given (or ``""``), write ``hap{_suffix}.npy`` (images),
        ``hap_pos{_suffix}.npy`` (SNP positions) and ``hap{_suffix}.parquet`` (metadata)
        into ``data_dir``.

    Returns
    -------
    images : (N, haplotypes, K) uint8
    positions : (N, K) int64, -1 in padded columns
    meta : DataFrame with ``iter, s, t, f_i, f_t, mu, r, model, n_snps`` in image order;
        ``model`` is the sweep subclass as in the feature vectors, or ``neutral``; for
        VCFs ``iter`` is the region string, parameters are 0 and ``model`` is
        ``empirical``. Malformed or empty
        loci are dropped (they have no row).
    """
    from . import Parallel, delayed
    from .data import Data

    for stage in (sort if isinstance(sort, (tuple, list)) else [sort]):
        if not callable(stage) and stage not in SORTS:
            raise ValueError(f"sort must be a callable or one of {SORTS}, got {stage!r}")
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
    if representation not in ("sorted", "trident"):
        raise ValueError(f"representation must be 'sorted' or 'trident', got {representation!r}")

    if not vcf:
        sims, params = Data(data_dir).read_simulations()
        # Labels from the full set (soft = f_i above the minimum over all sweeps)
        params = _sweep_subclass(params.rename({"saf": "f_i", "eaf": "f_t"}))
        params = params.with_columns(
            pl.when(pl.col("model") == "neutral").then(pl.lit("neutral"))
            .otherwise(pl.lit("sweep")).alias("class"), pl.col("model").alias("subclass"))
        jobs, metas = [], []
        for cls, files in sims.items():
            p = params.filter(pl.col("class") == cls).with_row_index("iter", offset=1)
            p = p.with_columns(pl.Series("file", np.asarray(files)))
            if iters is not None:
                p = p.filter(pl.col("iter").is_in(list(iters.get(cls, []))))
            metas.append(p)
            f = p["file"].to_list()
            jobs += [(f[i : i + batch_size],) for i in range(0, len(f), batch_size)]
        meta = pl.concat(metas)
        with Parallel(n_jobs=nthreads, backend="loky") as par:
            res = par(delayed(_batch_ms)(b[0], K, sort, locus_length, representation, dim,
                                        encoding, span, thin, offset, seed)
                      for b in jobs)
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "*vcf.gz"))
                       + glob.glob(os.path.join(data_dir, "*bcf.gz")))
        if not files:
            raise FileNotFoundError(f"No VCF/BCF files found in directory: {data_dir}")
        regions = {f: Data(f, window_size=locus_length, step=step_vcf).read_vcf()["region"]
                   for f in files}
        meta = pl.DataFrame(
            {"iter": [r for f in files for r in regions[f]]}
        ).with_columns(
            *(pl.lit(0.0).alias(c) for c in ("s", "t", "f_i", "f_t", "mu", "r")),
            pl.lit("empirical").alias("model"),
        )
        with Parallel(n_jobs=min(nthreads, len(files)), backend="loky") as par:
            res = par(delayed(_batch_vcf)(f, regions[f], K, sort, locus_length, representation,
                                         dim, encoding, span, thin, offset, seed)
                      for f in files)

    flat = [x for batch in res for x in batch]
    ok = np.array([x is not None and x[2] > 0 for x in flat], dtype=bool)
    if not ok.any():
        raise ValueError("no locus produced an image")
    n_hap = {x[0].shape[0] for x, k in zip(flat, ok) if k}
    if len(n_hap) != 1:
        raise ValueError(f"loci differ in number of haplotypes: {sorted(n_hap)}")
    images = np.stack([x[0] for x, k in zip(flat, ok) if k])
    positions = np.stack([x[1] for x, k in zip(flat, ok) if k])
    meta = meta.filter(pl.Series(ok)).with_columns(
        pl.Series("n_snps", [x[2] for x, k in zip(flat, ok) if k], dtype=pl.Int64)
    )
    meta = meta.select("iter", "s", "t", "f_i", "f_t", "mu", "r", "model", "n_snps")
    if (~ok).sum():
        print(f"haplotype_images: dropped {(~ok).sum()} malformed or empty loci")
    if suffix is not None:
        sfx = f"_{suffix}" if suffix else ""
        np.save(os.path.join(data_dir, f"hap{sfx}.npy"), images)
        np.save(os.path.join(data_dir, f"hap_pos{sfx}.npy"), positions)
        meta.write_parquet(os.path.join(data_dir, f"hap{sfx}.parquet"))
    return images, positions, meta
