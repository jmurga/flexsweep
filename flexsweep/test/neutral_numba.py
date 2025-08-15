import numpy as np
from numba import njit, float64, int64


@njit(float64[:](int64), cache=True)
def _harmonic_sums(n):
    a1 = 0.0
    a2 = 0.0
    for i in range(1, int(n)):
        inv = 1.0 / i
        a1 += inv
        a2 += inv * inv
    return np.array((a1, a2), dtype=np.float64)


@njit
def _harmonic_watterson(n):
    w = 0.0
    for i in range(1, n):
        w += 1 / n
    return w


@njit
def sfs_nb(dac, n):
    """
    Compute the site frequency spectrum given derived allele counts.

    Parameters
    ----------
    dac : int64[:]   # array of derived allele counts (0..n)
    n   : int64      # total number of chromosomes; if <= 0, inferred as max(dac)

    Returns
    -------
    sfs : int64[:]   # length n+1, where sfs[k] is count of sites with k derived alleles
    """
    # infer n if not provided or invalid
    if n <= 0:
        maxv = 0
        for i in range(dac.shape[0]):
            if dac[i] > maxv:
                maxv = dac[i]
        n = maxv

    # initialize spectrum
    s = np.zeros(n + 1, dtype=np.int64)

    # counts
    for i in range(dac.shape[0]):
        k = dac[i]
        if 0 <= k <= n:
            s[k] += 1
    return s


@njit
def theta_pi(ac):
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()
    denom_pairs = n * (n - 1.0)

    # pi = np.zeros(S)
    pi = np.zeros(S)
    for j in range(S):
        aj = ac[j, 1]
        pi[j] = 2.0 * aj * (n - aj) / denom_pairs
    return pi


@njit
def tajima_d(ac, min_sites=3):
    # count segregating variants
    S = hap.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()
    if S < min_sites:
        return np.nan

    # (n-1)th harmonic number
    an, bn = _harmonic_sums(n)

    # calculate Watterson's theta (absolute value)
    theta_hat_w_abs = S / an

    # calculate mean pairwise difference
    mpd = theta_pi(ac)

    # calculate theta_hat pi (sum differences over variants)
    theta_hat_pi_abs = mpd.sum()

    # N.B., both theta estimates are usually divided by the number of
    # (accessible) bases but here we want the absolute difference
    d = theta_hat_pi_abs - theta_hat_w_abs

    # calculate the denominator (standard deviation)
    a2 = np.sum(1 / (np.arange(1, n) ** 2))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - (1 / an)
    c2 = b2 - ((n + 2) / (an * n)) + (a2 / (an**2))
    e1 = c1 / an
    e2 = c2 / (an**2 + a2)
    d_stdev = np.sqrt((e1 * S) + (e2 * S * (S - 1)))

    # finally calculate Tajima's D
    D = d / d_stdev

    return D


@njit(float64(int64[:]), cache=True)
def achaz_y(fs):
    n = fs.shape[0] - 1
    if n < 3:
        return np.nan
    a1, a2 = _harmonic_sums(n)
    a1m1 = a1 - 1.0
    ff = (n - 2.0) / (n * a1m1)
    inv_n = 1.0 / n
    inv_n1 = 1.0 / (n - 1.0)
    inv_n2 = 1.0 / (n - 2.0)
    n2 = n * n
    alpha = (
        ff * ff * a1m1
        + ff
        * (
            a1 * (4.0 * (n + 1.0) * inv_n1 * inv_n1)
            - 2.0 * (n + 1.0) * (n + 2.0) * inv_n * inv_n1
        )
        - a1 * 8.0 * (n - 1.0) * inv_n * inv_n1 * inv_n1
        + (2.0 * n2 + 60.0 * n + 12.0) * (inv_n * inv_n) * (1.0 / 3.0) * inv_n1
    )
    beta = (
        ff * ff * (a2 + a1 * (4.0 * inv_n1 * inv_n2) - 4.0 * inv_n2)
        + ff
        * (
            -a1 * (4.0 * (n + 2.0) * inv_n * inv_n1 * inv_n2)
            - ((n2 - 3.0 * n2 - 16.0 * n + 20.0) * inv_n * inv_n1 * inv_n2)
        )
        + a1 * (8.0 * inv_n * inv_n1 * inv_n2)
        + (2.0 * (n2 * n2 - n2 * n - 17.0 * n2 - 42.0 * n + 72.0))
        * (inv_n * inv_n)
        * (inv_n1 * inv_n2)
        * (1.0 / 9.0)
    )
    y = fs.copy()
    y[0] = y[1] = y[n] = 0.0
    S = 0.0
    pi_sum = 0.0
    for i in range(1, n + 1):
        yi = y[i]
        if i > 1 and i < n:
            S += yi
        if i < n:
            pi_sum += yi * i * (n - i)
    pi_hat = pi_sum / (n * (n - 1.0) * 0.5)
    that = S / a1m1
    that_sq = S * (S - 1.0) / (a1m1 * a1m1)
    return (pi_hat - ff * S) / np.sqrt(alpha * that + beta * that_sq)


@njit
def fay_wu_h_norm_si(ac, position):
    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    fs = sfs_nb(ac[:, 1], n)[1:-1]

    i_arr = np.arange(1, int(n))
    a1 = np.sum(1.0 / i_arr)
    bn = np.sum(1.0 / (i_arr * i_arr)) + 1.0 / (n * n)
    theta_w = S / a1
    pi = 0.0
    th = 0.0
    for k in range(1, int(n)):
        si = fs[k - 1]
        pi += (2 * si * k * (n - k)) / (n * (n - 1.0))
        th += (2 * si * k * k) / (n * (n - 1.0))
    tl = 0.0
    for k in range(1, int(n)):
        tl += k * fs[k - 1]
    tl /= n - 1.0
    var1 = (n - 2.0) / (6.0 * (n - 1.0)) * theta_w
    theta_sq = S * (S - 1.0) / (a1 * a1 + bn)

    var2 = (
        ((18 * n * n * (3 * n + 2) * bn) - (88 * n * n * n + 9 * n * n - 13 * n - 6))
        / (9.0 * n * (n - 1.0) * (n - 1.0))
    ) * theta_sq
    h = pi - th
    return th, h, h / np.sqrt(var1 + var2)


@njit
def zeng_e_si(ac):
    # count segregating variants
    S = hap.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    fs = sfs_nb(ac[:, 1], n)[1:-1]

    i_arr = np.arange(1, int(n))
    a1 = np.sum(1.0 / i_arr)
    bn = np.sum(1.0 / (i_arr * i_arr))
    theta_w = S / a1
    tl = 0.0
    for k in range(1, int(n)):
        tl += k * fs[k - 1]
    tl /= n - 1.0
    theta_sq = S * (S - 1.0) / (a1 * a1 + bn)
    var1 = (n / (2.0 * (n - 1.0)) - 1.0 / a1) * theta_w
    var2 = (
        bn / a1 / a1
        + 2 * (n / (n - 1.0)) * (n / (n - 1.0)) * bn
        - 2 * (n * bn - n + 1) / ((n - 1.0) * a1)
        - (3 * n + 1) / (n - 1.0)
    ) * theta_sq
    return (tl - theta_w) / np.sqrt(var1 + var2)


@njit
def fuli_f_star_nb(ac):
    """Calculates Fu and Li's D* statistic"""
    # count segregating variants
    S = hap.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    an, bn = _harmonic_sums(n)
    an1 = an + np.true_divide(1, n)

    denom_pairs = n * (n - 1.0)
    pi = 0.0
    for j in range(S):
        aj = ac[j, 1]
        pi += 2.0 * aj * (n - aj) / denom_pairs

    ss = (ac[:, 1] == 1).sum()

    vfs = (
        (
            (2 * (n**3.0) + 110.0 * (n**2.0) - 255.0 * n + 153)
            / (9 * (n**2.0) * (n - 1.0))
        )
        + ((2 * (n - 1.0) * an) / (n**2.0))
        - ((8.0 * bn) / n)
    ) / ((an**2.0) + bn)
    ufs = (
        (
            n / (n + 1.0)
            + (n + 1.0) / (3 * (n - 1.0))
            - 4.0 / (n * (n - 1.0))
            + ((2 * (n + 1.0)) / ((n - 1.0) ** 2)) * (an1 - ((2.0 * n) / (n + 1.0)))
        )
        / an
    ) - vfs

    num = pi - ((n - 1.0) / n) * ss
    den = np.sqrt(ufs * S + vfs * (S * S))
    return num / den


@njit
def fuli_f_nb(ac):
    # count segregating variants
    S = hap.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()
    an, bn = _harmonic_sums(n)
    an1 = an + 1.0 / n

    ss = (ac[:, 1] == 1).sum()

    denom_pairs = n * (n - 1.0)
    pi = 0.0
    for j in range(S):
        aj = ac[j, 1]
        pi += 2.0 * aj * (n - aj) / denom_pairs

    if n == 2:
        cn = 1
    else:
        cn = 2.0 * (n * an - 2.0 * (n - 1.0)) / ((n - 1.0) * (n - 2.0))

    v = (
        cn + 2.0 * (np.power(n, 2) + n + 3.0) / (9.0 * n * (n - 1.0)) - 2.0 / (n - 1.0)
    ) / (np.power(an, 2) + bn)
    u = (
        1.0
        + (n + 1.0) / (3.0 * (n - 1.0))
        - 4.0 * (n + 1.0) / np.power(n - 1, 2) * (an1 - 2.0 * n / (n + 1.0))
    ) / an - v
    F = (pi - ss) / np.sqrt(u * S + v * np.power(S, 2))

    return F


@njit
def fuli_d_star_nb(ac):
    """Calculates Fu and Li's D* statistic"""

    # count segregating variants
    S = hap.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    an, bn = _harmonic_sums(n)
    an1 = an + np.true_divide(1, n)

    cn = 2 * (((n * an) - 2 * (n - 1))) / ((n - 1) * (n - 2))
    dn = (
        cn
        + np.true_divide((n - 2), ((n - 1) ** 2))
        + np.true_divide(2, (n - 1)) * (3.0 / 2 - (2 * an1 - 3) / (n - 2) - 1.0 / n)
    )

    vds = (
        ((n / (n - 1.0)) ** 2) * bn
        + (an**2) * dn
        - 2 * (n * an * (an + 1)) / ((n - 1.0) ** 2)
    ) / (an**2 + bn)
    uds = ((n / (n - 1.0)) * (an - n / (n - 1.0))) - vds

    ss = (ac[:, 1] == 1).sum()
    Dstar1 = ((n / (n - 1.0)) * S - (an * ss)) / (uds * S + vds * (S ^ 2)) ** 0.5
    return Dstar1


@njit
def fuli_d_nb(ac):
    # count segregating variants
    S = hap.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    an, bn = _harmonic_sums(n)

    ss = (ac[:, 1] == 1).sum()

    if n == 2:
        cn = 1
    else:
        cn = 2.0 * (n * an - 2.0 * (n - 1.0)) / ((n - 1.0) * (n - 2.0))

    v = 1.0 + (np.power(an, 2) / (bn + np.power(an, 2))) * (cn - (n + 1.0) / (n - 1.0))
    u = an - 1.0 - v
    D = (S - ss * an) / np.sqrt(u * S + v * np.power(S, 2))
    return D


@njit
def h_stats_numba(haplo):
    """
    Compute H1, H12, H123, and H2/H1 on a (L x n) uint8 matrix of 0/1 haplotypes.
    Returns a tuple (H1, H12, H123, H2_H1).
    """
    L, n = haplo.shape

    # 1) Count distinct columns via a rolling hash in a typed.Dict
    counts_dict = Dict.empty(
        key_type=int64,
        value_type=int64,
    )
    for j in range(n):
        h = np.int64(146527)  # arbitrary seed
        for i in range(L):
            # simple mix: multiply by prime, xor in the bit
            h = h * np.int64(1000003) ^ np.int64(haplo[i, j])
        # increment count
        if h in counts_dict:
            counts_dict[h] += 1
        else:
            counts_dict[h] = 1

    # 2) Move counts into an array
    m = len(counts_dict)
    counts = np.empty(m, dtype=np.int64)
    idx = 0
    for key in counts_dict:
        counts[idx] = counts_dict[key]
        idx += 1

    # 3) Convert to frequencies
    freqs = counts.astype(np.float64) / n

    # 4) Sort descending
    # Numba supports np.sort in nopython mode
    freqs = np.sort(freqs)[::-1]

    # 5) Extract top three (or 0)
    p1 = freqs[0] if freqs.size > 0 else 0.0
    p2 = freqs[1] if freqs.size > 1 else 0.0
    p3 = freqs[2] if freqs.size > 2 else 0.0

    # 6) Compute H1 = sum p_i^2
    H1 = 0.0
    for i in range(freqs.size):
        H1 += freqs[i] * freqs[i]

    # 7) Compute H12 = (p1+p2)^2 + sum_{i>=3} p_i^2
    H12 = (p1 + p2) * (p1 + p2)
    for i in range(2, freqs.size):
        H12 += freqs[i] * freqs[i]

    # 8) Compute H123 = (p1+p2+p3)^2 + sum_{i>=4} p_i^2
    H123 = (p1 + p2 + p3) * (p1 + p2 + p3)
    for i in range(3, freqs.size):
        H123 += freqs[i] * freqs[i]

    # 9) Compute H2/H1
    H2 = H1 - p1 * p1
    H2_H1 = H2 / H1 if H1 > 0.0 else np.nan
    return H1, H12, H123, H2_H1


@njit
def cluster_haplotypes_numba(haplotypes: np.ndarray, threshold: float = 0.8):
    L, n = haplotypes.shape
    # -1 means unassigned
    assignments = -1 * np.ones(n, dtype=np.int64)

    # pre-allocate for the worst case (n clusters)
    rep_indices = np.empty(n, dtype=np.int64)
    n_clusters = 0

    for i in range(n):
        if assignments[i] != -1:
            continue
        # new cluster seeded by i
        rep_indices[n_clusters] = i
        assignments[i] = n_clusters
        hap_i = haplotypes[:, i]
        # assign all j > i that meet the threshold
        for j in range(i + 1, n):
            if assignments[j] != -1:
                continue
            hap_j = haplotypes[:, j]
            identical = 0
            different = 0
            # count identical (1∧1) and differences
            for k in range(L):
                v1 = hap_i[k]
                v2 = hap_j[k]
                if v1 == 1 and v2 == 1:
                    identical += 1
                elif v1 != v2:
                    different += 1
            total = identical + different
            if total > 0 and (different / total) <= (1.0 - threshold):
                assignments[j] = n_clusters
        n_clusters += 1

    # shrink rep_indices to actual clusters
    rep_indices = rep_indices[:n_clusters]

    # build grouped_matrix of prototypes
    grouped_matrix = np.empty((L, n_clusters), dtype=haplotypes.dtype)
    for c in range(n_clusters):
        grouped_matrix[:, c] = haplotypes[:, rep_indices[c]]

    return assignments, rep_indices, grouped_matrix
