from . import np, Parallel, delayed, pl

from numba import njit, float64, int64


@njit
def calc_d(freq, core_freq, p):
    """Calculates the value of d, the similarity measure

    Parameters:
        freq: freq of SNP under consideration, ranges from 0 to 1
        core_freq: freq of coresite, ranges from 0 to 1
        p: the p parameter specifying sharpness of peak
    """
    xf = min(core_freq, 1.0 - core_freq)
    f = np.minimum(freq, 1.0 - freq)
    maxdiff = np.maximum(xf, 0.5 - xf)
    corr = ((maxdiff - np.abs(xf - f)) / maxdiff) ** p
    return corr


@njit
def calc_d_decay(freq, core_freq, distances, p, lambda_decay):
    """
    Calculates similarity measure d with frequency similarity + exponential distance decay.

    Parameters:
        freq: array of SNP frequencies
        core_freq: focal site frequency
        distances: array of distances from each SNP to core SNP (same shape as freq)
        p: sharpness parameter for frequency similarity
        lambda_decay: exponential decay rate (per bp or cM)
    """
    xf = min(core_freq, 1.0 - core_freq)
    f = np.minimum(freq, 1.0 - freq)
    maxdiff = np.maximum(xf, 0.5 - xf)
    corr = ((maxdiff - np.abs(xf - f)) / maxdiff) ** p
    decay = np.exp(-lambda_decay * distances)
    return corr, corr * decay


@njit
def omegai_nb(freqs, core_freq, n, p):
    """Calculates 9a

    Parameters:
        i:freq of SNP under consideration, ranges between 0 and 1
        snp_n: number of chromosomes used to calculate frequency of core SNP
        x: freq of coresite, ranges from 0 to 1
        p: the p parameter specifying sharpness of peak
    """
    n1num = calc_d(freqs, core_freq, p)
    n1denom = np.sum(calc_d(np.arange(1.0, n) / n, core_freq, p))
    n1 = n1num / n1denom
    n2 = (1.0 / (freqs * n)) / (np.sum(1.0 / np.arange(1.0, n)))
    return n1 - n2


@njit
def omegai_nb_decay(freqs, distances, core_freq, n, p, lambda_decay):
    """
    Computes ω_i with similarity and distance-based decay.
    """
    n1num, n1num_decay = calc_d_decay(freqs, core_freq, distances, p, lambda_decay)

    # Normalize using frequencies alone (distance = 0), to preserve original denominator structure
    denom_freqs = np.arange(1.0, n) / n
    denom_distances = np.zeros(denom_freqs.shape)

    # simulate no decay for normalization
    n1denom, n1denom_decay = calc_d_decay(
        denom_freqs, core_freq, denom_distances, p, lambda_decay
    )

    n1 = n1num / n1denom.sum()
    n1_decay = n1num_decay / n1denom_decay.sum()
    n2 = (1.0 / (freqs * n)) / np.sum(1.0 / np.arange(1.0, n))
    return n1 - n2, n1_decay - n2


@njit
def an_nb(n, core_freq, p):
    """
    Calculates alpha_n from Achaz 2009, eq 9b

        n: Sample size
        x: frequency, ranges from 0 to 1
        p: value of p parameter
    """
    i = np.arange(1, n)
    return np.sum(i * omegai_nb(i / n, core_freq, n, p) ** 2.0)


@njit
def fu_an_vec(n):
    """Calculates a_n from Fu 1995, eq 4 for a single integer value"""
    if n <= 1:
        return 0.0
    return np.sum(1.0 / np.arange(1.0, n))


@njit
def fu_Bn(n, i):
    """Calculates Beta_n(i) from Fu 1995, eq 5"""

    r = 2.0 * n / ((n - i + 1.0) * (n - i)) * (fu_an_vec(n + 1) - fu_an_vec(i)) - (
        2.0 / (n - i)
    )
    return r


def sigma(n, ij):
    """
    Returns sigma from eq 2 or 3 in Fu 1995

    Parameters:
        n: sample size
        ij: 2-d array of integers with 2 cols and no rows
    """
    np.seterr(all="raise")
    res = np.zeros(ij.shape[0])
    # i must be greater than j
    ij[:, 0], ij[:, 1] = ij.max(axis=1), ij.min(axis=1)
    ci = np.logical_and(ij[:, 0] == ij[:, 1], ij[:, 0] == n / 2)

    # Using eq 2
    if np.any(ci) > 0:
        res[ci] = 2.0 * (
            (fu_an_vec(n) - fu_an_vec(ij[ci, 0][0])) / (float(n) - ij[ci, 0][0])
        ) - (1.0 / (ij[ci, 0][0] ** 2.0))

    ci = np.logical_and(ij[:, 0] == ij[:, 1], ij[:, 0] < n / 2)
    if np.any(ci) > 0:
        res[ci] = fu_Bn(n, ij[ci, 0] + 1)

    ci = np.logical_and(ij[:, 0] == ij[:, 1], ij[:, 0] > n / 2)
    if np.any(ci) > 0:
        res[ci] = fu_Bn(n, ij[ci, 0]) - 1.0 / (ij[ci, 0] ** 2.0)

    # using eq 3
    ci = np.logical_and(ij[:, 0] > ij[:, 1], ij[:, 0] + ij[:, 1] == n)
    if np.any(ci) > 0:
        res[ci] = (fu_an_vec(n) - fu_an_vec(ij[ci, 0][0])) / (n - ij[ci, 0]) + (
            fu_an_vec(n) - fu_an_vec(ij[ci, 1][0])
        ) / (n - ij[ci, 1])
        -(fu_Bn(n, ij[ci, 0]) + fu_Bn(n, ij[ci, 1] + 1)) / 2.0 - 1.0 / (
            ij[ci, 0] * ij[ci, 1]
        )

    ci = np.logical_and(ij[:, 0] > ij[:, 1], ij[:, 0] + ij[:, 1] < n)
    if np.any(ci) > 0:
        res[ci] = (fu_Bn(n, ij[ci, 0] + 1) - fu_Bn(n, ij[ci, 0])) / 2.0

    ci = np.logical_and(ij[:, 0] > ij[:, 1], ij[:, 0] + ij[:, 1] > n)
    if np.any(ci) > 0:
        res[ci] = (fu_Bn(n, ij[ci, 1]) - fu_Bn(n, ij[ci, 1] + 1)) / 2.0 - (
            1.0 / (ij[ci, 0] * ij[ci, 1])
        )

    return res


@njit
def sigma_numba(n, _k, _j):
    num_rows = _k.shape[0]
    res = np.empty(num_rows)

    n_half = n / 2.0
    an_n = 0.0

    def harmonic(x):
        s = 0.0
        for m in range(1, int(x)):
            s += 1.0 / m
        return s

    an_n = harmonic(n)

    for idx in range(num_rows):
        k = _k[idx]
        j = _j[idx]
        if k < j:
            k, j = j, k
        if k == j:
            if k == n_half:
                val_k = harmonic(k)
                res[idx] = 2.0 * ((an_n - val_k) / (n - k)) - (1.0 / (k * k))
            elif k < n_half:
                s_k = harmonic(k + 1)
                term = 2.0 * n / ((n - (k + 1) + 1.0) * (n - (k + 1))) * (
                    an_n + 1.0 / n - s_k
                ) - (2.0 / (n - (k + 1)))
                res[idx] = term
            else:
                s_k = harmonic(k)
                term = 2.0 * n / ((n - k + 1.0) * (n - k)) * (an_n + 1.0 / n - s_k) - (
                    2.0 / (n - k)
                )
                res[idx] = term - 1.0 / (k * k)
        else:
            if k + j == n:
                val_k = harmonic(k)
                val_j = harmonic(j)
                term1 = (an_n - val_k) / (n - k)
                term2 = (an_n - val_j) / (n - j)
                bn_k_s = 2.0 * n / ((n - k + 1.0) * (n - k)) * (
                    an_n + 1.0 / n - val_k
                ) - (2.0 / (n - k))
                val_j1 = harmonic(j + 1)
                bn_j1_s = 2.0 * n / ((n - (j + 1) + 1.0) * (n - (k + 1))) * (
                    an_n + 1.0 / n - val_j1
                ) - (2.0 / (n - (j + 1)))
                res[idx] = term1 + term2 - (bn_k_s + bn_j1_s) / 2.0 - 1.0 / (k * j)
            elif k + j < n:
                val_k1 = harmonic(k + 1)
                val_k = harmonic(k)
                bn_kp1 = 2.0 * n / ((n - (k + 1) + 1.0) * (n - (k + 1))) * (
                    an_n + 1.0 / n - val_k1
                ) - (2.0 / (n - (k + 1)))
                bn_k = 2.0 * n / ((n - k + 1.0) * (n - k)) * (
                    an_n + 1.0 / n - val_k
                ) - (2.0 / (n - k))
                res[idx] = (bn_kp1 - bn_k) / 2.0
            else:
                val_j = harmonic(j)
                val_j1 = harmonic(j + 1)
                bn_j = 2.0 * n / ((n - j + 1.0) * (n - j)) * (
                    an_n + 1.0 / n - val_j
                ) - (2.0 / (n - j))
                bn_j1 = 2.0 * n / ((n - (j + 1) + 1.0) * (n - (j + 1))) * (
                    an_n + 1.0 / n - val_j1
                ) - (2.0 / (n - (j + 1)))
                res[idx] = (bn_j - bn_j1) / 2.0 - 1.0 / (i * j)

    return res


@njit
def sigma(n, ij):
    res = np.zeros(ij.shape[0])

    for k in range(ij.shape[0]):
        i = max(ij[k, 0], ij[k, 1])
        j = min(ij[k, 0], ij[k, 1])

        if i == j and 2 * i == n:
            res[k] = 2.0 * ((fu_an_vec(n) - fu_an_vec(i)) / (n - i)) - (1.0 / (i * i))
        elif i == j and 2 * i < n / 2.0:
            res[k] = fu_Bn(n, i + 1)
        elif i == j and i > n / 2.0:
            # print('3',i,j)
            res[k] = fu_Bn(n, i) - (1.0 / (i * i))
        elif i > j and (i + j == n):
            an_n = fu_an_vec(n)
            an_i = fu_an_vec(i)
            an_j = fu_an_vec(j)
            term1 = (an_n - an_i) / (n - i)
            term2 = (an_n - an_j) / (n - j)
            term3 = (fu_Bn(n, i) + fu_Bn(n, j + 1)) / 2.0
            term4 = 1.0 / (i * j)
            res[k] = (term1 + term2) - (term3 + term4)  # FIXED GROUPING!

        elif i > j and (i + j < n):
            res[k] = (fu_Bn(n, i + 1) - fu_Bn(n, i)) / 2.0
        elif i > j and (i + j > n):
            res[k] = (fu_Bn(n, j) - fu_Bn(n, j + 1)) / 2.0 - (1.0 / (i * j))

    return res


@njit
def Bn_nb(n, core_freq, p):
    """
    Returns Beta_N from Achaz 2009, eq 9c

    Parameters:
        n: Sample size
        x: frequency, ranges from 0 to 1
        p: value of p parameter
    """

    i = np.arange(1, n)
    n1 = np.sum(
        i**2.0
        * omegai_nb(i / n, core_freq, n, p) ** 2.0
        * sigma(n, np.column_stack((i, i)))
    )

    # coords = np.asarray([(j, i) for i in range(1, n) for j in range(1, i)])
    m = (n - 1) * (n - 2) // 2
    coords = np.empty((m, 2), dtype=np.int64)
    idx = 0
    for i in range(1, n):
        for j in range(1, i):
            coords[idx, 0] = j
            coords[idx, 1] = i
            idx += 1

    s2 = np.sum(
        coords[:, 0]
        * coords[:, 1]
        * omegai_nb(coords[:, 0] / n, core_freq, n, p)
        * omegai_nb(coords[:, 1] / n, core_freq, n, p)
        * sigma(n, coords)
    )

    n2 = 2.0 * s2
    return n1 + n2


def calc_thetaw_unfolded(snp_freq_list, num_ind):
    """Calculates watterson's theta

    Parameters:
        snp_freq_list: a list of frequencies, one for each SNP in the window,
            first column ranges from 1 to number of individuals, second columns is # individuals
        num_ind: number of individuals used to calculate the core site frequency
    """
    if snp_freq_list.size == 0:
        return 0

    a1 = np.sum(1.0 / np.arange(1, num_ind))

    thetaW = len(snp_freq_list[:, 0]) / a1
    return thetaW


def calc_t_unfolded(freqs, core_freq, n, p, theta, var_dic):
    """
    Using equation 8 from Achaz 2009

    Parameters:
        core_freq: freq of SNP under consideration, ranges from 1 to sample size
        snp_n: sample size of core SNP
        p: the p parameter specifying sharpness of peak
        theta: genome-wide estimate of the mutation rate
    """

    # x = float(core_freq)/snp_n

    num = np.sum(freqs * n * omegai_nb(freqs, core_freq, n, p))
    if not (n, core_freq, theta) in var_dic:
        denom = np.sqrt(
            an_nb(n, core_freq, p) * theta + Bn_nb(n, core_freq, p) * theta**2.0
        )
        var_dic[(n, core_freq, theta)] = denom
    else:
        denom = var_dic[(n, core_freq, theta)]
    return num / denom


@njit
def theta_watterson(ac, positions):
    # count segregating variants
    S = ac.shape[0]
    n = ac[0].sum()

    # (n-1)th harmonic number
    a1 = _harmonic_watterson(n)

    # calculate absolute value
    theta_hat_w_abs = S / a1

    # calculate value per base
    n_bases = positions[-1] - (positions[1] + 1)
    theta_hat_w = theta_hat_w_abs / n_bases

    return theta_hat_w_abs, theta_hat_w


@njit
def precompute_denoms(n, p, theta, omega_func):
    denom_array = np.zeros(n + 1)

    # Precompute shared structures
    i_vals = np.arange(1, n)
    diag_sigma = sigma(n, np.column_stack((i_vals, i_vals)))

    m = (n - 1) * (n - 2) // 2
    coords = np.empty((m, 2), dtype=np.int64)
    idx = 0
    for i in range(1, n):
        for j in range(1, i):
            coords[idx, 0] = j
            coords[idx, 1] = i
            idx += 1
    coords_i = coords[:, 0]
    coords_j = coords[:, 1]
    off_diag_sigma = sigma(n, coords)

    for cf in range(1, n + 1):
        x = cf / n
        omega = omega_func(i_vals / n, x, n, p)
        an = np.sum(i_vals * omega**2)

        omega_i = omega_func(coords_i / n, x, n, p)
        omega_j = omega_func(coords_j / n, x, n, p)
        s2 = np.sum(coords_i * coords_j * omega_i * omega_j * off_diag_sigma)

        b_n = np.sum(i_vals**2 * omega**2 * diag_sigma) + 2.0 * s2
        denom_array[cf] = np.sqrt(an * theta + b_n * theta**2)

    return denom_array, diag_sigma, off_diag_sigma


@njit
def calc_t_unfolded_cached(freqs, denom, core_freq, n, p, theta):
    num = np.sum(freqs * n * omegai_nb(freqs, core_freq, n, p))
    return num, num / denom


@njit
def calc_b_unfolded_custom(freqs, denom, core_freq, n, p, theta):
    num = np.sum(freqs * n * omegai_nb(freqs, core_freq, n, p))
    return num


@njit
def calc_t_unfolded_cached_decay(
    freqs, distances, denom, core_freq, n, p, theta, lambda_decay
):
    omega, omega_decay = omegai_nb_decay(
        freqs, distances, core_freq, n, p, lambda_decay
    )
    theta_beta = np.sum(freqs * n * omega)
    theta_beta_decay = np.sum(freqs * n * omega_decay)

    return theta_beta, theta_beta_decay, theta_beta / denom, theta_beta_decay / denom


@njit
def calc_t_unfolded_cached_decay(
    freqs, distances, denom, core_freq, n, p, theta, lambda_decay
):
    omega, omega_decay = omegai_nb_decay(
        freqs, distances, core_freq, n, p, lambda_decay
    )
    theta_beta = np.sum(freqs * n * omega)
    theta_beta_decay = np.sum(freqs * n * omega_decay)

    denom_decay = np.sqrt(np.var(freqs * n * omega_decay))

    return (
        theta_beta,
        theta_beta_decay,
        theta_beta / denom,
        theta_beta_decay / denom_decay,
    )


@njit
def _harmonic_watterson(n):
    w = 0.0
    for i in range(1, n):
        w += 1 / i
    return w


@njit
def find_win_indx(prev_start_i, prev_end_i, pos, snp_info, win_size):
    """Takes in the previous indices of the start_ing and end of the window,
    then returns the appropriate start_ing and ending index for the next SNP

    Parameters:
        prev_start_i: start_ing index in the array of SNP for the previous core SNP's window, inclusive
        prev_end_i: ending index in the array for the previous SNP's window, inclusive
        snp_i, the index in the array for the current SNP under consideration
        snp_info: the numpy array of all SNP locations & frequencies
    """

    win_start = pos - win_size / 2

    # array index of start of window, inclusive
    firstI = prev_start_i + np.searchsorted(
        snp_info[prev_start_i:, 0], win_start, side="left"
    )
    winEnd = pos + win_size / 2

    # array index of end of window, exclusive
    endI = (
        prev_end_i - 1 + np.searchsorted(snp_info[prev_end_i:, 0], winEnd, side="right")
    )
    return (firstI, endI)


@njit
def run_beta_window(snp_info, p=2, m=0.1, lambda_decay=2e-6, w=50000):
    S = int(snp_info.shape[0])
    n = int(snp_info[0, -1])
    theta = theta_watterson(snp_info[:, 2:4], snp_info[:, 0])[0]

    denom_array, sigma_term1, sigma_term2 = precompute_denoms(n, p, theta, omegai_nb)

    mask = (snp_info[:, 4] == n) & (snp_info[:, 1] < (1 - m)) & (snp_info[:, 1] > m)
    snp_info_masked = snp_info[mask]

    output = np.zeros((snp_info_masked.shape[0], 5))

    if w is None:
        for j, snp_i in enumerate(snp_info_masked):
            snp_set = np.concatenate((snp_info[:j], snp_info[j + 1 :]))
            core_freq = snp_i[1]
            denom = denom_array[int(round(core_freq * n))]
            distances = np.abs(snp_set[:, 0] - snp_i[0])

            # denom_decay = compute_denom_decay(core_freq, n, p, theta, lambda_decay, distances)
            # B,B_decay, T,T_decay = calc_t_unfolded_cached_decay(snp_set[:, 1], denom, core_freq, n, p, theta)

            B, B_decay, T, T_decay = calc_t_unfolded_cached_decay(
                snp_set[:, 1], distances, denom, core_freq, n, p, theta, lambda_decay
            )
            output[j] = np.array([snp_i[0], B, T, B_decay, T_decay])
    else:
        prev_start_i = 0
        prev_end_i = 0
        _idx = 0
        # for j, snp_i in enumerate(snp_info_masked):
        for j, snp_i in enumerate(snp_info):
            # snp_set = np.concatenate((snp_info[:j], snp_info[j + 1 :]))

            core_freq = snp_i[1]
            if not mask[j]:
                continue

            # print(prev_start_i, prev_end_i)
            sI, endI = find_win_indx(prev_start_i, prev_end_i, snp_i[0], snp_info, w)
            prev_start_i = sI
            prev_end_i = endI

            if endI == sI:
                B, T, B_decay, T_decay = 0, 0, 0, 0
            elif endI > sI:
                snp_set = np.concatenate(
                    (snp_info[sI:j], snp_info[(j + 1) : (endI + 1)])
                )
                distances = np.abs(snp_set[:, 0] - snp_i[0])
                denom = denom_array[int(core_freq * n)]
                # B,T = calc_t_unfolded_cached(snp_set[:, 1], denom, core_freq, n, p, theta * w)
                # B,T = calc_t_unfolded_cached(snp_set[:, 1], denom, core_freq, n, p, theta * w)
                B, B_decay, T, T_decay = calc_t_unfolded_cached_decay(
                    snp_set[:, 1],
                    distances,
                    denom,
                    core_freq,
                    n,
                    p,
                    theta * w,
                    lambda_decay,
                )
                # print(np.array([snp_i[0], B, T, B_decay, T_decay]))
            output[_idx] = np.array([snp_i[0], B, T, B_decay, T_decay])
            _idx += 1
            # output[j] = np.array([snp_i[0], B, T])

    return output


@njit
def run_beta_window_decay(snp_info, p=2, m=0.1):
    S = int(snp_info.shape[0])
    n = int(snp_info[0, -1])
    theta = theta_watterson(snp_info[:, 2:4], snp_info[:, 0])[0]

    denom_array = precompute_denoms(n, p, theta, omegai_nb)

    mask = (snp_info[:, 4] == n) & (snp_info[:, 1] < (1 - m)) & (snp_info[:, 1] > m)
    snp_info_masked = snp_info[mask]
    # mask_core = np.ones(snp_info.shape[0], dtype=bool_)
    output = np.zeros((snp_info_masked.shape[0], 2))
    output_decay = np.zeros((snp_info_masked.shape[0], 2))

    for j, snp_i in enumerate(snp_info_masked):
        # mask_core[j] = False
        # snp_set = snp_info[mask_core]
        snp_set = np.concatenate((snp_info[:j], snp_info[j + 1 :]))
        distances = np.abs(snp_set[:, 0] - snp_i[0])
        core_freq = snp_i[1]
        denom = denom_array[int(round(core_freq * n))]
        T = calc_t_unfolded_cached(snp_set[:, 1], denom, core_freq, n, p, theta)
        T_decay = calc_t_unfolded_cached_decay(
            snp_set[:, 1], distances, denom, core_freq, n, p, theta, 2e-6
        )
        # mask_core[j] = True
        output[j] = np.array([snp_i[0], T])
        output_decay[j] = np.array([snp_i[0], T_decay])
    print(
        pl.DataFrame(output)
        .filter(pl.col("column_0") > 5e5)
        .filter(pl.col("column_0") < 7e5)
    )
    print(
        pl.DataFrame(output_decay)
        .filter(pl.col("column_0") > 5e5)
        .filter(pl.col("column_0") < 7e5)
    )
    return output


def run_beta_window(snp_info, p=2, m=0.1):
    S = int(snp_info.shape[0])
    n = int(snp_info[0, -1])
    theta = theta_watterson(snp_info[:, 2:4], snp_info[:, 0])[0]

    # if len(snp_info) == 1 or n <= 3:
    #     T = 0
    #     # output.append((loc,round(T, 6)))
    #     return np.array([snp_info[0],0])

    # records variance calculations so don't need to be recalculated
    var_dic = {}

    # int(freq_count) != sample_n and freq < 1.0 - m and freq > m and sample_n > 3:
    mask = (snp_info[:, 4] == n) & (snp_info[:, 1] < (1 - m)) & (snp_info[:, 1] > m)
    snp_info_masked = snp_info[mask]
    mask_core = np.ones(snp_info.shape[0], dtype=bool)

    output = np.zeros((snp_info_masked.shape[0], 2))

    for j, snp_i in enumerate(snp_info_masked):
        # loc = snp_i[0]
        # freq = snp_i[1]
        # snp_set = np.delete(snp_info, snp_i, axis=0)
        mask_core[j] = False
        snp_set = snp_info[mask_core]
        core_freq = snp_i[1]
        T = calc_t_unfolded(snp_set[:, 1], core_freq, n, p, theta, var_dic)
        mask_core[j] = True
        output[j] = np.array([snp_i[0], T])

    # return output
    return pl.DataFrame(output)


@njit
def ncd1(position_masked, freqs, tf=0.5, w=3000, minIS=2):
    maf = np.minimum(freqs, 1 - freqs)
    w1 < -w / 2
    start = np.arange(position_masked[0], position_masked[-1], w1)
    end = start + w

    _ncd1 = []
    for i, j in zip(start, end):
        mask = (position_masked >= i) & (position_masked <= j)
        _tmp = ((maf[mask] - tf) ** 2).sum()
        IS = mask.sum()
        if IS < minIS:
            continue
        _ncd1.append(np.sqrt((_tmp) / IS))
    _ncd1 = np.array(_ncd1)
    return _ncd1
