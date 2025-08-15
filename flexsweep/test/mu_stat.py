
from jax import jit
import jax.numpy as jnp
from jax import config as jconfig
jconfig.update("jax_enable_x64", True)

# Updated μ_VAR
@njit
def compute_mu_var(start_idx, end_idx, snp_positions, D_ln, W_sz):
    l_start = snp_positions[start_idx]
    l_end = snp_positions[end_idx - 1]
    return ((l_end - l_start) / (D_ln * W_sz)) * snp_positions.shape[0]


@njit
def compute_mu_sfs(window, n, theta_W):
    if window.shape[0] == 0:
        return np.nan
    derived_counts = np.sum(window, axis=1)
    edge_mask = (derived_counts == 1) | (derived_counts == n - 1)
    n_edges = np.sum(edge_mask)
    W_sz = window.shape[0]
    return (n_edges / W_sz) * theta_W


@njit
def compute_mu_ld(start_idx: int, end_idx: int, r2: np.ndarray) -> float:
    """
    Numba‐accelerated version of mu_ld_from_r2matrix_lr.
    Operates entirely via loops over the relevant sub‐blocks.
    """
    length = end_idx - start_idx
    mid    = length // 2
    L      = mid
    R      = length - mid
    base   = start_idx

    # 1) within‐left block
    if L > 1:
        sumL = 0.0
        # sum strictly upper‐triangle
        for i in range(L):
            row_i = base + i
            for j in range(i+1, L):
                col_j = base + j
                sumL += r2[row_i, col_j]
        mean_ld_left = 2.0 * sumL / (L * (L - 1))
    else:
        mean_ld_left = 0.0

    # 2) within‐right block
    if R > 1:
        sumR = 0.0
        for i in range(R):
            row_i = base + mid + i
            for j in range(i+1, R):
                col_j = base + mid + j
                sumR += r2[row_i, col_j]
        mean_ld_right = 2.0 * sumR / (R * (R - 1))
    else:
        mean_ld_right = 0.0

    # 3) cross‐block
    if L > 0 and R > 0:
        sumC = 0.0
        for i in range(L):
            row_i = base + i
            for j in range(R):
                col_j = base + mid + j
                sumC += r2[row_i, col_j]
        mean_ld_cross = sumC / (L * R)
    else:
        mean_ld_cross = 1e-6

    # 4) final ratio, guarding zero‐division
    num = 0.5 * (mean_ld_left + mean_ld_right)
    den = mean_ld_cross if mean_ld_cross > 0.0 else 1e-6
    return num / den

@jit
def jax_dot(hap):
    return hap @ hap.T
@jit
def jax_dot_32(hap):
    _hap = hap.astype(jnp.float32)
    return hap @ hap.T


@jit
def compute_r2_matrix_upper_jax(hap):
    """
    Vectorized, JAX‐native computation of the strict upper triangle of
    the r^2 matrix for a (S x N) 0/1 haplotype array `hap`.

    Returns:
      r2_upper (S x S) float64 array where
        r2_upper[i,j] = r^2 between site i and j if i<j, else 0.
    """
    # 1) Cast to float64
    X = hap.astype(jnp.float64)
    S, N = X.shape

    # 2) Allele freqs p_i = mean over cols
    p = jnp.mean(X, axis=1)                    # (S,)

    # 3) Joint freqs P = (X @ X.T) / N
    P = jax_dot_32(X) / N                          # (S, S)

    # 4) Covariance D = P - p_i * p_j
    D = P - p[:, None] * p[None, :]            # (S, S)

    # 5) Variance denom = p*(1-p), and outer product
    var = p * (1.0 - p)                        # (S,)
    den = var[:, None] * var[None, :]          # (S, S)

    # 6) Compute r2 with guard: when den==0, set 0
    r2 = (D * D) / den
    r2 = jnp.where(den > 0.0, r2, 0.0)

    # 7) Strict upper triangle
    r2_upper = jnp.triu(r2, k=1)

    return r2_upper


@njit
def _harmonic_watterson(n):
    w = 0.0
    for i in range(1,n):
        w += 1/i
    return w


# @njit
def mu_stat(hap, snp_positions, r2_matrix,window_size=50):

    # full chromosome/region length
    D_ln = (snp_positions[-1] +1) - snp_positions[0]
    S, n = hap.shape

    theta_w_correction = _harmonic_watterson(n)
    # Match RAiSD -w option (default: 50)
    _window_size = window_size

    _iter_windows = list(range(S - _window_size + 1))
    mu_var_np= np.zeros(len(_iter_windows))
    mu_sfs_np= np.zeros(len(_iter_windows))
    mu_ld_np= np.zeros(len(_iter_windows))
    mu_total_np = np.zeros(len(_iter_windows))
    center_np = np.zeros(len(_iter_windows))

    for i in _iter_windows:
        # Indices for this window
        start_idx = i
        end_idx = i + _window_size  # exclusive in Python slicing

        # Window SNP positions (for plotting)
        center_pos = (snp_positions[start_idx] + snp_positions[end_idx - 1]) / 2
        window_positions = snp_positions[start_idx:end_idx]
        window = hap[start_idx:end_idx, :]

        if end_idx <= start_idx or end_idx > hap.shape[0]:
            mu_var_corr[i] = np.nan
            mu_sfs_corr[i] = np.nan
            mu_ld_corr[i] = np.nan
            mu_total_corr[i] = np.nan
            continue

        window = hap[start_idx:end_idx]
        mu_var = compute_mu_var(start_idx, end_idx, snp_positions, D_ln, end_idx - start_idx)
        mu_sfs = compute_mu_sfs(window,n,theta_w_correction)
        mu_ld = compute_mu_ld(start_idx,end_idx,r2_matrix)
        mu_total = mu_var * mu_sfs * mu_ld

        mu_var_np[i] = mu_var
        mu_sfs_np[i] = mu_sfs
        mu_ld_np[i] = mu_ld
        mu_total_np[i] = mu_total
        center_np[i] = center_pos

    df_mu = pl.DataFrame({'positions':center_np.astype(int),'mu_var':mu_var_np,'mu_sfs':mu_sfs_np,'mu_ld':mu_ld_np,'mu_total':mu_total_np})

    # return mu_var_np,mu_sfs_np,mu_ld_np,mu_total_np
    return df_mu



# raisd_full_df = pd.read_csv('/mnt/data/RAiSD_Report.ms_run.0', sep="\t", header=None)
# raisd_full_df = pd.read_csv('RAiSD_Report.ms_run', sep="\t", header=None)
raisd_full_df = pl.read_csv('RAiSD_Report.ms_run', separator="\t",skip_rows=1,has_header=False)


# Assign column names
raisd_full_df.columns = [
    "position", "window_start", "window_end",
    "mu_var_raisd", "mu_sfs_raisd", "mu_ld_raisd", "mu_total_raisd"
]


# Constants for normalization
hap_data='neutral_1.ms'
hap,r,p = ms_parser(hap_data)
S_total = hap.shape[0]  # total SNPs in the chromosome
n = hap.shape[1]        # number of individuals (haplotypes)

snp_positions = r[:,-1]


%time r2_matrix = compute_r2_matrix_upper_jax(hap)

# %time (mu_var_corr,mu_sfs_corr,mu_ld_corr,mu_total_corr) = mu_stat(hap,snp_positions,r2_matrix)
%time df_mu = mu_stat(hap,snp_positions,r2_matrix)

# Add corrected estimates to DataFrame
raisd_full_df = raisd_full_df.with_columns(pl.Series(mu_var_corr).alias("mu_var_ours"),
    pl.Series(mu_sfs_corr).alias("mu_sfs_ours"),
    pl.Series(mu_ld_corr).alias("mu_ld_ours"),
    pl.Series(mu_total_corr).alias("mu_total_ours"))


# Plot corrected comparisons
valid_df = raisd_full_df.drop_nans()

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

axes[0].plot(valid_df["position"], valid_df["mu_var_raisd"], label="RAiSD μ_VAR", color='blue')
axes[0].plot(valid_df["position"], valid_df["mu_var_ours"], label="Our μ_VAR", linestyle='--', color='blue')
axes[0].set_ylabel("μ_VAR")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(valid_df["position"], valid_df["mu_sfs_raisd"], label="RAiSD μ_SFS", color='green')
axes[1].plot(valid_df["position"], valid_df["mu_sfs_ours"], label="Our μ_SFS", linestyle='--', color='green')
axes[1].set_ylabel("μ_SFS")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(valid_df["position"], valid_df["mu_ld_raisd"], label="RAiSD μ_LD", color='orange')
axes[2].plot(valid_df["position"], valid_df["mu_ld_ours"], label="Our μ_LD", linestyle='--', color='orange')
axes[2].set_ylabel("μ_LD")
axes[2].legend()
axes[2].grid(True)

axes[3].plot(valid_df["position"], valid_df["mu_total_raisd"], label="RAiSD μ", color='red')
axes[3].plot(valid_df["position"], valid_df["mu_total_ours"], label="Our μ", linestyle='--', color='red')
axes[3].set_ylabel("μ Total")
axes[3].legend()
axes[3].grid(True)
axes[3].set_xlabel("Genomic Position")

fig.suptitle("Corrected μ Component Comparison: RAiSD vs Python", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
