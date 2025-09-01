# from jax import config

# config.update("jax_enable_x64", True)
# config.update("jax_platform_name", "cpu")

# import jax.numpy as jnp
# from jax import jit
# from jax.lax import fori_loop

# @jit
# def jax_dot(hap):
#     _hap = hap.astype(jnp.float32)
#     return hap @ hap.T

# def haf_top_jax(hap, pos, cutoff=0.1, start=None, stop=None, window_size=None):
#     """
#     Calculates the Haplotype Allele Frequency (HAF) for the top proportion of haplotypes,
#     which is a measure used to summarize haplotype diversity. The function computes the
#     HAF statistic for a filtered set of variants and returns the sum of the top `cutoff`
#     proportion of the HAF values.

#     Parameters
#     ----------
#     hap : numpy.ndarray
#         A 2D array where each row represents a SNP (variant), and each column represents
#         a haplotype for an individual. The entries are expected to be binary (0 or 1),
#         indicating the presence of ancestral or derived alleles.

#     pos : numpy.ndarray
#         A 1D array of physical positions corresponding to the SNPs in the `hap` matrix.
#         The length of `pos` should match the number of rows in `hap`.

#     cutoff : float, optional (default=0.1)
#         The proportion of HAF values to exclude from the top and bottom when calculating the final HAF score.
#         For example, a `cutoff` of 0.1 excludes the lowest 10% and highest 10% of HAF values,
#         and the function returns the sum of the remaining HAF values.

#     start : float or None, optional (default=None)
#         The starting physical position (in base pairs) for the genomic region of interest.
#         If provided, only SNPs at or after this position are included in the calculation.

#     stop : float or None, optional (default=None)
#         The ending physical position (in base pairs) for the genomic region of interest.
#         If provided, only SNPs at or before this position are included in the calculation.

#     Returns
#     -------
#     haf_top : float
#         The sum of the top `cutoff` proportion of HAF values, which represents the
#         higher end of the haplotype diversity distribution within the specified region.

#     Notes
#     -----
#     - The function first filters the SNPs by the specified genomic region (using `start` and `stop`).
#     - HAF (Haplotype Allele Frequency) is computed by summing the pairwise dot product of
#       haplotypes and dividing by the total number of haplotypes.
#     - The HAF values are sorted, and the top proportion (based on the `cutoff`) is returned.

#     """
#     if start is not None or stop is not None:
#         loc = (pos >= start) & (pos <= stop)
#         pos = pos[loc]
#         hap = hap[loc, :]
#     elif window_size is not None:
#         loc = (pos >= (6e5 - window_size // 2)) & (pos <= (6e5 + window_size // 2))
#         hap = hap[loc, :]

#     freqs = hap.sum(axis=1) / hap.shape[1]
#     hap_tmp = hap[(freqs > 0) & (freqs < 1)]
#     # haf_num = (np.dot(hap_tmp.T, hap_tmp) / hap.shape[1]).sum(axis=1)
#     haf_num = (jax_dot(hap_tmp.T) / hap.shape[1]).sum(axis=1)
#     haf_den = hap_tmp.sum(axis=0)
#     # haf = np.sort(haf_num / haf_den)

#     if 0 in haf_den:
#         mask_zeros = haf_den != 0
#         haf = np.full_like(haf_num, np.nan, dtype=np.float64)
#         haf[mask_zeros] = haf_num[mask_zeros] / haf_den[mask_zeros]
#         haf = jnp.sort(haf)
#     else:
#         haf = jnp.sort(haf_num / haf_den)

#     if cutoff <= 0 or cutoff >= 1:
#         cutoff = 1
#     idx_low = int(cutoff * haf.size)
#     idx_high = int((1 - cutoff) * haf.size)

#     # 10% higher
#     return haf[idx_high:].sum()


# def Ld_jax(r_2) -> tuple:
#     """
#     Compute Kelly Zns statistic (1997) and omega_max. Average r2
#     among every pair of loci in the genomic window.

#     Args:hap (numpy.ndarray): 2D array representing haplotype data. Rows correspond to different mutations, and columnscorrespond to chromosomes.
#     pos (numpy.ndarray): 1D array representing the positions of mutations.
#     min_freq (float, optional): Minimum frequency threshold. Default is 0.05.
#     max_freq (float, optional): Maximum frequency threshold. Default is 1.
#     window (int, optional): Genomic window size. Default is 500000.

#     Returns:tuple: A tuple containing two values:
#     - kelly_zns (float): Kelly Zns statistic.
#     - omega_max (float): Nielsen omega max.
#     """

#     # r2_matrix = r2_torch(hap_filter)

#     S = r_2.shape[0]
#     zns = r_2.sum() / comb(S, 2)
#     # Index combination to iter
#     omega_max = omega_linear_correct_jax(r_2)

#     # return zns, 0
#     return zns, omega_max


# @jit
# def compute_r2_matrix_upper_jax(hap):
#     """
#     Vectorized, JAX‐native computation of the strict upper triangle of
#     the r^2 matrix for a (S x N) 0/1 haplotype array `hap`.

#     Returns:
#       r2_upper (S x S) float64 array where
#         r2_upper[i,j] = r^2 between site i and j if i<j, else 0.
#     """
#     # 1) Cast to float64
#     X = hap.astype(jnp.float64)
#     S, N = X.shape

#     # 2) Allele freqs p_i = mean over cols
#     p = jnp.mean(X, axis=1)  # (S,)

#     # 3) Joint freqs P = (X @ X.T) / N
#     P = (X @ X.T) / N  # (S, S)

#     # 4) Covariance D = P - p_i * p_j
#     D = P - p[:, None] * p[None, :]  # (S, S)

#     # 5) Variance denom = p*(1-p), and outer product
#     var = p * (1.0 - p)  # (S,)
#     den = var[:, None] * var[None, :]  # (S, S)

#     # 6) Compute r2 with guard: when den==0, set 0
#     r2 = (D * D) / den
#     r2 = jnp.where(den > 0.0, r2, 0.0)

#     # 7) Strict upper triangle
#     r2_upper = jnp.triu(r2, k=1)

#     return r2_upper


# @jit
# def omega_linear_correct_jax(r2_matrix):
#     S = r2_matrix.shape[0]
#     if S < 3:
#         return 0.0

#     # Build row_sum[i] = sum_{j>i} r2[i,j]
#     #       col_sum[j] = sum_{i<j} r2[i,j]
#     mask = jnp.triu(jnp.ones((S, S)), k=1)
#     row_sum = jnp.sum(r2_matrix * mask, axis=1)
#     col_sum = jnp.sum(r2_matrix * mask, axis=0)
#     total = jnp.sum(row_sum)

#     # prefix_L[_l] = sum_{i<j<_l} r2[i,j] (cumsum over col_sum)
#     prefix_L = jnp.zeros(S)
#     prefix_L = prefix_L.at[1:].set(jnp.cumsum(col_sum[:-1]))

#     # suffix_R[_l] = sum_{_l≤i<j} r2[i,j] (reverse cumsum over row_sum)
#     suffix_R = jnp.zeros(S + 1)
#     suffix_R = suffix_R.at[:-1].set(jnp.cumsum(row_sum[::-1])[::-1])

#     # Sweep _l = 3..S-3
#     def omega_body(_l, state):
#         omega_max, omega_argmax = state
#         sum_L = prefix_L[_l]
#         sum_R = suffix_R[_l]
#         sum_LR = total - sum_L - sum_R
#         cond = sum_LR > 0.0
#         denom_L = (_l * (_l - 1) / 2.0) + ((S - _l) * (S - _l - 1) / 2.0)
#         denom_R = _l * (S - _l)
#         _omega = jnp.where(
#             cond,
#             ((sum_L + sum_R) / denom_L) / (sum_LR / denom_R),
#             -jnp.inf,
#         )
#         update = _omega > omega_max
#         omega_max = jnp.where(update, _omega, omega_max)
#         omega_argmax = jnp.where(update, _l + 2, omega_argmax)
#         return omega_max, omega_argmax

#     omega_max = 0.0
#     omega_argmax = -1.0

#     # Use JAX's fori_loop for efficient iteration
#     (omega_max, omega_argmax) = fori_loop(
#         3, S - 2, omega_body, (omega_max, omega_argmax)
#     )

#     return omega_max


# def filter_gt(hap, rec_map, region=None):
#     """
#     Convert the 2d numpy haplotype matrix to HaplotypeArray from scikit-allel and change type np.int8. It filters and processes the haplotype matrix based on a recombination map and
#     returns key information for further analysis such as allele frequencies and physical positions.

#     Parameters
#     ----------
#     hap : array-like, HaplotypeArray
#         The input haplotype data which can be in one of the following forms:
#         - A `HaplotypeArray` object.
#         - A genotype matrix (as a numpy array or similar).

#     rec_map : numpy.ndarray
#         A 2D numpy array representing the recombination map, where each row corresponds
#         to a genomic variant and contains recombination information. The third column (index 2)
#         of the recombination map provides the physical positions of the variants.

#     Returns
#     -------
#     tuple
#         A tuple containing the following elements:
#         - hap_01 (numpy.ndarray): The filtered haplotype matrix, with only biallelic variants.
#         - ac (AlleleCounts): An object that stores allele counts for the filtered variants.
#         - biallelic_mask (numpy.ndarray): A boolean mask indicating which variants are biallelic.
#         - hap_int (numpy.ndarray): The haplotype matrix converted to integer format (int8).
#         - rec_map_01 (numpy.ndarray): The recombination map filtered to include only biallelic variants.
#         - position_masked (numpy.ndarray): The physical positions of the biallelic variants.
#         - sequence_length (int): An arbitrary sequence length set to 1.2 million bases (1.2e6).
#         - freqs (numpy.ndarray): The frequencies of the alternate alleles for each biallelic variant.
#     """
#     try:
#         # Avoid unnecessary conversion if hap is already a HaplotypeArray
#         if not isinstance(hap, HaplotypeArray):
#             hap = HaplotypeArray(
#                 hap if isinstance(hap, np.ndarray) else hap.genotype_matrix()
#             )
#     except:
#         hap = HaplotypeArray(load(hap).genotype_matrix())

#     # positions = rec_map[:, -1]
#     # physical_position = rec_map[:, -2]

#     # HAP matrix centered to analyse whole chromosome
#     hap_01, ac, biallelic_mask = filter_biallelics(hap)
#     sequence_length = int(1.2e6)

#     rec_map_01 = rec_map[biallelic_mask]

#     mask_dup = (~pl.DataFrame(rec_map_01)[:, -1].is_duplicated()).to_numpy()

#     hap_01 = hap_01[mask_dup]
#     rec_map_01 = rec_map_01[mask_dup]
#     position_masked = rec_map_01[:, -1].astype(int)
#     genetic_position_masked = rec_map_01[:, -2]

#     ac = ac[mask_dup]

#     return (
#         hap_01,
#         rec_map_01,
#         ac,
#         biallelic_mask,
#         position_masked,
#         genetic_position_masked,
#     )


# def filter_biallelics(hap: HaplotypeArray) -> tuple:
#     """
#     Filter out non-biallelic loci from the haplotype data.

#     Args:
#         hap (allel.HaplotypeArray): Haplotype data represented as a HaplotypeArray.

#     Returns:
#         tuple: A tuple containing three elements:
#             - hap_biallelic (allel.HaplotypeArray): Filtered biallelic haplotype data.
#             - ac_biallelic (numpy.ndarray): Allele counts for the biallelic loci.
#             - biallelic_mask (numpy.ndarray): Boolean mask indicating biallelic loci.
#     """
#     ac = hap.count_alleles()
#     biallelic_mask = ac.is_biallelic_01()

#     # Use a subset to filter directly, minimizing intermediate memory usage
#     hap_biallelic = hap.subset(biallelic_mask)

#     ac_biallelic = ac[biallelic_mask]

#     return (hap_biallelic.values, ac_biallelic.values, biallelic_mask)


# def ms_parser(ms_file, param=None, seq_len=1.2e6):
#     """Read a ms file and output the positions and the genotypes.
#     Genotypes are a numpy array of 0s and 1s with shape (num_segsites, num_samples).
#     """

#     assert (
#         ms_file.endswith(".out")
#         or ms_file.endswith(".out.gz")
#         or ms_file.endswith(".ms")
#         or ms_file.endswith(".ms.gz")
#     )

#     open_function = gzip.open if ms_file.endswith(".gz") else open

#     with open_function(ms_file, "rt") as file:
#         file_content = file.read()

#     # Step 2: Split by pattern (e.g., `---`)
#     pattern = r"//"
#     partitions = re.split(pattern, file_content)

#     if len(partitions) == 1:
#         warn(f"File {ms_file} is malformed.")
#         return None
#     else:
#         positions = []
#         haps = []
#         rec_map = []
#         for r in partitions[1:]:
#             # Read in number of segregating sites and positions
#             data = []
#             for line in r.splitlines()[1:]:
#                 if line == "":
#                     continue
#                 # if "discoal" in line or "msout" in line:
#                 # seq_len = int(line.strip().split()[3])
#                 if line.startswith("segsites"):
#                     num_segsites = int(line.strip().split()[1])
#                     if num_segsites == 0:
#                         continue
#                         #     # Shape of data array for 0 segregating sites should be (0, 1)
#                         # return np.array([]), np.array([], ndmin=2, dtype=np.uint8).T
#                 elif line.startswith("positions"):
#                     tmp_pos = np.array([float(x) for x in line.strip().split()[1:]])
#                     tmp_pos = np.round(tmp_pos * seq_len).astype(int)

#                     # Find duplicates in the array
#                     duplicates = np.diff(tmp_pos) == 0

#                     # While there are any duplicates, increment them by 1
#                     for i in np.where(duplicates)[0]:
#                         tmp_pos[i + 1] += 1
#                     tmp_pos += 1
#                     positions.append(tmp_pos)
#                     tmp_map = np.column_stack(
#                         [
#                             np.repeat(1, tmp_pos.size),
#                             np.arange(tmp_pos.size),
#                             tmp_pos,
#                             tmp_pos,
#                         ]
#                     )
#                     rec_map.append(tmp_map)

#                 else:
#                     # Now read in the data
#                     data.append(np.array(list(line), dtype=np.int8))
#             try:
#                 data = np.vstack(data).T
#             except:
#                 data = None
#                 warn(f"File {ms_file} is malformed.")
#                 return None

#             # data = np.vstack(data).T
#             haps.append(data)

#     if param is None:
#         param = np.zeros(4)

#     return (haps[0], rec_map[0], param)


# def ms_parser_np(ms_file, param=None, seq_len=1.2e6):
#     """Read an ms/msms/hudson file and return (haps, rec_map, param) for the FIRST replicate.
#     haps: (num_segsites, num_samples) int8 array of 0/1
#     rec_map: (num_segsites, 4) int64 array: [1, idx, pos, pos]
#     """
#     if not ms_file.endswith((".out", ".out.gz", ".ms", ".ms.gz")):
#         warn(f"File {ms_file} has an unexpected extension.")

#     open_function = gzip.open if ms_file.endswith(".gz") else open

#     in_rep = False
#     num_segsites = None
#     pos_arr = None
#     hap_rows = []  # strings of '0'/'1', one per sample

#     def finish_and_return(p_in):
#         # Validate minimal structure
#         if num_segsites is None or pos_arr is None or not hap_rows:
#             warn(f"File {ms_file} is malformed.")
#             return None

#         # Build hap matrix fast: bytes -> uint8 -> 0/1 -> int8, then transpose
#         try:
#             n_samples = len(hap_rows)
#             n_sites = len(hap_rows[0])
#             H = np.empty((n_samples, n_sites), dtype=np.int8)
#             for i, s in enumerate(hap_rows):
#                 H[i, :] = (
#                     np.frombuffer(s.encode("ascii"), dtype=np.uint8) - 48
#                 ).astype(np.int8, copy=False)
#             H = H.T  # (num_segsites, num_samples)
#         except Exception:
#             warn(f"File {ms_file} is malformed.")
#             return None

#         n = pos_arr.size
#         rec_map = np.column_stack(
#             (np.repeat(1, n), np.arange(n, dtype=np.int64), pos_arr, pos_arr)
#         ).astype(np.int64, copy=False)

#         p_out = np.zeros(4) if p_in is None else p_in
#         return (H, rec_map, p_out)

#     with open_function(ms_file, "rt") as fh:
#         for raw in fh:
#             line = raw.strip()

#             # start of a replicate
#             if line.startswith("//"):
#                 if in_rep:
#                     out = finish_and_return(param)
#                     if out is not None:
#                         return out
#                     # else, malformed replicate; continue scanning for next
#                 in_rep = True
#                 num_segsites = None
#                 pos_arr = None
#                 hap_rows.clear()
#                 continue

#             if not in_rep or not line:
#                 continue

#             if line.startswith("segsites"):
#                 parts = line.split()
#                 if len(parts) >= 2:
#                     try:
#                         num_segsites = int(parts[1])
#                     except ValueError:
#                         warn(f"File {ms_file} is malformed.")
#                         return None
#                 else:
#                     warn(f"File {ms_file} is malformed.")
#                     return None
#                 continue

#             if line.startswith("positions"):
#                 # line like: "positions: 0.123 0.456 ..."
#                 try:
#                     _, values = line.split(":", 1)
#                     tmp_pos = np.fromstring(values, sep=" ", dtype=np.float64)
#                 except Exception:
#                     warn(f"File {ms_file} is malformed.")
#                     return None

#                 # EXACTLY match original: np.round then single-pass bump of consecutive duplicates, then +1
#                 tmp_pos = np.round(tmp_pos * seq_len).astype(np.int64, copy=False)

#                 dups = np.diff(tmp_pos) == 0
#                 if dups.any():
#                     idxs = np.nonzero(dups)[0]
#                     tmp_pos[idxs + 1] += 1  # bump NEXT duplicate once

#                 tmp_pos += 1
#                 pos_arr = tmp_pos
#                 continue

#             # haplotype row ('0'/'1' string)
#             c0 = line[0]
#             if c0 == "0" or c0 == "1":
#                 hap_rows.append(line)

#         # EOF: if a replicate was started, finalize it
#         if in_rep:
#             return finish_and_return(param)

#     warn(f"File {ms_file} is malformed.")
#     return None


# def cleaning_summaries(data, summ_stats, params, model):
#     # Cleaning params and simulations to remove malformed simulations
#     mask = []
#     summ_stats_filtered = []
#     malformed_files = []
#     for i, j in enumerate(summ_stats[0]):
#         if j is None:
#             mask.append(i)
#             malformed_files.append(
#                 f"Model {model}, index {i} is malformed."
#             )
#         else:
#             summ_stats_filtered.append(summ_stats[])

#     if len(mask) != 0:
#         params = np.delete(params, mask, axis=0)

#     return summ_stats_filtered, params, malformed_files


# def calculate_stats(
#     hap_data,
#     _iter=1,
#     center=[5e5, 7e5],
#     windows=[1000000],
#     step=1e4,
#     neutral=False,
#     region=None,
# ):
#     filterwarnings(
#         "ignore",
#         category=RuntimeWarning,
#         message="invalid value encountered in scalar divide",
#     )
#     np.seterr(divide="ignore", invalid="ignore")

#     if isinstance(hap_data, list) or isinstance(hap_data, tuple):
#         hap, rec_map, p = hap_data
#     elif isinstance(hap_data, str):
#         try:
#             hap, rec_map, p = ms_parser(hap_data)
#         except:
#             try:
#                 hap, rec_map, p = genome_reader(hap_data, region)
#             except:
#                 return None
#     else:
#         return None

#     # Open and filtering data
#     (
#         hap_int,
#         rec_map_01,
#         ac,
#         biallelic_mask,
#         position_masked,
#         genetic_position_masked,
#     ) = filter_gt(hap, rec_map, region=region)
#     freqs = ac[:, 1] / ac.sum(axis=1)

#     if len(center) == 1:
#         centers = np.arange(center[0], center[0] + step, step).astype(int)
#     else:
#         centers = np.arange(center[0], center[1] + step, step).astype(int)

#     df_dind_high_low, df_s_ratio, df_hapdaf_s, df_hapdaf_o = run_fs_stats(
#         hap_int[:, :], ac[:, :], rec_map_01[:, :]
#     )

#     d_stats = defaultdict(dict)

#     df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
#     df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
#     df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
#     df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

#     d_stats = defaultdict(dict)

#     df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
#     df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
#     df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
#     df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

#     d_stats["dind_high_low"] = df_dind_high_low
#     d_stats["s_ratio"] = df_s_ratio
#     d_stats["hapdaf_o"] = df_hapdaf_o
#     d_stats["hapdaf_s"] = df_hapdaf_s

#     try:
#         h12_v = h12_enard(
#             hap_int,
#             position_masked,
#             window_size=int(1e6),
#         )
#         # window_size=int(5e5) if neutral else int(1.2e6)
#     except:
#         h12_v = np.nan

#     haf_v = haf_top(
#         hap_int.astype(np.float64),
#         position_masked,
#         window_size=int(1e6),
#         # window_size=int(5e5) if neutral else int(1.2e6),
#     )

#     daf_w = 1.0
#     pos_w = int(6e5)
#     if 6e5 in position_masked:
#         daf_w = freqs[position_masked == 6e5][0]

#     df_window = pl.DataFrame(
#         {
#             "iter": pl.Series([_iter], dtype=pl.Int64),
#             "center": pl.Series([int(6e5)], dtype=pl.Int64),
#             "window": pl.Series([int(1e6)], dtype=pl.Int64),
#             "positions": pl.Series([pos_w], dtype=pl.Int64),
#             "daf": pl.Series([daf_w], dtype=pl.Float64),
#             "h12": pl.Series([h12_v], dtype=pl.Float64),
#             "haf": pl.Series([haf_v], dtype=pl.Float64),
#         }
#     )

#     d_stats["h12_haf"] = df_window

#     d_centers_stats = defaultdict(dict)
#     schema_center = {
#         "iter": pl.Int64,
#         "center": pl.Int64,
#         "window": pl.Int64,
#         "positions": pl.Int64,
#         "daf": pl.Float64,
#         "ihs": pl.Float64,
#         "delta_ihh": pl.Float64,
#         "isafe": pl.Float64,
#         "nsl": pl.Float64,
#     }

#     for c, w in product(centers, windows):
#         lower = c - w / 2
#         upper = c + w / 2

#         p_mask = (position_masked >= lower) & (position_masked <= upper)
#         p_mask
#         f_mask = freqs >= 0.05

#         # Check whether the hap subset is empty or not
#         if hap_int[p_mask].shape[0] == 0:
#             # df_centers_stats = pl.DataFrame({"iter": _iter,"center": c,"window": w,"positions": np.nan,"daf": np.nan,"isafe": np.nan,"ihs": np.nan,"nsl": np.nan,})
#             d_empty = pl.DataFrame(
#                 [
#                     [_iter],
#                     [c],
#                     [w],
#                     [None],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                 ],
#                 schema=schema_center,
#             )

#             d_centers_stats["ihs"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "ihs", "delta_ihh"]
#             )
#             d_centers_stats["isafe"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "isafe"]
#             )
#             d_centers_stats["nsl"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "nsl"]
#             )
#         else:
#             df_isafe = run_isafe(hap_int[p_mask], position_masked[p_mask])

#             # iHS and nSL
#             df_ihs = run_hapbin(
#                 hap_int[p_mask],
#                 rec_map_01[p_mask],
#                 min_ehh=0.05,
#                 gap_scale=0,
#                 max_extend=0,
#                 _iter=_iter,
#             )
#             # df_ihs = ihs_ihh(hap_int[p_mask],position_masked[p_mask],map_pos=genetic_position_masked[p_mask],min_ehh=0.05,min_maf=0.05,include_edges=False,)

#             nsl_v = nsl(hap_int[(p_mask) & (f_mask)], use_threads=False)
#             df_nsl = pl.DataFrame(
#                 {
#                     "positions": position_masked[(p_mask) & (f_mask)],
#                     "daf": freqs[(p_mask) & (f_mask)],
#                     "nsl": nsl_v,
#                 }
#             ).fill_nan(None)

#             df_isafe = center_window_cols(df_isafe, _iter=_iter, center=c, window=w)
#             df_ihs = center_window_cols(df_ihs, _iter=_iter, center=c, window=w)
#             df_nsl = center_window_cols(df_nsl, _iter=_iter, center=c, window=w)

#             d_centers_stats["ihs"][c] = df_ihs
#             d_centers_stats["isafe"][c] = df_isafe
#             d_centers_stats["nsl"][c] = df_nsl

#     d_stats["ihs"] = pl.concat(d_centers_stats["ihs"].values())
#     d_stats["isafe"] = pl.concat(d_centers_stats["isafe"].values())
#     d_stats["nsl"] = pl.concat(d_centers_stats["nsl"].values())

#     if region is not None:
#         for k, df in d_stats.items():
#             d_stats[k] = df.with_columns(pl.lit(region).alias("iter"))

#     ####### if neutral:
#     # Whole chromosome statistic to normalize

#     df_ihs = run_hapbin(
#         hap_int, rec_map_01, min_ehh=0.1, gap_scale=0, max_extend=0, _iter=_iter
#     )

#     nsl_v = nsl(hap_int[freqs >= 0.05], use_threads=False)

#     df_nsl = pl.DataFrame(
#         {
#             "positions": position_masked[freqs >= 0.05],
#             "daf": freqs[freqs >= 0.05],
#             "nsl": nsl_v,
#         }
#     ).fill_nan(None)

#     df_snps_norm = reduce(
#         lambda left, right: left.join(
#             right, on=["positions", "daf"], how="full", coalesce=True
#         ),
#         [df_nsl, df_ihs, df_isafe],
#     )

#     df_stats = reduce(
#         lambda left, right: left.join(
#             right,
#             on=["iter", "center", "window", "positions", "daf"],
#             how="full",
#             coalesce=True,
#         ),
#         [df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s, df_window],
#     ).sort(["iter", "center", "window", "positions"])
#     df_stats_norm = df_snps_norm.join(
#         df_stats.select(
#             pl.all().exclude(
#                 ["iter", "center", "window", "delta_ihh", "ihs", "isafe", "nsl"]
#             )
#         ),
#         on=["positions", "daf"],
#         how="full",
#         coalesce=True,
#     ).sort(["positions"])

#     df_stats_norm = (
#         df_stats_norm.with_columns(
#             [
#                 pl.lit(_iter).alias("iter"),
#                 pl.lit(600000).alias("center"),
#                 pl.lit(1200000).alias("window"),
#                 pl.col("positions").cast(pl.Int64),
#             ]
#         )
#         .select(
#             pl.col(["iter", "center", "window", "positions"]),
#             pl.all().exclude(["iter", "center", "window", "positions"]),
#         )
#         .sort(["iter", "center", "window", "positions"])
#     )

#     if region is not None:
#         df_stats_norm = df_stats_norm.with_columns(pl.lit(region).alias("iter"))

#     return d_stats, df_stats_norm


# Normalization 270525
# old
# def normalization_raw(
#     stats_values,
#     bins,
#     center=[5e5, 7e5],
#     windows=[50000, 100000, 200000, 500000, 1000000],
#     vcf=False,
#     nthreads=1,
#     parallel_manager=None,
# ):
#     """
#     Normalizes sweep statistics using neutral expectations or optionally using precomputed neutral normalized values.
#     The function applies normalization across different genomic windows and supports multi-threading.

#     Parameters
#     ----------
#     sweeps_stats : namedtuple
#         A Namedtuple containing the statistics for genomic sweeps and sweep parameters across
#         different genomic windows.

#     neutral_stats_norm : namedtuple
#         A Namedtuple containing the statistics for neutral region and neutral parameters across
#         different genomic windows, used as the baselinefor normalizing the sweep statistics.
#         This allows comparison of sweeps against neutral expectations.

#     norm_values : dict or None, optional (default=None)
#         A dictionary of precomputed neutral normalizated values. If provided, these values are
#         used to directly normalize the statistics. If None, the function computes
#         normalization values from the neutral statistics.

#     center : list of float, optional (default=[5e5, 7e5])
#         A list specifying the center positions (in base pairs) for the analysis windows.
#         If a single center value is provided, normalization is centered around that value.
#         Otherwise, it will calculate normalization for a range of positions between the two provided centers.

#     windows : list of int, optional (default=[50000, 100000, 200000, 500000, 1000000])
#         A list of window sizes (in base pairs) for which the normalization will be applied.
#         The function performs normalization for each of the specified window sizes.

#     nthreads : int, optional (default=1)
#         The number of threads to use for parallel processing. If set to 1, the function
#         runs in single-threaded mode. Higher values enable multi-threaded execution for
#         faster computation.

#     Returns
#     -------
#     normalized_stats : pandas.DataFrame
#         A DataFrame containing the normalized sweep statistics across the specified
#         windows and genomic regions. The sweep statistics are scaled relative to
#         neutral expectations.
#     """
#     df_stats, params = stats_values

#     if vcf:
#         df_h12_haf = df_stats.pop("h12_haf")
#         snps_genome = reduce(
#             lambda left, right: left.join(
#                 right,
#                 on=["iter", "center", "window", "positions", "daf"],
#                 how="full",
#                 coalesce=True,
#             ),
#             list(df_stats.values()),
#         ).sort(["iter", "center", "window", "positions"])
#         nchr = snps_genome["iter"].unique().first()

#         # Overwrite snps_values
#         # snps_genome = snps_genome.select(pl.exclude(["h12", "haf"]))
#         snps_values = snps_genome.select(pl.exclude(["h12", "haf"]))

#         stats_names = snps_values.select(snps_values.columns[5:]).columns

#         binned_values = bin_values(snps_values)

#         normalized_df = normalize_snps_statistics(
#             binned_values, bins, stats_names
#         ).sort("positions")

#         df_fv_n = vectorized_cut_vcf(
#             normalized_df, center, windows, stats_names
#         ).fill_nan(None)
#         df_fv_n_raw = vectorized_cut_vcf(
#             snps_genome, center, windows, stats_names
#         ).fill_nan(None)

#         window_values = (
#             df_h12_haf.with_columns(pl.col("positions").alias("iter"))
#             .filter(pl.col("positions").is_in(df_fv_n["iter"].unique().to_numpy()))
#             .select(pl.exclude(["center", "window", "positions", "daf"]))
#         )
#         df_fv_n = df_fv_n.join(
#             window_values,
#             on=["iter"],
#             how="full",
#             coalesce=True,
#         )
#         df_fv_n_raw = df_fv_n_raw.join(
#             window_values,
#             on=["iter"],
#             how="full",
#             coalesce=True,
#         )

#         df_params_unpack = pl.DataFrame(
#             np.repeat(
#                 np.zeros((1, 4)),
#                 df_fv_n.shape[0],
#                 axis=0,
#             ),
#             schema=["s", "t", "f_i", "f_t"],
#         )
#         df_fv_n = df_fv_n.with_columns(
#             pl.lit(snps_genome["iter"].unique().first()).alias("nchr")
#         )
#         df_fv_n_raw = df_fv_n_raw.with_columns(
#             pl.lit(snps_genome["iter"].unique().first()).alias("nchr")
#         )
#     else:
#         # df_stats_unzipped = dict(zip(
#         #     df_stats[0].keys(),
#         #     zip(*(d.values() for d in df_stats))
#         # ))

#         # df_stats_unzipped = {i:pl.concat(v) for i,v in df_stats_unzipped.items()}

#         # df_fv_n_l = []
#         # df_fv_n_raw_l = []
#         # for k,snps_genome in tqdm(df_stats_unzipped.items()):
#         #     if k == 'h12_haf':
#         #         continue
#         #     snps_values = snps_genome.select(pl.exclude(["h12", "haf"]))
#         #     binned_values = bin_values(snps_values)
#         #     stats_names = snps_genome.select(snps_genome.columns[5:]).columns

#         #     normalized_df = normalize_snps_statistics(binned_values, bins, stats_names)
#         #     normalized_df = normalized_df.with_columns(pl.lit(snps_genome['iter']))

#         #     df_fv_n.append(vectorized_cut(normalized_df, center, windows, stats_names).fill_nan(None))
#         #     df_fv_n_raw_l.append(vectorized_cut(snps_genome, center, windows, stats_names).fill_nan(None))

#         # df_fv_n = reduce(
#         #     lambda left, right: left.join(
#         #         right,
#         #         on=["iter", "center", "window"],
#         #         how="full",
#         #         coalesce=True,
#         #     ),
#         #     df_fv_n_l,
#         # ).sort(["iter", "center", "window"])

#         # df_fv_n_raw = reduce(
#         #     lambda left, right: left.join(
#         #         right,
#         #         on=["iter", "center", "window"],
#         #         how="full",
#         #         coalesce=True,
#         #     ),
#         #     df_fv_n_raw_l,
#         # ).sort(["iter", "center", "window"])

#         if parallel_manager is None:
#             df_fv_n_l, df_fv_n_l_raw = zip(
#                 *Parallel(n_jobs=nthreads, verbose=1)(
#                     delayed(normalize_cut_raw)(
#                         snps_values, bins, center=center, windows=windows
#                     )
#                     for _iter, snps_values in enumerate(df_stats, 1)
#                 )
#             )
#         else:
#             df_fv_n_l, df_fv_n_l_raw = zip(
#                 *parallel_manager(
#                     delayed(normalize_cut_raw)(
#                         snps_values, bins, center=center, windows=windows
#                     )
#                     for _iter, snps_values in enumerate(df_stats, 1)
#                 )
#             )

#         df_fv_n = pl.concat(df_fv_n_l).with_columns(
#             pl.col(["iter", "window", "center"]).cast(pl.Int64)
#         )
#         df_fv_n_raw = pl.concat(df_fv_n_l_raw).with_columns(
#             pl.col(["iter", "window", "center"]).cast(pl.Int64)
#         )

#         df_window = pl.concat([df["h12_haf"] for df in df_stats]).select(
#             pl.exclude(["center", "window", "positions", "daf"])
#         )
#         df_fv_n = df_fv_n.join(df_window, on=["iter"], how="full", coalesce=True)
#         df_fv_n_raw = df_fv_n_raw.join(
#             df_window, on=["iter"], how="full", coalesce=True
#         )

#         # params = params[:, [0, 1, 3, 4, ]]
#         df_params_unpack = pl.DataFrame(
#             np.repeat(
#                 params,
#                 df_fv_n.select(["center", "window"])
#                 .unique()
#                 .sort(["center", "window"])
#                 .shape[0],
#                 axis=0,
#             ),
#             schema=["s", "t", "f_i", "f_t"],
#         )

#     df_fv_n = pl.concat(
#         [df_params_unpack, df_fv_n],
#         how="horizontal",
#     )
#     df_fv_n_raw = pl.concat(
#         [df_params_unpack, df_fv_n_raw],
#         how="horizontal",
#     )

#     force_order = ["iter"] + [col for col in df_fv_n.columns if col != "iter"]
#     df_fv_n = df_fv_n.select(force_order)
#     df_fv_n_raw = df_fv_n_raw.select(force_order)

#     return df_fv_n, df_fv_n_raw


# # old
# def normalize_cut_raw(
#     snps_values,
#     bins,
#     center=[5e5, 7e5],
#     windows=[50000, 100000, 200000, 500000, 1000000],
#     step=int(1e4),
# ):
#     """
#     Normalizes SNP-level statistics by comparing them to neutral expectations, and aggregates
#     the statistics within sliding windows around specified genomic centers.

#     This function takes SNP statistics, normalizes them based on the expected mean and standard
#     deviation from neutral simulations, and computes the average values within windows
#     centered on specific genomic positions. It returns a DataFrame with the normalized values
#     for each window across the genome.

#     Parameters
#     ----------
#     _iter : int
#         The iteration or replicate number associated with the current set of SNP statistics.

#     snps_values : pandas.DataFrame
#         A DataFrame containing SNP-level statistics such as iHS, nSL, and iSAFE. The DataFrame
#         should contain derived allele frequencies ("daf") and other statistics to be normalized.

#     expected : pandas.DataFrame
#         A DataFrame containing the expected mean values of the SNP statistics for each frequency bin,
#         computed from neutral simulations.

#     stdev : pandas.DataFrame
#         A DataFrame containing the standard deviation of the SNP statistics for each frequency bin,
#         computed from neutral simulations.

#     center : list of float, optional (default=[5e5, 7e5])
#         A list specifying the center positions (in base pairs) for the analysis. Normalization is
#         performed around these genomic centers using the specified window sizes.

#     windows : list of int, optional (default=[50000, 100000, 200000, 500000, 1000000])
#         A list of window sizes (in base pairs) over which the SNP statistics will be aggregated
#         and normalized. The function performs normalization for each specified window size.

#     Returns
#     -------
#     out : pandas.DataFrame
#         A DataFrame containing the normalized SNP statistics for each genomic center and window.
#         The columns include the iteration number, center, window size, and the average values
#         of normalized statistics iSAFE within the window.

#     Notes
#     -----
#     - The function first bins the SNP statistics based on derived allele frequencies using the
#       `bin_values` function. The statistics are then normalized by subtracting the expected mean
#       and dividing by the standard deviation for each frequency bin.
#     - After normalization, SNPs are aggregated into windows centered on specified genomic positions.
#       The average values of the normalized statistics are calculated for each window.
#     - The window size determines how far upstream and downstream of the center position the SNPs
#       will be aggregated.

#     """
#     df_out = []
#     df_out_raw = []
#     _iter = np.unique([i.select("iter").unique() for i in snps_values.values()])

#     if len(center) == 2:
#         centers = np.arange(center[0], center[1] + step, step).astype(int)
#     else:
#         centers = center

#     for k, df in snps_values.items():
#         if k == "h12_haf":
#             continue

#         stats_names = df.select(df.columns[5:]).columns

#         binned_values = bin_values(df)
#         normalized_df = normalize_snps_statistics(binned_values, bins, stats_names)

#         fixed_center = int(6e5) if k not in ["isafe", "ihs", "nsl"] else None

#         df_out.append(
#             cut_snps(
#                 normalized_df,
#                 centers,
#                 windows,
#                 stats_names,
#                 fixed_center=None,
#                 iter_value=_iter,
#             )
#         )
#         df_out_raw.append(
#             cut_snps(
#                 df,
#                 centers,
#                 windows,
#                 stats_names,
#                 fixed_center=None,
#                 iter_value=_iter,
#             )
#         )

#     df_out = reduce(
#         lambda left, right: left.join(
#             right,
#             on=["iter", "center", "window"],
#             how="full",
#             coalesce=True,
#         ),
#         df_out,
#     ).sort(["iter", "center", "window"])

#     df_out_raw = reduce(
#         lambda left, right: left.join(
#             right,
#             on=["iter", "center", "window"],
#             how="full",
#             coalesce=True,
#         ),
#         df_out_raw,
#     ).sort(["iter", "center", "window"])

#     return df_out, df_out_raw


# # old
# def cut_snps(
#     normalized_df, centers, windows, stats_names, fixed_center=None, iter_value=1
# ):
#     """
#     Processes data within windows across multiple centers and window sizes.

#     Parameters
#     ----------
#     normalized_df : polars.DataFrame
#         DataFrame containing the positions and statistics.
#     iter_value : int
#         Iteration or replicate number.
#     centers : list
#         List of center positions to analyze.
#     windows : list
#         List of window sizes to use.
#     stats_names : list, optional
#         Names of statistical columns to compute means for.
#         If None, all columns except position-related ones will be used.
#     position_col : str, optional
#         Name of the column containing position values.
#     center_col : str, optional
#         Name of the column containing center values.
#     fixed_center : int, optional
#         If provided, use this fixed center value instead of the ones in centers list.

#     Returns
#     -------
#     polars.DataFrame
#         DataFrame with aggregated statistics for each center and window.
#     """
#     # If stats_names not provided, use all appropriate columns

#     results = []
#     for c, w in list(product(centers, windows)):
#         # Use fixed center if provided
#         c_fix = fixed_center if fixed_center is not None else c

#         # Define window boundaries
#         lower = c - w // 2
#         upper = c + w // 2

#         # Filter data by center and window boundaries
#         query = normalized_df.lazy()

#         # 1.2MB simulations derives into 21 center/windows combinations
#         if centers.size > 21:
#             query = query.filter(
#                 (pl.col("positions") >= c - int(6e5))
#                 & (pl.col("positions") <= c + int(6e5))
#             )
#         else:
#             query = query.filter(pl.col("center") == c_fix)

#         window_data = query.filter(
#             (pl.col("positions") >= lower) & (pl.col("positions") <= upper)
#         ).collect()

#         window_data = normalized_df.filter(
#             (pl.col("positions") >= lower) & (pl.col("positions") <= upper)
#         )

#         # Skip if no data in this window
#         # if window_data.height == 0:
#         #     continue

#         # Calculate mean statistics for window
#         window_stats = window_data.select(stats_names).fill_nan(None).mean()

#         # Add metadata columns
#         metadata_cols = [
#             pl.lit(iter_value).alias("iter"),
#             pl.lit(c).alias("center"),
#             pl.lit(w).alias("window"),
#         ]

#         results.append(window_stats.with_columns(metadata_cols))

#     return (
#         pl.concat(results, how="vertical").select(
#             ["iter", "center", "window"] + stats_names
#         )
#         if results
#         else None
#     )


#########################
# def calculate_stats(
#     hap_data,
#     _iter=1,
#     center=[5e5, 7e5],
#     windows=[1000000],
#     step=1e4,
#     neutral=False,
#     region=None,
# ):
#     filterwarnings(
#         "ignore",
#         category=RuntimeWarning,
#         message="invalid value encountered in scalar divide",
#     )
#     np.seterr(divide="ignore", invalid="ignore")

#     if isinstance(hap_data, list) or isinstance(hap_data, tuple):
#         hap, rec_map, p = hap_data
#     elif isinstance(hap_data, str):
#         try:
#             hap, rec_map, p = ms_parser(hap_data)
#         except:
#             try:
#                 hap, rec_map, p = genome_reader(hap_data, region)
#             except:
#                 return None
#     else:
#         return None

#     # Open and filtering data
#     (
#         hap_int,
#         rec_map_01,
#         ac,
#         biallelic_mask,
#         position_masked,
#         genetic_position_masked,
#     ) = filter_gt(hap, rec_map, region=region)
#     freqs = ac[:, 1] / ac.sum(axis=1)

#     if len(center) == 1:
#         centers = np.arange(center[0], center[0] + step, step).astype(int)
#     else:
#         centers = np.arange(center[0], center[1] + step, step).astype(int)

#     # df_dind_high_low = dind_high_low(hap_int, ac, rec_map_01)
#     # df_s_ratio = s_ratio(hap_int, ac, rec_map_01)
#     # df_hapdaf_o = hapdaf_o(hap_int, ac, rec_map_01)
#     # df_hapdaf_s = hapdaf_s(hap_int, ac, rec_map_01)
#     df_dind_high_low, df_s_ratio, df_hapdaf_s, df_hapdaf_o = run_fs_stats(
#         hap_int[:, :], ac[:, :], rec_map_01[:, :]
#     )

#     d_stats = defaultdict(dict)

#     df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
#     df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
#     df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
#     df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

#     d_stats = defaultdict(dict)

#     df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
#     df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
#     df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
#     df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

#     d_stats["dind_high_low"] = df_dind_high_low
#     d_stats["s_ratio"] = df_s_ratio
#     d_stats["hapdaf_o"] = df_hapdaf_o
#     d_stats["hapdaf_s"] = df_hapdaf_s

#     try:
#         h12_v = h12_enard(
#             hap_int, position_masked, window_size=int(5e5) if neutral else int(1.2e6)
#         )
#     except:
#         h12_v = np.nan

#     haf_v = haf_top(
#         hap_int.astype(np.float64),
#         position_masked,
#         window_size=int(5e5) if neutral else int(1.2e6),
#     )

#     daf_w = 1.0
#     pos_w = int(6e5)
#     if 6e5 in position_masked:
#         daf_w = freqs[position_masked == 6e5][0]

#     df_window = pl.DataFrame(
#         {
#             "iter": pl.Series([_iter], dtype=pl.Int64),
#             "center": pl.Series([int(6e5)], dtype=pl.Int64),
#             "window": pl.Series([int(1e6)], dtype=pl.Int64),
#             "positions": pl.Series([pos_w], dtype=pl.Int64),
#             "daf": pl.Series([daf_w], dtype=pl.Float64),
#             "h12": pl.Series([h12_v], dtype=pl.Float64),
#             "haf": pl.Series([haf_v], dtype=pl.Float64),
#         }
#     )

#     d_stats["h12_haf"] = df_window

#     d_centers_stats = defaultdict(dict)
#     schema_center = {
#         "iter": pl.Int64,
#         "center": pl.Int64,
#         "window": pl.Int64,
#         "positions": pl.Int64,
#         "daf": pl.Float64,
#         "ihs": pl.Float64,
#         "delta_ihh": pl.Float64,
#         "isafe": pl.Float64,
#         "nsl": pl.Float64,
#     }

#     for c, w in product(centers, windows):
#         lower = c - w / 2
#         upper = c + w / 2

#         p_mask = (position_masked >= lower) & (position_masked <= upper)
#         p_mask
#         f_mask = freqs >= 0.05

#         # Check whether the hap subset is empty or not
#         if hap_int[p_mask].shape[0] == 0:
#             # df_centers_stats = pl.DataFrame({"iter": _iter,"center": c,"window": w,"positions": np.nan,"daf": np.nan,"isafe": np.nan,"ihs": np.nan,"nsl": np.nan,})
#             d_empty = pl.DataFrame(
#                 [
#                     [_iter],
#                     [c],
#                     [w],
#                     [None],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                     [np.nan],
#                 ],
#                 schema=schema_center,
#             )

#             d_centers_stats["ihs"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "ihs", "delta_ihh"]
#             )
#             d_centers_stats["isafe"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "isafe"]
#             )
#             d_centers_stats["nsl"][c] = d_empty.select(
#                 ["iter", "center", "window", "positions", "daf", "nsl"]
#             )
#         else:
#             df_isafe = run_isafe(hap_int[p_mask], position_masked[p_mask])

#             # iHS and nSL
#             df_ihs = run_hapbin(
#                 hap_int[p_mask], rec_map_01[p_mask], gap_scale=0, max_extend=0
#             )
#             # df_ihs = ihs_ihh(hap_int[p_mask],position_masked[p_mask],map_pos=genetic_position_masked[p_mask],min_ehh=0.05,min_maf=0.05,include_edges=False,)

#             nsl_v = nsl(hap_int[(p_mask) & (f_mask)], use_threads=False)
#             df_nsl = pl.DataFrame(
#                 {
#                     "positions": position_masked[(p_mask) & (f_mask)],
#                     "daf": freqs[(p_mask) & (f_mask)],
#                     "nsl": nsl_v,
#                 }
#             ).fill_nan(None)

#             df_isafe = center_window_cols(df_isafe, _iter=_iter, center=c, window=w)
#             df_ihs = center_window_cols(df_ihs, _iter=_iter, center=c, window=w)
#             df_nsl = center_window_cols(df_nsl, _iter=_iter, center=c, window=w)

#             d_centers_stats["ihs"][c] = df_ihs
#             d_centers_stats["isafe"][c] = df_isafe
#             d_centers_stats["nsl"][c] = df_nsl

#     d_stats["ihs"] = pl.concat(d_centers_stats["ihs"].values())
#     d_stats["isafe"] = pl.concat(d_centers_stats["isafe"].values())
#     d_stats["nsl"] = pl.concat(d_centers_stats["nsl"].values())

#     if region is not None:
#         for k, df in d_stats.items():
#             d_stats[k] = df.with_columns(pl.lit(region).alias("iter"))

#     ####### if neutral:
#     # Whole chromosome statistic to normalize
#     df_isafe = run_isafe(hap_int, position_masked)
#     df_ihs = ihs_ihh(hap_int, position_masked, min_ehh=0.1, include_edges=True)

#     nsl_v = nsl(hap_int[freqs >= 0.05], use_threads=False)

#     df_nsl = pl.DataFrame(
#         {
#             "positions": position_masked[freqs >= 0.05],
#             "daf": freqs[freqs >= 0.05],
#             "nsl": nsl_v,
#         }
#     ).fill_nan(None)

#     df_snps_norm = reduce(
#         lambda left, right: left.join(
#             right, on=["positions", "daf"], how="full", coalesce=True
#         ),
#         [df_nsl, df_ihs, df_isafe],
#     )

#     df_stats = reduce(
#         lambda left, right: left.join(
#             right,
#             on=["iter", "center", "window", "positions", "daf"],
#             how="full",
#             coalesce=True,
#         ),
#         [df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s, df_window],
#     ).sort(["iter", "center", "window", "positions"])
#     df_stats_norm = df_snps_norm.join(
#         df_stats.select(
#             pl.all().exclude(
#                 ["iter", "center", "window", "delta_ihh", "ihs", "isafe", "nsl"]
#             )
#         ),
#         on=["positions", "daf"],
#         how="full",
#         coalesce=True,
#     ).sort(["positions"])

#     df_stats_norm = (
#         df_stats_norm.with_columns(
#             [
#                 pl.lit(_iter).alias("iter"),
#                 pl.lit(600000).alias("center"),
#                 pl.lit(1200000).alias("window"),
#                 pl.col("positions").cast(pl.Int64),
#             ]
#         )
#         .select(
#             pl.col(["iter", "center", "window", "positions"]),
#             pl.all().exclude(["iter", "center", "window", "positions"]),
#         )
#         .sort(["iter", "center", "window", "positions"])
#     )

#     if region is not None:
#         df_stats_norm = df_stats_norm.with_columns(pl.lit(region).alias("iter"))

#     return d_stats, df_stats_norm


# def run_haplosweep(
#     hap,
#     rec_map,
#     _iter=1,
#     min_ehh=0.05,
#     min_maf=0.05,
#     gap_scale=20000,
#     max_extend=1000000,
#     haplosweep="/home/jmurgamoreno/software/HaploSweep/bin/HaploSweep calc",
# ):
#     df_hap = pl.DataFrame(hap).transpose()
#     df_rec_map = pl.DataFrame(
#         rec_map,
#         pl.Schema(
#             [
#                 ("chr", pl.Int64),
#                 ("location", pl.Int64),
#                 ("cm_mb", pl.Float64),
#                 ("cm", pl.Float64),
#             ]
#         ),
#     )
#     hap_file = "/tmp/tmp_" + str(_iter) + ".hap"
#     map_file = "/tmp/tmp_" + str(_iter) + ".map"

#     df_hap.write_csv(hap_file, separator=" ", include_header=False)

#     df_rec_map.write_csv(map_file, include_header=False, separator=" ")

#     out_file = f"/tmp/tmp_{_iter}.out"
#     cmd_hapbin = f"{haplosweep} -i {hap_file} -m {map_file} -t 1 -cutoff {min_ehh} -maf {min_maf} -o1 {out_file}"

#     if gap_scale > 0:
#         cmd_hapbin += f" -gap-scale {gap_scale}"

#     if max_extend > 0:
#         cmd_hapbin += f" -e {max_extend}"

#     ihs_output = subprocess.run(cmd_hapbin, shell=True, capture_output=True, text=True)

#     df_ihs = pl.read_csv(
#         out_file,
#         separator="\t",
#         schema=pl.Schema(
#             [
#                 ("locusid", pl.Int64),
#                 ("chr", pl.Int64),
#                 ("positions", pl.Int64),
#                 ("daf", pl.Float64),
#                 ("ihh1", pl.Float64),
#                 ("ihh0", pl.Float64),
#                 ("ihhl1", pl.Float64),
#                 ("ihhl0", pl.Float64),
#                 ("ihs", pl.Float64),
#                 ("urihs1", pl.Float64),
#                 ("urihs0", pl.Float64),
#                 ("ihsl", pl.Float64),
#             ]
#         ),
#     ).select("positions", "daf", "ihs", "urihs1", "urihs0", "ihsl")

#     os.remove(hap_file)
#     os.remove(map_file)
#     os.remove(out_file)

#     # normalize and estimate rihsl
#     # rihsl = ihsl**2 + (urihs1 if ihsl > 0 else urihs0)**2

#     return df_ihs


# def mispolarize(hap, proportion=0.1):
#     """
#     Allele mispolarization by randomly flipping the alleles of a haplotype matrix (i.e., switching between 0 and 1). The proportion of rows to be flipped is determined by the `proportion` parameter.

#     Parameters
#     ----------
#     hap : numpy.ndarray
#         A 2D numpy array representing the haplotype matrix of shape (S, n),
#         where S is the number of variants (rows), and n is the number of samples (columns).
#         Each element is expected to be binary (0 or 1), representing the alleles.

#     proportion : float, optional (default=0.1)
#         A float between 0 and 1 specifying the proportion of rows (loci) in the haplotype
#         matrix to randomly flip. For example, if proportion=0.1, 10% of the rows in the
#         haplotype matrix will have their allele values flipped.

#     Returns
#     -------
#     hap_copy : numpy.ndarray
#         A new 2D numpy array of the same shape as `hap`, with a proportion of rows
#         randomly flipped (alleles inverted). The original matrix `hap` is not modified
#         in-place.

#     Notes
#     -----
#     The flipping operation is done using a bitwise XOR operation (`^= 1`), which
#     efficiently flips 0 to 1 and 1 to 0 for the selected rows.

#     """
#     # Get shape of haplotype matrix
#     S, n = hap.shape

#     # Select the column indices to flip based on the given proportion
#     to_flip = np.random.choice(np.arange(S), int(S * proportion), replace=False)

#     # Create a copy of the original hap matrix to avoid in-place modification
#     hap_copy = hap.copy()
#     hap_copy[to_flip, :] ^= 1
#     return hap_copy


# def filter_gt(hap, rec_map, region=None):
#     """
#     Convert the 2d numpy haplotype matrix to HaplotypeArray from scikit-allel and change type np.int8. It filters and processes the haplotype matrix based on a recombination map and
#     returns key information for further analysis such as allele frequencies and physical positions.

#     Parameters
#     ----------
#     hap : array-like, HaplotypeArray
#         The input haplotype data which can be in one of the following forms:
#         - A `HaplotypeArray` object.
#         - A genotype matrix (as a numpy array or similar).

#     rec_map : numpy.ndarray
#         A 2D numpy array representing the recombination map, where each row corresponds
#         to a genomic variant and contains recombination information. The third column (index 2)
#         of the recombination map provides the physical positions of the variants.

#     Returns
#     -------
#     tuple
#         A tuple containing the following elements:
#         - hap_01 (numpy.ndarray): The filtered haplotype matrix, with only biallelic variants.
#         - ac (AlleleCounts): An object that stores allele counts for the filtered variants.
#         - biallelic_mask (numpy.ndarray): A boolean mask indicating which variants are biallelic.
#         - hap_int (numpy.ndarray): The haplotype matrix converted to integer format (int8).
#         - rec_map_01 (numpy.ndarray): The recombination map filtered to include only biallelic variants.
#         - position_masked (numpy.ndarray): The physical positions of the biallelic variants.
#         - sequence_length (int): An arbitrary sequence length set to 1.2 million bases (1.2e6).
#         - freqs (numpy.ndarray): The frequencies of the alternate alleles for each biallelic variant.
#     """
#     try:
#         hap = HaplotypeArray(hap.genotype_matrix())
#     except:
#         try:
#             hap = HaplotypeArray(hap)
#         except:
#             hap = HaplotypeArray(load(hap).genotype_matrix())

#     # positions = rec_map[:, -1]
#     # physical_position = rec_map[:, -2]

#     # HAP matrix centered to analyse whole chromosome
#     hap_01, ac, biallelic_mask = filter_biallelics(hap)
#     hap_int = hap_01.astype(np.int8)
#     rec_map_01 = rec_map[biallelic_mask]
#     sequence_length = int(1.2e6)
#     freqs = ac.to_frequencies()[:, 1]

#     if region is not None:
#         tmp = list(map(int, region.split(":")[-1].split("-")))
#         d_pos = dict(
#             zip(np.arange(tmp[0], tmp[1] + 1), np.arange(1, sequence_length + 1))
#         )
#         for r in rec_map_01:
#             r[-1] = d_pos[r[-1]]

#     position_masked = rec_map_01[:, -1]
#     genetic_position_masked = rec_map_01[:, -2]

#     return (
#         hap_01,
#         ac,
#         biallelic_mask,
#         hap_int,
#         rec_map_01,
#         position_masked,
#         genetic_position_masked,
#         sequence_length,
#         freqs,
#     )


# def filter_biallelics(hap: HaplotypeArray) -> tuple:
#     """
#     Filter out non-biallelic loci from the haplotype data.

#     Args: hap (allel.HaplotypeArray): Haplotype data represented as a HaplotypeArray.

#     Returns:tuple: A tuple containing three elements:
#         - hap_biallelic (allel.HaplotypeArray): Filtered biallelic haplotype data.
#         - ac_biallelic (numpy.ndarray): Allele counts for the biallelic loci.
#         - biallelic_mask (numpy.ndarray): Boolean mask indicating biallelic loci.
#     """
#     ac = hap.count_alleles()
#     biallelic_mask = ac.is_biallelic_01()
#     return (hap.subset(biallelic_mask), ac[biallelic_mask, :], biallelic_mask)


# def normalize_cut_vcf(
#     snps_values,
#     bins,
#     center,
#     windows=[50000, 100000, 200000, 500000, 1000000],
#     nthreads=1,
#     parallel_manager=None,
# ):
#     """
#     Normalizes SNP-level statistics by comparing them to neutral expectations, and aggregates
#     the statistics within sliding windows around specified genomic centers.

#     This function takes SNP statistics, normalizes them based on the expected mean and standard
#     deviation from neutral simulations, and computes the average values within windows
#     centered on specific genomic positions. It returns a DataFrame with the normalized values
#     for each window across the genome.

#     Parameters
#     ----------
#     _iter : int
#         The iteration or replicate number associated with the current set of SNP statistics.

#     snps_values : pandas.DataFrame
#         A DataFrame containing SNP-level statistics such as iHS, nSL, and iSAFE. The DataFrame
#         should contain derived allele frequencies ("daf") and other statistics to be normalized.

#     expected : pandas.DataFrame
#         A DataFrame containing the expected mean values of the SNP statistics for each frequency bin,
#         computed from neutral simulations.

#     stdev : pandas.DataFrame
#         A DataFrame containing the standard deviation of the SNP statistics for each frequency bin,
#         computed from neutral simulations.

#     center : list of float, optional (default=[5e5, 7e5])
#         A list specifying the center positions (in base pairs) for the analysis. Normalization is
#         performed around these genomic centers using the specified window sizes.

#     windows : list of int, optional (default=[50000, 100000, 200000, 500000, 1000000])
#         A list of window sizes (in base pairs) over which the SNP statistics will be aggregated
#         and normalized. The function performs normalization for each specified window size.

#     Returns
#     -------
#     out : pandas.DataFrame
#         A DataFrame containing the normalized SNP statistics for each genomic center and window.
#         The columns include the iteration number, center, window size, and the average values
#         of normalized statistics iSAFE within the window.

#     Notes
#     -----
#     - The function first bins the SNP statistics based on derived allele frequencies using the
#       `bin_values` function. The statistics are then normalized by subtracting the expected mean
#       and dividing by the standard deviation for each frequency bin.
#     - After normalization, SNPs are aggregated into windows centered on specified genomic positions.
#       The average values of the normalized statistics are calculated for each window.
#     - The window size determines how far upstream and downstream of the center position the SNPs
#       will be aggregated.

#     """

#     # Overwrite snps_values
#     snps_values = snps_values.select(pl.exclude(["h12", "haf"]))

#     stats_names = snps_values.select(snps_values.columns[5:]).columns

#     binned_values = bin_values(snps_values)

#     normalized_df = normalize_snps_statistics(binned_values, bins, stats_names).sort(
#         "positions"
#     )

#     df_out = cut_vcf(normalized_df, center, windows, stats_names)
#     df_out_raw = cut_vcf(snps_values, center, windows, stats_names)

#     df_out = df_out.with_columns(
#         (
#             pl.lit(snps_values["iter"].unique())
#             + ":"
#             + (pl.col("iter") - int(6e5)).cast(pl.String)
#             + "-"
#             + (pl.col("iter") + int(6e5)).cast(pl.String)
#         ).alias("iter")
#     )
#     df_out_raw = df_out_raw.with_columns(df_out["iter"].alias("iter"))
#     return df_out, df_out_raw


# def slice_window(c, df):
#     positions = df["positions"].to_numpy()

#     lower = c - int(6e5)
#     upper = c + int(6e5)

#     start_idx = np.searchsorted(positions, lower, side="left")
#     end_idx = np.searchsorted(positions, upper, side="right")

#     return df.slice(start_idx, end_idx - start_idx)


# def vectorized_cut(normalized_df, center, windows, stats_names):
#     grouped = normalized_df.group_by("iter", maintain_order=True)

#     results = []

#     for group in grouped:
#         results.append(
#             vectorized_cut_vcf(group[1], center, windows, stats_names).with_columns(
#                 pl.lit(group[0][0]).alias("iter")
#             )
#         )

#     return pl.concat(results)


# def cut_vcf(normalized_df, center, windows, stats_names):
#     positions = normalized_df["positions"].to_numpy()

#     lower = center - int(6e5)
#     upper = center + int(6e5)

#     start_idx = np.searchsorted(positions, lower, side="left")
#     end_idx = np.searchsorted(positions, upper, side="right")

#     normalized_df = normalized_df.slice(start_idx, end_idx - start_idx)

#     results = []

#     # Slice the data around the center position using slice_window
#     inner_center = np.linspace(center - 1e5, center + 1e5, 21).astype(int)
#     results = []
#     for c, w in list(product(inner_center, windows)):
#         # Define window boundaries
#         lower = c - w // 2
#         upper = c + w // 2

#         # Filter data by center and window boundaries
#         window_data = normalized_df.filter(
#             (pl.col("positions") >= lower) & (pl.col("positions") <= upper)
#         )

#         # Calculate mean statistics for window
#         window_stats = window_data.select(stats_names).fill_nan(None).mean()

#         # Add metadata columns
#         metadata_cols = [
#             pl.lit(center).alias("iter").cast(pl.Int64),
#             pl.lit(c).alias("center").cast(pl.Int64),
#             pl.lit(w).alias("window").cast(pl.Int64),
#         ]

#         results.append(window_stats.with_columns(metadata_cols))

#     return (
#         pl.concat(results, how="vertical").select(
#             ["iter", "center", "window"] + stats_names
#         )
#         if results
#         else None
#     )


# def summary_statistics_v2(
#     data_dir,
#     vcf=False,
#     nthreads=1,
#     center=[500000, 700000],
#     windows=[1000000],
#     step=10000,
#     recombination_map=None,
# ):
#     """
#     Computes summary statistics across multiple simulations or empirical data, potentially using
#     multiple threads for parallel computation. The statistics are calculated over
#     defined genomic windows, with optional mispolarization applied to the haplotype data.
#     Save the dataframe to a parquet file


#     Only iHS, nSL and iSAFE are estimated across all windows/center combination. The other
#     statistics used the actual center (1.2e6 / 2) extended to 500Kb each flank.

#     Parameters
#     ----------
#     sims : str,
#         Discoal simulation path or VCF file. If VCF file ensure you're use `vcf=True` argument.
#     nthreads : int, optional (default=1)
#         The number of threads to use for parallel computation. If set to 1,
#         the function runs in single-threaded mode. Higher values will enable
#         multi-threaded processing to speed up calculations.
#     center : list of int, optional (default=[500000, 700000])
#         A list specifying the center positions (in base pairs) for the analysis windows.
#         If one center is provided, it will use that as a single point; otherwise,
#         the analysis will cover the range between the two provided centers.

#     windows : list of int, optional (default=[1000000])
#         A list of window sizes (in base pairs) over which summary statistics will be computed.

#     step : int, optional (default=10000)
#         The step size (in base pairs) for sliding windows in the analysis. This determines
#         how much the analysis window moves along the genome for each iteration.
#     vcf : bool,
#         If true parse vcf

#     Returns
#     -------
#     summary_stats : pandas.DataFrame
#         A DataFrame containing the computed summary statistics for each simulation and
#         for each genomic window.

#     """
#     # Validate data directories

#     fvs_file = defaultdict(str)
#     regions = defaultdict()

#     if vcf:
#         required_folders = ["vcfs"]
#         sims = defaultdict()
#         neutral_save = f"{data_dir}/empirical_bins.pickle"
#         df_params = []
#         # Process VCF files
#         vcf_files = np.sort(glob.glob(f"{data_dir}/vcfs/*vcf.gz"))

#         for vcf_path in vcf_files:
#             # Process each VCF file
#             fs_data = Data(vcf_path, nthreads=nthreads)
#             _sims = fs_data.read_vcf()

#             # Same folder custom fvs name based on input VCF.
#             f_name = os.path.basename(vcf_path)
#             for ext in [".vcf", ".bcf", ".gz"]:
#                 f_name = f_name.replace(ext, "")
#             f_name = f_name.replace(".", "_").lower()

#             # Extract key information
#             _df_params = pl.DataFrame(
#                 {
#                     "model": np.repeat(f_name, len(_sims["sweep"])),
#                     "s": np.zeros(len(_sims["sweep"])),
#                     "t": np.zeros(len(_sims["sweep"])),
#                     "saf": np.zeros(len(_sims["sweep"])),
#                     "eaf": np.zeros(len(_sims["sweep"])),
#                 }
#             )
#             df_params.append(_df_params)

#             fvs_file[f_name] = f"{data_dir}/vcfs/fvs_{f_name}.parquet"
#             sims[f_name] = _sims["sweep"]
#             regions[f_name] = _sims["region"]

#         df_params = pl.concat(df_params)

#         # Opening neutral expectations
#         try:
#             with open(f"{data_dir}/neutral_bins.pickle", "rb") as handle:
#                 neutral_stats_norm = pickle.load(handle)
#         except:
#             print(f"Please estimate fvs on simulations before continue")
#             return None
#     else:
#         required_folders = ["sweep", "neutral"]

#         for folder in required_folders:
#             folder_path = os.path.join(data_dir, folder)
#             if not os.path.exists(folder_path):
#                 raise ValueError(f"Required directory not found: {folder_path}")
#             if not glob.glob(os.path.join(folder_path, "*")):
#                 raise ValueError(f"Directory is empty: {folder_path}")

#         # Read simulation data
#         fs_data = Data(data_dir)
#         sims, df_params = fs_data.read_simulations()

#         # Define file paths
#         neutral_save = f"{data_dir}/neutral_bins.pickle"
#         fvs_file["sims"] = f"{data_dir}/fvs.parquet"

#         # Initialize regions dictionary
#         regions = {k: [None] * len(sims[k]) for k in ["neutral", "sweep"]}

#         # Validate simulation data
#         if not (
#             len(sims["sweep"]) > 0
#             and (len(sims["neutral"]) > 0 or neutral_save is not None)
#         ):
#             raise ValueError("Please input neutral and sweep simulations")

#     ########################
#     # Not using same parallel pool for simulation type and normalize to avoid RAM issues with multiprocessing. loky show errors increasing memory in some function unexpectedly
#     # with Parallel(n_jobs=nthreads, backend="loky", verbose=5) as parallel:

#     # Saving malformed simulations
#     results = defaultdict(lambda: None)
#     binned_data = defaultdict()
#     malformed_files = defaultdict()
#     d_centers = defaultdict()
#     tmp_bins = []

#     for sim_type, sim_data in sims.items():
#         print(sim_type)

#         params = df_params.filter(pl.col("model") == sim_type)[:, 1:].to_numpy()[:, :]

#         if vcf:
#             tmp_center = [
#                 tuple(map(int, r.split(":")[-1].split("-"))) for r in regions[sim_type]
#             ]
#             d_centers[sim_type] = np.array([(w[0] + w[1]) // 2 for w in tmp_center])

#             # Opening and closing parallel by each case
#             with Parallel(
#                 n_jobs=nthreads, backend="multiprocessing", verbose=5
#             ) as parallel:
#                 # joblib working inside, isafe/h12/haf will run by sliding windows using joblib configuration
#                 # hapdaf/sratio/freq and dind will use parallel numba
#                 stats = calculate_stats_vcf(
#                     sim_data,
#                     region=regions[sim_type],
#                     nthreads=nthreads,
#                     parallel_manager=parallel,
#                 )
#         else:
#             d_centers[sim_type] = center
#             # Limit to first 100 simulations for processing (as in original code)
#             paired_data = list(zip(sim_data, regions[sim_type]))[:]

#             # Opening and closing parallel by each case
#             with Parallel(
#                 n_jobs=nthreads, backend="multiprocessing", verbose=5
#             ) as parallel:
#                 stats = parallel(
#                     delayed(calculate_stats)(
#                         hap_data,
#                         _iter,
#                         center=center,
#                         step=step,
#                         neutral=True if sim_type == "neutral" else False,
#                         region=region,
#                     )
#                     for _iter, (hap_data, region) in enumerate(paired_data[:], 1)
#                 )

#         # Clean up results and handle malformed simulations
#         stats, params, malformed = cleaning_summaries(data_dir, stats, params, sim_type)
#         malformed_files[sim_type] = malformed

#         # Store binned data for specific simulation types
#         if sim_type == "neutral":
#             raw_stats, norm_stats = zip(*stats)
#             binned_data[sim_type] = binned_stats(*normalize_neutral(norm_stats))
#         elif sim_type == "sweep":
#             raw_stats, norm_stats = zip(*stats)
#         else:
#             raw_stats, norm_stats = stats
#             # Saving all available chr to normalize after stat estimations
#             tmp_bins.append(norm_stats)

#         # Create summary results

#         if ~np.all(params[:, 3] == 0):
#             params[:, 0] = -np.log(params[:, 0])

#         results[sim_type] = summaries(raw_stats, params)

#     df_fv_cnn = defaultdict()
#     df_fv_cnn_raw = defaultdict()

#     if vcf:
#         # Join all chromsomes windows and estimate expected and std values
#         binned_data["empirical"] = binned_stats(*normalize_neutral(tmp_bins))
#         binned_name = "empirical"

#         for k, stats_values in results.items():
#             print(k)
#             df_fv_w, df_fv_w_raw = normalize_stats(
#                 stats_values,
#                 binned_data[binned_name],
#                 center=d_centers[sim_type],
#                 parallel_manager=parallel,
#                 vcf=vcf,
#             )
#             df_fv_cnn[k] = df_fv_w
#             df_fv_cnn_raw[k] = df_fv_w_raw

#             df_fv_w.write_parquet(fvs_file[k])
#             df_fv_w_raw.write_parquet(fvs_file[k].replace(".parquet", "_raw.parquet"))

#         df_fv_training = pl.concat(df_fv_cnn.values(), how="vertical")
#         df_fv_training_raw = pl.concat(df_fv_cnn_raw.values(), how="vertical")

#     else:
#         binned_name = "neutral"
#         # Normalization is fast and does not consume high RAM. Opening same pool
#         with Parallel(n_jobs=nthreads, backend="loky", verbose=5) as parallel:
#             for k, stats_values in results.items():
#                 df_fv_w, df_fv_w_raw = normalize_stats(
#                     stats_values,
#                     binned_data[binned_name],
#                     center=d_centers[sim_type],
#                     parallel_manager=parallel,
#                     vcf=vcf,
#                 )
#                 df_fv_cnn[k] = df_fv_w
#                 df_fv_cnn_raw[k] = df_fv_w_raw

#         df_fv_training = pl.concat(df_fv_cnn.values(), how="vertical")
#         df_fv_training_raw = pl.concat(df_fv_cnn_raw.values(), how="vertical")

#         df_fv_w.write_parquet(fvs_file["sims"])
#         df_fv_w_raw.write_parquet(fvs_file["sims"].replace(".parquet", "_raw.parquet"))

#     # Save neutral_bins
#     with open(neutral_save, "wb") as handle:
#         pickle.dump(binned_data[binned_name], handle)

#     return df_fv_training, df_fv_training_raw


# def get_empir_freqs_np(hap):
#     """
#     Calculate the empirical frequencies of haplotypes.

#     Parameters:
#     - hap (numpy.ndarray): Array of haplotypes where each column represents an individual and each row represents a SNP.

#     Returns:
#     - k_counts (numpy.ndarray): Counts of each unique haplotype.
#     - h_f (numpy.ndarray): Empirical frequencies of each unique haplotype.
#     """
#     S, n = hap.shape

#     # Count occurrences of each unique haplotype
#     hap_f, k_counts = np.unique(hap, axis=1, return_counts=True)

#     # Sort counts in descending order
#     k_counts = np.sort(k_counts)[::-1]

#     # Calculate empirical frequencies
#     h_f = k_counts / n
#     return k_counts, h_f


# def T_m_statistic(K_counts, K_neutral, windows, K_truncation, sweep_mode=5, i=0):
#     output = []
#     m_vals = K_truncation + 1
#     epsilon_min = 1 / (K_truncation * 100)

#     _epsilon_values = list(map(lambda x: x * epsilon_min, range(1, 101)))
#     epsilon_max = K_neutral[-1]
#     epsilon_values = []

#     for ev in _epsilon_values:
#         # ev = e * epsilon_min
#         if ev <= epsilon_max:
#             epsilon_values.append(ev)
#     epsilon_values = np.array(epsilon_values)

#     for j, w in enumerate(windows):
#         # if(i==132):
#         # break
#         K_iter = K_counts[j]

#         null_likelihood = easy_likelihood(K_neutral, K_iter, K_truncation)

#         alt_likelihoods_by_e = []

#         for e in epsilon_values:
#             alt_likelihoods_by_m = []
#             for m in range(1, m_vals):
#                 alt_like = sweep_likelihood(
#                     K_neutral, K_iter, K_truncation, m, e, epsilon_max
#                 )
#                 alt_likelihoods_by_m.append(alt_like)

#             alt_likelihoods_by_m = np.array(alt_likelihoods_by_m)
#             likelihood_best_m = 2 * (alt_likelihoods_by_m.max() - null_likelihood)

#             if likelihood_best_m > 0:
#                 ml_max_m = (alt_likelihoods_by_m.argmax()) + 1
#             else:
#                 ml_max_m = 0

#             alt_likelihoods_by_e.append([likelihood_best_m, ml_max_m, e])

#         alt_likelihoods_by_e = np.array(alt_likelihoods_by_e)

#         likelihood_real = max(alt_likelihoods_by_e[:, 0])

#         out_index = np.flatnonzero(alt_likelihoods_by_e[:, 0] == likelihood_real)

#         out_intermediate = alt_likelihoods_by_e[out_index]

#         if out_intermediate.shape[0] > 1:
#             constarg = min(out_intermediate[:, 1])

#             outcons = np.flatnonzero(out_intermediate[:, 1] == constarg)

#             out_cons_intermediate = out_intermediate[outcons]

#             if out_cons_intermediate.shape[0] > 1:
#                 out_cons_intermediate = out_cons_intermediate[0]

#             out_intermediate = out_cons_intermediate

#         outshape = out_intermediate.shape

#         if len(outshape) != 1:
#             out_intermediate = out_intermediate[0]

#         out_intermediate = np.concatenate(
#             [out_intermediate, np.array([K_neutral[-1], sweep_mode, w]), K_iter]
#         )

#         output.append(out_intermediate)

#     # output = np.array(output)
#     # return output[output[:, 0].argmax(), :]

#     K_names = ["Kcounts_" + str(i) for i in range(1, K_iter.size + 1)]
#     output = pd.DataFrame(output)
#     output.insert(output.shape[1], "iter", i)

#     output.columns = (
#         [
#             "t_statistic",
#             "m",
#             "frequency",
#             "e",
#             "model",
#             "window_lassi",
#         ]
#         + K_names
#         + ["iter"]
#     )
#     return output


# def Ld(
#     hap: np.ndarray, pos: np.ndarray, min_freq=0.05, max_freq=1, start=None, stop=None
# ) -> tuple:
#     """
#     Compute Kelly Zns statistic (1997) and omega_max. Average r2
#     among every pair of loci in the genomic window.

#     Args:hap (numpy.ndarray): 2D array representing haplotype data. Rows correspond to different mutations, and columnscorrespond to chromosomes.
#     pos (numpy.ndarray): 1D array representing the positions of mutations.
#     min_freq (float, optional): Minimum frequency threshold. Default is 0.05.
#     max_freq (float, optional): Maximum frequency threshold. Default is 1.
#     window (int, optional): Genomic window size. Default is 500000.

#     Returns:tuple: A tuple containing two values:
#     - kelly_zns (float): Kelly Zns statistic.
#     - omega_max (float): Nielsen omega max.
#     """

#     if start is not None or stop is not None:
#         loc = (pos >= start) & (pos <= stop)
#         pos = pos[loc]
#         hap = hap[loc, :]

#     freqs = hap.sum(axis=1) / hap.shape[1]

#     hap_filter = hap[(freqs >= min_freq) & (freqs <= max_freq)]

#     r2_matrix = compute_r2_matrix_upper(hap_filter)
#     # r2_matrix = r2_torch(hap_filter)
#     S = hap_filter.shape[0]
#     zns = r2_matrix.sum() / comb(S, 2)
#     # Index combination to iter
#     omega_max = omega_linear_correct(r2_matrix)
#     # omega_max2 = dps.omega(r2_matrix)[0]

#     # return zns, 0
#     return zns, omega_max


# def r2_matrix(
#     hap: np.ndarray, pos: np.ndarray, min_freq=0.05, max_freq=1, start=None, stop=None
# ):
#     """
#     Compute Kelly Zns statistic (1997) and omega_max. Average r2
#     among every pair of loci in the genomic window.

#     Args:
#         hap (numpy.ndarray): 2D array representing haplotype data. Rows correspond to different mutations, and columns correspond to chromosomes.
#         pos (numpy.ndarray): 1D array representing the positions of mutations.
#         min_freq (float, optional): Minimum frequency threshold. Default is 0.05.
#         max_freq (float, optional): Maximum frequency threshold. Default is 1.
#         window (int, optional): Genomic window size. Default is 500000.

#     Returns: tuple: A tuple containing two values:
#         - kelly_zns (float): Kelly Zns statistic.
#         - omega_max (float): Nielsen omega max.
#     """

#     # if start is not None or stop is not None:
#     #     loc = (pos >= start) & (pos <= stop)
#     #     pos = pos[loc]
#     #     hap = hap[loc, :]

#     freqs = hap.sum(axis=1) / hap.shape[1]
#     freq_filter = (freqs >= min_freq) & (freqs <= max_freq)
#     hap_filter = hap[freq_filter]

#     r2_matrix = compute_r2_matrix(hap_filter)
#     # r2_matrix = r2_torch(hap_filter)
#     # S = hap_filter.shape[0]
#     # zns = r2_matrix.sum() / comb(S, 2)
#     # Index combination to iter
#     # omega_max = omega(r2_matrix)
#     # omega_max = dps.omega(r2_matrix)[0]

#     return r2_matrix, freq_filter
#     # return zns, omega_max


# def Ld(
#     r2_subset,
#     freq_filter,
#     pos: np.ndarray,
#     min_freq=0.05,
#     max_freq=1,
#     start=None,
#     stop=None,
# ):
#     pos_filter = pos[freq_filter]
#     if start is not None or stop is not None:
#         loc = (pos_filter >= start) & (pos_filter <= stop)
#         pos_filter = pos_filter[loc]
#         r2_subset = r2_subset[loc, :][:, loc]

#     # r2_subset_matrix = compute_r2_subset_matrix(hap_filter)
#     # r2_subset_matrix = r2_subset_torch(hap_filter)
#     S = r2_subset.shape[0]
#     kelly_zns = r2_subset.sum() / comb(S, 2)
#     # omega_max = omega(r2_subset)

#     return kelly_zns, 0


# @njit("float64[:,:](int8[:,:])", cache=True)
# def compute_r2_matrix(hap):
#     num_sites = hap.shape[0]

#     # r2_matrix = OrderedDict()
#     sum_r_squared = 0
#     r2_matrix = np.zeros((num_sites, num_sites))
#     # Avoid itertool.combination, not working on numba
#     # for pair in combinations(range(num_sites), 2):

#     # Check index from triangular matrix of size num_sites x num_sites. Each indices correspond to one one dimension of the array. Same as combinations(range(num_sites), 2)
#     c_1, c_2 = np.triu_indices(num_sites, 1)

#     for i, j in zip(c_1, c_2):
#         r2_matrix[i, j] = r2(hap[i, :], hap[j, :])
#         # r2_matrix[pair[0], pair[1]] = r2(hap[pair[0], :], hap[pair[1], :])

#     return r2_matrix


# @njit("float64(float64[:,:])", cache=True)
# def omega(r2_matrix):
#     """
#     Calculates Kim and Nielsen's (2004, Genetics 167:1513) omega_max statistic. Adapted from PG-Alignments-GAN

#     Args:r2_matrix (numpy.ndarray): 2D array representing r2 values.

#     Returns:
#         float: Kim and Nielsen's omega max.
#     """

#     omega_max = 0
#     S_ = r2_matrix.shape[1]

#     if S_ < 3:
#         omega_max = 0
#     else:
#         for l_ in range(3, S_ - 2):
#             sum_r2_L = 0
#             sum_r2_R = 0
#             sum_r2_LR = 0

#             for i in range(S_):
#                 for j in range(i + 1, S_):
#                     ld_calc = r2_matrix[i, j]
#                     if i < l_ and j < l_:
#                         sum_r2_L += ld_calc

#                     elif i >= l_ and j >= l_:
#                         sum_r2_R += ld_calc

#                     elif i < l_ and j >= l_:
#                         sum_r2_LR += ld_calc

#             # l_ ## to keep the math right outside of indexing
#             omega_numerator = (
#                 1 / ((l_ * (l_ - 1) / 2) + ((S_ - l_) * (S_ - l_ - 1) / 2))
#             ) * (sum_r2_L + sum_r2_R)
#             omega_denominator = (1 / (l_ * (S_ - l_))) * sum_r2_LR

#             if omega_denominator == 0:
#                 omega = 0
#             else:
#                 omega = np.divide(omega_numerator, omega_denominator)

#             if omega > omega_max:
#                 omega_max = omega

#     return omega_max


# @njit
# def sq_freq_pairs(hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size):
#     # Compute counts and frequencies
#     hap_derived = hap
#     hap_ancestral = np.bitwise_xor(hap_derived, 1)
#     derived_count = ac[:, 1]
#     ancestral_count = ac[:, 0]
#     freqs = ac[:, 1] / ac.sum(axis=1)

#     # Focal filter
#     focal_filter = (freqs >= min_focal_freq) & (freqs <= max_focal_freq)
#     focal_derived = hap_derived[focal_filter, :]
#     focal_derived_count = derived_count[focal_filter]
#     focal_ancestral = hap_ancestral[focal_filter, :]
#     focal_ancestral_count = ancestral_count[focal_filter]
#     focal_index = focal_filter.nonzero()[0]

#     # Allocate fixed-size lists to avoid growing lists
#     sq_out = [np.zeros((0, 3))] * len(focal_index)
#     # info = [None] * len(focal_index)
#     info = np.zeros((len(focal_index), 4))
#     # Main loop to calculate frequencies
#     for j in range(len(focal_index)):
#         i = focal_index[j]
#         size = window_size / 2

#         # Find indices within the window
#         # z = np.flatnonzero(np.abs(rec_map[i, -1] - rec_map[:, -1]) <= size)
#         mask = np.abs(rec_map[i, -1] - rec_map[:, -1]) <= size
#         z = np.where(mask)[0]

#         # Index range
#         x_r, y_r = i + 1, z[-1]
#         x_l, y_l = z[0], i - 1

#         # Calculate derived and ancestral frequencies
#         f_d_l = (
#             np.sum(focal_derived[j] & hap_derived[x_l : y_l + 1], axis=1)
#             / focal_derived_count[j]
#         )
#         f_a_l = (
#             np.sum(focal_ancestral[j] & hap_derived[x_l : y_l + 1], axis=1)
#             / focal_ancestral_count[j]
#         )
#         f_tot_l = freqs[x_l : y_l + 1]

#         f_d_r = (
#             np.sum(focal_derived[j] & hap_derived[x_r : y_r + 1], axis=1)
#             / focal_derived_count[j]
#         )
#         f_a_r = (
#             np.sum(focal_ancestral[j] & hap_derived[x_r : y_r + 1], axis=1)
#             / focal_ancestral_count[j]
#         )
#         f_tot_r = freqs[x_r : y_r + 1]

#         # Concatenate frequencies into a single array
#         sq_freqs = np.empty((f_d_l.size + f_d_r.size, 3))
#         sq_freqs[: f_d_l.size, 0] = f_d_l[::-1]
#         sq_freqs[: f_d_l.size, 1] = f_a_l[::-1]
#         sq_freqs[: f_d_l.size, 2] = f_tot_l[::-1]
#         sq_freqs[f_d_l.size :, 0] = f_d_r
#         sq_freqs[f_d_l.size :, 1] = f_a_r
#         sq_freqs[f_d_l.size :, 2] = f_tot_r

#         sq_out[j] = sq_freqs
#         info[j] = np.array(
#             [rec_map[i, -1], freqs[i], focal_derived_count[j], focal_ancestral_count[j]]
#         )

#     return sq_out, info


# def compare_haplos(haplo_1, haplo_2):
#     identical = haplo_1.count("1")  # Count "1"s in haplo_1
#     different = sum(1 for h1, h2 in zip(haplo_1, haplo_2) if h1 != h2)
#     total = identical + different  # Total equals identical + different

#     return identical, different, total


# def run_h12(
#     hap,
#     rec_map,
#     _iter=1,
#     neutral=True,
#     script="/home/jmurgamoreno/software/calculate_H12_modified.pl",
# ):
#     df_hap = pl.DataFrame(hap)
#     df_rec_map = pl.DataFrame(rec_map)
#     hap_file = "/tmp/tmp_" + str(_iter) + ".hap"
#     map_file = "/tmp/tmp_" + str(_iter) + ".map"
#     with open(hap_file, "w") as f:
#         for row in df_hap.iter_rows():
#             f.write("".join(map(str, row)) + "\n")

#     df_rec_map.write_csv(map_file, include_header=False, separator=" ")

#     h12_pl = "perl " + script + " " + hap_file + " " + map_file + " out "
#     h12_pl += "500000 " if neutral else "1200000"

#     with subprocess.Popen(h12_pl.split(), stdout=subprocess.PIPE) as process:
#         h12_v = float(process.stdout.read())

#     os.remove(hap_file)
#     os.remove(map_file)

#     return h12_v


# def run_haf(
#     hap,
#     rec_map,
#     _iter=1,
#     neutral=True,
#     script="/home/jmurgamoreno/software/calculate_HAF.pl",
# ):
#     df_hap = pl.DataFrame(hap)
#     df_rec_map = pl.DataFrame(rec_map)
#     hap_file = "/tmp/tmp_" + str(_iter) + ".hap"
#     map_file = "/tmp/tmp_" + str(_iter) + ".map"
#     with open(hap_file, "w") as f:
#         for row in df_hap.iter_rows():
#             f.write("".join(map(str, row)) + "\n")

#     df_rec_map.write_csv(map_file, include_header=False, separator=" ")

#     haf_pl = "perl " + script + " " + hap_file + " " + map_file + " out "
#     haf_pl += "500000 " if neutral else "1200000"

#     with subprocess.Popen(haf_pl.split(), stdout=subprocess.PIPE) as process:
#         haf_v = float(process.stdout.read())

#     os.remove(hap_file)
#     os.remove(map_file)

#     return haf_v


# def run_hapbin_og(
#     hap,
#     rec_map,
#     _iter=1,
#     min_ehh=0.05,
#     min_maf=0.05,
#     ihsbin=None,
# ):
#     if ihsbin is None:
#         ihsbin = shutil.which("ihsbin")
#     df_hap = pl.DataFrame(hap)
#     df_rec_map = pl.DataFrame(
#         rec_map,
#         pl.Schema(
#             [
#                 ("chr", pl.Int64),
#                 ("location", pl.Int64),
#                 ("cm_mb", pl.Float64),
#                 ("cm", pl.Float64),
#             ]
#         ),
#     )
#     hap_file = "/tmp/tmp_" + str(_iter) + ".hap"
#     map_file = "/tmp/tmp_" + str(_iter) + ".map"

#     with open(hap_file, "w") as f:
#         for row in df_hap.iter_rows():
#             f.write("".join(str(value) for value in row) + "\n")

#     df_rec_map.write_csv(map_file, include_header=False, separator=" ")

#     out_file = f"/tmp/tmp_{_iter}.out"
#     cmd_hapbin = f"{ihsbin} --hap {hap_file} --map {map_file} --cutoff {min_ehh} --minmaf {min_maf} --out {out_file}"

#     ihs_output = subprocess.run(cmd_hapbin, shell=True, capture_output=True, text=True)

#     tmp_ihs = pl.read_csv(
#         out_file,
#         separator="\t",
#         schema=pl.Schema(
#             [
#                 ("location", pl.Int64),
#                 ("ihh_0", pl.Float64),
#                 ("ihh_1", pl.Float64),
#                 ("ihs", pl.Float64),
#                 ("std_ihs", pl.Float64),
#             ]
#         ),
#     )

#     r_pos = dict(zip(rec_map[:, 1], rec_map[:, -1]))
#     _p = np.array([r_pos[i] for i in tmp_ihs["location"]])
#     df_daf = pl.DataFrame(
#         {"positions": rec_map[:, -1].astype(int), "daf": (hap.sum(axis=1) / hap.shape[1]) * 2}
#     )
#     df_ihs = (
#         tmp_ihs.with_columns(pl.lit(_p).cast(pl.Int64).alias("positions"))
#         .join(df_daf, on="positions", coalesce=True)
#         .with_columns(
#             (pl.col("ihh_1") - pl.col("ihh_0"))
#             .abs()
#             .cast(pl.Float64)
#             .alias("delta_ihh")
#         )
#         .select(["positions", "daf", "ihs", "delta_ihh"])
#     ).fill_nan(None)

#     os.remove(hap_file)
#     os.remove(map_file)
#     os.remove(out_file)

#     return df_ihs
