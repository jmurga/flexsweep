Advanced usage
==============

Custom summary statistics
-------------------------
<<<<<<< HEAD
You can extend Flex-sweep to any other summary statistic provided within the software or your own. For this purpose, you only need to mimic the following example function to estimate these stats, taking advantage of the current API. Note that the example functions follow the same logic as ``fs.fv.calculate_stats_simplify_custom`` to mimic the expected outputs to normalise and create the Flex-sweep feature vectors.

While the original Flex-sweep implementation not only avoids the windowed statistics employed (HAF and H12) but also repeats values across the selected windows/centre combination, the function examples below, based on ``fs.fv.calculate_stats_simplify_custom``, do. You must pay attention to how it creates dictionaries containing raw statistics data and a dictionary to normalise SNP-based and window-based statistics independently. SNP-based statistics are normalised following `Voight et al 2008 <https://doi.org/10.1371/journal.pbio.0040072>`_. For window-based statistics, we estimated the Z-score for each window and centre combination. E.g :math:`\pi` normalisation across window/centre:

=======

Select any combination of built-in statistics by passing a list of names to
the ``stats`` argument of ``summary_statistics``. The pipeline handles
computation, windowing, and normalisation automatically for both simulations
and VCF data.

By default (``stats=None``) the full Flex-sweep statistic set is used.

Normalisation follows two schemes. SNP-based statistics are normalised by
frequency bin following `Voight et al. 2006
<https://doi.org/10.1371/journal.pbio.0040072>`_. Window-based statistics are
Z-scored per window/centre combination:
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

.. math::

   Z\text{-score }\pi_i^{(w_1,c_1)}
   \;=\;
   \frac{\pi_i^{(w_1,c_1)} - \mu^{(w_1,c_1)}}{\sigma^{(w_1,c_1)}},
   \qquad \\
   \text{where }\mu^{(w_1,c_1)}=\operatorname{mean}_{i=1}^{n}\!\bigl(\pi_i^{(w_1,c_1)}\bigr)
<<<<<<< HEAD
   \text{ and }\sigma^{(w_1,c_1)}=\operatorname{sd}_{i=1}^{n}\!\bigl(\pi_i^{(w_1,c_1)}\bigr).


**It is crucial to mimic the outputs of the example functions, otherwise the API could not normalise or process the feature vector properly.**


First of all, import all the feature vector module content to make everything easier.
=======
   \text{, and }\sigma^{(w_1,c_1)}=\operatorname{sd}_{i=1}^{n}\!\bigl(\pi_i^{(w_1,c_1)}\bigr).


Available statistics
~~~~~~~~~~~~~~~~~~~~

**Window-based statistics** — computed for every centre × window-size
combination:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Name
     - Description
   * - ``pi``
     - Nucleotide diversity :math:`\pi` per base pair
   * - ``tajima_d``
     - Tajima's :math:`D`
   * - ``theta_w``
     - Watterson's :math:`\theta_W` per base pair
   * - ``theta_h``
     - Fay & Wu's :math:`\theta_H` (absolute)
   * - ``fay_wu_h``
     - Normalised Fay & Wu's :math:`H`
   * - ``k_counts``
     - Number of distinct haplotypes
   * - ``h1``
     - Garud's :math:`H_1`
   * - ``h12``
     - Garud's :math:`H_{12}`
   * - ``h2_h1``
     - Garud's :math:`H_2/H_1`
   * - ``zns``
     - Kelly's :math:`Z_{nS}`
   * - ``omega_max``
     - Kim & Nielsen's :math:`\omega_{max}`
   * - ``haf``
     - Haplotype allele frequency (HAF-top)
   * - ``max_fda``
     - Maximum derived allele frequency in window
   * - ``dist_var``
     - Variance of pairwise haplotype distances
   * - ``dist_skew``
     - Skewness of pairwise haplotype distances
   * - ``dist_kurtosis``
     - Kurtosis of pairwise haplotype distances

**SNP-based statistics** — computed per variant and normalised by frequency
bin:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Name
     - Description
   * - ``ihs``
     - Integrated haplotype score (iHS)
   * - ``delta_ihh``
     - :math:`\Delta iHH`
   * - ``nsl``
     - Number of segregating sites by length (:math:`nS_L`)
   * - ``isafe``
     - iSAFE sweep-focal SNP score
   * - ``dind`` / ``dind_high_low``
     - DIND (derived-background diversity ratio)
   * - ``highfreq`` / ``high_freq``
     - High-frequency derived allele statistic
   * - ``lowfreq`` / ``low_freq``
     - Low-frequency derived allele statistic
   * - ``s_ratio``
     - S-ratio (singletons vs high-frequency variants)
   * - ``hapdaf_o``
     - HapDAF observed
   * - ``hapdaf_s``
     - HapDAF simulated

Additional statistics (Python API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following statistics are implemented in ``flexsweep.fv_v2`` but are not
accessible via the ``stats=`` argument. They must be called directly from the
Python API (see :doc:`api` for full signatures).

**SFS-based neutrality tests:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``achaz_y``
     - Achaz's :math:`Y` (`Achaz 2008 <https://doi.org/10.1534/genetics.107.082198>`_)
   * - ``zeng_e``
     - Zeng's :math:`E` (`Zeng et al. 2006 <https://doi.org/10.1534/genetics.106.061432>`_)
   * - ``fuli_f_star``
     - Fu & Li's :math:`F^*` (`Fu & Li 1993 <https://doi.org/10.1093/genetics/133.3.693>`_)
   * - ``fuli_f``
     - Fu & Li's :math:`F`
   * - ``fuli_d_star``
     - Fu & Li's :math:`D^*`
   * - ``fuli_d``
     - Fu & Li's :math:`D`

**Composite sweep statistics:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``compute_t_m``
     - LASSI :math:`T` and :math:`m` statistics
       (`Harris & DeGiorgio 2020 <https://doi.org/10.1093/molbev/msaa115>`_)
   * - ``mu_stat``
     - RAiSD :math:`\mu` statistic
       (`Alachiotis & Pavlidis 2018 <https://doi.org/10.1038/s42003-018-0085-8>`_)

**Balancing selection:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``run_beta_window``
     - :math:`\beta^{(1)*}_{(std)}`
       (`Siewert & Voight 2020 <https://doi.org/10.1093/gbe/evaa013>`_)
   * - ``ncd1``
     - NCD1 (`Bitarello et al. 2018 <https://doi.org/10.1093/gbe/evy054>`_)


Selecting statistics
~~~~~~~~~~~~~~~~~~~~
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

.. code-block:: python

    import flexsweep as fs
<<<<<<< HEAD
    from flexsweep.fv import *


diploS/HIC like statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say we want to estimate the same statistics as `diploS/HIC <https://doi.org/10.1534/g3.118.200262>`_: :math:`\pi`, :math:`\theta_{w}`, :math:`\theta_{H}`, the number of distinct haplotypes, H1, H12 and H2/H1, :math:`Z_{nS}`, and :math:`\omega_{max}`. In addition, we will be using only one centre while decreasing the window sizes. You must modify as needed both functions, the function processing simulations as well as the function processing VCF files.


Note that diploS/HIC does not estimate any SNP-based statistics. Then we must return ``None`` in the ``fs.fv. calculate_stats_simplify_custom`` normalisation dictionary. Do the same if you are not estimating any windowed statistic. You will find below examples to estimate diploS/HIC statistics in simulations and VCF. You must start importing the Flex-sweep package as well as the entire feature vector module to easily access the statistics functions


Here you have the code to create diploS/HIC like feature vectors from simulated data

.. code-block:: python

    def calculate_stats_diploshic(
        hap_data,
        _iter=1,
        center=[6e5],
        windows=[10000, 20000, 50000, 100000, 200000],
        step=1e4,
    ):
        filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in scalar divide",
        )
        np.seterr(divide="ignore", invalid="ignore")

        try:
            (
                hap_int,
                rec_map_01,
                ac,
                biallelic_mask,
                position_masked,
                genetic_position_masked,
            ) = parse_and_filter_ms(hap_data)
            freqs = ac[:, 1] / ac.sum(axis=1)
            if hap_int.shape[0] != rec_map_01.shape[0]:
                return None, None, None
        except:
            return None, None, None

        if len(center) == 1:
            centers = np.arange(center[0], center[0] + step, step).astype(int)
        else:
            centers = np.arange(center[0], center[1] + step, step).astype(int)


        # COMPUTE WINDOWED STATS

        _r2 = compute_r2_matrix_upper(hap_int)

        _tmp_window = []
        for c, w in product(centers, windows):

            lower = c - w // 2
            upper = c + w // 2

            mask = (position_masked >= lower) & (position_masked <= upper)

            _tmp_hap = hap_int[mask]
            _tmp_pos = position_masked[mask]
            _ac = ac[mask]
            if _tmp_hap.size == 0:

                _theta_pi_v = np.nan
                _theta_w_v = np.nan
                _theta_h_v = np.nan
                k_counts = np.nan
                h12_v = np.nan
                h2_h1 = np.nan
                h1_v = np.nan
                zns_v = np.nan
                omega_max = np.nan

            else:

                _theta_pi_v = theta_pi(_ac).sum() / (upper - lower + 1)
                _theta_w_v = theta_watterson(_ac,_tmp_pos)
                _theta_h_v = fay_wu_h_norm(_ac)[0]
                k_counts = HaplotypeArray(_tmp_hap).distinct_counts().size

                try:
                    h12_v, h2_h1, h1_v, h123 = garud_h_numba(_tmp_hap)
                except:
                    h12_v, h2_h1, h1_v, h123 = np.nan, np.nan, np.nan

                zns_v, omega_max = Ld(_r2, mask)

            _tmp_window.append(
                np.array(
                    [
                        int(_iter),
                        int(c),
                        int(w),
                        _theta_pi_v,
                        _theta_w_v,
                        _theta_h_v,
                        k_counts,
                        h1_v,
                        h12_v,
                        h2_h1,
                        zns_v,
                        omega_max
                    ]
                )
            )

        # CREATE YOUR WINDOWED STAT DATAFRAME
        df_window_new = pl.DataFrame(
            np.vstack(_tmp_window),
            schema=pl.Schema(
                [
                    ("iter", pl.Int64),
                    ("center", pl.Int64),
                    ("window", pl.Int64),
                    ("pi", pl.Float64),
                    ("theta_w", pl.Float64),
                    ("theta_h", pl.Float64),
                    ("k_counts", pl.Float64),
                    ("h1", pl.Float64),
                    ("h12", pl.Float64),
                    ("h2_h1", pl.Float64),
                    ("omega_max", pl.Float64),
                    ("zns", pl.Float64),
                ]
            ),
        )

        # SAVE ANY STATISTIC DATAFRAME IN d_stats DICTIONARY
        d_stats = {}
        d_stats["window"] = df_window_new

        # SAVE STATS TO NORM
        d_stats_to_norm = {"snps": None, "windows": df_window_new}

        return d_stats, d_stats_to_norm


Here you have the code to create diploS/HIC like feature vectors from empirical data (VCF files)

.. code-block:: python


    def calculate_stats_diploshic_vcf(
        vcf_file,
        region,
        center=[int(6e5)],
        windows=[10000, 20000, 50000, 100000, 200000],
        step=1e4,
        _iter=1,
        recombination_map=None,
        nthreads=1,
    ):
        filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in scalar divide",
        )
        np.seterr(divide="ignore", invalid="ignore")

        try:
            (
                hap_int,
                rec_map_01,
                ac,
                biallelic_mask,
                position_masked,
                genetic_position_masked,
            ) = genome_reader(
                vcf_file, recombination_map=recombination_map, region=None
            )
            freqs = ac[:, 1] / ac.sum(axis=1)
        except:
            return None

        if recombination_map is None:
            genetic_position_masked = None

        genomic_windows = [tuple(map(int, r.split(":")[-1].split("-"))) for r in region]
        nchr = region[0].split(":")[0]

        if len(center) == 1:
            centers = np.arange(center[0], center[0] + step, step).astype(int)
        else:
            centers = np.arange(center[0], center[1] + step, step).astype(int)


        # ESTIMATE WINDOWED STATS
        def run_windowed_stats(hap, ac_subset, positions, genomic_window, center, window,_iter=1):

            if hap.size != 0:
                # USE 6E5 AS ACTUAL CENTER, CHANGE EMPIRICAL WINDOWS POSITIONS TO RANGE 1-1.2E6
                # CONCORDANCE WITH SUMMARY STATISTIC SIMULATION ESTIMATION
                positions_relative = relative_position(positions, genomic_window)

                # ESTIMATE STATS
                _r2 = compute_r2_matrix_upper(hap)

                _tmp_window = []
                for c, w in product(centers, windows):
                    lower = c - w // 2
                    upper = c + w // 2
                    mask = (positions_relative >= lower) & (positions_relative <= upper)

                    _tmp_hap = hap[mask]
                    _tmp_pos = positions[mask]
                    _ac = ac_subset[mask]
                    if _tmp_hap.size == 0:

                        _theta_pi_v = np.nan
                        _theta_w_v = np.nan
                        _theta_h_v = np.nan
                        k_counts = np.nan
                        h12_v = np.nan
                        h2_h1 = np.nan
                        h1_v = np.nan
                        zns_v = np.nan
                        omega_max = np.nan
                    else:
                        _theta_pi_v = theta_pi(_ac).sum() / (upper - lower + 1)
                        _theta_w_v = theta_watterson(_ac,_tmp_pos)
                        _theta_h_v = fay_wu_h_norm(_ac)[0]
                        k_counts = HaplotypeArray(_tmp_hap).distinct_counts().size

                        try:
                            h12_v, h2_h1, h1_v, h123 = garud_h_numba(_tmp_hap)
                        except:
                            h12_v, h2_h1, h1_v, h123 = np.nan, np.nan, np.nan

                        zns_v, omega_max = Ld(_r2, mask)

                    _tmp_window.append(
                        np.array(
                            [
                                _iter,
                                int(c),
                                int(w),
                                _theta_pi_v,
                                _theta_w_v,
                                _theta_h_v,
                                k_counts,
                                h1_v,
                                h12_v,
                                h2_h1,
                                zns_v,
                                omega_max
                            ]
                        )
                    )


                df_window_new = pl.DataFrame(
                    np.vstack(_tmp_window),
                    schema=pl.Schema(
                        [
                            ("iter", pl.Utf8),
                            ("center", pl.Int64),
                            ("window", pl.Int64),
                            ("pi", pl.Float64),
                            ("theta_w", pl.Float64),
                            ("theta_h", pl.Float64),
                            ("k_counts", pl.Float64),
                            ("h1", pl.Float64),
                            ("h12", pl.Float64),
                            ("h2_h1", pl.Float64),
                            ("omega_max", pl.Float64),
                            ("zns", pl.Float64),
                        ]
                    ),
                )
            else:
                df_window_new = pl.DataFrame(
                    schema=pl.Schema(
                        [
                            ("iter", pl.Utf8),
                            ("center", pl.Int64),
                            ("window", pl.Int64),
                            ("pi", pl.Float64),
                            ("theta_w", pl.Float64),
                            ("theta_h", pl.Float64),
                            ("k_counts", pl.Float64),
                            ("h1", pl.Float64),
                            ("h12", pl.Float64),
                            ("h2_h1", pl.Float64),
                            ("omega_max", pl.Float64),
                            ("zns", pl.Float64),
                        ]
                    ),
                )

            return df_window_new

        with Parallel(n_jobs=nthreads, backend="loky", verbose=5) as parallel:
            df_window_new = parallel(
                    delayed(run_windowed_stats)(
                        hap_int[
                            (position_masked >= genomic_window[0])
                            & (position_masked <= genomic_window[1])
                        ],
                        ac[
                            (position_masked >= genomic_window[0])
                            & (position_masked <= genomic_window[1])],
                        position_masked[
                            (position_masked >= genomic_window[0])
                            & (position_masked <= genomic_window[1])
                        ],
                        genomic_window,
                        centers,
                        windows,
                        _iter=f"{nchr}:{genomic_window[0]}-{genomic_window[1]}",
                    )
                    for _iter, (genomic_window) in enumerate(genomic_windows[:], 1)
                )

        df_window_new = pl.concat(df_window_new)
        d_stats = {}
        d_stats["window"] = df_window_new

        for k, df in d_stats.items():
            if k == 'window':
                continue
            d_stats[k] = df.with_columns([pl.lit(nchr).cast(pl.Utf8).alias("iter")])

        return d_stats, {"snps": None, "windows": df_window_new}


Custom statistics combination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here you have an example where both , SNP-based and window-based statistics are created. Pay attention how to save Polars DataFrames stasitics into the output dictionaries.

.. code-block:: python

    def calculate_stats_custom(
        hap_data,
        _iter=1,
        center=[5e5,7e5],
        windows=[10000, 20000, 50000, 100000, 200000],
        step=1e5,
    ):
        filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in scalar divide",
        )
        np.seterr(divide="ignore", invalid="ignore")

        try:
            (
                hap_int,
                rec_map_01,
                ac,
                biallelic_mask,
                position_masked,
                genetic_position_masked,
            ) = parse_and_filter_ms(hap_data)
            freqs = ac[:, 1] / ac.sum(axis=1)
            if hap_int.shape[0] != rec_map_01.shape[0]:
                return None, None, None
        except:
            return None, None, None

        if len(center) == 1:
            centers = np.arange(center[0], center[0] + step, step).astype(int)
        else:
            centers = np.arange(center[0], center[1] + step, step).astype(int)

        # COMPUTE SNPS STATS
        df_s_ratio = s_ratio(hap_int,ac,rec_map_01)
        df_hapdaf_o = hapdaf_o(hap_int,ac,rec_map_01)
        df_hapdaf_s = hapdaf_s(hap_int,ac,rec_map_01)

        df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
        df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
        df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)


        # COMPUTE WINDOWED STATS
        _tmp_window = []
        for c, w in product(centers, windows):

            lower = c - w // 2
            upper = c + w // 2

            mask = (position_masked >= lower) & (position_masked <= upper)

            _tmp_hap = hap_int[mask]
            _tmp_pos = position_masked[mask]
            _ac = ac[mask]
            if _tmp_hap.size == 0:

                _theta_pi_v = np.nan
                _h_v = np.nan
                h12_v = np.nan

            else:

                _theta_pi_v = theta_pi(_ac).sum() / (upper - lower + 1)
                _h_v = fay_wu_h_norm(_ac)[-1]

                try:
                    h12_v, h2_h1, h1_v, h123 = garud_h_numba(_tmp_hap)
                except:
                    h12_v, h2_h1, h1_v, h123 = np.nan, np.nan, np.nan


            _tmp_window.append(
                np.array(
                    [
                        int(_iter),
                        int(c),
                        int(w),
                        _theta_pi_v,
                        _h_v,
                        h12_v,
                    ]
                )
            )

        # CREATE YOUR WINDOWED STAT DATAFRAME
        df_window_new = pl.DataFrame(
            np.vstack(_tmp_window),
            schema=pl.Schema(
                [
                    ("iter", pl.Int64),
                    ("center", pl.Int64),
                    ("window", pl.Int64),
                    ("pi", pl.Float64),
                    ("fay_wu_h", pl.Float64),
                    ("h12", pl.Float64),
                ]
            ),
        )

        # SAVE ANY STATISTIC DATAFRAME IN d_stats DICTIONARY
        d_stats = {}
        d_stats["s_ratio"] = df_s_ratio
        d_stats["hapdaf_o"] = df_hapdaf_o
        d_stats["hapdaf_s"] = df_hapdaf_s
        d_stats["window"] = df_window_new

        # SAVE STATS TO NORM
        # MERGE ALL SNP-BASED STATS
        df_stats_norm = (
            reduce(
                lambda left, right: left.join(
                    right,
                    on=["iter", "positions", "daf"],
                    how="full",
                    coalesce=True,
                ),
                [
                    df_s_ratio.lazy(),
                    df_hapdaf_o.lazy(),
                    df_hapdaf_s.lazy(),
                ],
            )
            .sort("positions")
            .collect()
        )
        d_stats_to_norm = {"snps": df_stats_norm, "windows": df_window_new}

        return d_stats, d_stats_to_norm



.. code-block:: python


    def calculate_stats_custom_vcf(
        vcf_file,
        region,
        center=[int(6e5)],
        windows=[10000, 20000, 50000, 100000, 200000],
        step=1e4,
        _iter=1,
        recombination_map=None,
        nthreads=1,
    ):
        filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in scalar divide",
        )
        np.seterr(divide="ignore", invalid="ignore")

        try:
            (
                hap_int,
                rec_map_01,
                ac,
                biallelic_mask,
                position_masked,
                genetic_position_masked,
            ) = genome_reader(
                vcf_file, recombination_map=recombination_map, region=None
            )
            freqs = ac[:, 1] / ac.sum(axis=1)
        except:
            return None

        if recombination_map is None:
            genetic_position_masked = None

        genomic_windows = [tuple(map(int, r.split(":")[-1].split("-"))) for r in region]
        nchr = region[0].split(":")[0]

        if len(center) == 1:
            centers = np.arange(center[0], center[0] + step, step).astype(int)
        else:
            centers = np.arange(center[0], center[1] + step, step).astype(int)

        # ESTIMATE STATS
        df_s_ratio = s_ratio(hap_int,ac,rec_map_01)
        df_hapdaf_o = hapdaf_o(hap_int,ac,rec_map_01)
        df_hapdaf_s = hapdaf_s(hap_int,ac,rec_map_01)

        df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
        df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
        df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)

        # ESTIMATE WINDOWED STATS
        def run_windowed_stats(hap, ac_subset, positions, genomic_window, center, window,_iter=1):

            if hap.size != 0:
                # USE 6E5 AS ACTUAL CENTER, CHANGE EMPIRICAL WINDOWS POSITIONS TO RANGE 1-1.2E6
                # CONCORDANCE WITH SUMMARY STATISTIC SIMULATION ESTIMATION
                positions_relative = relative_position(positions, genomic_window)

                # ESTIMATE STATS
                _tmp_window = []
                for c, w in product(centers, windows):
                    lower = c - w // 2
                    upper = c + w // 2
                    mask = (positions_relative >= lower) & (positions_relative <= upper)

                    _tmp_hap = hap[mask]
                    _tmp_pos = positions[mask]
                    _ac = ac_subset[mask]
                    if _tmp_hap.size == 0:

                        _theta_pi_v = np.nan
                        _h_v = np.nan
                        h12_v = np.nan

                    else:
                        _theta_pi_v = theta_pi(_ac).sum() / (upper - lower + 1)
                        _h_v = fay_wu_h_norm(_ac)[-1]

                        try:
                            h12_v, h2_h1, h1_v, h123 = garud_h_numba(_tmp_hap)
                        except:
                            h12_v, h2_h1, h1_v, h123 = np.nan, np.nan, np.nan

                    _tmp_window.append(
                        np.array(
                            [
                                _iter,
                                int(c),
                                int(w),
                                _theta_pi_v,
                                _h_v,
                                h12_v,
                            ]
                        )
                    )


                df_window_new = pl.DataFrame(
                    np.vstack(_tmp_window),
                    schema=pl.Schema(
                        [
                            ("iter", pl.Utf8),
                            ("center", pl.Int64),
                            ("window", pl.Int64),
                            ("pi", pl.Float64),
                            ("fay_wu_h", pl.Float64),
                            ("h12", pl.Float64),
                        ]
                    )
                )
            else:
                df_window_new = pl.DataFrame(
                    schema=pl.Schema(
                        [
                            ("iter", pl.Utf8),
                            ("center", pl.Int64),
                            ("window", pl.Int64),
                            ("pi", pl.Float64),
                            ("fay_wu_h", pl.Float64),
                            ("h12", pl.Float64),
                        ]
                    ),
                )

            return df_window_new

        with Parallel(n_jobs=nthreads, backend="loky", verbose=5) as parallel:
            df_window_new = parallel(
                    delayed(run_windowed_stats)(
                        hap_int[
                            (position_masked >= genomic_window[0])
                            & (position_masked <= genomic_window[1])
                        ],
                        ac[
                            (position_masked >= genomic_window[0])
                            & (position_masked <= genomic_window[1])],
                        position_masked[
                            (position_masked >= genomic_window[0])
                            & (position_masked <= genomic_window[1])
                        ],
                        genomic_window,
                        centers,
                        windows,
                        _iter=f"{nchr}:{genomic_window[0]}-{genomic_window[1]}",
                    )
                    for _iter, (genomic_window) in enumerate(genomic_windows[:], 1)
                )

        df_window_new = pl.concat(df_window_new)

        # SAVE ANY STATISTIC DATAFRAME IN d_stats DICTIONARY
        d_stats = {}
        d_stats["s_ratio"] = df_s_ratio
        d_stats["hapdaf_o"] = df_hapdaf_o
        d_stats["hapdaf_s"] = df_hapdaf_s
        d_stats["window"] = df_window_new

        # SAVE STATS TO NORM
        # MERGE ALL SNP-BASED STATS
        df_stats_norm = (
            reduce(
                lambda left, right: left.join(
                    right,
                    on=["iter", "positions", "daf"],
                    how="full",
                    coalesce=True,
                ),
                [
                    df_s_ratio.lazy(),
                    df_hapdaf_o.lazy(),
                    df_hapdaf_s.lazy(),
                ],
            )
            .sort("positions")
            .collect()
        )
        d_stats_to_norm = {"snps": df_stats_norm, "windows": df_window_new}


        for k, df in d_stats.items():
            if k == 'window':
                continue
            d_stats[k] = df.with_columns([pl.lit(nchr).cast(pl.Utf8).alias("iter")])

        return d_stats, d_stats_to_norm


Once you have created the functions to work with discoal simulations and VCF, you can easily pass to the current API

.. code-block:: python

    df = fs.summary_statistics(data_dir,nthreads=24,center=[6e5],windows=[10000, 20000, 50000, 100000, 200000], func = calculate_stats_diploshic)

    df_vcf = fs.summary_statistics(vcf_dir,nthreads=24,center=[6e5],windows=[10000, 20000, 50000, 100000, 200000], func = calculate_stats_diploshic_vcf)


Note that ``func`` argument is ``False`` by default. When input the custom function you will estimate and normalize only the selected statistics avoiding the original Flex-sweep implementation.
=======

    # A subset of window and SNP statistics
    df = fs.summary_statistics(
        "./simulations",
        stats=["pi", "h12", "ihs", "nsl"],
        nthreads=8,
    )

The same ``stats`` argument applies to VCF data:

.. code-block:: python

    df_vcf = fs.summary_statistics(
        "./vcf_data",
        vcf=True,
        stats=["pi", "h12", "ihs", "nsl"],
        recombination_map="recomb_map.csv",
        nthreads=8,
    )

Passing an invalid stat name raises a ``ValueError`` listing all valid names.


diploS/HIC-like feature vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To estimate the same statistics as `diploS/HIC
<https://doi.org/10.1534/g3.118.200262>`_ — :math:`\pi`, :math:`\theta_W`,
:math:`\theta_H`, Fay & Wu's :math:`H`, Tajima's :math:`D`, distinct
haplotypes, :math:`H_1`, :math:`H_{12}`, :math:`H_2/H_1`, :math:`Z_{nS}`,
:math:`\omega_{max}`, maximum derived allele frequency, and pairwise distance
moments — use the parameters below. The locus is divided into contiguous
sub-windows matching the diploS/HIC subwindow approach.

From simulations:

.. code-block:: python

    import flexsweep as fs

    diploshic_stats = [
        "pi", "fay_wu_h", "theta_h", "max_fda", "theta_w", "tajima_d",
        "k_counts", "h1", "h12", "h2_h1", "zns", "omega_max",
        "dist_var", "dist_skew", "dist_kurtosis",
    ]

    df = fs.summary_statistics(
        "./simulations",
        stats=diploshic_stats,
        locus_length=100000,
        step=10000,
        windows=[10000],
        nthreads=8,
    )

From VCF data:

.. code-block:: python

    df_vcf = fs.summary_statistics(
        "./vcf_data",
        vcf=True,
        stats=diploshic_stats,
        locus_length=100000,
        step=10000,
        step_vcf=100000,
        windows=[10000],
        nthreads=8,
    )


Locus and window settings
~~~~~~~~~~~~~~~~~~~~~~~~~

Three arguments control the locus geometry. They must be consistent between
the simulation and VCF runs so that normalisation bins match.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Default
     - Description
   * - ``locus_length``
     - ``1200000``
     - Total locus length in base pairs. Together with ``step``, determines
       the centre grid: one centre every ``step`` bp from ``step//2`` to
       ``locus_length − step//2``.
   * - ``windows``
     - ``[100000]``
     - List of window sizes (bp). Multiple sizes produce a multi-scale
       feature vector. Each centre is evaluated at every window size, giving
       ``n_centres × len(windows)`` rows per replicate.
   * - ``step``
     - ``100000``
     - Sliding-window step size (bp) for simulations and the within-locus
       centre grid for VCF.
   * - ``step_vcf``
     - ``10000``
     - Step size (bp) for tiling VCF contigs into genomic windows. Independent
       of ``step``.

.. code-block:: python

    # Multi-scale windows over the default 1.2 Mb locus
    df = fs.summary_statistics(
        "./simulations",
        windows=[50000, 100000, 500000],
        nthreads=8,
    )

    # Shorter locus with a finer step
    df = fs.summary_statistics(
        "./simulations",
        locus_length=600000,
        step=50000,
        windows=[50000, 100000],
        nthreads=8,
    )


Recombination-rate stratified normalisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, SNP statistics are normalised within frequency bins only. When
``r_bins`` is provided, normalisation is additionally stratified by
recombination rate — a separate mean and standard deviation is computed for
each recombination-rate stratum defined by the breakpoints.

``r_bins`` accepts a list of breakpoints in cM/Mb (e.g. ``[1, 5, 10]``
creates four strata: <1, 1–5, 5–10, >10 cM/Mb). A recombination map must
also be supplied. The ``r_bins`` column is dropped from the output feature
matrix so the feature dimension stays unchanged regardless of whether
stratification is used.

.. code-block:: python

    df = fs.summary_statistics(
        "./simulations",
        recombination_map="recomb_map.csv",
        r_bins=[1, 5, 10],
        min_rate=0.0,
        nthreads=8,
    )

.. note::

   ``r_bins`` stratification is not recommended for domain-adaptive (DANN)
   training — it has been shown to increase domain shift in that context.
   Use it only with the standard CNN.
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)


Custom CNN
----------

Flex-sweep is now able to work with custom CNN architectures. The API includes a ``CNN`` class able to pre-process the feature vectors while being ready to use with custom CNN implementations. By default ``CNN`` class will work with the default Flex-sweep architecture. Nonetheless, we changed the old 2D CNN behaviour so we now input statistics as channels into the 2D CNN: ``(batch, windows, centers, stats)``. If you are planning to use custom CNN architectures, please be extremely careful, you must pay attention to feature vector reshaping as needed.

.. code-block:: python

    import flexsweep as fs
    from flexsweep.cnn import *

    def cnn_finer(model_input, num_classes=1):
        """
        Changing filter and kernels sizes to look for finer summary statistics dimensions.
        Includes bach normalization, global pooling (no flatten), and dropout.
        """

        initializer = tf.keras.initializers.HeNormal()

        x = tf.keras.layers.Conv2D(
            64, (2, 1), padding="same", kernel_initializer=initializer, name="conv_2x1"
        )(model_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            64, (1, 2), padding="same", kernel_initializer=initializer, name="conv_1x2"
<<<<<<< HEAD
        )(x)
=======
        )(model_input)
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            128, (2, 2), padding="same", kernel_initializer=initializer, name="conv_2x2"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # Optional deeper conv layer
        x = tf.keras.layers.Conv2D(
            128, (1, 1), padding="same", kernel_initializer=initializer, name="conv_1x1"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

<<<<<<< HEAD
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name="gentle_pool")(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_1")(x)

        # GlobalAveragePooling2D
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
=======
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name="gentle_pool")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), name="gentle_pool")(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_1")(x)

        # GlobalAveragePooling2D
        x = tf.keras.layers.Flatten()(x)
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
        # x = self.attention_pool_2d(x, name="attn_pool")

        x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)

        x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
        x = tf.keras.layers.Dropout(0.1, name="dropout_3")(x)

        output = tf.keras.layers.Dense(
            num_classes, activation="sigmoid", name="output"
        )(x)

        return output

    fs_cnn = fs.CNN(
<<<<<<< HEAD
            train_data = train_data,
            predict_data = predict_data,
            output_folder = output_folder,
    )
    fs_cnn.train(cnn = cnn_flexsweep_simplify)
=======
            train_data = "yri_test/fvs.parquet",
            predict_data = "yri_vcf/fvs_yri.parquet",
            output_folder = "yri_vcf",
    )
    fs_cnn.train(cnn = cnn_finer)
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)


Because we're providing new custom feature vectors (e.g, new genomic center and window size ranges), we're also providing an interface to train and predict using a 1D CNN. **When using 1D CNN, you must input your own CNN architecture**. We're providing a 1D CNN example with channel attention (Squeeze-and-Excitation) that learns local patterns across genomic positions and reweights features before classification. You can easily provide your own CNN similar to the example above:

.. code-block:: python

    def cnn_flexsweep_conv1d(model_input, num_classes=1):
        """
        Conv1D over spatial positions (steps) with stats as channels,
        followed by channel-wise (per-stat) attention.
        Expects model_input shape: (batch, positions=105, stats=11)
        """

        x = model_input

        # Conv1D over positions (channels_last): output (batch, positions, filters)
        x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv1D(256, 2, padding="same", activation="relu")(x)

        # Channel attention (Squeeze-and-Excitation)
        se = tf.keras.layers.GlobalAveragePooling1D()(x)
        se = tf.keras.layers.Dense(256, activation="sigmoid")(se)
        se = tf.keras.layers.Reshape((1, 256))(se)
        x  = tf.keras.layers.Multiply()([x, se])

        # Head
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        output = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

        return output

    fs_cnn_1d = CNN(
<<<<<<< HEAD
            train_data = train_data,
            predict_data = predict_data,
            output_folder = output_folder,
=======
            train_data = "yri_test/fvs.parquet",
            predict_data = "yri_vcf/fvs_yri.parquet",
            output_folder = "yri_vcf",
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
    )
    fs_cnn_1d.train(cnn = cnn_flexsweep_conv1d, one_dim = True)


<<<<<<< HEAD
=======
Haplotype sorting
~~~~~~~~~~~~~~~~~

Before feeding a raw haplotype matrix into a custom CNN you may want to
rearrange rows (haplotypes) or columns (SNPs) so that similar haplotypes are
placed adjacently — this can improve the spatial patterns a 2D CNN learns.
All functions below accept a binary ``(samples × sites)`` NumPy array.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - ``daf_sorting(matrix)``
     - Sort **columns** (SNPs) by descending derived allele frequency. Most
       common derived alleles appear first.
   * - ``freq_sorting(matrix)``
     - Sort **rows** (haplotypes) by descending number of derived alleles
       (Hamming weight). Most derived haplotypes appear first.
   * - ``corr_sorting(matrix)``
     - Sort **rows** by Pearson correlation coefficient with the most
       correlated haplotype. Groups similar haplotypes together.
   * - ``pcc_column_sort_numba(matrix)``
     - Sort **columns** by total PCC score — SNPs most correlated with the
       rest of the matrix appear first.
   * - ``haplotype_freq_sorting(matrix)``
     - Sort **columns** by haplotype frequency (most common haplotype group
       first). Returns ``(reordered, col_order, groups, freqs)``.
   * - ``haplotype_freq_sorting_hamming(matrix)``
     - Same as above but within each frequency group, haplotypes are
       additionally ordered by Hamming distance to the most frequent
       haplotype. Returns ``(reordered, freqs)``.

``daf_sorting``, ``freq_sorting``, ``corr_sorting``, and
``pcc_column_sort_numba`` are Numba-compiled and operate on integer arrays.
The two ``haplotype_freq_sorting`` variants operate on general NumPy arrays.

.. code-block:: python

    import numpy as np
    from flexsweep.utils import (
        daf_sorting, freq_sorting, haplotype_freq_sorting
    )

    # hap: binary (n_haplotypes × n_sites) array from your VCF window
    hap = np.random.randint(0, 2, (200, 500), dtype=np.int32)

    # Sort SNPs by DAF, then haplotypes by frequency
    hap_daf = daf_sorting(hap.copy())
    hap_sorted, col_order, groups, freqs = haplotype_freq_sorting(hap_daf)

    # hap_sorted is now ready for a (batch, haplotypes, sites, 1) CNN input


>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
Demography mis-specification
----------------------------

Flex-sweep is now more versatile to analyse non-model organisms where the quality or availability of simulated parameters, such as the mutation rate, recombination rate, and demography, is limited. We extend the CNN with the Domain Adaptive model proposed by `Mo, Z. and Siepel A. 2023 <https://doi.org/10.1371/journal.pgen.1011032>`_. If you plan to use Flex-sweep DA, please cite `Mo, Z. and Siepel A. 2023 <https://doi.org/10.1371/journal.pgen.1011032>`_. We highly recommend to read deep the paper along with the code `source code <https://github.com/ziyimo/popgen-dom-adapt>`_ provided by the authors.

<<<<<<< HEAD
Flex-sweep-DA trainining takes into account not only using labelled simulated data (source domain) as expected for a CNN, but also incorporates empirical unlabeled data (target domain) during the training. The goal then is to generalise the classification task across any feature distorting simulated feature vectors distribution from real data by learning a shared representation that is highly predictive for the CNN classifier but uninformative about the domain (whether the input is simulated or real).

Demography is known to highly shift summary statistics toward values that can mimic sweep signals, even assuming strict neutrality. It's been a matter of debate how realistic demographic, along with BGS, could explain most sweep signals. In the case of ML approaches like CNN, trained models under unrealistic demographies easily confound sweep prediction due to the overfitting of demographic artefacts. The DA model implemented is explicitly designed to account for and mitigate such a mismatch between simulated and real data. Note that when working with extremely out-of-range demographies (e.g, training over constant population sizes) or simulated parameters, DA implementation may still perform worse than the original CNN, so to work safer, the simulations should span plausible demographic scenarios.

Flex-sweep DA will subset the exact same number of ``source_data`` (labelled simulations) from ``target_data`` (empirical data) to balance the discriminator during training. Once trained the model is trained, the software will use the entire ``target_data`` dataset to make the predictions.
=======
Flex-sweep-DA trainining takes into account not only labelled simulated data (source domain) as expected for a CNN, but also incorporates empirical unlabelled data (target domain) during the training. The goal then is to generalise the classification task across any feature distorting simulated feature vectors distribution from real data by learning a shared representation that is highly predictive for the CNN classifier but uninformative about the domain (whether the input is simulated or real).

Demography is known to highly shift summary statistics toward values that can mimic sweep signals, even assuming strict neutrality. It's been a matter of debate how realistic demographic, along with BGS, could explain most sweep signals. In the case of ML approaches like CNN, trained models under unrealistic demographies easily confound sweep prediction due to the overfitting of demographic artifacts. The DA model implemented is explicitly designed to account for and mitigate such a mismatch between simulated and real data. Note that when working with extremely out-of-range demographies (e.g, training over constant population sizes) or simulated parameters, DA implementation may still perform worse than the original CNN, so to work safer, the simulations should span plausible demographic scenarios.

Flex-sweep DA will subset the exact same number of ``source_data`` (labelled simulations) from ``target_data`` (empirical data) to balance the discriminator during training. Once the model is trained, the software will use the entire ``target_data`` dataset to make the predictions.
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

.. code-block:: python

    import flexsweep as fs

<<<<<<< HEAD
    simulator = fs.Simulator(216, fs.DEMES_EXAMPLES['yri'], 'yri_test', num_simulations = int(2e4), nthreads = 24)

    # Prior parameters to simulate
    df_params = simulator.create_params()

    # Simulate
    sims_list = simulator.simulate()

    # Estimate fvs to train the CNN
    fvs_sims = fs.summary_statistics(data_dir = "yri_test", nthreads = 24)

    # Estimate fvs to predict
    fvs_vcf = fs.summary_statistics(data_dir = "yri_vcfs", vcf = True, nthreads = 24, recombination_map = fs.DECODE_MAP, population = 'yri')


=======
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
    fs_cnn = fs.CNN(
        source_data="yri_test/fvs.parquet",
        target_data="yri_vcfs/fvs_yri.parquet",
        output_folder="yri_vcfs",
    )

    fs_cnn.train_da()
    df_prediction = fs_cnn.predict_da()
<<<<<<< HEAD

=======
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
