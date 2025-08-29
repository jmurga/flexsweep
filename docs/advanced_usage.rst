Advanced usage
==============

Custom summary statistics
-------------------------
You can extend flexsweep to any other summary statistic provided within the software or your own. For this purpose, you only need to mimic the following example function to estimate these stats, taking advantage of the current API. Note that the example functions follow the same logic as fs.fv.calculate_stats to mimic the expected outputs to normalize and create the Flex-sweep feature vectors.

Let's say we want to estimate the same statistics as `diploS/HIC <https://doi.org/10.1534/g3.118.200262>`_: :math:`\pi`, :math:`\theta_{w}`, :math:`\theta_{H}`, the number of distinct haplotypes, H1, H12 and H2/H1, :math:`Z_{nS}`, and :math:`\omega_{max}`. In addition, we will be in one centre while decreasing the window sizes. You must modify as needed both functions, the function processing simulations as well as the function processing VCF files.

**It is crucial to mimic the outputs of the example functions, otherwise the API could not normalize nor process the feature vector properly.**

Note that since we're not estimating any SNP-based statistics, we must return None in the normalization dictionary. Do the same if you are not estimating any windowed statistic. You will find bellow examples to estimate diploS/HIC statistics in simulations and VCF. You must start importing the Flex-sweep package as well as the entire feature vector module to easily access the statistics functions

.. code-block:: python

    import flexsweep as fs
    from flexsweep.fv import *


Simulated summary statisitics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
                _theta_h_v = fay_wu_h_norm_si(_ac)[0]
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



VCF summary statisitics
~~~~~~~~~~~~~~~~~~~~~~~

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
                        _theta_h_v = fay_wu_h_norm_si(_ac)[0]
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



Once you have created the functions to work with discoal simulations and VCF, you can easily pass to the current API

.. code-block:: python

    df = fs.summary_statistics(data_dir,nthreads=24,center=[6e5],windows=[10000, 20000, 50000, 100000, 200000], func = calculate_stats_diploshic)

    df_vcf = fs.summary_statistics(vcf_dir,nthreads=24,center=[6e5],windows=[10000, 20000, 50000, 100000, 200000], func = calculate_stats_diploshic_vcf)


Note that ``func`` argument is ``False`` by default. When input the custom function you will estimate and normalize only the selected statistics avoiding the original Flex-sweep implementation.


Custom CNN
----------


Demography mis-specification
----------------------------


