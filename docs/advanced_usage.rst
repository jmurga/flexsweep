Advanced usage
==============

Custom summary statistics
-------------------------
You can extend Flex-sweep to any other summary statistic provided within the software or your own. For this purpose, you only need to mimic the following example function to estimate these stats, taking advantage of the current API. Note that the example functions follow the same logic as ``fs.fv.calculate_stats_simplify_custom`` to mimic the expected outputs to normalise and create the Flex-sweep feature vectors.

While the original Flex-sweep implementation not only avoids the windowed statistics employed (HAF and H12) but also repeats values across the selected windows/centre combination, the function examples below, based on ``fs.fv.calculate_stats_simplify_custom``, do. You must pay attention to how it creates dictionaries containing raw statistics data and a dictionary to normalise SNP-based and window-based statistics independently. SNP-based statistics are normalised following `Voight et al 2008 <https://doi.org/10.1371/journal.pbio.0040072>`_. For window-based statistics, we estimated the Z-score for each window and centre combination. E.g :math:`\pi` normalisation across window/centre:


.. math::

   Z\text{-score }\pi_i^{(w_1,c_1)}
   \;=\;
   \frac{\pi_i^{(w_1,c_1)} - \mu^{(w_1,c_1)}}{\sigma^{(w_1,c_1)}},
   \qquad \\
   \text{where }\mu^{(w_1,c_1)}=\operatorname{mean}_{i=1}^{n}\!\bigl(\pi_i^{(w_1,c_1)}\bigr)
   \text{ and }\sigma^{(w_1,c_1)}=\operatorname{sd}_{i=1}^{n}\!\bigl(\pi_i^{(w_1,c_1)}\bigr).


**It is crucial to mimic the outputs of the example functions, otherwise the API could not normalise or process the feature vector properly.**


First of all, import all the feature vector module content to make everything easier.

.. code-block:: python

    import flexsweep as fs
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
        )(x)
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

        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name="gentle_pool")(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_1")(x)

        # GlobalAveragePooling2D
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
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
            train_data = train_data,
            predict_data = predict_data,
            output_folder = output_folder,
    )
    fs_cnn.train(cnn = cnn_flexsweep_simplify)


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
            train_data = train_data,
            predict_data = predict_data,
            output_folder = output_folder,
    )
    fs_cnn_1d.train(cnn = cnn_flexsweep_conv1d, one_dim = True)


Demography mis-specification
----------------------------

Flex-sweep is now more versatile to analyse non-model organisms where the quality or availability of simulated parameters, such as the mutation rate, recombination rate, and demography, is limited. We extend the CNN with the Domain Adaptive model proposed by `Mo, Z. and Siepel A. 2023 <https://doi.org/10.1371/journal.pgen.1011032>`_. If you plan to use Flex-sweep DA, please cite `Mo, Z. and Siepel A. 2023 <https://doi.org/10.1371/journal.pgen.1011032>`_. We highly recommend to read deep the paper along with the code `source code <https://github.com/ziyimo/popgen-dom-adapt>`_ provided by the authors.

Flex-sweep-DA trainining takes into account not only using labelled simulated data (source domain) as expected for a CNN, but also incorporates empirical unlabeled data (target domain) during the training. The goal then is to generalise the classification task across any feature distorting simulated feature vectors distribution from real data by learning a shared representation that is highly predictive for the CNN classifier but uninformative about the domain (whether the input is simulated or real).

Demography is known to highly shift summary statistics toward values that can mimic sweep signals, even assuming strict neutrality. It's been a matter of debate how realistic demographic, along with BGS, could explain most sweep signals. In the case of ML approaches like CNN, trained models under unrealistic demographies easily confound sweep prediction due to the overfitting of demographic artefacts. The DA model implemented is explicitly designed to account for and mitigate such a mismatch between simulated and real data. Note that when working with extremely out-of-range demographies (e.g, training over constant population sizes) or simulated parameters, DA implementation may still perform worse than the original CNN, so to work safer, the simulations should span plausible demographic scenarios.

Flex-sweep DA will subset the exact same number of ``source_data`` (labelled simulations) from ``target_data`` (empirical data) to balance the discriminator during training. Once trained the model is trained, the software will use the entire ``target_data`` dataset to make the predictions.

.. code-block:: python

    import flexsweep as fs

    simulator = fs.Simulator(216, fs.DEMES_EXAMPLES['yri'], 'yri_test', num_simulations = int(2e4), nthreads = 24)

    # Prior parameters to simulate
    df_params = simulator.create_params()

    # Simulate
    sims_list = simulator.simulate()

    # Estimate fvs to train the CNN
    fvs_sims = fs.summary_statistics(data_dir = "yri_test", nthreads = 24)

    # Estimate fvs to predict
    fvs_vcf = fs.summary_statistics(data_dir = "yri_vcfs", vcf = True, nthreads = 24, recombination_map = fs.DECODE_MAP, population = 'yri')


    fs_cnn = fs.CNN(
        source_data="yri_test/fvs.parquet",
        target_data="yri_vcfs/fvs_yri.parquet",
        output_folder="yri_vcfs",
    )

    fs_cnn.train_da()
    df_prediction = fs_cnn.predict_da()

