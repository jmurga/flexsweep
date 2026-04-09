Advanced usage
==============

Custom summary statistics
Select any combination of built-in statistics by passing a list of names to
the ``stats`` argument of ``summary_statistics``. The pipeline handles
computation, windowing, and normalisation automatically for both simulations
and VCF data.

By default (``stats=None``) the full Flex-sweep statistic set is used.

Normalisation follows two schemes. SNP-based statistics are normalised by
frequency bin following `Voight et al. 2006
<https://doi.org/10.1371/journal.pbio.0040072>`_. Window-based statistics are
Z-scored per window/centre combination:

.. math::

   Z\text{-score }\pi_i^{(w_1,c_1)}
   \;=\;
   \frac{\pi_i^{(w_1,c_1)} - \mu^{(w_1,c_1)}}{\sigma^{(w_1,c_1)}},
   \qquad \\
   \text{where }\mu^{(w_1,c_1)}=\operatorname{mean}_{i=1}^{n}\!\bigl(\pi_i^{(w_1,c_1)}\bigr)


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

.. code-block:: python

    import flexsweep as fs

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
        )(model_input)

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

        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name="gentle_pool")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), name="gentle_pool")(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_1")(x)

        # GlobalAveragePooling2D
        x = tf.keras.layers.Flatten()(x)
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
            train_data = "yri_test/fvs.parquet",
            predict_data = "yri_vcf/fvs_yri.parquet",
            output_folder = "yri_vcf",
    )
    fs_cnn.train(cnn = cnn_finer)


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
            train_data = "yri_test/fvs.parquet",
            predict_data = "yri_vcf/fvs_yri.parquet",
            output_folder = "yri_vcf",
    )
    fs_cnn_1d.train(cnn = cnn_flexsweep_conv1d, one_dim = True)


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

Demography mis-specification
----------------------------

Flex-sweep is now more versatile to analyse non-model organisms where the quality or availability of simulated parameters, such as the mutation rate, recombination rate, and demography, is limited. We extend the CNN with the Domain Adaptive model proposed by `Mo, Z. and Siepel A. 2023 <https://doi.org/10.1371/journal.pgen.1011032>`_. If you plan to use Flex-sweep DA, please cite `Mo, Z. and Siepel A. 2023 <https://doi.org/10.1371/journal.pgen.1011032>`_. We highly recommend to read deep the paper along with the code `source code <https://github.com/ziyimo/popgen-dom-adapt>`_ provided by the authors.

Flex-sweep-DA trainining takes into account not only labelled simulated data (source domain) as expected for a CNN, but also incorporates empirical unlabelled data (target domain) during the training. The goal then is to generalise the classification task across any feature distorting simulated feature vectors distribution from real data by learning a shared representation that is highly predictive for the CNN classifier but uninformative about the domain (whether the input is simulated or real).

Demography is known to highly shift summary statistics toward values that can mimic sweep signals, even assuming strict neutrality. It's been a matter of debate how realistic demographic, along with BGS, could explain most sweep signals. In the case of ML approaches like CNN, trained models under unrealistic demographies easily confound sweep prediction due to the overfitting of demographic artifacts. The DA model implemented is explicitly designed to account for and mitigate such a mismatch between simulated and real data. Note that when working with extremely out-of-range demographies (e.g, training over constant population sizes) or simulated parameters, DA implementation may still perform worse than the original CNN, so to work safer, the simulations should span plausible demographic scenarios.

Flex-sweep DA will subset the exact same number of ``source_data`` (labelled simulations) from ``target_data`` (empirical data) to balance the discriminator during training. Once the model is trained, the software will use the entire ``target_data`` dataset to make the predictions.

.. code-block:: python

    import flexsweep as fs
    fs_cnn = fs.CNN(
        source_data="yri_test/fvs.parquet",
        target_data="yri_vcfs/fvs_yri.parquet",
        output_folder="yri_vcfs",
    )

    fs_cnn.train_da()
    df_prediction = fs_cnn.predict_da()

