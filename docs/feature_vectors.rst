.. _feature-vectors:

Feature vectors
===============

The CNN does not see haplotypes: it sees a **feature vector**, an image of summary
statistics computed around the center of each 1.2 Mb region. Each row of the image is a
statistic and each column a (window size, center) position. With 100 kb windows
centered every 100 kb (the defaults), each statistic contributes 12 columns, so the
11 statistics of the default CNN give an 11 x 12 image. Flex-sweep
computes the same feature vectors for the simulations, which carry labels, and for the
VCFs, which the trained model classifies.

Before training, every statistic is standardized (z-scored), so that statistics on
different scales are comparable:

* per-SNP statistics (iHS, nSL, DIND, ...) within derived allele frequency bins
  (`Voight et al. 2006 <https://doi.org/10.1371/journal.pbio.0040072>`_), and
* window statistics within each (window size, center) combination:

.. math::

   z_i^{(w,c)} = \frac{x_i^{(w,c)} - \mu^{(w,c)}}{\sigma^{(w,c)}}

For simulations, :math:`\mu` and :math:`\sigma` come from the **neutral** replicates
and are applied to neutral and sweep replicates alike. For VCFs they come from the
empirical windows themselves. Each dataset is therefore centered on its own
distribution, which removes part of the difference between simulated and real data.
Optionally, both are also stratified by local recombination rate
(:ref:`recombination-normalisation`).

Command line and Python interface
---------------------------------

From simulations
~~~~~~~~~~~~~~~~

.. code-block:: bash

    flexsweep fvs-discoal --simulations_path yri_sims --nthreads 24

.. code-block:: python

    import flexsweep as fs

    df_sims = fs.summary_statistics("yri_sims", nthreads=24)

The input is a folder written by ``flexsweep simulator`` (:doc:`simulation`). Output,
in that folder:

* ``fvs.parquet``: z-scored feature vectors with the simulation parameters (``model``,
  ``s``, ``t``, ``f_i``, ``f_t``, ``mu``, ``r``); the CNN training input;
* ``fvs_raw.parquet``: the same before standardization;
* ``neutral_bins.pickle``: neutral means and standard deviations.

``--suffix`` (``suffix=``) adds ``_<suffix>`` to these names.

From VCFs
~~~~~~~~~

.. code-block:: bash

    flexsweep fvs-vcf \
        --vcf_path yri_vcfs \
        --recombination_map decode \
        --nthreads 24 \
        --suffix yri

.. code-block:: python

    df_vcf = fs.summary_statistics(
        "yri_vcfs",
        vcf=True,
        recombination_map=fs.DECODE_MAP,
        nthreads=24,
        suffix="yri",
    )

The input is a folder of per-chromosome VCFs (:ref:`vcf-requirements`). Each
chromosome is tiled with regions of ``locus_length`` (1.2 Mb) sliding by ``step_vcf``
(10 kb), and every region gets a feature vector centered on its midpoint, as in the
simulations. The recombination map gives the genetic positions used by some statistics
and the local rate used by the optional stratification; ``decode`` is the bundled
deCODE map (GRCh38). Output, in the VCF folder:

* ``fvs_yri.parquet``: z-scored feature vectors, one row per region (``iter`` holds
  ``chr:start-end``); the CNN prediction input;
* ``fvs_raw_yri.parquet`` and ``empirical_bins_yri.pickle``.

.. note::

   **Temporary disk space.** Workers receive the haplotype arrays through
   joblib, which writes large arrays to a temporary folder. By default
   (``--tmp_dir`` unset) Flex-sweep keeps joblib's own choice: the folder in
   ``JOBLIB_TEMP_FOLDER`` if set, otherwise ``/dev/shm``, otherwise the system
   temporary folder (``TMPDIR``, typically ``/tmp``). The folder is emptied after
   each chromosome. On many-chromosome runs, or on servers with a small ``/tmp``,
   this can still fail with ``OSError: [Errno 28] No space left on device``. Point
   the temporary files to a larger disk with either option:

   .. code-block:: bash

       flexsweep fvs-vcf --vcf_path yri_vcfs --suffix yri --tmp_dir /path/to/large/disk/tmp
       # or, for any flexsweep command
       export JOBLIB_TEMP_FOLDER=/path/to/large/disk/tmp

   ``--tmp_dir`` (``tmp_dir=`` in Python) takes precedence over ``JOBLIB_TEMP_FOLDER``.

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 16 14 46

   * - CLI option
     - Python argument
     - Default
     - Meaning
   * - ``--simulations_path`` / ``--vcf_path``
     - ``data_dir`` (``vcf=True`` for VCFs)
     -
     - Input folder.
   * - ``--stats``
     - ``stats``
     - Flex-sweep set
     - Statistics to compute (:ref:`fv-statistics`).
   * - ``--windows``
     - ``windows``
     - ``100000``
     - Window size(s) in bp.
   * - ``--step``
     - ``step``
     - ``100000``
     - Spacing of the centers within a region.
   * - ``--centers``
     - ``centers``
     - grid from ``step``
     - Explicit center positions (simulations only on the CLI).
   * - ``--locus_length``
     - ``locus_length``
     - ``1200000``
     - Region length.
   * - ``--step_vcf``
     - ``step_vcf``
     - ``10000``
     - VCF only: distance between consecutive regions.
   * - ``--recombination_map``
     - ``recombination_map``
     - none
     - VCF only: map file or ``decode`` (CLI) / ``fs.DECODE_MAP`` (Python).
   * - ``--r_bins``
     - ``r_bins``
     - none
     - Recombination-rate strata (:ref:`recombination-normalisation`).
   * - ``--min_rate``
     - ``min_rate``
     - ``0.01`` (CLI), ``0.0`` (Python)
     - VCF only: with ``r_bins``, drop regions below this rate (cM/Mb).
   * - ``--suffix``
     - ``suffix``
     - none
     - Output name suffix; required by ``fvs-vcf``.
   * - ``--nthreads``
     - ``nthreads``
     - ``1``
     - Parallel workers.
   * - ``--save_stats``
     - ``save_stats``
     - ``False``
     - Also save the raw per-replicate statistics.
   * - ``--tmp_dir``
     - ``tmp_dir``
     - joblib default
     - VCF only: folder for joblib's temporary array files, cleaned after each VCF.
       Default order: ``JOBLIB_TEMP_FOLDER``, then ``/dev/shm``, then ``/tmp``. Point it
       (or ``JOBLIB_TEMP_FOLDER``) to a large disk when ``/tmp`` is small.
   * - ``--accessibility_mask``
     - ``accessibility_mask``
     - none
     - VCF only: BED of regions to exclude, GIAB convention (e.g.
       ``GRCh38_alldifficultregions.bed.gz``), all chromosomes. SNPs inside it are
       dropped, and ``pi``, ``theta_w`` and ``dist_var`` are divided by the accessible
       bases of each window instead of its length. See :ref:`accessibility-mask`.
   * - ``--min_accessible``
     - ``min_accessible``
     - 0.6
     - VCF only, with ``--accessibility_mask``: windows with a smaller accessible
       fraction get missing window statistics (handled like any missing feature).

.. _fv-statistics:

Available statistics
--------------------

Statistics fall into two families. **Window-based** statistics are computed for
every centre × window-size combination and summarise a genomic interval.
**SNP-based** statistics are computed per variant and are subsequently
aggregated over the same centre × window grid. Every name below is a valid
entry of the ``stats`` argument; passing an unknown name raises a ``ValueError``
listing all valid names.

**Window-based statistics**

.. list-table::
   :header-rows: 1
   :widths: 16 40 20 24

   * - Name
     - Description
     - Summarises
     - Reference
   * - ``pi``
     - Average number of pairwise nucleotide differences, per base pair
     - Nucleotide diversity
     - `Tajima 1983 <https://doi.org/10.1093/genetics/105.2.437>`_
   * - ``theta_w``
     - Watterson's :math:`\theta_W` based on segregating sites, per base pair
     - Nucleotide diversity
     - `Watterson 1975 <https://doi.org/10.1016/0040-5809(75)90020-9>`_
   * - ``tajima_d``
     - Tajima's :math:`D`; difference between :math:`\theta_\pi` and
       :math:`\theta_W`
     - SFS
     - `Tajima 1989 <https://doi.org/10.1093/genetics/123.3.585>`_
   * - ``theta_h``
     - Fay & Wu's :math:`\theta_H`; excess of high-frequency derived alleles
     - SFS
     - `Fay & Wu 2000 <https://doi.org/10.1093/genetics/155.3.1405>`_
   * - ``fay_wu_h``
     - Normalised Fay & Wu's :math:`H`
     - SFS
     - `Fay & Wu 2000 <https://doi.org/10.1093/genetics/155.3.1405>`_; `Zeng et al. 2006 <https://doi.org/10.1534/genetics.106.061432>`_
   * - ``zeng_e``
     - Zeng's :math:`E`; contrast between :math:`\theta_\pi` and Fay & Wu's
       :math:`H`
     - SFS
     - `Zeng et al. 2006 <https://doi.org/10.1534/genetics.106.061432>`_
   * - ``achaz_y``
     - Achaz's :math:`Y`; Tajima's :math:`D` analogue excluding derived
       singletons, robust to sequencing errors (polarized, requires outgroup)
     - SFS
     - `Achaz 2008 <https://doi.org/10.1534/genetics.107.082198>`_
   * - ``achaz_y_star``
     - Achaz's :math:`Y^*`; folded analogue excluding minor-allele singletons
     - SFS
     - `Achaz 2008 <https://doi.org/10.1534/genetics.107.082198>`_
   * - ``achaz_t``
     - Achaz's :math:`T_\Omega`; generalised neutrality test defined as the
       normalized difference between any two frequency-spectrum-based
       :math:`\hat{\theta}` estimators
     - SFS
     - `Achaz 2009 <https://doi.org/10.1534/genetics.109.104042>`_
   * - ``fuli_d``
     - Fu & Li's :math:`D`; excess of singletons relative to total mutations
       (polarized, requires outgroup)
     - SFS
     - `Fu & Li 1993 <https://doi.org/10.1093/genetics/133.3.693>`_
   * - ``fuli_d_star``
     - Fu & Li's :math:`D^*` (folded)
     - SFS
     - `Fu & Li 1993 <https://doi.org/10.1093/genetics/133.3.693>`_
   * - ``fuli_f``
     - Fu & Li's :math:`F`; excess of singletons relative to average pairwise
       differences (polarized, requires outgroup)
     - SFS
     - `Fu & Li 1993 <https://doi.org/10.1093/genetics/133.3.693>`_
   * - ``fuli_f_star``
     - Fu & Li's :math:`F^*` (folded)
     - SFS
     - `Fu & Li 1993 <https://doi.org/10.1093/genetics/133.3.693>`_
   * - ``max_fda``
     - :math:`\mathrm{DAF}_{\max}`; maximum derived allele frequency in a
       subwindow
     - SFS
     - `Kern & Schrider 2018 <https://doi.org/10.1534/g3.118.200262>`_
   * - ``k_counts``
     - Number of distinct haplotypes
     - Haplotype structure
     - `Kern & Schrider 2018 <https://doi.org/10.1534/g3.118.200262>`_
   * - ``h1``
     - Garud's :math:`H_1`; haplotype homozygosity
     - Haplotype structure
     - `Garud et al. 2015 <https://doi.org/10.1371/journal.pgen.1005004>`_
   * - ``h12``
     - Garud's :math:`H_{12}`; frequencies of the first and second most common
       haplotypes, modified to use an 80% identity threshold
     - Haplotype structure
     - `Garud et al. 2015 <https://doi.org/10.1371/journal.pgen.1005004>`_; `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``h2_h1``
     - Garud's :math:`H_2/H_1`
     - Haplotype structure
     - `Garud et al. 2015 <https://doi.org/10.1371/journal.pgen.1005004>`_
   * - ``haf``
     - Haplotype allele frequency, reporting only the summed upper 10% of the
       sorted scores (HAF-top)
     - Haplotype structure
     - `Ronen et al. 2015 <https://doi.org/10.1371/journal.pgen.1005527>`_; `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``dist_var``
     - :math:`\mathrm{Var}(d_{ij})`; variance of the distribution of pairwise
       haplotype mismatch distances within a subwindow
     - Haplotype structure
     - `Kern & Schrider 2018 <https://doi.org/10.1534/g3.118.200262>`_
   * - ``dist_skew``
     - :math:`\mathrm{Skew}(d_{ij})`; skewness of the same distribution
     - Haplotype structure
     - `Kern & Schrider 2018 <https://doi.org/10.1534/g3.118.200262>`_
   * - ``dist_kurtosis``
     - :math:`\mathrm{Kurt}(d_{ij})`; excess kurtosis of the same distribution
     - Haplotype structure
     - `Kern & Schrider 2018 <https://doi.org/10.1534/g3.118.200262>`_
   * - ``zns``
     - Kelly's :math:`Z_{nS}`; average linkage disequilibrium (:math:`r^2`)
       between segregating sites
     - Linkage disequilibrium
     - `Kelly 1997 <https://doi.org/10.1093/genetics/146.3.1197>`_
   * - ``omega_max``
     - :math:`\omega_{\max}`; maximum LD between selected and flanking regions
     - Linkage disequilibrium
     - `Kim & Nielsen 2004 <https://doi.org/10.1534/genetics.103.025387>`_
   * - ``ncd1``
     - NCD1; non-central deviation of the SFS from neutrality
     - SFS (balancing selection)
     - `Bitarello et al. 2018 <https://doi.org/10.1093/gbe/evy054>`_

**SNP-based statistics**

.. list-table::
   :header-rows: 1
   :widths: 16 40 20 24

   * - Name
     - Description
     - Summarises
     - Reference
   * - ``ihs``
     - Integrated haplotype score
     - Haplotype structure
     - `Voight et al. 2006 <https://doi.org/10.1371/journal.pbio.0040072>`_
   * - ``delta_ihh``
     - :math:`\Delta\text{-}iHH`; absolute iHH difference between ancestral and
       derived alleles
     - Haplotype structure
     - `Grossman et al. 2010 <https://doi.org/10.1126/science.1183863>`_
   * - ``nsl``
     - :math:`nS_L`; number of segregating sites by length
     - Haplotype structure
     - `Ferrer-Admetlla et al. 2014 <https://doi.org/10.1093/molbev/msu077>`_
   * - ``isafe``
     - Integrated selection of allele favoured by evolution
     - Haplotype structure
     - `Akbari et al. 2018 <https://doi.org/10.1038/nmeth.4606>`_
   * - ``hscan``
     - Average pairwise haplotype homozygosity tract length
     - Haplotype structure
     - `Schlamp et al. 2016 <https://doi.org/10.1111/mec.13485>`_
   * - ``dind`` / ``dind_high_low``
     - Derived intra-allelic nucleotide diversity
     - Diversity on derived background
     - `Barreiro et al. 2009 <https://doi.org/10.1371/journal.pgen.1000562>`_
   * - ``s_ratio``
     - Segregating sites ratio
     - Diversity on derived background
     - `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``low_freq`` / ``lowfreq``
     - Low-frequency alleles on derived background
     - Diversity on derived background
     - `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``high_freq`` / ``highfreq``
     - High-frequency alleles on derived background
     - Diversity on derived background
     - `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``hapdaf_o``
     - Haplotype-derived allele frequency (old)
     - SFS
     - `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``hapdaf_s``
     - Haplotype-derived allele frequency (standing)
     - SFS
     - `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``beta``
     - :math:`\beta^{(1)}`; correlation of allele frequency with local
       polymorphism
     - SFS (balancing selection)
     - `Siewert & Voight 2020 <https://doi.org/10.1093/gbe/evaa013>`_
   * - ``beta_t``
     - :math:`\beta^{(1)*}_{(std)}`; standardised form of the above
     - SFS (balancing selection)
     - `Siewert & Voight 2020 <https://doi.org/10.1093/gbe/evaa013>`_

.. note::

   ``ihs`` and ``nsl`` are computed with `scikit-allel
   <https://scikit-allel.readthedocs.io/>`_; every other statistic is a Numba
   or NumPy vectorised implementation. ``delta_ihh`` is computed together with
   ``ihs`` and is present in the output, but it is excluded from the default
   CNN feature set.

Statistics available through other modules
-------------------------------------------

The composite sweep statistics below are not selectable through the ``stats``
argument of the feature-vector pipeline. They are available through the
standalone outlier scan (see :doc:`scan`) and can be called directly from the
Python API (see :doc:`api` for full signatures).

.. list-table::
   :header-rows: 1
   :widths: 16 24 36 24

   * - Statistic
     - ``scan`` key / function
     - Description
     - Reference
   * - LASSI :math:`T`, :math:`\hat{m}`
     - ``lassi`` / ``compute_t_m``
     - Likelihood-based detection of selective sweeps using haplotype structure
     - `Harris & DeGiorgio 2020 <https://doi.org/10.1093/molbev/msaa115>`_
   * - RAiSD :math:`\mu`
     - ``raisd`` / ``mu_stat``
     - Composite detection of selective sweeps using SFS, LD, and diversity
     - `Alachiotis & Pavlidis 2018 <https://doi.org/10.1038/s42003-018-0085-8>`_

Selecting statistics
--------------------

``--stats`` (``stats=`` in Python) takes any combination of the statistics above. Use
the same list for the simulations and the VCFs.

.. code-block:: bash

    flexsweep fvs-discoal --simulations_path yri_sims --stats pi,h12,ihs,nsl --nthreads 8
    flexsweep fvs-vcf --vcf_path yri_vcfs --stats pi,h12,ihs,nsl --recombination_map decode \
        --nthreads 8 --suffix yri

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
        recombination_map=fs.DECODE_MAP,
        nthreads=8,
    )

Passing an invalid stat name raises a ``ValueError`` listing all valid names.

diploS/HIC-like feature vectors
--------------------------------

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
        locus_length=1100000,
        step=100000,
        windows=[100000],
        nthreads=8,
    )

From VCF data:

.. code-block:: python

    df_vcf = fs.summary_statistics(
        "./vcf_data",
        vcf=True,
        stats=diploshic_stats,
        locus_length=1100000,
        step=100000,
        step_vcf=1000000,
        windows=[100000],
        nthreads=8,
    )

Locus and window settings
-------------------------

These arguments set the windows and centers that make up the feature vector. Use the
same ``locus_length``, ``windows`` and ``step`` for the simulations and the VCFs, so
that every empirical position has a simulated counterpart.

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

.. code-block:: bash

    # Multi-scale windows over the default 1.2 Mb locus
    flexsweep fvs-discoal --simulations_path yri_sims --windows 50000,100000,500000 --nthreads 8

.. code-block:: python

    # Multi-scale windows over the default 1.2 Mb locus
    df = fs.summary_statistics(
        "./simulations",
        windows=[50000, 100000, 500000],
        nthreads=8,
    )

    # Larger locus with a finer step
    df = fs.summary_statistics(
        "./simulations",
        locus_length=2000000,
        step=50000,
        windows=[100000],
        nthreads=8,
    )

.. _recombination-normalisation:

Recombination-rate stratified normalisation
-------------------------------------------

Haplotype statistics correlate with the local recombination rate: low-recombination
regions have longer haplotypes and look more sweep-like. With ``r_bins``, Flex-sweep
standardizes each statistic within recombination-rate strata as well, so that a region
is compared with regions of similar recombination rate
(`Johnson and Voight 2018 <https://doi.org/10.1038/s41559-018-0478-6>`_). This makes
predictions more robust to differences in recombination rate between the simulations
and the data. In simulations the rate of each replicate is known (``r`` in
``params.txt.gz``); in VCFs it is interpolated from the recombination map. The strata
are dropped from the output, so the feature vector keeps the same shape.

**Recombination map format.** A tab-separated file with ``#`` comment lines, one
header line, and five columns: chromosome, start and end of the interval (bp), rate in
cM/Mb, and cumulative position in cM. This is the format of the deCODE maps
(`Halldorsson et al. 2019 <https://doi.org/10.1126/science.aau1043>`_); the shipped
``decode_sexavg_2019.txt.gz`` starts:

.. code-block:: text

    # Average genetic map computed from the paternal and maternal genetic maps.
    # ...
    Chr     Begin   End     cMperMb cM
    chr1    1431813 1515567 0.032449236802513395    0.0027177533791577072
    chr1    1515567 1530002 0.18959655991987387     0.0054545797216010855
    chr1    1530002 1534402 0.004630788906729801    0.005474955192790697

Column names are not used, but the order is. Chromosomes must be named ``chr`` plus
the VCF chromosome without its ``chr`` prefix: ``chr22`` for VCF records named ``22``
or ``chr22``. Coordinates must be on the same genome build as the VCFs (GRCh38 for the
shipped map).

**1. Compute the breakpoints** from your own VCFs and map. ``recombination-bins``
slides the regions over the VCFs, interpolates the rate of each region, drops regions
below ``--min_rate`` cM/Mb, and prints equal-frequency breakpoints (deciles with
``--bins 10``) plus the rounded-up maximum:

.. code-block:: bash

    flexsweep recombination-bins --vcf_path yri_vcfs --recombination_map decode \
        --bins 10 --min_rate 0.01
    # Recombination bins: 0.37,0.55,0.71,0.87,1.05,1.26,1.53,1.88,2.46,6.1

These are the breakpoints used for the 1000 Genomes analyses (deCODE map, 1.2 Mb
regions, ``--min_rate 0.01``); recompute them for other data or maps.

**2. Pass the same breakpoints and minimum rate to both runs**, so that simulated and
empirical statistics are standardized within matching strata:

.. code-block:: bash

    R_BINS=0.37,0.55,0.71,0.87,1.05,1.26,1.53,1.88,2.46,6.1
    flexsweep fvs-discoal --simulations_path yri_sims --r_bins "$R_BINS" --nthreads 24
    flexsweep fvs-vcf --vcf_path yri_vcfs --recombination_map decode \
        --r_bins "$R_BINS" --min_rate 0.01 --nthreads 24 --suffix yri

.. code-block:: python

    r_bins = [0.37, 0.55, 0.71, 0.87, 1.05, 1.26, 1.53, 1.88, 2.46, 6.1]
    df_sims = fs.summary_statistics("yri_sims", r_bins=r_bins, nthreads=24)
    df_vcf = fs.summary_statistics(
        "yri_vcfs", vcf=True, recombination_map=fs.DECODE_MAP,
        r_bins=r_bins, min_rate=0.01, nthreads=24, suffix="yri",
    )

``min_rate`` only filters empirical regions (VCF path); simulations are stratified by
the rate drawn for each replicate. The simulated rates must span the empirical ones:
with a fixed recombination rate (:ref:`rate-specs`) every neutral replicate falls in
one stratum.
