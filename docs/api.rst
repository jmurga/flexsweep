<<<<<<< HEAD
API
===
=======
API Reference
=============

.. contents:: Contents
   :local:
   :depth: 2

>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

Simulation
----------

<<<<<<< HEAD
.. autoclass:: flexsweep.Simulator()
=======
.. autoclass:: flexsweep.Simulator
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
    :members:
    :undoc-members:
    :show-inheritance:

<<<<<<< HEAD
Feature vectors
---------------

.. autofunction:: flexsweep.summary_statistics()
.. autofunction:: flexsweep.fv._process_sims()
.. autofunction:: flexsweep.fv._process_vcf()
.. autofunction:: flexsweep.fv.calculate_stats_simplify()
.. autofunction:: flexsweep.fv.calculate_stats_vcf()

Population genetics statisitcs
------------------------------

Site Frequency Spectrum (SFS)–based
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv.theta_watterson()
.. autofunction:: flexsweep.fv.sfs_nb()
.. autofunction:: flexsweep.fv.theta_pi()
.. autofunction:: flexsweep.fv.tajima_d()
.. autofunction:: flexsweep.fv.achaz_y()
.. autofunction:: flexsweep.fv.fay_wu_h_norm()
.. autofunction:: flexsweep.fv.zeng_e()
.. autofunction:: flexsweep.fv.fuli_f_star()
.. autofunction:: flexsweep.fv.fuli_f()
.. autofunction:: flexsweep.fv.fuli_d_star()
.. autofunction:: flexsweep.fv.fuli_d()

Derived-background diversity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv.dind_high_low()
.. autofunction:: flexsweep.fv.s_ratio()

=======

Data I/O
--------

.. autoclass:: flexsweep.data.Data
    :members:
    :undoc-members:
    :show-inheritance:


Feature vectors
---------------

.. autofunction:: flexsweep.fv_v2.summary_statistics
.. autofunction:: flexsweep.fv_v2._process_sims
.. autofunction:: flexsweep.fv_v2._process_vcf
.. autofunction:: flexsweep.fv_v2.calculate_stats_simulations
.. autofunction:: flexsweep.fv_v2.calculate_stats_vcf_flat
.. autofunction:: flexsweep.fv_v2.normalize_neutral
.. autofunction:: flexsweep.fv_v2.normalize_stats
.. autofunction:: flexsweep.fv_v2.normalize_cut_raw
.. autofunction:: flexsweep.fv_v2.resolve_stats


Outlier scan
------------

.. autofunction:: flexsweep.scan.scan
.. autofunction:: flexsweep.scan.available_stats
.. autofunction:: flexsweep.scan.stat_params
.. autofunction:: flexsweep.scan.empirical_pvalues


CNN / DANN
----------

.. autoclass:: flexsweep.cnn.CNN
    :members:
    :undoc-members:
    :show-inheritance:


Enrichment
----------

.. autofunction:: flexsweep.enrichment.run_enrichment
.. autofunction:: flexsweep.enrichment.run_bootstrap_nomatchomega
.. autofunction:: flexsweep.enrichment.estimate_fdr
.. autofunction:: flexsweep.enrichment.count_sweeps_singlepop
.. autofunction:: flexsweep.enrichment.count_sweeps_multipop


Ancestral polarization
----------------------

Flexsweep includes a Rust CLI (``flexsweep-polarize``) for annotating the
ancestral allele in a VCF from a multi-species alignment (MAF). The Python
wrappers below invoke the compiled binary; the binary must be built first with
``build_rust_polarization()`` or installed via conda-forge.

**Workflow**:

1. Sort the MAF by reference contig and position (``run_sort_maf``).
2. Polarize the VCF using the sorted MAF and one or more outgroup species
   (``run_polarize``). The binary streams both files together and writes a
   bgzipped VCF with REF/ALT swapped where the outgroup consensus supports the
   alternate allele as ancestral.

**Polarization methods** (``--method``):

- ``parsimony`` — majority vote over outgroup alleles; no model fitting.
- ``jc`` — Jukes–Cantor substitution model.
- ``kimura`` — Kimura two-parameter model (default).
- ``r6`` — General reversible model (6 free parameters); slowest, most accurate.

.. autofunction:: flexsweep.polarize.run_sort_maf
.. autofunction:: flexsweep.polarize.run_polarize
.. autofunction:: flexsweep.polarize.build_rust_polarization


Visualization
-------------

.. autofunction:: flexsweep.utils.plot_scan
.. autofunction:: flexsweep.utils.plot_manhattan
.. autofunction:: flexsweep.utils.plot_diversity
.. autofunction:: flexsweep.utils.plot_sweep_density
.. autofunction:: flexsweep.utils.plot_fv_pca
.. autofunction:: flexsweep.utils.plot_stat_distributions
.. autofunction:: flexsweep.utils.rank_probabilities


Population genetics statistics
-------------------------------

These are the low-level functions called internally by the feature-vector
pipeline and the scan module. They can also be called directly on numpy arrays.

Site Frequency Spectrum (SFS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv_v2.neutrality_stats
.. autofunction:: flexsweep.fv_v2.theta_watterson
.. autofunction:: flexsweep.fv_v2.sfs_nb
.. autofunction:: flexsweep.fv_v2.theta_pi
.. autofunction:: flexsweep.fv_v2.tajima_d
.. autofunction:: flexsweep.fv_v2.achaz_y
.. autofunction:: flexsweep.fv_v2.achaz_y_star
.. autofunction:: flexsweep.fv_v2.achaz_t
.. autofunction:: flexsweep.fv_v2.fay_wu_h_norm
.. autofunction:: flexsweep.fv_v2.zeng_e
.. autofunction:: flexsweep.fv_v2.fuli_f_star
.. autofunction:: flexsweep.fv_v2.fuli_f
.. autofunction:: flexsweep.fv_v2.fuli_d_star
.. autofunction:: flexsweep.fv_v2.fuli_d
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

Haplotype-based
~~~~~~~~~~~~~~~

<<<<<<< HEAD
.. autofunction:: flexsweep.fv.ihs_ihh()
.. autofunction:: flexsweep.fv.nsl()
.. autofunction:: flexsweep.fv.haf_top()
.. autofunction:: flexsweep.fv.run_isafe()
.. autofunction:: flexsweep.fv.garud_h_numba()
.. autofunction:: flexsweep.fv.h12_enard()
.. autofunction:: flexsweep.fv.hapdaf_o()
.. autofunction:: flexsweep.fv.hapdaf_s()


LD
~~

.. autofunction:: flexsweep.fv.Ld()
.. autofunction:: flexsweep.fv.r2()
.. autofunction:: flexsweep.fv.compute_r2_matrix_upper()
.. autofunction:: flexsweep.fv.omega_linear_correct()


Composite methods
~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv.compute_t_m()
.. autofunction:: flexsweep.fv.mu_stat()
=======
.. autofunction:: flexsweep.fv_v2.ihs_ihh
.. autofunction:: flexsweep.fv_v2.haf_top
.. autofunction:: flexsweep.fv_v2.garud_h
.. autofunction:: flexsweep.fv_v2.h12_enard
.. autofunction:: flexsweep.fv_v2.hscan
.. autofunction:: flexsweep.fv_v2.run_isafe
.. autofunction:: flexsweep.fv_v2.isafe

Derived-background diversity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv_v2.dind_high_low
.. autofunction:: flexsweep.fv_v2.s_ratio
.. autofunction:: flexsweep.fv_v2.hapdaf_o
.. autofunction:: flexsweep.fv_v2.hapdaf_s
.. autofunction:: flexsweep.fv_v2.fast_sq_freq_pairs
.. autofunction:: flexsweep.fv_v2.dind_high_low_from_pairs
.. autofunction:: flexsweep.fv_v2.s_ratio_from_pairs
.. autofunction:: flexsweep.fv_v2.hapdaf_from_pairs

Linkage disequilibrium
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv_v2.Ld
.. autofunction:: flexsweep.fv_v2.r2
.. autofunction:: flexsweep.fv_v2.compute_r2_matrix_upper
.. autofunction:: flexsweep.fv_v2.omega_linear_correct

Composite sweep statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv_v2.LASSI_spectrum_and_Kspectrum
.. autofunction:: flexsweep.fv_v2.T_m_statistic_fast
.. autofunction:: flexsweep.fv_v2.compute_t_m
.. autofunction:: flexsweep.fv_v2.Lambda_statistic_fast
.. autofunction:: flexsweep.fv_v2.run_lassip
.. autofunction:: flexsweep.fv_v2.mu_stat
.. autofunction:: flexsweep.fv_v2.run_raisd
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

Balancing selection
~~~~~~~~~~~~~~~~~~~

<<<<<<< HEAD
.. autofunction:: flexsweep.balancing.ncd1()
.. autofunction:: flexsweep.balancing.run_beta_window()


Convolutional Neural Network
----------------------------

.. autoclass:: flexsweep.CNN()
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: flexsweep.cnn.DAParquetSequence()
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: flexsweep.cnn.rank_probabilities()







=======
.. autofunction:: flexsweep.fv_v2.ncd1
.. autofunction:: flexsweep.fv_v2.run_beta_window
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
