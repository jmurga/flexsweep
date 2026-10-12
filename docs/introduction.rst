Introduction
============

`Flex-sweep <https://doi.org/10.1093/molbev/msad139>`_ is a convolutional neural
network–based method able to detect a wide range of selective sweep signals,
including those thousands of generations old, from single-species genomic data,
while being robust to background selection. It was primarily designed to be
flexible in terms of the types, strengths, and ages of sweeps, requiring only a
single phased population — without outgroups or admixture data — making it
particularly useful for non-model species.

Flex-sweep 2.0 is a more general-purpose update that streamlines the workflow,
vastly speeds up summary-statistic computation, relaxes CNN constraints by
supporting custom architectures, enables ancestral-state polarization, and adds
domain-adaptive training to reduce the effect of simulation mismatch. Flex-sweep now
scales to thousands of simulated training scenarios and enables genome-wide
inference on a standard workstation rather than an HPC.

Similarly to the first version, Flex-sweep works in three main steps:
simulation, summary statistics estimation (feature vectors), and
training/classification. Once installed, you can access the Command Line
Interface (CLI) to run any module as needed.


Pipeline overview
-----------------

Simulations
~~~~~~~~~~~

The simulations module reads demographic histories in
`demes <https://doi.org/10.1093/genetics/iyac131>`_ format and ships the
:math:`N_e(t)` trajectories of the 26 1000 Genomes populations. It includes a
precompiled `discoal <https://github.com/kr-colab/discoal>`_ binary, so no external
simulator is needed (a custom binary path can be given). Every replicate draws its
mutation rate, recombination rate, selection coefficient, sweep age and start and end
allele frequencies from ranges set on the command line or in Python. See :doc:`simulation`.

Feature vector estimation
~~~~~~~~~~~~~~~~~~~~~~~~~

The feature vector module was fully refactored for speed and traceability.
Summary statistics are estimated using Numba and NumPy vectorisation for all
statistics except :math:`nS_L` and :math:`iHS`, which rely on
`scikit-allel <https://scikit-allel.readthedocs.io/>`_ functions. All outputs
rely on `Polars <https://pola.rs/>`_ DataFrames, removing the large number of
intermediate files required by the previous version and substantially reducing
RAM consumption.

The software directly reads VCF files — no conversion to hap/map format is
needed. When using recombination maps, genetic positions and recombination rates
are interpolated automatically.

The refactored version includes optimised implementations of iSAFE, DIND,
hapDAF-o/s, Sratio, highfreq, lowfreq, as well as the custom HAF and H12
statistics from the original Flex-sweep paper. Additional statistics have been
added (see :doc:`feature_vectors` for the full list).

Users can select any custom combination of the available statistics via the
``stats=`` argument, choosing only those most informative for the organism and
sweep type under study. The software also supports user-defined genomic
intervals and window sizes, enabling feature extraction across highly
customisable regions. This design makes it straightforward to reproduce the
feature set of other tools (e.g. diploS/HIC) or to define a tailored feature
vector without writing custom code.

Normalization
~~~~~~~~~~~~~

Before training or prediction, Flex-sweep converts every
statistic to a z-score, so that each value measures how unusual it is
relative to a reference distribution:

.. math::

   z_i = \frac{x_i - \mu}{\sigma}

where :math:`x_i` is the value of a statistic, and :math:`\mu` and
:math:`\sigma` are the mean and standard deviation of the reference set it is
compared with. The reference depends on the input:

* **Simulations**: :math:`\mu` and :math:`\sigma` are estimated from the
  **neutral** replicates only and applied to neutral and sweep replicates
  alike, so a sweep appears as a departure from neutral expectations.
* **VCFs**: :math:`\mu` and :math:`\sigma` are estimated from the empirical
  regions themselves. Each dataset is therefore centred on its own
  distribution, which removes part of the difference between simulated and
  real data.

**Per-SNP statistics** (iHS, nSL, DIND, hapDAF, Sratio, ...) are standardized
within bins of derived allele frequency (DAF), following
`Voight et al. 2006 <https://doi.org/10.1371/journal.pbio.0040072>`_. A SNP is
compared only with SNPs of similar frequency, which removes the dependence of
these statistics on allele frequency:

.. math::
   z_i^{(f)} = \frac{x_i^{(f)} - \mu^{(f)}}{\sigma^{(f)}}

where :math:`f` is the DAF bin of SNP :math:`i`. The standardized SNP values
are then summarized in each window of the feature vector.

**Window statistics** (:math:`\pi`, Tajima's D, H12, HAF, ...) are
standardized within each combination of window size :math:`w` and center
:math:`c`, so that a value is compared only with values computed over the same
genomic geometry:

.. math::

   z_i^{(w,c)} = \frac{x_i^{(w,c)} - \mu^{(w,c)}}{\sigma^{(w,c)}}

When a high-quality recombination map is available,
Flex-sweep normalises both SNP-based and windowed statistic within bins defined by 
the empirical recombination distribution. For windowed statistic each input is standardized 
within bins defined by the joint combination of window position, window center,
and local recombination rate, extending the normalization scheme to explicitly
account for recombination rate heterogeneity across the genome:

.. math::
   Z\text{-score} = \frac{\pi_i^{(w,c,r)} - \mu^{(w,c,r)}}{\sigma^{(w,c,r)}}


where :math:`\pi_i` is any given statistic and :math:`\mu^{(w,c,r)}`,
:math:`\sigma^{(w,c,r)}` are the mean and standard deviation computed across
all windows sharing the same window size :math:`w`, center :math:`c`, and
recombination rate bin :math:`r`.

For SNP-based statistics, normalization is performed analogously but conditional 
on derived allele frequency (DAF) and local recombination rate. Specifically,
each SNP-level statistic is first standardized within bins defined
by the joint distribution of DAF and recombination rate:

.. math::

   \pi_{i(std)} = \pi_i^{(f,r)}=\frac{\pi_i^{(f,r)} - \mu^{(f,r)}}{\sigma^{(f,r)}}

where :math:`\pi_i` is the given statistics, :math:`f` denotes the derived allele frequency bin of SNP :math:`i`,
and :math:`\mu^{(f,r)}` and :math:`\sigma^{(f,r)}` are the mean and standard 
deviation computed across all SNPs falling within the same DAF bin :math:`f` and
recombination rate bin :math:`r`. Finally, once each window and center combination is defined
we normalized using the same approach described above.


Training and prediction
~~~~~~~~~~~~~~~~~~~~~~~

The CNN input is an image of summary statistics: one row per statistic and one
column per (window size, center) position. Flex-sweep reads the statistics, windows and
centers from the column names of the feature-vector file, so a feature vector with any
combination of them trains without extra settings, and any Keras architecture, 2D or 1D,
can replace the default network (see :doc:`cnn`). Separately, Flex-sweep can build
sorted haplotype-matrix images from the same simulations and VCFs, for CNNs that take
haplotypes rather than summary statistics as input (see :doc:`haplotype_matrix`); these
images are not part of the Flex-sweep feature vector so it will require advance usage of the API.

Domain-adaptive neural network (DANN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulations never match real data exactly: demography, mutation and recombination
landscapes, and background selection all shift the summary statistics. To reduce the
effect of this mismatch, Flex-sweep implements the domain-adversarial network of
`Mo and Siepel 2023 <https://doi.org/10.1371/journal.pgen.1011032>`_. It trains on
labelled simulations and on unlabelled empirical feature vectors together, and learns
features that predict sweeps but do not separate simulated from real windows during training. Please not that when the
training simulations is far from the truth, it can still perform worse than the plain
CNN (see :doc:`dann`).

Polarization
~~~~~~~~~~~~

Flex-sweep refactors and extends `est-sfs <https://doi.org/10.1534/genetics.118.301120>`_
to support straightforward allele polarization. Given a focal-species
polymorphism dataset (VCF) and a Multi-Alignment Format (MAF) file, the
pipeline:

1. Subsets up to three outgroup species from the alignment.
2. Extracts aligned outgroup bases at polymorphic sites.
3. Computes per-site posterior ancestral-state probabilities following
   `Keightley and Jackson 2018 <https://doi.org/10.1534/genetics.118.301120>`_.

These posteriors are used to probabilistically assign ancestral and derived
states, enabling downstream selection scans that are robust to ancestral
mis-specification. Any reference-anchored MAF works, including alignments
extracted from large comparative genomic resources such as the
`Zoonomia Project <https://doi.org/10.1038/s41586-020-2876-6>`_: the example data
are polarized with the great apes of the Zoonomia 447-mammal Cactus alignment.

Genome-wide outlier scan
~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the CNN/DANN prediction pipeline, Flex-sweep includes a
standalone genome-wide outlier scan that works directly from VCF files without
any simulation or training. Each statistic is computed at its native genomic
resolution and ranked against its empirical distribution to produce a p-value:

.. math::

   p_i = \frac{\text{rank}(-x_i)}{N_{\text{valid}}}

where a small :math:`p_i` indicates a high-scoring outlier window. Statistics
can be normalised jointly by allele frequency and recombination rate when a
recombination map is provided. This provides a fast, assumption-light complement
to the CNN approach, or a standalone analysis when simulations are not
available. See :doc:`scan` for full documentation.

Rank and enrichment
~~~~~~~~~~~~~~~~~~~

``flexsweep rank`` assigns sweep probabilities to genomic features. It takes any set
of genomic elements of interest in a BED file (e.g. genes, enhancers, transcription factors, etc.), computes the
distance from each element to the predicted windows with
`polars-bio <https://github.com/biodatageeks/polars-bio>`_, and gives each element the
highest sweep probability among its *k* nearest windows, so elements can be ranked by
their sweep signal. See :doc:`prediction` for usage.

To assess whether top-ranked genes or elements are sweep enriched, Flex-sweep implements
the enrichment pipeline described in `Enard et al. 2020 <https://doi.org/10.1098/rstb.2019.0575>`_ 
(see :doc:`enrichment` for details).


.. _requirements:

Recommended requirements
------------------------

Flex-sweep runs on a standard workstation. Memory and run time depend mainly on the
number of simulations, the sample size and the genome size. The examples and results
in this documentation were produced on:

* **Workstation**: AMD Ryzen 9 PRO 5945 (24 threads), 64 GB RAM, NVIDIA GeForce
  RTX 3080, Pop!_OS 22.04. All models were trained on RTX 3080 GPU.
* **Local server**: used only to speed up large simulation and feature-vector runs:
  2 x AMD EPYC 7713 (128 cores, 255 threads), 4 TB RAM.

A GPU is optional: training runs on the CPU, more slowly. A run of the
:doc:`quickstart` needs neither the server nor a GPU.
