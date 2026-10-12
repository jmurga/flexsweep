.. _simulation:

Simulations
===========

Flex-sweep learns what a selective sweep looks like from simulations. The simulator
produces the labelled training data: the same number of **neutral** and **sweep**
replicates of a 1.2 Mb locus, simulated with
`discoal <https://github.com/kr-colab/discoal>`_ (`Kern and Schrider 2016
<https://doi.org/10.1093/bioinformatics/btw556>`_) under a demographic model in
`demes <https://doi.org/10.1093/genetics/iyac131>`_ format. Each replicate draws its own
mutation and recombination rate, and each sweep replicate its own selection coefficient,
age, start and end frequency, from the ranges described below. The ranges define how
varied the training set is: the classifier only learns the sweeps and the neutral
variation it has seen. They are sampling ranges, not Bayesian priors; nothing in
Flex-sweep computes a posterior from them.

Command line and Python interface
---------------------------------

.. code-block:: bash

    flexsweep simulator \
        --sample_size 216 \
        --demes yri \
        --output_folder yri_sims \
        --num_simulations 22000 \
        --nthreads 24 \
        --seed 1

.. code-block:: python

    import flexsweep as fs

    sim = fs.Simulator(
        sample_size=216,
        demes="yri",
        output_folder="yri_sims",
        num_simulations=22000,
        nthreads=24,
        seed=1,
    )
    df_params = sim.create_params()   # draw and save the per-replicate parameters
    sim.simulate_batch()              # run discoal

``--sample_size`` counts haplotypes, so it is twice the number of diploid individuals
in your VCFs (216 for the 108 unrelated YRI). Options you do not set take the
``Simulator`` defaults, so the two interfaces above produce the same simulations.

Simulation parameters
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 17 43

   * - CLI option
     - Python argument
     - Default
     - Meaning
   * - ``--num_simulations``
     - ``num_simulations``
     - ``250000``
     - Neutral replicates, and the same number of sweep replicates.
   * - ``--locus_length``
     - ``locus_length``
     - ``1200000``
     - Length of the simulated locus (bp).
   * - ``--mutation_rate``
     - ``mutation_rate``
     - uniform 5e-9 to 2e-8
     - Per-bp mutation rate :math:`\mu` (see :ref:`rate-specs`).
   * - ``--recombination_rate``
     - ``recombination_rate``
     - exponential, mean 1e-8, in 1e-9 to 1e-7
     - Per-bp recombination rate :math:`r` (see :ref:`rate-specs`).
   * - ``--s``
     - ``s``
     - ``0.01,0.05``
     - Selection coefficient, uniform in [min, max].
   * - ``--time``
     - ``time``
     - ``0,5000``
     - Generations since the sweep ended, uniform in [min, max] (discoal ``-ws``).
   * - ``--saf``
     - ``saf``
     - ``0,0.1``
     - Start allele frequency :math:`f_i` of soft sweeps, uniform in [min, max].
   * - ``--eaf``
     - ``eaf``
     - ``0.5,1``
     - End allele frequency :math:`f_t` of incomplete sweeps, uniform in [min, max].
   * - ``--fixed_ratio``
     - ``fixed_ratio``
     - ``0.1``
     - Fraction of complete sweeps (see :ref:`sweep-classes`).
   * - ``--x``
     - ``x``
     - ``0.5``
     - Sweep position as a fraction of the locus (see :ref:`sweep-position`).
   * - ``--nthreads``
     - ``nthreads``
     - ``1``
     - Parallel discoal jobs.
   * - ``--seed``
     - ``seed``
     - unseeded
     - Seed for the parameter draws and every discoal replicate.
   * - ``--discoal_path``
     - ``discoal_path``
     - bundled binary
     - discoal executable (see :ref:`discoal-binary`).

On the command line, ranges are written ``min,max`` (``--s 0.005,0.1``); in Python they
are lists (``s=[0.005, 0.1]``).

.. _rate-specs:

Mutation and recombination rates
--------------------------------

Drawing the rates per replicate makes the classifier robust to local variation in
mutation and recombination rate along the genome. Both rates accept four forms:

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Distribution
     - CLI
     - Python
   * - Fixed
     - ``1e-8``
     - ``{"dist": "fixed", "value": 1e-8}``
   * - Uniform in [min, max]
     - ``5e-9,2e-8``
     - ``{"dist": "uniform", "min": 5e-9, "max": 2e-8}``
   * - Exponential with ``mean``, truncated to [min, max]
     - ``1e-9,1e-7,1e-8``
     - ``{"dist": "exponential", "min": 1e-9, "max": 1e-7, "mean": 1e-8}``
   * - Normal with ``mean`` and ``std``, truncated to [min, max]
     - Python only
     - ``{"dist": "truncnorm", "min": 1e-9, "max": 5e-8, "mean": 1e-8, "std": 5e-9}``

.. code-block:: bash

    # Same mutation rate in every replicate, recombination rate uniform
    flexsweep simulator --sample_size 216 --demes yri --output_folder yri_sims_fixed_mu \
        --mutation_rate 1.25e-8 --recombination_rate 1e-9,4e-8

.. code-block:: python

    sim = fs.Simulator(
        sample_size=216,
        demes="yri",
        output_folder="yri_sims_fixed_mu",
        mutation_rate={"dist": "fixed", "value": 1.25e-8},
        recombination_rate={"dist": "uniform", "min": 1e-9, "max": 4e-8},
    )

A fixed rate is useful for controlled experiments. A fixed recombination rate is not
compatible with recombination-stratified normalisation (``r_bins``, see
:doc:`feature_vectors`), which needs variation in :math:`r` among the neutral replicates.

.. _sweep-classes:

Sweep classes
-------------

Half of the sweep replicates are **hard** sweeps, from a new mutation
(:math:`f_i = 0`), and half are **soft** sweeps from standing variation, with
:math:`f_i` drawn from ``--saf``. Within each half, a fraction ``--fixed_ratio`` are
**complete** sweeps, in which the selected allele reaches :math:`f_t = 1`; the rest are
**incomplete**, with :math:`f_t` drawn from ``--eaf``. With the default
``fixed_ratio = 0.1`` and 22,000 sweep replicates:

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Class
     - Replicates
     - :math:`f_i`
     - :math:`f_t`
   * - Hard, complete
     - 1,100
     - 0
     - 1
   * - Hard, incomplete
     - 9,900
     - 0
     - ``--eaf``
   * - Soft, complete
     - 1,100
     - ``--saf``
     - 1
   * - Soft, incomplete
     - 9,900
     - ``--saf``
     - ``--eaf``

Raise ``--fixed_ratio`` if you expect mostly complete sweeps; ``--fixed_ratio 1`` gives
complete sweeps only and ``0`` incomplete sweeps only. The CNN is trained on two labels,
neutral and sweep; the classes only describe how varied the sweep set is.

.. note::

   When Flex-sweep groups sweeps by age, for example when reporting results by sweep
   class, a sweep is called **old** if it ended :math:`t \ge 2000` generations ago and
   **young** otherwise. This cutoff is a convention for grouping results. It is not a
   biological boundary, it is not a model input, and the classifier never sees it.

.. _sweep-position:

Sweep position
--------------

``--x`` places the selected site along the locus, as a fraction of ``--locus_length``
(discoal ``-x``). The default ``0.5`` puts it at the center, 600 kb into a 1.2 Mb locus.
A single value fixes the position; ``min,max`` draws one position per sweep replicate
uniformly from that range.

.. code-block:: bash

    # Sweep anywhere between about 400 and 800 kb of the 1.2 Mb locus
    flexsweep simulator --sample_size 216 --demes yri --output_folder yri_sims_offcenter \
        --x 0.333,0.667

.. code-block:: python

    sim = fs.Simulator(216, "yri", "yri_sims_offcenter", x=[0.333, 0.667])

Feature vectors and predictions refer to the **center** of the locus: with the default
``x = 0.5``, the CNN learns to recognise a sweep located at the center of each 1.2 Mb
window. Off-center sweeps label as "sweep" windows whose selected site lies away from
the center, so the classifier learns to call those too; keep the default unless that is
what you want.

.. _shipped-models:

Demographic model
-----------------

``--demes`` (``demes=`` in Python) takes a demes YAML file or one of these shipped
models, case-insensitive:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Alias
     - Model
   * - ``acb``, ``asw``, ``beb``, ``cdx``, ``ceu``, ``chb``, ``chs``, ``clm``, ``esn``,
       ``fin``, ``gbr``, ``gih``, ``gwd``, ``ibs``, ``itu``, ``jpt``, ``khv``, ``lwk``,
       ``msl``, ``mxl``, ``pel``, ``pjl``, ``pur``, ``stu``, ``tsi``, ``yri``
     - Effective population size trajectory of each 1000 Genomes population
       (``data/<pop>_spiedel_2019_full.yaml``, 30 piecewise-constant epochs).
   * - ``constant``
     - Constant size, :math:`N_e` = 20,000.
   * - ``africa_1t12``
     - African population of Tennessen et al. 2012 (stdpopsim ``Africa_1T12``):
       7,310 -> 14,474 at 5,920 generations ago, exponential growth to 432,125 over the
       last 204.6 generations.

.. note::

   The 26 population models are **single-population** :math:`N_e(t)` trajectories
   estimated by Relate from the 1000 Genomes data
   (`Speidel et al. 2019 <https://doi.org/10.1038/s41588-019-0484-x>`_). Epoch
   boundaries are Relate's log-spaced bins in years, from :math:`10^3` to :math:`10^7`
   years, converted with 28 years per generation. Each model describes one population on
   its own: it has no population splits, migration or admixture, and the population
   sizes are coalescence-rate estimates, not census sizes. Admixed populations (ACB,
   ASW, CLM, MXL, PEL, PUR) are therefore modelled only by their pooled size history.

.. code-block:: python

    print(fs.POPULATIONS_1KG)                                   # the 26 codes
    print(fs.DEMES_EXAMPLES["ceu"])                             # file behind an alias
    print(fs.simulate_discoal.demes.load(fs.DEMES_EXAMPLES["yri"]))

Only the first deme of a YAML file is simulated, as a single population; its epochs
become discoal ``-en`` size changes. discoal works in coalescent units, and Flex-sweep
converts with the size of the **oldest** epoch (the ancestral :math:`N_e`):
:math:`\theta = 4N_eL\mu`, :math:`\rho = 4N_eLr`, :math:`\alpha = 2N_es` and
:math:`\tau = t / 4N_e`. ``params.txt.gz`` keeps the per-bp and per-generation values.
:doc:`dann` describes how to test the effect of a wrong demographic model.

.. _discoal-binary:

The discoal binary
------------------

The wheel ships a precompiled discoal for x86-64 Linux (``flexsweep/data/discoal``,
exposed as ``flexsweep.DISCOAL``). The simulator uses the first executable it finds in
this order:

1. ``--discoal_path`` (``discoal_path=`` in Python);
2. the ``FLEXSWEEP_DISCOAL_BIN`` environment variable;
3. the bundled binary;
4. ``discoal`` on your ``PATH``.

If none is found, for example on macOS or ARM where the bundled binary cannot run,
Flex-sweep clones `kr-colab/discoal <https://github.com/kr-colab/discoal>`_ and compiles
it with ``make``; this needs ``git``, ``make`` and a C compiler. You can also compile
discoal yourself and pass its path.

Output
------

.. code-block:: text

    yri_sims/
        params.txt.gz         one row per replicate: iter, mu, r, eaf, saf, s, t, pos,
                              model (neutral or sweep), seed1, seed2
        neutral/neutral_<iter>.ms.gz
        sweep/sweep_<iter>.ms.gz

Neutral rows have ``s = t = eaf = saf = 0``. ``--seed`` seeds the parameter draws and
every discoal replicate (``-d seed1 seed2``). The per-replicate seeds are written to
``params.txt.gz`` even without ``--seed``, so any single replicate can be re-run
exactly, and two runs with the same seed give identical output.
