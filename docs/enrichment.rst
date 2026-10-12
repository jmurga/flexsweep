Enrichment analysis
===================

The enrichment analysis tests whether a user-defined set of genomic elements
shows a significant excess or deficit of positive selection signals relative to
matched control sets. It is an adaptation of the method from
`Enard et al. (2020) <https://doi.org/10.1098/rstb.2019.0575>`_, which was
originally developed to test whether **virus-interacting proteins (VIPs)**
accumulate sweep signals. The term *VIPs* appears throughout the code and
original pipeline as the label for the case element set; here it refers
generically to whatever set you provide.

The case set can be any genomic annotation: immune genes, disease-associated
loci, regulatory elements, conserved non-coding sequences, or any other
category. The key requirement is supplying relevant **confounding factors** for
your element type so that control sets can be properly matched (see
`Bootstrapping`_ below).


Biological background
---------------------

Gene ranking and the enrichment curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any per-element selection signal can be used — for example, the CNN sweep
probability from ``flexsweep cnn``. The score is averaged over a fixed-size
genomic window centred on each element, giving one score per element per
population. Elements are then **ranked genome-wide** from strongest signal
(rank 1) to weakest.

A **rank threshold** is a cutoff applied to these ranks. Threshold = 500 means
"the top 500 elements with the strongest selection signal". The analysis tests
many thresholds simultaneously (by default spanning top 20 to top 6000).

For each threshold *t* and population:

* **Case count** = number of case elements with rank ≤ *t*
* **Control mean** = mean count across all matched control sets
* **Ratio** = (case count + 0.1) / (control mean + 0.1)

Plotting the ratio against the threshold produces the **enrichment curve**. A
ratio > 1 means case elements are over-represented among the top-ranked
elements. Enrichment concentrated at stringent thresholds (small *t*) suggests
strong, rare sweeps; enrichment spread across lenient thresholds suggests a
broader, moderate signal.

Bootstrapping
~~~~~~~~~~~~~

A naive comparison would be confounded because many genomic properties
correlate with selection signals independently of biology — GC content, gene
length, expression level, and local recombination rate all influence sweep
scores. To avoid false positives, the bootstrap builds **matched control sets**:
elements drawn from a background pool (non-case elements at least
``min_distance`` bp from any case element) such that the mean of every
confounding factor stays within ± ``tolerance`` of the case-set mean.

The choice of confounders depends on your element type. For protein-coding
genes (as in Enard et al. 2020), typical confounders are coding-sequence
density, GC content, recombination rate, expression level, conservation scores,
or dN/dS. For regulatory elements, relevant confounders might include chromatin
accessibility, sequence conservation, or distance to the nearest transcription
start site. Factors with a non-monotonic relationship to sweep prevalence (e.g.
dN/dS) can be excluded from matching. Any numeric per-element values are
accepted via the ``--factors`` file.

Matching produces 1000 control sets (``n_control_sets`` in the Python API), a distribution of
expected counts reflecting what you would see by chance given the same genomic
properties as the case set.

False Discovery Rate
~~~~~~~~~~~~~~~~~~~~

The per-threshold p-values from the bootstrap are **not independent** — nearly
the same elements appear at adjacent thresholds, so corrections across
thresholds are not straightforward. Instead, the whole-curve FDR provides a
single p-value by:

1. **Genome shuffling** — generating null genomes where selection ranks are
   randomly reassigned between elements while preserving local clustering
   structure (blocks of ``shuffling_segs`` elements are randomly relocated).
2. **Re-running the sweep count** on each shuffled genome with the same case
   and control sets as the real analysis, ensuring bootstrap biases are the
   same in the null as in the real test.
3. **Computing the test statistic** as the total area above the control mean
   under the enrichment curve (summed excess across all thresholds passing
   filters).
4. **Reporting the FDR p-value** as the fraction of null replicates where the
   null statistic ≥ the real statistic.

Calibration against random gene sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The control-set p-values and the genome-shuffle FDR are not guaranteed to be
calibrated: whether ~5% of gene sets with no sweep enrichment reach p < 0.05
depends on the gene set, the confounding factors and the settings. In the
Flex-sweep 2.0 application (26 1000 Genomes populations, 1,212 HIV-interacting
proteins, 23 thresholds from 20 to 6000, ``--tolerance 0.05``,
``--min_distance 1200000``, 8 shuffles), 100 random sets of 1,212 non-case
genes reached a control-set p < 0.05 in 20-67% of cases, depending on the
threshold and group, and their whole-curve statistic exceeded all 8 shuffles in
41% of cases (11% expected). Random sets exceeded their own controls on average,
and their excess varied between sets 5-7 times more than the control counts
vary within one set. Both p-values were therefore anti-conservative there.

Before reading the p-values as p-values, run the same command, with the same
settings, on random gene sets of the same size and compare the case set with
them:

1. Draw ``N`` random sets (e.g. 100 or 1000) of the case-set size, uniformly
   from the elements present in both the factors table and the rank file,
   excluding the case elements (and HLA/histone genes for human genes).
2. Remove the case elements from the ``--factors`` table given to these runs,
   so they can never be controls.
3. Run ``flexsweep enrichment`` on each random set with the case-set settings
   and a different ``--seed`` per set, each into its own ``--output_folder``.
4. For each scope and threshold, compare the case excess
   (``vip_count - ctrl_mean`` in ``enrichment_curves.txt``) with the random-set
   excesses; for the whole curve, compare ``real_stat`` in ``fdr.txt``. The
   empirical p-value is ``(1 + k) / (N + 1)``, with ``k`` the number of random
   sets whose statistic is at least that of the case set.

The fraction of random sets with p < 0.05 in their own ``enrichment_curves.txt``
and ``fdr.txt`` shows how far the pipeline p-values are from nominal for your
data. The smallest attainable empirical p-value is ``1 / (N + 1)``. A random set
whose bootstrap stops (see ``--tolerance``) cannot be matched; replace it with
the next set and report how many were replaced.


How the pipeline works
----------------------

1. **Bootstrap control sets** — builds 1000 sets of control
   elements matched to the case set on all confounding factors, keeping a
   minimum genomic distance from case elements. Each control set has the same
   size as the case set. Matching constrains the running mean of the sampled
   controls, so the sets are matched on average, not one by one.
2. **Count sweeps** — for each rank threshold and population, counts case and
   control elements ranked within the threshold. A group or ``All`` count is the
   sum of its populations' counts (an element in a sweep in k populations counts
   k times). Elements are counted individually (no clustering).
3. **Genome shuffling** — creates a null distribution by shuffling blocks of the
   elements used by the bootstrap (case and control elements) across the genome
   ``n_shuffles`` times.
4. **FDR estimation** — computes an empirical whole-curve FDR p-value by
   comparing the real enrichment statistic to the null distribution.


Command line and Python interface
---------------------------------

First rank the genes in every population with ``flexsweep rank``. Given a folder with
one prediction file per population and ``--populations``, it ranks each file and writes a
single table, one rank column per population in ``--populations`` order, which is the
``--sweep_files`` input of ``flexsweep enrichment``:

.. code-block:: bash

    flexsweep rank \
        --prediction_dir predictions/ \
        --populations yri,ceu \
        --feature_coordinates genes.bed \
        --output predictions/all_ranks_v3

    flexsweep enrichment \
        --sweep_files predictions/all_ranks_v3 \
        --gene_set case_genes.txt \
        --factors confounders.tsv \
        --annotation genes.bed \
        --populations yri,ceu \
        --groups afr,eur \
        --thresholds 6000,5000,4000,3000,2000,1500,1000,500,200,100,50,20 \
        --nthreads 8 \
        --output_folder enrichment_out

``--pattern`` (default ``{pop}_*predictions.txt``) locates each population's prediction
file; it must match exactly one file per population. The command prints the column
assigned to each population and the number of genes tied at the highest sweep
probability. The text after the last ``_`` of the output name labels the dataset in the
enrichment output (``v3`` above).

.. code-block:: python

    from flexsweep.utils import find_prediction_files, rank_merge
    from flexsweep.enrichment import run_enrichment

    pops = ["yri", "ceu"]
    files = find_prediction_files("predictions/", pops)
    df_ranks, ties = rank_merge(
        files, pops, "genes.bed", output="predictions/all_ranks_v3"
    )

    df_case, df_control, df_curves, df_fdr = run_enrichment(
        sweep_files=["predictions/all_ranks_v3"],
        gene_set="case_genes.txt",
        factors_file="confounders.tsv",
        annotation_file="genes.bed",
        populations=pops,
        groups=["afr", "eur"],
        thresholds=[6000, 5000, 4000, 3000, 2000, 1500, 1000, 500, 200, 100, 50, 20],
        nthreads=8,
    )

``run_enrichment`` prints a short log (seed, gene set, bootstrap, one FDR line per sweep file, timings) and returns
four Polars DataFrames: the case genes, the bootstrap control sets, the per-threshold
enrichment curves, and the FDR results (one row per sweep file; columns ``scope``,
``p_value``, ``p_value_perm``, ``n_replicates``, ``real_stat``, ``max_null_stat``, ``dataset``).


Plotting the enrichment curves
------------------------------

``flexsweep enrichment --output_folder`` writes ``enrichment_curves.txt`` and one
``enrichment_curves_<sweep file name>.svg`` per sweep file. To redraw from the saved
table without rerunning the bootstrap, e.g. with other scopes or one panel per scope:

.. code-block:: bash

    flexsweep enrichment-plot \
        --curves enrichment_out/enrichment_curves.txt \
        --scopes afr,amr,eas,eur,sas \
        --facet

.. code-block:: python

    from flexsweep.enrichment import plot_enrichment

    figs = plot_enrichment(
        "enrichment_out/enrichment_curves.txt",
        scopes=["afr", "amr", "eas", "eur", "sas"],
        facet=True,
        output_folder="enrichment_out",
    )

Scopes are the population codes, the group labels and ``All``; without ``--scopes``
every scope is drawn. ``--ci`` shades the 95% interval of the ratio across matched
control sets: the case count divided by the 97.5% and 2.5% percentiles of the control
counts, ``(case + 0.1) / (ctrl_ci_hi + 0.1)`` to ``(case + 0.1) / (ctrl_ci_lo + 0.1)``.
It does not resample the case genes. At strict thresholds where almost every control set
has zero sweeps, both percentiles are 0 and the band collapses to a point.
``plot_enrichment`` also accepts the curves DataFrame returned by
``run_enrichment`` and returns one matplotlib figure per sweep file.


Input file formats
------------------

**sweep_files**
    Tab- or space-separated file (optionally gzipped), no header, as written by
    ``flexsweep rank --prediction_dir``. First column: ``gene_id``. Remaining columns:
    per-population sweep ranks (integers). Column order must match ``--populations``:
    columns are read by position.

**gene_set**
    One case gene id per line (no header). Every other gene of the universe (genes
    present in both ``factors`` and the first sweep file) is eligible as a control.
    Listed ids outside the universe are dropped and counted in the log.

**factors**
    Tab- or space-separated file (no header). First column: ``gene_id``.
    Remaining columns: numeric confounding factor values used for control
    matching.

**annotation**
    BED file (0-based, no header): ``chr``, ``start``, ``end``, ``gene_id``.
    Used for genomic distance computation and genome shuffling.


Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Option
     - Default
     - Description
   * - ``--sweep_files``
     - required
     - Comma-separated paths to sweep rank files.
   * - ``--gene_set``
     - required
     - Case gene ids, one per line (no header).
   * - ``--factors``
     - required
     - TSV confounding factors file.
   * - ``--annotation``
     - required
     - BED gene coordinates file (0-based, no header).
   * - ``--populations``
     - required
     - Comma-separated population codes matching sweep file column order.
   * - ``--groups``
     - required
     - Comma-separated group labels (same length as ``--populations``).
   * - ``--thresholds``
     - required
     - Comma-separated rank thresholds for the enrichment curve.
   * - ``--pop_interest``
     - ``All``
     - Perl ``Pop_interest``: the population, group or ``All`` whose p-values
       decide how many control sets are used, for the real data and for each
       shuffle: 20 sets, then 100 if one of its thresholds has p ≤ 0.05 or
       ≥ 0.95, then all if one has p ≤ 0.01 or ≥ 0.99 (``n_sets`` column). It is
       a regular expression on the scope label. The FDR statistic always uses
       ``All``.
   * - ``--cluster_distance``
     - ``500000``
     - Neighbour distance in bp for ``--count_sweeps`` (Perl
       ``Cluster_distance``): genes whose 10 kb bins of the gene centre are at
       most this far apart. Unused without ``--count_sweeps``.
   * - ``--count_sweeps``
     - ``False``
     - Perl ``Count_sweeps yes``: neighbouring genes in the same sweep count
       once, in the real data and in every shuffle. In the case set, the first
       gene of a cluster (by gene id, as the Perl ``VIPs/file_1``) counts and
       its in-sweep neighbours do not; in a control set, a gene and its
       in-sweep neighbours present in the set add their mean multiplicity.
   * - ``--tolerance``
     - ``0.05``
     - Allowed ± fraction deviation in factor averages when matching control
       elements to the case set.
   * - ``--min_distance``
     - ``1250000``
     - Minimum bp distance from any case element for an element to be eligible
       as a control.
   * - ``--flip``
     - ``False``
     - Flip test direction when the control pool is smaller than the case set.
       Increases statistical power in that scenario.
   * - ``--max_rep``
     - ``25``
     - Maximum average resamples per control element across all bootstrap sets.
   * - ``--nthreads``
     - ``1``
     - Number of parallel workers (joblib).
   * - ``--n_shuffles``
     - ``8``
     - Number of FDR genome-shuffle replicates. Must be a multiple of 8. The
       default is only a quick check; use 1000 or more for reported p-values.
   * - ``--shuffling_segs``
     - ``2``
     - Size of each genomic shuffle block in elements. Corresponds to
       ``Shuffling_segments_number`` in the current Enard-lab Perl shuffler.
   * - ``--shuffle_blocks``
     - none
     - Cut the shuffled genome into this many blocks of
       ``int(n_genes / shuffle_blocks)`` elements instead of
       ``--shuffling_segs``-element segments. This is the commented-out line of
       ``shuffle_sweep_script_nocut.pl`` and the design described by Enard &
       Petrov (2020), who used 100 blocks.
   * - ``--seed``
     - random
     - Seed for the bootstrap and the genome shuffles. When not given, one is
       drawn and printed (``Seed: ...``) so the run can be repeated.
   * - ``--reuse_bootstrap``
     - none
     - Output folder of a previous run. Its ``case_set.txt``,
       ``control_set.txt`` and ``used_genes.txt`` are loaded instead of running
       the bootstrap, e.g. to
       rerun with more shuffles or another rank table. The case genes must
       belong to the current gene set and every control gene to the current
       universe, otherwise the run stops.

**Key parameter notes**

* ``--min_distance`` (default 1.25 Mb) prevents nearby elements from acting as
  controls, ensuring controls are independent of the case set.

* ``--tolerance`` (default 0.05 = ±5%) controls how strictly controls must
  match the case set on each confounding factor. Loosen it (e.g. 0.1) if the
  bootstrap produces very few valid control sets.

* ``--shuffling_segs`` is the **size** of each genomic shuffle block (in
  elements), *not* the number of blocks. With ~5,000 shuffled genes and
  ``shuffling_segs=2``, the genome is cut into ~2,000-2,500 segments (cuts
  between two strong-signal genes are skipped) that are randomly re-arranged.
  The Perl manual's "Number of segments to shuffle" describes an earlier
  version of the shuffler (``seg_size = int(n_genes / segcut)``, still in the
  script as a comment); that behaviour is ``--shuffle_blocks``.

* ``--n_shuffles`` must be a **multiple of 8** (kept from the Perl pipeline).
  The shuffles are counted in ``--nthreads`` blocks; 1000 shuffles of 26
  populations take minutes.

* ``--seed`` makes the whole run repeatable: the same seed, inputs and
  ``--nthreads`` give identical control sets, curves and FDR values. The bootstrap
  runs one sampling chain per thread (the Perl ``Runs_number`` and
  ``Iterations_number`` have no CLI option; the 1000 sets match their 10 × 10 × 10).
  Each chain draws genes from its own numpy PCG64 stream (``SeedSequence(seed)``
  spawned once per chain), so control sets made with versions before 2026-10-10
  are not reproduced by the same seed.
  A sampling chain can reach a state in which no pair of control genes keeps the
  factor means within ``--tolerance`` (in practice within its first few hundred
  genes). A chain that rejects 1e8 consecutive proposals is restarted on a new stream
  derived from the seed, with a warning; after 10 failed restarts the run stops with an
  error asking for a lower ``--min_distance`` or a higher ``--tolerance``. The first
  attempt of every chain uses the seed's own stream, so runs without restarts are
  unchanged.

* ``--output_folder`` receives ``case_set.txt``, ``control_set.txt``,
  ``used_genes.txt`` (elements shuffled for the FDR), ``enrichment_curves.txt``,
  ``fdr.txt`` and the curve SVGs.


Understanding the output
------------------------

``run_enrichment`` returns ``(df_case, df_control, df_curves, df_fdr)``. ``df_fdr``
(written as ``fdr.txt``) has one row per sweep file with the whole-curve FDR result
for the ``All`` scope:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - ``scope``
     - Population code, group name, or ``"All"`` (aggregate across all
       populations).
   * - ``p_value``
     - Empirical FDR p-value as in the Perl pipeline: fraction of shuffles whose
       statistic is at least ``real_stat``. It is 0 when no shuffle reaches it,
       so its resolution is 1 / ``n_shuffles``.
   * - ``p_value_perm``
     - ``(k + 1) / (n_shuffles + 1)`` with ``k`` the number of shuffles at or
       above ``real_stat``: the standard permutation p-value, never 0. Report
       this one together with ``n_shuffles``, and see "Calibration against
       random gene sets" before reading it as a p-value. With the default 8
       shuffles it is at least 1/9.
   * - ``n_replicates``
     - Number of null shuffles used to estimate the p-value.
   * - ``real_stat``
     - The observed test statistic — total area above the control mean under
       the enrichment curve, summed across all thresholds passing the filters.
       Larger = more enrichment.
   * - ``max_null_stat``
     - The maximum test statistic across all null shuffle replicates. If
       ``real_stat > max_null_stat``, the FDR p-value is 0 / ``n_shuffles``.

``df_curves`` (``enrichment_curves.txt``) holds the per-threshold curve for
every population, group and ``All``: case count, control mean, 95% interval of the
control counts (sorted counts at ``int(0.025 n)`` and ``int(0.975 n)``, as Perl),
ratio, p-value (fraction of control sets with a larger count) and ``n_sets``, the
number of control sets used.

**Interpreting the enrichment curve**

* **Ratio > 1** — case elements are over-represented among the top-ranked
  elements at that threshold. The further above 1, the stronger the signal.
* **Ratio ≈ 1** — case elements behave like controls; no excess of selection.
* **Ratio < 1** — case elements are under-represented (possible depletion).
* Enrichment at **stringent thresholds** (top 50–200) points to a small number
  of elements with very strong sweep signals.
* Enrichment at **lenient thresholds** (top 1000–6000) suggests a broader,
  moderate signal across the set.

**Clustering note (``--count_sweeps``)**

By default ``run_enrichment`` counts elements individually (Perl
``Count_sweeps no``), so one sweep covering several case genes counts several
times. ``--count_sweeps`` reproduces Perl ``Count_sweeps yes`` (checked against
``count_sweeps_script.pl`` on synthetic data).


Notes and constraints
---------------------

* **Gene IDs** must start with ``ENSG`` (Ensembl format).
* ``n_shuffles`` must be a **multiple of 8**.
* Genome shuffling does not cut between two elements that both rank within the
  largest threshold (6000 by default; Perl ``maxi_rank``) in some population, so
  strong signals are moved as a block. Each block of 8 shuffles shares which
  segments are mirrored, as one call of the Perl shuffler.
* ``shuffling_segs`` is the block **size** in elements, not the number of blocks;
  ``--shuffle_blocks`` sets the number of blocks instead.
* Each segment is kept or mirrored to the opposite end of the ordered genome
  (Perl ``dice > 5``), so a shuffle is not a permutation: some genes give their
  ranks twice and others not at all (as in the Perl shuffler).
* HLA-region genes are removed from both the case and the control sets
  automatically.
* The FDR scope is always ``All`` (the ``All:`` scope of the Enard et al. (2020)
  Perl pipeline), which accumulates the test statistic over all individual
  populations.
* Counting, shuffling and the FDR statistic follow ``count_sweeps_script.pl``,
  ``shuffle_sweep_script_nocut.pl`` and ``estimate_FPR.pl ... All:``; the test
  suite checks them against those scripts on synthetic data.
