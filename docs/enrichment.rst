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

Matching is repeated ``n_runs × n_batches`` times, producing a distribution of
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


How the pipeline works
----------------------

1. **Bootstrap control sets** — builds ``n_runs × n_batches`` sets of control
   elements matched to the case set on all confounding factors, keeping a
   minimum genomic distance from case elements. Each control set has the same
   size as the case set.
2. **Count sweeps** — for each rank threshold and population, counts case and
   control elements overlapping a sweep signal, producing enrichment ratios
   with bootstrap confidence intervals and empirical p-values. Neighbouring
   elements within ``cluster_distance`` bp can optionally be merged into sweep
   clusters before counting.
3. **Genome shuffling** — creates a null distribution by randomly shuffling
   selection rank blocks across the genome ``n_shuffles`` times.
4. **FDR estimation** — computes an empirical whole-curve FDR p-value by
   comparing the real enrichment statistic to the null distribution.


Quick start
-----------

Command line
~~~~~~~~~~~~

.. code-block:: bash

    flexsweep enrichment \
        --sweep_files yri_ceu_ranks.tsv \
        --gene_set case_elements.tsv \
        --factors confounders.tsv \
        --annotation genes.bed \
        --populations YRI,CEU \
        --groups AFR,EUR \
        --thresholds 6000,5000,4000,3000,2000,1500,1000,500,200,100,50,20 \
        --pop_interest All \
        --n_runs 10 \
        --nthreads 8


Python interface
~~~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.enrichment import run_enrichment

    fdr_results = run_enrichment(
        sweep_files=["yri_ceu_ranks.tsv"],
        gene_set="case_elements.tsv",
        factors_file="confounders.tsv",
        annotation_file="genes.bed",
        populations=["YRI", "CEU"],
        groups=["AFR", "EUR"],
        thresholds=[6000, 5000, 4000, 3000, 2000, 1500, 1000, 500, 200, 100, 50, 20],
        pop_interest="All",
        n_runs=10,
        nthreads=8,
    )

    for df in fdr_results:
        print(df)

The function prints timing information (``TIMING_BOOTSTRAP``,
``TIMING_POSTBOOTSTRAP``, ``TIMING_TOTAL``) and a per-file FDR summary line
to stdout, and returns a list of Polars DataFrames — one per sweep file — with
columns ``scope``, ``p_value``, ``n_replicates``, ``real_stat``,
``max_null_stat``.


Input file formats
------------------

**sweep_files**
    Tab- or space-separated file (optionally gzipped). First column: ``gene_id``.
    Remaining columns: per-population sweep ranks (integers). Column order must
    match ``--populations``.

**gene_set**
    Two-column TSV (no header): ``gene_id``, ``label`` (``yes``/``no``). Elements
    labelled ``yes`` are the case set; ``no`` are the control-eligible pool.
    The file argument is named ``gene_set`` and the labels ``yes``/``no`` follow
    the convention of the original Enard et al. (2020) VIP pipeline.

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
     - TSV with ``gene_id`` and ``yes``/``no`` label columns.
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
     - Population or group scope for FDR output. Use ``All`` to aggregate
       across all populations.
   * - ``--cluster_distance``
     - ``500000``
     - Maximum bp distance between elements to count them as neighbours when
       clustering sweep events.
   * - ``--n_runs``
     - ``10``
     - Number of bootstrap batches. Total control sets =
       ``n_runs × iterations_per_run``.
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
     - Number of FDR shuffle replicates. Must be a multiple of 8.
   * - ``--shuffling_segs``
     - ``2``
     - Size of each genomic shuffle block in elements. Corresponds to
       ``Shuffling_segments_number`` in the Enard et al. (2020) Perl pipeline.
   * - ``--bootstrap_dir``
     - ``""``
     - Folder containing pre-computed bootstrap output in the format produced
       by the Enard et al. (2020) Perl pipeline (``VIPs/file_1`` and
       ``nonVIPs/file_1``). When provided, the bootstrap step is skipped.

**Key parameter notes**

* ``--min_distance`` (default 1.25 Mb) prevents nearby elements from acting as
  controls, ensuring controls are independent of the case set.

* ``--tolerance`` (default 0.05 = ±5%) controls how strictly controls must
  match the case set on each confounding factor. Loosen it (e.g. 0.1) if the
  bootstrap produces very few valid control sets.

* ``--shuffling_segs`` is the **size** of each genomic shuffle block (in
  elements), *not* the number of blocks. With ~16,000 valid genes and
  ``shuffling_segs=2``, the genome is cut into ~8,000 two-element blocks that
  are randomly re-arranged. Note: the Enard et al. (2020) Perl manual describes
  this parameter as "Number of segments to shuffle", which is incorrect —
  it is the block size.

* ``--n_shuffles`` must be a **multiple of 8** (parallelism constraint).


Understanding the output
------------------------

``run_enrichment`` returns one Polars DataFrame per sweep file. Each row
represents the whole-curve FDR result for one population/group scope:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - ``scope``
     - Population code, group name, or ``"All"`` (aggregate across all
       populations).
   * - ``p_value``
     - Empirical FDR p-value. Small values (< 0.05) indicate significant
       enrichment of sweep signals in the case set. Large values (> 0.95)
       indicate depletion.
   * - ``n_replicates``
     - Number of null shuffles used to estimate the p-value.
   * - ``real_stat``
     - The observed test statistic — total area above the control mean under
       the enrichment curve, summed across all thresholds passing the filters.
       Larger = more enrichment.
   * - ``max_null_stat``
     - The maximum test statistic across all null shuffle replicates. If
       ``real_stat > max_null_stat``, the FDR p-value is 0 / ``n_shuffles``.

The intermediate per-threshold enrichment curve (ratio, p-value per threshold,
confidence intervals) is produced internally by ``count_sweeps_multipop`` and
can be accessed directly via the Python API if needed.

**Interpreting the enrichment curve**

* **Ratio > 1** — case elements are over-represented among the top-ranked
  elements at that threshold. The further above 1, the stronger the signal.
* **Ratio ≈ 1** — case elements behave like controls; no excess of selection.
* **Ratio < 1** — case elements are under-represented (possible depletion).
* Enrichment at **stringent thresholds** (top 50–200) points to a small number
  of elements with very strong sweep signals.
* Enrichment at **lenient thresholds** (top 1000–6000) suggests a broader,
  moderate signal across the set.

**Clustering note (``use_clust``)**

By default, elements within ``cluster_distance`` bp of each other are merged
into a single genomic cluster before counting, preventing the same sweep event
from being counted multiple times. The ``count_sweeps=True`` Python-API option
counts distinct sweep *events* (connected components of the neighbour graph)
rather than individual elements — this extension is not present in the original
Enard et al. (2020) Perl pipeline.


Notes and constraints
---------------------

* **Gene IDs** must start with ``ENSG`` (Ensembl format).
* ``n_shuffles`` must be a **multiple of 8**.
* ``n_batches`` (``Iterations_number`` in the Enard et al. (2020) Perl pipeline)
  must be a **multiple of 10**.
* The maximum useful rank threshold is 2000 (hardcoded pipeline limit for speed).
* ``shuffling_segs`` is the block **size** in elements, not the number of blocks.
  The Enard et al. (2020) Perl manual describes it as "Number of segments to
  shuffle" — this is incorrect.
* HLA-region genes and histone genes should be excluded from the case set before
  passing to the pipeline.
* Use ``--pop_interest All`` (the default) to reproduce the ``All:`` FDR scope
  from the Enard et al. (2020) Perl pipeline, which accumulates the test
  statistic over all individual populations.
* ``count_sweeps=True`` (Python API only) counts sweep *events* via
  connected-component analysis instead of individual elements. The Enard et al.
  (2020) Perl pipeline always counts elements.
