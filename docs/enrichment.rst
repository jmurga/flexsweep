Enrichment analysis
===================

The gene set enrichment pipeline of
`Enard & Petrov (2020) <https://doi.org/10.1098/rstb.2019.0575>`_ tests whether a set of
genes of interest is enriched in sweeps compared with control genes matched for potential
confounding factors. Flex-sweep reimplements it in Python. The code and the original
pipeline call the set of interest *VIPs* (virus-interacting proteins, the set it was
built for); any set of elements of a BED file can be used.


Method
------

Ranking
~~~~~~~

``flexsweep rank`` ranks the elements of a BED file by their sweep predictions (see
:doc:`prediction`). Each element, represented by its midpoint, receives the highest
sweep probability among the windows whose centers are nearest to it; ties are broken by
the summed probability of the windows within 500 kb, then by their summed distance.
Elements are then ranked genome-wide from highest to lowest probability, one rank column
per population.

Control sets
~~~~~~~~~~~~

Variables other than belonging to the set may also influence sweep probability, so the
pipeline uses a bootstrap procedure that resamples control genes to build control sets
with as many genes as the set of interest, in which a control gene can be sampled more
than once, up to a limit (``--max_rep``). Control genes are at least a minimum distance
(``--min_distance``) from any gene of the set, to limit the overlap of the same sweeps
with genes of interest and control genes, and HLA and histone genes are excluded.

Controls match the set on average rather than gene by gene: candidates are accepted only
while the average of every confounding factor over the sampled controls is not higher, or
not lower, than the average over the set of interest by more than a tolerance
(``--tolerance``), set as a fraction of that average. Whether the limit is on the higher
or the lower side depends on whether the candidate controls average higher or lower than
the set.

Fold enrichment
~~~~~~~~~~~~~~~

At each sweep rank threshold, the pipeline counts the genes of interest ranked within the
threshold, and the genes of each control set in the same way, with control genes counted
as many times as they were sampled. When several populations are analyzed together, the
counts are summed over populations. Fold enrichment compares the number of genes of
interest, :math:`n`, with the mean number over the control sets, :math:`\bar{c}`,

.. math::

   \mathrm{Fold\ enrichment} = \frac{n + 0.1}{\bar{c} + 0.1},

where 0.1 keeps the ratio defined when no control gene ranks within the threshold. The
:math:`P` value is the fraction of control sets with a count larger than :math:`n`. The
95% confidence interval is obtained by dividing :math:`n` by the 97.5 and 2.5 percentiles
of the control set counts instead of by their mean.

False positive rate
~~~~~~~~~~~~~~~~~~~

The pipeline also estimates a single false positive rate (FPR) for the whole enrichment
curve, because the limited number of control genes, the clustering of genes in sweeps and
the overlap of genes between thresholds can all inflate the type I error. The same counts
are repeated on randomly shuffled genomes, with the same set of interest and control
sets. To shuffle a genome, the genes of interest and the control genes are ordered along
the genome and divided into segments, without separating neighboring genes that both rank
within the largest threshold, so the clustering of sweeps is kept. The segments are then
placed in random order, so that genes receive the sweep ranks of other genes. The
enrichment of the real data and of each shuffled genome is summarized by the sum over all
thresholds of the difference :math:`n - \bar{c}`. The FPR is the fraction of shuffled
genomes whose sum is equal to or larger than the sum in the real data,

.. math::

   \mathrm{FPR} = \frac{\mathrm{shuffled\ genomes\ with\ sum} \ge
   \mathrm{real\ sum}}{\mathrm{total\ shuffled\ genomes}}.

Because shuffling removes any association between the set of interest and sweeps, the FPR
estimates the type I error, the probability of an enrichment at least as strong as the
real one in the absence of enrichment. The FPR is computed for each scope of
``--pop_interest``: ``All`` (counts summed over every population), a group, or one
population.


Running the pipeline
--------------------

First rank the genes in every population. Given a folder with one prediction file per
population, ``flexsweep rank`` writes a single table with one rank column per population,
in ``--populations`` order:

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
        --min_distance 240000 \
        --pop_interest All,YRI \
        --seed 1 \
        --nthreads 8 \
        --output_folder enrichment_out

``--pattern`` (default ``{pop}_*predictions.txt``) locates each population's prediction
file; it must match exactly one file per population. The text after the last ``_`` of the
rank table name labels the dataset in the enrichment output (``v3`` above).

The same from Python:

.. code-block:: python

    from flexsweep.utils import find_prediction_files, rank_merge
    from flexsweep.enrichment import run_enrichment

    pops = ["yri", "ceu"]
    files = find_prediction_files("predictions/", pops)
    df_ranks, ties = rank_merge(files, pops, "genes.bed", output="predictions/all_ranks_v3")

    df_case, df_control, df_curves, df_fdr = run_enrichment(
        sweep_files=["predictions/all_ranks_v3"],
        gene_set="case_genes.txt",
        factors_file="confounders.tsv",
        annotation_file="genes.bed",
        populations=pops,
        groups=["afr", "eur"],
        thresholds=[6000, 5000, 4000, 3000, 2000, 1500, 1000, 500, 200, 100, 50, 20],
        min_distance=240_000,
        pop_interest="All,YRI",
        n_control_sets=100,
        seed=1,
        nthreads=8,
        output_folder="enrichment_out",
    )

``run_enrichment`` prints a timestamped log (seed, inputs, matching settings, bootstrap,
progress of the shuffles, one FPR line per sweep file and scope) and returns four Polars
DataFrames: the genes of interest, the control sets, the enrichment curves and the FPR.

To build only the control sets, call ``run_bootstrap`` (see :doc:`api`).


Inputs
------

**--sweep_files**
    Rank table written by ``flexsweep rank`` (tab- or space-separated, optionally
    gzipped, no header): ``gene_id``, then one rank column per population. Columns are
    read by position, in ``--populations`` order.

**--gene_set**
    Genes of interest, one id per line, no header. Every other gene present in both
    ``--factors`` and the first sweep file is a candidate control. Listed ids outside
    these genes are dropped and counted in the log.

**--factors**
    Confounding factors, no header, separated by single spaces: ``gene_id``, then one
    numeric column per factor. At least 19 factors are needed (as in the Perl pipeline,
    genes with a missing 19th factor are dropped).

**--annotation**
    BED file (0-based, no header): ``chr``, ``start``, ``end``, ``gene_id``. Used for the
    distances to the genes of interest and for the genome shuffles.


Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Option
     - Default
     - Description
   * - ``--sweep_files``
     - required
     - Comma-separated rank tables.
   * - ``--gene_set``
     - required
     - Genes of interest, one per line.
   * - ``--factors``
     - required
     - Confounding factors table.
   * - ``--annotation``
     - required
     - BED gene coordinates.
   * - ``--populations``
     - required
     - Comma-separated population codes, in the column order of the rank table.
   * - ``--groups``
     - required
     - Comma-separated group label of each population.
   * - ``--thresholds``
     - required
     - Comma-separated sweep rank thresholds (e.g. ``6000,...,20``).
   * - ``--pop_interest``
     - ``All``
     - Scopes for the FPR: ``All``, groups and/or populations, comma-separated (e.g.
       ``All,YRI``). One row of ``fdr.txt`` per scope, all from the same shuffled
       genomes; not corrected for the number of scopes.
   * - ``--min_distance``
     - ``200000``
     - Minimum distance (bp) of a control gene from any gene of interest.
   * - ``--tolerance``
     - ``0.05``
     - Allowed deviation of the control average of each factor, as a fraction of the
       average over the genes of interest.
   * - ``--max_rep``
     - ``25``
     - Limit on how often a control gene is sampled: in each sampling chain, at most
       ``max_rep`` times per set-sized batch of draws.
   * - ``--skip_factors``
     - none
     - Factors left out of the matching, numbered from 1 in ``--factors`` order (e.g.
       ``9``). All factors are matched by default.
   * - ``--flip``
     - off
     - Flip the test when there are fewer candidate controls than genes of interest.
   * - ``--n_shuffles``
     - ``10000``
     - Number of shuffled genomes for the FPR (a multiple of 8). Use 8 for a quick check.
   * - ``--shuffling_segs``
     - ``2``
     - Genes per shuffled segment.
   * - ``--shuffle_blocks``
     - none
     - Cut the genome into this many segments instead (100 in Enard & Petrov 2020).
   * - ``--count_sweeps``
     - off
     - Count neighboring genes in the same sweep once (Perl ``Count_sweeps yes``). Off
       in the current Enard-lab pipeline.
   * - ``--cluster_distance``
     - ``500000``
     - Neighbor distance (bp) for ``--count_sweeps``; unused otherwise.
   * - ``--distance_file``
     - none
     - Precomputed distances to the genes of interest (``gene_id`` and distance, no
       header) instead of computing them from ``--annotation``.
   * - ``--seed``
     - random
     - Seed of the bootstrap and of the shuffles; printed when not given.
   * - ``--reuse_bootstrap``
     - none
     - Output folder of a previous run whose control sets are reused, e.g. to rerun with
       more shuffles or another rank table.
   * - ``--nthreads``
     - ``1``
     - Threads for the bootstrap and the shuffles.
   * - ``--output_folder``
     - ``.``
     - Folder for the output files.
   * - ``--facet``, ``--plot_groups``, ``--ci``
     - off
     - How the curve plot is drawn (see `Plotting`_).

The number of control sets is set from Python, ``n_control_sets`` (default 100), in
``run_enrichment`` and ``run_bootstrap``.

**Notes**

* Same seed, inputs and ``--nthreads`` give identical control sets, curves and FPR. The
  bootstrap runs one sampling chain per thread, so the control sets change with
  ``--nthreads``.
* The matching is never relaxed. If no control can be added, a chain restarts from
  scratch, up to 10 times, and then the run stops and asks for a lower
  ``--min_distance`` or a higher ``--tolerance``. Large distances with a small tolerance
  can leave too few candidates to match.
* A matched factor whose average over the genes of interest is zero or negative is
  shifted by a constant, printed in the log, because the tolerance is relative to that
  average.
* Results of version 2.0.1 and earlier are not reproduced: that version skipped factor 9,
  added 2 to factor 28, matched within two-sided limits and mirrored shuffled segments.


Outputs
-------

``--output_folder`` receives ``case_set.txt``, ``control_set.txt``, ``used_genes.txt``
(genes shuffled for the FPR), ``enrichment_curves.txt``, ``fdr.txt`` and one
``enrichment_curves_<rank table>.svg`` per rank table.

``enrichment_curves.txt`` has one row per scope and threshold: ``vip_count``
(:math:`n`), ``ctrl_mean`` (:math:`\bar{c}`), ``ctrl_ci_lo`` and ``ctrl_ci_hi`` (2.5 and
97.5 percentiles of the control counts), ``ratio`` (fold enrichment), ``ci_lo_ratio`` and
``ci_hi_ratio`` (the two percentiles plus 0.1 over :math:`\bar{c} + 0.1`), ``p_value`` and
``n_sets`` (number of control sets). The plotted confidence interval is computed from
``vip_count``, ``ctrl_ci_lo`` and ``ctrl_ci_hi`` as described in `Fold enrichment`_.

``fdr.txt`` has one row per rank table and scope:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - ``scope``
     - ``All``, a group or a population.
   * - ``p_value``
     - FPR: fraction of shuffled genomes with a sum at least as large as the real one.
       0 when none reaches it.
   * - ``p_value_perm``
     - :math:`(k + 1) / (N + 1)`, with :math:`k` shuffled genomes at or above the real sum
       out of :math:`N`; never 0.
   * - ``n_replicates``
     - Number of shuffled genomes :math:`N`.
   * - ``real_stat``
     - Sum of :math:`n - \bar{c}` over the thresholds with at least one gene of
       interest.
   * - ``max_null_stat``
     - Largest sum over the shuffled genomes.


Plotting
--------

``--facet`` draws one panel per scope, ``--plot_groups`` draws the group curves instead
of the populations, and ``--ci`` shades the 95% confidence interval. To redraw from the
saved table without rerunning the pipeline:

.. code-block:: bash

    flexsweep enrichment-plot \
        --curves enrichment_out/enrichment_curves.txt \
        --scopes All,yri \
        --facet --ci

.. code-block:: python

    from flexsweep.enrichment import plot_enrichment

    figs = plot_enrichment(
        "enrichment_out/enrichment_curves.txt",
        scopes=["All", "yri"],
        facet=True,
        output_folder="enrichment_out",
    )
