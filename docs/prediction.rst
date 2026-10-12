.. _prediction:

From predictions to candidate sweeps
====================================

``flexsweep cnn`` and ``flexsweep dann`` write one row per 1.2 Mb region, sliding every
10 kb along each chromosome, with the probability that a sweep sits at the center of
the region (``prob_sweep``; columns in :doc:`cnn`). Neighbouring regions overlap
almost entirely, so a single sweep raises ``prob_sweep`` over many consecutive rows, and
``prob_sweep`` comes from a classifier trained on as many sweeps as neutral regions: it
ranks regions but is not the genome-wide fraction of sweeps. This page turns that table
into results, in the order you would use it:

1. **Check the inputs**: before trusting the predictions, check that the empirical
   feature vectors look like the simulated ones.
2. **Look genome-wide**: Manhattan plot and per-chromosome distributions of
   ``prob_sweep``.
3. **Call candidate regions**: merge consecutive regions above a probability threshold
   into non-overlapping candidate sweeps.
4. **Assign sweeps to genes or other elements**: give each element of a BED file the
   highest ``prob_sweep`` nearby and rank the elements. Gene ranks from one or more
   populations, combined into one table, are the input of :doc:`enrichment`.

Command line and Python interface
---------------------------------

Ranking elements has a command; the other steps are Python functions in
``flexsweep.utils``.

.. code-block:: bash

    flexsweep rank \
        --prediction yri_cnn/yri_predictions.txt \
        --feature_coordinates genes.bed

.. code-block:: python

    from flexsweep.utils import (
        plot_fv_pca, plot_manhattan, merge_regions, rank_probabilities,
    )

    plot_fv_pca("yri_sims/fvs.parquet", empirical_data="yri_vcfs/fvs_yri.parquet",
                output_path="fv_pca.svg")                              # 1. check inputs
    plot_manhattan("yri_cnn/yri_predictions.txt", out="manhattan.png")   # 2. genome-wide
    df_merged, summary = merge_regions("yri_cnn/yri_predictions.txt", p=0.9)  # 3. regions
    df_ranked = rank_probabilities(                                      # 4. genes
        prediction="yri_cnn/yri_predictions.txt",
        feature_coordinates="genes.bed",
    )

1. Check the inputs
-------------------

Simulations that do not resemble the data produce unreliable predictions, whatever
the test accuracy on simulations. These two plots compare simulated and empirical
feature vectors.

Feature vector PCA
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.utils import plot_fv_pca

    fig = plot_fv_pca(
        train_data="yri_sims/fvs.parquet",
        empirical_data="yri_vcfs/fvs_yri.parquet",
        subsample=5000,
        output_path="fv_pca.svg",
    )

Projects the feature matrix onto its first two principal components, coloured
by neutral (blue) and sweep (red). Pass ``empirical_data`` to overlay
empirical regions as a third colour. If the empirical cloud falls outside the
simulated clouds, the simulations do not resemble the data (domain shift): revisit
the demographic model (:doc:`simulation`) or consider DANN training (:doc:`dann`).

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Description
   * - ``train_data``
     - required
     - Path to ``fvs*.parquet`` or a Polars DataFrame. Must have a ``model``
       column (``neutral`` / sweep label).
   * - ``empirical_data``
     - ``None``
     - Path to empirical ``fvs*.parquet`` or DataFrame (no ``model`` column).
       When provided, plotted as a third distribution.
   * - ``subsample``
     - ``5000``
     - Maximum rows to use (avoids slow PCA on large datasets).
   * - ``output_path``
     - ``None``
     - Save path (SVG). If ``None``, shows interactively.

Statistic distributions
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.utils import plot_stat_distributions

    fig = plot_stat_distributions(
        train_data="yri_sims/fvs.parquet",
        empirical_data="yri_vcfs/fvs_yri.parquet",
        stats=["pi", "h12", "ihs", "nsl", "tajima_d"],
        output_path="stat_distributions.svg",
    )

Violin plots of each statistic split by neutral, sweep, and (optionally)
empirical data. This is the primary diagnostic for identifying which
statistics are shifted between simulations and real data — a stat whose
empirical distribution is far from both the neutral and sweep simulation
distributions is a candidate to exclude from training (``stats=`` in
:doc:`feature_vectors`), or a sign that DANN training (:doc:`dann`) may help.

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Description
   * - ``train_data``
     - required
     - Path to ``fvs*.parquet`` or Polars DataFrame with ``model`` column.
   * - ``empirical_data``
     - ``None``
     - Empirical ``fvs*.parquet`` or DataFrame (no ``model`` column).
   * - ``stats``
     - all stats
     - List of statistic base names to plot, e.g.
       ``["pi", "h12", "ihs"]``. Defaults to the full Flex-sweep stat set.
   * - ``output_path``
     - ``None``
     - Save path (SVG). If ``None``, shows interactively.

2. Look genome-wide
-------------------

Manhattan plot
~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.utils import plot_manhattan

    # CNN output — plots -log10(1 - prob_sweep) by default
    plot_manhattan(
        "yri_cnn/yri_predictions.txt",
        out="manhattan.png",
    )

    # Custom column — e.g. a scan pvalue column
    plot_manhattan(
        df,
        p_col="ihs_pvalue",
        chr_col="chrom",
        pos_col="pos",
        log_transform=True,
        out="manhattan_ihs.png",
    )

Generic genome-wide Manhattan plot. Accepts CNN prediction CSV/DataFrame or
any tabular data with a genomic position and a value column.

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Description
   * - ``input``
     - required
     - File path (CSV) or Polars DataFrame.
   * - ``p_col``
     - ``None``
     - Column to use as the value. ``None`` → computes ``1 − prob_sweep``
       (CNN default behaviour).
   * - ``chr_col``
     - ``None``
     - Chromosome column name. ``None`` → ``"chr"`` (CNN default).
   * - ``pos_col``
     - ``None``
     - Position column name. ``None`` → ``"start"`` (CNN default).
   * - ``log_transform``
     - ``True``
     - Plot ``-log10(value)`` on the y-axis.
   * - ``threshold_lines``
     - ``None``
     - List of ``(y_value, linestyle, label)`` for horizontal lines. ``None``
       → CNN defaults (y = 3 solid, y = 2 dashed). Pass ``[]`` to suppress.
   * - ``figsize``
     - ``(14, 5)``
     - Figure size in inches.
   * - ``out``
     - ``None``
     - Save path. If ``None``, shows interactively.
   * - ``title``
     - ``None``
     - Plot title.

Sweep probability density
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.utils import plot_sweep_density

    fig = plot_sweep_density(
        "yri_cnn/yri_predictions.txt",
        output_path="sweep_density.svg",
    )

Per-chromosome histograms of ``prob_sweep``. Each panel shows the
distribution of sweep probability for one contig and reports the percentage
of windows above 0.5. Useful for a quick genome-wide sanity check after
prediction.

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Description
   * - ``prediction``
     - required
     - Path to prediction CSV/Parquet or Polars DataFrame. Must have columns
       ``chr``, ``start``, ``end``, ``prob_sweep``.
   * - ``output_path``
     - ``None``
     - Save path (SVG). If ``None``, shows interactively.

3. Call candidate regions
-------------------------

Merge candidate regions
~~~~~~~~~~~~~~~~~~~~~~~

Merge contiguous windows above a probability threshold into non-overlapping
candidate sweep intervals:

.. code-block:: python

    from flexsweep.utils import merge_regions

    df_merged, summary = merge_regions(
        "yri_cnn/yri_predictions.txt",
        p=0.9,
    )
    print(summary)   # per-chromosome: merged_span, total_span, pct

``merge_regions`` filters windows where ``prob_sweep > p``, merges adjacent
intervals on the same chromosome, and returns:

* ``df_merged`` — Polars LazyFrame of merged intervals (``chr``, ``start``,
  ``end``, ``prob_sweep``).
* ``summary`` — DataFrame with ``chr``, ``merged_span`` (bp in merged
  intervals), ``total_span`` (total analysed bp), and ``pct`` (fraction of
  the analysed genome above the threshold).

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Description
   * - ``prediction``
     - required
     - File path (CSV) or Polars DataFrame with columns ``chr``, ``start``,
       ``end``, ``prob_sweep``.
   * - ``p``
     - required
     - Probability threshold. Windows with ``prob_sweep > p`` are merged.

4. Assign sweeps to genes and other elements
--------------------------------------------

Rank genes and other genomic elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assign sweep probabilities to any set of genomic elements of interest listed in a
BED file (e.g. genes or enhancers), using the prediction windows closest to each
element. Available as a Python function and via the ``flexsweep rank`` CLI.

**CLI:**

.. code-block:: bash

    flexsweep rank \
        --prediction yri_cnn/yri_predictions.txt \
        --feature_coordinates genes.bed

**Python:**

.. code-block:: python

    from flexsweep.utils import rank_probabilities

    df_ranked = rank_probabilities(
        prediction="yri_cnn/yri_predictions.txt",
        feature_coordinates="genes.bed",
        k=111,
    )

To rank several populations at once, pass a folder and the population codes. The
output is one table (``gene_id`` plus one rank column per population, sorted by
``gene_id``), the input of :doc:`enrichment`:

.. code-block:: bash

    flexsweep rank \
        --prediction_dir predictions/ \
        --populations yri,ceu,chb \
        --feature_coordinates genes.bed \
        --output predictions/all_ranks_v3

.. code-block:: python

    from flexsweep.utils import find_prediction_files, rank_merge

    pops = ["yri", "ceu", "chb"]
    df_ranks, ties = rank_merge(
        find_prediction_files("predictions/", pops), pops, "genes.bed",
        output="predictions/all_ranks_v3",
    )

For each element, ``rank_probabilities`` finds the *k* nearest
prediction windows on the same chromosome (using a ``bedtools closest -k``
equivalent), then assigns the maximum ``prob_sweep`` among those windows.
Elements are returned sorted by ``prob_sweep``, highest first: for a gene BED, the
output is a ranked gene list.

The ``feature_coordinates`` BED file must have columns ``chr``, ``start``,
``end``, ``gene_id``, ``strand`` (no header, 0-based). Chromosome labels
must be numeric (``1``–``22``); the function prepends ``chr`` automatically.

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Description
   * - ``prediction``
     - required
     - CNN prediction file (``<name>_predictions.txt``) or Polars DataFrame with ``chr``, ``start``,
       ``end``, ``prob_sweep`` columns.
   * - ``feature_coordinates``
     - required
     - BED file path (str) or Polars DataFrame of genomic features.
   * - ``rank_distance``
     - ``False``
     - If ``True``, additionally rank by distance to the nearest window.
   * - ``k``
     - ``111``
     - Number of nearest prediction windows to consider per gene. Equivalent
       to ``bedtools closest -k k``.
