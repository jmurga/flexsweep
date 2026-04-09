CNN prediction
==============

This page covers the full lifecycle of CNN/DANN predictions: checking that
your feature vectors are reasonable before training, visualising genome-wide
sweep probabilities, and post-processing predictions into candidate regions
and gene-level rankings.

All functions are in ``flexsweep.utils``.


Prediction visualization
------------------------

Manhattan plot
~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.utils import plot_manhattan

    # CNN output — plots -log10(1 - prob_sweep) by default
    plot_manhattan(
        "yri_vcf/predictions.csv",
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
        "yri_vcf/predictions.csv",
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


Post-processing predictions
---------------------------

Merge candidate regions
~~~~~~~~~~~~~~~~~~~~~~~

Merge contiguous windows above a probability threshold into non-overlapping
candidate sweep intervals:

.. code-block:: python

    from flexsweep.utils import merge_regions

    df_merged, summary = merge_regions(
        "yri_vcf/predictions.csv",
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

Rank genomic features
~~~~~~~~~~~~~~~~~~~~~

Assign sweep probabilities to genes or other genomic features using the
nearest prediction window. Available as a Python function and via the
``flexsweep rank`` CLI.

**CLI:**

.. code-block:: bash

    flexsweep rank \
        --prediction yri_vcf/predictions.csv \
        --feature_coordinates genes.bed

**Python:**

.. code-block:: python

    from flexsweep.utils import rank_probabilities

    df_ranked = rank_probabilities(
        prediction="yri_vcf/predictions.csv",
        feature_coordinates="genes.bed",
        k=111,
    )

For each gene (or BED feature), ``rank_probabilities`` finds the *k* nearest
prediction windows on the same chromosome (using a ``bedtools closest -k``
equivalent), then assigns the maximum ``prob_sweep`` among those windows.
Genes are returned sorted by ``prob_sweep`` descending — the output is a
ranked gene list.

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
     - CNN prediction CSV or Polars DataFrame with ``chr``, ``start``,
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



Training diagnostics
--------------------

Use these two functions to inspect your feature vectors before or after
training, particularly to check for domain shift between simulations and
empirical data.

Feature vector PCA
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.utils import plot_fv_pca

    fig = plot_fv_pca(
        train_data="yri_test/fvs.parquet",
        empirical_data="yri_vcf/fvs_yri.parquet",
        subsample=5000,
        output_path="fv_pca.svg",
    )

Projects the feature matrix onto its first two principal components, coloured
by neutral (blue) and sweep (red). Pass ``empirical_data`` to overlay
empirical windows as a third colour — a large separation between the
simulation cloud and the empirical cloud indicates domain shift that may
require DANN training.

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
        train_data="yri_test/fvs.parquet",
        empirical_data="yri_vcf/fvs_yri.parquet",
        stats=["pi", "h12", "ihs", "nsl", "tajima_d"],
        output_path="stat_distributions.svg",
    )

Violin plots of each statistic split by neutral, sweep, and (optionally)
empirical data. This is the primary diagnostic for identifying which
statistics are shifted between simulations and real data — a stat whose
empirical distribution is far from both the neutral and sweep simulation
distributions is a candidate to exclude from DANN training (see ``ihs``
exclusion in CLAUDE.md for an example).

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
