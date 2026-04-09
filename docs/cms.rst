Composite of Multiple Signals (CMS)
=====================================

CMS is a per-SNP Bayesian method that combines several summary statistics into a single
composite posterior probability of positive selection
`(Grossman et al. 2010) <https://doi.org/10.1126/science.1183863>`_.
Flex-sweep provides a fully standalone CMS implementation (``flexsweep/cms.py``) that
operates independently of the CNN pipeline — no neural network, no feature vector, and
no simulations are required beyond those already generated for the CNN workflow.

.. contents:: Contents
   :local:
   :depth: 2


Theoretical background
-----------------------

The core insight of CMS is that multiple signals of positive selection — extended
haplotypes, elevated derived allele frequency, reduced nucleotide diversity on the
derived background — are unlikely to occur simultaneously at a neutral SNP by chance.
A causal variant under selection will tend to score in the extreme tail of *every*
individual test at once, while a neutral SNP that happens to score highly on one test
will score near the mean on the others. Multiplying individual likelihoods therefore
amplifies the signal far more than any single test could.

Bayes factor per statistic
~~~~~~~~~~~~~~~~~~~~~~~~~~

For SNP *i* and statistic *k* with observed value :math:`s_k(i)`, the Bayes factor is:

.. math::

   \text{BF}_k(i) =
   \frac{P\!\left(s_k(i) \mid \text{selected},\, \text{DAF\_bin}(i)\right)}
        {P\!\left(s_k(i) \mid \text{neutral}\right)}

Both distributions are estimated from coalescent simulations as normalised frequency
histograms (60 equal-width bins across the observed value range).

Composite Bayes factor product
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **composite Bayes factor** (or composite score) at SNP *i* is the product of
individual Bayes factors across all *n* statistics:

.. math::

   \text{BF}_{\text{composite}}(i) = \prod_{k=1}^{n} \text{BF}_k(i)

In practice this is computed in log space for numerical stability:

.. math::

   \log \text{BF}_{\text{composite}}(i) = \sum_{k=1}^{n} \log \text{BF}_k(i)

This quantity is stored in the output column ``log_bf``. Missing values (NaN) contribute
:math:`\log \text{BF} = 0` (i.e. BF = 1), so a stat that cannot be computed at a
particular SNP is simply skipped rather than invalidating the composite score.


CMS posterior probability
~~~~~~~~~~~~~~~~~~~~~~~~~

Given a prior probability of selection :math:`\pi`, the posterior is (Grossman 2010, Eq. 3):

.. math::

   \text{CMS}(i) =
   \frac{\text{BF}_{\text{composite}}(i)\,\pi}
        {\text{BF}_{\text{composite}}(i)\,\pi + (1-\pi)}

With :math:`\pi = 0.5` (the default) this simplifies to:

.. math::

   \text{CMS}(i) = \frac{\text{BF}_{\text{composite}}(i)}{\text{BF}_{\text{composite}}(i) + 1}

This maps any composite BF to [0, 1], giving the probability that SNP *i* is under
positive selection given all observed statistics. Values near 1 indicate consistent,
strong evidence across all statistics.

Prior probability of selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The choice of :math:`\prior_p` determines the use case:

* **Genome-wide scanning** (``prior_p = 0.5``, default): no assumption on how many
  SNPs are under selection; a uniform prior puts equal weight on the two hypotheses.
  Scores are further ranked empirically via ``cms_rank``.
* **Fine-mapping / CMSlocal** (``prior_p = 1 / N_SNP``): assumes exactly one selected
  SNP in a candidate region of *N*\ :sub:`SNP` variants. This is the original
  Grossman et al. (2010) usage — identifying the causal variant *within* an
  already-detected sweep region. Set ``prior_p`` accordingly via ``--prior_p``.

selDAF stratification
~~~~~~~~~~~~~~~~~~~~~

Haplotype statistics have fundamentally different distributions depending on how common
the selected allele has become. A strongly extended haplotype at a SNP with DAF = 0.1
is very different evidence from the same haplotype length at DAF = 0.8. CMS conditions
the sweep distribution on the **observed derived allele frequency** at the test SNP,
using three frequency classes:

.. math::

   \text{sweep\_low}  &: \text{DAF} \leq 0.35  \\
   \text{sweep\_mid}  &: 0.35 < \text{DAF} \leq 0.65  \\
   \text{sweep\_high} &: \text{DAF} > 0.65

The boundaries 0.35 and 0.65 follow the broadinstitute/cms C++ reference implementation
(``combine_scores_gw.c``). The original Grossman et al. (2010) paper uses finer ten-bin
stratification (centres at 0.05, 0.15, ..., 0.95), but the three-group scheme from the
C++ implementation is computationally simpler and reproduces the key separation between
partial (low-frequency) and near-complete (high-frequency) sweeps.

For **simulation training data**, sweep replicates are grouped by their simulated final
allele frequency :math:`f_t` (``params[:, 3]`` from the discoal ``save_stats`` pickle).
This means the histogram is built from simulations whose sweep *outcome* matches the
frequency class of the test SNP, which is the correct conditioning.

Histogram construction
~~~~~~~~~~~~~~~~~~~~~~

For each statistic *k*, three sweep PMFs and one neutral PMF are constructed from
simulated per-SNP values:

1. Collect per-SNP values from the **interior** of each simulated locus (positions
   in [locus × 0.25, locus × 0.75]). The interior filter avoids biases from EHH
   statistics near locus boundaries where haplotype extension is artificially truncated.
   The CMS reference implementation uses a hard cutoff of 250 kb on each side of a 1.5
   Mb chromosome; Flex-sweep defaults to 25 % trimming on each side.

   **Note on locus length:** The original Grossman et al. papers simulate 1 Mb loci.
   Flex-sweep uses 1.2 Mb by default (the CNN training locus length). With 25 %
   trimming this gives an interior of [300 kb, 900 kb] = 600 kb, equivalent in
   coverage to the original 500 kb interior (250 kb each side of a 1 Mb locus).
   Using the same 1.2 Mb simulations for both CNN training and CMS table building is
   therefore valid — no separate simulation run is needed.

2. For statistics that are symmetric with respect to ancestral/derived encoding —
   specifically **iHS** and **nSL** — fold the values by taking the absolute value
   before histogram construction. This doubles effective sample size and ensures that
   both directions of selection (on the derived or the ancestral allele) are captured
   by the same histogram.

3. Fit a 60-bin PMF over the 0.1 %–99.9 % empirical range:

   .. math::

      h_j = \frac{\text{count of values in bin } j}{\text{total count}}

4. Apply a Laplace pseudocount to avoid zero-probability bins:
   zero entries are replaced with ``numpy.finfo(float).tiny`` ≈ 2.2 × 10\ :sup:`-308`.
   This prevents infinite log-Bayes-factors from isolated extreme observations.

Pseudocount rules from the reference C++ implementation:

+---------------------------------+-----------------------------+
| Condition                       | Bayes factor                |
+=================================+=============================+
| Both bins = 0                   | BF = 1 (no information)     |
+---------------------------------+-----------------------------+
| Sweep bin = 0, neutral bin > 0  | BF = min(BF in table)       |
+---------------------------------+-----------------------------+
| Neutral bin = 0, sweep bin > 0  | BF = max(BF in table)       |
+---------------------------------+-----------------------------+
| Stat value is NaN               | BF = 1 (skip)               |
+---------------------------------+-----------------------------+


Statistics
-----------

Original CMS statistics
~~~~~~~~~~~~~~~~~~~~~~~

The original Grossman et al. (2010) implementation uses five statistics, three of which
are **cross-population** (requiring data from at least two populations):

+------------+----------------------------------------------------------+------------------+
| Statistic  | Measures                                                 | Population req.  |
+============+==========================================================+==================+
| iHS        | Extended haplotype homozygosity ratio                    | Single           |
+------------+----------------------------------------------------------+------------------+
| ΔiHH       | Absolute iHH difference                                  | Single           |
+------------+----------------------------------------------------------+------------------+
| XP-EHH     | Cross-population haplotype homozygosity                  | **Two or more**  |
+------------+----------------------------------------------------------+------------------+
| F\ :sub:`ST`| Allele frequency differentiation                        | **Two or more**  |
+------------+----------------------------------------------------------+------------------+
| ΔdAF       | Derived allele frequency difference                      | **Two or more**  |
+------------+----------------------------------------------------------+------------------+

The statistical power of the original CMS is dominated by XP-EHH and F\ :sub:`ST`
(Fig. S3 of Grossman et al. 2010): these cross-population statistics are the most
effective for high-frequency sweeps (DAF > 50 %), while iHS and ΔiHH are more
informative for low-frequency partial sweeps. The combination of cross-population and
single-population signals provides broad coverage across sweep ages and strengths.

Flex-sweep single-population statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When only a single population is available, cross-population statistics cannot be
computed. Flex-sweep CMS replaces them with five statistics introduced by
`Lauterbur et al. (2023) <https://doi.org/10.1093/molbev/msad139>`_ that capture
complementary aspects of sweep signatures on the *derived-allele background* alone.
Together they cover the same mechanistic dimensions as the original five-stat CMS:
haplotype structure, site frequency spectrum, and nucleotide diversity patterns.

iHS — Integrated Haplotype Score
'''''''''''''''''''''''''''''''''

Proposed by `Voight et al. (2006) <https://doi.org/10.1371/journal.pbio.0040072>`_.
For a core SNP, separate all chromosomes into two haplotype groups by the ancestral (A)
and derived (D) alleles. Extend outward from the core SNP, computing EHH (extended
haplotype homozygosity) as a function of genetic distance until EHH drops below 0.05.
Integrate EHH over genetic distance to obtain :math:`\text{iHH}_A` and
:math:`\text{iHH}_D`. Then:

.. math::

   \text{iHS} = \ln\!\left(\frac{\text{iHH}_A}{\text{iHH}_D}\right)

A positive iHS indicates the derived allele resides on unusually long haplotypes — a
hallmark of incomplete sweeps. Values are normalised within genome-wide derived-allele
frequency bins (width 2 %) so that iHS is approximately N(0,1) under neutrality.

For CMS, iHS is **folded** (absolute value taken) before histogram lookup, because both
positive (derived allele favoured) and negative (ancestral allele favoured) extremes
are informative.

ΔiHH — Absolute iHH Difference
'''''''''''''''''''''''''''''''

Proposed by Grossman et al. (2010):

.. math::

   \Delta\text{iHH} = \left|\text{iHH}_A - \text{iHH}_D\right|

iHS is a ratio and can be inflated when :math:`\text{iHH}_A` is very short (neutral
SNPs linked to the causal variant). ΔiHH captures the *magnitude* of haplotype
asymmetry rather than the relative sizes, making it more robust to fluctuations in the
ancestral haplotype. ΔiHH and iHS are moderately correlated (sharing the same iHH
values) but complementary: Grossman et al. (2010) include both in the default stat set
because they capture different failure modes of each other.

**Note:** ΔiHH is intentionally excluded from the Flex-sweep CNN feature vector
(see ``CLAUDE.md``) but is explicitly recommended and included by default in CMS.

nSL — Number of Segregating Sites by Length
''''''''''''''''''''''''''''''''''''''''''''

Proposed by `Ferrer-Admetlla et al. (2014) <https://doi.org/10.1093/molbev/msu077>`_.
A haplotype-length statistic analogous to iHS but using the count of segregating sites
(rather than genetic distance) to measure haplotype extension. nSL is correlated with
iHS (Pearson r ≈ 0.7) because both are sensitive to extended haplotype length. However,
nSL is more robust to uncertainty in recombination rate estimates and performs better in
regions of very low recombination. Like iHS, nSL is folded before histogram lookup.

DIND — Derived Intra-allelic Nucleotide Diversity
''''''''''''''''''''''''''''''''''''''''''''''''''

Proposed by `Barreiro et al. (2009) <https://doi.org/10.1371/journal.pgen.1000562>`_ and extended by
`Lauterbur et al. (2023) <https://doi.org/10.1093/molbev/msad139>`_.

For a focal SNP with derived allele frequency between 0.25 and 0.95, define:

* Derived background: all chromosomes carrying the derived allele at the focal site.
* Ancestral background: all chromosomes carrying the ancestral allele.

DIND is the ratio of nucleotide diversity on the derived vs ancestral background
within a flanking window (default ±50 kb):

.. math::

   \text{DIND} = \frac{\pi_d}{\pi_a}

where :math:`\pi_d` and :math:`\pi_a` are the mean pairwise nucleotide diversity on
the derived and ancestral backgrounds respectively. Under a selective sweep, diversity
on the derived background is reduced (hitchhiking), so DIND < 1. DIND has high
independence from iHS because it measures diversity patterns rather than haplotype
structure.

lowfreq — Low-frequency Allele Excess on the Derived Background
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Introduced by `Lauterbur et al. (2023) <https://doi.org/10.1093/molbev/msad139>`_.

For each flanking variant *i* within a window around the focal SNP, compute the
frequency of the *derived* allele of variant *i* among chromosomes carrying the derived
focal allele:

.. math::

   f^\text{dd}_i = \frac{\text{chromosomes with derived focal AND derived flanking allele } i}
                        {\text{total chromosomes with derived focal allele}}

lowfreq is calculated over all flanking variants where :math:`f^\text{dd}_i < 0.25`:

.. math::

   \text{lowfreq} = \frac{1}{k} \sum_{i=1}^{k} \left(1 - f^\text{dd}_i\right)^2

This statistic exploits the expected excess of rare variants on the derived background
after a sweep (analogous to Tajima's D in the negative direction), but conditioned
on chromosomes carrying the derived focal allele only. Standardised genome-wide within
derived allele frequency bins so no theoretical null model is required.

highfreq — High-frequency Allele Excess on the Derived Background
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Introduced by `Lauterbur et al. (2023) <https://doi.org/10.1093/molbev/msad139>`_.

Computed over flanking variants where :math:`f^\text{dd}_i > 0.25`:

.. math::

   \text{highfreq} = \frac{1}{k} \sum_{i=1}^{k} \left(f^\text{dd}_i\right)^2

Analogous to Fay & Wu's H on the derived background: a sweep drives some derived
flanking alleles to high frequency along with the selected allele. highfreq detects
this excess of high-frequency alleles on the derived haplotype background.

Sratio — Segregating Sites Ratio
'''''''''''''''''''''''''''''''''

Introduced by `Lauterbur et al. (2023) <https://doi.org/10.1093/molbev/msad139>`_.

For a focal SNP with derived allele frequency between 0.25 and 0.95, let :math:`S_d`
be the number of variable sites within a flanking window on the derived background and
:math:`S_a` the number on the ancestral background:

.. math::

   \text{Sratio} = \frac{S_d}{S_a}

Under a selective sweep, :math:`S_d < S_a` because hitchhiking reduces variation near
the selected site. Sratio < 1 therefore indicates a sweep. Unlike DIND (which weights
by pairwise diversity), Sratio treats each variable site equally and is particularly
sensitive to incomplete and soft sweeps where diversity is only partially reduced.

hapDAF — Haplotype-derived Allele Frequency
''''''''''''''''''''''''''''''''''''''''''''

Introduced by `Lauterbur et al. (2023) <https://doi.org/10.1093/molbev/msad139>`_.

For a focal SNP (frequency 0.25–0.95), identify flanking variants *i* that satisfy:

* Present on at least one derived background and one ancestral background chromosome.
* More common on the derived background: :math:`f_{di} > f_{ai}`.
* Frequency on ancestral background below threshold :math:`T_a`.
* Total combined frequency above threshold :math:`T_\text{tot}`.

Frequencies are defined as:

.. math::

   f_{di} = \frac{\text{derived background chromosomes carrying variant } i}
                 {\text{total derived background chromosomes}}

   f_{ai} = \frac{\text{ancestral background chromosomes carrying variant } i}
                 {\text{total ancestral background chromosomes}}

The hapDAF score over the *k* qualifying variants is:

.. math::

   \text{hapDAF} = \frac{1}{k} \sum_{i=1}^{k} \left(f_{di}^2 - f_{ai}^2\right)

Two threshold calibrations are provided:

**hapDAF-o** (old/incomplete sweeps): :math:`T_a < 0.25`,
:math:`T_\text{tot} > 0.25`. Permissive thresholds detect alleles shared between
derived and ancestral backgrounds, which is typical of incomplete or standing-variation
sweeps where the selected haplotype has not yet outcompeted all ancestral chromosomes.

**hapDAF-s** (standing variation / recent sweeps): :math:`T_a < 0.10`,
:math:`T_\text{tot} > 0.10`. Stricter thresholds focus on variants nearly exclusive to
the derived background, characteristic of more complete sweeps from standing variation.

When *k* = 0 (no qualifying variants), hapDAF = 0. Both flavours are standardised
within derived allele frequency bins.


Statistical independence and the CMS approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CMS assumes **conditional independence** of the component statistics: the joint
likelihood factorises into the product of marginal likelihoods. This is an
approximation — in practice the statistics are pairwise correlated, particularly
under selection. Grossman et al. (2010) measured pairwise correlations in neutral
regions (their Fig. S2) and found:

+------------------+------------------+----------------------------+
| Stat 1           | Stat 2           | Correlation (neutral)      |
+==================+==================+============================+
| iHS              | ΔiHH             | −0.335                     |
+------------------+------------------+----------------------------+
| iHS              | ΔdAF             | −0.603                     |
+------------------+------------------+----------------------------+
| iHS              | F\ :sub:`ST`     | −0.339                     |
+------------------+------------------+----------------------------+
| XP-EHH           | F\ :sub:`ST`     |  0.279                     |
+------------------+------------------+----------------------------+
| F\ :sub:`ST`     | ΔdAF             |  0.382                     |
+------------------+------------------+----------------------------+
| XP-EHH           | iHS              |  0.04                      |
+------------------+------------------+----------------------------+

Crucially, correlations are substantially *smaller* in regions under selection than in
neutral regions. The independence approximation therefore becomes *more* accurate
precisely where it matters most — at genuine sweep loci. The product of near-independent
BFs at a causal variant nevertheless provides extreme composite scores.

For the single-population Flex-sweep statistics:

* **iHS / ΔiHH**: moderate correlation (share iHH values); include both following Grossman et al.
* **iHS / nSL**: correlation ≈ 0.7; nSL optional — adds limited independent information but
  improves robustness in regions of uncertain recombination.
* **DIND**: high independence from haplotype statistics (measures diversity, not haplotype
  extension); strongly recommended.
* **hapDAF-o**: moderate correlation with iHS (both sensitive to haplotype structure near
  the focal SNP); recommended because it captures an additional dimension (frequency of
  shared flanking variants).
* **hapDAF-s / lowfreq / highfreq / Sratio**: each provides a different view of the SFS
  and diversity on the derived background; together they resemble the information captured
  by the cross-population F\ :sub:`ST` and ΔdAF in the original CMS.
* **iSAFE**: is itself a composite score partly derived from iHS; including it would
  double-count iHS signal and is excluded from the default stat set.

The default stat set ``["ihs", "delta_ihh", "dind", "hapdaf_o"]`` has been chosen to
maximise independence while retaining broad coverage of sweep mechanisms.


Comparison: original CMS vs Flex-sweep CMS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+------------------------------+--------------------------------------+
| Aspect                    | Grossman et al. 2010         | Flex-sweep CMS                       |
+===========================+==============================+======================================+
| Default stats             | iHS, ΔiHH, XP-EHH,          | iHS, ΔiHH, DIND, hapDAF-o           |
|                           | F\ :sub:`ST`, ΔdAF           |                                      |
+---------------------------+------------------------------+--------------------------------------+
| Cross-pop stats           | Yes (XP-EHH, FST, ΔdAF)     | No — single population only         |
+---------------------------+------------------------------+--------------------------------------+
| Simulator                 | cosi2 (demographic fitting)  | discoal (demes YAML, hard/soft/      |
|                           |                              | incomplete sweeps)                   |
+---------------------------+------------------------------+--------------------------------------+
| selDAF bins               | 10 bins (0.05–0.95, Δ0.10)  | 3 groups (low/mid/high)              |
+---------------------------+------------------------------+--------------------------------------+
| Interior filter           | 250 kb each side (1.5 Mb)   | 25 % trim (default 1.2 Mb locus)    |
+---------------------------+------------------------------+--------------------------------------+
| Causal vs linked          | Separate distributions       | All interior sweep SNPs merged       |
+---------------------------+------------------------------+--------------------------------------+
| Genome-wide prior         | :math:`\pi = 0.5`            | :math:`\pi = 0.5` (default)         |
+---------------------------+------------------------------+--------------------------------------+
| Fine-mapping prior        | :math:`\pi = 1/N_\text{SNP}` | :math:`\pi = 1/N_\text{SNP}`        |
|                           |                              | via ``--prior_p``                    |
+---------------------------+------------------------------+--------------------------------------+
| Recombination correction  | Not standard                 | Optional r-bin stratification        |
+---------------------------+------------------------------+--------------------------------------+
| Output                    | Empirical P-value ranking    | Posterior + empirical rank           |
+---------------------------+------------------------------+--------------------------------------+

Without cross-population statistics, Flex-sweep CMS has lower power than the original
CMS for *completed* high-frequency sweeps, where F\ :sub:`ST` and XP-EHH provide the
strongest signal. For *partial* and *soft* sweeps (the primary design targets of
Flex-sweep), single-population statistics including hapDAF, DIND, lowfreq, and highfreq
are highly informative, and CMS performance approaches that of the cross-population
version.


CMSlocal vs CMS genome-wide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Grossman et al. papers describe two distinct modes of CMS with different goals,
backgrounds, and priors:

**CMSlocal** (Grossman et al. 2010, original paper):

* **Goal**: fine-map the causal variant *within* an already-detected candidate sweep
  region.
* **Background**: neutral SNPs from the **same population within 500 kb** of each
  candidate region (local neutral distribution, not genome-wide).
* **Prior**: :math:`\pi = 1/N_\text{SNP}` — uniform over all *N*\ :sub:`SNP` SNPs
  in the candidate region. This encodes the assumption that **exactly one** SNP in the
  region is selected.
* **Output**: per-SNP posterior P(selected | data), interpretable as a fine-mapping
  posterior within the region.
* **Use case**: you already know there is a sweep in a gene/region and want to identify
  which specific variant is the driver.

**CMS genome-wide** (Grossman et al. 2013, empirical application):

* **Goal**: rank all SNPs across the genome against a genome-wide neutral background
  to discover new sweep candidates.
* **Background**: neutral SNPs from **genome-wide neutral regions** — the same
  simulation-derived PMFs described above.
* **No explicit posterior**: Grossman et al. (2013) use the product of Bayes factors
  directly (the ``log_bf`` column) for ranking, without converting to a posterior.
  Candidate loci are identified by empirical percentile rather than by a probability
  threshold.
* **No prior assumption on number of selected SNPs** — the ranking is agnostic to how
  many sweep loci exist in the genome.
* **Use case**: an unbiased genome-wide screen with no pre-existing candidates.

**Flex-sweep implementation** uses the genome-wide approach (``prior_p = 0.5`` default)
because it is designed as a genome-wide discovery tool. The ``log_bf`` output (sum of
log Bayes factors) is the quantity used for ranking. The ``cms`` column converts this to
a posterior using the chosen ``--prior_p`` and is useful for threshold-based filtering.
For CMSlocal fine-mapping within a candidate region, pass ``--prior_p`` equal to
``1 / N_SNP`` for the region of interest.


Full pipeline
--------------

Flex-sweep CMS operates as an optional second-pass analysis layer after the core
simulation and feature-vector pipeline. There are two input modes:

**Option A (fast)** — pre-computed ``save_stats`` pickles from ``fvs-discoal`` and
``fvs-vcf`` with the ``--save_stats`` flag. No extra computation at all; the per-SNP
statistics were already computed during the CNN pipeline and are reused directly.

**Option A2 (fallback)** — raw simulation and VCF directories. Per-SNP statistics are
computed from scratch using the same functions as ``fv.py``, but without any window
aggregation or CNN feature-vector collapsing. Results are cached to disk as pickles for
faster reruns.

Step 1 — generate per-SNP statistics (Option A)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the standard Flex-sweep simulation and feature-vector pipeline with the
``--save_stats`` flag to produce per-SNP statistics pickles alongside the normal
``fvs.parquet`` feature matrix:

.. code-block:: bash

    # Simulations (neutral + sweep)
    flexsweep simulator \
        --model flexsweep/data/yri_spiedel_2019_full.yaml \
        --n_neutral 5000 --n_sweep 5000 \
        --data_dir sims/

    # Feature vectors + save per-SNP stats pickle
    flexsweep fvs-discoal \
        --data_dir sims/ \
        --save_stats \
        --out sims/

    # VCF feature vectors + save per-SNP stats pickle
    flexsweep fvs-vcf \
        --vcf_dir data/ \
        --save_stats \
        --out data/

This produces ``sims/raw_statistics.pickle`` (simulation statistics) and
``data/raw_statistics.pickle`` (VCF statistics) that CMS reads directly.

Step 2 — run CMS (CLI)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Default stat set: ihs, delta_ihh, dind, hapdaf_o
    flexsweep cms \
        --sims sims/raw_statistics.pickle \
        --vcf  data/raw_statistics.pickle \
        --out  results/yri_chr22

    # Custom stat set
    flexsweep cms \
        --sims  sims/raw_statistics.pickle \
        --vcf   data/raw_statistics.pickle \
        --stats ihs,delta_ihh,nsl,dind,hapdaf_o,hapdaf_s \
        --out   results/yri_chr22

    # Reuse prebuilt tables (much faster for multiple VCF files)
    flexsweep cms \
        --sims   sims/raw_statistics.pickle \
        --vcf    data/raw_statistics.pickle \
        --tables sims/cms_tables.pickle \
        --out    results/yri_chr22

    # Option A2: compute from raw directories (slower, cached on first run)
    flexsweep cms \
        --sims-dir sims/ \
        --vcf-dir  data/ \
        --out      results/yri_chr22

    # With recombination bin stratification
    flexsweep cms \
        --sims              sims/raw_statistics.pickle \
        --vcf               data/raw_statistics.pickle \
        --recombination_map flexsweep/data/decode_sexavg_2019.txt.gz \
        --r_bins            0.5,1,2,3,5 \
        --out               results/yri_chr22_rbins

Step 3 — inspect output
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import polars as pl

    df = pl.read_csv("results/yri_chr22.cms.txt", separator="\t")
    print(df.head(5))
    # ┌────────────────────┬───────────┬──────────┬───────────┬──────────┬───────────┐
    # │ region             │ positions │ daf      │ log_bf   │ cms      │ cms_rank  │
    # ╞════════════════════╪═══════════╪══════════╪═══════════╪══════════╪═══════════╡
    # │ chr22:0-1200000    │ 16062157  │ 0.312    │  4.821    │ 0.992    │ 0.9987    │
    # └────────────────────┴───────────┴──────────┴───────────┴──────────┴───────────┘

    # Top candidates
    top = df.sort("cms", descending=True).head(20)

    # Parquet version for larger datasets
    df_par = pl.read_parquet("results/yri_chr22.cms.parquet")


Python API
-----------

Build likelihood tables
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.cms import build_cms_tables, load_pickle

    # Build tables from sims pickle
    tables = build_cms_tables(
        sims_pickle="sims/raw_statistics.pickle",
        stats=["ihs", "delta_ihh", "dind", "hapdaf_o"],
        n_bins=60,
    )

    # tables is a nested dict:
    #   {stat: {"bin_edges": array, "neut": pmf, "sweep_low": pmf,
    #           "sweep_mid": pmf, "sweep_high": pmf}}

    # Optionally save for reuse:
    from flexsweep.fv import save_pickle
    save_pickle("sims/cms_tables.pickle", tables)

Score per-SNP observations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.cms import cms_score, build_cms_tables, load_pickle

    tables = load_pickle("sims/cms_tables.pickle")
    vcf_data = load_pickle("data/raw_statistics.pickle")

    for region_key, summary in vcf_data.items():
        snps_df = summary.stats["snps"]
        result = cms_score(
            snps_df,
            tables,
            stats=["ihs", "delta_ihh", "dind", "hapdaf_o"],
            prior_p=0.5,
        )
        print(result.head(3))
        # columns: positions, daf, log_bf, cms

Full pipeline in Python
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.cms import run_cms

    df = run_cms(
        sims_pickle="sims/raw_statistics.pickle",
        vcf_pickle="data/raw_statistics.pickle",
        stats=["ihs", "delta_ihh", "dind", "hapdaf_o"],
        n_bins=60,
        prior_p=0.5,
        cms_tables_path="sims/cms_tables.pickle",  # saved/loaded automatically
        locus_length=int(1.2e6),
        out_prefix="results/yri_chr22",
    )
    # df has columns: region, positions, daf, log_bf, cms, cms_rank

With recombination bins
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.cms import run_cms

    df = run_cms(
        sims_pickle="sims/raw_statistics.pickle",
        vcf_pickle="data/raw_statistics.pickle",
        stats=["ihs", "delta_ihh", "dind", "hapdaf_o"],
        recombination_map="flexsweep/data/decode_sexavg_2019.txt.gz",
        r_bins=[0.5, 1.0, 2.0, 3.0, 5.0],
        out_prefix="results/yri_chr22_rbins",
    )

When ``r_bins`` is provided, separate histograms are built for each recombination rate
interval. SNPs are scored against the histogram corresponding to their local
recombination rate (falling back to the ``"all"`` aggregate histogram for bins with
insufficient data). This corrects for the dependency of iHS, ΔiHH, and nSL on local
recombination rate.


Implementation details
-----------------------

Data structures
~~~~~~~~~~~~~~~

CMS reads from the ``save_stats`` pickles produced by ``fvs-discoal`` and ``fvs-vcf``.
The two pickles have slightly different internal structures:

**Simulations pickle** (``sims/raw_statistics.pickle``):

.. code-block:: text

    {
      "neutral": summaries(
          stats   = [{"snps": df_rep1, "window": df_rep1}, ...],  # list — one per replicate
          parameters = params_array  # shape (n_reps, 6): [s, t, f_i, f_t, mu, r]
      ),
      "sweep": summaries(
          stats   = [{"snps": df_rep1, ...}, ...],
          parameters = params_array
      )
    }

The final allele frequency :math:`f_t` = ``params[:, 3]`` is used to assign each sweep
replicate to the correct selDAF class when building the sweep histograms.

**VCF pickle** (``data/raw_statistics.pickle``):

.. code-block:: text

    {
      "chr22:0-1200000": summaries(
          stats      = {"snps": df, "window": df},  # dict — one per region
          parameters = None
      ),
      ...
    }

The ``snps`` DataFrame has columns:
``iter``, ``positions``, ``daf``, ``ihs``, ``delta_ihh``, ``nsl``, ``dind``,
``high_freq``, ``low_freq``, ``s_ratio``, ``hapdaf_o``, ``hapdaf_s``, ``isafe``.

Recombination bin stratification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``--recombination_map`` and ``--r_bins`` are provided, histograms are built and
scoring is conditioned on local recombination rate. Both paths use identical interval
label strings via ``pl.Series.cut``, ensuring consistency between simulation and VCF
paths.

*Simulations path:* Each replicate's recombination rate is stored in
``params[:, -1]`` (the last column of the parameters array, in units of
:math:`r` per bp). This is converted to cM/Mb as ``r × 10\ :sup:`8``` and then
assigned to a bin using ``pl.Series.cut(breaks=r_bins)``.

*VCF path:* For each genomic region, a 100 kb sliding window grid is constructed and
``get_cm`` is called to compute cM/Mb per window from the provided recombination map.
Each SNP is then assigned to the nearest window's recombination bin via
``snps_to_r_bins``. This matches the treatment in ``_process_vcf`` in ``fv.py``.

The resulting table structure when recombination bins are enabled:

.. code-block:: python

    tables = {
        "all":       {stat: {"bin_edges": ..., "neut": ..., ...}, ...},  # aggregate fallback
        "(-inf, 0.5]":  {stat: {...}, ...},
        "(0.5, 1.0]":   {stat: {...}, ...},
        "(1.0, 2.0]":   {stat: {...}, ...},
        ...
    }

Genome-wide ranking
~~~~~~~~~~~~~~~~~~~

After scoring all SNPs across all VCF regions, the ``log_bf`` values are ranked using
an empirical percentile transformation:

.. math::

   \text{cms\_rank}(i) = \frac{\text{rank}(i) - 0.5}{N}

where rank is computed without ties (``rank_with_duplicates`` from ``fv.py``) and *N*
is the total number of scored SNPs. This converts absolute log composite BF values into a
genome-wide relative ranking in [0, 1]. The highest-ranking SNPs are the most outlying
relative to the genome-wide distribution and are the primary candidates for follow-up.

Output columns
~~~~~~~~~~~~~~

+--------------+--------------------------------------------------------------+
| Column       | Description                                                  |
+==============+==============================================================+
| ``region``   | VCF locus key (e.g. ``chr22:0-1200000``)                     |
+--------------+--------------------------------------------------------------+
| ``positions``| SNP genomic position (bp)                                    |
+--------------+--------------------------------------------------------------+
| ``daf``      | Derived allele frequency                                     |
+--------------+--------------------------------------------------------------+
| ``log_bf``  | Log composite Bayes factor (sum of log BF\ :sub:`k`)         |
+--------------+--------------------------------------------------------------+
| ``cms``      | CMS posterior probability [0, 1]                             |
+--------------+--------------------------------------------------------------+
| ``cms_rank`` | Genome-wide empirical percentile rank [0, 1]                 |
+--------------+--------------------------------------------------------------+

Both ``{prefix}.cms.txt`` (tab-separated) and ``{prefix}.cms.parquet`` are written.


All available options
----------------------

.. code-block:: text

    flexsweep cms --help

    Options:
      --sims TEXT              Path to raw_statistics.pickle from fvs-discoal --save_stats
      --vcf TEXT               Path to raw_statistics.pickle from fvs-vcf --save_stats
      --sims-dir TEXT          Raw simulation directory (Option A2 fallback)
      --vcf-dir TEXT           Raw VCF directory (Option A2 fallback)
      --stats TEXT             Comma-separated per-SNP stat names.
                               Available: ihs, delta_ihh, nsl, isafe, dind,
                               high_freq, low_freq, s_ratio, hapdaf_o, hapdaf_s.
                               [default: ihs,delta_ihh,dind,hapdaf_o]
      --n_bins INTEGER         Histogram bins per stat per class. [default: 60]
      --prior_p FLOAT          Prior P(selection). Default 0.5 (genome-wide mode).
                               Use 1/N_SNP for within-region fine-mapping.
      --tables TEXT            Path to prebuilt cms_tables.pickle (saves/loads to
                               skip table building).
      --locus_length INTEGER   Simulation locus length in bp for interior-position
                               filtering.  [default: 1200000]
      --recombination_map TEXT Recombination map TSV (chr, end, cm columns).
                               Required when --r_bins is provided.
      --r_bins TEXT            Comma-separated r-rate bin breaks in cM/Mb
                               (e.g. '0.5,1,2,3,5'). Requires --recombination_map.
      --out TEXT               Output file prefix.  [required]
      --nthreads INTEGER       [default: 1]


Verification
-------------

Toy test with analytically known Bayes factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import scipy.stats
    import polars as pl
    from flexsweep.cms import cms_score

    # Build histograms analytically: neutral ~ N(0,1), sweep ~ N(3,1)
    edges = np.linspace(-5, 8, 61)
    neut_hist  = np.diff(scipy.stats.norm(0, 1).cdf(edges))
    sweep_hist = np.diff(scipy.stats.norm(3, 1).cdf(edges))
    neut_hist[neut_hist == 0] = np.finfo(float).tiny
    sweep_hist[sweep_hist == 0] = np.finfo(float).tiny

    tables_toy = {
        "ihs": {
            "bin_edges": edges,
            "neut": neut_hist,
            "sweep_low": sweep_hist,
            "sweep_mid": sweep_hist,
            "sweep_high": sweep_hist,
        }
    }

    # iHS = 3.0 (right at the sweep mean): log(BF) ≈ log[N(3;3,1)/N(3;0,1)]
    # = log(0.399) - log(0.004) ≈ 4.5
    snp = pl.DataFrame({"positions": [1000], "daf": [0.5], "ihs": [3.0]})
    r = cms_score(snp, tables_toy, stats=["ihs"])
    assert abs(r["log_bf"][0] - 4.5) < 0.3, f"Got {r['log_bf'][0]}"

    # iHS = 0.0 (at neutral mean): BF ≈ 1 → log_BF ≈ 0
    snp0 = pl.DataFrame({"positions": [1000], "daf": [0.5], "ihs": [0.0]})
    r0 = cms_score(snp0, tables_toy, stats=["ihs"])
    assert abs(r0["log_bf"][0]) < 0.5

Mathematical property tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from flexsweep.cms import build_cms_tables
    from flexsweep.fv import load_pickle
    import numpy as np

    tables = build_cms_tables(
        "sims/raw_statistics.pickle",
        stats=["ihs", "delta_ihh", "dind", "hapdaf_o"]
    )

    # Each PMF must sum to 1 and have no zeros (pseudocounts)
    for stat, t in tables.items():
        assert abs(t["neut"].sum() - 1.0) < 1e-9
        assert (t["neut"] > 0).all()
        assert not np.allclose(t["neut"], t["sweep_high"])

    # Missing stat → all BFs = 1 → composite BF = 1 → CMS = 0.5 (with prior 0.5)
    import polars as pl
    snp_nan = pl.DataFrame({
        "positions": [1000], "daf": [0.5],
        "ihs": [float("nan")], "dind": [float("nan")]
    })
    r = cms_score(snp_nan, tables, stats=["ihs", "dind"])
    assert abs(r["cms"][0] - 0.5) < 0.01


References
-----------

**CMS methods:**

* Grossman, S.R., Shlyakhter, I., Karlsson, E.K. *et al.* (2010).
  A composite of multiple signals distinguishes causal variants in regions of positive
  selection. *Science*, 327(5967), 883–886.
  https://doi.org/10.1126/science.1183863

* Grossman, S.R., Andersen, K.G., Shlyakhter, I. *et al.* (2013).
  Identifying recent adaptations in large-scale genomic data. *Cell*, 152(4), 703–713.
  https://doi.org/10.1016/j.cell.2013.01.035

* Ma, Y., Zhu, M., Ye, L. *et al.* (2015).
  A new composite of multiple signals for detecting signatures of natural selection.
  *Heredity*, 115, 426–436.

**Flex-sweep statistics:**

* Lauterbur, M.E., Munch, K. and Enard, D. (2023).
  Versatile detection of diverse selective sweeps with Flex-sweep.
  *Molecular Biology and Evolution*, 40(6).
  https://doi.org/10.1093/molbev/msad139

* Barreiro, L.B., Ben-Ali, M., Quach, H. *et al.* (2009).
  Evolutionary dynamics of human Toll-like receptors and disease susceptibility.
  *PLOS Genetics*, 5(7), e1000562.
  https://doi.org/10.1371/journal.pgen.1000562

**Component statistics:**

* Voight, B.F., Kudaravalli, S., Wen, X. and Pritchard, J.K. (2006).
  A map of recent positive selection in the human genome.
  *PLOS Biology*, 4(3), e72.
  https://doi.org/10.1371/journal.pbio.0040072

* Ferrer-Admetlla, A., Liang, M., Korneliussen, T. and Nielsen, R. (2014).
  On detecting incomplete soft or hard selective sweeps using haplotype structure.
  *Molecular Biology and Evolution*, 31(5), 1275–1291.
  https://doi.org/10.1093/molbev/msu077

* Akbari, A., Vitti, J.J., Iranmehr, A. *et al.* (2018).
  Identifying the favored mutation in a positive selective sweep.
  *Nature Methods*, 15(4), 279–282.
  https://doi.org/10.1038/nmeth.4606

* Kern, A.D. and Schrider, D.R. (2016).
  discoal: flexible coalescent simulations with selection.
  *Bioinformatics*, 32(24), 3839–3841.
  https://doi.org/10.1093/bioinformatics/btw556
