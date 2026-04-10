Ancestral state polarization
============================

Several statistics in Flexsweep require knowledge of the ancestral allele —
that is, which allele was present in the ancestral population before a mutation
arose. Examples include iHS, Fay-Wu H, Zeng E, Fu-Li D/F, hapdaf, DIND, and
iSAFE. Without polarization, REF is used as a proxy for the ancestral state,
which introduces error when REF is derived.

``flexsweep polarize`` assigns the ancestral allele by comparing each variant
in a VCF to one or more outgroup species in a multi-species alignment (MAF).
Where the outgroup consensus supports the alternate allele as ancestral, REF and
ALT are swapped in the output VCF and all associated fields (AC, AF, GT) are
updated accordingly.

The polarization engine is implemented in Rust for performance and is invoked
via a compiled binary (``flexsweep-polarize``).


How it works
------------

The pipeline has two steps:

**Step 1 — Sort the MAF** (if not already sorted by contig/position):

The Rust binary streams the MAF and VCF together in a merge-join. Both files
must be sorted by the same reference contig order and position. If your MAF is
unsorted, run ``sort-maf`` first.

**Step 2 — Polarize the VCF**:

For each biallelic SNP in the VCF, the binary looks up the aligned column in the
MAF for each requested outgroup. The ancestral allele is inferred by one of four
substitution models and compared to REF/ALT:

- If the inferred ancestral allele matches **REF** → no change.
- If the inferred ancestral allele matches **ALT** → REF and ALT are swapped;
  AC, AF, and all genotypes are flipped.
- If the ancestral allele cannot be determined → the site is left unchanged.

The output is a bgzipped VCF with an ``AA`` INFO field recording the inferred
ancestral allele.


Polarization methods
--------------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Method
     - Description
   * - ``parsimony``
     - Majority vote over outgroup alleles. Fast and parameter-free. Suitable
       when outgroups are closely related and substitution rate is low.
   * - ``jc``
     - Jukes–Cantor model. Assumes equal substitution rates among all four
       nucleotides. Fits one free parameter (transition rate).
   * - ``kimura``
     - Kimura two-parameter (K80) model. Distinguishes transitions from
       transversions. **Default.** Recommended for mammalian data.
   * - ``r6``
     - General time-reversible (GTR) model with six free parameters. Most
       accurate; slowest. Recommended when outgroups are distantly related.

Multiple random starts (``--nrandom``, default 10) are used for model fitting
to avoid local optima.


.. Requirements
.. ------------

.. The polarization binary is implemented in Rust. You need either:

.. - A **bioconda installation** — the binary is included automatically.
.. - A **manual build** — requires the Rust toolchain (``rustc`` and ``cargo``).
..   Run the following once before using the polarize commands:

.. .. code-block:: python

..     import flexsweep as fs
..     fs.build_rust_polarization()

.. The binary is searched in this order:

.. 1. ``FLEXSWEEP_RUST_BIN`` environment variable (explicit path).
.. 2. ``flexsweep/src/target/release/`` or ``debug/`` (local build).
.. 3. ``flexsweep-polarize`` or ``polarize`` on the system ``PATH``.

.. If none is found and ``cargo`` is available, the binary is compiled
.. automatically on first use.


Input files
-----------

**MAF** (Multiple Alignment Format):

A whole-genome alignment anchored to the reference species (the same genome
used for the VCF). Common sources: UCSC Genome Browser ``multiz`` alignments.
The file can be plain text or gzipped (``.maf.gz``).

**VCF**:

A bgzipped, sorted VCF (``*.vcf.gz``) of the focal population. Must be sorted
by the same contig order as the MAF reference.

**Outgroups**:

One or more species names as they appear in the MAF ``s`` lines (e.g.
``panTro4``, ``ponAbe2``, ``gorGor6`` for human data). Multiple outgroups
improve robustness — the model aggregates information across all specified
outgroups.


CLI reference
-------------

.. code-block:: text

    flexsweep polarize [OPTIONS]

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Option
     - Default
     - Description
   * - ``--maf PATH``
     - required
     - Input MAF file. Can be gzipped. Must be sorted by contig/position
       (or pass ``--sort`` to sort it automatically).
   * - ``--vcf PATH``
     - required
     - Input VCF file. Can be gzipped. Must be sorted by contig/position.
   * - ``--outgroups LIST``
     - required
     - Comma-separated outgroup species names, e.g.
       ``panTro4,ponAbe2,gorGor6``.
   * - ``--method STR``
     - required
     - Substitution model: ``parsimony``, ``jc``, ``kimura``, or ``r6``.
   * - ``--nrandom INT``
     - 10
     - Number of random starts for model fitting (ignored by ``parsimony``).
   * - ``--sort BOOL``
     - False
     - If True, sort the MAF by contig/position before polarizing.

Examples:

.. code-block:: bash

    # Sort MAF first, then polarize with Kimura model
    flexsweep polarize \
        --maf hg38.multiz100way.maf.gz \
        --vcf YRI.chr22.vcf.gz \
        --outgroups panTro4,ponAbe2,gorGor6 \
        --method kimura \
        --sort True

    # Use parsimony with a pre-sorted MAF
    flexsweep polarize \
        --maf hg38.multiz100way.sorted.maf.gz \
        --vcf YRI.chr22.vcf.gz \
        --outgroups panTro4,gorGor6 \
        --method parsimony


Python API
----------

.. code-block:: python

    import flexsweep as fs

    # Build the Rust binary (only needed once)
    fs.build_rust_polarization()

    # Step 1: sort MAF (skip if already sorted)
    sorted_maf = fs.run_sort_maf("hg38.multiz100way.maf.gz")

    # Step 2: polarize
    output_vcf = fs.run_polarize(
        maf=sorted_maf,
        vcf="YRI.chr22.vcf.gz",
        outgroup=["panTro4", "ponAbe2", "gorGor6"],
        method="kimura",
        nrandom=10,
        output="YRI.chr22.polarized.vcf.gz",
    )

The polarized VCF is written to ``output`` (default:
``{input}.polarized.vcf.gz`` in the same directory as the input VCF).


Using polarized VCFs with scan and feature vectors
---------------------------------------------------

Once polarized, pass the output VCF directly to any Flexsweep command that
reads VCF data. Polarization improves the accuracy of all statistics that
distinguish ancestral from derived alleles:

- **scan**: ``ihs``, ``isafe``, ``dind``, ``high_freq``, ``low_freq``,
  ``s_ratio``, ``hapdaf_o``, ``hapdaf_s``, ``fay_wu_h``, ``zeng_e``,
  ``achaz_y``, ``fuli_d``, ``fuli_f``.
- **fvs-vcf**: all SNP-level features in the feature-vector pipeline.

.. code-block:: bash

    # Scan with a polarized VCF
    flexsweep scan \
        --vcf_path polarized_vcfs/ \
        --out_prefix results/YRI \
        --stats ihs,isafe,hapdaf_o,hapdaf_s,fay_wu_h \
        --nthreads 8
