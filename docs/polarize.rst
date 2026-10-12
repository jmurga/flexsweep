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

The polarization engine is a refactor of
`est-sfs <https://doi.org/10.1534/genetics.118.301120>`_ written in Rust and compiled
into the Python package (``flexsweep.polarize``).

Command line and Python interface
---------------------------------

.. code-block:: bash

    flexsweep polarize \
        --maf 447.primates_chr22.maf.gz \
        --vcf yri_chr22.vcf.gz \
        --outgroups Pan_troglodytes,Gorilla_gorilla,Pongo_abelii \
        --method kimura

.. code-block:: python

    import flexsweep as fs

    fs.polarize.polarize(
        "447.primates_chr22.maf.gz",
        "yri_chr22.vcf.gz",
        "Pan_troglodytes,Gorilla_gorilla,Pongo_abelii",
        method="kimura",
        nrandom=10,
        output="yri_chr22.polarized.vcf.gz",
    )

The MAF is the human-anchored great-ape MAF of the Zoonomia 447-mammal alignment
(:ref:`Zenodo <example-data-zenodo>`); any reference-anchored MAF works, e.g. a UCSC
multiz20way file with ``--outgroups panTro4,gorGor3,ponAbe2``. The VCF needs ``AC``,
``AN`` and ``AF`` for the analysed samples (:ref:`polarize-inputs`).


How it works
------------

The pipeline has two steps:

**Step 1 — Sort the MAF** (if not already sorted by contig/position):

Polarization streams the MAF and VCF together in a merge-join. Both files
must be sorted by the same reference contig order and position. If your MAF is
unsorted, run ``flexsweep split-maf --sort`` first.

**Step 2 — Polarize the VCF**:

For each biallelic SNP in the VCF, it looks up the aligned column in the
MAF for each requested outgroup. The ancestral allele is inferred by one of four
substitution models and compared to REF/ALT:

- If the inferred ancestral allele matches **REF** → no change.
- If the inferred ancestral allele matches **ALT** → REF and ALT are swapped;
  AC, AF, and all genotypes are flipped.
- If the site has no usable alignment → the variant is dropped from the output.

The output is a bgzipped VCF in which REF is the inferred ancestral allele, with its
tabix index (``.tbi``) next to it, so it can be passed directly to ``fvs-vcf`` and
``scan``. The log
reports how many variants had a usable alignment, how many were swapped and how many
were dropped.


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


Requirements
------------

``flexsweep.polarize`` is a compiled Rust extension included in the published wheels,
so nothing else needs to be installed. Building Flex-sweep from source compiles it and
needs the Rust toolchain (see :doc:`installation`).


.. _polarize-inputs:

Input files
-----------

**MAF** (Multiple Alignment Format):

A whole-genome alignment anchored to the reference species (the same genome
used for the VCF). Common sources: UCSC Genome Browser ``multiz`` alignments.
The file can be plain text or gzipped (``.maf.gz``). For human GRCh38 data the
per-chromosome files of the UCSC hg38 20-way primate alignment are a practical
choice (e.g. ``chr22.maf.gz``, 147 MB), whereas the 100-way files are much larger:
https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz20way/maf/

**MAF from a Cactus HAL alignment** (e.g. Zoonomia). For species without a UCSC
multiz alignment, extract a reference-anchored MAF with ``cactus-hal2maf``,
keeping only the focal genome and the outgroups you will pass to
``--outgroups``. For the human analyses we extracted the three great apes from the
Zoonomia 447-mammal alignment (``447-mammalian-2022v1.hal``), anchored on human
(GRCh38). This primate MAF, genome-wide and for chr22, is on
:ref:`Zenodo <example-data-zenodo>`.

.. code-block:: bash

    set -euo pipefail
    # Genome names as stored in the HAL
    halStats --genomes 447-mammalian-2022v1.hal

    cactus-hal2maf ./js 447-mammalian-2022v1.hal 447.primates.maf.gz \
        --refGenome Homo_sapiens \
        --targetGenomes Pan_troglodytes,Gorilla_gorilla,Pongo_abelii \
        --noAncestors --onlyOrthologs --filterGapCausingDupes \
        --chunkSize 100000 \
        --batchSystem singleMachine --batchCores 240 --maxLocalJobs 240 \
        --logFile 447.primates.maf.gz.log

    # One sorted MAF per chromosome (writes 447.primates_chr22.maf.gz)
    flexsweep split-maf --maf 447.primates.maf.gz --contig chr22 --sort

    flexsweep polarize \
        --maf 447.primates_chr22.maf.gz \
        --vcf yri_chr22.vcf.gz \
        --outgroups Pan_troglodytes,Gorilla_gorilla,Pongo_abelii \
        --method kimura

``--noAncestors`` drops the reconstructed ancestral genomes, ``--onlyOrthologs``
keeps only sequences orthologous to the reference, and ``--filterGapCausingDupes``
removes duplicated rows that would open gaps in the reference. Scale
``--batchCores`` and ``--maxLocalJobs`` to your machine. In this alignment the
reference rows are named ``Homo_sapiens.chrN``, so ``split-maf --contig chrN``
matches the VCF contigs directly. Other HAL files name the reference contigs
differently, e.g. by GenBank accession (``CM000663.2`` is GRCh38 chr1). Check
with ``zcat file.maf.gz | grep -m1 '^s'``, and if the names differ, rename the
VCF contigs before polarizing (``bcftools annotate --rename-chrs``) and back
afterwards.

**VCF**:

A sorted VCF of the focal population, bgzipped (``*.vcf.gz``) or plain (``*.vcf``;
plain gzip is not read). Must be sorted
by the same contig order as the MAF reference.
It must carry ``AC``, ``AN`` and ``AF`` INFO fields computed for the analysed
samples (e.g. ``bcftools +fill-tags -- -t AC,AN,AF`` after subsetting): the
ancestral-state model is fitted on these counts, and sites without them are
treated as AC = AN = 0.

**Outgroups**:

One or more assembly names as they appear in the MAF ``s`` lines. In the hg38
multiz20way alignment the great apes are ``panTro4`` (chimpanzee), ``gorGor3``
(gorilla) and ``ponAbe2`` (orangutan); in the Zoonomia 447-mammal MAF above they
are ``Pan_troglodytes``, ``Gorilla_gorilla`` and ``Pongo_abelii``. Multiple outgroups
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
       (use ``flexsweep split-maf --sort`` to extract and sort one contig).
   * - ``--vcf PATH``
     - required
     - Input VCF file. Can be gzipped. Must be sorted by contig/position.
   * - ``--outgroups LIST``
     - required
     - Comma-separated outgroup species names, e.g.
       ``panTro4,gorGor3,ponAbe2``.
   * - ``--method STR``
     - required
     - Substitution model: ``parsimony``, ``jc``, ``kimura``, or ``r6``.
   * - ``--nrandom INT``
     - 10
     - Number of random starts for model fitting (ignored by ``parsimony``).

Examples:

.. code-block:: bash

    # Per-chromosome multiz20way MAF (already in hg38 position order)
    wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz20way/maf/chr22.maf.gz
    flexsweep polarize \
        --maf chr22.maf.gz \
        --vcf YRI.chr22.vcf.gz \
        --outgroups panTro4,gorGor3,ponAbe2 \
        --method kimura

    # A whole-genome MAF must first be split by contig and sorted
    flexsweep split-maf --maf genome.maf.gz --contig chr22 --sort

    # Use parsimony with a pre-sorted MAF
    flexsweep polarize \
        --maf chr22.maf.gz \
        --vcf YRI.chr22.vcf.gz \
        --outgroups panTro4,gorGor3 \
        --method parsimony


Python API
----------

.. code-block:: python

    import flexsweep as fs

    # Optional: sort a MAF by reference contig/position (skip for UCSC per-chromosome files)
    fs.polarize.maf_sort("genome.maf.gz", "genome.sorted.maf.gz")

    # Polarize (the Rust extension is compiled when the package is built)
    fs.polarize.polarize(
        "chr22.maf.gz",
        "YRI.chr22.vcf.gz",
        "panTro4,gorGor3,ponAbe2",
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
