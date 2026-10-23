.. _data-preparation:

Preparing VCF data
==================

Flex-sweep detects selective sweeps from a single phased population. The input is a
focal-species polymorphism dataset in VCF/BCF format, read directly without conversion
to hap/map or ms format, from any population or species. This page describes the VCF
requirements and then shows how to pre-process the 1000 Genomes Project (1000GP) data
used throughout this documentation.

The 1000GP data are an example of VCF pre-processing: we build the per-chromosome VCF
folder of the 108 unrelated Yoruba (YRI) individuals on GRCh38, the population
analysed in :doc:`basic_usage` and in the original Flex-sweep paper. The same steps
(sample subsetting, biallelic SNP filtering, allele count estimation, polarization and
masking of difficult regions) apply to any phased VCF. Run them before
:doc:`basic_usage`, or use ``flexsweep example`` (:doc:`quickstart`) for a
ready-made subset of the same data: 50 individuals, the whole of chr22 with
``--full`` or chr22:20-25 Mb without it.

.. _vcf-requirements:

VCF requirements
----------------

``fvs-vcf``, ``scan`` and ``recombination-bins`` take a **directory** of VCF/BCF files,
one per chromosome. Each file must be:

* **Phased.** Flex-sweep estimates haplotype-based statistics (iHS, nSL, H12, HAF,
  iSAFE, among others), which require phased genotypes (``0|1``, not ``0/1``). Phase
  the data beforehand, e.g. with SHAPEIT or Beagle. Flex-sweep does not check the
  phasing: unphased genotypes are read as phased and the haplotype-based statistics
  are then invalid. Check that genotypes are separated by ``|`` with
  ``bcftools view -H in.vcf.gz | head``.
* **Polarized.** REF must be the ancestral allele and ALT the derived allele, since
  several statistics depend on the derived allele frequency (see :doc:`polarize`).
  Unpolarized data still run, but those statistics are biased by ancestral allele
  mis-specification.
* **Biallelic SNPs.** Multiallelic sites are dropped; filter other variant types
  beforehand (``bcftools view -v snps -m2 -M2``).
* **bgzip-compressed and indexed** with tabix (``.tbi``) or ``bcftools index``
  (``.csi``). Flex-sweep stops with the command to fix the file if it is plain gzip or
  has no index.
* **Any chromosome naming** (``22``, ``chr22``, ``NC_000022.11``, ``scaffold_12``,
  ``Chr2``). The chromosome is taken from the first record and matched to its
  ``##contig`` header line, whose keys may appear in any order, so unmodified 1000GP
  phase 3 headers (``##contig=<ID=22,assembly=b37,length=51304566>``) are supported.
  If the first record has no matching ``##contig`` line, the chromosome in the filename
  (``yri_chr22.vcf.gz``) is used instead.
* **Chromosome length** in the ``##contig`` line (``length=``). If it is missing,
  Flex-sweep uses the position of the last variant and warns, so the 1.2 Mb regions
  stop at the last variant rather than at the chromosome end. Add the lengths from the
  reference index with ``bcftools reheader --fai reference.fa.fai in.vcf.gz -o out.vcf.gz``.
* **On the same genome build as the recombination map.** The bundled deCODE map
  (``--recombination_map decode``) is on GRCh38; Flex-sweep warns when a VCF header
  looks like GRCh37.

.. _accessibility-mask:

Accessibility mask (optional)
-----------------------------

Mask difficult regions before running Flex-sweep (step 5 below). Masking removes the
SNPs, but each window keeps its full length, so per-bp statistics of a partly masked
window are too low: a window with 40% of its bases masked shows about 40% less
nucleotide diversity, the same direction as a sweep. Pass the same BED with
``--accessibility_mask`` (``fvs-vcf`` and ``scan``) to correct this.

* The BED lists regions to **exclude** (GIAB convention, e.g.
  ``GRCh38_alldifficultregions.bed.gz``), one file for all chromosomes; ``chr22`` and
  ``22`` are matched.
* SNPs inside the BED are dropped when the VCF is read, so an unmasked VCF is masked
  on the fly and an already masked VCF is unchanged.
* Per-bp statistics are divided by the accessible bases of each window (window
  length minus masked bases) instead of its length: ``pi``, ``theta_w`` and
  ``dist_var`` in ``fvs-vcf``; ``pi`` and ``theta_w`` (also in ``neutrality``) in
  ``scan``. A fully masked window gives a missing value.
* Statistics without a per-bp denominator are unchanged. In bp windows, haplotype
  statistics (H12, number of haplotypes, HAF) see fewer SNPs when heavily masked,
  which mimics a sweep: on YRI chr20-22 (100 kb windows), H12 is inflated 1.3x in
  windows 40-50% accessible and 15x below 10%, and matches fully accessible windows
  from 50% up. In SNP-count windows (``scan`` h12, garud, haf) a masked gap instead
  stretches the window over more sequence, which lowers H12.
  Windows below ``--min_accessible`` (default 0.6) are therefore excluded: missing
  window statistics in ``fvs-vcf``, left out of the window-statistic tables in
  ``scan``. Per-SNP statistics (iHS, nSL, iSAFE, ...) and whole-chromosome
  window statistics in ``scan`` (lassi, lassip, raisd, beta, ncd) are not filtered.
* In ``fvs-vcf``, loci with many windows below ``--min_accessible`` can exceed the 10%
  missing-feature filter and are dropped from the output (13% of loci on YRI chr20-22
  at 0.6); lower the cutoff to keep more loci.
* Simulations are not masked, so their windows are fully accessible.

Sites with missing genotype calls (``.``) are always dropped when the VCF is read,
and Flex-sweep warns when they exceed 1% of the sites. Impute missing genotypes
during phasing (Beagle, SHAPEIT) rather than relying on this filter.

Example: 1000GP YRI pre-processing
----------------------------------

The steps below build the per-chromosome YRI VCF folder analysed in
:doc:`basic_usage`. For other data, replace the 1000GP panel, the sample list, the
outgroups (step 4) and the mask (step 5) with your own.

Genome build
~~~~~~~~~~~~

All coordinates are **GRCh38**. The data come from the NYGC 30x high-coverage
1000 Genomes panel, phased with SHAPEIT4 (`Byrska-Bishop et al. 2022
<https://doi.org/10.1016/j.cell.2022.08.004>`_), and the bundled deCODE
recombination map (``--recombination_map decode``) is on GRCh38 too. Do **not**
combine the bundled map with the phase 3 release (GRCh37): positions would not
match and the recombination-stratified normalization would be wrong. Flex-sweep
warns when a VCF header looks like GRCh37 and the bundled map is in use.

Tools
~~~~~

``bcftools`` and ``tabix`` (htslib ≥ 1.10), and ``flexsweep polarize`` for the
ancestral states.

1. Download the phased panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    set -euo pipefail
    BASE=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV
    for c in $(seq 1 22); do
        f=1kGP_high_coverage_Illumina.chr${c}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
        wget -c "${BASE}/${f}" "${BASE}/${f}.tbi"
    done

2. Select the 108 unrelated YRI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the phase 3 release list of unrelated individuals (108 YRI); the 3,202-sample
panel also contains their relatives.

.. code-block:: bash

    wget https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
    awk '$2 == "YRI" {print $1}' integrated_call_samples_v3.20130502.ALL.panel > yri_samples.txt
    wc -l yri_samples.txt   # 108

3. Subset, keep biallelic SNPs, drop monomorphic sites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    mkdir -p yri_raw
    for c in $(seq 1 22); do
        f=1kGP_high_coverage_Illumina.chr${c}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
        bcftools view -S yri_samples.txt --force-samples -v snps -m2 -M2 "$f" -Ou \
            | bcftools +fill-tags -Ou -- -t AC,AN,AF \
            | bcftools view -i 'AC>0 && AC!=AN' -Oz -o "yri_raw/yri_chr${c}.vcf.gz"
        tabix -p vcf "yri_raw/yri_chr${c}.vcf.gz"
    done

``+fill-tags`` recomputes ``AC``, ``AN`` and ``AF`` for the 108 individuals; the
panel values describe all 3,202 samples. ``flexsweep polarize`` fits its
ancestral-state model on these counts, so they must match the samples you analyse.

The ``##contig`` header lines of the NYGC panel already carry ``length=``, and
naming the files by chromosome (``yri_chr22.vcf.gz``) keeps the folder readable.

4. Polarize
~~~~~~~~~~~

Flex-sweep expects REF to be the ancestral allele. Polarize each chromosome with the
great-ape outgroups of the UCSC hg38 20-way alignment (see :doc:`polarize`). The
per-chromosome MAFs are in hg38 position order, so they need no sorting. The output,
``yri_raw/yri_chrN.polarized.vcf.gz``, has REF/ALT and genotypes swapped where the
ancestral allele was ALT; variants without a usable alignment are dropped.
The same step works with a MAF extracted from a Cactus alignment. The genome-wide
human-anchored great-ape MAF from the Zoonomia 447-mammal alignment is on
:ref:`Zenodo <example-data-zenodo>` (outgroups
``Pan_troglodytes,Gorilla_gorilla,Pongo_abelii``); split and sort it per chromosome
first with ``flexsweep split-maf``, as shown in :doc:`polarize`.

.. code-block:: bash

    MAF=https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz20way/maf
    mkdir -p maf
    for c in $(seq 1 22); do
        wget -c -P maf "${MAF}/chr${c}.maf.gz"
        flexsweep polarize \
            --maf "maf/chr${c}.maf.gz" \
            --vcf "yri_raw/yri_chr${c}.vcf.gz" \
            --outgroups panTro4,gorGor3,ponAbe2 \
            --method kimura
    done

5. Mask difficult regions
~~~~~~~~~~~~~~~~~~~~~~~~~

Keep only the GIAB v2.0 GRCh38 "easy" regions, the complement of tandem repeats,
homopolymers, low-mappability regions, segmental duplications and extreme-GC
regions (80% of the genome). This is the stratification used to evaluate the NYGC
small-variant calls (Byrska-Bishop et al. 2022, STAR Methods, "Evaluation of small
variant calls").

.. code-block:: bash

    wget https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/genome-stratifications/v2.0/GRCh38/union/GRCh38_notinalldifficultregions.bed.gz
    mkdir -p yri_vcfs
    for c in $(seq 1 22); do
        bcftools view -T GRCh38_notinalldifficultregions.bed.gz \
            "yri_raw/yri_chr${c}.polarized.vcf.gz" -Oz -o "yri_vcfs/yri_chr${c}.vcf.gz"
        tabix -p vcf "yri_vcfs/yri_chr${c}.vcf.gz"
    done

``yri_vcfs/`` is now ready for ``flexsweep fvs-vcf --vcf_path yri_vcfs
--recombination_map decode`` and ``flexsweep scan --vcf_path yri_vcfs``.
The data step of ``flexsweep example`` runs these same steps (sample draw, SNP filter,
polarization with the Zoonomia great-ape MAF, GIAB mask) on chr22:20-25 Mb or, with
``--full``, the whole of chr22. It removes sites inside
``GRCh38_alldifficultregions.bed.gz`` instead of keeping
``GRCh38_notinalldifficultregions.bed.gz``; the two differ only at reference gaps,
which carry no SNPs.

.. _example-data-zenodo:

Example data on Zenodo
----------------------

The inputs of ``flexsweep example`` and the alignment used by the human analyses are
deposited on Zenodo (`record 23260568 <https://zenodo.org/records/23260568>`_):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Content
   * - ``yri_chr22_hg38_high_coverage.vcf.gz`` (+ ``.tbi``, 103 MB)
     - chr22 genotypes of the 108 unrelated YRI of the 1000 Genomes phase 3 release,
       from the NYGC 30x phased panel (20220422), biallelic SNPs only
       (``bcftools view -M2 -m2 -v snps -S``); not polarized, not masked. AC and AN
       are those of the 108 YRI, AF that of the full 3,202-sample panel;
       ``flexsweep example`` recomputes all three for the drawn samples. Downloaded by
       ``flexsweep example``.
   * - ``447.primates.maf.gz`` (2.2 GB)
     - Human-anchored (GRCh38) MAF with the chimpanzee, gorilla and orangutan rows of
       the Zoonomia 447-mammal Cactus alignment, all chromosomes, extracted with
       ``cactus-hal2maf`` as shown in :doc:`polarize`.
   * - ``447.primates_chr22.maf.gz`` (30 MB)
     - Its chr22 part, sorted by human position; downloaded by ``flexsweep example``.
   * - ``GRCh38_alldifficultregions.bed.gz`` (33 MB)
     - GIAB v2.0 GRCh38 union of all difficult regions, unmodified copy; sites inside
       are removed. Downloaded by ``flexsweep example``.

If you use these files, cite
`Byrska-Bishop et al. 2022 <https://doi.org/10.1016/j.cell.2022.08.004>`_ (NYGC panel), the
`Zoonomia Consortium 2020 <https://doi.org/10.1038/s41586-020-2876-6>`_,
`Armstrong et al. 2020 <https://doi.org/10.1038/s41586-020-2871-y>`_ (Progressive
Cactus) and `Dwarshuis et al. 2024 <https://doi.org/10.1038/s41467-024-53260-y>`_
(GIAB stratifications).
