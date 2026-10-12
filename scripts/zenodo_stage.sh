#!/usr/bin/env bash
# Stage the Zenodo deposit for the Flex-sweep examples (published as record 23260568).
#
# Usage (from the repository root):
#     bash scripts/zenodo_stage.sh [SRC_DIR] [OUT_DIR]
#
# SRC_DIR (default ~/fs_validation/polemp) must hold:
#     yri_chr22_hg38_high_coverage.vcf.gz  108 unrelated YRI, chr22, biallelic SNPs, from the
#                                        NYGC 30x 1000 Genomes phased panel (GRCh38)
#     GRCh38_alldifficultregions.bed.gz  GIAB v2.0 GRCh38 difficult regions (excluded)
#     447.primates.maf.gz                Zoonomia 447-mammal great-ape MAF, all chromosomes
#     maf/447.primates_chr22.maf.gz      its chr22 part, sorted (flexsweep split-maf --sort)
# OUT_DIR (default ./zenodo_flexsweep_example) receives the files to upload, md5sums.txt
# and README.md. Upload every file in OUT_DIR, then set ZENODO_RECORD and ZENODO_MD5 in
# flexsweep/example.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$(realpath -m "${1:-${HOME}/fs_validation/polemp}")"
OUT="$(realpath -m "${2:-${REPO}/zenodo_flexsweep_example}")"

VCF="yri_chr22_hg38_high_coverage.vcf.gz"
BED="GRCh38_alldifficultregions.bed.gz"
MAF="447.primates.maf.gz"
MAF22="447.primates_chr22.maf.gz"

die() { echo "ERROR: $*" >&2; exit 1; }

# 1. Inputs exist and the VCF is complete (a VCF still being written has no BGZF EOF)
for f in "${SRC}/${VCF}" "${SRC}/${BED}" "${SRC}/${MAF}" "${SRC}/maf/${MAF22}"; do
    [[ -s "${f}" ]] || die "missing or empty: ${f}"
done
if command -v lsof > /dev/null && lsof "${SRC}/${VCF}" > /dev/null 2>&1; then
    die "${SRC}/${VCF} is still open by another process (being written); wait for it"
fi
bgzip -t "${SRC}/${VCF}" || die "${SRC}/${VCF} is truncated or not bgzip-compressed"
gzip -t "${SRC}/${MAF22}" || die "${SRC}/maf/${MAF22} is corrupt"
gzip -t "${SRC}/${BED}" || die "${SRC}/${BED} is corrupt"
[[ -f "${SRC}/${VCF}.tbi" ]] || tabix -p vcf "${SRC}/${VCF}"

# 2. Content checks on the VCF: samples, chromosome, contig length, SNPs only
n_samples=$(bcftools query -l "${SRC}/${VCF}" | wc -l)
chroms=$(bcftools index -s "${SRC}/${VCF}" | cut -f1 | paste -sd,)
contig=$(bcftools view -h "${SRC}/${VCF}" | grep -c '^##contig=<ID=chr22,length=50818468' || true)
n_records=$(bcftools index -n "${SRC}/${VCF}")
n_nonsnp=$(bcftools view -H -e 'TYPE="snp" && N_ALT=1' "${SRC}/${VCF}" | wc -l)
echo "VCF: ${n_samples} samples, chromosomes ${chroms}, ${n_records} records, ${n_nonsnp} non-biallelic-SNP"
[[ "${n_samples}" -eq 108 ]] || die "expected 108 YRI samples, found ${n_samples}"
[[ "${chroms}" == "chr22" ]] || die "expected chr22 only, found ${chroms}"
[[ "${contig}" -eq 1 ]] || die "no ##contig=<ID=chr22,length=50818468> header line"
[[ "${n_nonsnp}" -eq 0 ]] || die "${n_nonsnp} records are not biallelic SNPs"

# 3. Stage the files to upload (hard links when on the same filesystem, else copies)
mkdir -p "${OUT}"
for f in "${VCF}" "${VCF}.tbi" "${BED}" "${MAF}"; do
    ln -f "${SRC}/${f}" "${OUT}/${f}" 2> /dev/null || cp -v "${SRC}/${f}" "${OUT}/${f}"
done
ln -f "${SRC}/maf/${MAF22}" "${OUT}/${MAF22}" 2> /dev/null || cp -v "${SRC}/maf/${MAF22}" "${OUT}/${MAF22}"

# 4. Checksums (flexsweep example verifies them; paste into flexsweep/example.py)
(cd "${OUT}" && md5sum "${VCF}" "${VCF}.tbi" "${BED}" "${MAF22}" "${MAF}" > md5sums.txt)
cat "${OUT}/md5sums.txt"

cat > "${OUT}/README.md" <<EOF
# Flex-sweep example data

Input data for \`flexsweep example\` (5 Mb and whole-chr22 YRI examples), so that
the examples need no other server. Coordinates are GRCh38.

| File | Content |
|------|---------|
| \`${VCF}\` (+ \`.tbi\`) | chr22 genotypes of the 108 unrelated YRI of the 1000 Genomes phase 3 release, from the NYGC 30x high-coverage phased panel (20220422 release). Biallelic SNPs only; not polarized, not masked. AC and AN are those of the 108 YRI (bcftools recomputes them when subsetting samples); AF and the other INFO fields are those of the full 3,202-sample panel. Recompute them after subsetting samples. |
| \`${BED}\` | GIAB genome stratifications v2.0, GRCh38 union of all difficult regions (unmodified). Variants inside these intervals are removed. |
| \`${MAF22}\` | chr22 of the file below, sorted by human position. |
| \`${MAF}\` | Human-anchored (GRCh38) MAF with the chimpanzee, gorilla and orangutan rows of the Zoonomia 447-mammal Cactus alignment, all chromosomes. |
| \`md5sums.txt\` | MD5 checksums. |

## How the files were made

VCF:

    bcftools view -M2 -m2 -v snps -S yri_samples.txt.gz \\
        https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz \\
        -Oz -o ${VCF}
    tabix -p vcf ${VCF}

where \`yri_samples.txt.gz\` lists the YRI individuals of
\`integrated_call_samples_v3.20130502.ALL.panel\` (1000 Genomes phase 3).

MAF:

    cactus-hal2maf ./js 447-mammalian-2022v1.hal ${MAF} \\
        --refGenome Homo_sapiens \\
        --targetGenomes Pan_troglodytes,Gorilla_gorilla,Pongo_abelii \\
        --noAncestors --onlyOrthologs --filterGapCausingDupes --chunkSize 100000
    flexsweep split-maf --maf ${MAF} --contig chr22 --sort

Reference rows are named \`Homo_sapiens.chrN\`; outgroup rows \`Pan_troglodytes.*\`,
\`Gorilla_gorilla.*\`, \`Pongo_abelii.*\`. Use them with
\`flexsweep polarize --outgroups Pan_troglodytes,Gorilla_gorilla,Pongo_abelii\`.

## Sources (cite these when using the files)

- Byrska-Bishop M. et al. (2022) High-coverage whole-genome sequencing of the expanded
  1000 Genomes Project cohort including 602 trios. Cell 185:3426-3440.
  doi:10.1016/j.cell.2022.08.004
- Zoonomia Consortium (2020) A comparative genomics multitool for scientific discovery
  and conservation. Nature 587:240-245. doi:10.1038/s41586-020-2876-6
- Armstrong J. et al. (2020) Progressive Cactus is a multiple-genome aligner for the
  thousand-genome era. Nature 587:246-251. doi:10.1038/s41586-020-2871-y
- Dwarshuis N. et al. (2024) The GIAB genomic stratifications resource for human
  reference genomes. Nat Commun 15:9029. doi:10.1038/s41467-024-53260-y
EOF

echo
echo "Upload every file in ${OUT}:"
ls -la "${OUT}"
