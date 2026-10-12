"""Compare the fetch_example VCF with an independently prepared polarized VCF.

Usage:
    python scripts/compare_example.py qs_data/vcfs/yri_chr22_example.vcf.gz \
        ~/test/yri/yri_chr22_polarized_giab.vcf.gz [chr22:20000001-25000000]

The reference file (your production YRI chr22 VCF) was prepared with a different
outgroup set / alignment and mask, so exact identity is NOT expected. Reported:
  - site overlap (sites in both files, by position + allele pair);
  - polarization agreement on shared sites (same REF = same ancestral call);
  - genotype identity on shared samples and sites, after aligning polarization
    (should be 1.0: both come from the same NYGC panel).
"""

import gzip
import sys

import numpy as np


def header_samples(path):
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                return line.rstrip("\n").split("\t")[9:]
    raise ValueError(f"{path}: no #CHROM line")


def load(path, region, samples=None):
    """Biallelic records of ``region`` (chrom:start-end, 1-based inclusive) by streaming
    the (bgzipped) VCF as text; genotypes of ``samples`` (all if None)."""
    chrom, span = region.split(":")
    start, end = (int(x) for x in span.replace(",", "").split("-"))
    allsmp = header_samples(path)
    smp = list(allsmp) if samples is None else list(samples)
    idx = [9 + allsmp.index(s) for s in smp]
    sites, gts = {}, {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            pos = int(f[1])
            if f[0] != chrom or pos < start:
                continue
            if pos > end:
                break
            if "," in f[4]:
                continue
            key = (pos, frozenset((f[3], f[4])))
            sites[key] = f[3]
            gts[key] = np.array(
                [int(a) for i in idx for a in f[i].split(":", 1)[0].replace("|", "/").split("/")],
                dtype=np.int8,
            )
    return smp, sites, gts


def main(example, reference, region="chr22:20000001-25000000"):
    smp_a, sites_a, gts_a = load(example, region)
    ref_samples = set(header_samples(reference))
    shared_smp = [s for s in smp_a if s in ref_samples]
    smp_b, sites_b, gts_b = load(reference, region, shared_smp)

    shared = sorted(set(sites_a) & set(sites_b), key=lambda k: k[0])
    print(f"samples example/shared: {len(smp_a)}/{len(shared_smp)}")
    print(f"sites example: {len(sites_a)}  reference: {len(sites_b)}  shared: {len(shared)}")
    if not shared:
        return
    same_pol = sum(sites_a[k] == sites_b[k] for k in shared)
    print(f"polarization agreement on shared sites: {same_pol / len(shared):.4f}")

    # haplotype columns of each shared sample, in the same sample order for both files
    cols_a = np.array([[2 * smp_a.index(s), 2 * smp_a.index(s) + 1] for s in shared_smp]).ravel()
    cols_b = np.array([[2 * smp_b.index(s), 2 * smp_b.index(s) + 1] for s in shared_smp]).ravel()
    n_eq = 0
    for k in shared:
        ga = gts_a[k][cols_a]
        gb = gts_b[k][cols_b]
        if sites_a[k] != sites_b[k]:
            gb = 1 - gb
        n_eq += int(np.array_equal(ga, gb))
    print(f"genotype identity on shared sites (polarization-aligned): {n_eq / len(shared):.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
