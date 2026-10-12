"""polarize must swap REF/ALT according to the inferred ancestral allele, whichever
allele is the major one.

Regression: est_sfs.rs decided "keep REF if P(major ancestral) >= 0.5", treating REF as
the major allele, so every site where ALT was the more frequent allele was polarized the
wrong way (13% of sites, ~99% of all errors in the polarization simulations).

Oracle: synthetic sites with a known ancestral allele that all three outgroups carry;
REF is randomly the ancestral or the derived allele, and the derived allele is the major
allele at some sites (derived frequency > 0.5).
"""

import gzip
import os

import numpy as np
import pytest

from flexsweep import polarize

BASES = "ACGT"
OUTGROUPS = ("sister", "outgroup1", "outgroup2")


def _write_inputs(tmp, n_sites=400, n_hap=40, seed=1):
    rng = np.random.default_rng(seed)
    L = n_sites * 10 + 100
    ref_seq = list(rng.choice(list(BASES), size=L))
    sites = []
    for i in range(n_sites):
        pos = 50 + i * 10  # 0-based
        anc = ref_seq[pos]
        der = rng.choice([b for b in BASES if b != anc])
        # mostly low-frequency derived alleles (to fit the uSFS), plus derived-major sites
        k = int(rng.integers(n_hap // 2 + 2, n_hap - 1)) if i % 5 == 0 else int(rng.integers(1, n_hap // 3))
        ref_is_anc = bool(rng.integers(0, 2))
        ref, alt = (anc, der) if ref_is_anc else (der, anc)
        hap = np.zeros(n_hap, dtype=int)
        hap[rng.choice(n_hap, size=k, replace=False)] = 1  # 1 = carries derived
        gt = hap if ref_is_anc else 1 - hap  # 1 = carries ALT
        sites.append((pos, ref, alt, anc, gt))
    focal_seq = ref_seq.copy()
    for pos, ref, *_ in sites:
        focal_seq[pos] = ref  # reference genome carries REF

    vcf = os.path.join(tmp, "focal.vcf")
    with open(vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr1,length=%d>\n" % L)
        fh.write('##INFO=<ID=AC,Number=A,Type=Integer,Description="AC">\n')
        fh.write('##INFO=<ID=AN,Number=1,Type=Integer,Description="AN">\n')
        fh.write('##INFO=<ID=AF,Number=A,Type=Float,Description="AF">\n')
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 + "\t".join(f"i{j}" for j in range(n_hap // 2)) + "\n")
        for pos, ref, alt, _, gt in sites:
            ac = int(gt.sum())
            fh.write(f"chr1\t{pos + 1}\t.\t{ref}\t{alt}\t.\tPASS\t"
                     f"AC={ac};AN={n_hap};AF={ac / n_hap:.4f}\tGT\t"
                     + "\t".join(f"{gt[2 * j]}|{gt[2 * j + 1]}" for j in range(n_hap // 2))
                     + "\n")
    polarize.bgzip_index(vcf, vcf + ".gz")

    maf = os.path.join(tmp, "alignment.maf.gz")
    with gzip.open(maf, "wt") as fh:
        fh.write("##maf version=1\n\na score=0\n")
        fh.write(f"s focal.chr1 0 {L} + {L} {''.join(focal_seq)}\n")
        for o in OUTGROUPS:  # outgroups carry the ancestral sequence
            fh.write(f"s {o}.chr1 0 {L} + {L} {''.join(ref_seq)}\n")
        fh.write("\n")
    return vcf + ".gz", maf, {pos + 1: (anc, int(gt.sum()) > n_hap - int(gt.sum()))
                              for pos, _, _, anc, gt in sites}


@pytest.mark.parametrize("method", ["parsimony", "jc", "kimura"])
def test_polarized_ref_is_ancestral(tmp_path, method):
    vcf, maf, truth = _write_inputs(str(tmp_path))
    out = str(tmp_path / f"pol_{method}.vcf.gz")
    polarize.polarize(maf, vcf, ",".join(OUTGROUPS), method, 5, out)

    called = {}
    with gzip.open(out, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                f = line.split("\t", 5)
                called[int(f[1])] = f[3]
    assert called, "polarize wrote no records"
    wrong = [p for p, (anc, _) in truth.items() if p in called and called[p] != anc]
    wrong_alt_major = [p for p in wrong if truth[p][1]]
    n_alt_major = sum(1 for p, (_, am) in truth.items() if am and p in called)
    assert n_alt_major > 0, "test data must include sites where ALT is the major allele"
    # every outgroup carries the ancestral allele: calls must be (almost) all correct,
    # in particular where ALT is the major allele
    assert len(wrong) / len(called) < 0.02, (method, len(wrong), len(called))
    assert len(wrong_alt_major) / n_alt_major < 0.02, (method, len(wrong_alt_major), n_alt_major)
