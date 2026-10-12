"""Example dataset: YRI chr22 (GRCh38; 5 Mb or whole), from the Flex-sweep Zenodo record.

``fetch_example`` reproduces the empirical data preparation of Flex-sweep at small scale
(chr22:20-25 Mb, or the whole chromosome with ``full=True``). It is the first step of
``flexsweep example``, which then simulates, trains the CNN, predicts and scans.
All inputs come from Zenodo record ``ZENODO_RECORD`` (downloaded once to
``{out_dir}/raw/`` and checked against their md5):

1. Samples: ``n_samples`` individuals drawn at random (fixed ``seed``) from the 108
   unrelated YRI of the 1000 Genomes phase 3 release, i.e. the samples of
   ``yri_chr22_hg38_high_coverage.vcf.gz``.
2. Genotypes: that VCF holds chr22 of the NYGC 30x 1000 Genomes phased panel (GRCh38),
   biallelic SNPs only. The region is read from it; SNPs polymorphic in the drawn
   samples are kept and AC/AN/AF are recomputed for them.
3. Outgroups: the great-ape MAF extracted from the Zoonomia 447-mammal Cactus
   alignment (``447.primates_chr22.maf.gz``) is reduced to the region, keeping the
   human reference row plus the outgroup rows (chimpanzee ``Pan_troglodytes``,
   gorilla ``Gorilla_gorilla``, orangutan ``Pongo_abelii``).
4. Polarization: ``flexsweep polarize`` with the Kimura model (REF := ancestral
   allele; sites without a usable alignment are dropped).
5. Mask: sites inside the GIAB v2.0 GRCh38 difficult regions
   (``GRCh38_alldifficultregions.bed.gz``) are removed, i.e. sites are kept in the
   "easy" regions used for the NYGC call-set evaluation (Byrska-Bishop et al. 2022,
   Cell, STAR Methods).

The result is a bgzipped, tabix-indexed VCF in ``{out_dir}/vcfs/`` ready for
``flexsweep fvs-vcf --recombination_map decode`` and ``flexsweep scan``.
"""

import bisect
import gzip
import hashlib
import io
import os
import shutil
import time
import urllib.error
import urllib.request

import numpy as np

EXAMPLE_REGION = "chr22:20000001-25000000"
EXAMPLE_REGION_FULL = "chr22:1-50818468"
EXAMPLE_N_SAMPLES = 50
EXAMPLE_OUTGROUPS = "Pan_troglodytes,Gorilla_gorilla,Pongo_abelii"
EXAMPLE_CHROM = "chr22"

# Zenodo deposit of the example inputs (see scripts/zenodo_stage.sh): file name -> md5.
ZENODO_RECORD = "23260568"
ZENODO_VCF = "yri_chr22_hg38_high_coverage.vcf.gz"
ZENODO_MAF = "447.primates_chr22.maf.gz"
ZENODO_BED = "GRCh38_alldifficultregions.bed.gz"
ZENODO_MD5 = {
    ZENODO_VCF: "537a360275d630ec7860eb3dac99c96d",
    ZENODO_MAF: "4a6617dc20b0096c8296f992475e69e0",
    ZENODO_BED: "680cfe46a84aee3d889dc8aa4f4c1852",
}


def zenodo_url(name: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"


def _parse_region(region: str) -> tuple[str, int, int]:
    chrom, span = region.split(":")
    start, end = (int(x) for x in span.replace(",", "").split("-"))
    return chrom, start, end


def vcf_samples(vcf: str) -> list[str]:
    """Sample names of a (gzipped) VCF, from its ``#CHROM`` line."""
    with _open_text(vcf) as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                return line.rstrip("\n").split("\t")[9:]
            if not line.startswith("#"):
                break
    raise ValueError(f"{vcf}: no #CHROM header line")


def choose_samples(samples: list[str], n_samples: int | None, seed: int) -> list[str]:
    """Draw ``n_samples`` individuals without replacement (all if None), reproducibly."""
    samples = sorted(samples)
    if n_samples is None or n_samples >= len(samples):
        return samples
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=n_samples, replace=False)
    return sorted(samples[i] for i in idx)


def _fetch_genotypes(src, chrom, start, end, samples, out_vcf):
    """Read ``chrom:start-end`` (1-based, inclusive) of a coordinate-sorted VCF, keep
    biallelic SNPs polymorphic in ``samples`` and write them as a plain VCF with
    AC/AN/AF of ``samples``. Returns the number of records written."""
    contig_line = None
    n = 0
    with _open_text(src) as fh, open(out_vcf, "w") as out:
        for line in fh:
            if line.startswith("##"):
                if line.startswith(f"##contig=<ID={chrom},"):
                    contig_line = line
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
                col = {s: i for i, s in enumerate(header)}
                missing = sorted(set(samples) - set(col))
                if missing:
                    raise ValueError(f"Samples not in {src}: {missing}")
                idx = [col[s] for s in samples]
                out.write("##fileformat=VCFv4.2\n")
                out.write(contig_line or f"##contig=<ID={chrom}>\n")
                # AC/AN/AF of the selected samples, as bcftools +fill-tags would write
                # them: flexsweep polarize fits its ancestral-state model on these counts.
                out.write(
                    '##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes">\n'
                    '##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">\n'
                    '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
                    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                    f"##source=flexsweep.fetch_example {os.path.basename(src)} {chrom}:{start}-{end}\n"
                )
                out.write("\t".join(header[:9] + list(samples)) + "\n")
                continue
            c, p, rest = line.split("\t", 2)
            if c != chrom:
                continue
            pos = int(p)
            if pos < start:
                continue
            if pos > end:
                break
            f = line.rstrip("\n").split("\t")
            ref, alt = f[3], f[4]
            if len(ref) != 1 or len(alt) != 1 or alt in ("*", "."):
                continue
            gts = [f[i].split(":", 1)[0] for i in idx]
            alleles = [a for gt in gts for a in gt.replace("|", "/").split("/") if a != "."]
            an = len(alleles)
            ac = sum(a == "1" for a in alleles)
            if ac == 0 or ac == an:
                continue
            out.write(
                f"{chrom}\t{pos}\t{chrom}:{pos}:{ref}:{alt}\t{ref}\t{alt}\t.\t.\t"
                f"AC={ac};AN={an};AF={ac / an:.6g}\tGT\t" + "\t".join(gts) + "\n"
            )
            n += 1
    return n


def _is_url(src: str) -> bool:
    return src.startswith(("http://", "https://", "ftp://"))


def _open_text(src: str):
    """Open a local or remote, plain or gzipped text file for line iteration."""
    if _is_url(src):
        raw = urllib.request.urlopen(src, timeout=300)
        if src.split("?")[0].endswith(".gz"):
            return io.TextIOWrapper(gzip.GzipFile(fileobj=raw))
        return io.TextIOWrapper(raw)
    return gzip.open(src, "rt") if src.endswith(".gz") else open(src)


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(
    url: str, dest: str, md5: str | None = None, attempts: int = 6, wait: float = 30.0
) -> str:
    """Download ``url`` to ``dest`` once (checking ``md5`` if given).

    Zenodo rate-limits downloads (HTTP 429). On 429 and 5xx responses wait for the
    server's ``Retry-After`` (else ``wait`` seconds, doubled each time) and retry, up
    to ``attempts`` tries in total."""
    if not os.path.exists(dest):
        tmp = dest + ".part"
        for i in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as out:
                    shutil.copyfileobj(r, out)
                break
            except urllib.error.HTTPError as e:
                if (e.code != 429 and e.code < 500) or i == attempts:
                    raise RuntimeError(
                        f"Download of {url} failed (HTTP {e.code}) after {i} tries. Wait a "
                        f"few minutes and rerun (finished downloads are kept), or download "
                        f"it yourself to {dest}."
                    ) from e
                try:
                    delay = float(e.headers.get("Retry-After", ""))
                except ValueError:
                    delay = wait * 2 ** (i - 1)
                print(
                    f"  HTTP {e.code} from {url.split('?')[0]}; retrying in {delay:.0f} s "
                    f"({i}/{attempts - 1})",
                    flush=True,
                )
                time.sleep(delay)
        os.replace(tmp, dest)
    if md5 is not None and _md5(dest) != md5:
        raise ValueError(f"Checksum mismatch for {dest} (expected md5 {md5}); delete it and retry.")
    return dest


def _fetch_maf(src, chrom, start, end, keep_species, out_maf):
    """Read a reference-anchored MAF (path or URL) and keep the blocks overlapping
    [start, end] with the reference row plus the ``keep_species`` rows.

    The reference species is the species of the first ``s`` line in the file
    (``Homo_sapiens`` in the Cactus MAF, ``hg38`` in UCSC multiz). MAF fields may be
    separated by spaces or tabs. ``s`` lines are 0-based. Blocks are written sorted by
    reference start. Returns the number of blocks kept.
    """
    ref_src = None
    blocks = []
    with _open_text(src) as fh:
        block = []
        for line in fh:
            if line.startswith("a"):
                block = [line]
            elif line.startswith("s") and block:
                if ref_src is None:
                    ref_src = line.split()[1].split(".", 1)[0] + "." + chrom
                block.append(line)
            elif not line.strip() and block:
                kept = _filter_block(block, ref_src, keep_species, start, end)
                if kept is not None:
                    blocks.append(kept)
                elif blocks and _block_start(block, ref_src) > end:
                    # per-chromosome MAFs are in reference order
                    break
                block = []
        if block:
            kept = _filter_block(block, ref_src, keep_species, start, end)
            if kept is not None:
                blocks.append(kept)
    blocks.sort(key=lambda b: b[0])
    with gzip.open(out_maf, "wt") as out:
        out.write("##maf version=1\n")
        for _, lines in blocks:
            out.write("".join(lines) + "\n")
    return len(blocks)


def _block_start(block, ref_src):
    for ln in block[1:]:
        f = ln.split()
        if f[1] == ref_src:
            return int(f[2])
    return -1


def _filter_block(block, ref_src, keep_species, start, end):
    ref = None
    keep = [block[0]]
    for ln in block[1:]:
        f = ln.split()
        species = f[1].split(".", 1)[0]
        if f[1] == ref_src:
            ref = (int(f[2]), int(f[3]), f[4])
            keep.insert(1, ln)
        elif species in keep_species:
            keep.append(ln)
    if ref is None or ref[2] != "+":
        return None
    b0, b1 = ref[0], ref[0] + ref[1]  # 0-based half-open
    if b1 < start or b0 >= end:
        return None
    return (b0, keep)


def _read_bed(src, chrom, start, end):
    """BED intervals (path or URL) on ``chrom`` overlapping [start, end], merged and
    sorted, as (starts, ends) in 0-based half-open coordinates."""
    iv = []
    with _open_text(src) as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")):
                continue
            c, s, e = line.split("\t")[:3]
            if c != chrom:
                continue
            s, e = int(s), int(e)
            if e >= start and s <= end:
                iv.append((s, e))
    iv.sort()
    starts, ends = [], []
    for s, e in iv:
        if starts and s <= ends[-1]:
            ends[-1] = max(ends[-1], e)
        else:
            starts.append(s)
            ends.append(e)
    return starts, ends


def _in_intervals(pos0, starts, ends):
    i = bisect.bisect_right(starts, pos0) - 1
    return i >= 0 and pos0 < ends[i]


def _write_final(in_vcf, out_vcf, starts=None, ends=None):
    """Copy ``in_vcf`` to a bgzipped, tabix-indexed ``out_vcf``, dropping records whose
    1-based POS lies in a BED (0-based half-open) interval (none if ``starts`` is None).
    Returns the number of records written."""
    from flexsweep import polarize as _polarize

    tmp = out_vcf + ".tmp.vcf"
    n = 0
    with _open_text(in_vcf) as fh, open(tmp, "w") as out:
        for line in fh:
            if not line.startswith("#"):
                pos0 = int(line.split("\t", 2)[1]) - 1
                if starts is not None and _in_intervals(pos0, starts, ends):
                    continue
                n += 1
            out.write(line)
    _polarize.bgzip_index(tmp, out_vcf)
    os.remove(tmp)
    return n


def fetch_example(
    out_dir: str,
    full: bool = False,
    n_samples: int = EXAMPLE_N_SAMPLES,
    seed: int = 1,
    overwrite: bool = False,
) -> str:
    """Download, polarize and mask the example VCF (see module docstring).

    Parameters
    ----------
    out_dir:
        Output folder. Downloads and intermediate files go to ``{out_dir}/raw/``; the
        final VCF to ``{out_dir}/vcfs/`` so that folder can be passed directly as
        ``--vcf_path``.
    full:
        Whole chr22 instead of chr22:20-25 Mb.
    n_samples:
        Number of YRI individuals drawn at random (of 108). Simulate
        ``2 * n_samples`` haplotypes (``--sample_size``) to match.
    seed:
        Seed for the sample draw.
    overwrite:
        Rebuild even if the final VCF exists.

    Returns
    -------
    str
        Path to the polarized, masked, bgzipped and indexed VCF.
    """
    from flexsweep import polarize as _polarize

    region = EXAMPLE_REGION_FULL if full else EXAMPLE_REGION
    chrom, start, end = _parse_region(region)
    out_dir = os.path.abspath(out_dir)
    raw_dir = os.path.join(out_dir, "raw")
    vcf_dir = os.path.join(out_dir, "vcfs")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(vcf_dir, exist_ok=True)
    tag = f"yri_{chrom}"
    final_vcf = os.path.join(vcf_dir, f"{tag}_example.vcf.gz")
    if os.path.exists(final_vcf + ".tbi") and not overwrite:
        return final_vcf

    def fetch(name):
        if not os.path.exists(os.path.join(raw_dir, name)):
            print(f"  downloading {name} from Zenodo", flush=True)
        return _download(zenodo_url(name), os.path.join(raw_dir, name), ZENODO_MD5[name])

    src_vcf = fetch(ZENODO_VCF)
    maf = fetch(ZENODO_MAF)
    mask_bed = fetch(ZENODO_BED)

    samples = choose_samples(vcf_samples(src_vcf), n_samples, seed)
    with open(os.path.join(raw_dir, f"{tag}_samples.txt"), "w") as fh:
        fh.write("\n".join(samples) + "\n")

    print(f"  genotypes: {region}, {len(samples)} YRI", flush=True)
    raw_vcf = os.path.join(raw_dir, f"{tag}.vcf")
    n_raw = _fetch_genotypes(src_vcf, chrom, start, end, samples, raw_vcf)
    if n_raw == 0:
        raise ValueError(f"No polymorphic SNPs in {region}")

    print("  polarizing with the great-ape MAF", flush=True)
    region_maf = os.path.join(raw_dir, f"{tag}.outgroups.maf.gz")
    keep = set(EXAMPLE_OUTGROUPS.split(","))
    n_blocks = _fetch_maf(maf, chrom, start, end, keep, region_maf)
    if n_blocks == 0:
        raise ValueError(f"No alignment blocks for {region} in {maf}")
    pol_vcf = os.path.join(raw_dir, f"{tag}.polarized.vcf.gz")
    _polarize.polarize(region_maf, raw_vcf, EXAMPLE_OUTGROUPS, "kimura", 10, pol_vcf)

    print("  masking GIAB difficult regions", flush=True)
    starts, ends = _read_bed(mask_bed, chrom, start, end)
    n_final = _write_final(pol_vcf, final_vcf, starts, ends)

    with open(os.path.join(raw_dir, f"{tag}_summary.txt"), "w") as fh:
        fh.write(
            f"region\t{region}\nsamples\t{len(samples)}\nseed\t{seed}\n"
            f"snps_polymorphic\t{n_raw}\nmaf_blocks\t{n_blocks}\n"
            f"outgroups\t{EXAMPLE_OUTGROUPS}\nmethod\tkimura\n"
            f"snps_final\t{n_final}\n"
        )
    return final_vcf
