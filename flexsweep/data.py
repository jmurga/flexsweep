import glob
import gzip
import os
import re
import warnings
from collections import OrderedDict
from itertools import chain

# from subprocess import run
# from typing import Any, cast
from warnings import filterwarnings

from allel import GenotypeArray, index_windows, read_vcf, read_vcf_headers
from scipy.interpolate import interp1d

from . import Parallel, delayed, np, pl

filterwarnings("ignore", message="invalid INFO header", module="allel.io.vcf_read")


def _chrom_id(s):
    s = s.lower()

    # 1. Unify mitochondrial naming
    s = re.sub(r"chr?m(t)?", "mt", s)

    # 2. Extract token
    # We use (?:^|[^a-z0-9]) at the start to ensure we aren't mid-word.
    # Note: We now use m.group(1) for the chromosome because group 0
    # will include the leading separator.
    pattern = (
        r"(?:^|[^a-z0-9])(?:chr|contig|super)?[^a-z0-9]*([0-9]+|x|y|mt)(?=[^a-z0-9]|$)"
    )

    m = re.search(pattern, s)
    if not m:
        return None

    # The capturing group ([0-9]+|x|y|mt) is now the first group
    chrom = m.group(1)

    if chrom.isdigit():
        chrom = str(int(chrom))

    return chrom


_CONTIG_LINE = re.compile(r"^##contig=<(.*)>\s*$")
_CONTIG_FIELD = re.compile(r'([A-Za-z_][A-Za-z0-9_.]*)=("(?:[^"\\]|\\.)*"|[^,]*)')

# chr1 length -> build, used to infer the reference assembly from ##contig lines
_CHR1_LENGTH_TO_BUILD = {248956422: "GRCh38", 249250621: "GRCh37"}


def parse_contig_headers(headers: list[str]) -> dict[str, dict[str, str]]:
    """Parse ``##contig=<...>`` header lines into ``{ID: {key: value}}``.

    Keys may appear in any order (e.g. 1000 Genomes phase 3 writes
    ``##contig=<ID=1,assembly=b37,length=249250621>``). Quoted values are unquoted.
    """
    contigs = {}
    for line in headers:
        m = _CONTIG_LINE.match(line.rstrip("\n"))
        if not m:
            continue
        fields = {k: v.strip('"') for k, v in _CONTIG_FIELD.findall(m.group(1))}
        if "ID" in fields:
            contigs[fields["ID"]] = fields
    return contigs


def infer_genome_build(contigs: dict[str, dict[str, str]]) -> str | None:
    """Infer the human reference build from parsed contig headers, or None."""
    for cid, fields in contigs.items():
        if _chrom_id(cid) == "1" and "length" in fields:
            build = _CHR1_LENGTH_TO_BUILD.get(int(fields["length"]))
            if build is not None:
                return build
    for fields in contigs.values():
        assembly = fields.get("assembly", "").lower()
        if assembly in ("b37", "grch37", "hg19"):
            return "GRCh37"
        if assembly in ("b38", "grch38", "hg38"):
            return "GRCh38"
    return None


def check_vcf_file(path: str) -> None:
    """Fail early with a clear message if a VCF/BCF is not bgzip-compressed and indexed."""
    if path.endswith(".gz"):
        with open(path, "rb") as fh:
            block = fh.read(18)
        # BGZF = gzip member with FEXTRA flag and a 'BC' extra subfield
        if block[:4] != b"\x1f\x8b\x08\x04" or block[12:14] != b"BC":
            raise ValueError(
                f"{path} is not bgzip-compressed (plain gzip cannot be indexed). "
                f"Recompress with: zcat {path} | bgzip > fixed.vcf.gz && tabix -p vcf fixed.vcf.gz"
            )
    if not (os.path.exists(path + ".tbi") or os.path.exists(path + ".csi")):
        raise FileNotFoundError(
            f"No index found for {path} (.tbi or .csi). Index it with: tabix -p vcf {path}"
        )


def _first_record_chrom(path: str) -> str | None:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                return line.split("\t", 1)[0]
    return None


def _last_position(path: str, contig: str) -> int:
    """Largest POS of ``contig`` in a VCF/BCF (used when ``##contig`` has no length)."""
    region = contig if (os.path.exists(path + ".tbi") or os.path.exists(path + ".csi")) else None
    callset = read_vcf(path, fields=["variants/CHROM", "variants/POS"], region=region)
    if callset is None:
        raise ValueError(f"{path}: no records for contig '{contig}'")
    pos = callset["variants/POS"][callset["variants/CHROM"] == contig]
    if pos.size == 0:
        raise ValueError(f"{path}: no records for contig '{contig}'")
    return int(pos.max())


def get_contig_from_vcf_filename(path):
    """Return ``(contig_name, contig_length)`` for a single-chromosome VCF/BCF.

    The contig is resolved in this order:

    1. The CHROM of the first record, if a ``##contig`` line has exactly that ID
       (works for any naming: ``22``, ``chr22``, ``NC_000022.11``, ``scaffold_12``).
    2. The chromosome ID in the filename (e.g. ``yri_chr22.vcf.gz``), matched to the
       ``##contig`` IDs after normalizing ``chr``/``contig``/``super`` prefixes.
    3. The single ``##contig`` line of the header.
    4. The CHROM of the first record, with no ``##contig`` line.

    The length comes from ``length=`` in the matching ``##contig`` line (keys may be
    in any order). If it is missing, the largest POS of that contig is used and a
    warning is emitted: windows then end at the last variant instead of the
    chromosome end.

    Raises
    ------
    ValueError
        If the contig cannot be resolved or has no records.
    """
    contigs = parse_contig_headers(read_vcf_headers(path).headers)
    first = _first_record_chrom(path)
    file_chr = _chrom_id(os.path.basename(path))

    if first is not None and first in contigs:
        contig = first
    else:
        matches = [
            cid for cid in contigs if file_chr is not None and _chrom_id(cid) == file_chr
        ]
        if len(matches) > 1:
            raise ValueError(
                f"{path}: ambiguous ##contig match for chromosome {file_chr}: {matches}"
            )
        if matches:
            contig = matches[0]
        elif len(contigs) == 1:
            contig = next(iter(contigs))
        elif first is not None:
            contig = first
        else:
            found = ", ".join(list(contigs)[:30]) or "none"
            raise ValueError(
                f"{path}: cannot resolve the chromosome: the file has no records, the "
                f"filename has no chromosome ID, and the header has {len(contigs)} "
                f"##contig lines ({found})."
            )

    length = contigs.get(contig, {}).get("length")
    if length is not None:
        return contig, int(length)

    length = _last_position(path, contig)
    warnings.warn(
        f"{path}: ##contig '{contig}' has no length= in the header; using the last "
        f"variant position ({length}) as the contig end. Add the length with "
        f"bcftools reheader --fai to scan up to the chromosome end.",
        UserWarning,
        stacklevel=2,
    )
    return contig, length


def check_genome_build(vcf_files: list[str], recombination_map: str | None) -> None:
    """Warn if the bundled deCODE map (GRCh38) is used with non-GRCh38 VCFs."""
    if recombination_map is None:
        return
    from .simulate_discoal import DECODE_MAP

    try:
        is_decode = os.path.samefile(recombination_map, DECODE_MAP)
    except OSError:
        return
    if not is_decode:
        return
    for f in vcf_files:
        build = infer_genome_build(parse_contig_headers(read_vcf_headers(f).headers))
        if build is not None and build != "GRCh38":
            warnings.warn(
                f"{f} looks like {build}, but the bundled deCODE recombination map is "
                f"GRCh38. Positions will not match: lift the VCF over to GRCh38 or pass "
                f"a {build} map via --recombination_map.",
                stacklevel=2,
            )


class Data:
    def __init__(
        self,
        data,
        region=None,
        samples=None,
        recombination_map=None,
        window_size=int(1.2e6),
        step=int(1e4),
        nthreads=1,
        sequence_length=int(1.2e6),
        in_memory=False,
    ):
        """
        Utilities for loading VCF/ms-style simulations and producing windowed inputs.

        Args:
            data (str | Any):
                Path to input data:
                  - Path containing VCF/BCF files
                  - Path contiaining discoal simulations
            region (str, default=None):
                Optional genomic region string 'CHR:START-END' to read from VCF/BCF.
            samples (list[str] | np.ndarray | None, default=None):
                Optional subset of samples (names or indices) to read from VCF/BCF.
            recombination_map (str | None, default=None):
                Optional path to recombination map CSV with columns:
                [Chr, Begin, End, cMperMb, cM]. If None, assumes physical distance ~ genetic distance.
            window_size (int, default=1_200_000):
                Genomic window size in base pairs when scanning a VCF.
            step (int, default=10_000):
                Sliding step in base pairs between adjacent windows when scanning a VCF.
            nthreads (int, default=1):
                Number of parallel workers for joblib.
            sequence_length (int, default=1_200_000):
                Sequence length used to scale ms/discoal 'positions' (0..1) into bp.

        Notes:
            - The class handle both functions to read VCF files and discoal simulations.
            - Randomness is not used here; outputs are deterministic given inputs.
        """
        self.data = data
        self.region = region
        self.samples = samples
        self.recombination_map = recombination_map
        self.window_size = window_size
        self.step = step
        self.nthreads = nthreads
        self.sequence_length = sequence_length
        self.in_memory = in_memory

    def genome_reader(self, region, samples):
        """
        Read a genomic region from a VCF/BCF and return haplotypes and rec_map given the region  interval.

        Args:
            region (str): Region string 'CHR:START-END'.
            samples (list[str] | np.ndarray): Sample subset to read.

        Returns:
            dict[str, tuple[np.ndarray, np.ndarray] | None]:
                {region: (hap, rec_map_subset)} or {region: None} if no biallelic sites.
                - hap: (S x N) int8 array of phased haplotypes (derived-allele indicators).
                - rec_map_subset: (S x 4) array [chrom, idx, genetic_pos(cm or proxy), physical_pos].

        Notes:
            - Filters to biallelic variants via scikit-allel's AlleleCounts.is_biallelic_01.
            - If no recombination map is supplied, genetic distance is set to physical pos and
              later remapped to a relative [1..sequence_length] coordinate system.
        """
        filterwarnings(
            "ignore", message="invalid INFO header", module="allel.io.vcf_read"
        )

        raw_data = read_vcf(self.data, region=region, samples=samples)

        try:
            gt = GenotypeArray(raw_data["calldata/GT"])
        except Exception:
            return {region: None}

        pos = raw_data["variants/POS"]
        np_chrom = np.char.replace(raw_data["variants/CHROM"].astype(str), "chr", "")
        try:
            np_chrom = np_chrom.astype(int)
        except Exception:
            pass
        ac = gt.count_alleles()

        biallelic_filter = ac.is_biallelic_01()

        hap = gt.to_haplotypes()
        hap = hap.values[biallelic_filter]
        pos = pos[biallelic_filter]
        np_chrom = np_chrom[biallelic_filter]

        if hap.shape[0] == 0:
            return {region: None}

        tmp = list(map(int, region.split(":")[-1].split("-")))

        d_pos = dict(
            zip(np.arange(tmp[0], tmp[1] + 1), np.arange(self.sequence_length) + 1)
        )

        if self.recombination_map is None:
            rec_map = pl.DataFrame(
                {"chrom": np_chrom, "idx": np.arange(pos.size), "pos": pos, "cm": pos}
            ).to_numpy()

            for r in rec_map:
                r[-1] = d_pos[r[-1]]
                r[-2] = d_pos[r[-2]]
        else:
            df_recombination_map = pl.read_csv(self.recombination_map, separator=",")
            genetic_distance = self.get_cm(df_recombination_map, pos)
            rec_map = pl.DataFrame(
                [np_chrom, np.arange(pos.size), pos, genetic_distance]
            ).to_numpy()

            for r in rec_map:
                r[-2] = d_pos[r[-2]]

            if np.all(rec_map[:, -1] == 0):
                rec_map[:, -1] = rec_map[:, -2]

            # # physical position to relative physical position (1,1.2e6)
            # # this way we do not perform any change on summary_statistics center/windows combinations
            # f = interp1d(w, [1, int(self.window_size) + 1])
            # rec_map = np.column_stack([rec_map, f(rec_map[:, 2]).astype(int)])

        # return hap
        return {region: (hap, rec_map[:, [0, 1, -1, 2]])}

    def read_vcf(self):
        """
        Slide windows across a single-contig VCF/BCF and collect per-window data haplotypes

        Returns:
            dict | OrderedDict:
                If in_memory=True:
                    {'sweep': list[[hap, rec_map, zeros(4)]], 'region': list[str]}
                    where each element corresponds to a window with data.
                If in_memory=False:
                    OrderedDict({'sweep': <vcf_path>, 'region': list[str]})
                    with region strings to be used downstream for lazy reading.

        Raises:
            IOError: If the file extension is not one of [.vcf, .vcf.gz, .bcf, .bcf.gz].
            FileNotFoundError: If the VCF/BCF is not tabix-indexed (<path>.tbi is missing).

        Notes:
            - Determines contig name/length by reading last line of the vcf file.
            - Generates overlapping windows of size `window_size` and step `step`.
            - Uses multiprocessing when in_memory=True to read per-window haplotypes.
        """
        if (
            not self.data.endswith(".vcf")
            and not self.data.endswith(".vcf.gz")
            and not self.data.endswith(".bcf")
            and not self.data.endswith(".bcf.gz")
        ):
            raise IOError("VCF file must be vcf or bcf format")

        check_vcf_file(self.data)
        # check_contig_length = (
        # f"{'zcat' if '.gz' in self.data else 'cat'} {self.data} | tail -n 1"
        # )
        # contig_name, contig_length = run(
        #     check_contig_length, shell=True, capture_output=True, text=True
        # ).stdout.split("\t")[:2]

        contig_name, contig_length = get_contig_from_vcf_filename(self.data)

        # check_contig_start = (
        # f'zgrep -v "#" {self.data} | head -n 1'
        # if self.data.endswith(".gz")
        # else f'fgrep -v "^#" {self.data} | head -n 1'
        # )
        # contig_start = run(
        # check_contig_start, shell=True, capture_output=True, text=True
        # ).stdout.split("\t")[1]

        # if (int(contig_start) - 6e5) < 0:
        # contig_start = 1

        # if self.step is None:
        #     step = None
        # else:
        #     step = int(self.step)

        window_iter = list(
            index_windows(
                np.arange(1, int(contig_length)),
                int(self.window_size - 1),
                1,
                int(contig_length) + int(self.window_size - 1),
                int(self.step),
            )
        )

        if self.in_memory:
            with Parallel(
                n_jobs=self.nthreads, backend="multiprocessing", verbose=5
            ) as parallel:
                region_data = parallel(
                    delayed(self.genome_reader)(
                        contig_name + ":" + str(w[0]) + "-" + str(w[1]), self.samples
                    )
                    for w in window_iter
                )

            out_dict = dict(chain.from_iterable(d.items() for d in region_data))

            sims = {"sweep": [], "region": []}
            for k, v in out_dict.items():
                if v is not None:
                    tmp = list(v)
                    tmp.append(np.zeros(4))
                    sims["sweep"].append(tmp)
                    sims["region"].append(k)
        else:
            sims = OrderedDict({"sweep": [], "region": []})
            sims["sweep"] = self.data

            for w in window_iter:
                # if v is not None:
                sims["region"].append(contig_name + ":" + str(w[0]) + "-" + str(w[1]))

        return sims

    def read_simulations(self):
        """
        Load paths (or parsed data if in_memory=True) for discoal simulations.

        Expects a directory structure:
            <self.data>/neutral/*.ms.gz
            <self.data>/sweep/*.ms.gz
            <self.data>/params.txt.gz

        Returns:
            tuple[OrderedDict, pl.DataFrame]:
                sims: OrderedDict({'neutral': <list|array>, 'sweep': <list|array>})
                params: polars DataFrame with columns ['model', 's', 't', 'saf', 'eaf'].

        Raises:
            AssertionError: If self.data is not a string.
            ValueError: If required 'neutral' or 'sweep' subfolders are missing or empty.
        """
        assert isinstance(self.data, str)

        for folder in ("sweep", "neutral"):
            folder_path = os.path.join(self.data, folder)
            if not os.path.exists(folder_path):
                raise ValueError(f"Required directory not found: {folder_path}")
            if not glob.glob(os.path.join(folder_path, "*")):
                raise ValueError(f"Directory is empty: {folder_path}")

        df_params = pl.read_csv(self.data + "/params.txt.gz")
        params = df_params.select(["model", "s", "t", "saf", "eaf", "mu", "r"])
        df_neutral = df_params.filter(pl.col("model") == "neutral")
        df_sweeps = df_params.filter(pl.col("model") == "sweep")

        sweeps = (
            (self.data + "/sweep/sweep_" + (df_sweeps.select("iter") + ".ms.gz"))
            .to_numpy()
            .flatten()
        )
        neutral = (
            (self.data + "/neutral/neutral_" + (df_neutral.select("iter") + ".ms.gz"))
            .to_numpy()
            .flatten()
        )

        sims = OrderedDict(
            {
                "neutral": neutral,
                "sweep": sweeps,
            }
        )

        return sims, params

    def get_cm(self, df_rec_map, positions):
        """
        Interpolate genetic distances (cM) for given physical positions.

        Args:
            df_rec_map (pl.DataFrame):
                Recombination map with numeric columns where:
                  columns[1] holds physical coordinate (bp) and
                  columns[-1] holds cumulative genetic distance (cM).

            positions (np.ndarray[int]): Physical positions to interpolate.

        Returns:
            np.ndarray[float]: Interpolated cumulative cM values (negative values clamped to 0).

        Notes:
            Uses linear interpolation with extrapolation at ends.
        """
        interp_func = interp1d(
            df_rec_map.select(df_rec_map.columns[1]).to_numpy().flatten(),
            df_rec_map.select(df_rec_map.columns[-1]).to_numpy().flatten(),
            kind="linear",
            fill_value="extrapolate",
        )

        # Interpolate the cM values at the interval positions
        rr1 = interp_func(positions)
        # rr2 = interp_func(positions[:, 1])
        rr1[rr1 < 0] = 0

        # Calculate the recombination rate in cM/Mb
        # rate = (rr2 - rr1) / ((positions[:, 1] - positions[:, 0]) / 1e6)

        return rr1
