import os

# os.environ["POLARS_MAX_THREADS"] = "1"


from . import np, Parallel, delayed, pl


from allel import read_vcf, GenotypeArray, index_windows

# import numpy as np
# import polars as pl
# from joblib import Parallel, delayed

from scipy.interpolate import interp1d
from itertools import chain
from warnings import filterwarnings
import re
import gzip
import glob
from subprocess import run

filterwarnings("ignore", message="invalid INFO header", module="allel.io.vcf_read")


class Data:
    def __init__(
        self,
        data,
        sweep_parameters=None,
        region=None,
        samples=None,
        recombination_map=None,
        mask=None,
        window_size=int(1.2e6),
        step=int(1e4),
        nthreads=1,
        sequence_length=int(1.2e6),
    ):
        self.data = data
        self.region = region
        self.samples = samples
        self.recombination_map = recombination_map
        self.mask = mask
        self.window_size = window_size
        self.step = step
        self.nthreads = nthreads
        self.sequence_length = sequence_length

    def genome_reader(self, region, samples, _iter=1):
        filterwarnings(
            "ignore", message="invalid INFO header", module="allel.io.vcf_read"
        )

        raw_data = read_vcf(self.data, region=region, samples=samples)

        try:
            gt = GenotypeArray(raw_data["calldata/GT"])
        except:
            return {region: None}

        pos = raw_data["variants/POS"]
        np_chrom = np.char.replace(raw_data["variants/CHROM"].astype(str), "chr", "")
        try:
            np_chrom = np_chrom.astype(int)
        except:
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
        if (
            not self.data.endswith(".vcf")
            and not self.data.endswith(".vcf.gz")
            and not self.data.endswith(".bcf")
            and not self.data.endswith(".bcf.gz")
        ):
            raise IOError("VCF file must be vcf or bcf format")

        if not os.path.exists(self.data + ".tbi"):
            raise FileNotFoundError(
                f"Please index the vcf/bcf file before continue using: tabix -p vcf {self.data}"
            )

        check_contig_length = (
            f"{'zcat' if '.gz' in self.data else 'cat'} {self.data} | tail -n 1"
        )
        contig_name, contig_length = run(
            check_contig_length, shell=True, capture_output=True, text=True
        ).stdout.split("\t")[:2]

        if self.step is None:
            step = None
        else:
            step = int(self.step)

        window_iter = list(
            index_windows(
                np.arange(1, int(contig_length)),
                int(self.window_size - 1),
                1,
                int(contig_length) + int(self.window_size - 1),
                int(self.step),
            )
        )

        region_data = Parallel(n_jobs=self.nthreads, verbose=5)(
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
        return sims

    def read_simulations(self):
        assert isinstance(self.data, str)
        # df_sweeps = pd.read_csv(self.data + "/sweep_params.txt")
        df_params = pl.read_csv(self.data + "/params.txt.gz")
        params = df_params.select(["model", "s", "t", "saf", "eaf"])
        df_sweeps = df_params.filter(pl.col("model") != "neutral")
        df_neutral = df_params.filter(model="neutral")

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

        # ms_sweeps = list(
        #     chain(
        #         *Parallel(
        #             n_jobs=self.nthreads,
        #             pre_dispatch="10*n_jobs",
        #             batch_size=1000,
        #             verbose=5,
        #         )(
        #             delayed(self.ms_parser)(m, param=p, self.sequence_length=self.sequence_length)
        #             for (m, p) in zip(sweeps, params.iloc[:, 1:].values)
        #         )
        #     )
        # )

        # ms_neutral = list(
        #     chain(
        #         *Parallel(
        #             n_jobs=self.nthreads,
        #             pre_dispatch="10*n_jobs",
        #             batch_size=1000,
        #             verbose=5,
        #         )(delayed(self.ms_parser)(m, self.sequence_length=self.sequence_length) for m in neutral)
        #     )
        # )

        sims = {
            # "sweep": ms_sweeps,
            # "neutral": ms_neutral,
            "sweep": sweeps,
            "neutral": neutral,
        }

        return sims, params

    def read_ms_files(self):
        if not isinstance(self.data, list):
            self.data = [self.data]

        ms_data = Parallel(n_jobs=self.nthreads, verbose=5)(
            delayed(self.ms_parser)(m, self.sequence_length) for m in self.data
        )

        return tuple(chain(*ms_data))

    def ms_parser(self, ms_file, param=None):
        """Read a ms file and output the positions and the genotypes.
        Genotypes are a numpy array of 0s and 1s with shape (num_segsites, num_samples).
        """

        assert (
            ms_file.endswith(".out")
            or ms_file.endswith(".out.gz")
            or ms_file.endswith(".ms")
            or ms_file.endswith(".ms.gz")
        )

        open_function = gzip.open if ms_file.endswith(".gz") else open

        with open_function(ms_file, "rt") as file:
            file_content = file.read()

        # Step 2: Split by pattern (e.g., `---`)
        pattern = r"//"
        partitions = re.split(pattern, file_content)

        if len(partitions) == 1:
            raise IOError(f"File {ms_file} is malformed.")

        positions = []
        haps = []
        rec_map = []
        for r in partitions[1:]:
            # Read in number of segregating sites and positions
            data = []
            for line in r.splitlines()[1:]:
                if line == "":
                    continue
                # if "discoal" in line or "msout" in line:
                # self.sequence_length = int(line.strip().split()[3])
                if line.startswith("segsites"):
                    num_segsites = int(line.strip().split()[1])
                    if num_segsites == 0:
                        continue
                        #     # Shape of data array for 0 segregating sites should be (0, 1)
                        # return np.array([]), np.array([], ndmin=2, dtype=np.uint8).T
                elif line.startswith("positions"):
                    tmp_pos = np.array([float(x) for x in line.strip().split()[1:]])
                    tmp_pos = np.round(tmp_pos * self.sequence_length).astype(int)

                    # Find duplicates in the array
                    duplicates = np.diff(tmp_pos) == 0

                    # While there are any duplicates, increment them by 1
                    for i in np.where(duplicates)[0]:
                        tmp_pos[i + 1] += 1
                    tmp_pos += 1
                    positions.append(tmp_pos)
                    tmp_map = np.column_stack(
                        [
                            np.repeat(1, tmp_pos.size),
                            np.arange(tmp_pos.size),
                            tmp_pos,
                            tmp_pos,
                        ]
                    )
                    rec_map.append(tmp_map)

                else:
                    # Now read in the data
                    data.append(np.array(list(line), dtype=np.int8))
            try:
                data = np.vstack(data).T
            except:
                raise IOError(f"File {ms_file} is malformed.")

            # data = np.vstack(data).T
            haps.append(data)

        if param is None:
            param = np.zeros(4)

        return list(zip(haps, rec_map, [param]))

    def read_region(self):
        if self.region is None:
            raise ValueError("Please input a region")

        if not isinstance(self.data, list):
            self.data = [self.data]

            if isinstance(self.region, list):
                self.data = self.data * len(self.region)
            else:
                self.region = [self.region]

        sims = []
        for v, r in zip(self.data, self.region):
            assert (
                "zarr" in v
                or "vcf" in v
                or "vcf" in v
                or "bcf" in v
                or "bcf in self.data" "VCF file must be zarr, vcf or bcf format"
            )

            raw_data = allel.read_vcf(v, region=r, samples=self.samples)
            gt = allel.GenotypeArray(raw_data["calldata/GT"])
            pos = raw_data["variants/POS"]
            chrom = np.char.replace(raw_data["variants/CHROM"].astype(str), "chr", "")
            np_region = np.array(r.split(":")[-1].split("-")).astype(int)
            try:
                chrom = chrom.astype(int)
            except:
                pass
            ac = gt.count_alleles()
            biallelic_filter = ac.is_biallelic_01()

            hap = gt.to_haplotypes()
            hap = hap.subset(biallelic_filter)
            pos = pos[biallelic_filter]
            chrom = chrom[biallelic_filter]

            if self.recombination_map is None:
                rec_map = (
                    pl.DataFrame(
                        {
                            "chrom": chrom,
                            "idx": np.arange(pos.size),
                            "pos": pos,
                            "cm": pos,
                        }
                    )
                    .to_numpy()
                    .flatten()
                )
            else:
                df_recombination_map = pl.read_csv(
                    self.recombination_map, separator="\t"
                )
                genetic_distance = self.get_cm(df_recombination_map, pos)
                rec_map = (
                    pl.DataFrame([chrom, np.arange(pos.size), pos, genetic_distance])
                    .to_numpy()
                    .flatten()
                    .T
                )

            window_iter = list(
                allel.index_windows(
                    pos, int(self.window_size), pos[0], pos[-1], int(self.step)
                )
            )

            region_data = []
            for w in window_iter:
                tmp_hap, tmp_rec_map = self.get_hap_window(hap, pos, rec_map, w)

                if tmp_hap is not None:
                    # physical position to relative physical position (1,1.2e6)
                    # this way we do not perform any change on summary_statistics center/windows combinations
                    f = interp1d(w, [1, int(self.window_size) + 1])
                    tmp_rec_map = np.column_stack(
                        [tmp_rec_map, f(tmp_rec_map[:, 2]).astype(int)]
                    )

                    region_data.append((tmp_hap, tmp_rec_map[:, [0, 1, -1, 2, 3]]))

            sims.append(region_data)
        if len(sims) == 1:
            sims = sims[0]
        return sims

    def get_cm(self, df_rec_map, positions):
        # Create the interpolating function
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

    def get_hap_window(self, hap_data, positions, rec_map, window):
        flt = (positions >= window[0]) & (positions <= window[-1])

        if flt.sum() == 0:
            return None, None
        else:
            return np.array(hap_data[flt]), rec_map[flt]
