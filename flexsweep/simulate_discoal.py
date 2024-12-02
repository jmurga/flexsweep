import os

from . import pl, np, Parallel, delayed

# import numpy as np
# import pandas as pd
# from joblib import Parallel, delayed

import demes
import subprocess

import gzip
from scipy import stats
from itertools import chain
import re
import importlib.resources

# Extract the resource path using the recommended `files` API
DISCOAL = os.path.join(
    os.path.dirname(importlib.resources.files("data") / "discoal"), "discoal"
)

class Simulator:
    def __init__(
        self,
        sample_size,
        mutation_rate,
        recombination_rate,
        locus_length,
        demes,
        output_folder,
        discoal_path=DISCOAL,
        num_simulations=int(1e4),
        ne=int(1e4),
        time=[0, 5000],
        nthreads=1,
        fixed_ratio=0.1,
    ):
        """
        Initializes the Simulator with given parameters.

        Args:
                parameters (dataclass): A configuration dataclass containing simulation parameters and mean diffusion times.
        """
        self.ne_0 = ne
        self.ne = ne
        self.sample_size = sample_size
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.locus_length = int(locus_length)
        self.demes = demes
        self.output_folder = output_folder
        self.discoal_path = discoal_path
        self.nthreads = nthreads
        self.num_simulations = num_simulations
        self.f_t = [0.2, 1]
        self.f_i = [0, 0.1]
        self.time = [0, 5000]
        self.s = [0.001, 0.01]
        self.fixed_ratio = 0.1
        self.reset_simulations = False
        self.demes_data = None

    def check_inputs(self):
        assert isinstance(
            self.mutation_rate, dict
        ), "Please input distribution and mutation rates values"
        assert isinstance(
            self.recombination_rate, dict
        ), "Please input distribution and recombination_rate values"

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.output_folder + "/sweep/", exist_ok=True)
        os.makedirs(self.output_folder + "/neutral/", exist_ok=True)

        discoal_demes = self.read_demes()

        return discoal_demes

    def read_demes(self):
        assert ".yaml" in self.demes, "Please input a demes model"

        pop_history = demes.load(self.demes).asdict_simplified()["demes"][0]["epochs"]
        df_epochs = pl.DataFrame(pop_history).reverse()

        self.demes_data = df_epochs
        if df_epochs.shape[1] > 2:
            df_epochs = df_epochs.to_pandas()
            df_epochs.iloc[0, 1] = df_epochs.iloc[0, 2]
            self.ne_0 = df_epochs.start_size.iloc[0]
            self.ne = self.ne_0
        else:
            self.ne_0 = df_epochs["start_size"].to_numpy()[0]
            self.ne = self.ne_0

        epochs = df_epochs["end_time"].to_numpy()[1:] / (4 * self.ne_0)
        sizes = df_epochs["start_size"].to_numpy()[1:] / self.ne_0

        discoal_demes = " "
        for i, j in zip(epochs, sizes):
            discoal_demes += "-en {0:.20f} 0 {1:.20f} ".format(i, j)
        return discoal_demes

    def random_distribution(self, num):
        if (self.mutation_rate["dist"] == "exponential") or (
            self.mutation_rate["dist"] == "uniform"
        ):
            dist = getattr(np.random, self.mutation_rate["dist"])
            try:
                mu = dist(self.mutation_rate["lower"], self.mutation_rate["upper"], num)
            except:
                mu = dist(self.mutation_rate["mean"], num)
        elif self.mutation_rate["dist"] == "fixed":
            next
        else:
            mu = stats.truncnorm(
                (self.mutation_rate["min"] - self.mutation_rate["mean"])
                / self.mutation_rate["std"],
                (self.mutation_rate["max"] - self.mutation_rate["mean"])
                / self.mutation_rate["std"],
                loc=self.mutation_rate["mean"],
                scale=self.mutation_rate["std"],
            ).rvs(size=num)

        if (self.recombination_rate["dist"] == "exponential") or (
            self.recombination_rate["dist"] == "uniform"
        ):
            dist = getattr(np.random, self.recombination_rate["dist"])
            try:
                rho = dist(
                    self.recombination_rate["lower"],
                    self.recombination_rate["upper"],
                    num,
                )
            except:
                rho = dist(self.recombination_rate["mean"], num)
        elif self.recombination_rate["dist"] == "fixed":
            next
        else:
            rho = stats.truncnorm(
                (self.recombination_rate["min"] - self.recombination_rate["mean"])
                / self.recombination_rate["std"],
                (self.recombination_rate["max"] - self.recombination_rate["mean"])
                / self.recombination_rate["std"],
                loc=self.recombination_rate["mean"],
                scale=self.recombination_rate["std"],
            ).rvs(size=num)
        # mu = np.random.uniform(self.mutation_rate[0], self.mutation_rate[1], num)
        # rho = np.random.uniform(
        #     self.recombination_rate[0],
        #     self.recombination_rate[1],
        #     num,
        # )

        return mu, rho

    def simulate(self):
        discoal_demes = self.check_inputs()

        # Neutral simulations
        mu, rho = self.random_distribution(self.num_simulations)
        theta_neutral = 4 * self.ne * self.locus_length * mu
        rho_neutral = 4 * self.ne * self.locus_length * rho

        print("Performing neutral simulations")
        sims_n = Parallel(n_jobs=self.nthreads, backend="multiprocessing", verbose=5)(
            delayed(self.neutral)(v[0], v[1], discoal_demes, i)
            for (i, v) in enumerate(zip(theta_neutral, rho_neutral), 1)
        )

        df_neutral = pl.DataFrame(
            {
                "iter": np.arange(1, self.num_simulations + 1),
                "theta": theta_neutral / (4 * self.ne * self.locus_length),
                "rho": rho_neutral / (4 * self.ne * self.locus_length),
                "eaf": 0.0,
                "saf": 0.0,
                "s": 0.0,
                "t": 0.0,
                "model": "neutral",
            }
        )

        ms_neutral = list(
            chain(
                *Parallel(n_jobs=self.nthreads, verbose=0)(
                    delayed(self.ms_parser)(m, seq_len=1.2e6) for m in sims_n
                )
            )
        )

        # Sweep simulations
        mu, rho = self.random_distribution(self.num_simulations)
        theta_sweeps = 4 * self.ne * self.locus_length * mu
        rho_sweeps = 4 * self.ne * self.locus_length * rho

        sel_time = np.round(
            np.random.uniform(self.time[0], self.time[1], self.num_simulations)
            / (4 * self.ne),
            3,
        )

        sel_coef = np.random.uniform(
            2 * self.ne * self.s[0], 2 * self.ne * self.s[1], self.num_simulations
        )

        num_hard = int(self.num_simulations * 0.5)
        num_soft = self.num_simulations - num_hard

        hard_complete_f_t = np.repeat(1, int(num_hard * self.fixed_ratio))
        hard_complete_f_i = np.repeat(0, int(num_hard * self.fixed_ratio))

        hard_incomplete_f_t = np.random.uniform(
            self.f_t[0], self.f_t[1], int(num_hard * (1 - self.fixed_ratio))
        )
        hard_incomplete_f_i = np.repeat(0, int(num_hard * (1 - self.fixed_ratio)))

        soft_complete_f_t = np.repeat(1, int(num_soft * self.fixed_ratio))
        soft_complete_f_i = np.random.uniform(
            self.f_i[0], self.f_i[1], int(num_soft * self.fixed_ratio)
        )

        soft_incomplete_f_t = np.random.uniform(
            self.f_t[0], self.f_t[1], int(num_soft * (1 - self.fixed_ratio))
        )
        soft_incomplete_f_i = np.random.uniform(
            self.f_i[0], self.f_i[1], int(num_soft * (1 - self.fixed_ratio))
        )

        saf = np.concatenate(
            [
                hard_complete_f_i,
                hard_incomplete_f_i,
                soft_complete_f_i,
                soft_incomplete_f_i,
            ]
        )
        eaf = np.concatenate(
            [
                hard_complete_f_t,
                hard_incomplete_f_t,
                soft_complete_f_t,
                soft_incomplete_f_t,
            ]
        )

        print("Performing sweep simulations")

        sims_s = Parallel(n_jobs=self.nthreads, verbose=5)(
            delayed(self.sweep)(v[0], v[1], v[2], v[3], v[4], v[5], discoal_demes, i)
            for (i, v) in enumerate(
                zip(theta_sweeps, rho_sweeps, eaf, saf, sel_time, sel_coef), 1
            )
        )

        df_sweeps = pl.DataFrame(
            {
                "iter": np.arange(1, self.num_simulations + 1),
                "theta": theta_sweeps / (4 * self.ne * self.locus_length),
                "rho": rho_sweeps / (4 * self.ne * self.locus_length),
                "eaf": eaf,
                "saf": saf,
                "s": sel_coef / (2 * self.ne),
                "t": 4 * self.ne * sel_time,
                "model": "sweep",
            }
        )

        params = df_sweeps.select(["s", "t", "saf", "eaf"]).to_numpy()

        ms_sweeps = list(
            chain(
                *Parallel(n_jobs=self.nthreads, verbose=0)(
                    delayed(self.ms_parser)(m, param=p, seq_len=1.2e6)
                    for (m, p) in zip(sims_s, params)
                )
            )
        )

        df = pl.concat([df_neutral, df_sweeps], how="vertical")
        df.write_csv(self.output_folder + "/params.txt.gz")

        sims = {
            "sweep": ms_sweeps,
            "neutral": ms_neutral,
        }

        return sims

    def neutral(self, theta, rho, discoal_demes, _iter=1):
        discoal_job = (
            self.discoal_path
            + " "
            + str(self.sample_size)
            + " 1 "
            + str(self.locus_length)
            + " -t "
            + str(theta)
            + " -r "
            + str(rho)
        )

        if discoal_demes != "constant":
            discoal_job += discoal_demes

        output_file = self.output_folder + "/neutral/neutral_" + str(_iter) + ".ms.gz"

        with gzip.open(output_file, "wb") as output:
            result = subprocess.run(
                discoal_job.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output.write(result.stdout)

        return output_file

    def sweep(
        self,
        theta,
        rho,
        f_t,
        f_i,
        t,
        s,
        discoal_demes,
        _iter=1,
    ):
        # Default job is a hard/complete sweep in equilibrium population
        # -c, -f, and -en flags not defined
        discoal_job = (
            self.discoal_path
            + " "
            + str(self.sample_size)
            + " 1 "
            + str(self.locus_length)
            + " -t "
            + str(theta)
            + " -r "
            + str(rho)
            + " -x 0.5 -ws "
            + str(t)
            + " -a "
            + str(s)
        )

        # Simulate ongoing/partial sweep
        if f_t != 1:
            discoal_job += " -c " + str(f_t)

        # Simulate soft sweep
        if f_i != 0:
            discoal_job += " -f " + str(f_i)

        # Add demography
        if discoal_demes != "constant":
            discoal_job += discoal_demes

        output_file = self.output_folder + "/sweep/sweep_" + str(_iter) + ".ms.gz"
        with gzip.open(output_file, "wb") as output:
            result = subprocess.run(
                discoal_job.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output.write(result.stdout)

        return output_file

    def ms_parser(self, ms_file, param=None, seq_len=1.2e6):
        """Read a ms file and output the positions and the genotypes.
        Genotypes are a numpy array of 0s and 1s with shape (num_segsites, num_samples).
        """

        assert (
            ms_file.endswith(".ms.gz")
            or ms_file.endswith(".ms")
            or ms_file.endswith(".out.gz")
            or ms_file.endswith(".out")
        )

        open_function = gzip.open if ms_file.endswith(".gz") else open

        with open_function(ms_file, "rt") as file:
            file_content = file.read()

        # Step 2: Split by pattern (e.g., `---`)
        pattern = r"//"
        partitions = re.split(pattern, file_content)

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
                # seq_len = int(line.strip().split()[3])
                if line.startswith("segsites"):
                    num_segsites = int(line.strip().split()[1])
                    if num_segsites == 0:
                        continue
                        #     # Shape of data array for 0 segregating sites should be (0, 1)
                        # return np.array([]), np.array([], ndmin=2, dtype=np.uint8).T
                elif line.startswith("positions"):
                    tmp_pos = np.array([float(x) for x in line.strip().split()[1:]])
                    tmp_pos = np.round(tmp_pos * seq_len).astype(int)

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
            data = np.row_stack(data).T
            haps.append(data)

        if param is None:
            param = np.zeros(4)

        return list(zip(haps, rec_map, [param]))
