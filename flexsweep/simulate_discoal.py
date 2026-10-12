import gzip
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
import demes
from scipy import stats

from . import Parallel, delayed, np, pl

_DISCOAL_BIN_NAME = "discoal"


def _find_discoal(binary=None):
    """Locate the discoal binary. Priority: explicit path → env var → bundled data/ → PATH."""
    if binary:
        p = Path(binary)
        if p.is_file() and os.access(p, os.X_OK):
            return p

    env_bin = os.environ.get("FLEXSWEEP_DISCOAL_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file() and os.access(p, os.X_OK):
            return p

    bundled = Path(__file__).resolve().parent / "data" / _DISCOAL_BIN_NAME
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return bundled

    resolved = shutil.which(_DISCOAL_BIN_NAME)
    if resolved:
        return Path(resolved)

    return None


def build_discoal():
    """Clone discoal from GitHub and compile with make. Returns path to binary."""
    if not shutil.which("git"):
        raise FileNotFoundError("git not found; required to build discoal.")
    if not shutil.which("make"):
        raise FileNotFoundError("make not found; required to build discoal.")

    build_dir = Path(tempfile.mkdtemp(prefix="discoal_build_"))
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/kr-colab/discoal.git",
                str(build_dir),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["make", "discoal"],
            cwd=str(build_dir),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, end="")
        raise

    built = build_dir / _DISCOAL_BIN_NAME
    if not built.exists():
        raise FileNotFoundError("make succeeded but discoal binary not found.")

    dest = Path(__file__).resolve().parent / "data" / _DISCOAL_BIN_NAME
    try:
        shutil.copy2(built, dest)
        dest.chmod(dest.stat().st_mode | 0o111)
        return dest
    except OSError:
        return built


def _ensure_discoal(binary=None):
    """Return discoal path, building from source on first use if not found."""
    path = _find_discoal(binary)
    if path:
        return path
    warnings.warn(
        "discoal binary not found; attempting to build from source.",
        RuntimeWarning,
    )
    return build_discoal()


# Extract default data and binaries
DISCOAL = os.path.join(os.path.dirname(__file__), "data", "discoal")
DECODE_MAP = os.path.join(
    os.path.dirname(__file__), "data", "decode_sexavg_2019.txt.gz"
)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 1000 Genomes populations with a shipped Relate (Speidel et al. 2019) single-population
# Ne trajectory, file data/<pop>_spiedel_2019_full.yaml
POPULATIONS_1KG = (
    "ACB ASW BEB CDX CEU CHB CHS CLM ESN FIN GBR GIH GWD "
    "IBS ITU JPT KHV LWK MSL MXL PEL PJL PUR STU TSI YRI"
).split()

DEMES_EXAMPLES = {
    "constant": os.path.join(_DATA_DIR, "constant.yaml"),
    "africa_1t12": os.path.join(_DATA_DIR, "Africa_1T12.yaml"),
    **{
        pop.lower(): os.path.join(_DATA_DIR, f"{pop.lower()}_spiedel_2019_full.yaml")
        for pop in POPULATIONS_1KG
    },
}


def resolve_demes(value: str) -> str:
    """Resolve a demes model given as a shipped alias or a path.

    Aliases are the keys of ``DEMES_EXAMPLES``, case-insensitive: ``constant``,
    ``africa_1t12`` and the 26 1000 Genomes population codes (``yri``, ``ceu``,
    ``chb``, ...; see ``POPULATIONS_1KG``), or the basename of a shipped YAML (e.g.
    ``yri_spiedel_2019_full.yaml``). Anything else is treated as a path to a demes
    YAML file.
    """
    key = str(value).lower()
    if key in DEMES_EXAMPLES:
        return DEMES_EXAMPLES[key]
    for path in DEMES_EXAMPLES.values():
        if os.path.basename(path) == os.path.basename(str(value)) and not os.path.exists(value):
            return path
    if not os.path.exists(value):
        raise FileNotFoundError(
            f"Demes model not found: {value!r}. Pass a path to a demes YAML file or one "
            f"of the shipped models: constant, africa_1t12, or a 1000 Genomes population "
            f"code ({', '.join(p.lower() for p in POPULATIONS_1KG)})."
        )
    return value


def resolve_recombination_map(value: str | None) -> str | None:
    """Resolve ``'decode'`` to the bundled deCODE 2019 sex-averaged map (GRCh38)."""
    if value is None:
        return None
    if str(value).lower() == "decode":
        return DECODE_MAP
    if not os.path.exists(value):
        raise FileNotFoundError(
            f"Recombination map not found: {value!r}. Pass a path or 'decode' for the "
            f"bundled deCODE 2019 sex-averaged map (GRCh38)."
        )
    return value


class Simulator:
    """
    Initialize discoal coalescent simulations.

    The simulator draws per-replicate mutation (:math:`\\mu`) and recombination
    (:math:`r`) rates, scales them to :math:`\\theta = 4N_eL\\mu` and
    :math:`\\rho = 4N_eLr`, and prepares neutral and sweep configurations to be
    executed via the ``discoal`` binary.
    """

    def __init__(
        self,
        sample_size,
        demes,
        output_folder,
        mutation_rate={"dist": "uniform", "min": 5e-9, "max": 2e-8},
        recombination_rate={
            "dist": "exponential",
            "min": 1e-9,
            "max": 1e-7,
            "mean": 1e-8,
        },
        locus_length=int(1.2e6),
        discoal_path=DISCOAL,
        num_simulations=int(2.5e5),
        saf=[0, 0.1],
        eaf=[0.5, 1],
        s=[0.01, 0.05],
        time=[0, 5000],
        x=0.5,
        nthreads=1,
        ne=int(1e4),
        fixed_ratio=0.1,
        split=False,
        seed=None,
    ):
        """
        Initialize discoal coalescent simulations.

        The simulator prepares neutral and sweep configurations to be executed via
        the ``discoal`` binary.

        :param int sample_size: Number of haplotypes per replicate.
        :param str demes: Path to a demes YAML file describing demography, or a
            shipped alias (``constant``, ``africa_1t12`` or a 1000 Genomes population
            code such as ``yri``; see ``DEMES_EXAMPLES``).
        :param str output_folder: Directory to write ms outputs and parameters.
        :param dict mutation_rate: Mutation rate distribution (per-bp :math:`\\mu`).
            Supported forms:

            * ``{"dist": "uniform", "min": float, "max": float}``
            * ``{"dist": "exponential", "min": float, "max": float, "mean": float}``
            * ``{"dist": "truncnorm", "min": float, "max": float, "mean": float, "std": float}``
            * ``{"dist": "fixed", "value": float}`` (same value in every replicate)

            **Default:** ``{"dist": "uniform", "min": 5e-9, "max": 2e-8}``.
        :param dict recombination_rate: Recombination rate distribution (per-bp :math:`r`).
            Same schema as ``mutation_rate``. **Default:**
            ``{"dist": "exponential", "min": 1e-9, "max": 1e-7, "mean": 1e-8}``.
        :param int locus_length: Sequence length in base pairs.
            **Default:** ``1_200_000``.
        :param str discoal_path: Path to the discoal executable.
            **Default:** value of module/global ``DISCOAL``.
        :param int num_simulations: Number of neutral and number of sweep replicates (each).
            **Default:** ``250000``.
        :param int ne: Effective population size :math:`N_e`. Replaced by the size of the
            oldest epoch of the demes model when the model is read; that ancestral size
            scales :math:`\theta`, :math:`\rho`, ``s`` and ``time`` to discoal units.
            **Default:** ``10000``.
        :param list saf: Start allele frequency range ``[min, max]`` of soft sweeps.
            **Default:** ``[0, 0.1]``.
        :param list eaf: End allele frequency range ``[min, max]`` of incomplete sweeps.
            **Default:** ``[0.5, 1]``.
        :param list s: Selection coefficient range ``[min, max]``, drawn uniformly.
            **Default:** ``[0.01, 0.05]``.
        :param list time: Range ``[min, max]`` of generations before the present at which
            the sweep ends, drawn uniformly (discoal ``-ws``).
            **Default:** ``[0, 5000]``.
        :param x: Sweep position along the locus as a fraction in ``[0, 1]``
            (discoal ``-x``). A scalar fixes the position for every replicate;
            a two-element range ``[min, max]`` draws a random position uniformly
            per sweep replicate. **Default:** ``0.5``.
        :type x: float or list
        :param int nthreads: Maximum joblib workers. **Default:** ``1``.
        :param float fixed_ratio: Fraction of complete sweeps within hard/soft sets.
            **Default:** ``0.1``.

        :param int seed: Seed for parameter draws and for every discoal replicate
            (passed as ``-d seed1 seed2``; per-replicate seeds are stored in
            ``params.txt.gz``). ``None`` draws fresh entropy; the per-replicate seeds
            are still recorded, so any replicate can be re-run exactly.

        """
        self.ne_0 = ne
        self.ne = ne
        self.sample_size = sample_size
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.locus_length = int(locus_length)
        self.demes = resolve_demes(demes)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.output_folder = output_folder
        self.discoal_path = str(_ensure_discoal(discoal_path))
        self.nthreads = nthreads
        self.num_simulations = int(num_simulations)
        self.f_i = saf
        self.f_t = eaf
        self.time = time
        self.s = s
        self.x = x
        self.fixed_ratio = fixed_ratio
        self.reset_simulations = False
        self.demes_data = None
        self.parameters = None

    def check_inputs(self):
        """
        Validate inputs and prepare output directories.

        Ensures mutation and recombination distributions are dictionaries,
        creates subdirectories for sweep and neutral simulations,
        and reads demography from the provided demes YAML.

        Returns:
            str: Discoal demographic flags string (e.g., " -en <t> 0 <size> ...").
        """
        assert isinstance(
            self.mutation_rate, dict
        ), "Please input distribution and mutation rates values"
        assert isinstance(
            self.recombination_rate, dict
        ), "Please input distribution and recombination_rate values"
        for name, spec in (
            ("mutation_rate", self.mutation_rate),
            ("recombination_rate", self.recombination_rate),
        ):
            if spec.get("dist") == "fixed" and "value" not in spec:
                raise ValueError(f"{name}: 'fixed' needs a 'value' key, e.g. {{'dist': 'fixed', 'value': 1e-8}}")

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.output_folder + "/sweep/", exist_ok=True)
        os.makedirs(self.output_folder + "/neutral/", exist_ok=True)

        discoal_demes = self.read_demes()

        return discoal_demes

    def read_demes(self):
        """
        Parse the demes YAML and build the discoal -en demography string.

        Behavior:
            - Loads the first deme and reverses epochs (oldest first).
            - Sets self.ne_0 to initial population size.
            - Converts epoch end times to 4Ne units and start sizes to Ne ratios.
            - Produces strings of the form ' -en <time> 0 <size>'.

        Returns:
            str: Concatenated discoal '-en' flags
        """
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
            discoal_demes += f"-en {i:.20f} 0 {j:.20f} "
        return discoal_demes

    def random_distribution(self, num):
        """
        Draw mutation (μ) and recombination (ρ) rates.

        Args:
            num (int): Number of draws.

        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays (mu, rho) each of length `num`.

        Supported distributions:
            - 'uniform': Uniform[min, max]
            - 'exponential': Exponential(mean), truncated to [min, max]
            - 'truncnorm': Truncated normal with mean/std/min/max
            - 'fixed': the same value for every replicate

        Notes:
            - Values are in per-basepair units.
        """
        # eyre-walker = U(5e-9,2-e8 (1e-8 +- 50%)) two small to match mean segregating sites
        # mbe = U(2e-9,5e-8) if uniform to match sweep/neutral with high/low segregating sites, no way to discern
        # mbe
        # N('dist':'normal','mean':1.25e-8,'std':1e-8,'min':2e-9,'max':5.2e-8)
        if self.mutation_rate["dist"] == "uniform":
            mu = self.rng.uniform(
                self.mutation_rate["min"], self.mutation_rate["max"], num
            )
        elif self.mutation_rate["dist"] == "exponential":
            mu = []
            while len(mu) < num:
                tmp_mu = self.rng.exponential(self.mutation_rate["mean"], num)
                valid_values = tmp_mu[
                    (tmp_mu >= self.mutation_rate["min"])
                    & (tmp_mu <= self.mutation_rate["max"])
                ]
                remaining = num - len(mu)
                mu.extend(valid_values[:remaining])
            mu = np.array(mu)
        elif self.mutation_rate["dist"] == "fixed":
            mu = np.full(num, float(self.mutation_rate["value"]))
        else:
            mu = stats.truncnorm(
                (self.mutation_rate["min"] - self.mutation_rate["mean"])
                / self.mutation_rate["std"],
                (self.mutation_rate["max"] - self.mutation_rate["mean"])
                / self.mutation_rate["std"],
                loc=self.mutation_rate["mean"],
                scale=self.mutation_rate["std"],
            ).rvs(size=num, random_state=self.rng)

        #
        # mbe = U('dist': 'normal','min':1e-9,'max':5e-8,'mean':1e-8,'std':5e-9)
        # mbe high r = U('dist': 'normal','min':1e-9,'max':5e-8,'mean':1e-8,'std':1e-8)
        if self.recombination_rate["dist"] == "uniform":
            rho = self.rng.uniform(
                self.recombination_rate["min"],
                self.recombination_rate["max"],
                num,
            )
        elif self.recombination_rate["dist"] == "exponential":
            if len(self.recombination_rate.keys()) == 2:
                rho = self.rng.exponential(self.recombination_rate["mean"], num)
            else:
                rho = []
                while len(rho) < num:
                    tmp_rho = self.rng.exponential(
                        self.recombination_rate["mean"], num
                    )
                    valid_values = tmp_rho[
                        (tmp_rho >= self.recombination_rate["min"])
                        & (tmp_rho <= self.recombination_rate["max"])
                    ]
                    remaining = num - len(rho)
                    rho.extend(valid_values[:remaining])
                rho = np.array(rho)
        elif self.recombination_rate["dist"] == "fixed":
            rho = np.full(num, float(self.recombination_rate["value"]))
        else:
            rho = stats.truncnorm(
                (self.recombination_rate["min"] - self.recombination_rate["mean"])
                / self.recombination_rate["std"],
                (self.recombination_rate["max"] - self.recombination_rate["mean"])
                / self.recombination_rate["std"],
                loc=self.recombination_rate["mean"],
                scale=self.recombination_rate["std"],
            ).rvs(size=num, random_state=self.rng)

        return mu, rho

    def create_params(self):
        """
        Generate and save parameter sets for neutral and sweep simulations into a Polars DataFrame

        Steps:
            1. Draw μ and ρ, compute θ = 4NeLμ and ρ = 4NeLr.
            2. Sample sweep times (uniform between time[0], time[1]).
            3. Sample selection coefficients between s[0] and s[1].
            4. Partition simulations into hard/soft and complete/incomplete sweeps.
            5. Save all parameters to '<output_folder>/params.txt.gz'.

        Returns:
            pl.DataFrame: Parameters with columns
                ['iter', 'mu', 'r', 'eaf', 'saf', 's', 't', 'pos', 'model'].
        """

        _ = self.check_inputs()

        scaling = 4 * self.ne * self.locus_length

        # Neutral simulations
        mu_neutral, r_neutral = self.random_distribution(self.num_simulations)
        theta_neutral = scaling * mu_neutral
        rho_neutral = scaling * r_neutral

        # Sweep simulations
        mu_sweep, r_sweep = self.random_distribution(self.num_simulations)
        theta_sweep = scaling * mu_sweep
        rho_sweep = scaling * r_sweep

        sel_time = self.rng.uniform(
            self.time[0], self.time[-1], self.num_simulations
        ) / (4 * self.ne)

        sel_coef = (
            self.rng.uniform(self.s[0], self.s[-1], self.num_simulations) * 2 * self.ne
        )

        # Sweep position along the locus (discoal -x). Fixed if x is a scalar,
        # drawn uniformly per replicate if x is a [min, max] range.
        if isinstance(self.x, (list, tuple, np.ndarray)):
            sweep_position = self.rng.uniform(
                self.x[0], self.x[-1], self.num_simulations
            )
        else:
            sweep_position = np.repeat(float(self.x), self.num_simulations)

        num_hard = int(self.num_simulations * 0.5)
        num_soft = self.num_simulations - num_hard

        # Complete/incomplete counts per class; incomplete takes the remainder so the
        # four groups always sum to num_simulations (int() truncation of both parts
        # dropped rows, e.g. 48 parameter sets for 50 iterations).
        n_hard_complete = int(num_hard * self.fixed_ratio)
        n_hard_incomplete = num_hard - n_hard_complete
        n_soft_complete = int(num_soft * self.fixed_ratio)
        n_soft_incomplete = num_soft - n_soft_complete

        hard_complete_f_t = np.repeat(1, n_hard_complete)
        hard_complete_f_i = np.repeat(0, n_hard_complete)

        hard_incomplete_f_t = self.rng.uniform(
            self.f_t[0], self.f_t[-1], n_hard_incomplete
        )
        hard_incomplete_f_i = np.repeat(0, n_hard_incomplete)

        soft_complete_f_t = np.repeat(1, n_soft_complete)
        soft_complete_f_i = self.rng.uniform(
            self.f_i[0], self.f_i[-1], n_soft_complete
        )

        soft_incomplete_f_t = self.rng.uniform(
            self.f_t[0], self.f_t[1], n_soft_incomplete
        )
        soft_incomplete_f_i = self.rng.uniform(
            self.f_i[0], self.f_i[1], n_soft_incomplete
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

        # Per-replicate discoal seeds (discoal -d seed1 seed2), recorded in params
        sim_seeds = self.rng.integers(1, 2**31 - 1, size=(2 * self.num_simulations, 2))

        df_neutral = pl.DataFrame(
            {
                "iter": np.arange(1, self.num_simulations + 1),
                "mu": theta_neutral / scaling,
                "r": rho_neutral / scaling,
                "eaf": 0.0,
                "saf": 0.0,
                "s": 0.0,
                "t": 0.0,
                "pos": 0.0,
                "model": "neutral",
                "seed1": sim_seeds[: self.num_simulations, 0],
                "seed2": sim_seeds[: self.num_simulations, 1],
            }
        )

        df_sweeps = pl.DataFrame(
            {
                "iter": np.arange(1, self.num_simulations + 1),
                "mu": theta_sweep / scaling,
                "r": rho_sweep / scaling,
                "eaf": eaf,
                "saf": saf,
                "s": sel_coef / (2 * self.ne),
                "t": 4 * self.ne * sel_time,
                "pos": sweep_position,
                "model": "sweep",
                "seed1": sim_seeds[self.num_simulations :, 0],
                "seed2": sim_seeds[self.num_simulations :, 1],
            }
        )

        # params = df_sweeps.select(["s", "t", "saf", "eaf"]).to_numpy()

        # Simulating
        df_params = pl.concat([df_neutral, df_sweeps], how="vertical")
        df_params.write_csv(self.output_folder + "/params.txt.gz", compression="gzip")

        self.parameters = df_params

        return df_params

    def simulate(self):
        """
        Run neutral and sweep simulations via discoal.

        Returns:
            dict[str, list[str]]: Paths to gzipped ms files, with keys 'neutral' and 'sweep'.
        """
        discoal_demes = self.check_inputs()
        scaling = 4 * self.ne * self.locus_length

        try:
            df_params = pl.read_csv(self.output_folder + "/params.txt.gz")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found: {self.output_folder}/params.txt.gz"
            )

        with Parallel(n_jobs=self.nthreads, backend="loky", verbose=1) as parallel:
            print("Performing neutral simulations")

            sims_n = parallel(
                delayed(self.neutral)(
                    v["mu"] * scaling,
                    v["r"] * scaling,
                    discoal_demes,
                    v["iter"],
                    self._seeds(v),
                )
                for (i, v) in enumerate(
                    df_params.filter(pl.col("model") == "neutral").iter_rows(
                        named=True
                    ),
                    1,
                )
            )
            print("Performing sweep simulations")
            sims_s = parallel(
                delayed(self.sweep)(
                    v["mu"] * scaling,
                    v["r"] * scaling,
                    v["eaf"],
                    v["saf"],
                    v["t"] / (4 * self.ne),
                    2 * self.ne * v["s"],
                    discoal_demes,
                    v["iter"],
                    v["pos"],
                    self._seeds(v),
                )
                for (i, v) in enumerate(
                    df_params.filter(pl.col("model") != "neutral").iter_rows(
                        named=True
                    ),
                    1,
                )
            )

        sims = {
            "sweep": sims_s,
            "neutral": sims_n,
        }

        return sims

    @staticmethod
    def _seeds(row):
        """Per-replicate discoal seeds from a params row, or None for older params files."""
        if row.get("seed1") is None or row.get("seed2") is None:
            return None
        return (int(row["seed1"]), int(row["seed2"]))

    def neutral(self, theta, rho, discoal_demes, _iter=1, seeds=None):
        """
        Run a single neutral simulation.

        Args:
            theta (float): 4NeLμ.
            rho (float): 4NeLr.
            discoal_demes (str): Demography string (from read_demes).
            _iter (int, default=1): Iteration index.
            seeds (tuple[int, int] | None): discoal ``-d`` seeds.

        Returns:
            str: Path to output gzipped ms file.
        """
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

        if seeds is not None:
            discoal_job += f" -d {seeds[0]} {seeds[1]}"

        output_file = self.output_folder + "/neutral/neutral_" + str(_iter) + ".ms.gz"

        with gzip.GzipFile(output_file, "wb", mtime=0) as output:
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
        x=0.5,
        seeds=None,
    ):
        """
        Run a single sweep simulation

        Args:
            theta (float): 4NeLμ.
            rho (float): 4NeLr.
            f_t (float, default=1): Terminal frequency (completed sweep if =1).
            f_i (float, default=0): Initial frequency (0 = hard sweep).
            t (float): Sweep onset time in 4Ne generations.
            s (float): Selection coefficient scaled by 2Ne.
            discoal_demes (str): Demography string.
            _iter (int, default=1): Iteration index.
            x (float, default=0.5): Sweep position along the locus (discoal -x).
            seeds (tuple[int, int] | None): discoal ``-d`` seeds.

        Returns:
            str: Path to output gzipped ms file.
        """

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
            + " -x "
            + str(x)
            + " -ws "
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

        if seeds is not None:
            discoal_job += f" -d {seeds[0]} {seeds[1]}"

        output_file = self.output_folder + "/sweep/sweep_" + str(_iter) + ".ms.gz"
        with gzip.GzipFile(output_file, "wb", mtime=0) as output:
            result = subprocess.run(
                discoal_job.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output.write(result.stdout)

        return output_file

    def simulate_batch(self):
        """
        Run neutral and sweep simulations via discoal, in batches.


        Returns:
            dict[str, list[str]]: Paths to gzipped ms files, with keys 'neutral' and 'sweep'.
        """
        discoal_demes = self.check_inputs()
        scaling = 4 * self.ne * self.locus_length

        try:
            df_params = pl.read_csv(self.output_folder + "/params.txt.gz")
        except Exception:
            raise FileNotFoundError(
                f"File not found: {self.output_folder}/params.txt.gz"
            )

        # materialize rows once
        neutral_rows = list(
            df_params.filter(pl.col("model") == "neutral").iter_rows(named=True)
        )
        sweep_rows = list(
            df_params.filter(pl.col("model") != "neutral").iter_rows(named=True)
        )

        batch_size = int(max(1, len(neutral_rows) / self.nthreads))

        # build batched tasks (each task handles a chunk of rows)
        neutral_tasks = [
            delayed(self._batch_simulate)(
                neutral_rows[i : i + batch_size],
                scaling,
                discoal_demes,
                "neutral",
            )
            for i in range(0, len(neutral_rows), batch_size)
        ]

        sweep_tasks = [
            delayed(self._batch_simulate)(
                sweep_rows[i : i + batch_size],
                scaling,
                discoal_demes,
                "sweep",
            )
            for i in range(0, len(sweep_rows), batch_size)
        ]

        sims_n, sims_s = [], []

        with Parallel(n_jobs=self.nthreads, backend="loky", verbose=1) as parallel:
            print("Performing neutral simulations (batched)")
            neutral_batches = parallel(neutral_tasks)
            sims_n = [sim for batch in neutral_batches for sim in batch]

            print("Performing sweep simulations (batched)")
            sweep_batches = parallel(sweep_tasks)
            sims_s = [sim for batch in sweep_batches for sim in batch]

        sims = {
            "sweep": sims_s,
            "neutral": sims_n,
        }
        return sims

    def _batch_simulate(self, params_batch, scaling, discoal_demes, model):
        """
        Helper that runs either neutral or sweep simulations for a batch of
        parameter rows and returns a list of results.
        """
        out = []

        if model == "neutral":
            for v in params_batch:
                res = self.neutral(
                    v["mu"] * scaling,
                    v["r"] * scaling,
                    discoal_demes,
                    v["iter"],
                    self._seeds(v),
                )
                out.append(res)
        else:  # sweep
            for v in params_batch:
                res = self.sweep(
                    v["mu"] * scaling,
                    v["r"] * scaling,
                    v["eaf"],
                    v["saf"],
                    v["t"] / (4 * self.ne),
                    2 * self.ne * v["s"],
                    discoal_demes,
                    v["iter"],
                    v["pos"],
                    self._seeds(v),
                )
                out.append(res)

        return out
