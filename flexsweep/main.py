import glob
import math
import os
import click


def parse_float_list(value):
    """Parse a comma-separated list of floats."""
    if value:
        try:
            return [int(x) if x.isdigit() else float(x) for x in value.split(",")]
        except ValueError:
            raise click.BadParameter("Must be a comma-separated list of floats")
    return None


def _resolve_map_option(ctx, param, value):
    """Click callback: accept 'decode' for the bundled deCODE map, else a path."""
    if value is None:
        return None
    from .simulate_discoal import resolve_recombination_map

    try:
        return resolve_recombination_map(value)
    except FileNotFoundError as e:
        raise click.BadParameter(str(e))


def _resolve_demes_option(ctx, param, value):
    """Click callback: accept a shipped demes alias (constant, africa_1t12, a 1000GP code) or a path."""
    from .simulate_discoal import resolve_demes

    try:
        return resolve_demes(value)
    except FileNotFoundError as e:
        raise click.BadParameter(str(e))


MAP_HELP = (
    "Recombination map TSV (chr, start, end, cm_mb, cm), or 'decode' for the bundled "
    "deCODE 2019 sex-averaged map. The bundled map is GRCh38: VCF coordinates must be GRCh38."
)


@click.group()
def cli():
    """CLI for Simulator and CNN."""
    pass


@cli.command()
@click.option("--sample_size", type=int, required=True, help="Number of haplotypes.")
@click.option(
    "--demes",
    type=str,
    required=True,
    callback=_resolve_demes_option,
    help=(
        "Demes YAML file describing demography, or a shipped model: "
        "constant, africa_1t12, or any of the 26 1000 Genomes population codes "
        "(acb, asw, beb, cdx, ceu, chb, chs, clm, esn, fin, gbr, gih, gwd, ibs, itu, "
        "jpt, khv, lwk, msl, mxl, pel, pjl, pur, stu, tsi, yri; case-insensitive). The "
        "population models are single-population Ne trajectories estimated with Relate "
        "(Speidel et al. 2019)."
    ),
)
@click.option(
    "--output_folder",
    type=str,
    required=True,
    help="Directory where simulation outputs will be saved.",
)
@click.option(
    "--num_simulations",
    type=int,
    default=None,
    help="Number of neutral replicates, and of sweep replicates. [default: 250000]",
)
@click.option(
    "--locus_length",
    type=int,
    default=None,
    help="Length of the simulated locus in bp. [default: 1200000]",
)
@click.option(
    "--mutation_rate",
    type=str,
    default=None,
    help=(
        "Per-bp mutation rate. One value: fixed for every replicate (e.g. '1.25e-8'). "
        "Two values 'min,max': uniform. Three values 'min,max,mean': exponential with "
        "that mean, truncated to [min, max]. [default: 5e-9,2e-8]"
    ),
)
@click.option(
    "--recombination_rate",
    type=str,
    default=None,
    help=(
        "Per-bp recombination rate. One value: fixed for every replicate (e.g. '1e-8'). "
        "Two values 'min,max': uniform. Three values 'min,max,mean': exponential with "
        "that mean, truncated to [min, max]. [default: 1e-9,1e-7,1e-8]"
    ),
)
@click.option(
    "--s",
    type=str,
    default=None,
    help="Selection coefficient range 'min,max', drawn uniformly. [default: 0.01,0.05]",
)
@click.option(
    "--time",
    type=str,
    default=None,
    help=(
        "Range 'min,max' of generations since the sweep ended, drawn uniformly "
        "(discoal -ws). [default: 0,5000]"
    ),
)
@click.option(
    "--saf",
    type=str,
    default=None,
    help="Start allele frequency range 'min,max' of soft sweeps. [default: 0,0.1]",
)
@click.option(
    "--eaf",
    type=str,
    default=None,
    help="End allele frequency range 'min,max' of incomplete sweeps. [default: 0.5,1]",
)
@click.option(
    "--fixed_ratio",
    type=float,
    default=None,
    help=(
        "Fraction of complete sweeps (end frequency 1) among hard sweeps and among "
        "soft sweeps; the rest are incomplete. [default: 0.1]"
    ),
)
@click.option(
    "--x",
    type=str,
    default=None,
    help=(
        "Sweep position along the locus as a fraction of --locus_length (discoal -x). "
        "One value: fixed position; 'min,max': uniform per sweep replicate. [default: 0.5]"
    ),
)
@click.option(
    "--nthreads",
    type=int,
    default=None,
    help="Number of parallel discoal jobs. [default: 1]",
)
@click.option(
    "--discoal_path",
    type=str,
    default=None,
    help="Path to a discoal executable. [default: the bundled flexsweep.DISCOAL]",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help=(
        "Random seed for parameter draws and discoal replicates. Per-replicate discoal "
        "seeds are recorded in params.txt.gz. [default: unseeded]"
    ),
)
def simulator(
    sample_size,
    demes,
    output_folder,
    num_simulations,
    locus_length,
    mutation_rate,
    recombination_rate,
    s,
    time,
    saf,
    eaf,
    fixed_ratio,
    x,
    nthreads,
    discoal_path,
    seed,
):
    """
    Simulate neutral and sweep replicates with discoal.

    Options left unset take the defaults of flexsweep.Simulator.

    \b
    Example usage:
        flexsweep simulator --sample_size 216 --demes yri --output_folder ./sims --nthreads 24 --seed 1
    """
    import flexsweep as fs

    def _rate_spec(values, name):
        # One value: fixed; two values: uniform [min, max];
        # three values: exponential(mean) truncated to [min, max]
        if values is not None and len(values) == 1:
            return {"dist": "fixed", "value": values[0]}
        if values is not None and len(values) == 2:
            return {"dist": "uniform", "min": values[0], "max": values[1]}
        if values is not None and len(values) == 3:
            return {
                "dist": "exponential",
                "min": values[0],
                "max": values[1],
                "mean": values[2],
            }
        raise click.BadParameter(
            "expected 'value' (fixed), 'min,max' (uniform) or 'min,max,mean' "
            "(truncated exponential)",
            param_hint=f"--{name}",
        )

    def _range(value, name):
        values = parse_float_list(value)
        if values is None or len(values) != 2:
            raise click.BadParameter("expected 'min,max'", param_hint=f"--{name}")
        return values

    # Only options given on the command line are passed; the rest keep the
    # Simulator defaults, so CLI and Python share one set of defaults.
    kwargs = {}
    if num_simulations is not None:
        kwargs["num_simulations"] = num_simulations
    if locus_length is not None:
        kwargs["locus_length"] = locus_length
    if mutation_rate is not None:
        kwargs["mutation_rate"] = _rate_spec(
            parse_float_list(mutation_rate), "mutation_rate"
        )
    if recombination_rate is not None:
        kwargs["recombination_rate"] = _rate_spec(
            parse_float_list(recombination_rate), "recombination_rate"
        )
    for name, value in (("s", s), ("time", time), ("saf", saf), ("eaf", eaf)):
        if value is not None:
            kwargs[name] = _range(value, name)
    if fixed_ratio is not None:
        if not 0 <= fixed_ratio <= 1:
            raise click.BadParameter("must be in [0, 1]", param_hint="--fixed_ratio")
        kwargs["fixed_ratio"] = fixed_ratio
    if x is not None:
        xs = parse_float_list(x)
        if xs is None or len(xs) not in (1, 2) or not all(0 <= v <= 1 for v in xs):
            raise click.BadParameter(
                "expected 'x' or 'min,max' in [0, 1]", param_hint="--x"
            )
        kwargs["x"] = float(xs[0]) if len(xs) == 1 else [float(v) for v in xs]
    if nthreads is not None:
        kwargs["nthreads"] = nthreads
    if discoal_path is not None:
        kwargs["discoal_path"] = discoal_path

    simulator = fs.Simulator(
        sample_size=sample_size,
        demes=demes,
        output_folder=output_folder,
        seed=seed,
        **kwargs,
    )
    simulator.create_params()
    simulator.simulate_batch()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOWING SCHEMES for --windows / --centers (locus_length = 1.2e6, sweep @ 600 kb)
# --windows = comma-list of window sizes (bp); --centers = comma-list of center
# positions (bp). Omit --centers to use the uniform grid derived from --step.
# Same flags apply to `fvs-discoal` (sims) and `fvs-vcf` (centers are sim-relative,
# remapped per region). The five schemes used in the s/t bake-off:
#
#   base          --windows 100000
#                 (uniform centers from --step 100000 -> 50k..1.15M)              12 x 1
#
#   ms            --windows 20000,50000,100000,200000,500000
#                 (uniform centers from --step 100000)                            12 x 5
#
#   log_scale     --windows 2000,4000,8000,16000,32000,64000,128000,256000,512000
#                 --centers 10000,30000,...,1190000  (range(10000,1200000,20000)) 60 x 9
#
#   log_distance  --windows 4000,16000,64000,256000
#                 --centers 280000,440000,520000,560000,580000,600000,620000,
#                           640000,680000,760000,920000  (log-dist radial @ 600k) 11 x 4
#
#   radial        --windows 4000,8000,16000,32000,64000,128000,256000,512000
#                 --centers 600000  (single sweep center)                          1 x 8
#
#   caldas        Caldas et al. 2022 geometry (biorxiv 2022.07.19.500702), the
#                 d x d subwindow cube with d = 14 rescaled to L = 1.2 Mb.
#                 NOT a single call: summary_statistics applies ONE --centers to
#                 ALL --windows (rectangular product, fv.py:1934), but Caldas
#                 centers are SIZE-DEPENDENT (spacing = size/2, fanned on sweep).
#                 => run ONCE PER size s_i, each with its own 21-center fan, then
#                 hstack the 21 parquets on the index cols. Driver: scripts/caldas_geometry.
#                   sizes (21, geometric lmin=1e3 -> L/11=109091, ratio 1.2644):
#                     1000,1264,1599,2021,2556,3232,4086,5167,6533,8260,10445,
#                     13206,16698,21114,26696,33755,42681,53966,68235,86278,109091
#                   per size s: --centers = [600000 + (j-10)*(s/2) for j in 0..20]
#                     (21 windows overlapping neighbours by half, fanned @ 600k)  21 x 21
# ─────────────────────────────────────────────────────────────────────────────
@cli.command()
@click.option(
    "--simulations_path",
    type=str,
    required=True,
    help="Directory containing neutral and sweeps discoal simulations.",
)
@click.option(
    "--stats",
    type=str,
    required=False,
    default=None,
    help="Comma-separated list of summary statistics to compute, e.g:dind,fay_wu_h,isafe",
)
@click.option(
    "--nthreads",
    type=int,
    required=True,
    default=1,
    help="Number of thread to parallelize.",
)
@click.option(
    "--windows",
    type=str,
    required=False,
    default="100000",
    help="Comma-separated window size(s) in bp, e.g. 20000,50000,100000,200000,500000 (ms). See WINDOWING SCHEMES comment above.",
)
@click.option(
    "--step",
    type=float,
    required=False,
    default=int(1e5),
    help="Step size to sliding windows over the simulated regions",
)
@click.option(
    "--centers",
    type=str,
    required=False,
    default=None,
    help="Comma-separated center position(s) in bp. Omit for the uniform grid from --step. See WINDOWING SCHEMES comment above.",
)
@click.option(
    "--locus_length",
    type=float,
    required=False,
    default=int(1.2e6),
    help="Simulated locus size",
)
@click.option(
    "--r_bins",
    type=str,
    required=False,
    default=None,
    help="Recombination bins to normalize simulations",
)
@click.option(
    "--suffix",
    type=str,
    required=False,
    default=None,
    help="Custom suffix name to save feature vectors",
)
@click.option(
    "--save_stats",
    type=bool,
    required=False,
    default=False,
    help="Save raw statistics.",
)
def fvs_discoal(
    simulations_path,
    stats,
    windows,
    step,
    centers,
    locus_length,
    r_bins,
    save_stats,
    suffix,
    nthreads,
):
    """
    Estimate summary statistics from discoal simulations and build feature vectors.

    This command processes both neutral and sweep simulations in the given directory,
    computes a panel of summary statistics, and generates two outputs: a Parquet dataframe containing feature vectors, a Pickle dictionary containing neutral expectations and standard deviations (used for normalization during CNN training).
    """
    import flexsweep as fs

    r_bins_list = parse_float_list(r_bins)
    windows_list = parse_float_list(windows)
    center_list = parse_float_list(centers)

    stats_list = stats.split(",") if stats is not None else None

    print("Estimating summary statistics")
    fs.fv.summary_statistics(
        simulations_path,
        stats=stats_list,
        vcf=False,
        nthreads=nthreads,
        windows=windows_list,
        step=step,
        centers=center_list,
        locus_length=locus_length,
        recombination_map=None,
        r_bins=r_bins_list,
        min_rate=0.0,
        suffix=suffix,
        func=None,
        save_stats=save_stats,
    )


@cli.command()
@click.option(
    "--vcf_path",
    type=str,
    required=True,
    help="Directory containing vcfs folder with all the VCF files to analyze.",
)
@click.option(
    "--stats",
    type=str,
    required=False,
    default=None,
    help="Comma-separated list of summary statistics to compute, e.g:dind,fay_wu_h,isafe",
)
@click.option(
    "--nthreads",
    type=int,
    required=False,
    default=1,
    help="Number of threads for parallelization",
)
@click.option(
    "--windows",
    type=float,
    required=False,
    default=int(1e5),
    help="Window size to sliding windows over the selected region",
)
@click.option(
    "--step",
    type=int,
    required=False,
    default=int(1e5),
    help="Step size to sliding windows over the selected region",
)
@click.option(
    "--step_vcf",
    type=float,
    required=False,
    default=int(1e4),
    help="Step size to sliding windows over the VCF",
)
@click.option(
    "--locus_length",
    type=float,
    required=False,
    default=int(1.2e6),
    help="Windows size to sliding windows over the VCF",
)
@click.option(
    "--recombination_map",
    type=str,
    default=None,
    required=False,
    callback=_resolve_map_option,
    help=MAP_HELP,
)
@click.option(
    "--min_rate",
    type=float,
    required=False,
    default=0.01,
    help="Minimun recombination rate simulated",
)
@click.option(
    "--r_bins",
    type=str,
    required=False,
    default=None,
    help="Recombination bins to normalize simulations",
)
@click.option(
    "--suffix",
    type=str,
    required=True,
    default=None,
    help="Custom suffix name to save feature vectors",
)
@click.option(
    "--save_stats",
    type=bool,
    required=False,
    default=False,
    help="Save raw statistics.",
)
@click.option(
    "--tmp_dir",
    type=str,
    required=False,
    default=None,
    help="Folder for joblib temporary memmap files, cleaned after each VCF. Default: joblib's (JOBLIB_TEMP_FOLDER, then /dev/shm, then /tmp).",
)
@click.option(
    "--accessibility_mask",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional GIAB-style BED of regions to EXCLUDE (e.g. GRCh38_alldifficultregions.bed.gz), all chromosomes. SNPs inside it are dropped and per-bp statistics (pi, theta_w, dist_var) are divided by the accessible bases of each window instead of its length.",
)
@click.option(
    "--min_accessible",
    type=click.FloatRange(0, 1),
    default=0.6,
    show_default=True,
    help="With --accessibility_mask: windows whose accessible fraction is below this value get missing window statistics.",
)
def fvs_vcf(
    vcf_path,
    stats,
    windows,
    step,
    step_vcf,
    locus_length,
    recombination_map,
    min_rate,
    r_bins,
    save_stats,
    suffix,
    nthreads,
    tmp_dir,
    accessibility_mask,
    min_accessible,
):
    """
    Estimate summary statistics from VCF files and build feature vectors.

    This command parses VCF files in the given directory, computes summary statistics
    per genomic window, and writes feature vectors suitable as CNN input.

    \b
    Example usage:
        # Run summary statistics from VCFs using 8 threads, no recombination map
        flexsweep fvs-vcf --vcf_path ./data --nthreads 8
    \b
        # Run with the bundled deCODE map (GRCh38)
        flexsweep fvs-vcf --vcf_path ./data --nthreads 8 --recombination_map decode

    Notes: one bgzipped, tabix-indexed VCF/BCF per chromosome, REF = ancestral allele,
    ##contig header lines with length=, GRCh38 coordinates when using the bundled map.
    """
    import flexsweep as fs

    r_bins_list = parse_float_list(r_bins)

    stats_list = stats.split(",") if stats is not None else None

    vcf_files = sorted(
        glob.glob(os.path.join(vcf_path, "*vcf.gz"))
        + glob.glob(os.path.join(vcf_path, "*bcf.gz"))
    )
    from flexsweep.data import check_genome_build

    check_genome_build(vcf_files, recombination_map)

    fs.summary_statistics(
        vcf_path,
        vcf=True,
        stats=stats_list,
        nthreads=nthreads,
        windows=[windows],
        step=step,
        step_vcf=step_vcf,
        locus_length=locus_length,
        recombination_map=recombination_map,
        r_bins=r_bins_list,
        min_rate=min_rate,
        suffix=suffix,
        func=None,
        save_stats=save_stats,
        tmp_dir=tmp_dir,
        accessibility_mask=accessibility_mask,
        min_accessible=min_accessible,
    )


@cli.command()
@click.option(
    "--vcf_path",
    type=str,
    required=True,
    help="Directory containing vcfs folder with all the VCF files to analyze.",
)
@click.option(
    "--recombination_map",
    type=str,
    default=None,
    required=False,
    callback=_resolve_map_option,
    help=MAP_HELP,
)
@click.option(
    "--bins", type=int, default=10, required=False, help="Recombinations bins"
)
@click.option(
    "--window_size",
    type=int,
    default=int(1.2e6),
    required=False,
    help="Genome windows size",
)
@click.option(
    "--step", type=int, default=int(1e4), required=False, help="Step to slide windows"
)
@click.option(
    "--min_rate",
    type=float,
    default=0.01,
    required=False,
    help="Minimun recombination rate (cM/Mb)",
)
def recombination_bins(vcf_path, recombination_map, bins, window_size, step, min_rate):
    """
    Output recombination bins from empirical recombination map in decode format.

    \b
    Example usage:
        flexsweep recombination-bins --vcf_path ./data --recombination_map decode

    """
    import glob
    import os

    from joblib import Parallel, delayed

    import flexsweep as fs
    from flexsweep import pl
    from flexsweep.fv import get_cm

    def read_regions(vcf_path):
        fs_data = fs.Data(vcf_path, window_size=window_size, step=step, nthreads=1)
        vcf_dict = fs_data.read_vcf()

        regions = (
            pl.DataFrame(vcf_dict["region"])
            .with_columns(
                pl.col("column_0")
                .str.extract_groups(r"^(?P<chr>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
                .alias("g")
            )
            .unnest("g")
            .with_columns(
                pl.col("start").cast(pl.Int64),
                pl.col("end").cast(pl.Int64),
            )
            .drop("column_0")
        )
        tmp_r = get_cm(
            df_recombination_map.filter(
                pl.col("chr") == regions["chr"].unique().item()
            ),
            regions.select("start", "end").to_numpy(),
            cm_mb=True,
        )
        return tmp_r

    # fvs_file = {}
    # regions = {}

    vcf_files = sorted(
        glob.glob(os.path.join(vcf_path, "*vcf.gz"))
        + glob.glob(os.path.join(vcf_path, "*bcf.gz"))
    )

    if not vcf_files:
        raise FileNotFoundError(f"No VCF/BCF files found in directory: {vcf_path}")
    if recombination_map is None:
        raise click.UsageError("--recombination_map is required (a path or 'decode').")

    from flexsweep.data import check_genome_build

    check_genome_build(vcf_files, recombination_map)

    df_recombination_map = pl.read_csv(
        recombination_map,
        separator="\t",
        comment_prefix="#",
        schema=pl.Schema(
            [
                ("chr", pl.String),
                ("start", pl.Int64),
                ("end", pl.Int64),
                ("cm_mb", pl.Float64),
                ("cm", pl.Float64),
            ]
        ),
    )

    df_r_l = []
    nthreads = min(len(vcf_files), os.cpu_count() or 1)
    click.echo(click.style("Reading and sliding VCF file(s)", bold=True))
    with Parallel(n_jobs=nthreads, verbose=2) as parallel:
        df_r_l = parallel(delayed(read_regions)(i) for i in vcf_files)

    df_r = pl.concat(df_r_l)
    df_r = df_r.filter(pl.col("cm_mb") >= min_rate)

    lbl_fmt = ".2f"

    ps = [i / bins for i in range(1, bins)]

    qs_row = df_r.select(
        [pl.col("cm_mb").quantile(p).alias(f"q{i}") for i, p in enumerate(ps, start=1)]
    ).row(0)

    breaks = sorted({float(x) for x in qs_row if x is not None})

    hi = float(df_r.select(pl.col("cm_mb").max()).item())

    # Round intermediate edges normally
    edges = [round(x, 2) for x in breaks]

    # Force last edge to be >= true max (round UP, not nearest)
    hi_rounded_up = math.ceil(hi * 10) / 10  # → 21.2 in your case
    edges.append(hi_rounded_up)

    # Optional: ensure strictly increasing (important for binning)
    edges = sorted(set(edges))

    # Labels
    labels = [float(f"{x:{lbl_fmt}}") for x in edges]

    labels_str = ",".join(f"{x:{lbl_fmt}}" for x in edges)

    click.echo(click.style("Recombination bins:", bold=True) + f" {labels_str}")


@cli.command()
@click.option(
    "--train_data",
    type=str,
    required=False,
    help="Path to feature vectors from simulations for training the CNN.",
)
@click.option(
    "--predict_data",
    type=str,
    required=False,
    help="Path to feature vectors from empirical data for prediction.",
)
@click.option(
    "--center",
    type=str,
    required=False,
    default="0,1.2e6",
    help="Simulated start and end to sliding windows over the regions. Interally using the centers.",
)
@click.option(
    "--windows",
    type=int,
    required=False,
    default=int(1e5),
    help="Window size to sliding windows over the regions",
)
@click.option(
    "--step",
    type=int,
    required=False,
    default=int(1e5),
    help="Step size to sliding windows over the regions",
)
@click.option(
    "--output_folder",
    type=str,
    required=True,
    help="Directory to store the trained model, logs, and predictions.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Path to a pre-trained CNN model. If provided, the CNN will only perform prediction.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed (Python, NumPy, TensorFlow, data splits). GPU runs may still differ slightly.",
)
def cnn(train_data, predict_data, center, windows, step, output_folder, model, seed):
    """
    Run the Flex-sweep CNN for training or prediction.

    Depending on the inputs the software train, predict or train/predict.

    \b
    Train example:
        flexsweep cnn --train_data data/train.parquet --output_folder ./sims/

    \b
    Predict example:
        flexsweep cnn --model ./sims/model.keras --predict_data data/test.parquet --output_folder results/
    \b
    Train/predict example:
        flexsweep cnn --train_data data/train.parquet --predict_data data/train.parquet --output_folder ./sims/

    """
    import flexsweep as fs

    os.makedirs(output_folder, exist_ok=True)

    center_list = [int(x) for x in parse_float_list(center)]
    windows = [windows]

    if model is None:
        if not train_data:
            raise click.UsageError(
                "--train_data is required when --model is not provided."
            )
        fs_cnn = fs.CNN(
            train_data=train_data,
            output_folder=output_folder,
            center=center_list,
            windows=windows,
            step=step,
            seed=seed,
        )
        fs_cnn.train()
        if predict_data is not None:
            fs_cnn.predict_data = predict_data
            fs_cnn.predict()

    else:
        if predict_data is None:
            raise click.UsageError(
                "--predict_data is required when --model is provided."
            )
        fs_cnn = fs.CNN(
            predict_data=predict_data,
            output_folder=output_folder,
            model=model,
            center=center_list,
            windows=windows,
            step=step,
            seed=seed,
        )
        fs_cnn.predict()


@cli.command()
@click.option(
    "--source_data",
    type=str,
    required=False,
    help="Path to feature vectors from simulations for training the CNN.",
)
@click.option(
    "--target_data",
    type=str,
    required=True,
    help="Path to feature vectors from empirical data for prediction.",
)
@click.option(
    "--center",
    type=str,
    required=False,
    default="0,1.2e6",
    help="Simulated start and end to sliding windows over the regions. Interally using the centers.",
)
@click.option(
    "--windows",
    type=int,
    required=False,
    default=int(1e5),
    help="Window size to sliding windows over the regions",
)
@click.option(
    "--step",
    type=int,
    required=False,
    default=int(1e5),
    help="Step size to sliding windows over the regions",
)
@click.option(
    "--target_ratio",
    type=float,
    required=False,
    default=2.0,
    help="Target ratio",
)
@click.option(
    "--max_lambda",
    type=float,
    required=False,
    default=1.0,
    help="Maximum gradient-reversal lambda",
)
@click.option(
    "--ramp_epochs",
    type=int,
    required=False,
    default=30,
    help="Ramping epochs",
)
@click.option(
    "--batch_size",
    type=int,
    required=False,
    default=64,
    help="Batch size to train",
)
@click.option(
    "--output_folder",
    type=str,
    required=True,
    help="Directory to store the trained model, logs, and predictions.",
)
@click.option(
    "--preprocess",
    is_flag=True,
    default=False,
    help="Preprocessing stats",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Path to a pre-trained DANN model. If provided, the DANN will only perform prediction.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed (Python, NumPy, TensorFlow, data splits). GPU runs may still differ slightly.",
)
def dann(
    source_data,
    target_data,
    center,
    windows,
    step,
    target_ratio,
    max_lambda,
    ramp_epochs,
    batch_size,
    output_folder,
    preprocess,
    model,
    seed,
):
    """
    Run the Flex-sweep DANN for training or prediction.

    Depending on the inputs the software train, predict or train/predict.

    \b
    Train example:
        flexsweep dann --source_data data/train.parquet --target_data data/empirical.parquet --output_folder ./sims/

    \b
    Predict example:
        flexsweep dann --model ./sims/model.keras --target_data data/empirical.parquet --output_folder results/
    """
    import flexsweep as fs

    os.makedirs(output_folder, exist_ok=True)

    center_list = [int(x) for x in parse_float_list(center)]

    if model is None:
        if not source_data:
            raise click.UsageError(
                "--source_data is required when --model is not provided."
            )
        fs_dann = fs.CNN(
            source_data=source_data,
            target_data=target_data,
            output_folder=output_folder,
            center=center_list,
            windows=fs.np.asarray([windows]),
            step=step,
            seed=seed,
        )
        fs_dann.train_da(
            tgt_ratio=target_ratio,
            max_lambda=max_lambda,
            ramp_epochs=ramp_epochs,
            batch_size=batch_size,
            preprocess=preprocess,
        )
        fs_dann.predict_da(preprocess=preprocess)
    else:
        if target_data is None:
            raise click.UsageError(
                "--target_data is required when --model is provided."
            )
        fs_dann = fs.CNN(
            source_data=source_data,
            target_data=target_data,
            output_folder=output_folder,
            model=model,
            center=center_list,
            windows=fs.np.asarray([windows]),
            step=step,
            seed=seed,
        )
        # predict_da reads predict_data (train_da sets it from target_data)
        fs_dann.predict_data = target_data
        # train_da saves the Platt calibration next to the model; apply the same one so
        # predict-only output matches the training-run predictions
        cal_file = os.path.join(
            os.path.dirname(os.path.abspath(model)), "calibration.json"
        )
        if os.path.exists(cal_file):
            import json

            with open(cal_file) as f:
                fs_dann.calibration = json.load(f)
        else:
            click.secho(
                f"No calibration.json next to {model}: predictions are uncalibrated.",
                fg="yellow",
            )
        fs_dann.predict_da(preprocess=preprocess)


@cli.command()
@click.option(
    "--prediction",
    type=str,
    default=None,
    help="File containing predicted sweep probabilites (single-population mode).",
)
@click.option(
    "--prediction_dir",
    type=str,
    default=None,
    help="Folder with one prediction file per population (multi-population mode).",
)
@click.option(
    "--populations",
    type=str,
    default=None,
    help="Comma-separated population codes; sets the column order of the merged table.",
)
@click.option(
    "--pattern",
    type=str,
    default="{pop}_*predictions.txt",
    show_default=True,
    help="Glob for each population's file in --prediction_dir; {pop} is replaced by the code.",
)
@click.option(
    "--feature_coordinates",
    type=str,
    required=True,
    help="Bed file contaning genomic feature coordinates.",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Merged rank table (multi-population mode). Default: <prediction_dir>/all_ranks.txt.",
)
def rank(prediction, prediction_dir, populations, pattern, feature_coordinates, output):
    """
    Rank genomic features by their maximum nearby sweep probability.

    Reads the predictions file and a BED of genomic features, links each
    feature to the nearest prediction windows on the same chromosome, takes the
    **maximum** sweep probability among matched windows for that feature, and ranks
    all features by this value.

    Two modes:

    * ``--prediction``: one population. Writes ``<prediction>_rank_<n>.txt``
      (``gene_id``, ``rank``; tab-separated, no header), where ``n`` is the number of
      features tied at the highest sweep probability.
    * ``--prediction_dir`` + ``--populations``: ranks one file per population (found
      with ``--pattern``) and writes a single table to ``--output``: ``gene_id`` plus one
      rank column per population in ``--populations`` order, sorted by ``gene_id``,
      space-separated, no header. This is the ``--sweep_files`` input of
      ``flexsweep enrichment``. Name the output so that the text after its last ``_``
      labels the dataset (e.g. ``all_ranks_v3rbins``), since enrichment reports use it.

    Input formats
    -------------
    * **prediction**: delimited text (CSV) with at least the columns
      ``chr``, ``start``, ``end``, ``prob_sweep``. Additional columns (e.g. ``iter``)
      are ignored for ranking.
    * **feature_coordinates**: BED-like, tab-separated with no header, columns
      ``chr``, ``start``, ``end``, identifier (e.g. gene id) and strand.
      Coordinates are interpreted as 0-based half-open.

    \b
    Example usage:
        flexsweep rank \\
            --prediction_dir predictions/ \\
            --populations yri,ceu,chb \\
            --feature_coordinates genes.bed \\
            --output predictions/all_ranks_v3

    :notes:
        The ranking and feature-window association are performed by
        :func:`flexsweep.rank_probabilities`; the multi-population table by
        :func:`flexsweep.utils.rank_merge`.
    """
    import os

    import flexsweep as fs

    if (prediction is None) == (prediction_dir is None):
        raise click.UsageError("Give exactly one of --prediction or --prediction_dir.")

    if prediction_dir is not None:
        if populations is None:
            raise click.UsageError("--prediction_dir requires --populations.")
        from flexsweep.utils import find_prediction_files, rank_merge

        pops = populations.split(",")
        try:
            files = find_prediction_files(prediction_dir, pops, pattern)
        except ValueError as e:
            raise click.UsageError(str(e))
        if output is None:
            output = os.path.join(prediction_dir, "all_ranks.txt")

        df_rank, ties = rank_merge(files, pops, feature_coordinates, output=output)

        click.echo("column\tpopulation\tfile\ttied_at_max")
        for i, (pop, f) in enumerate(zip(pops, files)):
            click.echo(f"{i + 2}\t{pop}\t{os.path.basename(f)}\t{ties[pop]}")
        click.echo(f"{df_rank.height} features x {len(pops)} populations -> {output}")
        return

    df_rank, max_rank = fs.rank_probabilities(prediction, feature_coordinates)

    if max_rank > 1:
        click.echo(
            click.style(f"{max_rank}", bold=True)
            + " genomic features has the same highest sweep probability"
        )

    dir_name = os.path.dirname(prediction)
    base_name = os.path.splitext(os.path.basename(prediction))[0]

    output_path = os.path.join(dir_name, f"{base_name}_rank_{max_rank}.txt")

    df_rank[:, :2].write_csv(output_path, separator="\t", include_header=False)


@cli.command()
@click.option(
    "--sweep_files",
    type=str,
    required=True,
    help="Comma-separated paths to sweep rank files (gene_id + per-population rank columns).",
)
@click.option(
    "--gene_set",
    type=str,
    required=True,
    help="Case (VIP) gene ids, one per line (no header). All other genes are control candidates.",
)
@click.option(
    "--factors",
    type=str,
    required=True,
    help="TSV confounding factors file (gene_id + factor columns, no header).",
)
@click.option(
    "--annotation",
    type=str,
    required=True,
    help="BED gene coordinates file (0-based, no header).",
)
@click.option(
    "--populations",
    type=str,
    required=True,
    help="Comma-separated population codes matching sweep file column order.",
)
@click.option(
    "--groups",
    type=str,
    required=True,
    help="Comma-separated group label per population (same length as --populations).",
)
@click.option(
    "--thresholds",
    type=str,
    required=True,
    help="Comma-separated rank thresholds (e.g. 6000,5000,...,20).",
)
@click.option(
    "--pop_interest",
    type=str,
    default="All",
    help="Perl Pop_interest: scope (population, group or All) whose p-values set how many control sets are used (20/100/all).",
)
@click.option(
    "--cluster_distance",
    type=int,
    default=500_000,
    help="Neighbour distance (bp) for --count_sweeps (Perl Cluster_distance); unused otherwise.",
)
@click.option(
    "--count_sweeps",
    is_flag=True,
    default=False,
    help="Perl Count_sweeps yes: neighbouring genes in one sweep (within --cluster_distance) count once.",
)
@click.option(
    "--tolerance",
    type=float,
    default=0.05,
    help="Allowed ± fraction deviation in confounding factor averages.",
)
@click.option(
    "--min_distance",
    type=int,
    default=1_250_000,
    help="Minimum bp distance from VIPs for control gene eligibility.",
)
@click.option(
    "--flip",
    is_flag=True,
    default=False,
    help="Flip test direction when |VIPs| > |controls| (increases power).",
)
@click.option(
    "--max_rep",
    type=int,
    default=25,
    help="Max average resamples per control gene across all bootstrap sets.",
)
@click.option(
    "--nthreads",
    type=int,
    default=1,
    help="Parallel workers (joblib).",
)
@click.option(
    "--n_shuffles",
    type=int,
    default=8,
    help="FDR genome-shuffle replicates (multiple of 8; use >= 1000 for reported p-values).",
)
@click.option(
    "--shuffling_segs",
    type=int,
    default=2,
    help="Genes per genomic shuffle segment.",
)
@click.option(
    "--shuffle_blocks",
    type=int,
    default=None,
    help="Cut the shuffled genome into this many blocks instead of --shuffling_segs-gene segments (paper: 100).",
)
@click.option(
    "--distance_file",
    type=str,
    default=None,
    help="Pre-computed gene set distance",
)
@click.option(
    "--facet",
    is_flag=True,
    default=False,
    help="Facet populations in enrichment plot",
)
@click.option(
    "--plot_groups",
    is_flag=True,
    default=False,
    help="Plot group curves (--groups labels) instead of population curves.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Seed for the bootstrap and the genome shuffles (printed when not given).",
)
@click.option(
    "--reuse_bootstrap",
    type=str,
    default=None,
    help="Output folder of a previous run: reuse its case_set.txt/control_set.txt.",
)
@click.option(
    "--output_folder",
    type=str,
    default=None,
    help="Folder to output files).",
)
def enrichment(
    sweep_files,
    gene_set,
    factors,
    annotation,
    populations,
    groups,
    thresholds,
    pop_interest,
    cluster_distance,
    count_sweeps,
    tolerance,
    min_distance,
    flip,
    max_rep,
    nthreads,
    n_shuffles,
    shuffling_segs,
    shuffle_blocks,
    distance_file,
    facet,
    plot_groups,
    seed,
    reuse_bootstrap,
    output_folder,
):
    """
    Gene-set sweep enrichment and FDR analysis.

    Tests whether a set of genes of interest (VIPs) shows a significant excess
    or deficit of positive selection signals relative to matched control gene
    sets. Runs bootstrap control set generation, sweep counting across rank
    thresholds, genome shuffling for a null distribution, and FDR estimation.

    Input files
    -----------
    * **sweep_files**: tab- or space-separated, optionally gzipped. First column
      ``gene_id``; remaining columns are per-population sweep ranks matching
      ``--populations`` order.
    * **gene_set**: one case gene id per line (no header); every other gene of
      the universe (factors ∩ sweep file) is a control candidate.
    * **factors**: confounding factors TSV (no header); first column ``gene_id``.
    * **annotation**: BED file (0-based, no header) ``chr``, ``start``, ``end``,
      ``gene_id``.

    \b
    Example usage:
        flexsweep enrichment \\
            --sweep_files yri_ranks.tsv \\
            --gene_set vip_genes.txt \\
            --factors confounders.tsv \\
            --annotation genes.bed \\
            --populations YRI \\
            --groups AFR \\
            --thresholds 6000,5000,2000,1000,500,200,100,50,20 \\
            --nthreads 8
    """
    from flexsweep.enrichment import np, plot_enrichment, run_enrichment

    if len(groups.split(",")) != len(populations.split(",")):
        raise click.BadParameter(
            f"{len(groups.split(','))} labels for {len(populations.split(','))} "
            "populations; give one group per population, comma-separated",
            param_hint="--groups",
        )

    df_case, df_control, df_curves, df_fdr = run_enrichment(
        sweep_files=sweep_files.split(","),
        gene_set=gene_set,
        factors_file=factors,
        annotation_file=annotation,
        populations=populations.split(","),
        groups=groups.split(","),
        thresholds=[int(x) for x in thresholds.split(",")],
        pop_interest=pop_interest,
        cluster_distance=cluster_distance,
        tolerance=tolerance,
        min_distance=min_distance,
        flip=flip,
        max_rep=max_rep,
        nthreads=nthreads,
        n_shuffles=n_shuffles,
        shuffling_segs=shuffling_segs,
        distance_file=distance_file,
        seed=seed,
        reuse_bootstrap=reuse_bootstrap,
        output_folder=output_folder,
        count_sweeps=count_sweeps,
        shuffle_blocks=shuffle_blocks,
    )

    plot_enrichment(
        df_curves,
        scopes=list(np.unique(groups.split(",")))
        if plot_groups
        else list(np.unique(populations.split(","))),
        facet=facet,
        output_folder=output_folder,
        show=output_folder is None,
    )


@cli.command("enrichment-plot")
@click.option(
    "--curves",
    type=str,
    required=True,
    help="enrichment_curves.txt written by 'flexsweep enrichment --output_folder'.",
)
@click.option(
    "--scopes",
    type=str,
    default=None,
    help="Comma-separated scopes to draw (populations, groups or All). Default: all.",
)
@click.option("--facet", is_flag=True, default=False, help="One panel per scope.")
@click.option("--ci", is_flag=True, default=False, help="Shade bootstrap intervals.")
@click.option("--title", type=str, default="", help="Figure title.")
@click.option(
    "--output_folder",
    type=str,
    default=None,
    help="Folder for the SVGs. Default: the folder of --curves.",
)
def enrichment_plot(curves, scopes, facet, ci, title, output_folder):
    """
    Plot enrichment curves from a finished enrichment run, without rerunning it.

    Writes one ``enrichment_curves_<sweep file name>.svg`` per sweep file in the table.

    \b
    Example usage:
        flexsweep enrichment-plot \\
            --curves enrichment_out/enrichment_curves.txt \\
            --scopes afr,amr,eas,eur,sas \\
            --facet
    """
    from flexsweep.enrichment import _curve_svg_path, plot_enrichment

    if output_folder is None:
        output_folder = os.path.dirname(os.path.abspath(curves))
    try:
        figs = plot_enrichment(
            curves,
            scopes=scopes.split(",") if scopes else None,
            facet=facet,
            ci=ci,
            title=title,
            output_folder=output_folder,
        )
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--scopes")
    for dataset in figs:
        click.echo(f"{dataset} -> {_curve_svg_path(output_folder, dataset)}")


@cli.command()
@click.option(
    "--maf",
    type=str,
    required=True,
    help=" Input MAF file (sorted by contig/pos). Can be gzipped",
)
@click.option(
    "--vcf",
    type=str,
    required=True,
    help=" Input VCF file (sorted by contig/pos). Can be gzipped",
)
@click.option(
    "--outgroups",
    type=str,
    required=True,
    help="Comma-separated list of outgroups as named in the MAF, e.g. panTro4,gorGor3,ponAbe2 (hg38 multiz20way).",
)
@click.option(
    "--method",
    type=str,
    required=True,
    help="Method to polarize VCF file [possible values: parsimony, jc, kimura, r6]",
)
@click.option(
    "--nrandom",
    type=str,
    required=False,
    default=10,
    help="Number of random starts",
)
def polarize(maf, vcf, outgroups, method, nrandom):
    """
    Polarize VCF using rust est-sfs refactor. MAF file anchored to the VCF specie will be used to parse outgroups information.
    Please cite: https://doi.org/10.1534/genetics.118.301120

    Polarization is a streaming merge-join, so both inputs must contain a SINGLE reference
    contig, sorted by position, and must name that contig identically. Prepare them first:

    \b
        flexsweep split-maf --maf 447.primates.maf.gz --contig chr22
        bcftools view -r chr22 all.vcf.gz -Oz -o yri_chr22.vcf.gz

    \b
    Example usage:
        flexsweep polarize --maf 447.primates_chr22.maf.gz --vcf yri_chr22.vcf.gz --outgroups Pan_troglodytes,Gorilla_gorilla,Pongo_abelii --method kimura

    """

    import flexsweep as fs
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        fs.polarize.polarize(maf, vcf, outgroups, method, int(nrandom))
    except Exception as e:
        click.secho(f"Error during polarization: {e}", fg="red")
        raise click.Abort()


@cli.command(name="split-maf")
@click.option(
    "--maf",
    type=str,
    required=True,
    help="Input MAF file to split. Can be gzipped.",
)
@click.option(
    "--contig",
    type=str,
    required=True,
    help="Reference contig to extract, e.g: chr22. Must match the VCF's contig name.",
)
@click.option(
    "--outdir",
    type=str,
    required=False,
    default=None,
    help="Output directory. Defaults to the input MAF's directory.",
)
@click.option(
    "--sort/--no-sort",
    default=True,
    help="Sort the extracted contig by position. On by default: polarize requires it.",
)
@click.option(
    "--tmp-dir",
    type=str,
    required=False,
    default=None,
    help="Directory for the intermediate split and the sort's spill chunks.",
)
@click.option(
    "--chunk-bytes",
    type=int,
    required=False,
    default=268435456,
    help="Maximum bytes held in memory before the sort spills to disk.",
)
def split_maf(maf, contig, outdir, sort, tmp_dir, chunk_bytes):
    """
    Split a MAF by reference contig and sort it by position, ready for polarize.

    \b
    Polarization is a streaming merge-join over the MAF and VCF: its cursor only moves
    forward and never rewinds, so an unsorted or multi-contig MAF makes it silently skip
    variants. This command produces the single-contig, position-sorted MAF it needs.

    \b
    The VCF side is a standard bcftools job, so flexsweep does not duplicate it:
        bcftools view -r chr22 all.vcf.gz -Oz -o yri_chr22.vcf.gz
        tabix -p vcf yri_chr22.vcf.gz

    \b
    Example usage:
        flexsweep split-maf --maf 447.primates.maf.gz --contig chr22
    """

    import flexsweep as fs
    import logging
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.isfile(maf):
        click.secho(f"Error: MAF file not found: {maf}", fg="red")
        raise click.Abort()

    base_name = os.path.basename(maf).replace(".maf.gz", "").replace(".maf", "")
    out_dir = outdir if outdir is not None else os.path.dirname(maf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Named for the contig alone, whether or not sorting ran, so --no-sort output stays
    # drop-in interchangeable and the name never leaks which internal steps happened.
    maf_out = os.path.join(out_dir, f"{base_name}_{contig}.maf.gz")

    scratch = tmp_dir if tmp_dir is not None else tempfile.gettempdir()
    os.makedirs(scratch, exist_ok=True)

    try:
        if sort:
            # The unsorted extract is a full contig of MAF and can run to many GB, so it
            # goes to scratch and is dropped once the sort has consumed it.
            fd, split_tmp = tempfile.mkstemp(
                prefix=f"{base_name}_{contig}_", suffix=".maf.gz", dir=scratch
            )
            os.close(fd)
            try:
                click.echo(f"Splitting MAF for contig: {contig}")
                fs.polarize.maf_split(maf, split_tmp, contig)

                click.echo("Sorting by position...")
                fs.polarize.maf_sort(
                    split_tmp, maf_out, None, int(chunk_bytes), scratch
                )
            finally:
                if os.path.exists(split_tmp):
                    os.unlink(split_tmp)
        else:
            click.echo(f"Splitting MAF for contig: {contig}")
            fs.polarize.maf_split(maf, maf_out, contig)
    except Exception as e:
        click.secho(f"Error during split-maf: {e}", fg="red")
        raise click.Abort()

    click.secho(f"Wrote {maf_out}", fg="green")


@cli.command()
@click.option(
    "--vcf_path",
    "vcf_path",
    type=str,
    required=True,
    help="Directory of *.vcf.gz files, one per chromosome/contig.",
)
@click.option(
    "--out_prefix",
    "out_prefix",
    type=str,
    required=True,
    help="Output prefix. Writes {out}.{stat}.txt for each selected stat.",
)
@click.option(
    "--stats",
    type=str,
    required=True,
    help="Comma-separated stat keys, e.g. ihs,nsl,h12,lassip,raisd. "
    "Run 'python -c \"from flexsweep.scan import available_stats; "
    "print(available_stats())\"' to list all options.",
)
@click.option(
    "--w_size",
    type=int,
    default=201,
    show_default=True,
    help="SNP-count window size for sliding-window stats.",
)
@click.option(
    "--step",
    type=int,
    default=10,
    show_default=True,
    help="SNP step for sliding-window stats.",
)
@click.option(
    "--min_maf",
    type=float,
    default=0.05,
    show_default=True,
    help="Minimum minor allele frequency for iHS and nSL.",
)
@click.option(
    "--recombination_map",
    type=str,
    default=None,
    required=False,
    callback=_resolve_map_option,
    help=MAP_HELP,
)
@click.option(
    "--n_daf_bins",
    type=int,
    default=100,
    show_default=True,
    help="Number of DAF bins for iHS/nSL normalization.",
)
@click.option(
    "--max_extend",
    type=float,
    default=1e5,
    show_default=True,
    help="saltiLASSI spatial decay cutoff in bp.",
)
@click.option(
    "--K_truncation",
    "K_truncation",
    type=int,
    default=10,
    show_default=True,
    help="K truncation for LASSI/saltiLASSI.",
)
@click.option(
    "--sweep_mode",
    type=int,
    default=4,
    show_default=True,
    help="Sweep model for LASSI/saltiLASSI (1-5).",
)
@click.option(
    "--raisd_window",
    type=int,
    default=50,
    show_default=True,
    help="SNP window size for RAISD.",
)
@click.option(
    "--nthreads",
    type=int,
    default=1,
    show_default=True,
    help="Total worker threads for the global task pool. Window-batchable "
    "stats split each chromosome into nthreads window batches.",
)
@click.option(
    "--window_mode",
    default="auto",
    type=click.Choice(["auto", "snp", "bp"]),
    help="Window mode for sliding-window stats. 'auto' uses per-stat defaults "
    "(h12/garud/lassi/lassip/raisd=snp; neutrality/omega/beta=bp).",
)
@click.option(
    "--w_size_bp",
    default=1_000_000,
    type=int,
    help="Physical window size in bp for bp-mode stats (default 1Mb).",
)
@click.option(
    "--step_bp",
    default=10_000,
    type=int,
    help="Physical step size in bp for bp-mode stats (default 10kb).",
)
@click.option(
    "--window_size",
    default=50_000,
    type=int,
    show_default=True,
    help="Focal window size in bp for T3 per-SNP stats (dind, hapdaf_o, "
    "hapdaf_s, s_ratio, high_freq, low_freq).",
)
@click.option(
    "--r_bins",
    type=str,
    default=None,
    help=(
        "Comma-separated recombination-rate bin edges (cM/Mb) for joint DAF x recombination "
        "normalization of per-SNP stats; requires --recombination_map. Compute them with "
        "'flexsweep recombination-bins'. Default: DAF-only normalization."
    ),
)
@click.option(
    "--min_rate",
    type=float,
    default=0.0,
    show_default=True,
    help="Drop windows below this recombination rate (cM/Mb) when --r_bins is set.",
)
@click.option(
    "--tmp_dir",
    type=str,
    default=None,
    help="Folder for joblib temporary memmap files. Default: joblib's (JOBLIB_TEMP_FOLDER, then /dev/shm, then /tmp).",
)
@click.option(
    "--accessibility_mask",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional GIAB-style BED of regions to EXCLUDE (e.g. GRCh38_alldifficultregions.bed.gz), all chromosomes. SNPs inside it are dropped and per-bp pi and theta_w (also in neutrality) are divided by the accessible bases of each window instead of its length.",
)
@click.option(
    "--min_accessible",
    type=click.FloatRange(0, 1),
    default=0.6,
    show_default=True,
    help="With --accessibility_mask: sliding windows whose accessible fraction is below this value are left out of the window-statistic tables.",
)
def scan(
    vcf_path,
    out_prefix,
    stats,
    w_size,
    step,
    min_maf,
    recombination_map,
    n_daf_bins,
    max_extend,
    K_truncation,
    sweep_mode,
    raisd_window,
    nthreads,
    window_mode,
    w_size_bp,
    step_bp,
    window_size,
    r_bins,
    min_rate,
    tmp_dir,
    accessibility_mask,
    min_accessible,
):
    """Standalone outlier scan from a directory of per-chromosome VCF files.

    Computes selected statistics at their natural resolution (per-SNP or sliding
    window) and ranks each genome-wide. No neutral simulations required.
    Writes one tab-separated file per stat: {out}.{stat}.txt.

    Uses a global task pool: all chromosomes are loaded first, then all tasks
    (every stat × chromosome combination) are dispatched in one Parallel call
    to fully exploit --nthreads regardless of chromosome count.

    Global filters apply uniformly to all selected stats of the same class:
    --min_maf applies to both ihs and nsl; --window_size applies to dind,
    hapdaf_o, hapdaf_s, s_ratio, high_freq, and low_freq.

    \b
    Per-SNP stats:    ihs  nsl  isafe  dind  s_ratio  hapdaf_o  hapdaf_s  haf  hscan
    Window stats:     h12  garud  neutrality  omega  lassi  lassip  raisd  beta  ncd

    \b
    Examples:
        flexsweep scan --vcf_path vcf_folder/ --out_prefix YRI --stats ihs,nsl
        flexsweep scan --vcf_path vcf_folder/ --out_prefix YRI --stats ihs,nsl,h12,lassip --nthreads 8
        flexsweep scan --vcf_path vcf_folder/ --out_prefix YRI \\
            --stats dind,hapdaf_o,hapdaf_s --window_size 100000
    """
    from flexsweep.scan import scan as _scan

    if r_bins is not None and recombination_map is None:
        raise click.UsageError("--r_bins requires --recombination_map.")

    stat_list = [s.strip() for s in stats.split(",")]
    _scan(
        vcf_path=vcf_path,
        out_prefix=out_prefix,
        stats=stat_list,
        w_size=w_size,
        step=step,
        w_size_bp=w_size_bp,
        step_bp=step_bp,
        min_maf=min_maf,
        recombination_map=recombination_map,
        n_daf_bins=n_daf_bins,
        r_bins=parse_float_list(r_bins),
        min_rate=min_rate,
        nthreads=nthreads,
        window_mode=window_mode,
        tmp_dir=tmp_dir,
        accessibility_mask=accessibility_mask,
        min_accessible=min_accessible,
        config={
            "lassi": {"K_truncation": K_truncation, "sweep_mode": sweep_mode},
            "lassip": {
                "K_truncation": K_truncation,
                "sweep_mode": sweep_mode,
                "max_extend": max_extend,
            },
            "raisd": {"window_size": raisd_window},
            "dind": {"window_size": window_size},
            "high_freq": {"window_size": window_size},
            "low_freq": {"window_size": window_size},
            "s_ratio": {"window_size": window_size},
            "hapdaf_o": {"window_size": window_size},
            "hapdaf_s": {"window_size": window_size},
        },
    )


EXAMPLE_STEPS = ("data", "simulate", "fvs-sims", "fvs-vcf", "cnn", "scan")


@cli.command("example")
@click.option(
    "--output_folder", type=str, required=True, help="Folder for every output of the example."
)
@click.option(
    "--nthreads", type=int, required=True, help="Threads for every step (e.g. 24)."
)
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Whole chr22 instead of chr22:20-25 Mb.",
)
@click.option(
    "--num_simulations",
    type=int,
    default=1000,
    show_default=True,
    help="Neutral and sweep replicates (each) for the CNN training set.",
)
@click.option(
    "--seed", type=int, default=1, show_default=True, help="Seed for the sample draw, simulations and CNN."
)
@click.option(
    "--steps",
    type=str,
    default=",".join(EXAMPLE_STEPS),
    show_default=True,
    help="Comma-separated subset of steps to run, in this order.",
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Rerun steps already finished."
)
@click.pass_context
def example(ctx, output_folder, nthreads, full, num_simulations, seed, steps, overwrite):
    """
    Run the full example on 50 YRI: data, simulations, feature vectors, CNN and scan.

    \b
    Steps (all by default; on rerun, a step is skipped if it finished and its
    outputs still exist; rerunning a step also reruns the steps that use it):
      data      download the Zenodo record (YRI chr22 VCF, great-ape MAF, GIAB
                difficult regions), draw 50 YRI, polarize and mask chr22:20-25 Mb
                (--full: whole chr22)                            -> data/vcfs/
      simulate  YRI neutral and sweep replicates with discoal    -> sims/
      fvs-sims  feature vectors of the simulations               -> sims/fvs.parquet
      fvs-vcf   feature vectors of the VCF (deCODE map)          -> data/vcfs/fvs_yri.parquet
      cnn       train the CNN on the simulations, predict the VCF -> cnn/
      scan      iHS, nSL, H12 and RAiSD outlier scan and plots   -> scan/
    Wall-clock time per step is printed and appended to timings.tsv.

    \b
    Example usage:
        flexsweep example --output_folder yri_example --nthreads 24
        flexsweep example --output_folder yri_chr22 --nthreads 24 --full
    """
    import time

    from flexsweep.example import EXAMPLE_N_SAMPLES, fetch_example
    from flexsweep.simulate_discoal import resolve_demes, resolve_recombination_map

    todo = [s.strip() for s in steps.split(",") if s.strip()]
    unknown = sorted(set(todo) - set(EXAMPLE_STEPS))
    if unknown:
        raise click.BadParameter(
            f"unknown step(s) {unknown}; choose from {','.join(EXAMPLE_STEPS)}",
            param_hint="--steps",
        )
    todo = [s for s in EXAMPLE_STEPS if s in todo]

    out = os.path.abspath(output_folder)
    data_dir, vcf_dir = os.path.join(out, "data"), os.path.join(out, "data", "vcfs")
    sims_dir, cnn_dir = os.path.join(out, "sims"), os.path.join(out, "cnn")
    scan_prefix = os.path.join(out, "scan", "yri")
    done_dir = os.path.join(out, ".done")
    os.makedirs(done_dir, exist_ok=True)

    # A folder holds one region: refuse to mix 5 Mb and whole-chr22 outputs
    mode = "chr22" if full else "chr22_20_25Mb"
    mode_file = os.path.join(out, "mode.txt")
    if os.path.exists(mode_file):
        with open(mode_file) as fh:
            prev = fh.read().strip()
        if prev != mode and not overwrite:
            raise click.UsageError(
                f"{out} holds the {prev} example; use another --output_folder or --overwrite."
            )
        if prev != mode:  # --overwrite to another region: earlier steps are stale
            for f in os.listdir(done_dir):
                os.remove(os.path.join(done_dir, f))
    with open(mode_file, "w") as fh:
        fh.write(mode + "\n")

    def run_data():
        fetch_example(data_dir, full=full, seed=seed, overwrite=overwrite)

    def run_simulate():
        ctx.invoke(
            simulator,
            sample_size=2 * EXAMPLE_N_SAMPLES,
            demes=resolve_demes("yri"),
            output_folder=sims_dir,
            num_simulations=num_simulations,
            nthreads=nthreads,
            seed=seed,
        )

    def run_fvs_sims():
        ctx.invoke(fvs_discoal, simulations_path=sims_dir, nthreads=nthreads)

    def run_fvs_vcf():
        ctx.invoke(
            fvs_vcf,
            vcf_path=vcf_dir,
            recombination_map=resolve_recombination_map("decode"),
            suffix="yri",
            nthreads=nthreads,
        )

    def run_cnn():
        ctx.invoke(
            cnn,
            train_data=os.path.join(sims_dir, "fvs.parquet"),
            predict_data=os.path.join(vcf_dir, "fvs_yri.parquet"),
            output_folder=cnn_dir,
            seed=seed,
        )

    def run_scan():
        stats = ("ihs", "nsl", "h12", "raisd")
        os.makedirs(os.path.dirname(scan_prefix), exist_ok=True)
        ctx.invoke(
            scan,
            vcf_path=vcf_dir,
            out_prefix=scan_prefix,
            stats=",".join(stats),
            nthreads=nthreads,
        )
        # Outlier plots: values with the top 1% highlighted, and -log10 empirical p-values
        # (SVG keeps text editable; PNG for viewers that draw no SVG text)
        files = tuple(f"{scan_prefix}.{s}.txt" for s in stats)
        for ext in ("svg", "png"):
            ctx.invoke(
                scan_plot, stat_files=files, multi=True, out_file=f"{scan_prefix}.scan.{ext}"
            )
            ctx.invoke(
                scan_plot,
                stat_files=files,
                multi=True,
                pvalue=True,
                out_file=f"{scan_prefix}.scan_pvalue.{ext}",
            )

    runners = {
        "data": run_data,
        "simulate": run_simulate,
        "fvs-sims": run_fvs_sims,
        "fvs-vcf": run_fvs_vcf,
        "cnn": run_cnn,
        "scan": run_scan,
    }
    # A step is done when it finished (marker) and its outputs still exist
    outputs = {
        "data": [os.path.join(vcf_dir, "yri_chr22_example.vcf.gz.tbi")],
        "simulate": [os.path.join(sims_dir, "params.txt.gz")],
        "fvs-sims": [os.path.join(sims_dir, "fvs.parquet")],
        "fvs-vcf": [os.path.join(vcf_dir, "fvs_yri.parquet")],
        "cnn": [os.path.join(cnn_dir, "yri_predictions.txt")],
        "scan": [f"{scan_prefix}.scan{k}.{e}" for k in ("", "_pvalue") for e in ("svg", "png")],
    }
    # Steps that read a step's outputs; rerunning a step makes them stale
    users = {
        "data": ["fvs-vcf", "scan"],
        "simulate": ["fvs-sims"],
        "fvs-sims": ["cnn"],
        "fvs-vcf": ["cnn"],
        "cnn": [],
        "scan": [],
    }

    def is_done(step):
        marker = os.path.join(done_dir, step)
        return os.path.exists(marker) and all(os.path.exists(f) for f in outputs[step])

    def invalidate(step):
        for u in users[step]:
            marker = os.path.join(done_dir, u)
            if os.path.exists(marker):
                os.remove(marker)
            invalidate(u)

    n = len(EXAMPLE_STEPS)
    timings = []
    for step in todo:
        i = EXAMPLE_STEPS.index(step) + 1
        if is_done(step) and not overwrite:
            click.secho(f"[{i}/{n}] {step}: already done, skipping", fg="cyan")
            continue
        click.secho(f"[{i}/{n}] {step} ...", fg="cyan", bold=True)
        invalidate(step)
        t0 = time.perf_counter()
        runners[step]()
        dt = time.perf_counter() - t0
        open(os.path.join(done_dir, step), "w").close()
        timings.append((step, dt))
        click.secho(f"[{i}/{n}] {step} done in {dt / 60:.1f} min", fg="green")

    if timings:
        tsv = os.path.join(out, "timings.tsv")
        new = not os.path.exists(tsv)
        with open(tsv, "a") as fh:
            if new:
                fh.write("step\tseconds\tregion\tnthreads\tnum_simulations\tseed\n")
            for step, dt in timings:
                fh.write(f"{step}\t{dt:.1f}\t{mode}\t{nthreads}\t{num_simulations}\t{seed}\n")
        total = sum(dt for _, dt in timings)
        click.secho(f"Finished {len(timings)} step(s) in {total / 60:.1f} min ({tsv})", bold=True)
    click.echo(
        f"Predictions: {cnn_dir}/  Scan: {os.path.dirname(scan_prefix)}/  "
        f"VCF: {vcf_dir}/"
    )


@cli.command("scan-plot")
@click.option(
    "--stat",
    "stat_files",
    type=str,
    multiple=True,
    required=True,
    help="Scan output file ({prefix}.{stat}.txt). Repeat for several stats.",
)
@click.option(
    "--col",
    "stat_cols",
    type=str,
    multiple=True,
    help="Column to plot per --stat, in the same order. Default: inferred from the "
    "file name: the DAF-standardized {rank}_std column when the file has one "
    "(yri.ihs.txt -> ihs_std), else the stat's rank column (yri.raisd.txt -> mu_total).",
)
@click.option(
    "--multi",
    is_flag=True,
    default=False,
    help="Stack all stats in one figure. Default: one figure per stat.",
)
@click.option(
    "--split_chr",
    is_flag=True,
    default=False,
    help="One figure per chromosome. Default: all chromosomes in one figure, "
    "separated by dashed lines.",
)
@click.option(
    "--pvalue",
    is_flag=True,
    default=False,
    help="Plot -log10 of the {col}_pvalue column instead of raw values.",
)
@click.option(
    "--top_pct",
    type=click.FloatRange(0, 0.5, min_open=True, max_open=True),
    default=0.01,
    show_default=True,
    help="Fraction highlighted per tail; signed stats (iHS, nSL, ...) get both tails.",
)
@click.option(
    "--out",
    "out_file",
    type=str,
    default="scan_plot.svg",
    show_default=True,
    help="Output path; SVG unless the extension says otherwise (.png, .pdf). "
    "Without --multi, '.{col}' is added per stat; with --split_chr, '.chr{N}' "
    "per chromosome.",
)
def scan_plot(stat_files, stat_cols, multi, split_chr, pvalue, top_pct, out_file):
    """Visualize outlier scan results.

    \b
    One figure per stat, all chromosomes (scan_plot.ihs.svg, scan_plot.nsl.svg):
        flexsweep scan-plot --stat yri.ihs.txt --stat yri.nsl.txt
    All stats stacked in one figure:
        flexsweep scan-plot --stat yri.ihs.txt --stat yri.nsl.txt --multi
    Add --split_chr to either for one figure per chromosome (.chr{N}).
    """
    import os

    import polars as pl

    from flexsweep.scan import STAT_REGISTRY
    from flexsweep.utils import default_scan_col, plot_scan

    stat_files = list(stat_files)
    if stat_cols:
        stat_cols = list(stat_cols)
        if len(stat_files) != len(stat_cols):
            raise click.UsageError(
                "Give one --col per --stat, or omit --col to infer it."
            )
    else:
        stat_cols = []
        for f in stat_files:
            parts = os.path.basename(f).split(".")
            key = parts[-2] if len(parts) >= 3 else None
            if key not in STAT_REGISTRY:
                raise click.UsageError(
                    f"Cannot infer the column for '{f}' (expected "
                    "{prefix}.{stat}.txt from flexsweep scan); pass --col."
                )
            header = pl.read_csv(f, separator="\t", n_rows=0).columns
            stat_cols.append(default_scan_col(key, header))

    try:
        plot_scan(
            stat_files,
            stat_cols,
            pvalue=pvalue,
            top_pct=top_pct,
            out=out_file,
            multi=multi,
            split_chr=split_chr,
        )
    except ValueError as e:
        raise click.UsageError(str(e))


if __name__ == "__main__":
    cli()
