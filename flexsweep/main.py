import click
import ast
import os


def parse_float_list(ctx, param, value):
    """Parse a comma-separated list of floats."""
    if value:
        try:
            return [int(x) if x.isdigit() else float(x) for x in value.split(",")]
        except ValueError:
            raise click.BadParameter("Must be a comma-separated list of floats")
    return []


@click.group()
def cli():
    """CLI for Simulator and CNN."""
    pass


@cli.command()
@click.option("--sample_size", type=int, required=True, help="Number of haplotypes")
@click.option(
    "--mutation_rate",
    type=str,
    required=False,
    default="5e-9,2e-8",
    help=(
        "Mutation rate specification. "
        "Please input:\n"
        "  - Two comma-separated values: lower,upper (uniform distribution bounds):"
        "  - Three values: min, max and mean of an exponential distribution."
        "Example: '5e-9,2e-8' or '5e-9,2e-8,1e-8'"
    ),
)
@click.option(
    "--recombination_rate",
    type=str,
    required=False,
    default="1e-9,4e-8,1e-8",
    help=(
        "Recombination rate specification. "
        "Please input:"
        "  - Two comma-separated values: lower,upper (uniform distribution bounds):"
        "  - Three values: min, max and mean of an exponential distribution."
        "Example: '1e-9,4e-8' or '1e-9,4e-8,1e-8'"
    ),
)
@click.option(
    "--locus_length",
    type=int,
    required=False,
    default=int(1.2e6),
    help="Length of the simulated locus in base pairs.",
)
@click.option(
    "--demes",
    type=str,
    required=True,
    help="Path to the demes YAML file describing demography.",
)
@click.option(
    "--output_folder",
    type=str,
    required=True,
    help="Directory where simulation outputs will be saved.",
)
@click.option(
    "--time",
    type=str,
    default="0,5000",
    help=(
        "Adaptive mutation time range in generations. "
        "Two comma-separated values: start,end. "
        "Default: '0,5000'"
    ),
)
@click.option(
    "--num_simulations",
    type=int,
    default=int(1e4),
    help="Number of neutral and sweep simulations to generate. Default: 10000.",
)
@click.option(
    "--nthreads",
    type=int,
    default=1,
    help="Number of threads for parallelization. Default: 1.",
)
@click.option(
    "--discoal_path",
    type=str,
    default=None,
    help=(
        "Path to the discoal executable. If not provided, using pre-compiled flexsweep.DISCOAL."
    ),
)
def simulator(
    sample_size,
    mutation_rate,
    recombination_rate,
    locus_length,
    demes,
    output_folder,
    discoal_path,
    num_simulations,
    time,
    nthreads,
):
    """
    Run the discoal Simulator with user-specified parameters.

    flexsweep.Simulator class, parsing mutation and
    recombination rate specifications from the command line, and dispatches
    neutral and sweep simulations to discoal.

    \b
    Example usage:
        flexsweep simulate --sample_size 20 --demes model.yaml --output_folder ./sims --nthreads 24

    """
    import flexsweep as fs

    # If not provided explicitly, use default path from flexsweep
    if discoal_path is None:
        discoal_path = fs.DISCOAL

    # Parse mutation and recombination inputs
    mutation_rate_list = parse_float_list(None, None, mutation_rate)
    recombination_rate_list = parse_float_list(None, None, recombination_rate)

    # Build mutation rate distribution spec
    if len(mutation_rate_list) == 2:
        mu_rate = {
            "dist": "uniform",
            "min": mutation_rate_list[0],
            "max": mutation_rate_list[1],
        }
    elif len(mutation_rate_list) == 3:
        mu_rate = {
            "dist": "exponential",
            "min": mutation_rate_list[0],
            "max": mutation_rate_list[1],
            "mean": mutation_rate_list[2],
        }

    # Build recombination rate distribution spec
    if len(recombination_rate_list) == 2:
        rho_rate = {
            "dist": "uniform",
            "lower": recombination_rate_list[0],
            "upper": recombination_rate_list[1],
        }
    elif len(recombination_rate_list) == 3:
        rho_rate = {
            "dist": "exponential",
            "min": recombination_rate_list[0],
            "max": recombination_rate_list[1],
            "mean": recombination_rate_list[2],
        }

    # Parse time range (not directly used in this wrapper, passed to Simulator internally)
    time_list = parse_float_list(None, None, time)

    # Instantiate Simulator and run simulations
    simulator = fs.Simulator(
        sample_size=sample_size,
        mutation_rate=mu_rate,
        recombination_rate=rho_rate,
        locus_length=locus_length,
        demes=demes,
        output_folder=output_folder,
        discoal_path=fs.DISCOAL,
        num_simulations=num_simulations,
        nthreads=nthreads,
    )
    simulator.create_params()
    simulator.simulate()


@cli.command()
@click.option(
    "--simulations_path",
    type=str,
    required=True,
    help="Directory containing neutral and sweeps discoal simulations.",
)
@click.option(
    "--nthreads", type=int, required=True, help="Number of threads for parallelization"
)
def fvs_discoal(simulations_path, nthreads):
    """
    Estimate summary statistics from discoal simulations and build feature vectors.

    This command processes both neutral and sweep simulations in the given directory,
    computes a panel of summary statistics, and generates two outputs: a Parquet dataframe containing feature vectors, a Pickle dictionary containing neutral expectations and standard deviations (used for normalization during CNN training).
    """
    import flexsweep as fs

    print("Estimating summary statistics")
    df_fv = fs.summary_statistics(simulations_path, nthreads=nthreads)


@cli.command()
@click.option(
    "--vcf_path",
    type=str,
    required=True,
    help="Directory containing vcfs folder with all the VCF files to analyze.",
)
@click.option(
    "--nthreads", type=int, required=True, help="Number of threads for parallelization"
)
@click.option(
    "--recombination_map",
    type=str,
    default=None,
    required=False,
    help="Recombination map. Decode CSV format: Chr,Begin,End,cMperMb,cM",
)
@click.option("--pop", type=str, default="pop", required=False, help="Population ID")
def fvs_vcf(vcf_path, recombination_map, nthreads):
    """
    Estimate summary statistics from VCF files and build feature vectors.

    This command parses VCF files in the given directory, computes summary statistics
    per genomic window, and writes feature vectors suitable as CNN input.

    \b
    Example usage:
        # Run summary statistics from VCFs using 8 threads, no recombination map
        flexsweep fvs-vcf --vcf_path ./data --nthreads 8
    \b
        # Run with a recombination map
        flexsweep fvs-vcf --vcf_path ./data --nthreads 8 --recombination_map recomb_map.csv

    Notes: VCF files must be bgzipped and tabix-indexed.
    """
    import flexsweep as fs

    df_fv = fs.summary_statistics(
        vcf_path,
        nthreads=nthreads,
        vcf=True,
        recombination_map=recombination_map,
        population=pop,
    )


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
def cnn(train_data, predict_data, output_folder, model):
    """
    Run the Flexsweep CNN for training or prediction.

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

    if model is None:
        if not train_data:
            raise click.UsageError(
                "--train_data is required when --model is not provided."
            )
        fs_cnn = fs.CNN(
            train_data=train_data,
            predict_data=predict_data,
            output_folder=output_folder,
        )
        fs_cnn.train()
        fs_cnn.predict()

    else:
        if not predict_data:
            raise click.UsageError(
                "--predict_data is required when --model is provided."
            )
        fs_cnn = fs.CNN(
            predict_data=predict_data, output_folder=output_folder, model=model
        )
        fs_cnn.predict()


if __name__ == "__main__":
    cli()
