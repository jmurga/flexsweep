import click
import ast


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
@click.option(
    "--sample_size", type=int, required=True, help="Sample size for the simulation"
)
@click.option(
    "--mutation_rate",
    type=str,
    required=True,
    help="Mutation rate. For two comma-separated values, the first will be used as the lower bound and the second as the upper bound for a uniform distribution. A single value will be treated as the mean for an exponential distribution.",
)
@click.option(
    "--recombination_rate",
    type=str,
    required=True,
    help="Mutation rate. For two comma-separated values, the first will be used as the lower bound and the second as the upper bound for a uniform distribution. A single value will be treated as the mean for an exponential distribution.",
)
@click.option("--locus_length", type=int, required=True, help="Length of the locus")
@click.option(
    "--demes", type=str, required=True, help="Path to the demes YAML model file"
)
@click.option(
    "--output_folder",
    type=str,
    required=True,
    help="Folder where outputs will be saved",
)
@click.option(
    "--time",
    type=str,
    default="0,5000",
    help="Start/end adaptive mutation range timing",
)
@click.option(
    "--num_simulations",
    type=int,
    default=int(1e4),
    help="Number of neutral and sweep simulations",
)
@click.option(
    "--nthreads", type=int, default=1, help="Number of threads for parallelization"
)
@click.option(
    "--discoal_path",
    type=str,
    default=None,
    help="Path to the discoal executable",
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
    """Run the discoal Simulator"""

    import flexsweep as fs

    if discoal_path is None:
        discoal_path = fs.DISCOAL

    mutation_rate_list = parse_float_list(None, None, mutation_rate)
    recombination_rate_list = parse_float_list(None, None, recombination_rate)
    time_list = parse_float_list(None, None, time)

    if len(mutation_rate_list) == 2:
        mu_rate = {
            "dist": "uniform",
            "lower": mutation_rate_list[0],
            "upper": mutation_rate_list[1],
        }
    elif len(mutation_rate_list) == 1:
        mu_rate = {
            "dist": "exponential",
            "mean": mutation_rate_list[0],
        }

    if len(recombination_rate_list) == 2:
        rho_rate = {
            "dist": "uniform",
            "lower": recombination_rate_list[0],
            "upper": recombination_rate_list[1],
        }
    elif len(recombination_rate_list) == 1:
        rho_rate = {
            "dist": "exponential",
            "mean": recombination_rate_list[0],
        }

    # Instantiate Simulator and run it
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
    simulator.simulate()


@cli.command()
@click.option(
    "--simulations_path",
    type=str,
    required=True,
    help="Path containing neutral and sweeps discoal simulations.",
)
@click.option("--nthreads", type=int, required=True, help="Number of threads")
def fvs_discoal(simulations_path, nthreads):
    """Run the summary statistic estimation from discoal simulation to create CNN input feature vectors.
    Will create two file: a parquet dataframe and a pickle dictionary containing neutral expectation and stdev
    """
    import flexsweep as fs

    print("Estimating summary statistics")
    df_fv = fs.summary_statistics(simulations_path, nthreads=nthreads)


@cli.command()
@click.option(
    "--vcf", type=str, required=True, help="VCF file to parse. Must be indexed"
)
@click.option(
    "--neutral_bin",
    type=str,
    required=True,
    help="Neutral bin data from discoal simulations",
)
@click.option("--contig_name", type=str, required=True, help="Chromosome name")
@click.option(
    "--contig_len", type=str, required=True, help="Chromosome length for sliding"
)
@click.option("--window_size", type=int, required=True, help="Window size")
@click.option("--step", type=int, required=True, help="Sliding step")
def fvs_vcf(vcf, contig_name, contig_length, window_size, step, nthreads):
    """Run the summary statistic estimation from a VCF file to create CNN input feature vectors."""
    import flexsweep as fs

    data = fs.Data(vcf, window_size=window_size, step=step, nthreads=nthreads)
    data_gt = fs.read_vcf(contig_name, contig_length)


@cli.command()
@click.option("--data", type=str, required=True, help="Path to the training data")
@click.option(
    "--output_folder",
    type=str,
    required=True,
    help="Output folder for the CNN model and logs",
)
@click.option(
    "--output_prediction",
    type=str,
    default="predictions.txt",
    help="Prediction file name. Saved on output_folder",
)
@click.option(
    "--mode",
    type=click.Choice(["train", "predict"]),
    required=True,
    help="Mode: 'train' or 'predict'",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Input a pretrained model",
)
def cnn(data, output_folder, mode):
    """Run the Flexsweep CNN"""
    import flexsweep as fs

    fs_cnn = fs.CNN(data, output_folder, output_prediction)
    if mode == "train":
        fs_cnn.train()
        d_prediction = fs_cnn.predict()

    if mode == "predict":
        if model is not None:
            fs_cnn.model = model
        fs_cnn.predict()


if __name__ == "__main__":
    cli()
