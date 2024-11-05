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
@click.option(
    "--sample_size", type=int, required=True, help="Sample size for the simulation"
)
@click.option(
    "--mutation_rate",
    type=str,
    required=True,
    help="Mutation rate. Two comma-separated values, the first will be used as the lower bound and the second as the upper bound for a uniform distribution.",
)
@click.option(
    "--recombination_rate",
    type=str,
    required=True,
    help="Mutation rate. Two comma-separated values, the first will be used as the lower bound and the second as the upper bound for a uniform distribution.",
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

    mu_rate = parse_float_list(None, None, mutation_rate)
    rho_rate = parse_float_list(None, None, recombination_rate)

    assert (
        len(mu_rate) == 2 or len(rho_rate) == 2
    ), "Please input two comma-separated values as lower and upper values to draw values uniform distribution, e.g: 5e-9,2e-8"

    time_list = parse_float_list(None, None, time)

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
@click.option("--step", type=int, required=True, help="Sliding step")
@click.option("--nthreads", type=int, required=True, help="Number of threads")
@click.option("--rec_map", type=str, required=False, help="Recombination map")
def fvs_vcf(vcf, contig_name, contig_length, window_size, step, nthreads):
    """Run the summary statistic estimation from a VCF file to create CNN input feature vectors."""
    import flexsweep as fs

    fs_data = fs.Data(
        vcf,
        step=step,
        nthreads=nthreads,
        recombination_map=rec_map,
    )
    data = fs_data.read_vcf(contig_name, contig_length)

    df_fv = fs.summary_statistics(data, nthreads=nthreads)


@cli.command()
@click.option(
    "--mode",
    type=click.Choice(["train", "predict"]),
    required=True,
    help="Mode: 'train' or 'predict'",
)
@click.option("--data", type=str, required=True, help="Path to the training data")
@click.option(
    "--output_folder",
    type=str,
    required=True,
    help="Output folder for the CNN model and logs",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Input a pretrained model",
)
def cnn(data, output_folder, mode, model):
    """Run the Flexsweep CNN"""
    import flexsweep as fs

    os.makedirs(output_folder, exist_ok=True)

    fs_cnn = fs.CNN(data, output_folder)
    if mode == "train":
        fs_cnn.train()
        df_prediction = fs_cnn.predict()
        p_roc, p_history = fs_cnn.roc_curve()

    elif mode == "predict":
        assert model is None, "Please input a model to make predictions"

        fs_cnn.model = model
        fs_cnn.predict()


if __name__ == "__main__":
    cli()
