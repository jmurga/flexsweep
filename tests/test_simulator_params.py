"""Simulator rate specifications and shipped demes aliases (no discoal run).

Oracle: a fixed rate must give the same per-bp value in every replicate; every
shipped alias must resolve to a file that demes can load.
"""

import demes
import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

import flexsweep as fs
from flexsweep.main import cli


def _params(tmp_path, mu, r, n=50):
    sim = fs.Simulator(
        sample_size=20,
        demes="constant",
        output_folder=str(tmp_path / "sims"),
        mutation_rate=mu,
        recombination_rate=r,
        num_simulations=n,
        seed=1,
    )
    sim.demes = fs.resolve_demes(sim.demes)
    sim.create_params()
    return pl.read_csv(tmp_path / "sims" / "params.txt.gz")


def test_fixed_rates(tmp_path):
    df = _params(tmp_path, {"dist": "fixed", "value": 1.25e-8}, {"dist": "fixed", "value": 1e-8})
    assert np.allclose(df["mu"].to_numpy(), 1.25e-8)
    assert np.allclose(df["r"].to_numpy(), 1e-8)


def test_all_1000gp_aliases_load():
    assert len(fs.POPULATIONS_1KG) == 26
    for pop in fs.POPULATIONS_1KG:
        for key in (pop, pop.lower()):
            path = fs.resolve_demes(key)
            graph = demes.load(path)
            assert graph.demes[0].name == pop, (key, graph.demes[0].name)


def test_other_aliases_and_basename():
    assert demes.load(fs.resolve_demes("constant"))
    assert demes.load(fs.resolve_demes("Africa_1T12"))
    assert fs.resolve_demes("yri_spiedel_2019_full.yaml") == fs.DEMES_EXAMPLES["yri"]
    with pytest.raises(FileNotFoundError, match="1000 Genomes population code"):
        fs.resolve_demes("nope")


def _capture(monkeypatch, tmp_path, extra):
    seen = {}

    class Dummy:
        def __init__(self, **kw):
            seen.update(kw)

        def create_params(self):
            pass

        def simulate_batch(self):
            pass

    monkeypatch.setattr(fs, "Simulator", Dummy)
    res = CliRunner().invoke(
        cli,
        ["simulator", "--sample_size", "10", "--demes", "constant", "--output_folder",
         str(tmp_path), *extra],
    )
    assert res.exit_code == 0, res.output
    return seen


@pytest.mark.parametrize("bad", [["--x", "1.5"], ["--fixed_ratio", "2"], ["--s", "0.1"]])
def test_cli_rejects_bad_values(bad, tmp_path):
    res = CliRunner().invoke(
        cli, ["simulator", "--sample_size", "10", "--demes", "constant",
              "--output_folder", str(tmp_path), *bad],
    )
    assert res.exit_code != 0


def test_cli_help_defaults_match_simulator_signature():
    """The [default: ...] shown by --help must be the Simulator default."""
    import inspect
    import re

    sig = inspect.signature(fs.Simulator.__init__).parameters
    cmd = cli.commands["simulator"]
    helps = {p.name: p.help for p in cmd.params}

    def shown(name):
        return re.search(r"\[default: ([^\]]+)\]", helps[name]).group(1)

    def nums(text):
        return [float(v) for v in text.split(",")]

    mu, r = sig["mutation_rate"].default, sig["recombination_rate"].default
    assert nums(shown("mutation_rate")) == [mu["min"], mu["max"]]
    assert nums(shown("recombination_rate")) == [r["min"], r["max"], r["mean"]]
    for name in ("s", "time", "saf", "eaf"):
        assert nums(shown(name)) == [float(v) for v in sig[name].default], name
    for name in ("num_simulations", "locus_length", "fixed_ratio", "x", "nthreads"):
        assert float(shown(name)) == float(sig[name].default), name


def test_cli_single_value_is_fixed(monkeypatch, tmp_path):
    seen = _capture(monkeypatch, tmp_path, ["--mutation_rate", "1e-8", "--recombination_rate", "1e-8"])
    assert seen["mutation_rate"] == seen["recombination_rate"] == {"dist": "fixed", "value": 1e-8}
