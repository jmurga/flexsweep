"""CNN.train exposes its optimizer and early-stopping hyperparameters.

Reviewer 2 (MBE-26-0872) asked whether hyperparameters can be searched over a grid. The
learning rate, batch size, early-stopping patience/min_delta and maximum epochs were
hardcoded in CNN.train, so an external optimizer (Optuna, KerasTuner) could not vary them.
Oracle: a non-default call must reach the compiled optimizer and stop within the requested
number of epochs, and the `dann` command's defaults must equal `CNN.train_da`'s.
"""

import inspect

import numpy as np
import polars as pl
import pytest

from flexsweep.cnn import CNN

STATS = [
    "dind", "dist_kurtosis", "dist_skew", "dist_var", "h1", "h12", "h2_h1", "haf",
    "hapdaf_o", "hapdaf_s", "high_freq", "ihs", "isafe", "k_counts", "low_freq", "max_fda",
    "nsl", "omega_max", "pi", "s_ratio", "tajima_d", "theta_h", "theta_w", "zns",
]
CENTERS = np.arange(50_000, 1_200_000, 100_000)  # CNN default: 12 centers, 100 kb windows


def _toy_parquet(path, n_per_class=64, seed=0):
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    cols = {
        "iter": np.arange(1, n + 1),
        "s": np.zeros(n), "t": np.zeros(n), "f_i": np.zeros(n), "f_t": np.zeros(n),
        "mu": np.full(n, 1e-8), "r": np.full(n, 1e-8),
        "model": ["neutral"] * n_per_class + ["hard_old_complete"] * n_per_class,
    }
    shift = np.r_[np.zeros(n_per_class), np.ones(n_per_class)]
    for s in STATS:
        for c in CENTERS:
            cols[f"{s}_100000_{c}"] = rng.normal(size=n) + shift
    pl.DataFrame(cols).write_parquet(path)


@pytest.mark.slow
def test_hyperparameters_reach_training(tmp_path):
    data = tmp_path / "fvs_toy.parquet"
    _toy_parquet(data)
    m = CNN(train_data=str(data), output_folder=str(tmp_path), seed=1)
    m.train(learning_rate=3e-3, batch_size=16, patience=1, min_delta=0.0, epochs=2)
    opt = m.model.optimizer
    sched = getattr(opt, "_learning_rate", opt.learning_rate)  # Keras 3 keeps the schedule here
    initial = getattr(sched, "initial_learning_rate", sched)
    assert float(initial) == pytest.approx(3e-3)
    assert 1 <= m.history.height <= 2


def test_dann_api_defaults_match_the_cli():
    """`CNN.train_da()` and `flexsweep dann` must not drift apart.

    They used to disagree on three of them (tgt_ratio 1 vs 2, batch_size 128 vs 64,
    preprocess True vs False), so calling the API without arguments trained a different
    model from the one the command trains, and from the one the paper reports.
    """
    from click.testing import CliRunner  # noqa: F401  (import guards a missing click)

    from flexsweep.main import cli

    dann = cli.commands["dann"]
    cli_defaults = {o.name: o.default for o in dann.params}
    sig = inspect.signature(CNN.train_da).parameters
    for cli_name, api_name in (("target_ratio", "tgt_ratio"), ("max_lambda", "max_lambda"),
                               ("ramp_epochs", "ramp_epochs"), ("batch_size", "batch_size"),
                               ("preprocess", "preprocess")):
        assert cli_defaults[cli_name] == sig[api_name].default, (
            f"{cli_name} (CLI {cli_defaults[cli_name]}) != {api_name} "
            f"(API {sig[api_name].default})")
    # predict_da must not standardize twice either
    assert inspect.signature(CNN.predict_da).parameters["preprocess"].default is False
