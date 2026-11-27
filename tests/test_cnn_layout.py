"""CNN input layout for 2D and 1D models and for non-uniform window/center geometries.

Reviewer 1 (MBE-26-0872, point 1d) asked for custom architectures and inputs. The flat
feature row is stat-major: S blocks of P = windows x centers columns. Oracle: every
column holds a constant code(stat, position), so a model that returns one input cell
must output exactly the code of that cell. The 2D model reads x[:, s, p, 0]; a 1D model
(one_dim=True) reads x[:, p, s]. A geometry with several windows and centers off the
uniform grid of the CNN constructor must load from the column names alone.
"""

import numpy as np
import polars as pl
import pytest
import tensorflow as tf

from flexsweep.cnn import CNN

STATS = ["h12", "ihs", "pi", "tajima_d"]
BASE = [(100000, c) for c in range(50_000, 1_200_000, 100_000)]  # 1 window x 12 centers
UNEVEN = [(w, c) for w in (4000, 64000) for c in (280000, 600000, 920000)]


def code(s: int, p: int, n_pos: int) -> float:
    return (s * n_pos + p + 1) / (len(STATS) * n_pos + 2)


def toy(path, keys, n_per_class=20):
    n = 2 * n_per_class
    cols = {
        "iter": np.arange(1, n + 1),
        "s": np.zeros(n), "t": np.zeros(n), "f_i": np.zeros(n), "f_t": np.zeros(n),
        "mu": np.full(n, 1e-8), "r": np.full(n, 1e-8),
        "model": ["neutral"] * n_per_class + ["hard_old_complete"] * n_per_class,
    }
    for si, s in enumerate(STATS):
        for p, (w, c) in enumerate(keys):
            cols[f"{s}_{w}_{c}"] = np.full(n, code(si, p, len(keys)))
    pl.DataFrame(cols).write_parquet(path)
    return str(path)


def picker(shape, index):
    inp = tf.keras.Input(shape)
    out = tf.keras.layers.Lambda(lambda x: tf.reshape(x[(slice(None),) + index], (-1, 1)))(inp)
    return tf.keras.Model(inp, out)


@pytest.mark.parametrize("keys", [BASE, UNEVEN], ids=["base", "uneven"])
@pytest.mark.parametrize("one_dim", [False, True], ids=["2d", "1d"])
def test_training_tensor_layout(tmp_path, keys, one_dim):
    m = CNN(train_data=toy(tmp_path / "fvs.parquet", keys), seed=1)
    m.num_stats = len(STATS)
    X = m.load_training_data(_stats=STATS, one_dim=one_dim)[0]
    P = len(keys)
    if one_dim:
        assert X.shape[1:] == (P, len(STATS))
        got = X[0]
    else:
        assert X.shape[1:] == (len(STATS), P, 1)
        got = X[0, :, :, 0].T
    want = np.array([[code(s, p, P) for s in range(len(STATS))] for p in range(P)])
    np.testing.assert_allclose(got, want)


@pytest.mark.parametrize("keys", [BASE, UNEVEN], ids=["base", "uneven"])
@pytest.mark.parametrize("one_dim", [False, True], ids=["2d", "1d"])
def test_predict_reads_the_right_cell(tmp_path, keys, one_dim):
    P, s0, p0 = len(keys), 2, len(keys) - 1
    shape, index = ((P, len(STATS)), (p0, s0)) if one_dim else ((len(STATS), P, 1), (s0, p0, 0))
    m = CNN(predict_data=toy(tmp_path / "fvs.parquet", keys), model=picker(shape, index),
            output_folder=str(tmp_path))
    pred = m.predict(_stats=STATS)  # layout inferred from the model's input rank
    np.testing.assert_allclose(pred["prob_sweep"].to_numpy(), code(s0, p0, P), atol=1e-6)


def test_mismatched_columns_across_stats_raise(tmp_path):
    path = toy(tmp_path / "fvs.parquet", BASE)
    pl.read_parquet(path).drop(f"pi_100000_{BASE[-1][1]}").write_parquet(path)
    m = CNN(train_data=path, seed=1)
    with pytest.raises(ValueError, match="window/center"):
        m.load_training_data(_stats=STATS)


CENTER_MAJOR = sorted(UNEVEN, key=lambda k: (k[1], k[0]))  # same columns, other order


def test_predict_rejects_a_layout_other_than_training(tmp_path):
    """Same column names in another order would reshape into the wrong cells silently."""
    m = CNN(train_data=toy(tmp_path / "train.parquet", UNEVEN), seed=1)
    m.num_stats = len(STATS)
    m.load_training_data(_stats=STATS)
    m.model = picker((len(STATS), len(UNEVEN), 1), (0, 0, 0))
    m.output_folder = str(tmp_path)

    m.predict_data = toy(tmp_path / "same.parquet", UNEVEN)
    assert m.predict(_stats=STATS).height == 40
    m.predict_data = toy(tmp_path / "center_major.parquet", CENTER_MAJOR)
    with pytest.raises(ValueError, match="window/center columns differ"):
        m.predict(_stats=STATS)
    m.predict_data = toy(tmp_path / "same2.parquet", UNEVEN)
    with pytest.raises(ValueError, match="statistics differ"):
        m.predict(_stats=STATS[::-1])


def test_layout_saved_next_to_the_model(tmp_path):
    m = CNN(train_data=toy(tmp_path / "train.parquet", UNEVEN), seed=1)
    m.num_stats = len(STATS)
    m.load_training_data(_stats=STATS)
    m._save_layout(str(tmp_path / "model.keras"))

    p = CNN(model=str(tmp_path / "model.keras"))
    p._check_layout(STATS, UNEVEN, "same.parquet")
    with pytest.raises(ValueError, match="features.json"):
        p._check_layout(STATS, CENTER_MAJOR, "center_major.parquet")
    # a model saved without features.json is not checked
    CNN(model=str(tmp_path / "old" / "model.keras"))._check_layout(STATS, CENTER_MAJOR, "x")


def test_dann_target_layout_must_match_source(tmp_path):
    src = toy(tmp_path / "src.parquet", BASE)
    ok = CNN(source_data=src, target_data=toy(tmp_path / "tgt.parquet", BASE), seed=1)
    ok.load_da_data(_stats=STATS)
    bad = CNN(source_data=src, target_data=toy(tmp_path / "tgt_rev.parquet", BASE[::-1]), seed=1)
    with pytest.raises(ValueError, match="tgt_rev.parquet: its window/center columns differ"):
        bad.load_da_data(_stats=STATS)
