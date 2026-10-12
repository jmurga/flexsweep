"""Haplotype-image architecture (flexsweep.cnn.hap_image_cnn).

Oracle: the parameter count worked out by hand from the layer sizes, and the layer
sequence itself. The head must flatten (not globally pool) the last feature map, since
position along the SNP axis is the signal a haplotype image carries.
"""

import pytest

tf = pytest.importorskip("tensorflow")

from flexsweep.cnn import hap_image_cnn  # noqa: E402


def build(shape=(32, 41, 1), **kw):
    inp = tf.keras.Input(shape)
    return tf.keras.Model(inp, hap_image_cnn(inp, **kw))


def test_parameter_count_matches_hand_computation():
    # 32x41 -> conv 3x3 valid 30x39x64 -> pool 15x19 -> conv 13x17x64 -> pool 6x8
    # flatten 6*8*64 = 3072
    want = ((3 * 3 * 1 * 64 + 64) + (3 * 3 * 64 * 64 + 64)        # convolutions
            + (3072 * 128 + 128) + (128 * 128 + 128)              # dense head
            + (128 * 1 + 1))                                      # sigmoid output
    m = build()
    assert m.count_params() == want == 447_553
    assert m.output_shape == (None, 1)
