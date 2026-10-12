"""Non-finite feature values must be rejected before CNN/DANN training or prediction.

Regression: +/-inf z-scores (zero-std normalization groups) made DANN loss NaN and gave
NaN CNN predictions for 32/158 example windows without any error.
Oracle: hand-built frames with known NaN/inf positions.
"""

import numpy as np
import polars as pl
import pytest

from flexsweep.cnn import CNN


def _frame(value):
    return pl.DataFrame(
        {
            "iter": ["chr22:1-1200000", "chr22:10001-1210000"],
            "s": [0.0, 0.0],
            "t": [np.nan, 0.0],  # parameter columns are not features: ignored
            "model": ["neutral", "neutral"],
            "isafe_100000_850000": [0.5, value],
            "h12_100000_850000": [0.1, 0.2],
        }
    )


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_non_finite_feature_rejected(value):
    with pytest.raises(ValueError, match="isafe_100000_850000"):
        CNN._check_finite(_frame(value), "fvs_yri.parquet")


def test_finite_features_pass():
    CNN._check_finite(_frame(0.3), "fvs_yri.parquet")
