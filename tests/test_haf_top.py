"""haf_top accepts integer and float haplotype matrices and its float32 path matches float64.

No independent oracle: the float32 check compares against the float64 path on bundled
neutral and sweep replicates.
"""

import os

import numpy as np

from flexsweep.fv import haf_top, parse_ms_numpy

TESTS_DIR = os.path.dirname(__file__)
MS_FILES = sorted(
    os.path.join(TESTS_DIR, d, f)
    for d in ("neutral", "sweep")
    for f in os.listdir(os.path.join(TESTS_DIR, d))
    if f.endswith(".ms.gz")
)


def test_input_dtypes_and_zero_haplotype_give_finite_result():
    rng = np.random.default_rng(0)
    freqs = rng.uniform(0.05, 0.95, size=100)
    hap = rng.binomial(1, freqs[:, None], size=(100, 50))
    hap[:, 0] = 0  # one haplotype with no derived alleles
    pos = np.sort(rng.integers(0, 1_200_000, size=100))
    for dtype in (np.int8, np.int64, np.float32, np.float64):
        assert np.isfinite(haf_top(hap.astype(dtype), pos)), dtype


def test_float32_matches_float64_on_simulations():
    assert MS_FILES
    for ms_path in MS_FILES:
        hap, _, _, _, pos, _ = parse_ms_numpy(ms_path, seq_len=1_200_000)
        for c in range(50_000, 1_200_000, 100_000):
            start, stop = c - 50_000, c + 50_000
            if ((pos >= start) & (pos <= stop)).sum() < 3:
                continue
            v64 = haf_top(hap.astype(np.float64), pos, start=start, stop=stop)
            v32 = haf_top(hap.astype(np.float32), pos, start=start, stop=stop)
            np.testing.assert_allclose(v32, v64, rtol=1e-4, atol=1e-8, err_msg=f"{ms_path} c={c}")
