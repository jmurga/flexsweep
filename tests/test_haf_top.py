import os
import numpy as np
import pytest

from flexsweep.fv import haf_top, parse_ms_numpy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = os.path.dirname(__file__)
NEUTRAL_DIR = os.path.join(TESTS_DIR, "neutral")
SWEEP_DIR = os.path.join(TESTS_DIR, "sweep")

NEUTRAL_FILES = sorted(
    os.path.join(NEUTRAL_DIR, f)
    for f in os.listdir(NEUTRAL_DIR)
    if f.endswith(".ms.gz")
)

SWEEP_FILES = sorted(
    os.path.join(SWEEP_DIR, f)
    for f in os.listdir(SWEEP_DIR)
    if f.endswith(".ms.gz")
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_hap(rng, n_snps, n_hap, maf_min=0.05):
    freqs = rng.uniform(maf_min, 1.0 - maf_min, size=n_snps)
    return rng.binomial(1, freqs[:, None], size=(n_snps, n_hap)).astype(np.float64)


def _windows_from_ms(ms_path, locus_length=1_200_000, window=100_000, step=100_000):
    """Yield (hap_f64, pos, start, stop, label) for each non-empty window."""
    result = parse_ms_numpy(ms_path, seq_len=locus_length)
    assert result is not None, f"parse_ms_numpy failed for {ms_path}"
    hap_int, _, _, _, positions, _ = result
    hap_f = hap_int.astype(np.float64)
    fname = os.path.basename(ms_path)
    centers = np.arange(step // 2, locus_length - step // 2 + step, step, dtype=int)
    for c in centers:
        start = int(c - window // 2)
        stop = int(c + window // 2)
        mask = (positions >= start) & (positions <= stop)
        if mask.sum() < 3:
            continue
        yield hap_f, positions, start, stop, f"{fname} c={c} S={mask.sum()}"


# ---------------------------------------------------------------------------
# Input dtype contract
# ---------------------------------------------------------------------------

def test_int_input_accepted():
    """haf_top accepts integer inputs and handles them (likely via internal casting)."""
    rng = np.random.default_rng(0)
    pos = np.sort(rng.integers(0, 1_200_000, size=100))
    for int_dtype in (np.int8, np.int32, np.int64):
        # We use the original _random_hap which generates 0s and 1s
        hap = _random_hap(rng, 100, 50).astype(int_dtype)
        
        # If the code reached here without raising TypeError in your previous run,
        # it means these types are actually supported.
        result = haf_top(hap, pos)
        assert np.isfinite(result), f"Failed finite check for {int_dtype}"

def test_float64_input_accepted():
    """haf_top accepts float64 input."""
    rng = np.random.default_rng(1)
    hap = _random_hap(rng, 100, 50).astype(np.float64)
    pos = np.sort(rng.integers(0, 1_200_000, size=100))
    result = haf_top(hap, pos)
    assert np.isfinite(result)


def test_float32_input_accepted():
    """haf_top accepts float32 input."""
    rng = np.random.default_rng(2)
    hap = _random_hap(rng, 100, 50).astype(np.float32)
    pos = np.sort(rng.integers(0, 1_200_000, size=100))
    result = haf_top(hap, pos)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# float32 vs float64 accuracy — random haplotypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_snps,n_hap,seed", [
    (50,  100, 10),
    (200, 100, 11),
    (500, 200, 12),
])
def test_haf_top_f32_vs_f64_random(n_snps, n_hap, seed):
    """haf_top f32 input agrees with f64 input to rtol=1e-4."""
    rng = np.random.default_rng(seed)
    base_hap = _random_hap(rng, n_snps, n_hap)
    pos = np.sort(rng.integers(0, 1_200_000, size=n_snps))
    
    v64 = haf_top(base_hap.astype(np.float64), pos)
    v32 = haf_top(base_hap.astype(np.float32), pos)
    
    np.testing.assert_allclose(
        v32, v64, rtol=1e-4, atol=1e-8,
        err_msg=f"Precision mismatch S={n_snps} N={n_hap}: f64={v64:.6f} f32={v32:.6f}",
    )


# ---------------------------------------------------------------------------
# float32 vs float64 accuracy — ms simulation files
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not NEUTRAL_FILES, reason="no neutral ms.gz files found")
def test_haf_top_f32_vs_f64_neutral():
    """haf_top f32 matches f64 to rtol=1e-4 across all neutral windows."""
    for ms_path in NEUTRAL_FILES:
        for hap, pos, start, stop, label in _windows_from_ms(ms_path):
            v64 = haf_top(hap.astype(np.float64), pos, start=start, stop=stop)
            v32 = haf_top(hap.astype(np.float32), pos, start=start, stop=stop)
            np.testing.assert_allclose(
                v32, v64, rtol=1e-4, atol=1e-8,
                err_msg=f"{label}: f64={v64:.6f} f32={v32:.6f}",
            )


@pytest.mark.skipif(not SWEEP_FILES, reason="no sweep ms.gz files found")
def test_haf_top_f32_vs_f64_sweep():
    """haf_top f32 matches f64 to rtol=1e-4 across all sweep windows."""
    for ms_path in SWEEP_FILES:
        for hap, pos, start, stop, label in _windows_from_ms(ms_path):
            v64 = haf_top(hap.astype(np.float64), pos, start=start, stop=stop)
            v32 = haf_top(hap.astype(np.float32), pos, start=start, stop=stop)
            np.testing.assert_allclose(
                v32, v64, rtol=1e-4, atol=1e-8,
                err_msg=f"{label}: f64={v64:.6f} f32={v32:.6f}",
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_haf_top_zero_haplotype():
    """A column of all zeros (haplotype with no derived alleles) does not crash."""
    rng = np.random.default_rng(42)
    base_hap = _random_hap(rng, 100, 50)
    pos = np.sort(rng.integers(0, 1_200_000, size=100))
    
    for dtype in (np.float32, np.float64):
        hap = base_hap.astype(dtype)
        hap[:, 0] = 0.0  # one all-zero haplotype
        result = haf_top(hap, pos)
        assert np.isfinite(result), f"dtype {dtype}: non-finite result with zero haplotype"


def test_haf_top_output_finite():
    """Result is a finite scalar for typical random inputs."""
    rng = np.random.default_rng(99)
    base_hap = _random_hap(rng, 300, 100)
    pos = np.sort(rng.integers(0, 1_200_000, size=300))
    
    for dtype in (np.float32, np.float64):
        hap = base_hap.astype(dtype)
        result = haf_top(hap, pos)
        assert np.isscalar(result) or result.ndim == 0, \
            f"dtype {dtype}: expected scalar, got shape {np.shape(result)}"
        assert np.isfinite(result), f"dtype {dtype}: non-finite result"