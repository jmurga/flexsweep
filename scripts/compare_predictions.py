"""Compare two Flex-sweep prediction tables row by row.

Usage: python scripts/compare_predictions.py a.txt b.txt [tolerance]

Exits 0 if both tables have the same rows/columns, no NaN probabilities,
identical predicted classes, and max |prob_sweep_a - prob_sweep_b| <= tolerance
(default 1e-5; use ~1e-3 across separate GPU processes, where cuDNN kernel choice
changes float32 results by ~1e-4).
"""

import sys

import polars as pl


def main(a_path: str, b_path: str, tol: str = "1e-5") -> int:
    a, b = pl.read_csv(a_path), pl.read_csv(b_path)
    if a.columns != b.columns or a.height != b.height:
        print(f"shape/columns differ: {a.shape} {b.shape}\n{a.columns}\n{b.columns}")
        return 1
    n_nan = int(a["prob_sweep"].is_nan().sum() + b["prob_sweep"].is_nan().sum())
    diff = (a["prob_sweep"] - b["prob_sweep"]).abs()
    max_diff = float(diff.max())
    same_class = float((a["predicted_model"] == b["predicted_model"]).mean())
    print(f"rows: {a.height}  NaN prob_sweep: {n_nan}")
    print(f"max |diff| prob_sweep: {max_diff:.3e}  (tolerance {float(tol):.0e})")
    print(f"identical predicted_model: {same_class:.4f}")
    return 0 if n_nan == 0 and same_class == 1.0 and max_diff <= float(tol) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
