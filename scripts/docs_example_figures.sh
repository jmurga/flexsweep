#!/usr/bin/env bash
# Run `flexsweep example` (docs/quickstart.rst) on the workstation for the 5 Mb and the
# whole-chr22 data, record per-step timings, and write the scan figures embedded in the
# documentation (docs/images/quickstart_*.svg).
#
# Usage (from the repository root):
#     bash scripts/docs_example_figures.sh [OUT_DIR] [NTHREADS]
#
# Fixed before measuring: NTHREADS (default 24), --seed 1, 1000 + 1000 simulations,
# 50 YRI individuals. One cold run per setting: the times are indicative, not a
# benchmark. Per-step times are in OUT_DIR/{ex_5mb,ex_chr22}/timings.tsv; the data step
# of the first run includes the Zenodo download (170 MB), the second reuses it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "${1:-${REPO}/fs_docs_example}")"
NT="${2:-24}"
IMG="${REPO}/docs/images"
mkdir -p "${OUT}" "${IMG}"

cd "${REPO}"
rm -rf "${OUT}/venv"
uv venv --python 3.12 "${OUT}/venv"
uv build --wheel
WHEEL="$(ls -t dist/flexsweep-*.whl | head -n 1)"
uv pip install --python "${OUT}/venv/bin/python" "${WHEEL}"
FS="${OUT}/venv/bin/flexsweep"

cd "${OUT}"
echo "# $(date -Is) $(hostname) nthreads=${NT} seed=1 commit=$(git -C "${REPO}" rev-parse --short HEAD)" \
    > provenance.txt
rm -rf ex_5mb ex_chr22

"${FS}" example --output_folder ex_5mb --nthreads "${NT}" --seed 1 2>&1 | tee ex_5mb.log

# Whole chr22: reuse the downloads of the 5 Mb run
mkdir -p ex_chr22/data/raw
for f in ex_5mb/data/raw/{yri_chr22_hg38_high_coverage.vcf.gz,447.primates_chr22.maf.gz,GRCh38_alldifficultregions.bed.gz}; do
    ln -f "${f}" ex_chr22/data/raw/ 2> /dev/null || cp "${f}" ex_chr22/data/raw/
done
"${FS}" example --output_folder ex_chr22 --nthreads "${NT}" --seed 1 --full 2>&1 | tee ex_chr22.log

# Figures for docs/quickstart.rst (5 Mb run)
S=ex_5mb/scan/yri
"${FS}" scan-plot --stat "${S}.ihs.txt" --stat "${S}.nsl.txt" --stat "${S}.h12.txt" \
    --stat "${S}.raisd.txt" --multi --out "${IMG}/quickstart_scan_multi.svg"
"${FS}" scan-plot --stat "${S}.ihs.txt" --pvalue --out "${IMG}/quickstart_scan_ihs_pvalue.svg"

echo
for d in ex_5mb ex_chr22; do
    echo "== ${d}"
    column -t -s $'\t' "${d}/timings.tsv"
done
echo "Figures: ${IMG}/quickstart_scan_*.svg  Outputs: ${OUT}"
