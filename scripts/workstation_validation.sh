#!/usr/bin/env bash
# Build flexsweep from a clean checkout and run every documented command end to end.
#
# Usage (from the repository root on the workstation):
#     bash scripts/workstation_validation.sh [OUT_DIR] [NTHREADS]
#
# Writes logs and a timing table to OUT_DIR (default: ./fs_validation). Every step
# appends "step<TAB>exit_code<TAB>seconds" to OUT_DIR/timings.tsv; failures do not
# stop the script, so one run reports everything. Requires: uv, a Rust toolchain
# (cargo) for the polarize extension, network access (flexsweep example), and
# optionally REF_VCF pointing at your production polarized YRI chr22 VCF.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "${1:-${REPO}/fs_validation}")"
NT="${2:-24}"
REF_VCF="${REF_VCF:-${HOME}/test/yri/yri_chr22_polarized_giab.vcf.gz}"
mkdir -p "${OUT}/logs"
TIMES="${OUT}/timings.tsv"
printf "step\texit\tseconds\n" > "${TIMES}"

run() {
    # run <step-name> <command...>; log to OUT/logs/<step>.log, record exit code and time
    local name="$1"; shift
    local t0 t1 rc
    t0=$(date +%s)
    echo ">>> ${name}: $*"
    "$@" > "${OUT}/logs/${name}.log" 2>&1
    rc=$?
    t1=$(date +%s)
    printf "%s\t%s\t%s\n" "${name}" "${rc}" "$((t1 - t0))" >> "${TIMES}"
    [[ ${rc} -eq 0 ]] || echo "    FAILED (exit ${rc}), see ${OUT}/logs/${name}.log"
}

cd "${REPO}"

# 0. Fresh environment and build (wheel, as a user would install it)
rm -rf "${OUT}/venv" dist
run 00_venv   uv venv --python 3.12 "${OUT}/venv"
PY="${OUT}/venv/bin/python"
FS="${OUT}/venv/bin/flexsweep"
run 01_build  uv build --wheel
WHEEL="$(ls dist/flexsweep-*.whl | head -n 1)"
run 02_install uv pip install --python "${PY}" "${WHEEL}" pytest

# 1. Static check and tests (tests import the repo source; the CLI uses the wheel)
# --isolated: ignore ruff configs above the repo (pyproject.toml has no [tool.ruff]);
# test_code/ holds local scratch scripts that are never packaged or uploaded
run 10_lint_undefined_names uvx ruff check --isolated --select F821,F822,F823 \
    --output-format concise --extend-exclude test_code flexsweep/
run 11_pytest_fast env CUDA_VISIBLE_DEVICES=-1 "${PY}" -m pytest tests -m "not slow" -q -p no:warnings
run 12_pytest_slow env CUDA_VISIBLE_DEVICES=-1 "${PY}" -m pytest tests -m slow -q -p no:warnings

# 2. Quickstart, exactly as in docs/quickstart.rst (one command, six steps)
cd "${OUT}"
rm -rf qs qs_dann qs_dann_pred
run 20_example       "${FS}" example --output_folder qs --nthreads "${NT}" --seed 1
run 26_dann          "${FS}" dann --source_data qs/sims/fvs.parquet \
                     --target_data qs/data/vcfs/fvs_yri.parquet --output_folder qs_dann --seed 1
run 27_dann_predict  "${FS}" dann --model qs_dann/model_da.keras \
                     --target_data qs/data/vcfs/fvs_yri.parquet --output_folder qs_dann_pred --seed 1

# 3. Reproducibility
rm -rf seed_a seed_b
run 30_seed_sim_a "${FS}" simulator --sample_size 100 --demes yri --output_folder seed_a \
                  --num_simulations 50 --nthreads "${NT}" --seed 7
run 31_seed_sim_b "${FS}" simulator --sample_size 100 --demes yri --output_folder seed_b \
                  --num_simulations 50 --nthreads "${NT}" --seed 7
run 32_seed_sim_identical diff -r seed_a seed_b
run 33_cnn_cpu_a env CUDA_VISIBLE_DEVICES=-1 "${FS}" cnn --train_data qs/sims/fvs.parquet \
                 --predict_data qs/data/vcfs/fvs_yri.parquet --output_folder cnn_cpu_a --seed 1
run 34_cnn_cpu_b env CUDA_VISIBLE_DEVICES=-1 "${FS}" cnn --train_data qs/sims/fvs.parquet \
                 --predict_data qs/data/vcfs/fvs_yri.parquet --output_folder cnn_cpu_b --seed 1
run 35_cnn_cpu_identical cmp cnn_cpu_a/yri_predictions.txt cnn_cpu_b/yri_predictions.txt
run 36_cnn_gpu_b "${FS}" cnn --train_data qs/sims/fvs.parquet \
                 --predict_data qs/data/vcfs/fvs_yri.parquet --output_folder cnn_gpu_b --seed 1
run 37_cnn_gpu_identical cmp qs/cnn/yri_predictions.txt cnn_gpu_b/yri_predictions.txt
# Same model and calibration, separate GPU processes: cuDNN kernel choice changes
# float32 outputs by ~1e-4 (observed 1.04e-4; CPU matches to ~1e-7). Require identical
# classes, no NaN, and max |diff| <= 1e-3; prints the max |diff| to the log.
run 38_dann_train_vs_predict "${PY}" "${REPO}/scripts/compare_predictions.py" \
    qs_dann/yri_da_predictions.txt qs_dann_pred/yri_da_predictions.txt 1e-3

# 4. Example data versus your production polarized chr22 (independent preparation)
if [[ -f "${REF_VCF}" ]]; then
    run 40_compare_example "${PY}" "${REPO}/scripts/compare_example.py" \
        qs/data/vcfs/yri_chr22_example.vcf.gz "${REF_VCF}"
fi

# 5. Every documented command parses (also covered by tests/test_docs_cli.py)
for c in simulator fvs-discoal fvs-vcf recombination-bins cnn dann rank enrichment \
         polarize split-maf scan scan-plot example; do
    run "50_help_${c}" "${FS}" "${c}" --help
done

echo
echo "Timings and exit codes: ${TIMES}"
column -t "${TIMES}"
