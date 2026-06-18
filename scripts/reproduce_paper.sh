#!/usr/bin/env bash
#
# Reproduce the paper's quantitative results.
#
# For every config in retargeting_configs/, runs contact-based retargeting (our
# method) at each loss threshold, then builds the per-threshold comparison
# tables (paper Tables II-V) as Markdown and CSV under results/.
#
# Usage:
#   bash scripts/reproduce_paper.sh
#   THRESHOLDS="0.0035 0.01" bash scripts/reproduce_paper.sh   # custom sweep
#   BASELINES=1 bash scripts/reproduce_paper.sh                # also run naive baselines
#
# Prerequisites:
#   - conda env `trajwarp` (see README), activated.
#   - demonstration data downloaded (bash scripts/download_data.sh).
set -euo pipefail

cd "$(dirname "$0")/.."

# Loss thresholds to sweep (space-separated). Override via env var.
THRESHOLDS="${THRESHOLDS:-0.0035 0.01 0.02}"
# Set BASELINES=1 to additionally run the two naive baselines.
BASELINES="${BASELINES:-0}"

CONFIGS=(retargeting_configs/*.json)
mkdir -p results

for thr in ${THRESHOLDS}; do
  echo "=============================================================="
  echo "Loss threshold: ${thr}"
  echo "=============================================================="
  out_dir="final_trajectories_${thr}"

  for cfg in "${CONFIGS[@]}"; do
    name="$(basename "${cfg}" .json)"
    [ "${name}" = "schema" ] && continue
    echo "--- retarget (ours): ${name} @ ${thr} ---"
    python retarget.py "${cfg}" --no-view --loss-threshold "${thr}"

    if [ "${BASELINES}" = "1" ]; then
      echo "--- retarget (naive_spline): ${name} @ ${thr} ---"
      python retarget_naive_spline.py "${cfg}" --no-view --loss-threshold "${thr}"
    fi
  done

  echo "--- building tables for threshold ${thr} ---"
  python evaluate.py --all --dir "${out_dir}" \
    --markdown "results/table_${thr}.md" --csv "results/table_${thr}.csv"

  if [ "${BASELINES}" = "1" ]; then
    python evaluate.py --all --dir "${out_dir}" --suffix _naive \
      --markdown "results/table_${thr}_naive.md" --csv "results/table_${thr}_naive.csv"
  fi
done

echo "Done. Tables written to results/."
