#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: RTX3090_NODELIST=gpu02[,gpuXX] bash submit_rtx3090.sh job.slurm [job arguments...]" >&2
  echo "   or: RTX3090_CONSTRAINT=rtx3090 bash submit_rtx3090.sh job.slurm [job arguments...]" >&2
  exit 2
fi

SBATCH_FILTER=()
if [[ -n "${RTX3090_CONSTRAINT:-}" ]]; then
  SBATCH_FILTER=(--constraint="$RTX3090_CONSTRAINT")
elif [[ -n "${RTX3090_NODELIST:-}" ]]; then
  SBATCH_FILTER=(--nodelist="$RTX3090_NODELIST")
else
  echo "Refusing unrestricted submission." >&2
  echo "Set RTX3090_CONSTRAINT if Slurm exposes a 3090 feature, or RTX3090_NODELIST to confirmed RTX 3090 nodes." >&2
  exit 2
fi

echo "Submitting with RTX 3090 filter: ${SBATCH_FILTER[*]}"
sbatch "${SBATCH_FILTER[@]}" "$@"
