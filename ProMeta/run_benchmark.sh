#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${DATA_DIR:?Set DATA_DIR to the directory containing the six term2pre_*.pkl files.}"
: "${PROTEOMICS_CSV:?Set PROTEOMICS_CSV to the preprocessed proteomics CSV path.}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CPDB_FILE="${CPDB_FILE:-../resource/CPDB_pathways_genes.tab}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments_output}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTER_LR="${OUTER_LR:-1e-4}"
INNER_LR="${INNER_LR:-0.005}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
MAX_SUPPORT_SIZE="${MAX_SUPPORT_SIZE:-32}"
ENCODER_TYPE="${ENCODER_TYPE:-transformer}"
NUM_TASK_GROUPS="${NUM_TASK_GROUPS:-5}"
TSA_SELECTOR_STEPS="${TSA_SELECTOR_STEPS:-10}"
TSA_PARAM_KEYS="${TSA_PARAM_KEYS:-classifier,tokenizer.gate_logits}"
TSA_SELECTOR_SOURCE="${TSA_SELECTOR_SOURCE:-frozen_warmup}"
TSA_ASSIGNMENT_SOURCE="${TSA_ASSIGNMENT_SOURCE:-current_group}"
TSA_DISTANCE_MODE="${TSA_DISTANCE_MODE:-block_mean_l2}"
TSA_GATE_DISTANCE_WEIGHT="${TSA_GATE_DISTANCE_WEIGHT:-1.0}"
TSA_SELECTOR_L1_LAMBDA="${TSA_SELECTOR_L1_LAMBDA:-0.001}"
TSA_ROUTING_SCHEDULE="${TSA_ROUTING_SCHEDULE:-epoch_snapshot}"
TSA_SWITCH_THRESHOLD="${TSA_SWITCH_THRESHOLD:-0.05}"
TSA_MIN_GROUP_FRACTION="${TSA_MIN_GROUP_FRACTION:-0.05}"
TSA_MAX_GROUP_FRACTION="${TSA_MAX_GROUP_FRACTION:-0.50}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ProMeta_vs_TSA}"

SHOT_LIST="${SHOT_LIST:-4 8 16 32}"
SEED_LIST="${SEED_LIST:-42 43 44 45 46}"

mkdir -p "$OUTPUT_DIR"

for shot in $SHOT_LIST; do
  for seed in $SEED_LIST; do
    echo "=== ProMeta | shot=${shot} | seed=${seed} ==="
    "$PYTHON_BIN" main.py \
      --data_dir "$DATA_DIR" \
      --proteomics_csv "$PROTEOMICS_CSV" \
      --cpdb_path "$CPDB_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --gpu_id "$GPU_ID" \
      --random_seed "$seed" \
      --support_size "$shot" \
      --max_support_size "$MAX_SUPPORT_SIZE" \
      --batch_size "$BATCH_SIZE" \
      --outer_lr "$OUTER_LR" \
      --inner_lr "$INNER_LR" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --encoder_type "$ENCODER_TYPE" \
      --experiment_name "$EXPERIMENT_NAME"

    warmup_checkpoint="$OUTPUT_DIR/checkpoints/support_${shot}/ProMeta_best_seed${seed}.pth"
    if [[ ! -f "$warmup_checkpoint" ]]; then
      echo "Missing warmup checkpoint: $warmup_checkpoint" >&2
      exit 1
    fi

    echo "=== TSA-ProMeta | shot=${shot} | seed=${seed} ==="
    "$PYTHON_BIN" main.py \
      --data_dir "$DATA_DIR" \
      --proteomics_csv "$PROTEOMICS_CSV" \
      --cpdb_path "$CPDB_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --gpu_id "$GPU_ID" \
      --random_seed "$seed" \
      --support_size "$shot" \
      --max_support_size "$MAX_SUPPORT_SIZE" \
      --batch_size "$BATCH_SIZE" \
      --outer_lr "$OUTER_LR" \
      --inner_lr "$INNER_LR" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --encoder_type "$ENCODER_TYPE" \
      --experiment_name "$EXPERIMENT_NAME" \
      --tsa_enable \
      --num_task_groups "$NUM_TASK_GROUPS" \
      --tsa_selector_steps "$TSA_SELECTOR_STEPS" \
      --tsa_param_keys "$TSA_PARAM_KEYS" \
      --tsa_selector_source "$TSA_SELECTOR_SOURCE" \
      --tsa_assignment_source "$TSA_ASSIGNMENT_SOURCE" \
      --tsa_distance_mode "$TSA_DISTANCE_MODE" \
      --tsa_gate_distance_weight "$TSA_GATE_DISTANCE_WEIGHT" \
      --tsa_selector_l1_lambda "$TSA_SELECTOR_L1_LAMBDA" \
      --tsa_routing_schedule "$TSA_ROUTING_SCHEDULE" \
      --tsa_switch_threshold "$TSA_SWITCH_THRESHOLD" \
      --tsa_min_group_fraction "$TSA_MIN_GROUP_FRACTION" \
      --tsa_max_group_fraction "$TSA_MAX_GROUP_FRACTION" \
      --tsa_warmup_checkpoint "$warmup_checkpoint"
  done
done

echo "Benchmark runs finished. Summarize with:"
echo "$PYTHON_BIN summarize_benchmark.py --output_dir \"$OUTPUT_DIR\""
