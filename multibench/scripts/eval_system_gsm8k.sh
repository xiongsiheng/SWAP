#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
multibench_root=$(cd "$script_dir/.." && pwd)
repo_root=$(cd "$multibench_root/.." && pwd)
python_bin="${PYTHON_BIN:-python}"
base_model="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
benchmark=gsm8k
split="${SPLIT:-test}"
eval_seed="${EVAL_SEED:-$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')}"
eval_seed=$((eval_seed % 2147483647))
generator_adapter="${GENERATOR_ADAPTER:-$repo_root/checkpoints/SWAP_GSM8K_Gen_Llama3-8B-LoRA}"
discriminator_adapter="${DISCRIMINATOR_ADAPTER:-$repo_root/checkpoints/SWAP_GSM8K_Disc_Llama3-8B-LoRA}"
output_dir="${OUTPUT_DIR:-$multibench_root/output/$benchmark}"
generator_output="$output_dir/generator_output.json"
system_output="$output_dir/system_output.json"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$multibench_root/src:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output_dir"

if [[ "${SKIP_GRADING:-0}" != "1" ]]; then
    "$python_bin" -c "from utils import ensure_grading_available; ensure_grading_available('$benchmark')"
fi

limit_args=()
if [[ -n "${LIMIT:-}" ]]; then
    limit_args=(--limit "$LIMIT")
fi

"$python_bin" "$multibench_root/src/eval_generator_vllm.py" \
    --benchmark "$benchmark" \
    --split "$split" \
    --base-model "$base_model" \
    --adapter-path "$generator_adapter" \
    --output "$generator_output" \
    --samples "${GENERATOR_SAMPLES:-31}" \
    --temperature "${GENERATOR_TEMPERATURE:-1.0}" \
    --top-p "${GENERATOR_TOP_P:-0.95}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --max-tokens "${GENERATOR_MAX_TOKENS:-6144}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
    "${limit_args[@]}"

grading_args=()
if [[ "${SKIP_GRADING:-0}" == "1" ]]; then
    grading_args=(--skip-grading)
fi

"$python_bin" "$multibench_root/src/eval_system_vllm.py" \
    --generator-output "$generator_output" \
    --base-model "$base_model" \
    --adapter-path "$discriminator_adapter" \
    --output "$system_output" \
    --max-options "${MAX_OPTIONS:-8}" \
    --max-trajectory-chars "${MAX_TRAJECTORY_CHARS:-1400}" \
    --include-graph \
    --max-tokens "${DISCRIMINATOR_MAX_TOKENS:-384}" \
    --temperature 0 \
    --top-p 1 \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
    --seed "$eval_seed" \
    "${grading_args[@]}"

printf 'Generator output: %s\nSystem output: %s\n' "$generator_output" "$system_output"
