GENERATOR_ADAPTER=${1:-}
DISCRIMINATOR_ADAPTER=${2:-}
if [[ -n "${GENERATOR_ADAPTER}" ]]; then
  GENERATOR_ADAPTER=$(realpath "${GENERATOR_ADAPTER}")
  GENERATOR_ADAPTER_ARGS=(--generator_adapter_path "${GENERATOR_ADAPTER}")
else
  GENERATOR_ADAPTER_ARGS=(--generator_hf_repo_id sxiong/SWAP_v2_MATH500_Gen_Llama3-8B-LoRA)
fi
if [[ -n "${DISCRIMINATOR_ADAPTER}" ]]; then
  DISCRIMINATOR_ADAPTER=$(realpath "${DISCRIMINATOR_ADAPTER}")
  DISCRIMINATOR_ADAPTER_ARGS=(--discriminator_adapter_path "${DISCRIMINATOR_ADAPTER}")
else
  DISCRIMINATOR_ADAPTER_ARGS=(--discriminator_hf_repo_id sxiong/SWAP_v2_MATH500_Disc_Llama3-8B-LoRA)
fi

NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
OUTPUT_PATH=${OUTPUT_PATH:-../output/eval_system_v2_math500_test_shard_${SHARD_INDEX}_of_${NUM_SHARDS}.json}

python src/eval_system_v2_vllm.py \
  --data math500 \
  --split test \
  --output_path ${OUTPUT_PATH} \
  --num_shards ${NUM_SHARDS} \
  --shard_index ${SHARD_INDEX} \
  --save_every 1 \
  --generator_base_model meta-llama/Meta-Llama-3-8B-Instruct \
  "${GENERATOR_ADAPTER_ARGS[@]}" \
  --generation_temperature 0.6 \
  --generation_top_p 0.95 \
  --generation_max_tokens 2048 \
  --generation_max_model_len 2560 \
  --generation_gpu_memory_utilization 0.3 \
  --discriminator_base_model meta-llama/Meta-Llama-3-8B-Instruct \
  "${DISCRIMINATOR_ADAPTER_ARGS[@]}" \
  --discrimination_temperature 0.0 \
  --discrimination_max_tokens 1536 \
  --discrimination_max_model_len 8192 \
  --discriminator_gpu_memory_utilization 0.6 \
  --num_candidates 32 \
  --cmp_per_opt 8 \
  --group_size 3
