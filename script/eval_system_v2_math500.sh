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
  --generator_hf_repo_id sxiong/SWAP_LLM_v2 \
  --generator_hf_adapter_subpath MATH_Gen_llama3_8B/final \
  --generation_temperature 0.6 \
  --generation_top_p 0.95 \
  --generation_max_tokens 2048 \
  --generation_max_model_len 2560 \
  --generation_gpu_memory_utilization 0.3 \
  --discriminator_base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --discriminator_hf_repo_id sxiong/SWAP_LLM_v2 \
  --discriminator_hf_adapter_subpath MATH_Disc_llama3_8B/final \
  --discrimination_temperature 0.0 \
  --discrimination_max_tokens 1536 \
  --discrimination_max_model_len 8192 \
  --discriminator_gpu_memory_utilization 0.6 \
  --num_candidates 32 \
  --cmp_per_opt 8 \
  --group_size 3