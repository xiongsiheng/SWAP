NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
OUTPUT_PATH=${OUTPUT_PATH:-../output/eval_system_gsm8k_test_shard_${SHARD_INDEX}_of_${NUM_SHARDS}.json}

python src/eval_system_vllm.py \
  --data gsm8k \
  --output_path ${OUTPUT_PATH} \
  --num_shards ${NUM_SHARDS} \
  --shard_index ${SHARD_INDEX} \
  --save_every 1 \
  --generator_base_model ../model_weights/GSM8K_Gen_llama3_8B \
  --generation_temperature 0.6 \
  --generation_top_p 0.95 \
  --generation_max_tokens 512 \
  --generation_max_model_len 1024 \
  --generation_gpu_memory_utilization 0.3 \
  --discriminator_base_model ../model_weights/GSM8K_Disc_llama3_8B \
  --discrimination_temperature 0.0 \
  --discrimination_max_tokens 1024 \
  --discrimination_max_model_len 3072 \
  --discriminator_gpu_memory_utilization 0.6 \
  --num_candidates 32 \
  --keep_top_k 1 \
  --cmp_per_opt 8 \
  --group_size 3 \
  --max_steps 20 \
  --search_per_N_steps 10 \
  --future_N_steps 2