GENERATOR_ADAPTER=${1:-}
if [[ -n "${GENERATOR_ADAPTER}" ]]; then
  GENERATOR_ADAPTER=$(realpath "${GENERATOR_ADAPTER}")
  ADAPTER_ARGS=(--adapter_path "${GENERATOR_ADAPTER}")
else
  ADAPTER_ARGS=(--hf_repo_id sxiong/SWAP_v2_GSM8K_Gen_Llama3-8B-LoRA)
fi

python src/eval_generator_v2_vllm.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  "${ADAPTER_ARGS[@]}" \
  --data gsm8k \
  --split test \
  --output_path ../output/eval_gen_v2_gsm8k_test.json \
  --max_model_len 1280 \
  --max_new_tokens 1024 \
  --temperature 0 \
  --top_p 1.0
