GENERATOR_ADAPTER=${1:-}
if [[ -n "${GENERATOR_ADAPTER}" ]]; then
  GENERATOR_ADAPTER=$(realpath "${GENERATOR_ADAPTER}")
  ADAPTER_ARGS=(--adapter_path "${GENERATOR_ADAPTER}")
else
  ADAPTER_ARGS=(--hf_repo_id sxiong/SWAP_v2_MATH500_Gen_Llama3-8B-LoRA)
fi

python src/eval_generator_v2_vllm.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  "${ADAPTER_ARGS[@]}" \
  --data math500 \
  --split test \
  --output_path ../output/eval_gen_v2_math500_test.json \
  --max_model_len 2560 \
  --max_new_tokens 2048 \
  --temperature 0 \
  --top_p 1.0
