GENERATOR_MODEL=${1:-sxiong/SWAP_v1_GSM8K_Gen_Llama3-8B}

python src/eval_generator_vllm.py \
  --base_model "${GENERATOR_MODEL}" \
  --data gsm8k \
  --split test \
  --output_path ../output/eval_gen_gsm8k_test.json \
  --max_model_len 1024 \
  --max_new_tokens 512 \
  --temperature 0 \
  --top_p 1.0
