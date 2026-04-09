python src/eval_generator_vllm.py \
  --base_model ../model_weights/GSM8K_Gen_llama3_8B \
  --data gsm8k \
  --output_path ../output/eval_gen_gsm8k_test.json \
  --max_model_len 1024 \
  --max_new_tokens 512 \
  --temperature 0 \
  --top_p 1.0