import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from grading.grader import grade_answer


SENTINEL = "<|resp|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the generator_v2 with vLLM."
    )
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--hf_repo_id", default=None)
    parser.add_argument("--hf_adapter_subpath", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


def load_test_samples(
    data: str,
    split: str = "test",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if data == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        for row_id, sample in enumerate(dataset):
            sample = dict(sample)
            sample["id"] = row_id
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    elif data == "math500":
        dataset = load_dataset("sxiong/MATH-500", split=split)
        for sample in dataset:
            sample = dict(sample)
            unique_id = str(sample.get("unique_id", "unknown"))
            sample["id"] = unique_id.removesuffix(".json").replace("/", "_")
            sample["question"] = sample.get("problem", "")
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    return samples


def build_prompt(question: str, cleaned_trajectory: str, tokenizer=None) -> str:
    prompt = (
        f"{question.strip()}\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        f"{SENTINEL}"
    )
    if cleaned_trajectory:
        prompt = f"{prompt}\n{cleaned_trajectory}"
    return prompt

def build_inference_prompt(question: str) -> str:
    return build_prompt(question.strip(), "")



def extract_gold_answer(answer_text: str) -> str:
    if "####" in answer_text:
        answer_text = answer_text.split("####", 1)[1]
    return answer_text.strip()


def normalize_answer(text: Any) -> str:
    normalized = str(text).strip()
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = normalized.replace("\\(", "").replace("\\)", "")
    normalized = normalized.strip().rstrip(".")
    return normalized


def parse_boxed_result(s):
    '''
    Parse the boxed result.
    '''
    s = str(s)
    # Find the start of the boxed content
    start = s.find('\\boxed{')
    if start == -1:
        return s
    
    # Skip past '\\boxed{' to start content capture
    start += len('\\boxed{')
    brace_count = 1  # We start after finding the first '{'
    content = []
    
    # Iterate over the string starting after '\boxed{'
    for i in range(start, len(s)):
        if s[i] == '{':
            brace_count += 1
        elif s[i] == '}':
            brace_count -= 1
        
        # If brace_count returns to 0, we've found the matching '}'
        if brace_count == 0:
            return ''.join(content)
        content.append(s[i])
    
    return s


def extract_prediction_answer(response_text: str) -> str:
    text = response_text.strip()

    if '\\boxed' in text:
        return parse_boxed_result(text)

    final_answer_patterns = [
        r'"Final answer"\s*:\s*"?([^"\n]+)"?',
        r"Final answer\s*:\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ]
    for pattern in final_answer_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_answer(match.group(1))

    fallback_matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", text)
    if fallback_matches:
        return normalize_answer(fallback_matches[-1])

    return ""


def evaluate_predictions(
    samples: List[Dict[str, Any]], responses: List[str]
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    correct = 0

    for sample, response in zip(samples, responses):
        gold_answer = extract_gold_answer(sample["answer"])
        pred_answer = extract_prediction_answer(response)
        is_correct = grade_answer(pred_answer, gold_answer)
        correct += int(is_correct)
        results.append(
            {
                "id": sample.get("id"),
                "question": sample.get("question"),
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "correct": is_correct,
                "response": response,
            }
        )

    total = len(results)
    accuracy = correct / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


def save_results(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_adapter_path(script_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    if args.hf_repo_id:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "huggingface_hub is required for --hf_repo_id. Install it with `pip install huggingface_hub`."
            ) from error

        snapshot_path = Path(snapshot_download(repo_id=args.hf_repo_id, repo_type="model"))
        if args.hf_adapter_subpath:
            adapter_path = snapshot_path / args.hf_adapter_subpath
        else:
            adapter_path = snapshot_path
        if not adapter_path.exists():
            raise RuntimeError(f"Resolved Hugging Face adapter path does not exist: {adapter_path}")
        return adapter_path

    if not args.adapter_path:
        return None

    adapter_path = Path(args.adapter_path)
    if adapter_path.is_absolute():
        return adapter_path
    return script_dir / adapter_path


def main() -> None:
    args = parse_args()

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as error:
        raise RuntimeError(
            "vLLM is not installed. Install it before running this evaluator."
        ) from error

    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / args.output_path
    samples = load_test_samples(args.data, split=args.split, limit=args.limit)
    prompts = [build_inference_prompt(sample["question"]) for sample in samples]

    adapter_path = resolve_adapter_path(script_dir, args)

    llm = LLM(
        model=args.base_model,
        enable_lora=bool(adapter_path),
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    if adapter_path is not None:
        lora_request = LoRARequest("swap_lora", 1, str(adapter_path))
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    responses = [output.outputs[0].text for output in outputs]

    metrics = evaluate_predictions(samples, responses)
    payload = {
        "base_model": args.base_model,
        "adapter_path": str(adapter_path),
        "hf_repo_id": args.hf_repo_id,
        "hf_adapter_subpath": args.hf_adapter_subpath,
        "data": args.data,
        "split": args.split,
        "total": metrics["total"],
        "correct": metrics["correct"],
        "accuracy": metrics["accuracy"],
        "results": metrics["results"],
    }
    save_results(output_path, payload)

    print(f"Evaluated {metrics['total']} local test samples.")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()