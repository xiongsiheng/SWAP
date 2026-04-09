import argparse
import json
import random
import re
from collections import defaultdict
from itertools import combinations
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from grading.grader import grade_answer
from tqdm import tqdm



LABEL_PATTERN = re.compile(
    r"(?m)^(Goal|Initial state|Plan|Action\s+\d+|State\s+\d+|Final answer):"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the full system with vLLM."
    )
    parser.add_argument("--data", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from_output", default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=10)

    parser.add_argument("--generator_base_model", default=None)
    parser.add_argument("--generator_adapter_path", default=None)
    parser.add_argument("--generation_max_model_len", type=int, default=3072)
    parser.add_argument("--generator_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--generation_gpu_memory_utilization", type=float, default=0.3)
    
    parser.add_argument("--generation_temperature", type=float, default=0.6)
    parser.add_argument("--generation_top_p", type=float, default=0.95)
    parser.add_argument("--generation_max_tokens", type=int, default=2048)

    parser.add_argument("--discriminator_base_model", default=None)
    parser.add_argument("--discriminator_adapter_path", default=None)
    parser.add_argument("--discrimination_max_model_len", type=int, default=8192)
    parser.add_argument("--discriminator_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--discriminator_gpu_memory_utilization", type=float, default=0.6)

    parser.add_argument("--discrimination_temperature", type=float, default=0.0)
    parser.add_argument("--discrimination_top_p", type=float, default=1.0)
    parser.add_argument("--discrimination_max_tokens", type=int, default=4096)

    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--keep_top_k", type=int, default=2)
    parser.add_argument("--cmp_per_opt", type=int, default=3)
    parser.add_argument("--group_size", type=int, choices=[2, 3], default=3)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--search_per_N_steps", type=int, default=1)
    parser.add_argument("--future_N_steps", type=int, default=0)
    return parser.parse_args()


def load_test_samples(data: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if data == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        for row_id, sample in enumerate(dataset):
            sample = dict(sample)
            sample["id"] = row_id
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    return samples


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


def extract_prediction_answer(response_text: str) -> str:
    final_answer_patterns = [
        r'"Final answer"\s*:\s*"?([^"\n]+)"?',
        r"Final answer\s*:\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ]
    for pattern in final_answer_patterns:
        match = re.search(pattern, response_text, flags=re.IGNORECASE)
        if match:
            return normalize_answer(match.group(1))

    fallback_matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", response_text)
    if fallback_matches:
        return normalize_answer(fallback_matches[-1])

    return normalize_answer(response_text)


def evaluate_prediction(sample: Dict[str, Any], response: str) -> Dict[str, Any]:
    gold_answer = extract_gold_answer(sample["answer"])
    gold_answer_normalized = normalize_answer(gold_answer)
    pred_answer = extract_prediction_answer(response)
    is_correct = grade_answer(pred_answer, gold_answer)
    return {
        "gold_answer": gold_answer,
        "gold_answer_normalized": gold_answer_normalized,
        "pred_answer": pred_answer,
        "correct": is_correct,
    }


def save_results(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_optional_path(script_dir: Path, path_arg: Optional[str]) -> Optional[Path]:
    if not path_arg:
        return None
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return script_dir / path


def validate_shard_args(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise RuntimeError("--num_shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("--shard_index must satisfy 0 <= shard_index < num_shards.")


def filter_samples_for_shard(
    samples: List[Dict[str, Any]],
    num_shards: int,
    shard_index: int,
) -> List[Dict[str, Any]]:
    if num_shards == 1:
        return samples
    return [sample for index, sample in enumerate(samples) if index % num_shards == shard_index]


def build_ordered_results(
    samples: List[Dict[str, Any]],
    result_by_id: Dict[Any, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ordered_results: List[Dict[str, Any]] = []
    for sample in samples:
        sample_id = sample.get("id")
        if sample_id in result_by_id:
            ordered_results.append(result_by_id[sample_id])
    return ordered_results


def build_payload(
    args: argparse.Namespace,
    data: str,
    generator_adapter_path: Optional[Path],
    discriminator_adapter_path: Optional[Path],
    results: List[Dict[str, Any]],
    latest_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    num_correct = sum(int(result.get("correct", False)) for result in latest_results)
    return {
        "generator_base_model": args.generator_base_model,
        "generator_adapter_path": str(generator_adapter_path) if generator_adapter_path is not None else None,
        "generation_max_model_len": args.generation_max_model_len,
        "generator_tensor_parallel_size": args.generator_tensor_parallel_size,
        "generation_gpu_memory_utilization": args.generation_gpu_memory_utilization,
        "discriminator_base_model": args.discriminator_base_model,
        "discriminator_adapter_path": str(discriminator_adapter_path) if discriminator_adapter_path is not None else None,
        "discrimination_max_model_len": args.discrimination_max_model_len,
        "discriminator_tensor_parallel_size": args.discriminator_tensor_parallel_size,
        "discriminator_gpu_memory_utilization": args.discriminator_gpu_memory_utilization,
        "data": data,
        "resume": args.resume,
        "resume_from_output": args.resume_from_output,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "generation_temperature": args.generation_temperature,
        "generation_top_p": args.generation_top_p,
        "generation_max_tokens": args.generation_max_tokens,
        "discrimination_temperature": args.discrimination_temperature,
        "discrimination_top_p": args.discrimination_top_p,
        "discrimination_max_tokens": args.discrimination_max_tokens,
        "num_candidates": args.num_candidates,
        "keep_top_k": args.keep_top_k,
        "cmp_per_opt": args.cmp_per_opt,
        "group_size": args.group_size,
        "max_steps": args.max_steps,
        "search_per_N_steps": args.search_per_N_steps,
        "future_N_steps": args.future_N_steps,
        "total": len(latest_results),
        "correct": num_correct,
        "accuracy": (num_correct / len(latest_results)) if latest_results else 0.0,
        "results": latest_results
    }


def count_completed_steps(trajectory: str) -> int:
    completed_steps = len(re.findall(r"(?m)^State\s+\d+:", trajectory))
    if re.search(r"(?m)^Goal:", trajectory) and re.search(r"(?m)^Initial state:", trajectory):
        completed_steps += 1
    if re.search(r"(?m)^Plan:", trajectory):
        completed_steps += 1
    return completed_steps


def count_action_steps(trajectory: str) -> int:
    return len(re.findall(r"(?m)^State\s+\d+:", trajectory))


def has_final_answer(trajectory: str) -> bool:
    return bool(re.search(r"(?m)^Final answer:", trajectory))


def extract_step_number(label: str, prefix: str) -> Optional[int]:
    match = re.fullmatch(rf"{prefix}\s+(\d+)", label)
    if not match:
        return None
    return int(match.group(1))


def extract_labeled_segments(text: str) -> List[str]:
    matches = list(LABEL_PATTERN.finditer(text))
    if not matches:
        return []

    segments: List[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
    return segments


def clean_segment_text(label: str, segment: str) -> str:
    if label == "Final answer":
        first_line = segment.splitlines()[0].strip()
        return first_line

    contamination_patterns = [
        r"(?im)^return ordered steps only\.?\s*$",
        r"(?im)^reasoning return ordered steps only:?\s*$",
        r"(?im)^reasoning skill required:.*$",
        r"(?im)^problem resolution ordered return only:?\s*$",
    ]

    cleaned = segment
    for pattern in contamination_patterns:
        cleaned = re.sub(pattern, "", cleaned)
    return cleaned.strip()


def parse_generation_chunk(
    raw_text: str,
    existing_trajectory: str,
    max_new_steps: int,
    future_n_steps: int = 0,
) -> Tuple[str, int, bool, str]:
    segments = extract_labeled_segments(raw_text)
    if not segments:
        return "", 0, False, ""

    accepted: List[str] = []
    future_segments: List[str] = []
    existing_has_content = bool(existing_trajectory.strip())
    pending_action = False
    new_steps = 0
    terminal = False
    expected_step_number = count_action_steps(existing_trajectory) + 1
    seen_prefix_labels = set()
    future_steps = 0
    collecting_future = False
    future_pending_action = False
    prefix_has_goal = bool(re.search(r"(?m)^Goal:", existing_trajectory))
    prefix_has_initial_state = bool(re.search(r"(?m)^Initial state:", existing_trajectory))
    prefix_has_plan = bool(re.search(r"(?m)^Plan:", existing_trajectory))

    for segment in segments:
        label = segment.split(":", 1)[0].strip()
        segment = clean_segment_text(label, segment)
        if not segment:
            continue

        if label == "Goal":
            if existing_has_content or accepted or collecting_future:
                continue
            if label in seen_prefix_labels:
                break
            seen_prefix_labels.add(label)
            accepted.append(segment)
            prefix_has_goal = True
            continue

        if label == "Initial state":
            if existing_has_content or collecting_future:
                continue
            if not prefix_has_goal or label in seen_prefix_labels:
                break
            seen_prefix_labels.add(label)
            accepted.append(segment)
            prefix_has_initial_state = True
            new_steps = 1
            continue

        if label == "Plan":
            if existing_has_content or collecting_future:
                continue
            if not (prefix_has_goal and prefix_has_initial_state) or label in seen_prefix_labels:
                break
            seen_prefix_labels.add(label)
            accepted.append(segment)
            prefix_has_plan = True
            new_steps = 2
            continue

        if label.startswith("Action "):
            if not existing_has_content and not (prefix_has_goal and prefix_has_initial_state and prefix_has_plan):
                break
            step_number = extract_step_number(label, "Action")
            if step_number != expected_step_number:
                break

            if not collecting_future:
                if new_steps >= max_new_steps:
                    collecting_future = True
                elif pending_action:
                    break

            if collecting_future:
                if future_n_steps <= 0 or future_steps >= future_n_steps:
                    break
                if future_pending_action:
                    break
                future_segments.append(segment)
                future_pending_action = True
            else:
                accepted.append(segment)
                pending_action = True
            continue

        if label.startswith("State "):
            step_number = extract_step_number(label, "State")
            if step_number != expected_step_number:
                break

            if collecting_future:
                if not future_pending_action:
                    continue
                future_segments.append(segment)
                future_pending_action = False
                future_steps += 1
                expected_step_number += 1
                if future_steps >= future_n_steps:
                    break
            else:
                if not pending_action:
                    continue
                accepted.append(segment)
                pending_action = False
                new_steps += 1
                expected_step_number += 1
                if new_steps >= max_new_steps:
                    collecting_future = future_n_steps > 0
            continue

        if label == "Final answer":
            if not existing_has_content and not (prefix_has_goal and prefix_has_initial_state and prefix_has_plan):
                break
            if collecting_future:
                if future_pending_action and future_segments:
                    future_segments.pop()
                    future_pending_action = False
                future_segments.append(segment)
            else:
                if pending_action and accepted:
                    accepted.pop()
                    pending_action = False
                accepted.append(segment)
                terminal = True
            break

    if pending_action and accepted and accepted[-1].startswith("Action "):
        accepted.pop()
    if future_pending_action and future_segments and future_segments[-1].startswith("Action "):
        future_segments.pop()

    parsed = "\n".join(accepted).strip()
    future_preview = "\n".join(future_segments).strip()
    return parsed, new_steps, terminal, future_preview


def append_trajectory(existing_trajectory: str, parsed_chunk: str) -> str:
    if not existing_trajectory.strip():
        return parsed_chunk.strip()
    if not parsed_chunk.strip():
        return existing_trajectory.strip()
    return f"{existing_trajectory.strip()}\n{parsed_chunk.strip()}"


def strip_graph_segments_for_discriminator(trajectory: str) -> str:
    if not trajectory:
        return ""

    kept_lines: List[str] = []
    for raw_line in trajectory.splitlines():
        line = raw_line.strip()
        if re.match(r"(?i)^(initial\s+graph|graph\s+\d+)\s*:", line):
            continue
        kept_lines.append(raw_line)

    return "\n".join(kept_lines).strip()


def render_messages_without_system_prompt(messages: List[Dict[str, str]]) -> str:
    prompt_parts = ["<|begin_of_text|>"]
    for message in messages:
        role = message["role"].strip()
        content = message["content"]
        prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(prompt_parts)


def build_generation_messages(
    question: str,
    existing_partial_trajectory: str,
) -> List[Dict[str, str]]:
    user_content = (
        "Solve the problem step by step using a structured reasoning format. Return ordered steps only.\n\n"
        f"Problem:\n{question.strip()}"
    )
    return [
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": existing_partial_trajectory,
        },
    ]


def build_generation_prompt(question: str, existing_partial_trajectory: str) -> str:
    messages = build_generation_messages(question, existing_partial_trajectory)
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{messages[0]['content']}"
        "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{messages[1]['content']}"
    )


def build_discrimination_messages(
    question: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    lines = [
        "Compare the provided candidates, considering both their current attributes and potential future outcomes (if applicable).",
        "First, present a detailed comparison, showing every step without skipping any.",
        "Then, provide a conclusion, selecting only one answer.",
        "",
        "Problem:",
        question.strip(),
        "",
    ]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(f"## Candidate {index}")
        candidate_text = candidate.get("comparison_trajectory") or candidate["trajectory"]
        candidate_text = strip_graph_segments_for_discriminator(candidate_text)
        lines.append(candidate_text.strip())
        lines.append("----------------------------------------")
        lines.append('')
    lines.extend(
        [
            "Comparison:",
            "Provide a detailed comparison of all candidates.",
            "Conclusion:",
            "Select exactly one candidate.",
            "The last non-empty line of your response must be exactly: Candidate X",
            "Do not add any explanation after the final line.",
        ]
    )
    return [{"role": "user", "content": "\n".join(lines).strip()}]


def parse_best_candidate(response_text: str, num_candidates: int) -> Optional[int]:
    normalized_response = response_text.replace("**", "")

    non_empty_lines = [line.strip() for line in normalized_response.splitlines() if line.strip()]
    if non_empty_lines:
        last_line = non_empty_lines[-1]
        exact_match = re.fullmatch(r"Candidate\s+([1-9]\d*)", last_line, flags=re.IGNORECASE)
        if exact_match:
            index = int(exact_match.group(1))
            if 1 <= index <= num_candidates:
                return index - 1

    try:
        payload = json.loads(normalized_response)
        value = payload.get("best_candidate")
        if isinstance(value, int) and 1 <= value <= num_candidates:
            return value - 1
        if isinstance(value, str) and value.isdigit():
            index = int(value)
            if 1 <= index <= num_candidates:
                return index - 1
    except json.JSONDecodeError:
        pass

    conclusion_match = re.search(
        r"Conclusion\s*:?[ \t]*\n*(.*)$",
        normalized_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    search_space = conclusion_match.group(1) if conclusion_match else normalized_response

    match = re.search(r"best_candidate\s*[:=]\s*([1-9]\d*)", search_space, flags=re.IGNORECASE)
    if match:
        index = int(match.group(1))
        if 1 <= index <= num_candidates:
            return index - 1

    matches = re.findall(r"candidate\s*([1-9]\d*)", search_space, flags=re.IGNORECASE)
    if matches:
        index = int(matches[-1])
        if 1 <= index <= num_candidates:
            return index - 1

    return None


def build_discrimination_repair_messages(response_text: str, num_candidates: int) -> List[Dict[str, str]]:
    options = ", ".join(f"Candidate {index}" for index in range(1, num_candidates + 1))
    content = (
        "Extract the selected winner from the response below.\n"
        f"Return exactly one line and nothing else. It must be one of: {options}.\n\n"
        "Response:\n"
        f"{response_text.strip()}"
    )
    return [{"role": "user", "content": content}]


def schedule_random_comparisons(
    candidate_ids: List[int],
    cmp_per_opt: int,
    group_size: int,
    rng: random.Random,
) -> List[List[int]]:
    num_candidates = len(candidate_ids)
    target_total_comparisons = ceil((cmp_per_opt * num_candidates) / group_size)

    if num_candidates <= group_size:
        return [candidate_ids[:]]

    all_comparisons = list(combinations(candidate_ids, group_size))
    rng.shuffle(all_comparisons)

    participation_count: Dict[int, int] = defaultdict(int)
    scheduled: List[List[int]] = []

    for comparison in all_comparisons:
        if all(participation_count[candidate_id] < cmp_per_opt for candidate_id in comparison):
            scheduled.append(list(comparison))
            for candidate_id in comparison:
                participation_count[candidate_id] += 1
            if len(scheduled) >= target_total_comparisons:
                break

    if not scheduled:
        return [list(all_comparisons[0])]
    return scheduled


class VLLMDiscriminatorClient:
    def __init__(
        self,
        llm: Any,
        sampling_params_cls: Any,
        lora_request: Optional[Any],
    ) -> None:
        self.llm = llm
        self.sampling_params_cls = sampling_params_cls
        self.lora_request = lora_request
        self.tokenizer = llm.get_tokenizer()

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        return render_messages_without_system_prompt(messages)

    def batch_chat_completion(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not jobs:
            return []

        grouped_jobs: Dict[Tuple[float, float, int], List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for index, job in enumerate(jobs):
            grouped_jobs[(job["temperature"], job["top_p"], job["max_tokens"])].append((index, job))

        results: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
        for (temperature, top_p, max_tokens), indexed_jobs in grouped_jobs.items():
            prompts = [self.build_prompt(job["messages"]) for _, job in indexed_jobs]
            sampling_params = self.sampling_params_cls(
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            if self.lora_request is not None:
                outputs = self.llm.generate(prompts, sampling_params, lora_request=self.lora_request, use_tqdm=False)
            else:
                outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

            for (job_index, job), prompt, output in zip(indexed_jobs, prompts, outputs):
                response_text = output.outputs[0].text if output.outputs else ""
                results[job_index] = {
                    "messages": job["messages"],
                    "prompt": prompt,
                    "response": response_text,
                    "raw": {
                        "prompt": prompt,
                        "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
                        "stop_reason": getattr(output.outputs[0], "stop_reason", None) if output.outputs else None,
                    },
                }

        return [result for result in results if result is not None]


def resolve_adapter_path(script_dir: Path, adapter_path_arg: Optional[str]) -> Optional[Path]:
    if adapter_path_arg is None:
        return None
    adapter_path_str = str(adapter_path_arg).strip()
    if not adapter_path_str or adapter_path_str.lower() == "none":
        return None
    adapter_path = Path(adapter_path_str)
    if adapter_path.is_absolute():
        return adapter_path
    return script_dir / adapter_path


def generate_with_vllm(
    llm: Any,
    sampling_params: Any,
    question: str,
    frontier: List[Dict[str, Any]],
    lora_request: Optional[Any],
    round_index: int,
) -> List[Dict[str, Any]]:
    prompts = [build_generation_prompt(question, node["trajectory"]) for node in frontier]
    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
    else:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    generation_logs: List[Dict[str, Any]] = []
    for node, prompt, output in zip(frontier, prompts, outputs):
        for sample_index, candidate_output in enumerate(output.outputs):
            generation_logs.append(
                {
                    "call_type": "generation",
                    "round_index": round_index,
                    "sample_index": sample_index,
                    "parent_node_id": node["node_id"],
                    "prompt": prompt,
                    "response": candidate_output.text,
                    "finish_reason": candidate_output.finish_reason,
                    "stop_reason": getattr(candidate_output, "stop_reason", None),
                }
            )
    return generation_logs


def execute_discriminator_jobs(
    client: Any,
    jobs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not jobs:
        return []

    responses = client.batch_chat_completion(jobs)
    merged_results: List[Dict[str, Any]] = []
    for job, response in zip(jobs, responses):
        merged = {
            "call_type": job["call_type"],
            "round_index": job["round_index"],
            "comparison_index": job["comparison_index"],
            "candidate_node_ids": job["candidate_node_ids"],
            "temperature": job["temperature"],
            "top_p": job["top_p"],
            "max_tokens": job["max_tokens"],
            "prompt": response["prompt"],
            "response": response["response"],
        }
        merged_results.append(merged)
    return merged_results


def discriminate_candidates(
    question: str,
    candidates: List[Dict[str, Any]],
    client: Any,
    args: argparse.Namespace,
    rng: random.Random,
    round_index: int,
    target_keep_k: Optional[int] = None,
    force_rank: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    effective_keep_k = args.keep_top_k if target_keep_k is None else target_keep_k

    if len(candidates) <= 1:
        return candidates, [], []

    if (not force_rank) and len(candidates) <= effective_keep_k:
        return candidates, [], []

    comparisons = schedule_random_comparisons(
        [candidate["node_id"] for candidate in candidates],
        cmp_per_opt=args.cmp_per_opt,
        group_size=args.group_size,
        rng=rng,
    )
    node_lookup = {candidate["node_id"]: candidate for candidate in candidates}

    jobs: List[Dict[str, Any]] = []
    for comparison_index, comparison in enumerate(comparisons):
        shuffled_comparison = list(comparison)
        rng.shuffle(shuffled_comparison)
        batch = [node_lookup[node_id] for node_id in shuffled_comparison]
        jobs.append(
            {
                "call_type": "discrimination",
                "round_index": round_index,
                "comparison_index": comparison_index,
                "candidate_node_ids": shuffled_comparison,
                "original_candidate_node_ids": comparison,
                "messages": build_discrimination_messages(question, batch),
                "temperature": args.discrimination_temperature,
                "top_p": args.discrimination_top_p,
                "max_tokens": args.discrimination_max_tokens,
            }
        )

    logs = execute_discriminator_jobs(client, jobs)

    for candidate in candidates:
        candidate["score"] = 0

    for log in logs:
        candidate_node_ids = log["candidate_node_ids"]
        local_winner_index = parse_best_candidate(log["response"], len(candidate_node_ids))
        if local_winner_index is None:
            repair_job = {
                "call_type": "discrimination_repair",
                "round_index": round_index,
                "comparison_index": log["comparison_index"],
                "candidate_node_ids": candidate_node_ids,
                "messages": build_discrimination_repair_messages(log["response"], len(candidate_node_ids)),
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 32,
            }
            repair_log = execute_discriminator_jobs(client, [repair_job])[0]
            local_winner_index = parse_best_candidate(repair_log["response"], len(candidate_node_ids))
            log["repair_log"] = repair_log

        if local_winner_index is None:
            log["winner_node_id"] = None
            log["parse_failed"] = True
            continue

        winner_node_id = candidate_node_ids[local_winner_index]
        node_lookup[winner_node_id]["score"] += 1
        log["winner_node_id"] = winner_node_id
        log["parse_failed"] = False

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate["score"],
            int(candidate["terminal"]),
            candidate["steps"],
        ),
        reverse=True,
    )
    kept = ranked_candidates[: effective_keep_k]
    return kept, ranked_candidates, logs


def run_tree_search_for_sample(
    sample: Dict[str, Any],
    llm: Any,
    sampling_params: Any,
    lora_request: Optional[Any],
    client: Any,
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, Any]:
    max_rounds = ceil(args.max_steps / args.search_per_N_steps)
    next_node_id = 1

    frontier: List[Dict[str, Any]] = [
        {
            "node_id": 0,
            "parent_id": None,
            "trajectory": "",
            "steps": 0,
            "terminal": False,
            "score": 0,
        }
    ]
    all_nodes: List[Dict[str, Any]] = [dict(frontier[0])]
    generation_logs_all: List[Dict[str, Any]] = []
    discrimination_logs: List[Dict[str, Any]] = []
    round_summaries: List[Dict[str, Any]] = []

    for round_index in range(max_rounds):
        expandable = [
            node for node in frontier
            if (not node["terminal"]) and node["steps"] < args.max_steps
        ]
        if not expandable:
            break

        generation_logs = generate_with_vllm(
            llm,
            sampling_params,
            sample["question"],
            expandable,
            lora_request,
            round_index,
        )
        generation_logs_all.extend(generation_logs)

        child_lookup: Dict[str, Dict[str, Any]] = {}
        for log in generation_logs:
            parent_node = next(node for node in expandable if node["node_id"] == log["parent_node_id"])
            parsed_chunk, new_steps, terminal, future_preview = parse_generation_chunk(
                log["response"],
                parent_node["trajectory"],
                args.search_per_N_steps,
                args.future_N_steps,
            )
            log["parsed_chunk"] = parsed_chunk
            log["new_steps"] = new_steps
            log["terminal"] = terminal
            log["future_preview"] = future_preview

            if not parsed_chunk:
                continue

            trajectory = append_trajectory(parent_node["trajectory"], parsed_chunk)
            if trajectory in child_lookup:
                continue

            child = {
                "node_id": next_node_id,
                "parent_id": parent_node["node_id"],
                "trajectory": trajectory,
                "future_preview": future_preview,
                "steps": min(count_completed_steps(trajectory), args.max_steps),
                "terminal": terminal or has_final_answer(trajectory),
                "score": 0,
            }
            all_nodes.append(dict(child))
            child["comparison_trajectory"] = strip_graph_segments_for_discriminator(
                    append_trajectory(trajectory, future_preview) if future_preview else trajectory
                )
            child_lookup[trajectory] = child
            
            next_node_id += 1

        children = list(child_lookup.values())
        if not children:
            break

        kept, ranked_candidates, discrimination_logs_round = discriminate_candidates(
            sample["question"],
            children,
            client,
            args,
            rng,
            round_index,
        )
        discrimination_logs.extend(discrimination_logs_round)
        frontier = [
            {
                "node_id": candidate["node_id"],
                "parent_id": candidate["parent_id"],
                "trajectory": candidate["trajectory"],
                "steps": candidate["steps"],
                "terminal": candidate["terminal"],
                "score": candidate["score"],
            }
            for candidate in kept
        ]

        round_summaries.append(
            {
                "round_index": round_index,
                "expanded_node_ids": [node["node_id"] for node in expandable],
                "num_generation_calls": len(generation_logs),
                "num_children": len(children),
                "num_discriminations": len(discrimination_logs_round),
                "kept_node_ids": [node["node_id"] for node in frontier],
                "ranked_node_ids": [candidate["node_id"] for candidate in ranked_candidates],
            }
        )

        if any(node["terminal"] for node in frontier):
            terminal_frontier = [node for node in frontier if node["terminal"]]
            if terminal_frontier:
                frontier = terminal_frontier
                break

    if len(frontier) > 1:
        final_kept, final_ranked, final_logs = discriminate_candidates(
            sample["question"],
            frontier,
            client,
            args,
            rng,
            round_index=max_rounds,
            target_keep_k=1,
            force_rank=True,
        )
        discrimination_logs.extend(final_logs)
        frontier = final_kept[:1]
        round_summaries.append(
            {
                "round_index": max_rounds,
                "expanded_node_ids": [],
                "num_generation_calls": 0,
                "num_children": len(final_ranked),
                "num_discriminations": len(final_logs),
                "kept_node_ids": [node["node_id"] for node in frontier],
                "ranked_node_ids": [candidate["node_id"] for candidate in final_ranked],
            }
        )

    best_node = frontier[0] if frontier else {"trajectory": "", "steps": 0, "terminal": False, "node_id": None}
    final_trajectory = best_node["trajectory"].strip()
    evaluation = evaluate_prediction(sample, final_trajectory)

    return {
        "id": sample.get("id"),
        "question": sample.get("question"),
        "gold_answer": evaluation["gold_answer"],
        "gold_answer_normalized": evaluation["gold_answer_normalized"],
        "pred_answer": evaluation["pred_answer"],
        "correct": evaluation["correct"],
        "final_trajectory": final_trajectory,
        "best_node_id": best_node.get("node_id"),
        "completed_steps": best_node["steps"],
        "terminated": best_node["terminal"],
        "all_nodes": all_nodes,
        "round_summaries": round_summaries,
        "generation_logs": generation_logs_all,
        "discrimination_logs": discrimination_logs,
    }


def load_existing_results(output_path: Path) -> Dict[str, Any]:
    if not output_path.exists():
        return {"results": []}
    with output_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_discriminator_base_model(args: argparse.Namespace) -> str:
    if args.discriminator_base_model:
        return args.discriminator_base_model
    return args.generator_base_model


def resolve_discriminator_client(
    args: argparse.Namespace,
    LLM: Any,
    SamplingParams: Any,
    LoRARequest: Any,
    discriminator_adapter_path: Optional[Path],
) -> Any:
    discriminator_tensor_parallel_size = (
        args.discriminator_tensor_parallel_size
        if args.discriminator_tensor_parallel_size is not None
        else args.generator_tensor_parallel_size
    )
    discriminator_llm = LLM(
        model=resolve_discriminator_base_model(args),
        enable_lora=bool(discriminator_adapter_path),
        max_model_len=args.discrimination_max_model_len,
        tensor_parallel_size=discriminator_tensor_parallel_size,
        gpu_memory_utilization=args.discriminator_gpu_memory_utilization,
        trust_remote_code=True,
    )
    discriminator_lora_request = (
        LoRARequest("discriminator_lora", 2, str(discriminator_adapter_path))
        if discriminator_adapter_path is not None else None
    )
    return VLLMDiscriminatorClient(
        llm=discriminator_llm,
        sampling_params_cls=SamplingParams,
        lora_request=discriminator_lora_request,
    )


def resolve_gpu_memory_utilizations(args: argparse.Namespace) -> Tuple[float, Optional[float]]:
    generation_util = args.generation_gpu_memory_utilization
    discriminator_util = args.discriminator_gpu_memory_utilization
    return generation_util, discriminator_util


def main() -> None:
    args = parse_args()
    validate_shard_args(args)
    rng = random.Random(args.seed)

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as error:
        raise RuntimeError(
            "vLLM is not installed. Install it before running this evaluator."
        ) from error

    script_dir = Path(__file__).resolve().parent
    client = None
    generation_gpu_memory_utilization, discriminator_gpu_memory_utilization = resolve_gpu_memory_utilizations(args)
    args.generation_gpu_memory_utilization = generation_gpu_memory_utilization
    args.discriminator_gpu_memory_utilization = discriminator_gpu_memory_utilization

    output_path = Path(__file__).resolve().parent / args.output_path
    resume_output_path = resolve_optional_path(script_dir, args.resume_from_output) or output_path
    generator_adapter_path = resolve_adapter_path(script_dir, args.generator_adapter_path)
    discriminator_adapter_path = resolve_adapter_path(script_dir, args.discriminator_adapter_path)
    if args.discriminator_tensor_parallel_size is None:
        args.discriminator_tensor_parallel_size = args.generator_tensor_parallel_size
    client = resolve_discriminator_client(
        args,
        LLM,
        SamplingParams,
        LoRARequest,
        discriminator_adapter_path,
    )

    if args.num_shards > 1 and args.resume:
        if resume_output_path.resolve() == output_path.resolve() and not output_path.exists():
            raise RuntimeError(
                "Sharded resume requires each shard to write to its own --output_path. "
                "Use --resume_from_output to point to the shared existing result file and set a shard-specific --output_path."
            )

    llm = LLM(
        model=args.generator_base_model,
        enable_lora=bool(generator_adapter_path),
        max_model_len=args.generation_max_model_len,
        tensor_parallel_size=args.generator_tensor_parallel_size,
        gpu_memory_utilization=args.generation_gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        n=args.num_candidates,
        temperature=args.generation_temperature,
        top_p=args.generation_top_p,
        max_tokens=args.generation_max_tokens,
    )
    lora_request = (
        LoRARequest("tree_search_lora", 1, str(generator_adapter_path))
        if generator_adapter_path is not None else None
    )

    samples = load_test_samples(args.data, limit=args.limit)
    shard_samples = filter_samples_for_shard(samples, args.num_shards, args.shard_index)
    should_resume = args.resume and not args.overwrite
    if should_resume:
        resume_load_path = output_path if output_path.exists() else resume_output_path
        payload = load_existing_results(resume_load_path)
    else:
        payload = {"results": []}
    existing_results = list(payload.get("results", []))
    if not existing_results:
        existing_results = list(payload.get("latest_results", []))
    result_by_id = {
        result.get("id"): result
        for result in existing_results
        if result.get("id") is not None
    }
    all_results = list(existing_results)

    processed_since_save = 0
    processed_new_samples = 0
    existing_ids = set(result_by_id.keys())
    evaluation_samples = [
        sample for sample in shard_samples
        if sample.get("id") not in existing_ids
    ]

    progress_bar = tqdm(evaluation_samples, desc="Evaluation")

    for sample in progress_bar:
        sample_id = sample.get("id")
        result = run_tree_search_for_sample(
            sample,
            llm,
            sampling_params,
            lora_request,
            client,
            args,
            rng,
        )
        result_by_id[sample_id] = result
        all_results.append(result)

        processed_since_save += 1
        processed_new_samples += 1

        current_results = build_ordered_results(samples, result_by_id)
        current_correct = sum(int(item.get("correct", False)) for item in current_results)

        if processed_new_samples % 10 == 0:
            progress_bar.set_postfix(
                running_acc=f"{(current_correct / len(current_results)):.4f}" if current_results else "0.0000",
                solved=current_correct,
                evaluated=len(current_results),
            )

        if processed_since_save >= args.save_every:
            payload = build_payload(
                args,
                args.data,
                generator_adapter_path,
                discriminator_adapter_path,
                all_results,
                current_results,
            )
            save_results(output_path, payload)
            processed_since_save = 0

    current_results = build_ordered_results(samples, result_by_id)
    current_correct = sum(int(item.get("correct", False)) for item in current_results)
    if processed_new_samples % 10 != 0 and current_results:
        progress_bar.set_postfix(
            running_acc=f"{(current_correct / len(current_results)):.4f}",
            solved=current_correct,
            evaluated=len(current_results),
        )
    progress_bar.close()

    final_results = build_ordered_results(samples, result_by_id)
    payload = build_payload(
        args,
        args.data,
        generator_adapter_path,
        discriminator_adapter_path,
        all_results,
        final_results,
    )
    save_results(output_path, payload)
    print(f"Evaluated {payload['total']} test samples.")
    print(f"Correct: {payload['correct']}")
    print(f"Accuracy: {payload['accuracy']:.4f}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()