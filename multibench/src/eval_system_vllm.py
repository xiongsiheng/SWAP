#!/usr/bin/env python3
"""Select generator candidates with a listwise discriminator LoRA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from utils import (
    accuracy_summary,
    build_selector_example,
    ensure_grading_available,
    grade_answer,
    majority_candidate,
    parse_selected_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-output", type=Path, required=True)
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-options", type=int, default=8)
    parser.add_argument("--max-trajectory-chars", type=int, default=1400)
    parser.add_argument("--include-graph", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant-index", type=int, default=0)
    parser.add_argument(
        "--skip-grading",
        action="store_true",
        help="Produce predictions without benchmark correctness metrics.",
    )
    args = parser.parse_args()
    if not 2 <= args.max_options <= 17:
        parser.error("--max-options must be in [2, 17]")
    if args.variant_index < 0:
        parser.error("--variant-index must be non-negative")
    return args


def _grade(answer: str, record: dict[str, Any], skip: bool) -> dict[str, Any]:
    return {"correct": None, "error": None} if skip else grade_answer(answer, record)


def main() -> None:
    args = parse_args()
    config_path = args.adapter_path / "adapter_config.json"
    weight_path = args.adapter_path / "adapter_model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Incomplete LoRA adapter: {args.adapter_path}")

    generator_payload = json.loads(args.generator_output.read_text(encoding="utf-8"))
    if generator_payload.get("stage") != "generator":
        raise ValueError("Input is not a multibench generator output")
    if not args.skip_grading:
        ensure_grading_available(str(generator_payload["benchmark"]))
    source_results = generator_payload["results"]
    examples: dict[str, dict[str, Any]] = {}
    for result in source_results:
        example = build_selector_example(
            result,
            max_options=args.max_options,
            max_trajectory_chars=args.max_trajectory_chars,
            include_graph=args.include_graph,
            seed=args.seed,
            variant_index=args.variant_index,
        )
        if example is not None:
            examples[str(result["record"]["id"])] = example

    responses: dict[str, str] = {}
    started = time.perf_counter()
    if examples:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        ordered = list(examples.values())
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": example["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in ordered
        ]
        llm = LLM(
            model=args.base_model,
            enable_lora=True,
            max_lora_rank=args.max_lora_rank,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        lora_request = LoRARequest(
            f"{generator_payload['benchmark']}-discriminator",
            1,
            str(args.adapter_path.resolve()),
        )
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed,
            ),
            lora_request=lora_request,
        )
        responses = {
            example["id"]: output.outputs[0].text.strip()
            for example, output in zip(ordered, outputs)
        }

    results: list[dict[str, Any]] = []
    greedy_grades: list[dict[str, Any]] = []
    system_grades: list[dict[str, Any]] = []
    for source in source_results:
        record = source["record"]
        record_id = str(record["id"])
        candidates = source["candidates"]
        greedy = candidates[0]
        example = examples.get(record_id)
        response = responses.get(record_id, "")
        parsed_label = parse_selected_label(response) if response else None
        selected = None
        selection_source = "majority_fallback"
        if example is not None and parsed_label is not None:
            option = next(
                (row for row in example["options"] if row["label"] == parsed_label),
                None,
            )
            if option is not None:
                selected = {
                    "answer": option["answer"],
                    "answer_key": option["answer_key"],
                }
                selection_source = "discriminator"
        if selected is None:
            selected = majority_candidate(candidates)

        grade_cache: dict[str, dict[str, Any]] = {}

        def cached_grade(candidate: dict[str, Any]) -> dict[str, Any]:
            key = str(candidate.get("answer_key") or candidate.get("answer") or "")
            if key not in grade_cache:
                grade_cache[key] = _grade(str(candidate.get("answer") or ""), record, args.skip_grading)
            return grade_cache[key]

        greedy_grade = cached_grade(greedy)
        selected_grade = cached_grade(selected)
        greedy_grades.append(greedy_grade)
        system_grades.append(selected_grade)
        results.append({
            "id": record["id"],
            "benchmark": record["benchmark"],
            "split": record["split"],
            "question": record["question"],
            "gold_answer": record.get("gold_answer"),
            "greedy_answer": greedy.get("answer", ""),
            "selected_answer": selected.get("answer", ""),
            "selection_source": selection_source,
            "discriminator_invoked": example is not None,
            "discriminator_prompt": example["prompt"] if example else None,
            "discriminator_options": example["options"] if example else [],
            "discriminator_response": response or None,
            "parsed_label": parsed_label,
            "greedy_grade": greedy_grade,
            "system_grade": selected_grade,
        })

    payload = {
        "schema_version": 1,
        "stage": "system",
        "benchmark": generator_payload["benchmark"],
        "split": generator_payload["split"],
        "base_model": args.base_model,
        "generator_adapter_path": generator_payload["adapter_path"],
        "discriminator_adapter_path": str(args.adapter_path.resolve()),
        "config": {
            "max_options": args.max_options,
            "max_trajectory_chars": args.max_trajectory_chars,
            "include_graph": args.include_graph,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "variant_index": args.variant_index,
            "skip_grading": args.skip_grading,
        },
        "summary": {
            "records": len(results),
            "discriminator_prompts": len(examples),
            "discriminator_selections": sum(
                row["selection_source"] == "discriminator" for row in results
            ),
            "invalid_discriminator_responses": sum(
                row["discriminator_invoked"] and row["parsed_label"] is None
                for row in results
            ),
            "generator_greedy": accuracy_summary(greedy_grades),
            "system": accuracy_summary(system_grades),
            "wall_seconds": time.perf_counter() - started,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved system output to {args.output}")


if __name__ == "__main__":
    main()
