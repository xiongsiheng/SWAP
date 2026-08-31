#!/usr/bin/env python3
"""Generate one greedy and N sampled SWAP trajectories per benchmark item."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from utils import BENCHMARK_SPLITS, build_candidate, build_generator_messages, load_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_SPLITS), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    return args


def _prompt(tokenizer: Any, record: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        build_generator_messages(record["question"], record["benchmark"]),
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    args = parse_args()
    config_path = args.adapter_path / "adapter_config.json"
    weight_path = args.adapter_path / "adapter_model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Incomplete LoRA adapter: {args.adapter_path}")

    records = load_benchmark(args.benchmark, args.split)[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No benchmark records selected")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    prompts = [_prompt(tokenizer, record) for record in records]
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    lora_request = LoRARequest(
        f"{args.benchmark}-generator", 1, str(args.adapter_path.resolve())
    )

    started = time.perf_counter()
    greedy_outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            logprobs=0,
            seed=args.seed,
        ),
        lora_request=lora_request,
        use_tqdm=True,
    )
    sampled_outputs = llm.generate(
        prompts,
        SamplingParams(
            n=args.samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            logprobs=0,
            seed=args.seed + 1009,
        ),
        lora_request=lora_request,
        use_tqdm=True,
    )

    results: list[dict[str, Any]] = []
    for record, greedy_output, sampled_output in zip(
        records, greedy_outputs, sampled_outputs
    ):
        greedy = greedy_output.outputs[0]
        candidates = [
            build_candidate(
                record,
                greedy.text,
                greedy.token_ids,
                greedy.cumulative_logprob,
                "greedy",
                0.0,
                0,
            )
        ]
        for index, completion in enumerate(sampled_output.outputs, start=1):
            candidates.append(
                build_candidate(
                    record,
                    completion.text,
                    completion.token_ids,
                    completion.cumulative_logprob,
                    "sample",
                    args.temperature,
                    index,
                )
            )
        results.append({"record": record, "candidates": candidates})

    payload = {
        "schema_version": 1,
        "stage": "generator",
        "benchmark": args.benchmark,
        "split": args.split,
        "base_model": args.base_model,
        "adapter_path": str(args.adapter_path.resolve()),
        "config": {
            "offset": args.offset,
            "limit": args.limit,
            "samples": args.samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "summary": {
            "records": len(results),
            "candidates": sum(len(row["candidates"]) for row in results),
            "parse_valid": sum(
                candidate["trajectory"] is not None
                for row in results
                for candidate in row["candidates"]
            ),
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
    print(f"Saved generator output to {args.output}")


if __name__ == "__main__":
    main()

