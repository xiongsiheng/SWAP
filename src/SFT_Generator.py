import argparse
import ast
import json
import os
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, load_dataset





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT with LoRA for Generator."
    )
    parser.add_argument("--model_id", default=None)

    parser.add_argument("--hf_dataset", default=None)
    parser.add_argument("--hf_config", default=None)
    parser.add_argument("--hf_split", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    
    parser.add_argument("--use_graph", action="store_true")
    parser.add_argument("--convert_graph_format", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    
    parser.add_argument("--print_samples", type=int, default=3)
    parser.add_argument("--analyze_seq_len", action="store_true")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
        
    return parser.parse_args()


def setup_wandb(args: argparse.Namespace, train_size: int, val_size: int) -> Optional[Any]:
    if not args.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return None

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install wandb`."
        ) from error

    run_name = args.wandb_run_name or Path(args.output_dir).name
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_id": args.model_id,
            "hf_dataset": args.hf_dataset,
            "hf_config": args.hf_config,
            "hf_split": args.hf_split,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "num_train_epochs": args.num_train_epochs,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "train_size": train_size,
            "val_size": val_size,
        },
    )
    return wandb


def is_positive_label(label: Any) -> bool:
    if label is None:
        return True
    if isinstance(label, bool):
        return label
    if isinstance(label, (int, float)):
        return label > 0
    normalized = str(label).strip().lower()
    return normalized in {"positive", "pos", "true", "1", "correct", "yes"}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)

    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    text = text.replace("\\t", "\t")
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def stringify_step_value(value: Any) -> str:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, (dict, list)):
        return clean_text(json.dumps(value, ensure_ascii=False))
    return clean_text(str(value))


def try_parse_literal(text: str) -> Optional[Any]:
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def format_graph_value(value: Any) -> str:
    graph = value
    if isinstance(value, str):
        graph = try_parse_literal(value)

    if not isinstance(graph, dict):
        return stringify_step_value(value)

    statements = graph.get("Statement")
    entailment = graph.get("Entailment")
    if not isinstance(statements, dict) or not isinstance(entailment, dict):
        return stringify_step_value(value)

    formatted_lines: List[str] = []
    for statement_id, statement_text in statements.items():
        statement_key = clean_text(statement_id)
        statement_value = clean_text(statement_text)
        if statement_value.endswith("."):
            statement_value = statement_value[:-1].rstrip()
        dependencies = entailment.get(statement_id)

        if isinstance(dependencies, list) and dependencies:
            dependency_text = " & ".join(clean_text(item) for item in dependencies if clean_text(item))
            if dependency_text:
                formatted_lines.append(f"{dependency_text} -> {statement_key}: {statement_value}")
                continue

        formatted_lines.append(f"{statement_key}: {statement_value}")

    return "; ".join(formatted_lines)


def maybe_format_graph_step(key: str, value: Any, convert_graph_format: bool = False) -> str:
    if convert_graph_format and "graph" in key.lower():
        return format_graph_value(value)
    return stringify_step_value(value)


def extract_steps_from_mapping(
    mapping: Dict[str, Any],
    use_graph: bool = False,
    convert_graph_format: bool = False,
) -> List[Tuple[str, str]]:
    steps: List[Tuple[str, str]] = []
    for key, value in mapping.items():
        key_str = clean_text(key)
        if "graph" in key_str.lower() and not use_graph:
            continue
        steps.append((key_str, maybe_format_graph_step(key_str, value, convert_graph_format=convert_graph_format)))
    return steps


def extract_steps_from_list(
    items: List[Any],
    use_graph: bool = False,
    convert_graph_format: bool = False,
) -> List[Tuple[str, str]]:
    steps: List[Tuple[str, str]] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            nested_steps = extract_steps_from_mapping(
                item,
                use_graph=use_graph,
                convert_graph_format=convert_graph_format,
            )
            if nested_steps:
                steps.extend(nested_steps)
            else:
                steps.append((f"Step {index}", stringify_step_value(item)))
        else:
            steps.append((f"Step {index}", stringify_step_value(item)))
    return steps


def extract_steps_from_text(
    text: str,
    use_graph: bool = False,
    convert_graph_format: bool = False,
) -> List[Tuple[str, str]]:
    parsed_literal = try_parse_literal(text)
    if isinstance(parsed_literal, dict):
        return extract_steps_from_mapping(
            parsed_literal,
            use_graph=use_graph,
            convert_graph_format=convert_graph_format,
        )
    if isinstance(parsed_literal, list):
        return extract_steps_from_list(
            parsed_literal,
            use_graph=use_graph,
            convert_graph_format=convert_graph_format,
        )

    steps: List[Tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().rstrip(",")
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = clean_text(key)
        if "graph" in key.lower() and not use_graph:
            continue
        steps.append((key, maybe_format_graph_step(key, clean_text(value), convert_graph_format=convert_graph_format)))
    return steps


def trajectory_to_steps(
    trajectory: Any,
    use_graph: bool = False,
    convert_graph_format: bool = False,
) -> List[Tuple[str, str]]:
    if isinstance(trajectory, dict):
        return extract_steps_from_mapping(
            trajectory,
            use_graph=use_graph,
            convert_graph_format=convert_graph_format,
        )
    if isinstance(trajectory, list):
        return extract_steps_from_list(
            trajectory,
            use_graph=use_graph,
            convert_graph_format=convert_graph_format,
        )
    if isinstance(trajectory, str):
        return extract_steps_from_text(
            trajectory,
            use_graph=use_graph,
            convert_graph_format=convert_graph_format,
        )
    return [("Trajectory", stringify_step_value(trajectory))]


def format_steps(
    steps: Iterable[Tuple[str, str]],
    convert_graph_format: bool = False,
) -> str:
    formatted_steps = []
    for key, value in steps:
        if not value:
            continue
        if "\n" in value:
            formatted_steps.append(f"{key}:\n{value}")
        else:
            formatted_steps.append(f"{key}: {value}")
    return "\n".join(formatted_steps)


def build_messages(question: str, cleaned_trajectory: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Solve the problem step by step using a structured reasoning format. Return ordered steps only.\n\n"
                f"Problem:\n{question.strip()}"
            ),
        },
        {
            "role": "assistant",
            "content": cleaned_trajectory,
        },
    ]


def build_prompt(question: str, cleaned_trajectory: str, tokenizer=None) -> str:
    messages = build_messages(question, cleaned_trajectory)
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{messages[0]['content']}"
        "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{messages[1]['content']}"
        "<|eot_id|>"
    )


def normalize_sample(
    sample: Dict[str, Any],
    source: str,
    use_graph: bool = False,
    convert_graph_format: bool = False,
) -> Optional[Dict[str, Any]]:
    if not is_positive_label(sample.get("label")):
        return None

    question = sample.get("question") or sample.get("problem")
    trajectory = sample.get("trajectory")
    if not question or trajectory is None:
        return None

    steps = trajectory_to_steps(
        trajectory,
        use_graph=use_graph,
        convert_graph_format=convert_graph_format,
    )
    cleaned_trajectory = format_steps(
        steps,
        convert_graph_format=convert_graph_format,
    )
    if not cleaned_trajectory:
        return None

    return {
        "source": source,
        "id": sample.get("id") or sample.get("idx") or sample.get("filename") or "unknown",
        "question": clean_text(question),
        "trajectory": cleaned_trajectory,
        "num_steps": len(steps),
    }



def load_hf_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset = load_dataset(args.hf_dataset, args.hf_config)
    if args.hf_split not in dataset:
        raise ValueError(
            f"Split '{args.hf_split}' not found in {args.hf_dataset}/{args.hf_config}. Available: {list(dataset.keys())}"
        )

    normalized: List[Dict[str, Any]] = []
    for sample in dataset[args.hf_split]:
        normalized_sample = normalize_sample(
            sample,
            source=f"hf:{args.hf_split}",
            use_graph=args.use_graph,
            convert_graph_format=args.convert_graph_format,
        )
        if normalized_sample is not None:
            normalized.append(normalized_sample)
    return normalized


def build_dataset(args: argparse.Namespace) -> Dataset:
    samples = load_hf_samples(args)

    if not samples:
        raise RuntimeError("Could not load any usable samples from the Hugging Face dataset.")

    dataset = Dataset.from_list(samples)
    dataset = dataset.shuffle(seed=args.seed)

    return dataset


def maybe_load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as error:
        print(f"Warning: tokenizer could not be loaded for chat template formatting. {error}")
        return None


def add_text_column(dataset: Dataset, tokenizer=None) -> Dataset:
    def build_row(sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": build_prompt(sample["question"], sample["trajectory"], tokenizer=tokenizer),
        }

    text_dataset = dataset.map(build_row, remove_columns=dataset.column_names)
    return text_dataset


def split_train_val_dataset(dataset: Dataset, args: argparse.Namespace) -> Tuple[Dataset, Optional[Dataset]]:
    if args.val_ratio > 0:
        split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)
        train_dataset = split["train"]
        val_dataset = split["test"]
    else:
        train_dataset = dataset
        val_dataset = None

    if val_dataset is not None and len(val_dataset) == 0:
        val_dataset = None

    return train_dataset, val_dataset


def compute_quantile(sorted_values: List[int], quantile: float) -> int:
    if not sorted_values:
        return 0
    index = max(0, min(len(sorted_values) - 1, ceil(quantile * len(sorted_values)) - 1))
    return sorted_values[index]


def analyze_sequence_lengths(dataset: Dataset, tokenizer) -> Dict[str, Any]:
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for sequence length analysis.")

    lengths: List[int] = []
    max_length = -1
    max_sample: Optional[Dict[str, Any]] = None

    for sample in dataset:
        prompt = build_prompt(sample["question"], sample["trajectory"], tokenizer=tokenizer)
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        lengths.append(token_count)
        if token_count > max_length:
            max_length = token_count
            max_sample = sample

    lengths.sort()
    average_length = sum(lengths) / len(lengths)
    return {
        "count": len(lengths),
        "min": lengths[0],
        "max": lengths[-1],
        "mean": average_length,
        "p95": compute_quantile(lengths, 0.95),
        "p99": compute_quantile(lengths, 0.99),
        "max_sample": max_sample,
    }


def print_sequence_length_stats(stats: Dict[str, Any]) -> None:
    max_sample = stats["max_sample"] or {}
    print("Sequence length stats:")
    print(f"  Count: {stats['count']}")
    print(f"  Min: {stats['min']}")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  P95: {stats['p95']}")
    print(f"  P99: {stats['p99']}")
    print(f"  Max: {stats['max']}")
    print(
        "  Longest sample: "
        f"source={max_sample.get('source', 'unknown')} id={max_sample.get('id', 'unknown')} steps={max_sample.get('num_steps', 'unknown')}"
    )


def print_prompt_samples(dataset: Dataset, tokenizer=None, num_samples: int = 3) -> None:
    sample_count = min(num_samples, len(dataset))
    for index in range(sample_count):
        sample = dataset[index]
        prompt = build_prompt(sample["question"], sample["trajectory"], tokenizer=tokenizer)
        print(f"===== Sample {index + 1} | {sample['source']} | {sample['id']} =====")
        print(prompt)
        print()


def train_lora(args: argparse.Namespace, dataset: Dataset, tokenizer) -> None:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for training.")

    train_source_dataset, val_source_dataset = split_train_val_dataset(dataset, args)
    train_dataset = add_text_column(train_source_dataset, tokenizer=tokenizer)
    val_dataset = add_text_column(val_source_dataset, tokenizer=tokenizer) if val_source_dataset is not None else None
    wandb_module = setup_wandb(args, len(train_dataset), len(val_dataset) if val_dataset is not None else 0)

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=12,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset is not None else "no",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.wandb_run_name or Path(args.output_dir).name,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(Path(args.output_dir) / "final"))
    if wandb_module is not None:
        wandb_module.finish()


def main() -> None:
    args = parse_args()

    dataset = build_dataset(args)
    tokenizer = maybe_load_tokenizer(args.model_id)

    print(f"Loaded {len(dataset)} positive trajectories after filtering negatives.")
    source_counts: Dict[str, int] = {}
    for source in dataset["source"]:
        source_counts[source] = source_counts.get(source, 0) + 1
    print(f"Source counts: {source_counts}")

    if args.analyze_seq_len:
        length_stats = analyze_sequence_lengths(dataset, tokenizer)
        print_sequence_length_stats(length_stats)

    print_prompt_samples(dataset, tokenizer=tokenizer, num_samples=args.print_samples)

    if args.train:
        train_lora(args, dataset, tokenizer)


if __name__ == "__main__":
    main()