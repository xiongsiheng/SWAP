import argparse
import json
import os
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT with LoRA for Discriminator."
    )
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--adapter_path", default=None)

    parser.add_argument("--hf_dataset", default=None)
    parser.add_argument("--hf_config", default=None)
    parser.add_argument("--hf_split", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--max_seq_len", type=int, default=4096)
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
    if not args.use_wandb or not is_main_process():
        os.environ["WANDB_DISABLED"] = "true"
        return None

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install wandb`."
        ) from error

    run_name = args.wandb_run_name or Path(args.output_dir).name
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
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


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    text = text.replace("\\t", "\t")
    return text.strip()


def build_messages(prompt: str, response: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": response.strip()},
    ]


def build_prompt(prompt: str, response: str) -> str:
    messages = build_messages(prompt, response)
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{messages[0]['content']}"
        "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{messages[1]['content']}"
        "<|eot_id|>"
    )


def normalize_sample(sample: Dict[str, Any], source: str, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    prompt = clean_text(sample.get("prompt"))
    response = clean_text(sample.get("response"))
    if not prompt or not response:
        return None

    return {
        "source": source,
        "prompt": prompt,
        "response": response,
    }




def load_hf_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset = load_dataset(args.hf_dataset, args.hf_config)
    if args.hf_split not in dataset:
        raise ValueError(
            f"Split '{args.hf_split}' not found in {args.hf_dataset}/{args.hf_config}. Available: {list(dataset.keys())}"
        )

    normalized: List[Dict[str, Any]] = []
    for row_index, sample in enumerate(dataset[args.hf_split]):
        normalized_sample = normalize_sample(sample, source=f"hf:{args.hf_split}", args=args)
        if normalized_sample is not None:
            normalized_sample["id"] = str(row_index)
            normalized.append(normalized_sample)
        if args.max_samples is not None and len(normalized) >= args.max_samples:
            break
    return normalized


def build_dataset(args: argparse.Namespace) -> Dataset:
    samples = load_hf_samples(args)

    if not samples:
        raise RuntimeError("Could not load any usable discriminator SFT samples from the Hugging Face dataset.")

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


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed_training() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_rank() == 0


def configure_cuda_device(torch_module) -> Optional[int]:
    if not torch_module.cuda.is_available():
        return None

    if not is_distributed_training():
        return torch_module.cuda.current_device()

    local_rank = get_local_rank()
    device_count = torch_module.cuda.device_count()
    if local_rank < 0 or local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is out of range for {device_count} visible CUDA device(s)."
        )

    torch_module.cuda.set_device(local_rank)
    return local_rank


def add_text_column(dataset: Dataset) -> Dataset:
    def build_row(sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"text": build_prompt(sample["prompt"], sample["response"])}

    return dataset.map(build_row, remove_columns=dataset.column_names)


def split_train_val_dataset(dataset: Dataset, args: argparse.Namespace) -> Tuple[Dataset, Optional[Dataset]]:
    if args.val_ratio <= 0 or len(dataset) < 2:
        return dataset, None

    split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_dataset = split["train"].shuffle(seed=args.seed)
    val_dataset = split["test"]
    if len(val_dataset) == 0:
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
        prompt = build_prompt(sample["prompt"], sample["response"])
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
        f"source={max_sample.get('source', 'unknown')} id={max_sample.get('id', 'unknown')}"
    )


def print_prompt_samples(dataset: Dataset, num_samples: int = 3) -> None:
    sample_count = min(num_samples, len(dataset))
    for index in range(sample_count):
        sample = dataset[index]
        prompt = build_prompt(sample["prompt"], sample["response"])
        print(f"===== Sample {index + 1} | {sample['source']} | {sample['id']} =====")
        print(prompt)
        print()


def save_prompt_samples(
    dataset: Dataset,
    output_dir: Path,
    num_samples: int = 3,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(num_samples, len(dataset))
    for index in range(sample_count):
        sample = dataset[index]
        prompt = build_prompt(sample["prompt"], sample["response"])
        file_name = f"sample_{index + 1:03d}_{sample['source'].replace(':', '_')}_{sample['id']}.txt"
        (output_dir / file_name).write_text(prompt, encoding="utf-8")
    print(f"Saved {sample_count} prompt samples to {output_dir}")


def train_lora(args: argparse.Namespace, dataset: Dataset, tokenizer) -> None:
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for training.")

    configure_cuda_device(torch)
    train_source_dataset, val_source_dataset = split_train_val_dataset(dataset, args)
    train_dataset = add_text_column(train_source_dataset)
    val_dataset = add_text_column(val_source_dataset) if val_source_dataset is not None else None
    wandb_module = setup_wandb(args, len(train_dataset), len(val_dataset) if val_dataset is not None else 0)

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "torch_dtype": torch_dtype,   # use torch_dtype, not dtype
    }
    if torch.cuda.is_available() and not is_distributed_training():
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    base_model.config.use_cache = False

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset is not None else "no",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
        report_to="wandb" if args.use_wandb and is_main_process() else "none",
        run_name=args.wandb_run_name or Path(args.output_dir).name,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )

    if args.adapter_path:
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            is_trainable=True,
        )
        model.enable_input_require_grads()
        if is_main_process():
            model.print_trainable_parameters()

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        trainer = SFTTrainer(
            model=base_model,
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

    if args.train:
        import torch

        configure_cuda_device(torch)

    dataset = build_dataset(args)
    tokenizer = maybe_load_tokenizer(args.model_id)

    print(f"Loaded {len(dataset)} discriminator SFT samples after filtering.")
    print(f"Distributed training: {is_distributed_training()} (WORLD_SIZE={get_world_size()})")
    print(f"RANK={get_rank()}")
    print(f"LOCAL_RANK={get_local_rank()}")
    source_counts: Dict[str, int] = {}
    for source in dataset["source"]:
        source_counts[source] = source_counts.get(source, 0) + 1
    print(f"Source counts: {source_counts}")

    if args.analyze_seq_len:
        length_stats = analyze_sequence_lengths(dataset, tokenizer)
        print_sequence_length_stats(length_stats)

    print_prompt_samples(dataset, num_samples=args.print_samples)

    if args.train:
        train_lora(args, dataset, tokenizer)


if __name__ == "__main__":
    main()