import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

from SFT_discriminator import (
    clean_text,
    configure_cuda_device,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed_training,
    is_main_process,
    maybe_load_tokenizer,
    setup_wandb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DPO with LoRA for Discriminator."
    )
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--ref_adapter_path", default=None)

    parser.add_argument("--hf_dataset", default=None)
    parser.add_argument("--hf_config", default=None)
    parser.add_argument("--hf_split", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--print_samples", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=2560)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", default="sigmoid")
    
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


def resolve_path(script_dir: Path, path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return script_dir / path


def load_dataset_split(args: argparse.Namespace):
    resolved_path = resolve_path(Path(__file__).resolve().parent, args.hf_dataset)
    if resolved_path.exists():
        if resolved_path.is_dir():
            candidate_files = [
                resolved_path / f"{args.hf_split}.jsonl",
                resolved_path / f"{args.hf_split}.json",
            ]
            data_file = next((path for path in candidate_files if path.exists()), None)
            if data_file is None:
                raise RuntimeError(
                    f"Could not find a split file for '{args.hf_split}' in {resolved_path}. "
                    "Expected one of: "
                    f"{candidate_files[0].name}, {candidate_files[1].name}"
                )
            return load_dataset("json", data_files={args.hf_split: str(data_file)})

        if resolved_path.is_file():
            return load_dataset("json", data_files={args.hf_split: str(resolved_path)})

    if args.hf_config:
        return load_dataset(args.hf_dataset, args.hf_config)
    return load_dataset(args.hf_dataset)


def build_dpo_prompt(prompt: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt.strip()}"
        "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_dpo_response(response: str) -> str:
    return f"{response.strip()}<|eot_id|>"


def normalize_sample(sample: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    prompt = clean_text(sample.get("prompt"))
    chosen = clean_text(sample.get("chosen"))
    rejected = clean_text(sample.get("rejected"))
    if not prompt or not chosen or not rejected:
        return None

    return {
        "source": source,
        "prompt": build_dpo_prompt(prompt),
        "chosen": build_dpo_response(chosen),
        "rejected": build_dpo_response(rejected)
    }




def load_hf_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset = load_dataset_split(args)
    if args.hf_split not in dataset:
        dataset_name = args.hf_dataset if args.hf_config is None else f"{args.hf_dataset}/{args.hf_config}"
        raise ValueError(
            f"Split '{args.hf_split}' not found in {dataset_name}. Available: {list(dataset.keys())}"
        )

    normalized: List[Dict[str, Any]] = []
    for row_index, sample in enumerate(dataset[args.hf_split]):
        normalized_sample = normalize_sample(sample, source=f"hf:{args.hf_split}")
        if normalized_sample is not None:
            normalized_sample["id"] = str(row_index)
            normalized.append(normalized_sample)
        if args.max_samples is not None and len(normalized) >= args.max_samples:
            break
    return normalized


def build_dataset(args: argparse.Namespace) -> Dataset:
    samples = load_hf_samples(args)

    if not samples:
        raise RuntimeError("Could not load any usable DPO samples from the Hugging Face dataset.")

    dataset = Dataset.from_list(samples)
    dataset = dataset.shuffle(seed=args.seed)
    return dataset


def split_train_val_dataset(dataset: Dataset, args: argparse.Namespace) -> Tuple[Dataset, Optional[Dataset]]:
    if args.val_ratio <= 0 or len(dataset) < 2:
        return dataset, None

    split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_dataset = split["train"].shuffle(seed=args.seed)
    val_dataset = split["test"]
    if len(val_dataset) == 0:
        val_dataset = None
    return train_dataset, val_dataset


def print_samples(dataset: Dataset, num_samples: int) -> None:
    sample_count = min(num_samples, len(dataset))
    for index in range(sample_count):
        sample = dataset[index]
        print(
            f"===== Sample {index + 1} | {sample['source']} | {sample['id']} ====="
        )
        print(sample["prompt"])
        print("========== chosen ==========")
        print(sample["chosen"])
        print("========== rejected ==========")
        print(sample["rejected"])
        print()


def train_dpo(args: argparse.Namespace, dataset: Dataset, tokenizer) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from trl import DPOConfig, DPOTrainer

    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for DPO training.")

    configure_cuda_device(torch)
    train_dataset, val_dataset = split_train_val_dataset(dataset, args)
    if not hasattr(args, "max_seq_len"):
        args.max_seq_len = args.max_length
    wandb_module = setup_wandb(args, len(train_dataset), len(val_dataset) if val_dataset is not None else 0)

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"torch_dtype": torch_dtype}
    ref_model_kwargs = {"torch_dtype": torch_dtype}
    if torch.cuda.is_available() and not is_distributed_training():
        model_kwargs["device_map"] = "auto"
        ref_model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    base_model.config.use_cache = False

    if args.adapter_path:
        model = PeftModel.from_pretrained(
            base_model,
            resolve_path(Path(__file__).resolve().parent, args.adapter_path),
            is_trainable=True,
        )
        model.enable_input_require_grads()
        if is_main_process():
            model.print_trainable_parameters()
    else:
        raise RuntimeError("--adapter_path is required for DPO fine-tuning in this script.")

    ref_base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **ref_model_kwargs)
    ref_base_model.config.use_cache = False
    ref_adapter_path = args.ref_adapter_path or args.adapter_path
    ref_model = PeftModel.from_pretrained(
        ref_base_model,
        resolve_path(Path(__file__).resolve().parent, ref_adapter_path),
        is_trainable=False,
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
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

    print(f"Loaded {len(dataset)} DPO samples after filtering.")
    print(f"Distributed training: {is_distributed_training()} (WORLD_SIZE={get_world_size()})")
    print(f"RANK={get_rank()}")
    print(f"LOCAL_RANK={get_local_rank()}")
    print_samples(dataset, num_samples=args.print_samples)

    if args.train:
        train_dpo(args, dataset, tokenizer)


if __name__ == "__main__":
    main()