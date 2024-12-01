import sys
import os
import argparse

import torch
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
from datasets import load_dataset, Dataset

from utils import *
from prompt_generation import *




def parse_args():
    """
    Parses command-line arguments provided to the script.
    Returns:
        argparse.Namespace: An object containing all the arguments and their values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MATH')
    parser.add_argument('--subset', default='algebra')
    parser.add_argument('--train', action='store_true')  # Whether to train the model
    parser.add_argument('--print_example', action='store_true')  # Whether to print examples
    parser.add_argument('--use_wandb', action='store_true')  # Whether to use wandb
    parser.add_argument('--output_dir', default='../model_weights')  # Output directory for model
    parser.add_argument('--resume_from', default=None)  # Path to resume checkpoint
    parser.add_argument('--batch_size', type=int, default=6)  # Batch size for training
    parser.add_argument('--max_seq_len', type=int, default=512)  # Maximum sequence length
    return parser.parse_args()


def setup_wandb(args, model_name, path_name):
    """
    Sets up Weights and Biases (wandb) for tracking training and evaluation.
    Args:
        args (argparse.Namespace): Command-line arguments.
        path_name (str): Identifier for the model architecture.
    """
    if args.use_wandb:
        wandb.init(
            project=f"{args.dataset}_{args.subset}_sem_equ_{path_name}",
            config={
                "learning_rate": 5e-4,
                "lora": 'r16-alpha32',
                "architecture": model_name,
                "dataset": f"{args.dataset}_{args.subset}",
            }
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"


def filter_dataset(dataset, subset):
    """
    Filters the dataset based on the subset and group size specified in the arguments.
    Args:
        dataset (DatasetDict): The dataset loaded from HuggingFace.
        args (argparse.Namespace): Command-line arguments.
    Returns:
        Tuple[Dataset, Dataset]: The filtered training and validation datasets.
    """
    filtered = [
        sample for sample in dataset['train']
        if 'subset' in sample and (subset == 'all' or subset == sample['subset'])
    ]
    train_size = int(0.8 * len(filtered))
    train_data = Dataset.from_dict(obtain_sem_equ_data(filtered[:train_size]))
    val_data = Dataset.from_dict(obtain_sem_equ_data(filtered[train_size:]))
    return train_data, val_data


def print_examples(dataset):
    """
    Prints example prompts from the dataset for verification purposes.
    Args:
        dataset (Dataset): The dataset to sample examples from.
    """
    for i in range(5):
        sample = dataset[i]
        prompt = generate_prompt_for_sem_equ_lora_train(sample['instruction'], sample['input'], sample['output'], eos_token="<|eot_id|>")
        print(prompt)
        print('===============================')


def load_model_and_tokenizer(model_name):
    """
    Loads the model and tokenizer from HuggingFace.
    Args:
        model_name (str): Name of the model to load.
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: The model and tokenizer objects.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def SFT_with_LoRA(model, tokenizer, output_dir, formatting_func, data_train, data_val, batch_size, max_seq_length,
                  resume_from_checkpoint=None, collator=None):
    """
    Trains a model with LoRA (Low-Rank Adaptation) for Supervised Fine-Tuning (SFT).
    Args:
        model (PreTrainedModel): The base model to be fine-tuned.
        tokenizer (PreTrainedTokenizer): The tokenizer for the model.
        output_dir (str): Directory to save the trained model.
        formatting_func (Callable): Function to format the input data.
        data_train (Dataset): Training dataset.
        data_val (Dataset): Validation dataset.
        batch_size (int): Batch size for training.
        max_seq_length (int): Maximum sequence length for input.
        resume_from_checkpoint (str, optional): Path to resume training from.
        collator (DataCollator, optional): Data collator for tokenization.
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        save_steps=50,
        learning_rate=5e-4,
        logging_steps=10,
        max_grad_norm=0.3,
        evaluation_strategy="epoch",
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=16,
        per_device_eval_batch_size=batch_size,
        resume_from_checkpoint=resume_from_checkpoint,
        bf16=True,
        num_train_epochs=10
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data_train,
        eval_dataset=data_val,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(f"{output_dir}/final")


def main():
    """
    Main function to run the script. Handles argument parsing, dataset preparation, 
    model initialization, and training execution.
    """
    args = parse_args()

    # Model and path selection
    model_selection = 0
    model_name = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"][model_selection]
    path_name = ["llama3_8B", "mistral_7B"][model_selection]

    output_dir = f"{args.output_dir}/{args.dataset}_{args.subset}_sem_equ_{path_name}"

    setup_wandb(args, model_name, path_name)

    # Load and filter dataset
    dataset = load_dataset("sxiong/SWAP", f"{args.dataset}_semantic_equivalence")
    dataset_train, dataset_val = filter_dataset(dataset, args.subset)

    # Print examples if required
    if args.print_example:
        print_examples(dataset_train)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Train the model
    if args.train:
        def formatting_func(sample):
            output = []
            for instruction, input_text, output_text in zip(sample['instruction'], sample['input'], sample['output']):
                prompt = generate_prompt_for_sem_equ_lora_train(instruction, input_text, output_text, eos_token="<|eot_id|>")
                output.append(prompt)
            return output
        
        response_template = " ### Output:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
        
        SFT_with_LoRA(
            model, tokenizer, output_dir, formatting_func, dataset_train, dataset_val,
            args.batch_size, args.max_seq_len, args.resume_from, collator
        )


if __name__ == "__main__":
    main()