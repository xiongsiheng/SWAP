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
    parser.add_argument('--prob_type', default='math')  # 'math', 'logical reasoning', 'coding'
    parser.add_argument('--group_size', type=int, default=3)  # Group size for single-time comparison (recommend: 2 or 3)
    parser.add_argument('--train', action='store_true')  # Whether to train the model
    parser.add_argument('--use_graph', action='store_true')  # Use graph for reasoning
    parser.add_argument('--use_meta_knowledge', action='store_true')  # Use meta-knowledge for discriminator
    parser.add_argument('--print_example', action='store_true')  # Print example prompts
    parser.add_argument('--use_wandb', action='store_true')  # Use wandb for tracking
    parser.add_argument('--output_dir', default='../model_weights')  # Model output directory
    parser.add_argument('--resume_from', default=None)  # Resume from checkpoint
    parser.add_argument('--batch_size', type=int, default=1)  # Training batch size
    parser.add_argument('--max_seq_len', type=int, default=2048)  # Maximum sequence length
    parser.add_argument('--finetune_method', type=str, choices=['lora', 'full'], default='lora')  # Finetuning method: LoRA or full finetuning

    return parser.parse_args()


def setup_wandb(args, path_name):
    """
    Sets up Weights and Biases (wandb) for tracking training and evaluation.
    Args:
        args (argparse.Namespace): Command-line arguments.
        path_name (str): Identifier for the model architecture.
    """
    if args.use_wandb:
        wandb.init(
            project=f"{args.dataset}_{args.subset}_Dis_{path_name}",
            config={
                "learning_rate": 5e-4,
                "lora": 'r16-alpha32',
                "architecture": path_name,
                "dataset": f"{args.dataset}_{args.subset}",
            }
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"


def filter_dataset(dataset, args):
    """
    Filters the dataset based on the subset and group size specified in the arguments.
    Args:
        dataset (DatasetDict): The dataset loaded from HuggingFace.
        args (argparse.Namespace): Command-line arguments.
    Returns:
        Tuple[Dataset, Dataset]: The filtered training and validation datasets.
    """
    dataset_filtered = []
    for sample in dataset['train']:
        if 'subset' in sample and args.subset != 'all' and args.subset != sample['subset']:
            continue
        input_dict = eval(sample['input'])
        responses = input_dict["Search steps"]
        if len(responses) > args.group_size:
            continue
        dataset_filtered.append(sample)

    train_size = int(0.8 * len(dataset_filtered))
    return (
        Dataset.from_list(dataset_filtered[:train_size]),
        Dataset.from_list(dataset_filtered[train_size:])
    )


def print_examples(dataset, args):
    """
    Prints example prompts from the dataset for verification purposes.
    Args:
        dataset (Dataset): The dataset to sample examples from.
        args (argparse.Namespace): Command-line arguments.
    """
    for i in range(5):
        sample = dataset[i]
        prompt = generate_prompt_for_discriminator(
            args.prob_type, 
            sample['input'], 
            sample['output'], 
            eos_token="<|eot_id|>", 
            use_graph=args.use_graph, 
            use_meta_knwoledge=args.use_meta_knowledge
        )
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


def SFT_with_LoRA(model, tokenizer, output_dir, formatting_func, data_train, data_val, batch_size, max_seq_length, resume_from_checkpoint=None, collator=None):
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
        bf16=True,
        num_train_epochs=10,
        resume_from_checkpoint=resume_from_checkpoint
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


def SFT_with_full_finetuning(model, tokenizer, output_dir, formatting_func, data_train, data_val, batch_size, max_seq_length, resume_from_checkpoint=None, collator=None):
    """
    Trains a model with full finetuning for Supervised Fine-Tuning (SFT).
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
        bf16=True,
        num_train_epochs=10,
        resume_from_checkpoint=resume_from_checkpoint
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data_train,
        eval_dataset=data_val,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator
    )

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

    model_selection = 0
    model_name = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"][model_selection]
    path_name = ["llama3_8B", "mistral_7B"][model_selection]
    output_dir = f"{args.output_dir}/{args.dataset}_{args.subset}_Dis_{path_name}"

    setup_wandb(args, path_name)

    dataset = load_dataset("sxiong/SWAP", f"{args.dataset}_contrastive_ranking")
    dataset_train, dataset_val = filter_dataset(dataset, args)

    if args.print_example:
        print_examples(dataset_train, args)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Train the model
    if args.train:
        def formatting_func(sample):
            output = []
            for x, y in zip(sample['input'], sample['output']):
                op = generate_prompt_for_discriminator(
                    args.prob_type, x, y, eos_token="<|eot_id|>", 
                    use_graph=args.use_graph, 
                    use_meta_knwoledge=args.use_meta_knowledge
                )
                output.append(op)
            return output
        
        response_template = " ### Output:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

        if args.finetune_method == "lora":
            # Apply LoRA for SFT
            SFT_with_LoRA(
                model, tokenizer, output_dir, formatting_func, 
                dataset_train, dataset_val, args.batch_size, 
                args.max_seq_len, args.resume_from, collator
            )
        else:
            # Apply full finetuning without PEFT modifications
            SFT_with_full_finetuning(
                model, tokenizer, output_dir, formatting_func, 
                dataset_train, dataset_val, args.batch_size, 
                args.max_seq_len, args.resume_from, collator
            )


if __name__ == "__main__":
    main()