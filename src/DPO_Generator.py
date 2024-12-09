import sys
import os
import argparse


from datasets import load_dataset
from trl import DPOConfig, DPOTrainer, PreferenceCollator
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import torch
from peft import PeftModel

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
    parser.add_argument('--prob_type', default='math') # 'math', 'logical reasoning', 'coding'

    parser.add_argument('--train', action='store_true')   # whether to train the model
    parser.add_argument('--use_graph', action='store_true')  # whether to use the graph for reasoning

    parser.add_argument('--print_example', action='store_true') # whether to print the example
    parser.add_argument('--use_wandb', action='store_true') # whether to use wandb

    parser.add_argument('--output_dir', default='../model_weights') # the output directory for the model

    parser.add_argument('--resume_frome', default=None)  # the path to the resume checkpoint
    parser.add_argument('--batch_size', type=int, default=5)  # the batch size for training
    parser.add_argument('--max_seq_len', type=int, default=1024)  # the maximum sequence length for the model

    return parser.parse_args()


def setup_wandb(args, path_name):
    """
    Sets up Weights and Biases (wandb) for tracking training and evaluation.
    Args:
        args (argparse.Namespace): Command-line arguments.
        path_name (str): Identifier for the model architecture.
    """
    if args.use_wandb:
        # start a new wandb run to track this script
        wandb.init(
            project=f"{args.dataset}_{args.subset}_Gen_{path_name}_DPO",
            # track hyperparameters and run metadata
            config={
            "architecture": path_name,
            "dataset": f"{args.dataset}_{args.subset}",
            }
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"


def filter_dataset(dataset, args):
    """
    Filters the dataset based on the subset specified in the arguments.
    Args:
        dataset (DatasetDict): The dataset loaded from HuggingFace.
        args (argparse.Namespace): Command-line arguments.
    Returns:
        Tuple[Dataset, Dataset]: The filtered training and validation datasets.
    """
    dataset_filtered = []
    for sample in dataset['train']:
        if 'subset' in sample and args.subset != 'all':
            if args.subset != sample['subset']:
                continue
        dataset_filtered.append(sample)

    dataset_train = Dataset.from_list(dataset_filtered[:int(0.8*len(dataset_filtered))])
    dataset_val = Dataset.from_list(dataset_filtered[int(0.8*len(dataset_filtered)):])

    return dataset_train, dataset_val


def print_examples(dataset, args):
    """
    Prints example prompts from the dataset for verification purposes.
    Args:
        dataset (Dataset): The dataset to sample examples from.
        args (argparse.Namespace): Command-line arguments.
    """
    for i in range(5):
        sample = dataset[i]
        prompt = generate_prompt_for_generator_DPO(
                    sample['prompt'], 
                    sample['chosen'], 
                    sample['rejected'], 
                    use_graph=args.use_graph
                )
        print(prompt)
        print('===============================')


def load_model_and_tokenizer(model_name, output_dir_sft):
    """
    Loads the model and tokenizer from HuggingFace.
    Args:
        model_name (str): Name of the model to load.
        output_dir_sft (str): Directory to save the SFT model.
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: The model and tokenizer objects.
    """
    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    
    # Load the base model.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))

    # Load the adapter.
    model = PeftModel.from_pretrained(
        model,
        output_dir_sft,
        is_trainable=True,
        adapter_name="train",
    )
    # Load the adapter a second time, with a different name, which will be our reference model.
    model.load_adapter(output_dir_sft, adapter_name="reference")

    return model, tokenizer


def DPO_with_LoRA(model, tokenizer, output_dir, data_train, data_val, batch_size, max_seq_length, resume_from_checkpoint=None, collator=None):
    """
    Trains a model with LoRA (Low-Rank Adaptation) for Directed Preference Optimization (DPO).
        Args:
        model (PreTrainedModel): The base model to be fine-tuned.
        tokenizer (PreTrainedTokenizer): The tokenizer for the model.
        output_dir (str): Directory to save the trained model.
        data_train (Dataset): Training dataset.
        data_val (Dataset): Validation dataset.
        batch_size (int): Batch size for training.
        max_seq_length (int): Maximum sequence length for input.
        resume_from_checkpoint (str, optional): Path to resume training from.
        collator (Callable, optional): Function to format the input data.
    """

    # Initialize the trainer, without a ref_model param.
    training_args = DPOConfig(
        model_adapter_name="train",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=16,
        ref_adapter_name="reference",
        resume_from_checkpoint=resume_from_checkpoint,
        logging_steps=10,
        num_train_epochs=3,
        max_length=max_seq_length
    )

    dpo_trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=data_train, eval_dataset=data_val, 
                             data_collator=collator)
    dpo_trainer.train()
    dpo_trainer.save_model(f"{output_dir}/final")


def main():
    """
    Main function to run the script. Handles argument parsing, dataset preparation, 
    model initialization, and training execution.
    """
    args = parse_args()

    # You can add more base models here
    model_selection = 0
    model_name = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"][model_selection]
    path_name = ["llama3_8B", "mistral_7B"][model_selection]
    output_dir_dpo = f"{args.output_dir}/{args.dataset}_{args.subset}_Gen_{path_name}_DPO"
    output_dir_sft = f"{args.output_dir}/{args.dataset}_{args.subset}_Gen_{path_name}"

    setup_wandb(args, path_name)

    dataset = load_dataset("sxiong/SWAP", f"{args.dataset}_trajectory_DPO")
    print(dataset)
    
    dataset_train, dataset_val = filter_dataset(dataset, args)

    if args.print_example:
        print_examples(dataset_train, args)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, output_dir_sft)
    
    # Train the model
    if args.train:
        collator = PreferenceCollator(pad_token_id=len(tokenizer)-1)
        DPO_with_LoRA(model, tokenizer, output_dir_dpo, dataset_train, dataset_val, args.batch_size, args.max_seq_len, args.resume_frome, collator)



if __name__ == "__main__":
    main()