import sys
import os
import argparse


import torch
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import (
        get_peft_model, 
        LoraConfig
    )
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb

from datasets import load_dataset, Dataset

from utlis import *
from prompt_generation import *







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



args = parser.parse_args()





f_print_example = args.print_example
f_train = args.train
f_use_wandb = args.use_wandb
resume_from_checkpoint = args.resume_frome


# You can add more base models here
model_selection = 0
model_name = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"][model_selection]
path_name = ["llama3_8B", "mistral_7B"][model_selection]

output_dir = f"{args.output_dir}/{args.dataset}_{args.subset}_Gen_{path_name}"


if f_use_wandb:
    # start a new wandb run to track this script
    wandb.init(
        project=f"{args.dataset}_{args.subset}_Gen_{path_name}",
        # track hyperparameters and run metadata
        config={
        "learning_rate": 5e-4,
        "lora": 'r16-alpha32',
        "architecture": model_name,
        "dataset": f"{args.dataset}_{args.subset}",
        }
    )
else:
    os.environ["WANDB_DISABLED"] = "true"




## Load the dataset
dataset = load_dataset("sxiong/SWAP", f"{args.dataset}_trajectory")
print(dataset)

dataset_filtered = []
for sample in dataset['train']:
    if 'subset' in sample and args.subset != 'all':
        if args.subset != sample['subset']:
            continue
    dataset_filtered.append(sample)

dataset_train = Dataset.from_list(dataset_filtered[:int(0.8*len(dataset_filtered))])
dataset_val = Dataset.from_list(dataset_filtered[int(0.8*len(dataset_filtered)):])

print(dataset_train)
print(dataset_val)





# print the example prompt for the generator
if f_print_example:
    for i in range(5):
        sample = dataset_train[i]
        prompt = generate_prompt_for_generator(args.prob_type, sample['question'], sample['trajectory'], eos_token="<|eot_id|>", use_graph=args.use_graph)
        
        print(prompt)
        print('===============================')





# load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# this should be set for finutning and batched inference
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model.resize_token_embeddings(len(tokenizer))



def SFT_with_LoRA(model, tokenizer, output_dir, formatting_func, data_train, data_val, batch_size, max_seq_length, resume_from_checkpoint=None,
                  collator=None):
    '''
    Perform the SFT with LoRA training.
    '''
    # lora config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    per_device_train_batch_size = batch_size
    gradient_accumulation_steps = 16
    per_device_eval_batch_size = batch_size
    eval_accumulation_steps = 16
    optim = "paged_adamw_32bit"
    save_steps = 50
    logging_steps = 10
    learning_rate = 5e-4
    max_grad_norm = 0.3
    warmup_ratio = 0.03
    evaluation_strategy="epoch"
    lr_scheduler_type = "cosine"

    training_args = transformers.TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                optim=optim,
                save_steps=save_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                max_grad_norm=max_grad_norm,
                evaluation_strategy=evaluation_strategy,
                warmup_ratio=warmup_ratio,
                group_by_length=True,
                lr_scheduler_type=lr_scheduler_type,
                ddp_find_unused_parameters=False,
                eval_accumulation_steps=eval_accumulation_steps,
                per_device_eval_batch_size=per_device_eval_batch_size,
                resume_from_checkpoint=resume_from_checkpoint,  # Resume from checkpoint if provided
                bf16=True,
                num_train_epochs=10
            )

    # SFT with lora
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

    # We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
    
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_model(f"{output_dir}/final")
    return



# Perform the training
if f_train:
    def formatting_func(sample):
        '''
        Given the sample, generate the prompt for the model.
        '''
        output = []
        for x, y in zip(sample['question'], sample['trajectory']):
            op = generate_prompt_for_generator(args.prob_type, x, y, eos_token="<|eot_id|>", use_graph=args.use_graph)
            output.append(op)
        return output
        
    response_template = " ### Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    SFT_with_LoRA(model, tokenizer, output_dir, formatting_func, dataset_train, dataset_val, args.batch_size, args.max_seq_len, resume_from_checkpoint, collator)