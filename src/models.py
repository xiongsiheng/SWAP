import sys
import json
import os
import random
import itertools
from math import ceil
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel

from utils import *
from prompt_generation import *
from grading.grader import grade_answer






def distributed_inference(model, tokenizer, accelerator, prompts, batch_size, max_new_tokens, top_k, top_p, temperature):
    '''
    Function to perform distributed inference on a list of prompts using the given model.
    
    Args:
        model (torch.nn.Module): The model to perform inference with.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenizing the prompts.
        accelerator (accelerate.Accelerator): The accelerator object to use for distributed inference.
        prompts (List[str]): The list of prompts to generate completions for.
        batch_size (int): The batch size to use for inference.
        max_new_tokens (int): The maximum number of tokens to generate for each prompt.
        top_k (int): The number of top-k tokens to sample from the probability distribution.
        top_p (float): The nucleus sampling threshold.
        temperature (float): The temperature to use for sampling from the probability distribution.
    
    Returns:
        List[str]: The list of generated completions for the prompts.
    '''
    def prepare_prompts(prompts, tokenizer, batch_size):
        batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        batches_tok=[]
        for prompt_batch in batches:
            batches_tok.append(
                tokenizer(
                    prompt_batch, 
                    return_tensors="pt", 
                    padding='longest', 
                    truncation=False, 
                    pad_to_multiple_of=8,
                    add_special_tokens=False).to("cuda") 
                )
        return batches_tok

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    

    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts) as prompts_process:
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches = prepare_prompts(prompts_process, tokenizer, batch_size)

        for prompts_tokenized in prompt_batches:
            outputs_tokenized = model.generate(**prompts_tokenized, max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p, temperature=temperature)

            # remove prompt from gen.tokens
            outputs_tokenized = [tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)] 

            # count and decode gen.tokens
            num_tokens = sum([len(t) for t in outputs_tokenized])
            outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)

            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens

        results = [results] # transform to list, otherwise gather_object() will not collect correctly

    # collect results from all the GPUs
    results_gathered = gather_object(results)

    outputs = []
    for r in results_gathered:
        outputs += r["outputs"]
    
    return outputs




class Generator:
    def __init__(self, gen_model_id, sem_model_id, model_name, enable_DBM=True, show_prompt_only=False, prob_type=None):
        '''
        Initialize the Generator object.

        Args:
            gen_model_id (str): The ID of the generator model to use.
            sem_model_id (str): The ID of the semantic equivalent model to use.
            model_name (str): The name of the model to use.
            enable_DBM (bool): Whether to enable diversity-based modelling.
            show_prompt_only (bool): Whether to only show the prompts without generating completions.
            prob_type (str): The type of the problem.
        
        Returns:
            None
        '''
        assert prob_type in ['math', 'logical reasoning', 'coding'], "Invalid problem type."   # We mainly test these types of problems. You can adapt to other types.
        self.enable_DBM = enable_DBM
        self.show_prompt_only = show_prompt_only  # For debugging purposes
        self.prob_type = prob_type
        if not show_prompt_only:
            self._build_model(gen_model_id, sem_model_id, model_name)


    def _build_model(self, gen_model_id, sem_model_id, model_name):
        '''
        Build the generator model.

        Args:
            gen_model_id (str): The ID of the generator model to use.
            sem_model_id (str): The ID of the semantic equivalent model to use.
            model_name (str): The name of the model to use.

        Returns:
            None
        '''
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    device_map="auto",
                                                    torch_dtype=torch.float16
                                                    )

        if self.enable_DBM:
            model_sem = AutoModelForCausalLM.from_pretrained(model_name,
                                                            device_map="auto",
                                                            torch_dtype=torch.float16
                                                            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # this should be set for finutning and batched inference
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.resize_token_embeddings(len(self.tokenizer))
        if self.enable_DBM:
            model_sem.resize_token_embeddings(len(self.tokenizer))

        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        if self.enable_DBM:
            model_sem.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.peft_model = PeftModel.from_pretrained(model, gen_model_id, torch_dtype=torch.float32, offload_folder="lora_results/temp")
        self.peft_model.eval()

        self.accelerator = Accelerator()

        if self.enable_DBM:
            self.peft_model_sem = PeftModel.from_pretrained(model_sem, sem_model_id, torch_dtype=torch.float32, offload_folder="lora_results/temp")
            self.peft_model_sem.eval()
        else:
            self.peft_model_sem = None


    def _to_binary_distribution(self, probabilities):
        '''
        Function to convert a probability distribution to a binary distribution.

        Args:
            probabilities (torch.Tensor): The probability distribution tensor.

        Returns:
            binary_distribution (torch.Tensor): The binary distribution tensor.
        '''
        # Get the index of the maximum probability for each batch
        max_indices = torch.argmax(probabilities, dim=-1)
        
        # Create a binary distribution tensor with the same shape as `probabilities`
        binary_distribution = torch.zeros_like(probabilities)
        
        # Set the maximum index in each batch to 1
        binary_distribution[torch.arange(probabilities.size(0)), max_indices] = 1.0
        
        return binary_distribution


    def _get_last_token_prob_distribution(self, model, input_ids, attention_mask, past_key_values=None, temperature=1.0):
        '''
        Function to get the probability distribution of the last token in the generated sequence.

        Args:
            model (torch.nn.Module): The model to use for generation.
            input_ids (torch.Tensor): The input token IDs tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
            past_key_values (Tuple[torch.Tensor], optional): The past key values tensor. Defaults to None.
            temperature (float, optional): The temperature to use for scaling the logits. Defaults to 1.0.

        Returns:
            probabilities (torch.Tensor): The probability distribution tensor.
            past_key_values (Tuple[torch.Tensor]): The past key values tensor.
        '''
        with torch.no_grad():
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]
                attention_mask = attention_mask[:, -1:]

            outputs = model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            last_token_logits = logits[:, -1, :]

            # Scale logits by temperature before softmax
            if temperature > 0:
                scaled_logits = last_token_logits / temperature
            else:
                scaled_logits = last_token_logits
            probabilities = torch.softmax(scaled_logits, dim=-1)
            
            if temperature == 0:
                probabilities = self._to_binary_distribution(probabilities)

        return probabilities, outputs.past_key_values

    
    def _sample_from_probabilities(self, scaled_probabilities, top_k=10, top_p=0.9):
        '''
        Function to sample from the averaged probability distributions.

        Args:
            scaled_probabilities (torch.Tensor): The scaled probability distributions tensor.
            top_k (int, optional): The number of top-k tokens to sample from the probability distribution. Defaults to 10.
            top_p (float, optional): The nucleus sampling threshold. Defaults to 0.9.

        Returns:
            next_token (torch.Tensor): The sampled token tensor.
        '''
        # # Apply temperature scaling
        # scaled_probabilities = probabilities / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(scaled_probabilities, top_k)
            top_k_mask = torch.full_like(scaled_probabilities, float('-inf'))
            top_k_mask.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
            scaled_probabilities = torch.where(scaled_probabilities < top_k_mask, float('-inf'), scaled_probabilities)

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(scaled_probabilities, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            sorted_probs[sorted_indices_to_remove] = float('-inf')
            scaled_probabilities.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

        # Sample from the filtered distribution
        next_token = torch.multinomial(torch.softmax(scaled_probabilities, dim=-1), num_samples=1)
        return next_token


    def _geometric_sequence(self, start, ratio, num_terms, offset):
        '''
        Function to generate a geometric sequence.

        Args:
        start: starting point, ratio: common ratio, num_terms: number of terms, offset: offset

        Returns:
        sequence: The generated geometric sequence.
        '''
        sequence = []
        term = start
        for _ in range(num_terms):
            sequence.append(term + offset)
            term *= ratio
        return sequence


    def _generate_with_diversity_based_sampling(self, peft_model1, peft_model2, input_ids1, input_ids2, attention_mask1, attention_mask2, 
                                                max_length, top_k, top_p, temperature, stop_count):
        '''
        Function to generate tokens iteratively with diversity based probabilities.

        Args:
            peft_model1: The model to use for the first probability distribution.
            peft_model2: The model to use for the second probability distribution.
            input_ids1: The input token IDs tensor for the first model.
            input_ids2: The input token IDs tensor for the second model.
            attention_mask1: The attention mask tensor for the first model.
            attention_mask2: The attention mask tensor for the second model.
            max_length: The maximum length of the generated sequence.
            top_k: The number of top-k tokens to sample from the probability distribution.
            top_p: The nucleus sampling threshold.
            temperature: The temperature to use for scaling the logits.
            stop_count: The number of times the stop token should be generated before stopping.

        Returns:
            input_ids1: The generated token IDs tensor.
        '''
        # decay factor for the second probability distribution.
        gamma = self._geometric_sequence(start=0.7, ratio=0.6, num_terms=max_length, offset=0)

        batch_size = input_ids1.shape[0]
        eos_token_id = self.tokenizer.eos_token_id  # Assuming tokenizer is defined globally

        # Flag to track finished sequences and counters for token `10246` occurrences
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool).to(input_ids1.device)
        stop_token_count = torch.zeros(batch_size, dtype=torch.int).to(input_ids1.device)

        past_key_values1 = None
        past_key_values2 = None
        for t in range(max_length):
            # Get probability distributions P1 and P2
            P1, past_key_values1 = self._get_last_token_prob_distribution(peft_model1, input_ids1, attention_mask1, past_key_values=past_key_values1, 
                                                                    temperature=temperature)
            if input_ids2 is not None:
                P2, past_key_values2 = self._get_last_token_prob_distribution(peft_model2, input_ids2, attention_mask2, past_key_values=past_key_values2,
                                                                        temperature=0.3)
                combined_prob = P1 - gamma[t] * P2
                combined_prob = torch.clamp(combined_prob, min=0)
                combined_prob /= combined_prob.sum(dim=-1, keepdim=True)
            else:
                combined_prob = P1

            # Sample the next token from the combined probability distribution
            next_token = self._sample_from_probabilities(combined_prob, top_k=top_k, top_p=top_p)

            # Update stop token count for token `10246`
            stop_token_count += (next_token.squeeze(-1) == 10246).int()
            
            # print(stop_token_count)

            # Replace next_token with eos_token_id for finished sequences
            next_token[finished_sequences] = eos_token_id

            # Mark sequences as finished if they hit the stop count or reach the eos token
            finished_sequences = finished_sequences | (next_token.squeeze(-1) == eos_token_id) | (stop_token_count >= stop_count)
            
            # Append the new token to the input_ids and update attention_mask
            input_ids1 = torch.cat([input_ids1, next_token], dim=-1)
            attention_mask1 = torch.cat([attention_mask1, torch.ones((batch_size, 1), dtype=attention_mask1.dtype).to(input_ids1.device)], dim=-1)

            if input_ids2 is not None:
                input_ids2 = torch.cat([input_ids2, next_token], dim=-1)
                attention_mask2 = torch.cat([attention_mask2, torch.ones((batch_size, 1), dtype=attention_mask2.dtype).to(input_ids2.device)], dim=-1)

            # If all sequences are finished, break
            if finished_sequences.all():
                break

        return input_ids1


    def _decode_new_steps(self, generation_output, input_tokens):
        '''
        Decode the generated tokens.

        Args:
            generation_output: The generated token IDs tensor.
            input_tokens: The input token IDs tensor.

        Returns:
            outputs: The list of decoded outputs.
        '''
        responses = self.tokenizer.batch_decode(generation_output[:, input_tokens.shape[-1]:], skip_special_tokens=True)
        outputs = [response.strip() for response in responses]
        outputs = [convert_escape_sequences(replace_all_escape_sequences(output)) for output in outputs]
        return outputs


    def _run_one_batch(self, samples, num_future_steps, force_termination, output_dir, rollout_id, visualize):
        '''
        Run one batch of samples through the generator model.
        
        Args:
            samples (List[Dict]): The list of samples to process.
            num_future_steps (int): The number of future steps to generate.
            force_termination (bool): Whether to force termination.
            output_dir (str): The output directory to save the results.
            rollout_id (str): The ID of the rollout.
            visualize (bool): Whether to visualize the results.

        Returns:
            None
        '''
        prompts = [sample['rollout'][rollout_id]['prompt'] for sample in samples]
        
        # Determine the tags (necessary for the semantic equivalent model)
        if force_termination:
            tags = ['"Final answer":' for _ in range(len(samples))]
        else:
            responses = distributed_inference(self.peft_model, self.tokenizer, self.accelerator, prompts, len(samples), 
                                              max_new_tokens=7, top_k=10, top_p=0.9, temperature=0.3)
            outputs = [response.strip() for response in responses]
            tags = [output[:len(output.split(':')[0])+1].strip() for output in outputs]
        
        prompts_with_tag = [f'{prompt}{tag} "' for prompt, tag in zip(prompts, tags)]

        tokenized_inputs = self.tokenizer(prompts_with_tag, return_tensors="pt", padding=True)
        input_tokens = tokenized_inputs["input_ids"].to("cuda")
        attention_mask = tokenized_inputs['attention_mask'].to("cuda")
        prev_preds = [sample['rollout'][rollout_id]['responses'] for sample in samples]


        # semantic equivalent model
        input_tokens_sem = None
        attention_mask_sem = None
        if (self.peft_model_sem is not None) and (len(prev_preds[0]) > 0):
            prompts_sem = generate_prompt_for_sem_equ_lora_inference(prev_preds)
            # print('Prompts for semantic equivalent model:')
            # for prompt in prompts_sem:
            #     print(prompt)
            tokenized_inputs_sem = self.tokenizer(prompts_sem, return_tensors="pt", padding=True)
            input_tokens_sem = tokenized_inputs_sem["input_ids"].to("cuda")
            attention_mask_sem = tokenized_inputs_sem['attention_mask'].to("cuda")
        
        with torch.cuda.amp.autocast():
            generation_output = self._generate_with_diversity_based_sampling(self.peft_model, self.peft_model_sem, input_tokens, input_tokens_sem, 
                                                                             attention_mask, attention_mask_sem, max_length=128, top_k=50, top_p=0.9, 
                                                                             temperature=0.7, stop_count=1)

        results = self._decode_new_steps(generation_output, input_tokens)
        results = [f'{tag} "{res}' for tag, res in zip(tags, results)]

        futures = [[] for _ in range(len(samples))]
        if num_future_steps > 0:
            prompts_new = [f"{prompt}{res.strip()}\n" for (prompt, res) in zip(prompts, results)]
            tokenized_inputs = self.tokenizer(prompts_new, return_tensors="pt", padding=True)
            input_tokens = tokenized_inputs["input_ids"].to("cuda")
            attention_mask = tokenized_inputs['attention_mask'].to("cuda")

            generation_output = self._generate_with_diversity_based_sampling(self.peft_model, None, input_tokens, None, attention_mask, 
                                                                None, max_length=896, top_k=50, top_p=0.9, temperature=0.3, 
                                                                stop_count=num_future_steps)
            futures = self._decode_new_steps(generation_output, input_tokens)
        
        if visualize:
            print('Generation results:')
        for idx_sample in range(len(prompts)):
            if visualize:
                print('Prompt:')
                print(prompts[idx_sample])
                print('-------------------')
                print('Result:')
                print(results[idx_sample])
                print('-------------------')
                print('Future:')
                print(futures[idx_sample])
                print('-------------------')
            

            op_dict = samples[idx_sample]

            op_dict['rollout'][rollout_id]['responses'].append(results[idx_sample])
            op_dict['rollout'][rollout_id]['futures'].append(futures[idx_sample])
                            
            with open(f'{output_dir}/{samples[idx_sample]["id"]}.json', 'w') as f:
                json.dump(op_dict, f)
        
        if visualize:
            print('===================')


    def inference(self, dataset, output_dir, rollout_id, batch_size, num_generations, num_future_steps, force_termination=False, visualize=False):
        '''
        Perform inference on the given dataset.

        Args:
            dataset (List[Dict]): The dataset to perform inference on.
            output_dir (str): The output directory to save the results.
            rollout_id (str): The ID of the rollout.
            batch_size (int): The batch size to use for inference.
            num_generations (int): The number of generations to perform.
            num_future_steps (int): The number of future steps to generate.
            force_termination (bool): Whether to force termination.
            visualize (bool): Whether to visualize the results.

        Returns:
            flag_finish (bool): Whether the inference is finished.
        '''
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        instruction = f'Given a {self.prob_type} problem, solve it with the reinforcement learning format.'
        flag_finish = False
        for _ in range(num_generations):
            num_processed_samples = 0
            samples = []
            for sample in tqdm(dataset, total=len(dataset)):
                if os.path.exists(f'{output_dir}/{sample["id"]}.json'):
                    with open(f'{output_dir}/{sample["id"]}.json', 'r') as f:
                        sample = json.load(f)

                if 'rollout' not in sample:
                    sample['rollout'] = {}

                if rollout_id not in sample['rollout']:        
                    sample['rollout'][rollout_id] = {}
                    Input = convert_element_format('Problem', sample['question'], convert_json=True)
                    sample['rollout'][rollout_id]['prompt'] = f'{instruction}\n\n ### Input:\n{Input}\n ### Output: \n'
                    sample['rollout'][rollout_id]['responses'] = []
                    sample['rollout'][rollout_id]['futures'] = []
                    sample['rollout'][rollout_id]['search_tree'] = []
                else:
                    sample['rollout'][rollout_id]['prompt'] = f"{sample['rollout'][rollout_id]['prompt'].strip()}\n"

                if self.show_prompt_only:
                    # For debugging purposes
                    print(sample['rollout'][rollout_id]['prompt'])
                    print('-------------------')
                    continue
                
                if '"Final answer":' in sample['rollout'][rollout_id]['prompt']:
                    continue

                samples.append(sample)
                num_processed_samples += 1
                if len(samples) >= batch_size:
                    self._run_one_batch(samples, num_future_steps, force_termination, output_dir, rollout_id, visualize)
                    samples = []

            if len(samples) > 0:
                self._run_one_batch(samples, num_future_steps, force_termination, output_dir, rollout_id, visualize)
            
            if self.show_prompt_only:
                break
            
            if num_processed_samples == 0:
                flag_finish = True
                break

        return flag_finish




class Discriminator():
    def __init__(self, disc_model_id, model_name, use_meta_knwoledge=False, show_prompt_only=False, prob_type=None):
        '''
        Initialize the Discriminator object.

        Args:
            disc_model_id (str): The ID of the discriminator model to use.
            model_name (str): The name of the model to use.
            use_meta_knwoledge (bool): Whether to use meta-knowledge.
            show_prompt_only (bool): Whether to only show the prompts without generating completions.
            prob_type (str): The type of the problem.

        Returns:
            None
        '''
        assert prob_type in ['math', 'logical reasoning', 'coding'], "Invalid problem type."   # We mainly test these types of problems. You can adapt to other types.
        self.use_meta_knwoledge = use_meta_knwoledge
        self.show_prompt_only = show_prompt_only  # For debugging purposes
        self.prob_type = prob_type
        if not show_prompt_only:
            self._build_model(disc_model_id, model_name)


    def _build_model(self, disc_model_id, model_name):
        '''
        Build the discriminator model.

        Args:
            disc_model_id (str): The ID of the discriminator model to use.
            model_name (str): The name of the model to use.

        Returns:
            None
        '''
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    device_map="auto",
                                                    torch_dtype=torch.float16
                                                    )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # this should be set for finutning and batched inference
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.resize_token_embeddings(len(self.tokenizer))

        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.peft_model = PeftModel.from_pretrained(model, disc_model_id, torch_dtype=torch.float32, offload_folder="lora_results/temp")
        self.peft_model.eval()

        self.accelerator = Accelerator()


    def _schedule_all_comparisons(self, options, group_size=3):
        """
        Schedules all possible comparisons with up to 3 options.
        Each comparison is between 2 or 3 options.

        Args:
            options (List[Option]): List of options to compare.

        Returns:
            List[List[Option]]: List of comparisons.
        """
        comparisons = []
        
        # Schedule all possible 2-option comparisons
        comparisons.extend(list(itertools.combinations(options, 2)))
        
        if group_size > 2:
            # Schedule all possible 3-option comparisons
            comparisons.extend(list(itertools.combinations(options, 3)))
        
        return [list(comparison) for comparison in comparisons]


    def _schedule_random_comparisons(self, options, cmp_per_opt=3, group_size=3):
        """
        Schedules a random subset of comparisons ensuring each option participates
        in approximately 'cmp_per_opt' comparisons.
        Useful for larger N to limit the number of comparisons.

        Args:
            options (List[Option]): List of options to compare.
            cmp_per_opt (int): Number of comparisons each option should participate in.
            group_size (int): Number of options in each comparison.

        Returns:
            List[List[Option]]: List of comparisons.
        """
        N = len(options)
        target_total_comparisons = ceil((cmp_per_opt * N) / group_size)
        
        if len(options) < group_size:
            # Generate combinations of all available elements
            all_comparisons = list(itertools.combinations(options, len(options)))
        else:
            # Generate all possible group_size-opt comparisons
            all_comparisons = list(itertools.combinations(options, group_size))
        random.shuffle(all_comparisons)
        
        comparisons = []
        participation_count = defaultdict(int)
        
        for comparison in all_comparisons:
            if all(participation_count[option.id] < cmp_per_opt for option in comparison):
                comparisons.append(list(comparison))
                for option in comparison:
                    participation_count[option.id] += 1
                if len(comparisons) >= target_total_comparisons:
                    break
        
        return comparisons


    def _rank_options(self, options):
        """
        Ranks options based on their scores.
        Returns the list of options sorted by score descending.

        Args:
            options (List[Option]): List of options to rank.

        Returns:
            List[Option]: List of options sorted by score descending.
        """
        return sorted(options, key=lambda x: x.score, reverse=True)


    def _prepare_meta_knowledge(meta_knowledge_path, test_q_id, num_references=1):
        '''
        Prepare the meta-knowledge for the given test question ID.

        Args:
            meta_knowledge_path (str): The path to the meta-knowledge.
            test_q_id (str): The test question ID.
            num_references (int): The number of references to include.

        Returns:
            meta_knowledge (str): The meta knowledge for the test question.
        '''
        with open(f'{meta_knowledge_path}/similar_question_ids.json', 'r') as f:
            data = json.load(f)
        similar_filename_ls = data[test_q_id]
        meta_knowledge = ''
        cnt = 0
        for file in similar_filename_ls:
            file_mapped = f'{meta_knowledge_path}/{file}.json'
            if not os.path.exists(file_mapped):
                continue
            with open(file_mapped, 'r') as f:
                data = json.load(f)
            meta_knowledge += '\n\n' + data['Knowledge']
            cnt += 1
            if cnt >= num_references:
                break
        return meta_knowledge.strip()


    def _post_process(self, data, rollout_id, selected_option, disc_data, output_dir, filename=None):
        '''
        Post-process the generated results.

        Args:
            data (Dict): The data dictionary.
            rollout_id (str): The ID of the rollout.
            selected_option (str): The selected option.
            disc_data (List): The discrimination data.
            output_dir (str): The output directory to save the results.
            filename (str): The filename to save the results.

        Returns:
            data (Dict): The updated data dictionary.
        '''
        if rollout_id == 'Agg':
            data[rollout_id]['prompt'] = f"{data[rollout_id]['prompt'].strip()}\n{selected_option}"
            data[rollout_id]['search_tree'].append([data[rollout_id]['responses'], data[rollout_id]['futures'], disc_data]) 
            data[rollout_id]['responses'] = []
            data[rollout_id]['futures'] = []  
        else:
            data['rollout'][rollout_id]['prompt'] = f"{data['rollout'][rollout_id]['prompt'].strip()}\n{selected_option}"
            data['rollout'][rollout_id]['search_tree'].append([data['rollout'][rollout_id]['responses'], data['rollout'][rollout_id]['futures'], disc_data]) 
            data['rollout'][rollout_id]['responses'] = []
            data['rollout'][rollout_id]['futures'] = []    
        
        if filename is not None:
            with open(f'{output_dir}/{filename}', 'w') as f:
                json.dump(data, f)

        return data


    def _reshape_res(self, prompts_ls, result):
        '''
        Reshape the results to the original list.

        Args:
            prompts_ls (List[List[str]]): The list of prompts.
            result (List[str]): The list of results.

        Returns:
            original_dist (List[List[str]]): The reshaped list of results.
        '''
        original_dist = []
        index = 0
        for sublist in prompts_ls:
            length = len(sublist)
            original_dist.append(result[index:index + length])
            index += length
        return original_dist


    def _run_one_batch(self, batch_size, output_dir, rollout_id, samples, prompts_ls, options_ls, comparisons_ls, filenames, visualize):
        '''
        Run one batch of samples through the discriminator model.

        Args:
            batch_size (int): The batch size to use for inference.
            output_dir (str): The output directory to save the results.
            rollout_id (str): The ID of the rollout.
            samples (List[Dict]): The list of samples to process.
            prompts_ls (List[List[str]]): The list of prompts.
            options_ls (List[List[Option]]): The list of options.
            comparisons_ls (List[List[List[Option]]]): The list of comparisons.
            filenames (List[str]): The list of filenames.
            visualize (bool): Whether to visualize the results.

        Returns:
            None
        '''
        flat_prompts_ls = [item for sublist in prompts_ls for item in sublist]    
        flat_results = distributed_inference(self.peft_model, self.tokenizer, self.accelerator, flat_prompts_ls, batch_size, max_new_tokens=256, top_k=10, top_p=0.9, temperature=0.2)
        flat_results = [f'{prompt.strip()}\n{res}' for prompt, res in zip(flat_prompts_ls, flat_results)]
        recovered_res = self._reshape_res(prompts_ls, flat_results)

        for i in range(len(samples)):
            disc_res = recovered_res[i]
            comparisons = comparisons_ls[i]
            
            if visualize:
                print('Discrimination result:')
            for idx_res in range(len(disc_res)):
                cur_res = disc_res[idx_res]
                if visualize:
                    print(cur_res)
                    print('------------------------------------------')
                if '"Conclusion":' in cur_res:
                    cur_res = cur_res.split('"Conclusion":')[1].strip()
                cur_res = cur_res.lower()
                winner = None
                if 'answer 1' in cur_res:
                    winner = comparisons[idx_res][0]
                elif 'answer 2' in cur_res and len(comparisons[idx_res]) > 1:
                    winner = comparisons[idx_res][1]
                elif 'answer 3' in cur_res and len(comparisons[idx_res]) > 2:
                    winner = comparisons[idx_res][2]
                
                if winner is not None:
                    winner.score += 1

            ranked_options = self._rank_options(options_ls[i])
            final_winner = ranked_options[0]

            if rollout_id == 'Agg':
                samples[i]['flag_correct'] = self._judge_final_answer(final_winner.description, samples[i]['answer'])

            self._post_process(samples[i], rollout_id, final_winner.description, disc_res, output_dir, filenames[i])
        if visualize:
            print('=============================================')


    def _judge_final_answer(self, pred, gt):
        '''
        Judge the final answer.

        Args:
            pred (str): The predicted answer.
            gt (str): The ground truth answer.

        Returns:
            flag_correct (bool): Whether the answer is correct.
        '''
        gt_result = parse_boxed_result(gt)
        flag_correct = None
        if '"Final answer":' in pred:
            boxed_result = extract_final_answer(pred)
            flag_correct = grade_answer(boxed_result, gt_result)
        return flag_correct


    def inference(self, output_dir, meta_knowledge_path, rollout_id, batch_size, max_future_len, cmp_per_opt, group_size, deduplicate=True, 
                  visualize=False, final_agg=False, structure_check=False):
        '''
        Perform inference on the given dataset.

        Args:
            output_dir (str): The output directory to save the results.
            meta_knowledge_path (str): The path to the meta-knowledge.
            rollout_id (str): The ID of the rollout.
            batch_size (int): The batch size to use for inference.
            max_future_len (int): The maximum length of the future steps.
            cmp_per_opt (int): The number of comparisons per option.
            group_size (int): The number of options in each comparison.
            deduplicate (bool): Whether to deduplicate the responses.
            visualize (bool): Whether to visualize the results.
            final_agg (bool): Whether to perform final aggregation.
            structure_check (bool): Whether to check the structure of the responses.

        Returns:
            None
        '''
        src_files = os.listdir(output_dir)
        src_files = [filename for filename in src_files if filename.endswith('.json')]
        src_files = sorted(src_files)

        if self.use_meta_knwoledge:
            instruction = f'Given the meta knowledge and a {self.prob_type} reasoning problem,'
        else:
            instruction = f'Given a {self.prob_type} reasoning problem,'
        instruction += ' tell me which answer is better. You should provide me their comparison, your thought and conclusion. In the thought, you should show me all the details. DONT skip any step. Only return me JSON. You should choose only ONE answer.'

        samples = []
        prompts_ls = []
        options_ls = []
        comparisons_ls = []
        filenames = []
        for filename in tqdm(src_files):
            with open(f'{output_dir}/{filename}', 'r') as f:
                # The format is a dictionary of rollouts with rollout_id as the key, each containing the prompt, responses, and futures.
                # {'rollout': {"0": {'prompt': String, 'responses': List[String], 'futures': List[String], 'search_tree': List[List]}, "1": {}, "2": {}, ...},
                #  'Agg': {'prompt': String, 'responses': List[String], 'futures': List[String], 'search_tree': List[List]}}
                sample = json.load(f)
            
            problem = sample['question']
            if not final_agg:
                context = sample['rollout'][rollout_id]['prompt'].split('### Output:')[1].strip()
                context = context.replace('"\n', '",\n')
                responses = sample['rollout'][rollout_id]['responses']
                futures = sample['rollout'][rollout_id]['futures']
            else:
                context = ''
                responses = []
                futures = []
                for rollout in sample['rollout']:
                    # responses.apppend(rollout['prompt'].split('### Output:')[1].split('"Final answer":')[1].strip())
                    # futures.append('"Final answer": ' + rollout['prompt'].split('"Final answer":')[1].strip())
                    responses.append(sample['rollout'][rollout]['prompt'].split('### Output:')[1].strip())
                    futures.append([])
                
                sample[rollout_id] = {'prompt': '', 'responses': [], 'futures': [], 'search_tree': []}


            if deduplicate:
                # Initialize a dictionary to maintain unique responses and corresponding futures
                unique_responses = {}
                for response, future in zip(responses, futures):
                    if response not in unique_responses:
                        unique_responses[response] = future

                # Extract the deduplicated responses and their corresponding futures
                responses = list(unique_responses.keys())
                futures = list(unique_responses.values())


            if structure_check:
                # Filter out responses that have incorrect structure
                responses_filtered = []
                futures_filtered = []
                for response, future in zip(responses, futures):
                    if 'graph' not in response.split(':')[0].lower() or check_graph_structure(eval(response.split(':')[1])):
                        responses_filtered.append(response)
                        futures_filtered.append(future)

                responses = responses_filtered
                futures = futures_filtered
                
                
            options = [Option(i, responses[i], futures[i]) for i in range(len(responses))]
            num_options = len(responses)

            if num_options == 0:
                continue
            elif num_options == 1:
                if final_agg: 
                    sample['flag_correct'] = self._judge_final_answer(responses[0], sample['answer'])
                self._post_process(sample, rollout_id, responses[0], None, output_dir, filename)
                continue
            
            meta_knowledge = None
            if self.use_meta_knwoledge:
                meta_knowledge = self._prepare_meta_knowledge(meta_knowledge_path, sample['id'])


            comparisons = self._schedule_random_comparisons(options, cmp_per_opt, group_size)
            prompts = [f'{instruction}\n ### Input: \n{prepare_prompt_for_disciminator(problem, context, [option.description for option in cur_batch], [option.future for option in cur_batch], meta_knowledge, future_range=range(max_future_len))}\n ### Output: \n' for cur_batch in comparisons]

            if self.show_prompt_only:
                # For debugging purposes
                for prompt in prompts:
                    print(prompt)
                    print('------------------------------------------')
                print('=============================================')
                continue

            samples.append(sample)
            prompts_ls.append(prompts)
            options_ls.append(options)
            comparisons_ls.append(comparisons)
            filenames.append(filename)

            if len(samples) >= batch_size:
                self._run_one_batch(batch_size, output_dir, rollout_id, samples, prompts_ls, options_ls, comparisons_ls, filenames, visualize)
                samples = []
                prompts_ls = []
                options_ls = []
                comparisons_ls = []
                filenames = []
                
        if len(samples) > 0:
            self._run_one_batch(batch_size, output_dir, rollout_id, samples, prompts_ls, options_ls, comparisons_ls, filenames, visualize)




class Option:
    '''
    Class to represent an option in the comparison task for the discriminator model.
    '''
    def __init__(self, option_id: int, description: str, future: str):
        '''
        Initialize the Option object.
        
        Args:
            option_id (int): The ID of the option.
            description (str): The description of the option.
            future (str): The future of the option.

        Returns:
            None
        '''
        self.id = option_id
        self.description = description
        self.future = future
        self.score = 0