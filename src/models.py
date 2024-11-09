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
from peft import PeftModel

from utlis import *
from prompt_generation import *
from grading.grader import grade_answer












class Generator:
    def __init__(self, gen_model_id, sem_model_id, enable_DBM=True, show_prompt_only=False, prob_type=None):
        assert prob_type in ['math', 'logical reasoning', 'coding'], "Invalid problem type."   # In our paper, we only use these three types of problems. You can adapt to other types.
        self.enable_DBM = enable_DBM
        self.show_prompt_only = show_prompt_only
        self.prob_type = prob_type
        if not show_prompt_only:
            self._build_model(gen_model_id, sem_model_id)


    def _build_model(self, gen_model_id, sem_model_id, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
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


    def _to_binary_distribution(self, probabilities):
        # Get the index of the maximum probability for each batch
        max_indices = torch.argmax(probabilities, dim=-1)
        
        # Create a binary distribution tensor with the same shape as `probabilities`
        binary_distribution = torch.zeros_like(probabilities)
        
        # Set the maximum index in each batch to 1
        binary_distribution[torch.arange(probabilities.size(0)), max_indices] = 1.0
        
        return binary_distribution


    def _get_last_token_prob_distribution(self, model, input_ids, attention_mask, past_key_values=None, temperature=1.0):
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
        Function to sample from the averaged probability distributions
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
        start: starting point, ratio: common ratio, num_terms: number of terms, offset: offset
        '''
        sequence = []
        term = start
        for _ in range(num_terms):
            sequence.append(term + offset)
            term *= ratio
        return sequence


    def _generate_with_averaged_sampling(self, peft_model1, peft_model2, input_ids1, input_ids2, attention_mask1, attention_mask2, 
                                        max_length, top_k, top_p, temperature, stop_count):
        '''
        Function to generate tokens iteratively with averaged sampling probabilities
        '''
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
        responses = self.tokenizer.batch_decode(generation_output[:, input_tokens.shape[-1]:], skip_special_tokens=True)
        outputs = [response.strip() for response in responses]
        outputs = [convert_escape_sequences(replace_all_escape_sequences(output)) for output in outputs]
        return outputs


    def _run_one_batch(self, samples, num_future_steps, force_termination, output_dir, visualize):
        prompts = [sample['prompt'] for sample in samples]
        
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
        prev_preds = [sample['responses'] for sample in samples]

        # semantic equivalent model
        input_tokens_sem = None
        attention_mask_sem = None
        if (self.peft_model_sem is not None) and (len(prev_preds[0]) > 0):
            prompts_sem = my_generate_prompt_sem_equ(prev_preds)
            tokenized_inputs_sem = self.tokenizer(prompts_sem, return_tensors="pt", padding=True)
            input_tokens_sem = tokenized_inputs_sem["input_ids"].to("cuda")
            attention_mask_sem = tokenized_inputs_sem['attention_mask'].to("cuda")
        
        with torch.cuda.amp.autocast():
            generation_output = self._generate_with_averaged_sampling(self.peft_model, self.peft_model_sem, input_tokens, input_tokens_sem, attention_mask, 
                                                                attention_mask_sem, max_length=512, top_k=50, top_p=0.9, temperature=0.7, 
                                                                stop_count=1)

        results = self._decode_new_steps(generation_output, input_tokens)
        results = [f'{tag} "{res}' for tag, res in zip(tags, results)]

        futures = [[] for _ in range(len(samples))]
        if num_future_steps > 0:
            prompts_new = [f"{prompt}{res.strip()}\n" for (prompt, res) in zip(prompts, results)]
            tokenized_inputs = self.tokenizer(prompts_new, return_tensors="pt", padding=True)
            input_tokens = tokenized_inputs["input_ids"].to("cuda")
            attention_mask = tokenized_inputs['attention_mask'].to("cuda")

            generation_output = self._generate_with_averaged_sampling(self.peft_model, None, input_tokens, None, attention_mask, 
                                                                None, max_length=2048, top_k=50, top_p=0.9, temperature=0.3, 
                                                                stop_count=num_future_steps)
            futures = self._decode_new_steps(generation_output, input_tokens)
        
        if visualize:
            print('Results:')
        for idx_sample in range(len(prompts)):
            if visualize:
                print(prompts[idx_sample])
                print('-------------------')
                print(results[idx_sample])
                print('-------------------')
                print(futures[idx_sample])
                print('-------------------')
            
            flag_correct = None
            # if 'Final answer' in results[idx_sample]:
            #     boxed_result = extract_final_answer(results[idx_sample])
            #     gt_result = samples[idx_sample]['answer'].split('####')[-1].strip()
            #     flag_correct = grade_answer(boxed_result, gt_result)

            samples[idx_sample]['responses'].append(results[idx_sample])
            samples[idx_sample]['futures'].append(futures[idx_sample])
            op_dict = samples[idx_sample]
            op_dict['flag_correct'] = flag_correct
            with open(f'{output_dir}/{samples[idx_sample]["idx"]}.json', 'w') as f:
                json.dump(op_dict, f)
        if visualize:
            print('===================')


    def inference(self, dataset, output_dir, batch_size, num_generations, num_future_steps, force_termination=False, visualize=False):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        instruction = f'Given a {self.prob_type} problem, solve it with the reinforcement learning format.'
        flag_finish = False
        for _ in range(num_generations):
            num_processed_samples = 0
            samples = []
            for sample in tqdm(dataset, total=len(dataset)):
                sample['responses'] = []
                sample['futures'] = []
                sample['search_tree'] = []
                Input = convert_ip_format('Problem', sample['problem'])
                sample['prompt'] = f'{instruction}\n\n ### Input:\n{Input}\n ### Output: \n'

                if os.path.exists(f'{output_dir}/{sample["idx"]}.json'):
                    with open(f'{output_dir}/{sample["idx"]}.json', 'r') as f:
                        data = json.load(f)
                    sample['responses'] = data['responses']
                    sample['futures'] = data['futures']
                    sample['search_tree'] = data['search_tree']
                    sample['prompt'] = f"{data['prompt'].strip()}\n"

                if self.show_prompt_only:
                    print(sample['prompt'])
                    print('-------------------')
                    continue
                
                if 'Final answer' in sample['prompt']:
                    continue

                samples.append(sample)
                num_processed_samples += 1
                if len(samples) >= batch_size:
                    self._run_one_batch(samples, num_future_steps, force_termination, output_dir, visualize)
                    samples = []

            if len(samples) > 0:
                self._run_one_batch(samples, num_future_steps, force_termination, output_dir, visualize)
            
            if self.show_prompt_only:
                break
            
            if num_processed_samples == 0:
                flag_finish = True
                break

        return flag_finish




class Discriminator():
    def __init__(self, disc_model_id, enable_meta_knwoledge=False, show_prompt_only=False, prob_type=None):
        assert prob_type in ['math', 'logical reasoning', 'coding'], "Invalid problem type."   # In our paper, we only use these three types of problems. You can adapt to other types.
        self.enable_meta_knwoledge = enable_meta_knwoledge
        self.show_prompt_only = show_prompt_only
        self.prob_type = prob_type
        if not show_prompt_only:
            self._build_model(disc_model_id)


    def _build_model(self, disc_model_id, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
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


    def _schedule_all_comparisons(self, options):
        """
        Schedules all possible comparisons with up to 3 options.
        Each comparison is between 2 or 3 options.
        """
        comparisons = []
        
        # Schedule all possible 2-option comparisons
        comparisons.extend(list(itertools.combinations(options, 2)))
        
        # Schedule all possible 3-option comparisons
        comparisons.extend(list(itertools.combinations(options, 3)))
        
        return [list(comparison) for comparison in comparisons]


    def _schedule_random_comparisons(self, options, cmp_per_opt=3):
        """
        Schedules a random subset of comparisons ensuring each option participates
        in approximately 'cmp_per_opt' comparisons.
        Useful for larger N to limit the number of comparisons.
        """
        N = len(options)
        target_total_comparisons = ceil((cmp_per_opt * N) / 3)
        
        # Generate all possible 3-opt comparisons
        all_comparisons = list(itertools.combinations(options, 3))
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
        """
        return sorted(options, key=lambda x: x.score, reverse=True)


    def _post_process(self, data, selected_option, disc_data, output_dir, filename=None):
        data['prompt'] = f"{data['prompt'].strip()}\n{selected_option}"
        data['search_tree'].append([data['responses'], data['futures'], disc_data]) 
        data['responses'] = []
        data['futures'] = []    
        
        if filename is not None:
            with open(f'{output_dir}/{filename}', 'w') as f:
                json.dump(data, f)

        return data


    def _reshape_res(self, prompts_ls, result):
        original_dist = []
        index = 0
        for sublist in prompts_ls:
            length = len(sublist)
            original_dist.append(result[index:index + length])
            index += length
        return original_dist


    def _run_one_batch(self, batch_size, output_dir, samples, prompts_ls, options_ls, comparisons_ls, filenames, visualize):
        flat_prompts_ls = [item for sublist in prompts_ls for item in sublist]    
        flat_results = distributed_inference(self.peft_model, self.tokenizer, self.accelerator, flat_prompts_ls, batch_size, 1024, top_k=10, top_p=0.9, temperature=0.3)
        flat_results = [f'{prompt.strip()}\n{res}' for prompt, res in zip(flat_prompts_ls, flat_results)]
        recovered_res = self._reshape_res(prompts_ls, flat_results)

        for i in range(len(samples)):
            disc_res = recovered_res[i]
            comparisons = comparisons_ls[i]
            
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

            self._post_process(samples[i], final_winner.description, disc_res, output_dir, filenames[i])
        if visualize:
            print('=============================================')


    def inference(self, output_dir, batch_size, max_future_len, cmp_per_opt, deduplicate=True, visualize=False):
        src_files = os.listdir(output_dir)
        src_files = [filename for filename in src_files if filename.endswith('.json')]
        src_files = sorted(src_files)

        if self.enable_meta_knwoledge:
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
                sample = json.load(f)
            problem = sample['prompt'].split('"Problem":')[1].split('### Output:')[0].strip()

            context = sample['prompt'].split('### Output:')[1].strip()
            context = context.replace('"\n', '",\n')

            responses = sample['responses']
            futures = sample['futures']
            if deduplicate:
                # Initialize a dictionary to maintain unique responses and corresponding futures
                unique_responses = {}
                for response, future in zip(responses, futures):
                    if response not in unique_responses:
                        unique_responses[response] = future

                # Extract the deduplicated responses and their corresponding futures
                responses = list(unique_responses.keys())
                futures = list(unique_responses.values())

            options = [Option(i, responses[i], futures[i]) for i in range(len(responses))]
            num_options = len(responses)

            if num_options < 2:
                self._post_process(sample, responses[0], None, output_dir, filename)
                continue
            
            meta_knowledge = None
            if self.enable_meta_knwoledge:
                meta_knowledge = self._prepare_meta_knowledge(sample['idx'])


            comparisons = self._schedule_random_comparisons(options, cmp_per_opt)
            prompts = [f'{instruction}\n ### Input: \n{prepare_prompt_for_disciminator(problem, context, [option.description for option in cur_batch], [option.future for option in cur_batch], meta_knowledge, future_range=range(max_future_len))}\n ### Output: \n' for cur_batch in comparisons]

            if self.show_prompt_only:
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
                self._run_one_batch(batch_size, output_dir, samples, prompts_ls, options_ls, comparisons_ls, filenames, visualize)
                samples = []
                prompts_ls = []
                options_ls = []
                comparisons_ls = []
                filenames = []
                
        if len(samples) > 0:
            self._run_one_batch(batch_size, output_dir, samples, prompts_ls, options_ls, comparisons_ls, filenames, visualize)




class Option:
    def __init__(self, option_id: int, description: str, future: str):
        self.id = option_id
        self.description = description
        self.future = future
        self.score = 0