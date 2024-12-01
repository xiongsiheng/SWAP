from utils import *



def generate_prompt_for_generator(prob_type, question, trajectory, eos_token="", use_graph=True):
    '''
    Generate the prompt for the generator.

    Args:
        prob_type (str): The type of the problem.
        question (str): The question.
        trajectory (str): The trajectory.
        eos_token (str): The end-of-sequence token.
        use_graph (bool): Whether to use the graph format.

    Returns:
        prompt (str): The prompt.
    '''
    instruction = f'Given a {prob_type} problem, solve it with the reinforcement learning format.'
    input = convert_element_format('Problem', question, convert_json=True)
    prompt = f'{instruction}\n\n ### Input: \n{input}\n ### Output: \n'
    output = convert_dict_format(eval(trajectory), use_graph=use_graph)
    
    if output is not None:
        # Append the output to the prompt
        prompt += f"{output}\n"
        
        # Replace unwanted escape characters with their respective representations
        prompt = prompt.replace('\x0c', '\f') \
                    .replace('\x07', '\a') \
                    .replace('\x08', '\b') \
                    .replace('\x0b', '\v') \
                    .replace('\x0d', '\r') \
                    .replace('\x0a', '\n')

    # Add the end-of-sequence token
    prompt += eos_token
    
    return prompt





def obtain_sem_equ_data(samples, allow_ori_sen=True):
    '''
    Obtain the semantic equivalent data.

    Args:
        samples (list): The list of samples.
        allow_ori_sen (bool): Whether to allow the original sentences.

    Returns:
        data_dict (dict): The data dictionary.
    '''
    data_dict = {'instruction': [], 'input': [], 'output': []}
    
    instruction = 'Given the original sentences, find me a sentence that is semantically equivalent to them.'
    if allow_ori_sen:
        instruction += ' You CAN return me the original sentences.'
    else:
        instruction += ' You should NOT return me the original sentences.'

    for sample in samples:
        ori_sen = sample['input']
        new_sen = sample['output']
        
        if not isinstance(ori_sen, str):
            ori_sen = str(ori_sen)

        if not isinstance(new_sen, list):
            new_sen = [new_sen]
        
        new_sen = [str(sen) for sen in new_sen]
                    
        data_dict['instruction'].extend([instruction for _ in range(len(new_sen))])
        data_dict['input'].extend([ori_sen for _ in range(len(new_sen))])
        data_dict['output'].extend(new_sen)

        if allow_ori_sen:
            data_dict['instruction'].append(instruction)
            data_dict['input'].append(ori_sen)
            data_dict['output'].append(ori_sen)
        
    return data_dict





def generate_prompt_for_sem_equ_lora_train(instruction, input, output, eos_token=""):
    '''
    Generate the prompt for the semantic equivalent LoRA training.

    Args:
        instruction (str): The instruction.
        input (str): The input.
        output (str): The output.
        eos_token (str): The end-of-sequence token.

    Returns:
        prompt (str): The prompt.
    '''
    prompt = f'{instruction}\n\n ### Input: \n{input}\n ### Output: \n'
    if output is not None:
        prompt += f"{output}\n"
    prompt = replace_all_escape_sequences(prompt)
    prompt = convert_escape_sequences(prompt)
    prompt += eos_token
    return prompt



def generate_prompt_for_sem_equ_lora_inference(inputs, mask=None, allow_ori_sen=True):
    '''
    Generate the prompt for the semantic equivalent LoRA inference.

    Args:
        inputs (list): The list of inputs.
        mask (list): The mask.
        allow_ori_sen (bool): Whether to allow the original sentences.

    Returns:
        prompts (list): The list of prompts.
    '''
    # Format of inputs: [[sen] * num_regenerate] * batch_size                                              
    inputs = ['\n\n'.join(compact_list([remove_tag(sen) for sen in input])) for input in inputs]
    if mask is not None:
        inputs = compact_list(inputs, mask=mask)
        
    instruction = 'Given the original sentences, find me a sentence that is semantically equivalent to them.'
    if allow_ori_sen:
        instruction += ' You CAN return me the original sentences.'
    else:
        instruction += ' You should NOT return me the original sentences.'
    prompts = [f'{instruction}\n\n ### Input: \n{input}\n ### Output: \n' for input in inputs]
    
    # print('Prompt Generation:')
    # for prompt in prompts:
    #     print(prompt)
    #     print('-'*20)
    # print('=============================================')
    return prompts






def generate_prompt_for_discriminator(prob_type, input, output, eos_token="", use_meta_knwoledge=True, use_graph=True):
    '''
    Generate the prompt for the discriminator during training.

    Args:
        prob_type (str): The type of the problem.
        input (str): The input.
        output (str): The output.
        eos_token (str): The end-of-sequence token.
        use_meta_knwoledge (bool): Whether to use the meta-knowledge.
        use_graph (bool): Whether to use the graph format.

    Returns:
        prompt (str): The prompt.
    '''
    input_dict = eval(input)
    if use_meta_knwoledge:
        instruction = f'Given the meta knowledge and a {prob_type} reasoning problem,'
    else:
        instruction = f'Given a {prob_type} reasoning problem,'
        del input_dict['Meta-knowledge']
    instruction += ' tell me which answer is better. You should provide me their comparison, your thought and conclusion. In the thought, you should show me all the details. DONT skip any step. Only return me JSON. You should choose only ONE answer.'
    
    responses = input_dict["Search steps"]
    del input_dict["Search steps"]
    
    if "Future" in input_dict:
        futures = input_dict["Future"]
        del input_dict["Future"]
    else:
        futures = {"Future 1": {}, "Future 2": {}}
        if len(responses) > 2:
            futures["Future 3"] = {}

    responses = convert_dict_format(responses, use_graph=use_graph)
    futures = convert_dict_format(futures, use_graph=use_graph)

    input = convert_dict_format(input_dict, use_graph=use_graph)
    input += f',\n"Search steps":\n{{\n{responses}\n}},\n"Futures":\n{{\n{futures}\n}}'

    prompt = f'{instruction}\n\n ### Input: \n{input}\n ### Output: \n'
    output = convert_dict_format(eval(output), use_graph=use_graph)
    
    if output is not None:
        # Append the output to the prompt
        prompt += f"{output}\n"
        
        # Replace unwanted escape characters with their respective representations
        prompt = prompt.replace('\x0c', '\f') \
                    .replace('\x07', '\a') \
                    .replace('\x08', '\b') \
                    .replace('\x0b', '\v') \
                    .replace('\x0d', '\r') \
                    .replace('\x0a', '\n')

    # Add the end-of-sequence token
    prompt += eos_token
    
    return prompt




def prepare_prompt_for_disciminator(problem, context, options, futures, external_knowledge, future_range=None, gt=None, 
                                    provide_gt=False, use_external_knowledge=False):
    '''
    Prepare the prompt for the discriminator during inference.

    Args:
        problem (str): The problem.
        context (str): The context.
        options (list): The list of options.
        futures (list): The list of futures.
        external_knowledge (list): The external knowledge.
        future_range (list): The range of the future.
        gt (str): The ground truth.
        provide_gt (bool): Whether to provide the ground truth.
        use_external_knowledge (bool): Whether to use the external knowledge.

    Returns:
        prompt (str): The prompt.
    '''
    q = convert_element_format("Problem", problem, convert_json=True)
    
    options = [option.replace('\\\n', '\\n') for option in options]
    option1 = options[0].strip().split('\n')
    if len(options) >= 2:
        option2 = options[1].strip().split('\n')
    if len(options) >= 3:
        option3 = options[2].strip().split('\n')

    futures = [future.replace('\\\n', '\\n') if len(future) > 0 else [] for future in futures]
    future1 = futures[0].strip().split('\n') if len(futures[0]) > 0 else []
    if len(futures) >= 2:
        future2 = futures[1].strip().split('\n') if len(futures[1]) > 0 else []
    if len(futures) >= 3:
        future3 = futures[2].strip().split('\n') if len(futures[2]) > 0 else []

    # q is question, context is the known steps before the search steps
    merged_context = f'{q}'
    if len(context) > 0:
        merged_context += f',\n{context}'
    context = merged_context.strip()
    if context.endswith(','):
        context = context[:-1]

    answer1 = convert_list_into_dict(option1)
    future1 = convert_list_into_dict(future1) if future_range is None else \
                convert_list_into_dict([future1[idx] for idx in future_range if idx < len(future1)])
    if len(options) >= 2:
        answer2 = convert_list_into_dict(option2)
        future2 = convert_list_into_dict(future2) if future_range is None else \
                convert_list_into_dict([future2[idx] for idx in future_range if idx < len(future2)])
    if len(options) >= 3:
        answer3 = convert_list_into_dict(option3)
        future3 = convert_list_into_dict(future3) if future_range is None else \
                convert_list_into_dict([future3[idx] for idx in future_range if idx < len(future3)])

    key = 'Search steps'

    prompt = ''
    if use_external_knowledge:
        prompt += f'"External knowledge": {json.dumps(external_knowledge)},\n'
    prompt += f'{context},\n"{key}":\n{{\n"Option 1": {answer1},\n'
    if len(options) >= 2:
        prompt += f'"Option 2": {answer2},\n'
    if len(options) >= 3:
        prompt += f'"Option 3": {answer3}\n'
    prompt += '}'

    prompt_future = f',\n"Futures":\n{{\n"Future 1": {future1},\n'
    if len(options) >= 2:
        prompt_future += f'"Future 2": {future2},\n'
    if len(options) >= 3:
        prompt_future += f'"Future 3": {future3}\n'
    prompt_future += '}'
    prompt += prompt_future
    prompt = convert_escape_sequences(my_unicode_to_latex(prompt))

    if provide_gt and gt is not None:
        prompt += f',\n{gt}'

    return prompt