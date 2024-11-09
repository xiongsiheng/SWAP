from utlis import *




def my_generate_prompt_sem_equ(inputs, mask=None, allow_ori_sen=True):
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
    
    print('Prompt Generation:')
    for prompt in prompts:
        print(prompt)
        print('-'*20)
    print('=============================================')
    return prompts



def prepare_prompt_for_disciminator(problem, context, options, futures, external_knowledge, future_range=None, gt=None, 
                                    provide_gt=False, use_external_knowledge=False):
    ip = convert_ip_format2(problem)
    
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

    # ip is question, context is the known steps before the search, op is the search steps before the target step
    # target step can be middle steps in the search steps

    merged_context = f'{ip}'
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
    prompt += f'{context},\n"{key}":\n{{\n"Answer 1": {answer1},\n'
    if len(options) >= 2:
        prompt += f'"Answer 2": {answer2},\n'
    if len(options) >= 3:
        prompt += f'"Answer 3": {answer3}\n'
    prompt += '}'

    prompt_future = f',\n"Future":\n{{\n"Future 1": {future1},\n'
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