import json
from itertools import islice
from accelerate.utils import gather_object


def convert_op_format(output):
    op = ''
    for key in output:
        if key in ['Graph', 'graph']:
            continue
        content = replace_all_escape_sequences(str(output[key]))
        op += f'"{key}": "{content}"\n'
    return op.strip()

def convert_ip_format(key, input):
    return f'"{key}": {json.dumps(input)}'

def convert_ip_format2(input):
    return f'"Problem": {input}'


def convert_list_into_dict(ls):
    return '{' + ', '.join(ls) + '}'


def merge_dicts(dict1, dict2):
    for key in dict1:
        dict1[key] += dict2[key]
    return dict1


def create_subset(dataset, size=None, indices=None, shuffle=False, seed=None):
    '''
    Create a subset of the dataset.
    '''
    if size == -1:
        return dataset
    
    # Define the indices for the subset
    if indices is None:
        indices = list(range(len(dataset)))[:size]

    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    subset = dataset.select(indices)
    return subset




def obtain_data_dict(file_ls):
    data_dict = {'problem': [], 'solution': [], 'idx': []}
    for file in file_ls:
        with open(file, 'r') as f:
            data = json.load(f)
        data_dict['problem'].append(data['problem'])
        data_dict['solution'].append(data['solution'])
        data_dict['idx'].append(file.split('/')[-1].split('.')[0])

    return data_dict


def replace_all_escape_sequences(line):
    line = line.replace('\\\\', '\\')
    
    # Use repr to get a string representation of text with escape sequences
    line = repr(line)[1:-1]  # Strip the quotes added by repr
    
    line = line.replace('\\n', '\n')
    line = line.replace("\\'", "'")
    
    return line


def convert_escape_sequences(text):
    return text.replace('\\x0c', '\\f').replace('\\x07', '\\a').replace('\\x08', '\\b').replace('\\x0b', '\\v').replace('\\x0d', '\\r').replace('\\x0a', '\\n')



def prepare_external_knowledge(test_q_id):
    with open(f'../results/MATH_algebra_similar_question_ids_meta_knowledge.json', 'r') as f:
        data = json.load(f)
    _filename_map = lambda name: f'../results/MATH_test_algebra_rewrite_problem_gpt_4o_from_concepts_meta_knowledge/{name.split("/")[-1]}_meta_knowledge.json'
    similar_filename_ls = data[test_q_id]
    external_knowledge = ''
    cnt = 0
    for file in similar_filename_ls:
        file_mapped = _filename_map(file)
        if not os.path.exists(file_mapped):
            continue
        with open(file_mapped, 'r') as f:
            data = json.load(f)
        external_knowledge += '\n\n' + data['response']['Knowledge']
        cnt += 1
        if cnt >= 1:
            break
    return external_knowledge.strip()




def my_unicode_to_latex(text):
    unicode_to_latex = {
        "\\u03c0": r"\\pi",
        "\\u03b1": r"\\alpha",
        "\\u03b2": r"\\beta",
        "\\u03b3": r"\\gamma",
        "\\u03b4": r"\\delta",
        "\\u03b5": r"\\epsilon",
        "\\u03b6": r"\\zeta",
        "\\u03b7": r"\\eta",
        "\\u03b8": r"\\theta",
        "\\u03b9": r"\\iota",
        "\\u03ba": r"\\kappa",
        "\\u03bb": r"\\lambda",
        "\\u03bc": r"\\mu",
        "\\u03bd": r"\\nu",
        "\\u03be": r"\\xi",
        "\\u03bf": r"\\omicron",
        "\\u03c1": r"\\rho",
        "\\u03c3": r"\\sigma",
        "\\u03c4": r"\\tau",
        "\\u03c5": r"\\upsilon",
        "\\u03c6": r"\\phi",
        "\\u03c7": r"\\chi",
        "\\u03c8": r"\\psi",
        "\\u03c9": r"\\omega",
    }
    unicode_to_latex.update({
        "\\u03b4": r"\\delta",       # Greek letter delta
        "\\u00b2": r"^2",           # Superscript two
        "\\u0007": r"\\",             # Bell character, typically not used in LaTeX
        "\\u2248": r"\\approx",      # Approximately equal to
        "\\u2260": r"\\neq",         # Not equal to
        "\\u2309": r"\\rceil",       # Right ceiling
        "\\u00d7": r"\\times",       # Multiplication sign
        "\\u221e": r"\\infty",       # Infinity
        "\\u2265": r"\\geq",         # Greater than or equal to
        "\\u230b": r"\\rfloor",      # Right floor
        "\\u2014": r"---",          # Em dash
        "\\u2019": r"'",            # Right single quotation mark
        "\\u0127": r"\\textit{h}",        # Latin small letter h with stroke
        "\\u2308": r"\\lceil",       # Left ceiling
        "\\u03c0": r"\\pi",          # Greek letter pi
        "\\u221a": r"\\sqrt",      # Square root
        "\\u230a": r"\\lfloor",      # Left floor
        "\\u00b1": r"\\pm",          # Plus-minus sign
        "\\u2264": r"\\leq"          # Less than or equal to
    })

    # Iterate over the dictionary and replace Unicode characters with LaTeX
    for unicode_char, latex in unicode_to_latex.items():
        text = text.replace(unicode_char, latex)
    return text


def chunked(iterable, chunk_size):
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk



def compact_list(cur_list, mask=None):
    if mask is None:
        return [x for x in cur_list if x is not None]
    return [x for x, m in zip(cur_list, mask) if m]


def convert_mask_into_idx(mask):
    return [idx for idx, m in enumerate(mask) if m]



def distributed_inference(model, tokenizer, accelerator, prompts, batch_size, max_new_tokens, top_k, top_p, temperature):
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


def remove_tag(sentence):
    if sentence is None:
        return None
    if ':' in sentence:
        sentence = sentence.split(':')[-1].strip()
    if sentence[0] == '"':
        sentence = sentence[1:]
    if sentence[-1] == '"':
        sentence = sentence[:-1]
    return sentence.strip()