import json
from itertools import islice





def convert_element_format(key, value, convert_json=False):
    '''
    Convert the key-value pair into a string format.

    Args:
        key (str): The key.
        value (str): The value.
        convert_json (bool): Whether to convert the value into JSON format.

    Returns:
        str: The formatted key-value pair.
    '''
    if convert_json:
        return f'"{key}": {json.dumps(value)}'
    return f'"{key}": "{value}"'


def convert_dict_format(Dict, use_graph=True):
    '''
    Convert the dictionary into a string format.

    Args:
        Dict (dict): The dictionary.
        use_graph (bool): Whether to use the graph.

    Returns:
        str: The formatted dictionary.
    '''
    if Dict is None:
        return None
    op = ''
    for key in Dict:
        if (not use_graph) and ('graph' in key.lower()):
            continue
        content = replace_all_escape_sequences(str(Dict[key]))
        op += f'"{key}": "{content}"\n'
    return op.strip()


def convert_list_into_dict(ls):
    '''
    Convert the list into a dictionary.
    '''
    return '{' + ', '.join(ls) + '}'


def merge_dicts(dict1, dict2):
    '''
    Merge two dictionaries.
    '''
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
    '''
    Obtain the data dictionary.
    '''
    data_dict = {'problem': [], 'label': [], 'filename': []}
    for file in file_ls:
        with open(file, 'r') as f:
            data = json.load(f)
        data_dict['problem'] += data['problems']
        data_dict['label'] += data['labels']
        data_dict['filename'] += [file.split('/')[-1].split('.')[0] + f'_idx_{i}' for i in range(len(data['problems']))]
        # print(data_dict['filename'][-2:])
    return data_dict


def obtain_data_dict2(file_ls):
    '''
    Obtain the data dictionary.
    '''
    data_dict = {'problem': [], 'solution': [], 'idx': []}
    for file in file_ls:
        with open(file, 'r') as f:
            data = json.load(f)
        data_dict['problem'].append(data['problem'])
        data_dict['solution'].append(data['solution'])
        data_dict['idx'].append(file.split('/')[-1].split('.')[0])

    return data_dict


def replace_all_escape_sequences(line):
    '''
    Replace all escape sequences in the line.
    '''
    line = line.replace('\\\\', '\\')
    
    # Use repr to get a string representation of text with escape sequences
    line = repr(line)[1:-1]  # Strip the quotes added by repr
    
    line = line.replace('\\n', '\n')
    line = line.replace("\\'", "'")
    
    return line


def convert_escape_sequences(text):
    '''
    Convert the escape sequences in the text.
    '''
    return text.replace('\\x0c', '\\f').replace('\\x07', '\\a').replace('\\x08', '\\b').replace('\\x0b', '\\v').replace('\\x0d', '\\r').replace('\\x0a', '\\n')




def my_unicode_to_latex(text):
    '''
    Convert Unicode characters to LaTeX.
    '''
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
    '''
    Chunk the iterable into chunks of size chunk_size.
    '''
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk



def compact_list(cur_list, mask=None):
    '''
    Compact the list based on the mask or remove None values.
    '''
    if mask is None:
        return [x for x in cur_list if x is not None]
    return [x for x, m in zip(cur_list, mask) if m]


def convert_mask_into_idx(mask):
    '''
    Convert the mask into indices.
    '''
    return [idx for idx, m in enumerate(mask) if m]




def remove_tag(sentence):
    '''
    Remove the tag from the sentence.
    '''
    if sentence is None:
        return None
    if ':' in sentence:
        sentence = sentence.split(':')[-1].strip()
    if sentence[0] == '"':
        sentence = sentence[1:]
    if sentence[-1] == '"':
        sentence = sentence[:-1]
    return sentence.strip()




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



def extract_final_answer(response):
    '''
    Extract the final answer from the response.
    '''
    if '"Final answer"' in response:
        # Extract the part after "Final answer"
        answer_part = response.split('"Final answer"', 1)[1]
        # Remove unwanted characters and extra whitespace
        cleaned_answer = (
            answer_part.replace(':', '')
                        .replace('$', '')
                        .replace('"', '')
                        .replace('\\(', '')
                        .replace('\\)', '')
                        .strip()
        )
        return cleaned_answer
    return None



def parse_boxed_result(s):
    '''
    Parse the boxed result.
    '''
    s = str(s)
    # Find the start of the boxed content
    start = s.find('\\boxed{')
    if start == -1:
        return s
    
    # Skip past '\\boxed{' to start content capture
    start += len('\\boxed{')
    brace_count = 1  # We start after finding the first '{'
    content = []
    
    # Iterate over the string starting after '\boxed{'
    for i in range(start, len(s)):
        if s[i] == '{':
            brace_count += 1
        elif s[i] == '}':
            brace_count -= 1
        
        # If brace_count returns to 0, we've found the matching '}'
        if brace_count == 0:
            return ''.join(content)
        content.append(s[i])
    
    return s