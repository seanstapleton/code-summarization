from preprocess_utilities import *
import ast
from tqdm import tqdm
import multiprocessing as mp
import sys

def get_and_format_examples(examples_path, labels_path):
    code = get_file(examples_path).split('\n')
    labels = get_file(labels_path, latin=True).split('\n')

    ### Remove quotes and new lines from labels 
    labels = [label.replace('DCNL', '').replace('`', '').replace('\'', '') for label in labels]

    ### convert single line examples to indented code
    code_expanded = [form_code_sample(example) for example in code]

    ### Ignore all examples with > 36 lines of code
    code_limited = [example for example in code_expanded if example.count('\n') <= 36]
    labels_limited = [labels[idx] for idx in range(len(code_expanded)) if code_expanded[idx].count('\n') <= 36]
    
    return code_limited, labels_limited

def preprocess_subset(codeset, label_set, floor_number, ceil_number, output_path):
    messups = []
    idx = 0
    for example,label in zip(codeset, label_set):
        try:
            paths = convert_code_to_ast_paths(example)
            paths = list(map(lambda x: x.replace('\n', 'DCNL').replace(' ', 'DCSP').replace('\r', 'DCRC'), paths))
            paths = ' '.join(paths)
            label = label.replace('  ', ' ')
            formed_str = label.replace(' ', '|') + ' ' + paths
            append_to_file(output_path + '/' + 'preprocessed_train_' + str(ceil_number) + '.txt', formed_str)
        except:
            messups.append(floor_number + idx)
        idx += 1
    return messups

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('You must include paths to your training set and labels')
        sys.exit()
    output_path = sys.argv[3] if len(sys.argv) > 3 else ''
    examples_path,labels_path = sys.argv[1],sys.argv[2]
    code_limited,labels_limited = get_and_format_examples(examples_path, labels_path)

    ### Run preprocessing in batches of 10 w/ 2,000 examples at a time
    pool = mp.Pool(processes=10)
    results = []
    batch_size = 2000
    for i in range(len(code_limited)//batch_size + 1):
        floor = i*batch_size
        ceil = min(len(code_limited), floor + batch_size) 
        codeset = code_limited[floor:ceil]
        labelset = labels_limited[floor:ceil]
        results.append(pool.apply_async(preprocess_subset, args=(codeset,labelset,floor,ceil,output_path)))

    output = [r.get() for r in results]
    messups = [messup for elt in output for messup in elt]
    print('Finished preprocessing. Examples ' +', '.join(str(m) for m in messups) + ' were not able to be processed.')
