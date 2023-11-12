import os
import sys
import numpy as np
import time
import re
import random
import json
from functools import partial

from collections import defaultdict
import glob

import pickle
from tqdm import tqdm 

import pyarrow # need this before torch!
import torch
import transformers
from transformers import AutoTokenizer

from datasets import load_dataset


open_instruct_dir = '/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/'
assets_dir = '/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/assets/'
scripts_dir = os.path.join(open_instruct_dir, 'scripts')
data_raw_dir = os.path.join(open_instruct_dir, 'data', 'raw_train')
processed_dir = os.path.join(open_instruct_dir, 'data', 'processed')
data_inds_dir = os.path.join(scripts_dir, 'data_inds')
lm_output_dir = os.path.join(scripts_dir, 'model_outputs')
text_viz_path = os.path.join(scripts_dir, 'text_viz')
curriculum_dir = os.path.join(scripts_dir, 'curriculum')



def get_dataset_size(data_dir = 'data/processed'):
    """
        ```
        from note_pruning_analysis import get_dataset_size
        df = get_dataset_size()
        markdown_table = df.to_markdown(index=False)
        print(markdown_table)
        ```
    """
    import pandas as pd

    paths = glob.glob(os.path.join(data_dir, '*/*.jsonl'))
    ds_len = {}
    for path in paths:
        ds = load_dataset('json', 
                          data_files={'train': path}, 
                          split='train', 
                          cache_dir=os.path.dirname(path))
        basename = os.path.basename(path)
        if 'data' in basename:
            name = basename.split('_data')[0]
        else:
            name = basename.split('.')[0]
        ds_len[name] = len(ds)

    df = pd.DataFrame(list(ds_len.items()), columns=['dataset', 'length'])
    df = df.sort_values('dataset')
    df['length'] = df['length'].apply(lambda x: '{:,.0f}'.format(x))
    return df


    
def get_dataset(dataset, processed=True):
    if dataset.endswith(('jsonl', 'json')):
        train_file = dataset
    else:
        if processed:
            if 'tulu' in dataset:
                train_file = os.path.join(processed_dir, 'tulu', f'{dataset}.jsonl')
            elif 'flan2022' in dataset:
                train_file = os.path.join(processed_dir, 'flan2022', f'{dataset}_data.jsonl')
            elif 'ultrachat' in dataset:
                if dataset == 'ultrachat200k':
                    train_file = os.path.join(processed_dir, 'ultrachat', f'{dataset}_train_data.jsonl')
                else:
                    train_file = os.path.join(processed_dir, 'ultrachat', f'{dataset}_data.jsonl')
            else:
                train_file = os.path.join(processed_dir, dataset, f'{dataset}_data.jsonl')
        else:
            if dataset == 'lima':
                train_file = os.path.join(data_raw_dir, 'lima', 'train.jsonl')
            elif 'flan2022' in dataset:
                train_file = os.path.join(data_raw_dir, 'flan2022', f'{dataset}.jsonl')
            elif 'tulu' in dataset:
                train_file = os.path.join(data_raw_dir, 'tulu', f'{dataset}.jsonl')
            else:
                train_file = os.path.join(data_raw_dir, dataset)
    ds = load_dataset(
        'json', 
        data_files={'train': train_file}, 
        split='train', 
        cache_dir=os.path.dirname(train_file))
    return ds


def get_dataset_token_lengths(dataset, tokenizer, inds=None):
    from open_instruct.finetune_trainer import encode_with_messages_format
    if isinstance(dataset, str):
        ds = get_dataset(dataset)
    else:
        ds = dataset
    if inds is not None: ds = ds.select(inds)
    encode_fn = partial(encode_with_messages_format, tokenizer=tokenizer, max_seq_length=2048)
    ds = ds.map(encode_fn, batched=False, num_proc=64)
    ds.set_format(type='np')

    def count_token_lengths(d):
        x = d['labels']
        input_len = x[x==-100].shape[0]
        output_len = x.shape[0] - input_len
        return {'input_len': input_len, 'output_len': output_len}

    ds = ds.map(count_token_lengths, num_proc=64)
    return {'input_len': ds['input_len'], 
            'output_len': ds['output_len']}


def get_lm_output(dataset, model_name, return_text_embedding=True, fill_nan=True):
    """`model_name` is name of directory under `model_outputs`. """
    save_path = os.path.join(lm_output_dir, model_name, f'{dataset}.pkl')
    if dataset == 'ultrachat15' and os.path.isfile(save_path):
        ## concat ultrachat data shards.
        output = {}
        for i in range(10):
            with open(os.path.join(lm_output_dir, model_name, f'{dataset}_{i}.pkl'), 'rb') as f:
                output_i = pickle.load(f)
                for k, v in output_i.items():
                    if k not in output:
                        output[k] = []
                    output[k].append(v)
        output = {k: np.vstack(v) for k, v in output.items()}     
        with open(save_path, 'wb') as f:
            pickle.dump(output, f)
    else:
        with open(save_path, 'rb') as f:
            output = pickle.load(f)
    if not return_text_embedding:
        for k in ['text_embedding', 'text_embeddings', 'grad_rp_loraB']:
            if k in output:
                del output[k]
    if fill_nan:
        for k in [k for k, v in output.items() if v.squeeze().ndim==1]:
            output[k] = np.nan_to_num(output[k], nan=np.nanmean(output[k])) 
    return output


def get_sorted_inds(dataset, model_name, sort_by):
    save_path = os.path.join(data_inds_dir, model_name, dataset, f'{sort_by}.pkl')
    with open(save_path, 'rb') as f:
        output = pickle.load(f)
    return output


def get_prune_results(path):
    """Note `S` in pkl is sorted - here we convert `S` to before sorting.
    """
    pkl_filename = os.path.basename(path)
    sort_by = os.path.splitext(pkl_filename)[0]
    dataset = os.path.basename(os.path.dirname(path))
    model_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    output = {
        'model_name': model_name, 
        'dataset': dataset, 
        'sort_by': sort_by,
        'pkl_path': path}
    sorted_inds = get_sorted_inds(dataset, model_name, sort_by)
    sorted_inds = {k: np.array(v) if v else None for k, v in sorted_inds.items()}
    if 'S' in sorted_inds and 'inds' in sorted_inds:
        sorted_inds['S'] = sorted_inds['S'][np.argsort(sorted_inds['inds'])]
    output.update(sorted_inds)
    return output



def compute_correlations(xs, ys, corr_type='pearsonr'):
    xs = xs.squeeze()
    ys = ys.squeeze()
    import scipy
    if corr_type == 'pearsonr':
        return scipy.stats.pearsonr(xs, ys).statistic
    elif corr_type == 'spearmanr':
        return scipy.stats.spearmanr(xs, ys).statistic
    else:
        raise ValueError(f"Invalid corr_type={corr_type}")


def convert_example_to_str(idx, example):
    import json

    metadata = {k: v for k, v in example.items() if k!='messages'}
    messages = example['messages']
    metadata['idx'] = idx
    metadata['n_turns'] = len(messages)

    s = ''
    s += 'metadata: ' + json.dumps(metadata, indent=4) + '\n'
    for message in messages:
        s += '\n'
        s += '='*15 + '\n'
        s += f"= {message['role'].upper():11} =\n"
        s += '='*15 + '\n'
        s += message['content'] + '\n'
    s += '\n'*15
    
    return s


def write_ds_to_file_for_reading(dataset, output_path, num_examples=None):
    """
    from note_pruning_analysis import write_ds_to_file_for_reading
    write_ds_to_file_for_reading(ds, 'text_viz/data/processed/lima.txt', num_examples=200)
    """

    # output_dir = os.path.join(text_viz_path, os.path.dirname(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # full_path = os.path.join(output_dir, os.path.basename(output_path))

    if num_examples is None:
        num_examples = len(dataset)

    random.seed(0)
    inds = random.sample(range(len(dataset)), num_examples)
    inds = sorted(inds)

    with open(output_path, 'w') as f:
        for idx in inds:
            example = dataset[idx]
            s = convert_example_to_str(idx, example)
            f.write(s)
            
    print(f'Writing {num_examples} examples to {output_path} completed!')



def sample_indices_given_scores(scores, portion):
    """Given `scores`, sample indices from corresponding regions of indices
        indicated by `portion`.
        Used to generate a few examples for text visualization.
        """
    np.random.seed(0)

    match = re.search(r'num=([^_]+)', portion)
    num_printed = int(match.group(1))

    if portion.startswith('sorted'):
        inds_sorted = np.argsort(scores)
        inds_to_inds = np.arange(len(scores))

        if portion.startswith(('sorted_beg', 'sorted_end')):
            if portion.startswith('sorted_end'):
                inds_to_inds = inds_to_inds[::-1]
            inds_to_inds = inds_to_inds[:num_printed]
        elif portion.startswith('sorted_partition'):
            match = re.search(r'partition=([\d.:]+)', portion)
            part, npart = match.group(1).split(':')
            part, npart = int(part)-1, int(npart) # make 1 ordered, e.g., 1:10, ..., 10:10
            def split_section_sizes(n, sections):
                """Get size of sections for an array of size `n` into `sections` sections. """
                L = []
                for _ in range(sections-1):
                    L.append(int(n//sections))
                L.append(int(n-np.sum(L)))
                return L
            indices_or_sections = np.cumsum(split_section_sizes(len(inds_to_inds), npart)[:-1])
            inds_to_inds_split = np.split(inds_to_inds, indices_or_sections)
            assert(len(inds_to_inds_split) == npart)
            assert(np.sum([x.shape[0] for x in inds_to_inds_split]) == len(inds_to_inds))
            inds_to_inds = inds_to_inds_split[part]

            inds_to_inds = np.random.choice(
                inds_to_inds, min(num_printed, len(inds_to_inds)), replace=False)
            inds_to_inds = np.sort(inds_to_inds)
        elif portion.startswith('sorted_random'):
            inds_to_inds = np.random.choice(
                inds_to_inds, min(num_printed, len(inds_to_inds)), replace=False)
        else:
            raise ValueError(f'portion={portion} not supported')

        inds_to_inds = np.sort(inds_to_inds)    
        inds = inds_sorted[inds_to_inds]
    else:
        raise ValueError(f'portion={portion} not supported')
        
    return inds



def viz_tokenizer_outputs(outputs, tokenizer=None):
    import pandas as pd
    outputs = {k: v for k, v in outputs.items() if isinstance(v, (list, torch.Tensor, np.ndarray))}
    outputs = {k: v.squeeze().tolist() if isinstance(v, torch.Tensor) else v 
               for k, v in outputs.items()}
    outputs = {k: v[0] if isinstance(v, list) and isinstance(v[0], list) and len(v)==1 else v
               for k, v in outputs.items()}
    print(outputs)
    if tokenizer is not None:
        toks = tokenizer.convert_ids_to_tokens(outputs['input_ids'])
        data = {'toks': toks}
    else:
        data = {}
    data.update(outputs)
    l = max(len(v) for v in data.values())
    for k, v in data.items():
        if len(v) < l:
            data[k] = [np.nan]*(l-len(v)) + v
    df = pd.DataFrame(data, index=range(len(data['input_ids'])))
    return df


def encode_just_one_role(example, tokenizer, max_seq_length, encode_fn_type):
    messages = example['messages']
    assert(len(messages) == 2)
    
    if encode_fn_type == 'input':
        text = messages[0]['content']
    elif encode_fn_type == 'output':
        text = messages[1]['content']
    else:
        raise ValueError(f'encode_fn_type={encode_fn_type} not supported.')
    tokenized_example = tokenizer(
        text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    attention_mask = tokenized_example.attention_mask
    labels = input_ids.clone()
    
    return {'text': text,
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten()}