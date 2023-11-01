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

import matplotlib.pyplot as plt

import pickle
from tqdm import tqdm 

import pyarrow
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
    data_dir = processed_dir if processed else data_raw_dir

    if processed:
        if 'tulu' in dataset:
            train_file = os.path.join(processed_dir, 'tulu', f'{dataset}.jsonl')
        elif 'flan2022' in dataset:
            train_file = os.path.join(processed_dir, 'flan2022', f'{dataset}_data.jsonl')
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


def get_dataset_token_lengths(dataset, model_name_or_path, tokenizer, inds=None):
    from open_instruct.finetune_trainer import encode_with_messages_format

    ds = get_dataset(dataset)
    if inds is not None: ds = ds.select(inds)
    encode_fn = partial(encode_with_messages_format, tokenizer=tokenizer, max_seq_length=2048)
    ds = ds.map(encode_fn, batched=False, num_proc=16)
    ds.set_format(type='np')

    def count_token_lengths(d):
        x = d['labels']
        input_len = x[x==-100].shape[0]
        output_len = x.shape[0] - input_len
        return {'input_len': input_len, 'output_len': output_len}

    ds = ds.map(count_token_lengths, num_proc=16)
    return {'input_len': ds['input_len'], 
            'output_len': ds['output_len']}


def get_lm_output(dataset, model_name, return_text_embedding=True):
    """`model_name` is name of directory under `model_outputs`. """
    save_path = os.path.join(lm_output_dir, model_name, f'{dataset}.pkl')
    with open(save_path, 'rb') as f:
        output = pickle.load(f)
    for k in output.keys():
        if k != 'text_emedding':
            output[k] = np.nan_to_num(output[k], nan=np.nanmean(output[k])) 
    if not return_text_embedding:
        for k in ['text_embedding', 'text_embeddings']:
            if k in output:
                del output[k]
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


def write_ds_to_file_for_reading(dataset, num_examples, output_path):

    output_dir = os.path.join(text_viz_path, os.path.dirname(output_path))
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, os.path.basename(output_path))

    random.seed(0)
    inds = random.sample(range(len(dataset)), num_examples)
    inds = sorted(inds)

    with open(full_path, 'w') as f:
        for idx in inds:
            example = dataset[idx]
            s = convert_example_to_str(idx, example)
            f.write(s)
            
    print(f'Writing {num_examples} examples to {full_path} completed!')