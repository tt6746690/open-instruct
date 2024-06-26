import os
import sys
import time
import re
import random
import json
from functools import partial

from collections import defaultdict
import glob

import scipy
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm 
import matplotlib.pyplot as plt

import pyarrow # need this before torch!
import torch
import transformers
from transformers import AutoTokenizer

from datasets import load_dataset


open_instruct_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
mitibm_dir = os.path.dirname(os.path.dirname(open_instruct_dir))
cache_dir = os.path.join(mitibm_dir, 'cache')
assets_dir = os.path.join(mitibm_dir, 'assets')
scripts_dir = os.path.join(open_instruct_dir, 'scripts')
data_raw_dir = os.path.join(open_instruct_dir, 'data', 'raw_train')
processed_dir = os.path.join(open_instruct_dir, 'data', 'processed')
data_inds_dir = os.path.join(scripts_dir, 'data_inds')
lm_output_dir = os.path.join(scripts_dir, 'model_outputs')
text_viz_path = os.path.join(scripts_dir, 'text_viz')
curriculum_dir = os.path.join(scripts_dir, 'curriculum')



def save_lm_output_for_50k_subset(dataset, dataset_50k, num_proc=64):
    """Already have model output for full dataset, save model output for 50k subset.
    
        ```
        from note_pruning_analysis import save_lm_output_for_50k_subset
        for dataset, dataset_50k in [
            ('ultrachat200kv2', 'ultrachat50k'),
            ('sharegptv2', 'sharegpt50k'),
            ('wizardlmv2', 'wizardlm50k'),
        ]:
            save_lm_output_for_50k_subset(dataset, dataset_50k)
        ```
    
    """
    from open_instruct.finetune_trainer import encode_with_messages_format
    
    ds = get_dataset(dataset)
    ds50k = get_dataset(dataset_50k)

    tokenizer_name_or_path = get_tokenizer_name_or_path('llama-7b')
    tokenizer = get_fast_tokenizer(tokenizer_name_or_path)
    encode_fn = partial(encode_with_messages_format, tokenizer=tokenizer, max_seq_length=10_000)
    def fn(example):
        output = encode_fn(example)
        return {'text': tokenizer.decode(output['input_ids'])}

    ds = ds.map(fn, batched=False, num_proc=num_proc)
    ds50k = ds50k.map(fn, batched=False, num_proc=num_proc)

    df = ds.to_pandas()
    df = df.reset_index()
    df50k = ds50k.to_pandas()
    df50k = df50k.reset_index()

    if df['text'].nunique() != len(df):
        raise ValueError(f'`df` has duplicates text nunique={df["text"].nunique()} len={len(df)}')
    if df50k['text'].nunique() != len(df50k):
        raise ValueError(f'`df50k` has duplicates text nunique={df50k["text"].nunique()} len={len(df50k)}')
    
    dfm = pd.merge(df, df50k, on='text', suffixes=(['_full', '_50k']), how='inner')
    if len(dfm) != len(df50k):
        raise ValueError(f'len(dfm)={len(dfm)} != len(df50k)={len(df50k)}')

    dfinds = dfm.sort_values(by='index_50k')[['index_full', 'index_50k']]
    # index into original dataset that gives 50k subset's ordering
    inds = dfinds['index_full'].tolist()

    for md, encode_fn_type in [
        ('llama7br512p4096', 'sft'), 
        ('mpnet', 'input'),
    ]:
        model_name = get_full_model_name(md)
        d = get_lm_output(dataset, model_name, encode_fn_type=encode_fn_type, return_text_embedding=True, fill_nan=False)
        assert(all( v.shape[0]==len(ds) for k,v in d.items() ))
        dout = {k: v[inds] for k, v in d.items()}
        assert(all( v.shape[0]==len(ds50k) for k,v in dout.items() ))

        save_dir = (f"model_outputs/{encode_fn_type}/{model_name}")
        save_path = os.path.join(save_dir, f'{dataset_50k}.pkl')
        if os.path.isfile(save_path):
            print(f'save_path={save_path} already exist! dont overwrite.')
        else:
            with open(save_path, 'wb') as f:
                pickle.dump(dout, f, protocol=pickle.HIGHEST_PROTOCOL)
        din = get_lm_output(dataset_50k, model_name, encode_fn_type=encode_fn_type, return_text_embedding=True, fill_nan=False)
        for k in din.keys():
            assert((din[k]==dout[k]).all())



def save_to_pickle(save_path, output, verbose=True):
    if verbose and 'inds' in output:
        print(f'save inds (length = {len(output["inds"])}) to {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)


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
                          split='train')
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


dataset_with_multiple_version = [
    'alpagasus',
    'baize',
    'code_alpaca',
    'cot',
    'dolly',
    'flan_v2',
    'flan2022',
    'gpt4_alpaca',
    'hh_rlhf',
    'lima',
    'oasst',
    'open_orca',
    'openai_sum',
    'self_instruct',
    'sharegpt',
    'shp',
    'stanford_alpaca',
    'starcoder',
    'super_ni',
    'tulu_v1',
    'tulu_v2',
    'tulu',
    'ultrachat',
    'ultrafeedback',
    'unnatural_instructions',
    'wizardlm',
    'mix',
]

dataset_with_train_val_split = [
    'ultrachat200k',
    'ultrachat50k',
]


def get_dataset_train_file(dataset, processed=True):
    if dataset.endswith(('jsonl', 'json')):
        train_file = dataset
    else:
        if processed:
            has_multiple_versions = [x in dataset for x in dataset_with_multiple_version]
            if any(has_multiple_versions):
                dataset_dir = dataset_dir = dataset_with_multiple_version[has_multiple_versions.index(True)]
                if any(dataset.startswith(x) for x in dataset_with_train_val_split):
                    dataset += '_train'
                if 'starcoder' not in dataset:
                    dataset += '_data'
                train_file = os.path.join(processed_dir, dataset_dir, f'{dataset}.jsonl')
            else:
                train_file = os.path.join(processed_dir, dataset, f'{dataset}_data.jsonl')
        else:
            if dataset in ['tulu_v1', 'tulu_v2']:
                train_file = os.path.join(processed_dir, dataset, f'{dataset}_data.jsonl')
            elif 'tulu' in dataset:
                train_file = os.path.join(data_raw_dir, 'tulu', f'{dataset}.jsonl')
            elif dataset == 'lima':
                train_file = os.path.join(data_raw_dir, 'lima', 'train.jsonl')
            elif 'flan2022' in dataset:
                train_file = os.path.join(data_raw_dir, 'flan2022', f'{dataset}.jsonl')
            elif 'starcoder' in dataset:
                train_file = os.path.join(data_raw_dir, 'starcoder', f'{dataset}.json')
            else:
                train_file = os.path.join(data_raw_dir, dataset)
    return train_file

    
def get_dataset(dataset, processed=True):
    """Get dataset from `data/processed` or `data/raw_train`. """
    train_file = get_dataset_train_file(dataset, processed=processed)
    ds = load_dataset(
        'json', 
        data_files={'train': train_file}, 
        split='train')
    return ds



def combine_existing_datasets_processed_texts_and_lm_output(mix_name, mix_datasets, data_processed_dir, model_output_dir):
    """combine jsonl text files & combine model_output for llama7b

        from note_pruning_analysis import combine_existing_datasets_processed_texts_and_lm_output
        model_output_dir = '/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/model_outputs/sft/llama-7b+lora:r=512:a=11585+proj=4096'
        data_processed_dir = 'data/processed'    
        mix_name = 'mix_all50k'
        mix_datasets = ['dolly', 'flan_v250k', 'stanford_alpaca50k', 'oasst2', 'wizardlm50k', 'sharegpt50k', 'ultrachat50k'] # order matters!
        combine_existing_datasets_processed_texts_and_lm_output(mix_name, mix_datasets, data_processed_dir, model_output_dir)
    """
    from note_pruning_analysis import get_dataset_train_file
    import numpy as np
    import os, json, pickle

    ## processed texts
    counts = 0
    save_path = os.path.join(data_processed_dir, 'mix', f'{mix_name}_data.jsonl')
    with open(save_path, 'w') as fout:
        for dataset in mix_datasets:
            json_file = get_dataset_train_file(dataset)
            print(json_file)
            with open(json_file, 'r') as fin:
                for line in fin:
                    example = json.loads(line)
                    fout.write(json.dumps({
                        'dataset': example['dataset'],
                        'id': example['id'],
                        'messages': [{'role': x['role'], 'content': x['content']} for x in example['messages']],
                    }) + "\n")
                    counts += 1
    print(f'save {counts} examples to {save_path}')

    ## lm outputs
    output_list = []
    for dataset in mix_datasets:
        save_path = os.path.join(model_output_dir, f'{dataset}.pkl')
        with open(save_path, 'rb') as f:
            output = pickle.load(f)
        output_list.append(output)

    output = {}
    for k in output_list[0].keys():
        output[k] = np.vstack([x[k] for x in output_list])

    save_path = os.path.join(model_output_dir, f'{mix_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Combining lm_outputs for {mix_name}')
    print(f'Save output={[(k, v.shape) for k, v in output.items()]} to {save_path}')




def encode_with_messages_format_dpo(example, tokenizer, max_seq_length):
    '''Taken from `dpo_tune.py`.
    Here we assume each example has a rejected and chosen field, both of which are a list of messages.
    Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    We assume only the last message is different, and the prompt is contained in the list of messages.
    '''
    chosen_messages = example['chosen']
    rejected_messages = example['rejected']
    if len(chosen_messages) == 0:
        raise ValueError('chosen messages field is empty.')
    if len(rejected_messages) == 0:
        raise ValueError('rejected messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    def encode_messages(messages):
        example_text = _concat_messages(messages).strip()
        tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()

        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
                else:
                    messages_so_far = _concat_messages(messages[:message_idx+1])
                message_end_idx = tokenizer(
                    messages_so_far,
                    return_tensors='pt', 
                    max_length=max_seq_length, 
                    truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = -100
                
                if message_end_idx >= max_seq_length:
                    break

        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }
    chosen_encoded = encode_messages(chosen_messages)
    rejected_encoded = encode_messages(rejected_messages)
    # labels are useful for working out where the loss is valid.
    return {
        'chosen_input_ids': chosen_encoded['input_ids'],
        'chosen_labels': chosen_encoded['labels'],
        'chosen_attention_mask': chosen_encoded['attention_mask'],
        'rejected_input_ids': rejected_encoded['input_ids'],
        'rejected_labels': rejected_encoded['labels'],
        'rejected_attention_mask': rejected_encoded['attention_mask'],
    }


def get_dataset_token_lengths(dataset, tokenizer, inds=None, num_proc=128, max_seq_length=10_000):
    """Get token lengths for dataset when `encode_with_messages_format` is used.
        Change `max_seq_length` to a very large number to get an idea if any example is truncated. 
        
        Takes into account of both sft/preference dataset
            - sft dataset: requires `dataset` to have `messages` field.
            - preference dataset: requires `dataset` to have `chosen` and `rejected` fields, which are themselves messages.
        """
    if isinstance(dataset, str):
        ds = get_dataset(dataset)
    else: 
        ds = dataset
    if inds is not None: ds = ds.select(inds)

    is_preference_data = True if ('chosen' in ds.column_names and 'rejected' in ds.column_names) else False

    if is_preference_data:
        encode_fn = partial(encode_with_messages_format_dpo, tokenizer=tokenizer, max_seq_length=max_seq_length)
    else:
        from open_instruct.finetune_trainer import encode_with_messages_format
        encode_fn = partial(encode_with_messages_format, tokenizer=tokenizer, max_seq_length=max_seq_length)

    ds = ds.map(encode_fn, batched=False, num_proc=num_proc)
    ds.set_format(type='np')
    
    if is_preference_data:
        def count_token_lengths(d):
            output = {}
            for k in ['chosen', 'rejected']:
                x = d[f'{k}_labels']
                numtoks_input = x[x==-100].shape[0]
                numtoks_output = x.shape[0] - numtoks_input
                output.update({
                    f'{k}_numtoks_input': numtoks_input, 
                    f'{k}_numtoks_output': numtoks_output,
                    f'{k}_numtoks_total': numtoks_input + numtoks_output
                })
            # take max of chosen/rejected numtoks
            output = {
                'numtoks_input': max(output['chosen_numtoks_input'], output['rejected_numtoks_input']),
                'numtoks_output': max(output['chosen_numtoks_output'], output['rejected_numtoks_output']),
                'numtoks_total': max(output['chosen_numtoks_total'], output['rejected_numtoks_total']),
            }
            return output
    else:
        def count_token_lengths(d):
            x = d['labels']
            numtoks_input = x[x==-100].shape[0]
            numtoks_output = x.shape[0] - numtoks_input
            return {'numtoks_input': numtoks_input, 
                    'numtoks_output': numtoks_output,
                    'numtoks_total': numtoks_input + numtoks_output}

    ds = ds.map(count_token_lengths, num_proc=num_proc)
    for k in ds.column_names:
        for k in ['input_ids', 'labels', 'attention_mask']:
            if k in ds.column_names:
                ds = ds.remove_columns(k)
    return ds


def get_lm_output(dataset, model_name, encode_fn_type='sft', return_text_embedding=True, fill_nan=True):
    """`model_name` is name of directory under `model_outputs`. """
    save_path = os.path.join(lm_output_dir, encode_fn_type, model_name, f'{dataset}.pkl')
    print(save_path)

    if os.path.isfile(save_path):
        with open(save_path, 'rb') as f:
            output = pickle.load(f)
    else:
        if dataset == 'ultrachat15':
            ## concat ultrachat data shards.
            output = {}
            for i in range(10):
                save_path_shard = os.path.join(lm_output_dir, encode_fn_type, model_name, f'{dataset}_{i}.pkl')
                with open(save_path_shard, 'rb') as f:
                    output_i = pickle.load(f)
                    for k, v in output_i.items():
                        if k not in output:
                            output[k] = []
                        output[k].append(v)
            output = {k: np.vstack(v) for k, v in output.items()}     
            with open(save_path, 'wb') as f:
                pickle.dump(output, f)
        else:
            raise ValueError(f'save_path={save_path} does not exist, and cannot assemble shards for dataset={dataset}')
    if not return_text_embedding:
        for k in ['text_embedding', 'text_embeddings', 'grad_rp_loraB']:
            if k in output:
                del output[k]
    if fill_nan:
        for k in [k for k, v in output.items() if v.squeeze().ndim==1]:
            output[k] = np.nan_to_num(output[k], nan=np.nanmean(output[k])) 
    return output



def random_uniform_hypersphere_surface(N, D):
    v = np.random.randn(N, D)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v



def generate_randsphere_model_output(dataset_or_N, D, model_name='randspherep4096', encode_fn_type='sft', embed_type='grad_rp_loraB', seed=0):
    """Generate uniformlys sampled points on hypersphere surface
        to `model_outputs` under `randsphere`
        
        ## as reference dataset to compare between datasets.
        ```
        from note_pruning_analysis import generate_randsphere_model_output
        dataset_list = [
            'flan_v2', # ~100k data
        ]
        for dataset in dataset_list:
            generate_randsphere_model_output(dataset, 4096, model_name='randspherep4096', encode_fn_type='sft', embed_type='grad_rp_loraB')
            generate_randsphere_model_output(dataset, 768, model_name='randspherep768', encode_fn_type='input', embed_type='text_embedding')
        ```

        ## for measuring randomness from seed
        ```
        from note_pruning_analysis import generate_randsphere_model_output
        dataset_list = [
            'flan_v2', # ~100k data
        ]

        for dataset in dataset_list:
            for seed in [0,1,2]:
                generate_randsphere_model_output(dataset, 4096, model_name=f'randspherep4096s{seed}', encode_fn_type='sft', embed_type='grad_rp_loraB', seed=seed)
                generate_randsphere_model_output(dataset, 768, model_name=f'randspherep768s{seed}', encode_fn_type='input', embed_type='text_embedding', seed=seed)
        ```
        
        ## for computing time efficiency of DPP MAP
        ```
        Ns = [10_000, 50_000, 100_000]
        Ds = [256, 1024, 4096]
        
        for N in Ns:
            for D in Ds:
                print(N, D)
                generate_randsphere_model_output(N, D, model_name=f'randspherep{D}', encode_fn_type='sft', embed_type='grad_rp_loraB')
        ```
    """
    if isinstance(dataset_or_N, int):
        N = dataset_or_N
        save_filename = f'Ref_N={N}.pkl'
    else:
        dataset = dataset_or_N
        ds = get_dataset(dataset)
        N = len(ds)
        save_filename = f'{dataset}.pkl'

    np.random.seed(seed)
    X = random_uniform_hypersphere_surface(N, D)
    X = X.astype(np.float32)

    save_dir = os.path.join(lm_output_dir, encode_fn_type, model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_filename)
    with open(save_path, 'wb') as f:
        output = {
            embed_type: X,
        }
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)



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



def get_clustering_results(dataset, model_name, clustering_fn, encode_fn_type='input', return_data=False):
    
    save_dir = os.path.join(scripts_dir, 'clustering', encode_fn_type, model_name, dataset, clustering_fn)

    d = {}
    with open(os.path.join(save_dir, 'info.json'), 'r') as f:
        d['info'] = json.load(f)

    if return_data:
        with open(os.path.join(save_dir, 'data.pkl'), 'rb') as f:
            d['data'] = pickle.load(f)
    return d



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

    if 'chosen' in example:
        messages_key = 'chosen'
    elif 'messages' in example:
        messages_key = 'messages'
    else:
        raise ValueError(f'Cannot find `messages` or `chosen` in example')

    metadata = {k: v for k, v in example.items() if k!='messages' and not isinstance(v, list)}
    messages = example[messages_key]
    metadata['print_idx'] = idx
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



def sample_indices_given_scores(scores, portion, num_printed=100):
    """Given `scores`, sample indices from corresponding regions of indices
        indicated by `portion`.
        Used to generate a few examples for text visualization.
        """
    np.random.seed(0)

    if portion.startswith('sorted'):
        inds_sorted = np.argsort(scores)
        inds_to_inds = np.arange(len(inds_sorted))

        # take first `M` examples to sample scores
        match = re.search(r'sorted(\d+)_', portion)
        if match:
            inds_to_inds = inds_to_inds[:int(match.group(1))]
            portion = re.sub(r'sorted(\d+)_', 'sorted_', portion)

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


def chat_completion_openai(
    messages,
    model='gpt-3.5-turbo-1106',
    temperature=0,
    max_tokens=256,
    ):

    import openai
    output = {}
    for _ in range(16):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output['completion'] = response['choices'][0]['message']['content']
            output['usage'] = {'prompt_tokens': response['usage']['prompt_tokens'],
                               'completion_tokens': response['usage']['completion_tokens'],
                               'total_tokens': response['usage']['total_tokens'],}
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(10)

    return response



def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.

    Parameters:
    - d: The input dictionary.
    - parent_key: Used for recursion to keep track of the parent keys.
    - sep: Separator used to concatenate keys.

    Returns:
    A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)




def save_text_viz_for_curriculum(path):
    """Saves some text snippets for visualization for a curriculum. """
    from note_curriculum import get_curriculum_scores
    num_examples = 100
    portion_list = [ 
        f'sorted_beg',
        f'sorted_end',
        f'sorted_random',
        f'sorted_partition=1:10',
        f'sorted_partition=10:10',
        f'sorted1000_beg',
        f'sorted1000_end',
        f'sorted1000_random',
        f'sorted10000_beg',
        f'sorted10000_end',
        f'sorted10000_random',
    ]
    output = get_curriculum_scores(path)
    scores = output['scores']
    dataset = output['dataset']
    
    ds = get_dataset(dataset)
    for portion in portion_list:
        # sample subsets for printing
        inds = sample_indices_given_scores(scores, portion, num_printed=100)
        def add_score_fn(example, idx):
            example.update({'score': scores[inds[idx]],
                            'ind': inds[idx],
                            'ind_pct': inds[idx]/len(ds)})
            return example
        ds_subset = ds.select(inds).map(add_score_fn, with_indices=True, keep_in_memory=True)
        write_ds_to_file_for_reading(
            dataset=ds_subset, 
            output_path=os.path.join('text_viz', os.path.dirname(path), f"{portion}.txt"),
            num_examples=num_examples)



def get_tokenizer_name_or_path(model_name):

    if model_name.startswith('llama-7b'):
        tokenizer_name_or_path = os.path.join(
            scripts_dir, 'results', 'baselines', 'huggyllama/llama-7b')
    elif model_name.startswith('codellama-7b'):
        tokenizer_name_or_path = os.path.join(
            scripts_dir, 'results', 'baselines', 'codellama/CodeLlama-7b-hf')
    elif model_name.startswith('mistral-7b'):
        tokenizer_name_or_path = os.path.join(
            scripts_dir, 'results', 'baselines', 'mistralai/Mistral-7B-v0.1')
    elif model_name.startswith('llama2-7b'):
        tokenizer_name_or_path = os.path.join(
            scripts_dir, 'results', 'baselines', 'NousResearch/Llama-2-7b-hf')
    else:
        raise ValueError(f'Cannot find corresponding `tokenizer_name_or_path` for model_name: {model_name}')
    
    return tokenizer_name_or_path
    
    


def get_fast_tokenizer(model_name_or_path):
    """get fast tokenizer that tokenize special tokens properly. """
    from transformers import AddedToken, LlamaTokenizerFast

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True)
    num_added_tokens = tokenizer.add_special_tokens({
        "bos_token": AddedToken("<s>", normalized=False, special=True),
        "eos_token": AddedToken("</s>", normalized=False, special=True),
        "unk_token": AddedToken("<unk>", normalized=False, special=True),
        "pad_token": AddedToken("<pad>", normalized=False, special=True),
    })
    if isinstance(tokenizer, LlamaTokenizerFast):
        if os.path.isdir(model_name_or_path):
            tmp_tok_path = os.path.join(
                os.path.dirname(model_name_or_path),
                os.path.basename(model_name_or_path)+'_fixtok')
            if not os.path.isdir(tmp_tok_path):
                raise ValueError(f'Not valid fixtok path: {tmp_tok_path}')
        else:
            from secrets import token_hex
            tmp_tok_path = f'/tmp/wpq_tok_{token_hex(16)}'
            tokenizer.save_pretrained(tmp_tok_path)
        tokenizer = AutoTokenizer.from_pretrained(tmp_tok_path, use_fast=True)
    for s, s_tokenized in [
        ("Hi<s>Hey</s>sir<unk>what<pad><pad>", 
        ['▁Hi', '<s>', '▁Hey', '</s>', '▁sir', '<unk>', '▁what', '<pad>', '<pad>']),
    ]:
        assert(tokenizer.tokenize(s, add_special_tokens=False)==s_tokenized)
        
    return tokenizer




def filter_examples_by_numtoks(examples, tokenizer_name='llama', num_proc=64, max_seq_length=2048):
    ## given a list of examples, filter them so that when applied 
    # the chat template, will be shorter than `max_seq_length`.
    from datasets import Dataset
    
    if tokenizer_name.startswith('llama'):
        tokenizer_name_or_path = get_tokenizer_name_or_path('llama-7b')
    elif tokenizer_name.startswith('codellama'):
        tokenizer_name_or_path = get_tokenizer_name_or_path('codellama-7b')
    elif tokenizer_name.startswith('mistral'):
        tokenizer_name_or_path = get_tokenizer_name_or_path('mistral-7b')
    else:
        raise ValueError(f'Unknown tokenizer_name={tokenizer_name}')

    tokenizer = get_fast_tokenizer(tokenizer_name_or_path)
    ds = Dataset.from_list(examples)

    num_examples = len(ds)

    if 'messages' not in ds.column_names and 'conversations' in ds.column_names:
        def convert_conversations_to_messages_fn(example):
            messages = []
            for x in example['conversations']:
                if x['from'] == 'human':
                    messages.append({'role': 'user', 'content': x['value']})
                elif x['from'] == 'gpt':
                    messages.append({'role': 'assistant', 'content': x['value']})
                elif x['from'] == 'system':
                    pass
                else:
                    raise ValueError(f"Unknown message sender: {x['from']}")
            return {'messages': messages}
        ds = ds.map(convert_conversations_to_messages_fn, num_proc=num_proc)
        if 'conversations' in ds.column_names:
            ds = ds.remove_columns('conversations')

    ds = get_dataset_token_lengths(ds, tokenizer, max_seq_length=10_000)

    ds = ds.map(lambda x, idx: {'idx': idx}, with_indices=True, num_proc=num_proc)
    dsf = ds.filter(lambda x: x['numtoks_total'] <= max_seq_length, num_proc=num_proc)

    inds = dsf['idx']
    examples = [examples[i] for i in inds]

    print(f'[filter_examples_by_numtoks] Filter to <={max_seq_length}, the number of examples {num_examples} -> {len(examples)} examples')
    
    return examples



def filter_json_by_numtoks(jsonl_path, tokenizer_name='llama7b', max_seq_length=2048):
    """Filter dataset specified by `jsonl_path`,
        so that each example when applied tulu's chat template has numtoks < `max_seq_length`.
        Write the dataset to `jsonl_path`. """
    examples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    examples = filter_examples_by_numtoks(
        examples,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length,
        num_proc=64)

    with open(jsonl_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


def plt_dataset_numtoks(dataset, tokenizer_name_or_path):
    """Plot the number of tokens in a dataset under `data/processed/`. """

    # convert path -> shorter names
    tokenizer_name = os.path.basename(tokenizer_name_or_path)
    dataset_name = os.path.basename(dataset).split('.jsonl')[0]
    if dataset_name.endswith('_data'):
        dataset_name = dataset_name[:-5]
             
    os.makedirs(os.path.join(open_instruct_dir, 'data', 'stats', 'numtoks', tokenizer_name), exist_ok=True)

    tokenizer = get_fast_tokenizer(tokenizer_name_or_path)
    ds = get_dataset(dataset)
    ds = get_dataset_token_lengths(ds, tokenizer, max_seq_length=100_000)

    fig, axs = plt.subplots(1, 3,figsize=(12,4), sharex=True, sharey=True)
    for axi, k in enumerate(['numtoks_input', 
                             'numtoks_output', 
                             'numtoks_total']):
        ax = axs[axi]
        ys = np.array(ds[k])
        ax.hist(ys)
        ax.set_title(k)
        ax.grid()
        for k, v in {
            'mean': .5,
            '99% pct': .99,
            'max': 1,
        }.items():
            s = int(np.quantile(ys, v))
            ax.axvline(x=s, color='red', linestyle='dashed', linewidth=2, label=f'{k}={s}')
        ax.legend()

    fig.suptitle(f'{dataset}:{tokenizer_name}')
    fig.tight_layout()
    save_path = os.path.join(open_instruct_dir, 'data', 'stats', 'numtoks', tokenizer_name, f'{dataset_name}.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    
    return fig, axs


def count_repeating_substrings(s, length=3):
    substring_counts = {}

    def count_substrings(s, length):
        counts = {}
        for i in range(len(s) - length + 1):
            substring = s[i:i+length]
            counts[substring] = counts.get(substring, 0) + 1
        return counts

    counts = count_substrings(s, length)
    for substring, count in counts.items():
        if count > 1:
            substring_counts[substring] = count

    substring_counts = sorted(substring_counts.items(), key=lambda x: x[1], reverse=True)
    substring_counts = {k: v for k, v in substring_counts}

    return substring_counts


def compute_alpacaeval_numtoks():
    """mean, median: 127.84844720496895 95.0 """
    import datasets
    from note_pruning_analysis import get_tokenizer_name_or_path, get_fast_tokenizer
    import numpy as np

    tokenizer_name_or_path = get_tokenizer_name_or_path('llama-7b')
    tokenizer = get_fast_tokenizer(tokenizer_name_or_path)
    alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    def fn(example):
        output = tokenizer(example['output'])
        return {'len': len(output['input_ids'])}
    alpaca_eval_data = alpaca_eval_data.map(fn, num_proc=8)
    L = np.array(alpaca_eval_data['len'])
    print(f"alpacaeval_output_lens={alpaca_eval_data['len']}")
    print(np.mean(L), np.median(L))


def compute_win_rate(preference):
    preference = np.array([x for x in preference if x is not None])
    n_wins = np.sum(preference == 2.)
    n_wins_base = np.sum(preference == 1.)
    n_draws = np.sum(preference == 0.)
    n_total = n_wins + n_wins_base + n_draws
    win_rate = (n_wins/n_total + .5*n_draws/n_total)
    return win_rate



alpacaeval_output_lens = np.array([90, 123, 114, 105, 163, 174, 117, 58, 85, 452, 95, 125, 288, 47, 87, 119, 131, 109, 42, 148, 40, 48, 56, 160, 15, 67, 34, 111, 86, 105, 268, 181, 127, 17, 18, 56, 197, 61, 201, 83, 20, 145, 93, 50, 72, 70, 78, 226, 31, 150, 16, 21, 212, 190, 222, 84, 200, 290, 182, 80, 456, 183, 140, 146, 363, 169, 120, 458, 133, 53, 35, 16, 118, 99, 86, 127, 248, 380, 77, 175, 50, 88, 204, 39, 236, 335, 130, 102, 31, 86, 168, 20, 368, 12, 312, 88, 104, 9, 108, 55, 404, 8, 48, 214, 187, 11, 217, 158, 115, 148, 330, 224, 42, 2, 82, 433, 219, 168, 10, 38, 22, 17, 22, 58, 105, 25, 93, 93, 27, 294, 395, 48, 70, 31, 262, 85, 390, 136, 397, 251, 124, 127, 20, 13, 6, 249, 176, 96, 748, 11, 11, 199, 105, 764, 172, 92, 1050, 100, 15, 2, 99, 45, 242, 81, 8, 95, 81, 73, 10, 29, 307, 528, 18, 72, 21, 104, 733, 344, 35, 304, 369, 64, 396, 184, 103, 6, 165, 163, 277, 80, 5, 200, 90, 255, 111, 10, 140, 16, 65, 2, 30, 132, 151, 632, 236, 317, 418, 17, 344, 95, 2, 3, 96, 75, 128, 142, 193, 241, 73, 250, 378, 306, 82, 177, 31, 23, 136, 106, 699, 374, 37, 177, 3, 114, 45, 282, 95, 164, 47, 146, 346, 104, 33, 261, 91, 4, 148, 113, 477, 16, 209, 149, 146, 110, 122, 149, 108, 89, 150, 50, 147, 78, 2, 390, 18, 812, 16, 110, 4, 209, 227, 218, 27, 122, 208, 161, 375, 87, 160, 732, 70, 151, 160, 352, 601, 37, 31, 167, 107, 260, 111, 115, 24, 93, 313, 57, 28, 72, 463, 164, 307, 102, 82, 40, 216, 120, 164, 68, 97, 107, 201, 28, 110, 127, 14, 184, 77, 43, 35, 260, 235, 114, 165, 63, 233, 96, 99, 95, 157, 26, 51, 488, 12, 3, 137, 146, 548, 119, 95, 29, 13, 95, 168, 222, 131, 88, 67, 101, 99, 22, 10, 112, 44, 149, 274, 97, 44, 105, 411, 7, 215, 190, 2, 267, 53, 59, 14, 75, 90, 367, 10, 8, 103, 149, 163, 11, 46, 44, 24, 492, 70, 108, 172, 120, 89, 12, 136, 53, 23, 11, 171, 39, 134, 93, 240, 121, 220, 333, 39, 345, 72, 58, 7, 60, 8, 153, 90, 96, 34, 64, 201, 109, 141, 125, 265, 146, 41, 52, 72, 4, 34, 211, 127, 14, 14, 143, 274, 180, 70, 67, 69, 267, 66, 164, 318, 172, 35, 52, 22, 16, 73, 65, 119, 45, 24, 221, 30, 41, 4, 72, 181, 184, 199, 112, 114, 58, 158, 29, 16, 83, 56, 11, 10, 103, 172, 66, 75, 94, 140, 50, 807, 9, 80, 16, 29, 32, 55, 20, 116, 92, 58, 67, 181, 75, 173, 15, 102, 168, 59, 19, 144, 51, 58, 79, 18, 21, 182, 164, 66, 70, 65, 74, 118, 56, 206, 78, 125, 62, 62, 10, 70, 190, 120, 25, 106, 114, 56, 211, 84, 159, 73, 267, 227, 72, 174, 26, 103, 50, 41, 204, 64, 122, 90, 285, 78, 180, 76, 14, 177, 164, 26, 64, 37, 104, 244, 18, 90, 165, 119, 7, 255, 111, 8, 177, 85, 43, 181, 146, 119, 167, 94, 336, 65, 36, 4, 105, 67, 86, 119, 126, 60, 174, 196, 170, 115, 64, 325, 113, 77, 134, 316, 165, 199, 205, 158, 114, 397, 61, 70, 359, 82, 97, 172, 134, 87, 42, 152, 5, 19, 37, 12, 138, 49, 60, 331, 389, 76, 180, 16, 146, 70, 93, 4, 12, 47, 168, 2, 26, 237, 95, 23, 279, 6, 15, 41, 60, 10, 11, 17, 18, 21, 5, 25, 88, 41, 13, 4, 2, 4, 4, 15, 69, 149, 28, 91, 127, 39, 90, 112, 60, 226, 12, 126, 133, 75, 135, 3, 2, 12, 115, 24, 21, 9, 8, 153, 76, 13, 3, 7, 24, 3, 57, 31, 25, 16, 46, 101, 5, 24, 24, 32, 37, 294, 29, 126, 68, 326, 54, 68, 217, 72, 29, 88, 16, 106, 348, 111, 31, 6, 6, 23, 92, 9, 46, 31, 3, 200, 4, 3, 24, 230, 4, 88, 11, 4, 13, 2, 4, 33, 95, 6, 92, 82, 18, 59, 143, 211, 112, 209, 108, 127, 192, 254, 159, 251, 372, 211, 187, 124, 139, 412, 180, 158, 196, 190, 48, 41, 153, 77, 134, 116, 108, 104, 232, 257, 95, 93, 95, 85, 91, 153, 104, 109, 84, 70, 331, 147, 235, 185, 335, 336, 311, 245, 304, 328, 117, 105, 127, 92, 91, 110, 87, 82, 78, 79, 222, 323, 88, 281, 274, 140, 335, 4, 5, 13, 185, 207, 259, 255, 131, 291, 262, 235, 289, 253])


def compute_win_rate_adjust_by_output_tok_length(preference, output_tok_length):
    """win-rate adjusted by length for each example.
        originally each win has weight 1.
        now weight by 
            `reference model output length / model output length`
        don't need to adjust for repetition, since in that case this number will be very small.
    """

    ref_vs_model_len_ratio = alpacaeval_output_lens / output_tok_length
    preference = np.array(preference)
    n_wins = np.sum(ref_vs_model_len_ratio[preference==2.])
    n_wins_base = np.sum(ref_vs_model_len_ratio[preference==1.])
    n_draws = np.sum(ref_vs_model_len_ratio[preference==0.])
    n_total = n_wins + n_wins_base + n_draws
    win_rate = (n_wins/n_total + .5*n_draws/n_total)
    
    return win_rate

def update_metrics_with_highly_repeated_chars(
        save_dir,
        num_repeated_substr_threshold=100,
        repeated_str_len=3, 
        update_metrics_file=False,
        num_proc=1,
    ):
    """Update `metrics.json` in alpacaeval with 
            - percentage counts of highly repeated generations.
            - win-rate adjusted for repetitive outputs (always give base model a win). 
            
        ```
        import glob
        from note_pruning_analysis import update_metrics_with_highly_repeated_chars, count_repeating_substrings
        save_dirs = glob.glob('results/*/*/eval/alpacafarm*/')
        save_dirs = glob.glob('results/baselines/*/*/eval/alpacafarm*/')
        from tqdm import tqdm
        for save_dir in tqdm(save_dirs):
            print(save_dir)
            ds, d = update_metrics_with_highly_repeated_chars(
                save_dir,
                num_repeated_substr_threshold=100,
                repeated_str_len=3,
                num_proc=8,
                update_metrics_file=True)
        ```
            
    """
    from datasets import Dataset

    tokenizer_name_or_path = get_tokenizer_name_or_path('llama-7b')
    tokenizer = get_fast_tokenizer(tokenizer_name_or_path)

    metrics_file = os.path.join(save_dir, 'metrics.json')
    if not os.path.isfile(metrics_file):
        print(f'metrics_file={metrics_file} does not exist, skip.')
        return (None, None)
    
    ann_file = os.path.join(save_dir, 'annotations.json')
    if not os.path.isfile(ann_file):
        print(f'ann_file={ann_file} does not exist.')
        ann_files = glob.glob(os.path.join(save_dir, "*", 'annotations.json'))
        if len(ann_files) == 1:
            ann_file = ann_files[0]
            print(f'=1 ann_files found in sub-directories. ann_file={ann_file}')
        else:
            print('>1 ann_files found. ann_files={ann_files}')
            return (None, None)

    with open(ann_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    ds = Dataset.from_pandas(df)

    def compute_preference_adjust_by_repetitiveness(example):
        output = {}
        for k in ['output_1', 'output_2']:
            d = count_repeating_substrings(example[k], length=repeated_str_len)
            if ' '*repeated_str_len in d: del d[' '*repeated_str_len] # skip white space for coding
            output[f'{k}_repchar'] = str(d)
            output[f'{k}_maxrepchar'] = max(list(d.values())) if d else 0

        # Always set base model win if following holds
        #     - base model not repetitive  
        #     - evaluated model highly repetitive 
        output_1_highlyrepeated = output['output_1_maxrepchar'] >= num_repeated_substr_threshold
        output_2_highlyrepeated = output['output_2_maxrepchar'] >= num_repeated_substr_threshold
        preference = example['preference']
        if not output_1_highlyrepeated and output_2_highlyrepeated:
            preference = 1.
        if preference is None: # just favors gpt3 if preference is not available
            preference = 1
        output.update({'preference_repetition_adjusted': float(preference),
                       'output_1_highlyrepeated': float(output_1_highlyrepeated),
                       'output_2_highlyrepeated': float(output_2_highlyrepeated),})
        
        # adjust output token lengths as well
        d = tokenizer(example['output_2'])
        output.update({'output_2_tok_length': len(d['input_ids']),
                       'output_2_tok_length_adjusted': -1 if output_2_highlyrepeated else len(d['input_ids'])})
        
        return output
    ds = ds.map(compute_preference_adjust_by_repetitiveness, num_proc=num_proc)

    d = {
        'output_1_highlyrepeated': np.sum(ds['output_1_highlyrepeated'])/len(ds) * 100,
        'output_2_highlyrepeated': np.sum(ds['output_2_highlyrepeated'])/len(ds) * 100,
        'win_rate_repetition_adjusted': compute_win_rate(ds['preference_repetition_adjusted']) * 100,
        'win_rate_adjusted_by_tok_length': compute_win_rate_adjust_by_output_tok_length(ds['preference'], ds['output_2_tok_length']) * 100,
        'output_2_tok_length_median': np.median(ds['output_2_tok_length']),
        'output_2_tok_length_adjusted': np.mean([x for x in ds['output_2_tok_length_adjusted'] if x!=-1]),
    }
    
    if update_metrics_file:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        metrics.update(d)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)

    dsf = ds.filter(lambda x: x['output_2_highlyrepeated']==1., num_proc=4)
    dsf.to_json(os.path.join(save_dir, 'repeated_generation.jsonl'))
            
    return ds, d




def get_alpacafarm_preference_and_generation(save_dir, task='alpacafarm_ann=chatgpt_chatfmt'):
    ann_file = os.path.join(save_dir, 'eval', task, 'annotations.json')
    with open(ann_file, 'r') as f:
        anns = json.load(f)
    return {'instruction': [x['instruction'] for x in anns],
            'generation_base': [x['output_1'] for x in anns],
            'preference': [x['preference'] for x in anns],
            'generation': [x['output_2'] for x in anns],}


def get_alpacafarm_generations(save_dirs, filter_fn_name=None, task='alpacafarm_ann=chatgpt_chatfmt', save_path=None):
    """Given a list of models, get respective model's generation for alpacaeval's instructions. """

    data = {}
    for i, save_dir in enumerate(save_dirs):
        run_name = os.path.basename(save_dir)
        d = get_alpacafarm_preference_and_generation(save_dir, task=task)
        d['run_name'] = [run_name]*len(d['generation'])
        if i == 0:
            d.update({'run_name_base': [task]*len(d['generation'])})
        else:
            del d['generation_base']
            del d['instruction']
        d = {f'{k}_{i}' if 'base' not in k and 'instruction' not in k else k: v 
             for k, v in d.items()}
        data.update(d)
        
    df = pd.DataFrame(data)
    df = df.sort_index(axis=1)

    if filter_fn_name is not None:
        if filter_fn_name == 'lw':
            filter_fn = lambda row: row['preference_0'] == 1. and row['preference_1'] == 2.
        elif filter_fn_name == 'wl':
            filter_fn = lambda row: row['preference_0'] == 2. and row['preference_1'] == 1.
        elif filter_fn_name == 'll':
            filter_fn = lambda row: row['preference_0'] == 1. and row['preference_1'] == 1.
        elif filter_fn_name == 'ww':
            filter_fn = lambda row: row['preference_0'] == 2. and row['preference_1'] == 2.
        else:
            raise ValueError(f'Unknown filter_fn_name={filter_fn_name}')
        df = df[df.apply(filter_fn, axis=1)]

    d = {1: 'lose (vs. davinci003)', 2: 'win (vs. davinci003)'}
    df = df.replace({"preference_0": d, 'preference_1': d})

    df = df.reset_index(drop=False)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_json(save_path.replace('.json', f'_{len(df)}.json'), orient='records', indent=4)

    return df



def is_preference_dataset(dataset):
    if any(x in dataset for x in [
        'openai_sum',
        'shp',
        'hh_rlhf',
        'ultrafeedback',
    ]):
        return True
    else:
        return False

    
def get_encode_fn_type(md, dataset):
    """Get encode_fn_type given `md` and `dataset`.
        - if dataset is ultrafeedback, always use `pref`.
    """
    if md is not None and any(x in md for x in ['mpnet', 'bge']):
        return 'input'
    else:
        if dataset is not None and is_preference_dataset(dataset):
            return 'pref'
        else:
            return 'sft'


md_to_model_name = {
    'mpnet': 'all-mpnet-base-v2',
    'bge': 'bge-large-en-v1.5',
    'llama7b': 'llama-7b+lora:r=256:a=256',
    'llama2:7b': 'llama2-7b+lora:r=256:a=256',
    'codellama7b': 'codellama-7b+lora:r=256:a=256',
    'mistral7b': 'mistral-7b+lora:r=256:a=256',
    'llama7b+lima': 'llama-7b+lima+lora:r=256:a=256',
}

# after fixed bug in lora init -> unbiased pairwise distance.
# try different projection dimension (default 2048)
md_to_model_name.update({
    'llama7br256p4096': 'llama-7b+lora:r=256:a=4096+proj=4096',
    'llama7br512p4096': 'llama-7b+lora:r=512:a=11585+proj=4096',
    'pythia1br512p4096': 'pythia-1b+lora:r=512:a=11585+proj=4096',
    'mistral7br512p4096': 'mistral-7b+lora:r=512:a=11585+proj=4096'
})

# randsphere baselines 
md_to_model_name.update({
    'randspherep4096': 'randspherep4096',
    'randspherep768': 'randspherep768',
    'randspherep1024': 'randspherep1024',
    'randspherep256': 'randspherep256',
    # for measuring randomnessin ldd
    'randspherep4096s0': 'randspherep4096s0',
    'randspherep4096s1': 'randspherep4096s1',
    'randspherep4096s2': 'randspherep4096s2',
    'randspherep4096s3': 'randspherep4096s3',
    'randspherep4096s4': 'randspherep4096s4',
})

# preference data
md_to_model_name.update({
    'llama7b+sharegptv2ep2+r512p4096': 'llama-7b+sharegptv2ep2+lora:r=512:a=11585+proj=4096',
})


def get_full_model_name(md):
    if md in md_to_model_name:
        model_name = md_to_model_name[md]
    else:
        raise ValueError(f'Dont know full name for model_name: {md}')
    return model_name

