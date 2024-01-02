from collections import defaultdict
from functools import partial
import os
import time
import re
import math

import random
import pickle
from tqdm import tqdm 
import datetime

import numpy as np
import pandas as pd
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

import pyarrow # import before `torch`, `transformers`, `datasets`
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from open_instruct.finetune_trainer import encode_with_prompt_completion_format, encode_with_messages_format
from note_pruning_analysis import get_dataset


from sklearn.random_projection import SparseRandomProjection
class TorchSparseRandomProjection(SparseRandomProjection):
    
    # overrrides https://github.com/scikit-learn/scikit-learn/blob/3f89022fa04d293152f1d32fbc2a5bdaaf2df364/sklearn/random_projection.py#L810C16-L810C86
    def transform(self, X):
        device = 'cuda'
        X = torch.from_numpy(X).to(device)
        if hasattr(self, 'Pt_'):
            Pt = self.Pt_
        else:
            Pt = self.components_.transpose().tocsr()
            Pt = torch.sparse_csr_tensor(torch.from_numpy(Pt.indptr),
                                         torch.from_numpy(Pt.indices),
                                         torch.from_numpy(Pt.data),)
            Pt = Pt.to(device)
            self.Pt_ = Pt
        XPt = X @ Pt
        XPt = XPt.to('cpu').numpy()
        return XPt



def torch_cdist(X, device):
    """Compute cdist on gpu."""
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    X = X.to(torch.float32).to(device).unsqueeze(0)
    D = torch.cdist(X, X)
    D = D.cpu().numpy()
    return D
    


def torch_cdist_chunked(X, device, chunk_size=None, mem_total=12, normalize=False):
    """Compute cdist on GPU with reduced peak memory usage.
    
        ```
        from note_llama_embeddings import torch_cdist, torch_cdist_chunked
        X = torch.from_numpy(np.random.rand(1000, 1000))
        D0 = torch_cdist(X, 'cpu')
        D1 = torch_cdist(X, 'cuda')
        D2 = torch_cdist_chunked(X, 'cpu', chunk_size=None)
        D3 = torch_cdist_chunked(X, 'cuda', chunk_size=500)

        print(np.abs((D0-D1)).max())
        print(np.abs((D0-D2)).max())
        print(np.abs((D2-D3)).max())
        print(np.abs((D1-D3)).max(), np.abs((D1-D3)).mean())
        ```
    
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    X = X.to(torch.float32)
    N = X.shape[0]
    D = X.shape[1]
    
    if chunk_size is None:
        chunk_size = N
    elif chunk_size == 'auto':
        mem_X = N*D*4 / (1024**3)
        chunk_size = math.floor((mem_total / (mem_X*2)) * N)
        chunk_size = min(chunk_size, N)
        print(f'[torch_cdist_chunked] set chunk_size = {chunk_size} for (N, D) = ({N}, {D})')

    Ds = []
    for i in range(0, N, chunk_size):
        Xi = X[i:i+chunk_size, :].to(device)
        if normalize:
            Xi = Xi / torch.linalg.norm(Xi, ord=2, dim=1, keepdim=True)
        for j in range(0, N, chunk_size):
            Xj = X[j:j+chunk_size, :].to(device)
            if normalize:
                Xj = Xj / torch.linalg.norm(Xj, ord=2, dim=1, keepdim=True)
            D = torch.cdist(Xi, Xj)
            Ds.append(D.cpu().numpy())
            del Xj, D
        del Xi
    
    nchunks = np.sqrt(len(Ds))
    if not nchunks.is_integer():
        raise ValueError(f'[torch_cdist_chunked] nchunks={nchunks} not integer')
    nchunks = int(nchunks)

    Ds_mat = [Ds[i:i+nchunks] for i in range(0, len(Ds), nchunks)]
    D = np.vstack([np.hstack(x) for x in Ds_mat])

    return D


def plt_pair_of_dists(D1, D2, use_hexbin=True):
    """visualize pair-wise distance distribution of two sets of samples.

        ```
        from note_llama_embeddings import get_grad_vectors_cdist, plt_pair_of_dists
        dataset = 'lima'
        model_names = [
            'pythia-160m',
            'pythia-160m+lora:r=256:a=4096',
            'pythia-160m+lora:r=512:a=11585',
            'pythia-160m+lora:r=512:a=11585+proj=1024',
            'pythia-160m+lora:r=512:a=11585+proj=4096',
            'pythia-160m+lora:r=512:a=11585+proj=8192',
        ]
        Ds, df = get_grad_vectors_cdist(dataset, model_names)
                
        filter_key_fn = lambda k: 'grad_rp_loraB_r=512' in k
        for k in sorted(list(Ds.keys())):
            if not filter_key_fn(k): continue
            if k == 'grad_qkv':
                continue
            fig, axs = plt_pair_of_dists(Ds['grad_qkv'], Ds[k], use_hexbin=False)
            fig.suptitle(f"[{dataset}:{model_name}] D1=grad_qkv {sizes['grad_qkv']}, D2={k} {sizes[k]}")
            fig.tight_layout()
        ```
    
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 16})

    fig, axs = plt.subplots(1,2,figsize=(12,6))
    ax = axs[0]
    min_dist = min(D2.min(), D1.min())
    max_dist = max(D2.max(), D1.max())
    max_dist = max(np.quantile(D2, .95), np.quantile(D1, .95))
    if use_hexbin:
        hb = ax.hexbin(
            D1,
            D2,
            gridsize=100,
            cmap=plt.cm.PuBu,
            extent=[min_dist, max_dist, min_dist, max_dist],
        )
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label("Sample pairs counts")
    else:
        ax.scatter(D1, D2, s=1, alpha=0.1)
        ax.set_xlim(min_dist, max_dist)
        ax.set_ylim(min_dist, max_dist)
    ax.plot([min_dist, max_dist], [min_dist, max_dist], color='red', linestyle='--')
    ax.set_xlabel("Pairwise dist D1")
    ax.set_ylabel("Pairwise dist D2")


    rates = D2 / D1

    ax = axs[1]
    ax.hist(rates, bins=50, range=(0., 2.), edgecolor="k", density=True)

    rates_mean, rates_std = np.mean(rates), np.std(rates)
    ax.axvline(rates_mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {rates_mean:.4f}')
    ax.axvline(rates_mean + 2 * rates_std, color='g', linestyle='dashed', linewidth=2, label=f'2*std: {2*rates_std:.4f}')
    ax.axvline(rates_mean - 2 * rates_std, color='g', linestyle='dashed', linewidth=2)
    ax.legend()
    ax.set_xlabel("distances rate: D2 / D1")
    ax.set_ylabel("Distribution of samples pairs")

    fig.tight_layout()

    return fig, axs



def grad_vectors_compute_cdist(dataset, model_names):
    """compute pair-wise distance to compare different random projection schemes

        ```
        from note_llama_embeddings import grad_vectors_compute_cdist
        dataset = 'lima'
        model_names = [
            'pythia-160m',
            'pythia-160m+lora:r=256:a=4096',
            'pythia-160m+lora:r=512:a=11585',
            'pythia-160m+lora:r=512:a=11585+proj=1024',
            'pythia-160m+lora:r=512:a=11585+proj=4096',
            'pythia-160m+lora:r=512:a=11585+proj=8192',
        ]
        grad_vectors_compute_cdist(dataset, model_names)
        ```
    """

    for model_name in model_names:
        sizes = {}
        Ds = {}
        Ds_norm = {}

        save_path = os.path.join('model_outputs', 'sft', model_name, f'{dataset}.pkl')
        with open(save_path, 'rb') as f:
            o = pickle.load(f)
            for k in [x for x in o.keys() if x.startswith('grad') and 'l2n' not in x]:
                print(f'Computing distance for {k} of size {o[k].shape}')
                Ds[k] = torch_cdist_chunked(o[k], device='cuda', chunk_size='auto', mem_total=12, normalize=False)
                Ds_norm[k] = torch_cdist_chunked(o[k], device='cuda', chunk_size='auto', mem_total=12, normalize=True)
                sizes[k] = o[k].shape
            del o

        save_path = os.path.join('model_outputs', 'sft', model_name, f'{dataset}_dist.pkl')
        with open(save_path, 'wb') as f:
            output = {'Ds': Ds, 'Ds_norm': Ds_norm, 'sizes': sizes}
            pickle.dump(output, f)




def get_grad_vectors_cdist(dataset, model_names, compute_dist_with_normalized_vector=False, base_vector_name='grad_qkv'):
    """Gather results from `grad_vectors_compute_cdist`
        
        ```
        from note_llama_embeddings import get_grad_vectors_cdist
        dataset = 'lima'
        model_names = [
            'pythia-160m',
            'pythia-160m+lora:r=256:a=4096',
            'pythia-160m+lora:r=512:a=11585',
            'pythia-160m+lora:r=512:a=11585+proj=1024',
            'pythia-160m+lora:r=512:a=11585+proj=4096',
            'pythia-160m+lora:r=512:a=11585+proj=8192',
        ]
        Ds, df = get_grad_vectors_cdist(dataset, model_names)
        ```
    """
    Ds_key = 'Ds_norm' if compute_dist_with_normalized_vector else 'Ds'

    Ds = {}
    sizes = {}
    for model_name in model_names:
        match = re.search(r'r=(\d+)', model_name)
        lora_rank = int(match.group(1)) if match else None
        match = re.search(r'a=(\d+)', model_name)
        lora_alpha = int(match.group(1)) if match else None
        match = re.search(r'proj=(\d+)', model_name)
        proj = int(match.group(1)) if match else 2048
        base_model = model_name.split('+')[0]

        save_path = os.path.join('model_outputs', 'sft', model_name, f'{dataset}_dist.pkl')
        with open(save_path, 'rb') as f:
            o = pickle.load(f)
            o_Ds = {f'{k}_r={lora_rank}_a={lora_alpha}_p={proj}_md={base_model}' if lora_rank else k: v for k, v in o[Ds_key].items()}
            o_sizes = {f'{k}_r={lora_rank}_a={lora_alpha}_p={proj}_md={base_model}' if lora_rank else k: v for k, v in o['sizes'].items()}
            if lora_rank is None:
                o_Ds = {k: v for k, v in o_Ds.items() if k in ['grad_qkv', 'grad_rp_qkv']}
                o_sizes = {k: v for k, v in o_sizes.items() if k in ['grad_qkv', 'grad_rp_qkv']}
            Ds.update(o_Ds)
            sizes.update(o_sizes)

    ## flatten & select non-identical pairs.
    nonzero = Ds[base_vector_name]!=0
    for k in list(Ds.keys()):
        v = Ds[k]
        Ds[k] = v[nonzero].ravel()


    statistics = []
    for k, D in Ds.items():
        if k == base_vector_name:
            continue
        rates = D / Ds[base_vector_name]
        statistics.append({
            'name': k,
            'shape': sizes[k],
            'rates_mean': np.mean(rates),
            'rates_std': np.std(rates),
        })

    df = pd.DataFrame(statistics)
    def parse_params_fn(name):
        match = re.search(r'r=(\d+)', name)
        lora_rank = int(match.group(1)) if match else None
        match = re.search(r'a=(\d+)', name)
        lora_alpha = int(match.group(1)) if match else None
        match = re.search(r'p=(\d+)', name)
        proj = int(match.group(1)) if match else 2048
        match = re.search(r'md=([\w-]+)', name)
        base_model = match.group(1) if match else None

        return {
            'base_model': base_model,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'proj': proj,
        }
    df = df.join(pd.DataFrame(list(df['name'].apply(parse_params_fn))))
    
    return Ds, df


def plt_rates_std_vs_projected_dim(df):
    """Visualize effects of JLT projected dimension & Lora rank on pairwise distance ratio std.
    
        ```
        from note_llama_embeddings import get_grad_vectors_cdist, plt_rates_std_vs_projected_dim
        dataset = 'lima'
        model_names = [
            'pythia-160m',
            'pythia-160m+lora:r=256:a=4096',
            'pythia-160m+lora:r=512:a=11585',
            'pythia-160m+lora:r=512:a=11585+proj=1024',
            'pythia-160m+lora:r=512:a=11585+proj=4096',
            'pythia-160m+lora:r=512:a=11585+proj=8192',
        ]
        Ds, df = get_grad_vectors_cdist(dataset, model_names)
        title = f"{dataset}:{model_names[0]}"
        fig, ax = plt_rates_std_vs_projected_dim(df)
        fig.suptitle(title)
        ```
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,1,figsize=(8,5))

    dft = df[df['name']=='grad_rp_qkv']
    xs = dft['proj']
    ys = dft['rates_std']
    ax.plot(xs, ys, '.-', label='grad_rp_qkv')

    dft = df[df['name'].apply(lambda x: 'grad_rp_loraB_' in x)]
    for k, dfg in dft.groupby('lora_rank'):
        dfg = dfg.sort_values(['proj'])
        xs = dfg['proj']
        ys = dfg['rates_std']
        ax.plot(xs, ys, '.-', label=f'lora_r={int(k)}')

    ax.set_ylabel('pairwise dist ($\pi dL/dL$) ratio std')
    ax.set_xlabel('JLT projected dim')
    ax.set_xticks([1024, 2048, 4096, 8192])
    ax.set_ylim(bottom=0.03)

    fig.legend(loc='center right')

    return fig, ax


def encode_just_one_role(example, tokenizer, max_seq_length, encode_fn_type, add_eos_token):
    """Takes first-turn conversation and encode it as a single sequence."""
    messages = example['messages']
    assert(len(messages) >= 2)

    # if single-turn, take the message
    # if multi-turn, join the messages with '\n\n'
    if encode_fn_type == 'input':
        contents = [x['content'] for x in messages if x['role']=='user']
        text = '\n\n'.join(contents)
    elif encode_fn_type == 'output':
        contents = [x['content'] for x in messages if x['role']=='assistant']
        text = '\n\n'.join(contents)
    else:
        raise ValueError(f'encode_fn_type={encode_fn_type} not supported.') 
        
    if add_eos_token:
        text += tokenizer.eos_token

    tokenized_example = tokenizer(
        text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    attention_mask = tokenized_example.attention_mask
    labels = input_ids.clone()
    
    return {'text': text,
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten()}


def sklearn_rp_mat_size(rp):
    if hasattr(rp, "components_"):
        if isinstance(rp.components_, np.ndarray):
            n_bytes = rp.components_.nbytes
        else:
            n_bytes = rp.components_.data.nbytes
            n_bytes += rp.components_.indices.nbytes
        return n_bytes
    

    
def get_grad_statistic_pattern(model_name_or_path, use_lora):
    if use_lora:
        grad_statistic_patterns = {
            'loraB': r'lora_B\.[a-zA-Z_]+\.weight',
        }
    else:
        if any(x in model_name_or_path.lower() for x in ['llama', 'mistral']):
            grad_statistic_patterns = {
                'all': r'.*',
                'qkv': r'(q_proj\.weight|k_proj\.weight|v_proj\.weight|o_proj\.weight)',
                'mlp': r'\bmlp\..*?\.weight\b',
                'last': r'\blm_head\.weight\b',
            }
        elif 'pythia' in model_name_or_path:
            grad_statistic_patterns = {
                'all': r'.*',
                'qkv': r'\bquery_key_value\.weight\b',
                'mlp': r'\bmlp\..*?\.weight\b',
                'last': r'\bembed_out\.weight\b',
            }
        else:
            grad_statistic_patterns = {
                'all': r'.*'
            }
    return grad_statistic_patterns


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


@torch.inference_mode()
def compute_grad_statistic(model, patterns):

    param_names = []
    grads = []
    for param_name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            param_names.append(param_name)
            grads.append(param.grad.to(torch.float32))

    statistic = {}
    for pattern_name, pattern in patterns.items():
        grads_filtered = list(filter(lambda x: True if re.search(pattern, x[0]) else False,
                                     zip(param_names, grads)))
        grads_filtered = [x[1] for x in grads_filtered]
        if not grads_filtered:
            continue
        norms = compute_grad_norm(grads_filtered)
        for norm_type, v in norms.items():
            statistic[f'{pattern_name}_{norm_type}'] = v

    return statistic
    


@torch.inference_mode()
def compute_grad_norm(l):
    """Given a list of Tensors `l`, compute 
        - sum of norm, 
        - norm of concatenated vectors. """
    device = l[0].device
    output = {}
    # `l2n_sum` and `l2n` seems quite correlated, just compute 1.
    # output['l2n_sum'] = torch.tensor(
    #     sum([torch.linalg.norm(x.reshape(-1), ord=2).cpu().item() for x in l]), device=device)
    g = torch.vstack([x.reshape(-1, 1) for x in l]).squeeze()
    output['l2n'] = torch.linalg.norm(g, ord=2)
    return output


def np_shift_rows_then_sum_cols(A):
    A_shifted = np.zeros_like(A)
    for i in range(A.shape[0]):
        A_shifted[i, :] = np.roll(A[i, :], i)
    s = np.sum(A_shifted, axis=0)
    return s


@torch.inference_mode()
def gather_grad_embeddings(model, patterns, stacked=True, add_rsum=False):
    """Gather grad vectors from `model`.
        if `add_rsum` is True, then sum over rows for grad matrices with & without row-wise cyclic shift"""

    param_names = []
    grads = []
    for param_name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            param_names.append(param_name)
            grads.append(param.grad.cpu().to(torch.float32).numpy())

    grad_embeddings = {}
    for pattern_name, pattern in patterns.items():
        grads_filtered = list(filter(lambda x: True if re.search(pattern, x[0]) else False,
                                    zip(param_names, grads)))
        g = [x[1] for x in grads_filtered]
        if not grads_filtered:
            continue
        grad_embeddings[pattern_name] = g
        
        if add_rsum:
            grad_embeddings[f'{pattern_name}rsum'] = [x.sum(0) for x in g]
            grad_embeddings[f'{pattern_name}rsumcyc'] = [np_shift_rows_then_sum_cols(x) for x in g]
            
    if stacked:
        for k in list(grad_embeddings.keys()):
            g = grad_embeddings[k]
            # the resulting g has the correct ordering, i.e.,
            # parameters associated with any weight matrix is grouped together.
            g = [x.reshape(-1) for x in g]
            g = np.hstack(g).reshape(-1)
            grad_embeddings[k] = g
    return grad_embeddings


@torch.inference_mode()
def compute_losses(logits, labels):
    """ Computes a few different scores based on log probabilities 
        logits (Bsz, |Seq|, |Vocab|)
        labels (Bsz, |Seq|)
    """
    if logits.shape[0]!=1:
        raise ValueError('compute_el2n supports bsz=1 only.')

    vocab_size = logits.shape[-1]
    device = logits.device
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    # (Bsz*|Seq|, |Vocab|)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_probs = torch.nn.functional.softmax(shift_logits, dim=-1)
    shift_labels = shift_labels.view(-1)
    # only compute loss on the output tokens
    output_tok_indices = (shift_labels != -100).nonzero().reshape(-1)
    shift_labels = shift_labels[output_tok_indices]
    shift_probs = shift_probs[output_tok_indices]
    shift_logits = shift_logits[output_tok_indices]
    # Enable model parallelism
    shift_labels = shift_labels.to(device)

    losses = {}
    # Compute EL2N = || prob - one-hot-label ||_2
    shift_probs_minus_onehot_target = shift_probs.clone()
    shift_probs_minus_onehot_target[torch.arange(shift_probs.size(0)), shift_labels] -= 1
    loss_tokenwise = torch.linalg.norm(shift_probs_minus_onehot_target, dim=-1, ord=2)
    losses['el2n_agg=mean'] = loss_tokenwise.mean()
    losses['el2n_agg=l2n'] =  torch.linalg.norm(loss_tokenwise, ord=2)

    # Classification logit margin
    shift_logits_true = torch.gather(shift_logits, 1, shift_labels.view(-1, 1)).squeeze()
    shift_logits_other = shift_logits.clone()
    shift_logits_other[torch.arange(shift_logits.size(0)), shift_labels] = float('-inf')
    shift_logits_other_max, _ = torch.max(shift_logits_other, 1)
    losses['logit_margin'] = (shift_logits_true-shift_logits_other_max).mean()

    # if no output tokens return nan
    if output_tok_indices.size()[0] == 0:
        losses = {k: torch.tensor(np.nan, device=device) for k in losses.keys()}
    return losses


def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def combine_lm_outputs_for_mixes(dataset, save_dir, test_run):
        
    # Need to keep the same order as specified in `prepare_train_data.sh`
    mixes = {
        'tulu_v1_human_mix': ['flan_v2', 'cot', 'dolly', 'oasst1'],
        'tulu_v1_mix': ['flan_v2', 'cot', 'dolly', 'oasst1', 'gpt4_alpaca', 'code_alpaca', 'sharegpt'],
        'tulu_v2_human_mix': ['flan_v2', 'cot', 'oasst1', 'lima'],
        'tulu_v2_mix': ['flan_v2', 'cot', 'oasst1', 'lima', 'code_alpaca', 'sharegpt', 'wizardlm', 'open_orca']
    }

    mix_name = dataset
    mix_datasets = mixes[dataset] 

    output_list = []
    for dataset in mix_datasets:
        save_path = os.path.join(save_dir, f'{dataset}.pkl')
        with open(save_path, 'rb') as f:
            output = pickle.load(f)
        output_list.append(output)

    output = {}
    for k in output_list[0].keys():
        output[k] = np.vstack([x[k] for x in output_list])

    save_path = os.path.join(save_dir, ('test_' if test_run else '')+f'{mix_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'dataset: {dataset}')
    print(f'Save output={[(k, v.shape) for k, v in output.items()]} to {save_path}')


def datasets_shard_chunk_size(N, num_shards, index):
    """Get chunk size for `datasets.shard(..., contiguous=True)`. """
    div = N // num_shards
    mod = N % num_shards
    start = div * index + min(index, mod)
    end = start + div + (1 if index < mod else 0)
    return end-start

    

def dist_gather_and_vstack_2d_tensors(tensor, tensor_list_sizes, rank):
    """ For ncll backend: requires roughly (world_size+1)/world_size x of 
            stacked tensor's memory to fit in 1 gpu. 
        For gloo backend, as long as everything fits in cpu memory.
            however, timeout if send/recv hangs for 180000ms. transfering 
            large data gives timeout error.

            ```
            text_embeddings = dist_gather_and_vstack_2d_tensors(
                text_embeddings, train_dataset_chunk_sizes, rank)
            ```
        """
    D = tensor.shape[1]
    max_chunk_size = max(tensor_list_sizes)
    tensor_list = [torch.zeros((max_chunk_size, D), 
                                dtype=tensor.dtype, device=tensor.device) 
                    for _ in range(len(tensor_list_sizes))]
    # Note `tensor_list` have to be same shape, 
    # pad `tensor` in all processes to same length before gather.
    if tensor.shape[0] != max_chunk_size:
        tensor = torch.vstack([
            tensor,
            torch.zeros((max_chunk_size-tensor.shape[0], D),
                            device=tensor.device, dtype=tensor.dtype)
        ])
    if rank == 0:
        dist.gather(tensor, gather_list=tensor_list, dst=0)
    else:
        dist.gather(tensor, gather_list=[], dst=0)
    # remove padding 
    tensor_list = [x[:B] for x, B in zip(tensor_list, tensor_list_sizes)]
    tensor = torch.vstack(tensor_list)
    return tensor


def compute_lm_outputs(
        dataset,
        model_name_or_path='../results/baselines/huggyllama/llama-7b',
        save_dir='/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/llama-7b_outputs',
        use_dist=False,
        test_run=False,
        shuffle=False,
        compute_loss=True,
        compute_grad=False,
        use_lora=False,
        lora_rank=128,
        lora_alpha=128,
        compute_grad_embeddings=False,
        save_grad_embeddings=False,
        grad_randproj_components=2048,
        max_seq_len=2048,
        encode_fn_type='sft',
        text_pooling_type='meanpool',
        add_rsum=False,
        torch_dtype='float16',
    ):
    """
        `shuffle` to allow each process to process roughly similar workload cross the dataset 
            to avoid unnecessary waiting.
    """

    os.makedirs(save_dir, exist_ok=True)

    if dataset in ['tulu_v1_human_mix', 
                   'tulu_v1_mix',
                   'tulu_v2_human_mix',
                   'tulu_v2_mix']:
        combine_lm_outputs_for_mixes(dataset, save_dir, test_run)
        return

    if use_dist:
        dist.init_process_group("gloo", timeout=datetime.timedelta(hours=6))
        world_size = dist.get_world_size()
        rank = dist.get_rank() # global rank
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    print(f'rank/local_rank/world_size: {rank}/{local_rank}/{world_size}\n')

    device = f'cuda:{str(local_rank)}'

    if 'sentence-transformers' in model_name_or_path:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_name_or_path,
            device_map=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device,
            torch_dtype=getattr(torch, torch_dtype),
        )

    if use_lora:
        if not compute_grad:
            raise ValueError('compute_grad must be True if use LoRA!')
        
        print(f'Initializing lora(r={lora_rank},a={lora_alpha})')
        # ensure the same initialization
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        if any(x in model_name_or_path.lower() for x in ['llama', 'mistral']):
            # # the following also applies lora to MLP layers.
            # target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        elif 'pythia' in model_name_or_path:
            target_modules = ['query_key_value']
        else:
            raise ValueError(f'Define new `target_modules` for LoraConfig for {model_name_or_path}')

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            bias='none',
            r=lora_rank,
            lora_alpha=lora_alpha, 
            lora_dropout=0.,
            target_modules=target_modules,
            init_lora_weights='gaussian',
        )
        
        # https://github.com/huggingface/peft/issues/137
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        
        ## don't need to compute gradient to `lora_A`, saves computation (i think) but not space.
        for param_name, param in model.named_parameters():
            if param.requires_grad and 'lora_A' in param_name:
                param.requires_grad = False

    print_trainable_parameters(model)
        
    if compute_grad:
        if any(x in model_name_or_path.lower() for x in ['llama', 'mistral']):
            # Computing full gradient for llama or even lora's weight matrix is 
            # computationally prohibitive. Use gradient checkpointing to prevent oom issues.
            # Note gradient checkpointing is only applied when in training mode
            #     https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L908
            # So need to set `model.train()`. This is harmless because
            # llama's eval/train computation is exactly the same, since there's no dropout layer.
            model.gradient_checkpointing_enable()
            model.train()
    else:
        model.eval()


    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


    # processed_dir = '../data/processed'
    # if 'flan2022' in dataset:
    #     train_file = os.path.join(processed_dir, 'flan2022', f'{dataset}_data.jsonl')
    # elif 'ultrachat' in dataset:
    #     train_file = os.path.join(processed_dir, 'ultrachat', f'{dataset}_data.jsonl')
    # elif 'open_orca' in dataset:
    #     train_file = os.path.join(processed_dir, 'open_orca', f'{dataset}_data.jsonl')
    # elif 'sharegpt' in dataset:
    #     train_file = os.path.join(processed_dir, 'sharegpt', f'{dataset}_data.jsonl')
    # elif 'wizardlm' in dataset:
    #     train_file = os.path.join(processed_dir, 'wizardlm', f'{dataset}_data.jsonl')
    # elif 'starcoder' in dataset:
    #     train_file = os.path.join(processed_dir, 'starcoder', f'{dataset}.jsonl')
    # else:
    #     train_file = os.path.join(processed_dir, dataset, f'{dataset}_data.jsonl')
    # assert(os.path.isfile(train_file))
     

    if encode_fn_type in ['input', 'output']:
        encode_function = partial(
            encode_just_one_role,
            tokenizer=tokenizer,
            max_seq_length=max_seq_len,
            encode_fn_type=encode_fn_type,
            # add eos token to causal models, e.g., llama, since its not added by default.
            add_eos_token=False if any(y in model_name_or_path for y in ['mpnet', 'bge']) else True,
        )
    elif encode_fn_type == 'sft':
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_len,
        )
    else:
        raise ValueError(f'encode_fn_type={encode_fn_type} not implemented.')


    if rank == 0:
        # raw_datasets = load_dataset("json", 
        #                             data_files={'train': train_file},
        #                             cache_dir=os.path.dirname(train_file))
        # if test_run:
        #     raw_datasets['train'] = raw_datasets['train'].select(range(100))
        train_dataset = get_dataset(dataset, processed=True)
        print(f"{dataset} dataset length = {len(train_dataset)}")
        train_dataset = train_dataset.map(
            encode_function, batched=False, num_proc=64,
            desc="Tokenizing and reformatting instruction data")
    if use_dist:
        dist.barrier()
    if rank!= 0:
        # raw_datasets = load_dataset("json", 
        #                             data_files={'train': train_file},
        #                             cache_dir=os.path.dirname(train_file))
        # if test_run:
        #     raw_datasets['train'] = raw_datasets['train'].select(range(100))
        train_dataset = get_dataset(dataset, processed=True)
        print(f"{dataset} dataset length = {len(train_dataset)}")
        train_dataset = train_dataset.map(
            encode_function, batched=False, num_proc=64,
            desc="Tokenizing and reformatting instruction data")


    # train_dataset = lm_datasets['train']
        
    train_dataset.set_format(
        type="torch",
        output_all_columns=False,
        columns=['input_ids', 'labels', 'attention_mask'])
    if shuffle:
        random.seed(0)
        shuffle_inds = list(range(len(train_dataset)))
        random.shuffle(shuffle_inds)
        reverse_shuffle_inds = [(i, ind) for i, ind in enumerate(shuffle_inds)]
        reverse_shuffle_inds = sorted(reverse_shuffle_inds, key=lambda x: x[1])
        reverse_shuffle_inds = [x[0] for x in reverse_shuffle_inds]
        train_dataset = train_dataset.select(shuffle_inds)
    train_dataset_chunk_sizes = [datasets_shard_chunk_size(len(train_dataset), num_shards=world_size, index=i) 
                for i in range(world_size)]
    train_dataset = train_dataset.shard(
        num_shards=world_size, 
        index=rank,
        contiguous=True)
    loader = DataLoader(train_dataset, shuffle=False, batch_size=1, pin_memory=True) 

    grad_statistic_patterns = get_grad_statistic_pattern(model_name_or_path, use_lora)

    output = defaultdict(list)
    for i, batch in tqdm(enumerate(loader), disable=rank!=0, total=len(loader)):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        if compute_grad:
            outputs = model(**batch, output_hidden_states=True, use_cache=False)
            model.zero_grad()
            outputs['loss'].backward()
        else:
            with torch.inference_mode():
                outputs = model(**batch, output_hidden_states=True)
        
        # (bsz, seq_len, hidden_size) -> (bsz, hidden_size)
        last_hidden_state = outputs['hidden_states'][-1]
        if text_pooling_type == 'meanpool':
            text_embedding = mean_pooling(last_hidden_state, batch['attention_mask'])
        elif text_pooling_type == 'cls':
            text_embedding = last_hidden_state[:, 0]
        else:
            raise ValueError(f'text_pooling_type={text_pooling_type} not supported.')
        output['text_embedding'].append(text_embedding.to(torch.float32).detach().cpu())

        if compute_loss:
            # average of output token log probs
            if 'loss' in outputs:
                output['log_prob'].append(-outputs['loss'].detach().cpu())
            
            # el2n scores
            losses = compute_losses(outputs['logits'], batch['labels'])
            for k in ['el2n_agg=mean', 'el2n_agg=l2n', 'logit_margin']:
                output[k].append(losses[k].detach().cpu())
            
        if compute_grad:
            ## gradient statistic
            grad_statistics = compute_grad_statistic(model, grad_statistic_patterns)
            for k, v in grad_statistics.items():
                output[f'grad_{k}'].append(v.detach().cpu())

            ## gradient embeddings
            if compute_grad_embeddings:
                grad_embeddings = gather_grad_embeddings(
                    model,
                    {k: v for k, v in grad_statistic_patterns.items() if k in ['qkv', 'loraB']},
                    stacked=True,
                    add_rsum=add_rsum,
                )
                if save_grad_embeddings:
                    for k, v in grad_embeddings.items():
                        output[f'grad_{k}'].append(v)
                if i==0:
                    rps = {}
                    for k, v in grad_embeddings.items():
                        t0 = time.time()
                        if 'rsum' not in k:
                            rps[k] = SparseRandomProjection(n_components=grad_randproj_components, random_state=0)
                        else:
                            rps[k] = GaussianRandomProjection(n_components=grad_randproj_components, random_state=0)
                        print(f"Fitting random projection for {k} ({v.size} -> {grad_randproj_components})")
                        rps[k] = rps[k].fit(v[np.newaxis,...])
                        print(f"Fitting random projection in {time.time() - t0:0.3f}s "
                            f"with random matrix size {sklearn_rp_mat_size(rps[k]) / 1e6:0.3f} MB")
                        print(f'Log statistics of projection matrix for {k} to ensure same initialization cross procs:\n'
                            f"{np.mean(rps[k].components_ != 0)}, {np.max(rps[k].components_)}, {np.mean(rps[k].components_[0])}")
                for k in grad_embeddings.keys():
                    rp = rps[k]
                    g = grad_embeddings[k]
                    output[f'grad_rp_{k}'].append(rp.transform(g[np.newaxis,...]).squeeze())


    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            output[k] = torch.vstack(v).to(torch.float32).numpy()
        else:
            output[k] = np.vstack(v)

    print(f'[local_rank/global={local_rank}/{rank}] '
          f'output={[(k, v.shape) for k, v in output.items()]}')

    if use_dist:
        save_path = os.path.join(save_dir, f'{dataset}_rank={rank}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    if use_dist:
        dist.barrier()

    if rank == 0:
        if use_dist:
            ## load and concat outputs from all ranks
            output = defaultdict(list)
            for r in range(world_size):
                save_path = os.path.join(save_dir, f'{dataset}_rank={r}.pkl')
                with open(save_path, 'rb') as f:
                    output_per_ranks = pickle.load(f)
                for k, v in output_per_ranks.items():
                    output[k].append(v)
                os.remove(save_path)
            for k, v in output.items():
                output[k] = np.vstack(v)
        if shuffle:
            output = {k: v[reverse_shuffle_inds] for k, v in output.items()}
        save_path = os.path.join(save_dir, ('test_' if test_run else '')+f'{dataset}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'dataset: {dataset}')
        print(f'Save output={[(k, v.shape) for k, v in output.items()]} to {save_path}')


if __name__ == "__main__":

    """
    torchrun \
        --nnodes=1 \
        --nproc_per_node=2 \
        --rdzv-id=$SLURM_JOB_ID \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$RDZV_ENDPOINT \
    note_llama_embeddings.py \
        --dataset lima \
        --model_name_or_path ../results/baselines/huggyllama/llama-7b \
        --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/model_outputs/llama-7b \
        --use_dist \
        --shuffle \
        --compute_grad \
        --use_lora \
        --compute_grad_embeddings \
        --grad_randproj_components 2048 \
        --max_seq_len 2048 \
        --encode_fn_type sft \
        --text_pooling_type meanpool
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lima")
    parser.add_argument("--model_name_or_path", type=str, default="../results/baselines/huggyllama/llama-7b")
    parser.add_argument("--save_dir", type=str, default="/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/llama-7b_outputs")
    parser.add_argument("--use_dist", action='store_true', default=False)
    parser.add_argument("--test_run", action='store_true', default=False)
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument("--compute_loss", action='store_true', default=False)
    parser.add_argument("--compute_grad", action='store_true', default=False)
    parser.add_argument("--use_lora", action='store_true', default=False)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--compute_grad_embeddings", action='store_true', default=False)
    parser.add_argument("--save_grad_embeddings", action='store_true', default=False)
    parser.add_argument("--grad_randproj_components", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--encode_fn_type", type=str, default='sft')
    parser.add_argument("--text_pooling_type", type=str, default="meanpool")
    parser.add_argument("--add_rsum", action='store_true', default=False)
    parser.add_argument("--torch_dtype", type=str, default="float16")



    args = parser.parse_args()

    compute_lm_outputs(**vars(args))