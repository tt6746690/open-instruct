import os
import re
import pickle
import json
import random
import time
import numpy as np
import pyarrow # need this before torch!
import torch

from transformers import AutoTokenizer

from rosemary import parse_kv_from_string, create_string_from_kv
from note_pruning_analysis import lm_output_dir, get_dataset_token_lengths, save_text_viz_for_curriculum

import note_pruning_dpp
import note_pruning_clustering

def save_to_pickle(save_path, output):
    if 'inds' in output:
        print(f'save inds (length = {len(output["inds"])}) to {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_sorted_inds(save_dir, S, sort_by, reverse=False, extra=None):
    save_path = os.path.join(save_dir, f'{sort_by}_{"decr" if reverse else "incr"}.pkl')
    inds = np.argsort(S).tolist()
    if reverse:
        inds = inds[::-1]
    output = {'inds': inds, 'S': S[inds].tolist()}
    if extra:
        output.update(extra)
    save_to_pickle(save_path, output)

    
def sort_kmeans_dist_to_cluster_centers(X, n_clusters, kmeans_type='minibatch_kmeans', dist_fn='l2'):
    """
        dist_fn
            l2: euclidean distance
            cd: cosine distance
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans

    if dist_fn not in ['l2', 'cd']:
        raise ValueError(f'Invalid dist_fn={dist_fn}')

    if kmeans_type == 'auto':
        kmeans_type = 'kmeans' if len(X) <= 100000 else 'minibatch_kmeans'
    if kmeans_type == 'minibatch_kmeans':
        kmeans_cls = MiniBatchKMeans
        # need to increase the batch size! otherwise makes no progress and stops early.
        # https://stackoverflow.com/questions/21447351/minibatchkmeans-parameters
        # - might want to decrease reassignment_ratio for low n_clusters.
        kmeans_fn_kwargs = {'batch_size': 512, 
                            'max_no_improvement': 100,
                            'reassignment_ratio': 1e-4,}
        print(kmeans_fn_kwargs)
    elif kmeans_type == 'kmeans':
        kmeans_cls = KMeans
        kmeans_fn_kwargs = {}
    else:
        raise ValueError(f'Invalid kmeans_type={kmeans_type}')
        
    X = X.astype(np.float64)
    kmeans = kmeans_cls(
        n_clusters=n_clusters, 
        init='k-means++',
        random_state=0,
        n_init=10,
        verbose=True,
        **kmeans_fn_kwargs)

    if dist_fn == 'cd':
        X = X / np.linalg.norm(X, axis=1, ord=2, keepdims=True)
    kmeans.fit(X)
    P = kmeans.cluster_centers_[kmeans.labels_]
    if dist_fn == 'cd':
        P = P / np.linalg.norm(P, axis=1, ord=2, keepdims=True)
        D = 1 - np.sum(X*P, axis=1) # cosine distance!
    else:
        D = np.linalg.norm(X - P, axis=1)
    
    return D, kmeans



def cholesky_jitter(K, jitter=1e-5):
    K[np.diag_indices_from(K)] += jitter
    return K


def sort_dpp_map(X, logP, kernel_type='Kcos'):
    import sys
    sys.path.insert(0, "/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/fast-map-dpp")
    from dpp import dpp
    
    logP = torch.from_numpy(logP).to('cuda')
    P = logP.exp().to('cpu')
    
    N = P.shape[0]
    
    X = torch.from_numpy(X).to('cuda')
    X = torch.nn.functional.normalize(X, dim=-1)
    # S = T@T.T out-of-memory
    # use block-wise matmul to reduce peak memory usage.
    L = []
    for Xn in torch.split(X, 3000):
        L.append((Xn@X.T).to('cpu'))
    S = torch.vstack(L)
    
    if kernel_type == 'Kcos':
        K = S
    elif kernel_type == 'Kcosp':
        K = P.reshape(N,1)*S*P.reshape(1,N)
    elif kernel_type == 'Kcos1np':
        K = (1-P).reshape(N,1)*S*(1-P).reshape(1,N)
    else:
        raise ValueError(f'Invalid kernel_type={kernel_type}')
        
    K = K.numpy()
    K = cholesky_jitter(K, jitter=1e-3)

    inds = dpp(K, N)
    if len(inds) != N:
        print(f'dpp map len(indices)={len(inds)} != {N} = N')
        
    return inds 


def sort_dpp_map_memefficient(X, logP, kernel_type='Kcos', torch_compile=False):
    """O(N) memory instead of O(N^2) memory, due to lazily evalute kernel matrix."""
    import sys
    sys.path.insert(0, "/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/fast-map-dpp")
    from dpp import dpp_lazy

    logP = torch.from_numpy(logP).to('cuda')
    P = logP.exp()

    N = P.shape[0]

    if torch_compile:
        X = X / np.linalg.norm(X, axis=-1, ord=2, keepdims=True)
    else:
        X = torch.from_numpy(X).to('cuda')
        X = torch.nn.functional.normalize(X, dim=-1)

    jitter = 1e-3

    def kernel_matrix_ith_row(i):
        """Returns i-th row of kernel matrix `K`"""
        if kernel_type == 'Kcos':
            Ki = X[i]@X.T
        elif kernel_type == 'Kcosp':
            Ki = P[i]*X[i]@X.T*P.reshape(1,N)
        elif kernel_type == 'Kcos1np':
            Ki = (1-P[i])*X[i]@X.T*(1-P.reshape(1,N))
        else:
            raise ValueError(f'kernel_type={kernel_type} not supported')
        Ki = Ki.squeeze()
        Ki[i] += jitter
        if torch_compile:
            return Ki
        else:
            return Ki.to('cpu').numpy()

    def kernel_matrix_diag(): 
        if kernel_type == 'Kcos':
            Kdiag = (X*X).sum(-1)
        elif kernel_type == 'Kcosp':
            Kdiag = (X*X).sum(-1) * (P*P)
        elif kernel_type == 'Kcos1np':
            Kdiag = (X*X).sum(-1) * ((1-P)*(1-P))
        else:
            raise ValueError(f'kernel_type={kernel_type} not supported')
        Kdiag = Kdiag.squeeze()
        Kdiag += jitter
        if torch_compile:
            return Kdiag
        else:
            return Kdiag.to('cpu').numpy()
         
    max_length = min(50000, int(.3*N))
    if torch_compile:
        dpp_lazy = torch.compile(dpp_lazy)
        kernel_matrix_ith_row = torch.compile(kernel_matrix_ith_row)
        kernel_matrix_diag = torch.compile(kernel_matrix_diag)
    inds = dpp_lazy(N, kernel_matrix_ith_row, kernel_matrix_diag, max_length, jitter)
    if len(inds) != N:
        print(f'dpp map len(indices)={len(inds)} != {N} = N')
    
    return inds


def save_prune_results(save_dir, inds, S, pkl_extra, sort_by, model_name, dataset):
    
    if any(sort_by.startswith(x) for x in ['dpp_']):
        output = {'inds': inds}
        if pkl_extra:
            output.update(pkl_extra)
        save_to_pickle(
            save_path=os.path.join(save_dir, f'{sort_by}.pkl'),
            output=output)
    else:
        save_sorted_inds(save_dir, S, sort_by, extra=pkl_extra, reverse=False)
        save_sorted_inds(save_dir, S, sort_by, extra=pkl_extra, reverse=True)

        ## use `note_pruning` to generate scores for curriculum learning.
        if any(sort_by.startswith(x) for x in ['dppmap', 'semdedup']):
            # cases when the reverse ordering does not make sense.
            pacing_fn_list = [sort_by]
        else:
            pacing_fn_list = [sort_by, sort_by+'_neg']
        for pacing_fn in pacing_fn_list:
            curriculum_output_dir = os.path.join('curriculum', model_name, dataset, pacing_fn)
            os.makedirs(curriculum_output_dir, exist_ok=True)
            save_path = os.path.join(curriculum_output_dir, 'scores.pkl')
            output = {'S': -S if pacing_fn.endswith('_neg') else S}
            save_to_pickle(save_path=save_path, output=output)
            save_text_viz_for_curriculum(save_path)


def main(dataset, sort_by, save_dir, model_name, test_run, encode_fn_type):

    from note_pruning_analysis import get_lm_output

    d = get_lm_output(dataset, model_name, encode_fn_type=encode_fn_type, return_text_embedding=True)
    if test_run:
        d = {k: v[:1000] for k, v in d.items()}
        
    # some entries are nan, impute with mean value.
    N = d['text_embedding'].shape[0]

    pkl_extra = {}
    inds = None

    t0 = time.time()
    if any(sort_by.startswith(x) for x in [
            'log_prob', 
            'el2n',  # el2n_agg={l2n|mean}
            'logit_margin', 
            'grad',  # grad_{loraB|qkv|all|last}_l2n
        ]):
        if sort_by not in d:
            print(f'sort_by={sort_by} not in model output: ({dataset}, {model_name})')
            return
        S = np.nan_to_num(d[sort_by], nan=np.nanmean(d[sort_by])).squeeze()
    elif sort_by.startswith('ifd'):
        print(f'encode_fn_type={encode_fn_type} not used! since sort_by={sort_by}')
        S = note_pruning_dpp.get_ifd_and_pmi(dataset, model_name)['ifd']
    elif sort_by.startswith('log_pmi'):
        print(f'encode_fn_type={encode_fn_type} not used! since sort_by={sort_by}')
        S = note_pruning_dpp.get_ifd_and_pmi(dataset, model_name)['log_pmi']
    elif sort_by.startswith('random'):
        match = re.search(r's=(\d+)', sort_by)
        seed = int(match.group(1))
        np.random.seed(seed)
        S = np.random.rand(N)
        assert(S.shape == np.unique(S).shape)
    if sort_by.startswith('kmeans'):
        dist_fn = 'l2' if sort_by.startswith('kmeansl2') else 'cd'
        match = re.search(r'nc=(\d+)', sort_by)
        n_clusters = int(match.group(1)) if match else None
        match = re.search(r'emb=([^_]+)', sort_by)
        embed_type = re.sub(r'[+]', '_', match.group(1)) if match else 'text_embedding'
        if embed_type not in set(d.keys()).intersection(set(['text_embedding', 'grad_rp_loraB'])):
            raise ValueError(f'Invalid embed_type = {embed_type}')
        emb = d[embed_type]
        print(f'Running kmeans(n_clusters={n_clusters}) {{ {embed_type} }} to compute {"euclidean" if dist_fn == "l2" else "cosine"} distance to cluster centers.')
        S, kms = sort_kmeans_dist_to_cluster_centers(emb, n_clusters, dist_fn=dist_fn)
        pkl_extra['kmeans'] = kms
    elif sort_by.startswith('semdedup'):
        kvs = parse_kv_from_string(sort_by)
        md = kvs['md']
        if (md == 'mpnet' and model_name != 'all-mpnet-base-v2') or \
        (md == 'bge' and model_name != 'bge-large-en-v1.5') or \
        (md == 'llama7b' and not model_name.lower().startswith('llama-7b')) or \
        (md == 'mistral7b' and not model_name.lower().startswith('mistral-7b')):
            raise ValueError(f'md={md} does not match with model_name={model_name}')
        clustering_fn = create_string_from_kv(
            {k: v for k, v in kvs.items() if k in ['cl', 'nc', 'bsz', 'ms', 'emb']})
        dist = kvs['dist']
        assert(dist in ['cd', 'l2'])
        embed_type = re.sub(r'[+]', '_', kvs['emb'])
        save_dir_clustering = os.path.join('clustering', encode_fn_type, model_name, dataset, clustering_fn)
        os.makedirs(save_dir_clustering, exist_ok=True)
        # normalize embeddings to unit norm if the model that generated the embeddings does the 
        # same, e.g., mpnet, bge, or if using spherical kmeans clustering.
        if any(x in model_name for x in ['mpnet', 'bge']) or 'kmeansfaisscd' in clustering_fn:
            normalize_embeddings = True
        else:
            normalize_embeddings = False
        kwargs = {
            'model_name': model_name,
            'dataset': dataset,
            'encode_fn_type': encode_fn_type,
            'clustering_fn': clustering_fn,
            'embed_type': embed_type,
            'normalize_embeddings': normalize_embeddings,
            'first_N': None,
            'save_dir': save_dir_clustering,
        }
        print(f'Calling note_pruning_clustering.main with kwargs={json.dumps(kwargs, indent=4)}')
        X, Y, C = note_pruning_clustering.main(**kwargs)
        print('Apply SemDeDup to discard duplicates.')
        S = note_pruning_clustering.semdedup(X, Y, dist=dist, device='cuda')
    elif sort_by.startswith('dedup'):
        kvs = parse_kv_from_string(sort_by)
        md = kvs['md']
        if (md == 'mpnet' and model_name != 'all-mpnet-base-v2') or \
        (md == 'bge' and model_name != 'bge-large-en-v1.5') or \
        (md == 'llama7b' and not model_name.lower().startswith('llama-7b')) or \
        (md == 'mistral7b' and not model_name.lower().startswith('mistral-7b')):
            raise ValueError(f'md={md} does not match with model_name={model_name}')
        if md in ['mpnet', 'bge']:
            normalize_embeddings = True
            dist = 'cd'
        else:
            normalize_embeddings = False
            dist = 'l2'
        embed_type = re.sub(r'[+]', '_', kvs['emb'])
        X = d[embed_type]
        Y = np.zeros(X.shape[0])
        S = note_pruning_clustering.semdedup(X, Y, dist=dist, device='cpu')
    elif sort_by.startswith('dpp_'):
        match = re.search(r'k=(\w+)', sort_by)
        kernel_type = match.group(1) if match else None
        match = re.search(r'emb=([^_]+)', sort_by)
        embed_type = re.sub(r'[+]', '_', match.group(1)) if match else 'text_embedding'
        emb = d[embed_type]
        log_prob = d['log_prob']
        inds = sort_dpp_map_memefficient(emb, log_prob, kernel_type=kernel_type, torch_compile=False)
    elif sort_by.startswith('dppmap_'):
        kvs = parse_kv_from_string(sort_by)
        if kvs['k'] == 'vmf':
            kernel_kwargs = {'gamma': kvs['gamma']}
        elif kvs['k'] == 'rbf':
            kernel_kwargs = {'sigma': kvs['sigma']}
        else:
            kernel_kwargs = {}
        kwargs = {
            'dppmap_type': 'dppmap',
            'dataset': dataset,
            'kernel_type': kvs['k'],
            'kernel_embed_model': kvs['kmd'],
            'kernel_embed_type': re.sub(r'[+]', '_', kvs['kemb']) if 'kemb' in kvs else 'text_embedding',
            'kernel_kwargs': kernel_kwargs,
            'quality_score_type': re.sub(r'[+]', '_', kvs['q']) if 'q' in kvs else None,
            'quality_score_embed_model': kvs.get('qmd', None),
            'theta': kvs.get('theta', 0.), # defaults to just diversity no quality
            'device': 'cuda',
            'max_length': min(20_000, N), # balance finish job within 6 hrs with wanting to prune a lot. meaning resulting scores will only be valid up to 20k for large datasets.
            'run_name': sort_by,
        }
        print(f'Calling note_pruning_dpp.compute_dppmap with kwargs={json.dumps(kwargs, indent=4)}')
        S, output = note_pruning_dpp.compute_dppmap(**kwargs)
        pkl_extra['info'] = output
    elif sort_by.startswith('dppmapbd'):
        kvs = parse_kv_from_string(sort_by)
        md = kvs['kmd']
        if (md == 'mpnet' and model_name != 'all-mpnet-base-v2') or \
            (md == 'bge' and model_name != 'bge-large-en-v1.5') or \
            (md == 'llama7b' and not model_name.lower().startswith('llama-7b')) or \
            (md == 'mistral7b' and not model_name.lower().startswith('mistral-7b')):
            raise ValueError(f'md={md} does not match with model_name={model_name}')
        if kvs['k'] == 'vmf':
            kernel_kwargs = {'gamma': kvs['gamma']}
        elif kvs['k'] == 'rbf':
            kernel_kwargs = {'sigma': kvs['sigma']}
        else:
            kernel_kwargs = {}
        kwargs = {
            'dppmap_type': 'dppmapbd',
            'dataset': dataset,
            'kernel_type': kvs['k'],
            'kernel_embed_model': kvs['kmd'],
            'kernel_embed_type': re.sub(r'[+]', '_', kvs['kemb']) if 'kemb' in kvs else 'text_embedding',
            'kernel_kwargs': kernel_kwargs,
            'quality_score_type': re.sub(r'[+]', '_', kvs['q']) if 'q' in kvs else None,
            'quality_score_embed_model': kvs.get('qmd', None),
            'theta': kvs.get('theta', 0.), # defaults to just diversity no quality
            'device': 'cuda',
            'max_length': 5_000, # per-cluster max length. 
            'run_name': sort_by,
        }
        clustering_fn = create_string_from_kv({
            'cl': kvs.get('cl', 'kmeansfaisscd'),
            'md': kwargs['kernel_embed_model'],
            'emb': kvs['kemb'] if 'kemb' in kvs else 'text+embedding',
            'nc': kvs['nc'],
        })
        save_dir_clustering = os.path.join(
            'clustering', encode_fn_type, model_name, dataset, clustering_fn)
        os.makedirs(save_dir_clustering, exist_ok=True)
        clustering_data_path = os.path.join(save_dir_clustering, 'data.pkl')
        if not os.path.isfile(clustering_data_path):
            normalize_embeddings = True if \
                (any(x in model_name for x in ['mpnet', 'bge']) or 'kmeansfaisscd' in clustering_fn) else False
            kwargs_clustering = {
                'model_name': model_name,
                'dataset': dataset,
                'encode_fn_type': encode_fn_type,
                'clustering_fn': clustering_fn,
                'embed_type': kwargs['kernel_embed_type'],
                'normalize_embeddings': normalize_embeddings,
                'first_N': None,
                'save_dir': save_dir_clustering,
            }
            print(f'Calling note_pruning_clustering.main with kwargs={json.dumps(kwargs_clustering, indent=4)}')
            X, Y, C = note_pruning_clustering.main(**kwargs_clustering)
        else:
            with open(clustering_data_path, 'rb') as f:
                data = pickle.load(f)
            Y = data['Y']
        print(f'Calling note_pruning_dpp.compute_dppmap with kwargs={json.dumps(kwargs, indent=4)}')
        kwargs.update({'Y': Y})
        S, output = note_pruning_dpp.compute_dppmap(**kwargs)
        pkl_extra['info'] = output
    elif sort_by.startswith('rho'):
        if sort_by == 'rhov1': 
            model_names = ['mistral-7b+lora:r=256:a=256',
                           'mistral-7b-ultrachat200k-v1+lora:r=256:a=256']
            assert(model_name == model_names[0])
        else:
            raise ValueError(f'sort_by={sort_by} not implemented.')
        assert(len(model_names) == 2)
        ds = []
        for x in model_names:
            ds.append(get_lm_output(dataset, x, return_text_embedding=False, fill_nan=False))
        ks = [set(d.keys()) for d in ds]
        ks = ks[0] & ks[1]
        for k in ks:
            S0 = ds[0][k]
            S1 = ds[1][k]
            # handle nan entries properly.
            nan_mask = np.logical_or(np.isnan(S0), np.isnan(S1))
            S = np.subtract(S0, S1)
            S[nan_mask] = np.nan
            S = S.squeeze()
            save_prune_results(save_dir, None, S, {}, f'{sort_by}_{k}', model_name, dataset)
    elif sort_by.startswith('numtoks'):
        if 'llama' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('results/baselines/huggyllama/llama-7b', use_fast=False)
        elif 'mistral' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('results/baselines/mistralai/Mistral-7B-v0.1', use_fast=False)
        else:
            raise ValueError('Need to supply appropriate tokenizer to count token lengths,')
        d = get_dataset_token_lengths(dataset, tokenizer)

        d['total_len'] = d['input_len'] + d['output_len']
        for k in ['input', 'output', 'total']:
            S = d[f'{k}_len']
            save_prune_results(save_dir, None, S, {}, f'{sort_by}_{k}', model_name, dataset)


    t1 = time.time()
    print(f'Rank datapoints with {sort_by} took {t1-t0:.2f} seconds.')

    if not any(sort_by.startswith(x) for x in ['rho', 'numtoks']):
        save_prune_results(save_dir, inds, S, pkl_extra, sort_by, model_name, dataset)



if __name__ == '__main__':
    """
    python note_explore_data_pruning.py --dataset lima --sort_by prob --test_run
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lima")
    parser.add_argument("--sort_by", type=str, default="prob")
    parser.add_argument("--model_name", type=str, default="llama-7b")
    parser.add_argument("--save_dir", type=str, default="/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/")
    parser.add_argument("--encode_fn_type", type=str, default="sft")
    parser.add_argument("--test_run", action='store_true', default=False)
    args = parser.parse_args()

    print(json.dumps(vars(args), indent=2))

    # args.save_dir = os.path.join(args.save_dir, args.model_name, args.dataset)
    os.makedirs(args.save_dir, exist_ok=True)

    main(**vars(args))