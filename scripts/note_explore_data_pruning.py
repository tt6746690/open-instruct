import os
import re
import pickle
import json

import numpy as np
import torch


def save_to_pickle(save_path, output):
    if 'inds' in output:
        print(f'save inds (length = {len(output["inds"])}) to {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_sorted_inds(save_dir, S, sort_by, reverse=False):
    save_path = os.path.join(save_dir, f'{sort_by}_{"decr" if reverse else "incr"}.pkl')
    inds = np.argsort(S).tolist()
    if reverse:
        inds = inds[::-1]
    output = {'inds': inds, 'S': S[inds].tolist()}
    save_to_pickle(save_path, output)

    
def sort_kmeans_l2_to_prototypes(X, n_clusters):
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto', verbose=True)
    kmeans.fit(X)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    P = cluster_centers[cluster_labels]
    D = np.linalg.norm(X - P, axis=1)
    
    return D


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
    for Xn in torch.split(X, 10000):
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
        raise ValueError(f'dpp map len(indices)={len(inds)} != {N} = N')
        
    return inds


def prune_data(dataset, sort_by, save_dir, lm_output_dir, test_run):

    save_path = os.path.join(lm_output_dir, f'{dataset}.pkl')
    with open(save_path, 'rb') as f:
        d = pickle.load(f)
    # some entries are nan, impute with mean value.
    d['log_probs'] = np.nan_to_num(d['log_probs'], nan=np.nanmean(d['log_probs']))
    text_embeddings = d['text_embeddings']
    log_probs = d['log_probs'].squeeze()

    if test_run:
        text_embeddings = text_embeddings[:1000]
        log_probs = log_probs[:1000]

    if sort_by.startswith('kmeansl2'):
        match = re.search(r'(?<=\=)\d+', sort_by)
        n_clusters = int(match.group()) if match else None
        S = sort_kmeans_l2_to_prototypes(text_embeddings, n_clusters)
    elif sort_by == 'prob':
        S = log_probs
    elif sort_by.startswith('dpp'):
        match = re.search(r'k=(\w+)', sort_by)
        kernel_type = match.group(1) if match else None  
        inds = sort_dpp_map(text_embeddings, log_probs, kernel_type=kernel_type)
    else:
        raise ValueError('sort_by={sort_by} not supported')

        
    if sort_by.startswith('dpp'):
        save_to_pickle(
            save_path=os.path.join(save_dir, f'{sort_by}.pkl'),
            output={'inds': inds})
    else:
        save_sorted_inds(save_dir, S, sort_by, reverse=False)
        save_sorted_inds(save_dir, S, sort_by, reverse=True)


if __name__ == '__main__':
    """
    python note_explore_data_pruning.py --dataset lima --sort_by prob --test_run
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lima")
    parser.add_argument("--sort_by", type=str, default="prob")
    parser.add_argument("--lm_output_dir", type=str, default="/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/llama-7b_outputs/")
    parser.add_argument("--save_dir", type=str, default="/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b/")
    parser.add_argument("--test_run", action='store_true', default=False)
    args = parser.parse_args()

    print(json.dumps(vars(args), indent=2))

    args.save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(args.save_dir, exist_ok=True)

    prune_data(**vars(args))