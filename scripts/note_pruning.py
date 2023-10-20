import os
import re
import pickle
import json
import random
import time
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
        X = X / np.linalg.norm(X, axis=1, ord=2)[:, np.newaxis]
    kmeans.fit(X)
    P = kmeans.cluster_centers_[kmeans.labels_]
    if dist_fn == 'cd':
        P = P / np.linalg.norm(P, axis=1, ord=2)[:, np.newaxis]
        D = 1 - np.sum(X*P, axis=1) # cosine distance!
    else:
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
        print(f'dpp map len(indices)={len(inds)} != {N} = N')
        
    return inds 


def sort_dpp_map_memefficient(X, logP, kernel_type='Kcos'):
    """O(N) memory instead of O(N^2) memory, due to lazily evalute kernel matrix."""
    import sys
    sys.path.insert(0, "/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/fast-map-dpp")
    from dpp import dpp_lazy

    logP = torch.from_numpy(logP).to('cuda')
    P = logP.exp()

    N = P.shape[0]

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
        return Kdiag.to('cpu').numpy()
         
    max_length = min(30000, int(.3*N))
    inds = dpp_lazy(N, kernel_matrix_ith_row, kernel_matrix_diag, max_length, jitter)
    if len(inds) != N:
        print(f'dpp map len(indices)={len(inds)} != {N} = N')
    
    return inds


def prune_data(dataset, sort_by, save_dir, lm_output_dir, test_run):

    save_path = os.path.join(lm_output_dir, f'{dataset}.pkl')
    with open(save_path, 'rb') as f:
        d = pickle.load(f)
    if test_run:
        d = {k: v[:1000] for k, v in d.items()}
        
    # some entries are nan, impute with mean value.
    N = d['text_embedding'].shape[0]
    log_prob = np.nan_to_num(d['log_prob'], nan=np.nanmean(d['log_prob'])).squeeze()

    t0 = time.time()
    if any(sort_by.startswith(x) for x in [
            'log_prob', 
            'el2n',  # el2n_agg={l2n|mean}
            'logit_margin', 
            'grad',  # grad_{loraB|qkv|all|last}_l2n
        ]):
        if sort_by not in d:
            print(f'sort_by={sort_by} not in lm_output_dir={lm_output_dir}')
            return
        S = np.nan_to_num(d[sort_by], nan=np.nanmean(d[sort_by])).squeeze()
    elif sort_by.startswith('random'):
        match = re.search(r's=(\d+)', sort_by)
        seed = int(match.group(1))
        random.seed(seed)
        inds = list(range(N))
        random.shuffle(inds)
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
        S = sort_kmeans_dist_to_cluster_centers(emb, n_clusters, dist_fn=dist_fn)
    elif sort_by.startswith('dpp'):
        match = re.search(r'k=(\w+)', sort_by)
        kernel_type = match.group(1) if match else None
        match = re.search(r'emb=([^_]+)', sort_by)
        embed_type = re.sub(r'[+]', '_', match.group(1)) if match else 'text_embedding'
        if embed_type not in set(d.keys()).intersection(set(['text_embedding', 'grad_rp_loraB'])):
            raise ValueError(f'Invalid embed_type = {embed_type}')
        emb = d[embed_type]
        inds = sort_dpp_map(emb, log_prob, kernel_type=kernel_type)
    t1 = time.time()
    print(f'Rank datapoints with {sort_by} took {t1-t0:.2f} seconds.')

    if any(sort_by.startswith(x) for x in ['dpp', 'random']):
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
    parser.add_argument("--lm_output_dir", type=str, default="/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/model_outputs/llama-7b")
    parser.add_argument("--save_dir", type=str, default="/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b/")
    parser.add_argument("--test_run", action='store_true', default=False)
    args = parser.parse_args()

    print(json.dumps(vars(args), indent=2))

    args.save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(args.save_dir, exist_ok=True)

    prune_data(**vars(args))