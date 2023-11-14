import os
import re
import json
import pickle
import time

import numpy as np
import scipy
import pandas as pd
import sklearn

import pyarrow
import torch

from functools import partial



from note_pruning_analysis import (
    get_lm_output,
    get_dataset,
)


def clustering_sort_by_cluster_size(Y, C, descending=True):
    """Sort and reassign cluster labels s.t. such that 
        lower indexed cluster center class has more points.
        
        ```
        X = np.array([[1,1,1],[2,2,2],[3,3,3]])
        Y = np.array([0,1,1])
        C = np.array([[0,0,0],[1,1,1]])
        Y, C = clustering_sort_by_cluster_size(Y, C)
        assert(np.array_equal(Y, np.array([1,0,0])))
        assert(np.array_equal(C, np.array([[1,1,1],[0,0,0]])))
        classes, class_counts = np.unique(Y, return_counts=True)
        assert(sorted(class_counts, reverse=True) == list(class_counts))
        ```
    """
    # Get the unique cluster labels and their counts
    classes, class_counts = np.unique(Y, return_counts=True)
    inds = np.argsort(class_counts)
    if descending:
        inds = inds[::-1]
    classes = classes[inds]

    Y_sorted = np.zeros_like(Y)
    C_sorted = np.zeros_like(C)
    for class_new, class_old in enumerate(classes):
        Y_sorted[Y == class_old] = class_new
        C_sorted[class_new] = C[class_old]
        
    return Y_sorted, C_sorted



def pairwise_cosine_similarity(a, b, eps=1e-8):
    """Compute cosine similarity 
        a (N, D)
        b (M, D)
        Returns (N, M)
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = a.norm(dim=1)[:,None]
    b_norm = b.norm(dim=1)[:,None]
    a_norm = a / torch.max(a_norm, eps * torch.ones_like(a_norm))
    b_norm = b / torch.max(b_norm, eps * torch.ones_like(b_norm))
    S = torch.mm(a_norm, b_norm.transpose(0, 1))
    return S


def pairwise_cosine_distance(a, b, eps=1e-8):
    """Compute cosine distance = 1 - cosine_similarity
        a (N, D)
        b (M, D)
        Returns (N, M)
    """
    return 1 - pairwise_cosine_similarity(a, b, eps=eps)
    


def clustering_dist_to_centroids(X, Y, C, dist='cd', device='cuda'):
    """Compute distances of each point to their centroid.

    each cluster
    cluster size, mean/std of distance to centroids

    each point
    cluster assignment, distance to centroid, 
    """
    n_clusters = C.shape[0]

    if X.shape[0]!=Y.shape[0]:
        raise ValueError('Each data point should have a cluster assignment')

    if dist == 'cd':
        dist_fn = pairwise_cosine_distance
    elif dist == 'l2':
        dist_fn = lambda x1, x2: torch.cdist(x1, x2, p=2.0)
    else:
        raise ValueError(f'dist_fn={dist_fn} not supported.')

    # faster processing on gpu
    X = torch.from_numpy(X).float().to(device)
    C = torch.from_numpy(C).float().to(device)
    # Y = torch.from_numpy(Y).float().to(device)

    D = torch.zeros(X.shape[0])
    for i in range(n_clusters):
        mask = (Y==i)
        C_i = C[i]
        X_i = X[mask]
        C_i = C_i.reshape(1, -1) if C_i.ndim==1 else C_i
        D_i = dist_fn(C_i, X_i).squeeze()
        D[mask] = D_i.to('cpu')
    D = D.tolist()
    return D


def clustering_knn_withincluster(X, Y, k=5, dist='cd', device='cuda'):
    """Find k-nearest-neighbor for each data point within the same cluster.
        Returns the indices and distances of the k-nearest-neighbors.
    """

    clusters = sorted(list(np.unique(Y)))

    if dist == 'cd':
        dist_fn = pairwise_cosine_distance
    elif dist == 'l2':
        dist_fn = lambda x1, x2: torch.cdist(x1, x2, p=2.0)
    else:
        raise ValueError(f'dist_fn={dist_fn} not supported.')
    
    X = torch.from_numpy(X).float().to(device)
    Y = torch.from_numpy(Y)

    # indices of k-nearest neighbors
    I = torch.zeros(X.shape[0], k, dtype=torch.int64)
    # distance to k-nearest neighbors
    D = torch.zeros(X.shape[0], k, dtype=torch.float32)
    for i in clusters:
        mask = (Y==i)
        inds_global = torch.nonzero(mask).squeeze()
        Xi = X[mask]
        Di = dist_fn(Xi, Xi)
        values, indices = torch.topk(Di, k + 1, largest=False, dim=1)
        I[mask] = inds_global[indices[:,1:].to('cpu')]
        D[mask] = values[:,1:].to('cpu')

    X = X.to('cpu')
    I = I.tolist()
    D = D.tolist()
        
    return I, D


def clustering_algorithm_scores(X, Y):

    sample_size = min(10_000, len(X))
    scores_fn = {
        'silhouette_score_cd': partial(sklearn.metrics.silhouette_score, metric='cosine', sample_size=sample_size, random_state=0),
        'silhouette_score_l2': partial(sklearn.metrics.silhouette_score, metric='euclidean', sample_size=sample_size, random_state=0),
        'variance_ratio': sklearn.metrics.calinski_harabasz_score,
        'davies_bouldin_index': sklearn.metrics.davies_bouldin_score,
    }

    scores = {}
    for k, fn in scores_fn.items():
        scores[k] = float(fn(X, Y))

    return scores


def clustering_run(run_name, X):
    from sklearn.cluster import KMeans, MiniBatchKMeans

    match = re.search(r'cl=([^_]+)', run_name)
    clustering_algo = match.group(1)
    match = re.search(r'nc=([^_]+)', run_name)
    n_clusters = int(match.group(1))

    if clustering_algo.startswith('kmeans'):
        if clustering_algo == 'kmeans':
            clustering_model = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init="auto",
                random_state=0,
                verbose=True,
            )
        elif clustering_algo == 'kmeansminibatch':
            match = re.search(r'bsz=([^_]+)', run_name)
            batch_size = int(match.group(1)) if match else 1024
            clustering_model = MiniBatchKMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init="auto",
                random_state=0,
                verbose=True,
                batch_size=batch_size,
                max_no_improvement=100,
                reassignment_ratio=1e-4,
            )
        clustering_model.fit(X)
        Y = clustering_model.labels_
        C = clustering_model.cluster_centers_
    else:
        raise ValueError(f'clustering_algo={clustering_algo} not implemented.')
    
    ## sort by decreasing cluster size by default
    Y, C = clustering_sort_by_cluster_size(Y, C)
    
    return Y, C, clustering_model



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



def clustering_compute_and_save_results(X, Y, C, ds, save_dir):
    """
        - top-k closest text/distance for each cluster (viz text near centroids)
        - top-k closest neighbor within each cluster (viz text with similar embeddings)
        - 2d tsne of clustering algorithm
    """
    data = {
        'data_ind': np.arange(len(Y)),
        'cluster_assignment': Y,
        'cent_dist_l2': clustering_dist_to_centroids(X, Y, C, device='cuda', dist='l2'),
        'cent_dist_cd': clustering_dist_to_centroids(X, Y, C, device='cuda', dist='cd'),
    }
    for dist in ['l2', 'cd']:
        I, D = clustering_knn_withincluster(X, Y, device='cuda', dist=dist)
        data.update({f'knn_inds_{dist}': I, 
                     f'knn_dist_{dist}': D,})

    df = pd.DataFrame(data)

    cent_dist_l2_vs_cd_spearmanr = float(scipy.stats.spearmanr(
        df['cent_dist_l2'].to_numpy(), df['cent_dist_cd'].to_numpy()).statistic)
    print(f'cent_dist_l2_vs_cd_spearmanr: {cent_dist_l2_vs_cd_spearmanr}')

    with open(os.path.join(save_dir, 'data.pkl'), 'wb') as f:
        output = {'df': df, 'C': C, 'Y': Y}
        pickle.dump(output, f)


    ## visualize text near cluster centers

    cluster_sizes = np.unique(Y, return_counts=True)[1].tolist()

    for dist in ['l2', 'cd']:
        df_topk = df.sort_values(by=['cluster_assignment', f'cent_dist_{dist}']) \
                    .groupby('cluster_assignment') \
                    .head(20)
        output = []
        for i, count in enumerate(cluster_sizes):
            dfi = df_topk[df_topk['cluster_assignment']==i]
            output.append(
                {
                    'cluster': i,
                    'cluster_size': count,
                    'examples': [{'text': ds[ind]['text'], f'cent_dist_{dist}': x}
                         for ind, x in zip(dfi['data_ind'].to_list(), dfi[f'cent_dist_{dist}'].to_list())]
                }
            )

        with open(os.path.join(save_dir, f"text_clusterwise_topk_dist={dist}.json"), 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    ## visualize text with similar embeddings

    n_examples_per_cluster = 5
    k_neighbors = 3

    for dist in ['l2', 'cd']:

        output = []
        for i, count in enumerate(cluster_sizes):
            dfi = df[df['cluster_assignment']==i]
            dfi = dfi.sample(n=min(n_examples_per_cluster, len(dfi)), random_state=0)
            ID = dfi.apply(lambda row: ([row['data_ind']]+row[f'knn_inds_{dist}'],
                                         [0]+row[f'knn_dist_{dist}'],), axis=1).tolist()
            output += [
                {
                    'cluster': i,
                    'cluster_size': count,
                    'examples': [{'text': ds[ind]['text'], f'knn_dist_{dist}': v}
                                for ind, v in zip(inds[:k_neighbors], vals[:k_neighbors])]
                } for inds, vals in ID
            ]

        with open(os.path.join(save_dir, f'text_knn_dist={dist}.json'), 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)




def main(
    model_name,
    dataset,
    encode_fn_type,
    clustering_fn,
    normalize_embeddings,
    first_N,
    save_dir,
):

    ds = get_dataset(dataset, processed=True)
    if encode_fn_type == 'input':
        def get_user_prompt_fn(example):
            example['text'] = example['messages'][0]['content']
            return example
        ds = ds.map(get_user_prompt_fn, num_proc=16)

    d = get_lm_output(dataset, 
                  model_name, 
                  encode_fn_type=encode_fn_type,
                  return_text_embedding=True,)
    X = d['text_embedding']
    if normalize_embeddings:
        X = X / np.linalg.norm(X, axis=-1, keepdims=True)

    if first_N:
        ds = ds.select(range(first_N))
        X = X[:first_N]


    info = {}
    info['N'] = len(X)
    info['dataset'] = 'dataset'
    info['model_name'] = model_name
    info['encode_fn_type'] = encode_fn_type

    t0 = time.time()
    Y, C, clustering_model = clustering_run(clustering_fn, X)
    info['time_elapsed'] = time.time()-t0
    info['scores'] = {}
    info['scores'].update({'inertia': clustering_model.inertia_})
    info['scores'].update(clustering_algorithm_scores(X, Y))
    info['cluster_sizes'] = np.unique(Y, return_counts=True)[1].tolist()

    with open(os.path.join(save_dir, 'info.json'), 'w') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)

    clustering_compute_and_save_results(X, Y, C, ds=ds, save_dir=save_dir)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--dataset', type=str, default='wizardlm')
    parser.add_argument('--encode_fn_type', type=str, default='input')
    parser.add_argument('--clustering_fn', type=str, default='cl=kmeans_nc=100')
    parser.add_argument("--normalize_embeddings", action='store_true', default=False)
    parser.add_argument("--first_N", type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='clustering/input/all-mpnet-base-v2/wizardlm/cl=kmeans_nc=100')

    args = parser.parse_args()

    print(json.dumps(vars(args), indent=2))

    os.makedirs(args.save_dir, exist_ok=True)
    main(**vars(args))
