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
from note_pruning_analysis import lm_output_dir, get_dataset_token_lengths, save_text_viz_for_curriculum, get_tokenizer_name_or_path, get_fast_tokenizer, save_to_pickle
from note_pruning_analysis import get_lm_output, md_to_model_name, get_full_model_name

import note_pruning_dpp
import note_pruning_clustering





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


def parse_sort_by_and_compute_dppmap(sort_by, dataset):
    kvs = parse_kv_from_string(sort_by)

    if kvs['k'] == 'vmf':
        kernel_kwargs = {'gamma': kvs['gamma']}
    elif kvs['k'] == 'rbf':
        kernel_kwargs = {'gamma': kvs['gamma']}
    elif kvs['k'] == 'lin':
        kernel_kwargs = {'gamma': kvs.get('gamma', 1.)}
    else:
        kernel_kwargs = {}

    if dataset != 'ultrachat15':
        max_length = 55_000
    else:
        max_length = 20_000

    prespecified_ordering = re.sub(':', '_', re.sub('@', '=', kvs['ord'])) if 'ord' in kvs else None
    if prespecified_ordering:
        ## just run for target_size
        assert('auto' in kvs['gamma'])
        match = re.search(r'auto([\d.e+-]+)', sort_by)
        max_length = int(match.group(1))
        ## fetch prev run autotune-ed gamma, and use it for this run
        kvs_ = {k: v for k, v in kvs.items() if k not in [0, 'ord']}
        sort_by_without_ord = kvs[0]+'_'+create_string_from_kv(kvs_)
        d = get_dppmap_autotune_gamma_search_result(sort_by_without_ord, dataset)
        kernel_kwargs = {'gamma': d['gamma']}
        sort_by = re.sub(r'auto\d+', str(d['gamma']), sort_by)

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
        'max_length': max_length, # balance finish job within 6 hrs with wanting to prune a lot. meaning resulting scores will only be valid up to 20k for large datasets.
        'prespecified_ordering': prespecified_ordering,
        'run_name': sort_by,
    }
    print(f'Calling note_pruning_dpp.compute_dppmap with kwargs={json.dumps(kwargs, indent=4)}')
    return note_pruning_dpp.compute_dppmap(**kwargs)



def get_dppmap_autotune_gamma_search_result(sort_by, dataset, rel_tol=0.01):
    
    match = re.search(r'gamma=auto([\d.e+-]+)', sort_by)
    target_size = int(match.group(1))

    df = note_pruning_dpp.get_dppmap_run_info(
        re.sub('gamma=auto([\d.e+-]+)', 'gamma=*', sort_by),
        dataset)
    M_closest, gamma_closest = df.loc[(df['M']-target_size).abs().idxmin()][['M', 'gamma']]
    if np.abs(((M_closest-target_size)/target_size)) < rel_tol:
        print(f'[autotune gamma result] Found gamma={gamma_closest} / M={M_closest} '
              f'is within {rel_tol} of target_size={target_size}')
    d = df.loc[(df['M']-target_size).abs().idxmin()].to_dict()
    return d



def parse_sort_by_and_compute_dppmap_autotune_gamma(
        sort_by,
        dataset,
        rel_tol=0.01,
        max_iterations=20,
    ):
    """run dppmap, but auto-tunes `gamma` to reach target_size. """

    match = re.search(r'gamma=auto([\d.e+-]+)', sort_by)
    target_size = int(match.group(1))


    results = None, None
    for it in range(max_iterations):
        # 
        # - Since gamma and subset_size roughly linear. Fit a 
        # linear function to predict subset_size from gamma from past runs.
        # - Since large gamma -> large compute cost, try small gamma first!
        # instead of using bisection where you are doing binary search over [low, high]
        # 
        
        ## Initialize `gamma`
        df = note_pruning_dpp.get_dppmap_run_info(
            re.sub('gamma=auto([\d.e+-]+)', 'gamma=*', sort_by), 
            dataset)
        d = df[['gamma', 'M']].to_dict(orient='list')

        if it == 0:
            if len(df) >= 1:
                M_closest, gamma = df.loc[(df['M']-target_size).abs().idxmin()][['M', 'gamma']]
            else:
                M_closest, gamma = None, 1e-8
            print(f'[autotune gamma] Set initial gamma={gamma} / M={M_closest} to reach target_size={target_size}')

        if len(df) > 0:
            M_closest, gamma_closest = df.loc[(df['M']-target_size).abs().idxmin()][['M', 'gamma']]
            if np.abs(((M_closest-target_size)/target_size)) < rel_tol:
                print(f'[autotune gamma] Terminate since gamma={gamma_closest} / M={M_closest} '
                    f'is within {rel_tol} of target_size={target_size}')
                return results

        if len(df) >= 2:
            # Fit quadratic fn: M = quadratic_fn(gamma)
            # give more weight to closeby gamma & M
            deg = 1
            xs, ys = np.array(d['gamma']), np.array(d['M'])
            w = 1/np.abs(target_size-ys)

            # switch to bisection if target_size is inbetween two previously found subset sizes.
            # to prevent estimated function being affected by many points somewhat close to `target_size`
            inbetween = np.any(ys > target_size) and np.any(ys < target_size)
            if inbetween:
                c = np.zeros_like(ys)
                c[np.argmin(np.array([np.inf if y < target_size else y-target_size for y in ys]))] = 1
                c[np.argmin(np.array([np.inf if y > target_size else target_size-y for y in ys]))] = 1
                w = w*c
            coeff = np.polyfit(xs, ys, deg=deg, w=w)
            print(f'[autotune gamma] Fit fn given data:\n gamma={d["gamma"]}\nM={d["M"]}')
            print(f'[autotune gamma] Fit fn: M = quad_fn(γ), coeff={coeff}')
            if deg == 1:
                slope = coeff[0]
            elif deg == 2:
                slope = 2*coeff[0]*gamma + coeff[1]
            else:
                raise ValueError(f'Invalid deg={deg}')
            if slope <= 1:
                gamma = gamma*2 # increase prev gamma by 2x
            else:
                gamma = np.max((np.poly1d(coeff)-target_size).roots)
                gamma = np.round(gamma, 3 - int(np.floor(np.log10(abs(gamma)))) - 1) # round to 3 sig-dig
                print(gamma)
                gamma= max(min(1., gamma), 1e-8)
        else:
            if len(df) > 0:
                gamma = 2*gamma if target_size > M_closest else .5*gamma
            else:
                gamma = 2*gamma


        results = parse_sort_by_and_compute_dppmap(
            sort_by=re.sub(r'gamma=auto([\d.e+-]+)', f'gamma={gamma}', sort_by), 
            dataset=dataset)
        print(f"[autotune gamma] Iteration {it} tried gamma={gamma}, got M={results[1]['M']}")

    print(f'[autotune gamma] Reached max_iterations without finding a good gamma (gamma={gamma})')
    return None, None



def save_prune_results(save_dir, S, pkl_extra, sort_by, model_name, dataset):

    save_sorted_inds(save_dir, S, sort_by, extra=pkl_extra, reverse=False)
    save_sorted_inds(save_dir, S, sort_by, extra=pkl_extra, reverse=True)

    ## use `note_pruning` to generate scores for curriculum learning.
    if any(sort_by.startswith(x) for x in ['dppmap', 'semdedup', 'dedup']):
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

    

def compute_ranking_copyover(sort_by, dataset, model_name):
    d = get_lm_output(dataset, model_name, encode_fn_type='sft', return_text_embedding=False)
    if sort_by not in d:
        raise ValueError(f'sort_by={sort_by} not in model output: ({dataset}, {model_name})')
    S = d[sort_by]
    S = np.nan_to_num(S, nan=np.nanmean(S)).squeeze()
    return {sort_by: S}
    

def compute_ranking_ifd_and_pmi(dataset, model_name):
    d = note_pruning_dpp.get_ifd_and_pmi(dataset, model_name)
    Sd = {}
    for k in ['ifd', 'log_pmi']:
        Sd[k] = d[k]
    return Sd


def compute_ranking_random(sort_by, dataset, model_name):
    d = get_lm_output(dataset, model_name, encode_fn_type='sft', return_text_embedding=False)
    N = d['log_prob'].shape[0]
    match = re.search(r's=(\d+)', sort_by)
    seed = int(match.group(1))
    np.random.seed(seed)
    S = np.random.rand(N)
    assert(S.shape == np.unique(S).shape)
    return {sort_by: S}


def compute_ranking_kmeans_dist_to_centroids(sort_by, dataset):
    """
        ```
        from note_pruning import compute_ranking_kmeans_dist_to_centroids
        dataset = 'lima'
        sort_by = 'kmeansl2_md=mpnet_emb=text+embedding_nc=30'
        Sd = compute_ranking_kmeans_dist_to_centroids(sort_by, dataset)
        ```
    """
    kvs = parse_kv_from_string(sort_by)
    dist_fn = 'l2' if sort_by.startswith('kmeansl2') else 'cd'
    n_clusters = kvs['nc']
    embed_type = re.sub(r'[+]', '_', kvs['emb'])
    encode_fn_type='input' if kvs['md'] in ['mpnet', 'bge'] else 'sft'
    model_name = get_full_model_name(kvs['md'])
    d = get_lm_output(dataset, model_name, encode_fn_type=encode_fn_type, return_text_embedding=True)
    emb = d[embed_type]
    print(f'Running kmeans(n_clusters={n_clusters}) {{ {model_name}\'s {embed_type} }} to compute {"euclidean" if dist_fn == "l2" else "cosine"} distance to cluster centers.')
    S, kms = sort_kmeans_dist_to_cluster_centers(emb, n_clusters, dist_fn=dist_fn)
    return {sort_by: S}, {'kmeans': kms}


def compute_ranking_semdedup(sort_by, dataset):
    """
        ```
        from note_pruning import compute_ranking_semdedup
        sort_by = 'semdedup_cl=kmeansfaisscd_md=mpnet_dist=cd_emb=text+embedding_nc=200'
        Sd = compute_ranking_semdedup(sort_by, dataset)
        ```
    """
    
    kvs = parse_kv_from_string(sort_by)
    model_name = get_full_model_name(kvs['md'])
    clustering_fn = create_string_from_kv(
        {k: v for k, v in kvs.items() if k in ['cl', 'nc', 'bsz', 'ms', 'emb']})
    dist = kvs['dist']
    assert(dist in ['cd', 'l2'])
    embed_type = re.sub(r'[+]', '_', kvs['emb'])
    encode_fn_type='input' if kvs['md'] in ['mpnet', 'bge'] else 'sft'
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
    
    return {sort_by: S}


def compute_ranking_dedup(sort_by, dataset):
    """
        ```
        from note_pruning import compute_ranking_dedup
        dataset = 'lima'
        sort_by = 'dedup_md=mpnet_emb=text+embedding'
        Sd = compute_ranking_dedup(sort_by, dataset)
        ```
    """

    kvs = parse_kv_from_string(sort_by)
    model_name = get_full_model_name(kvs['md'])
    dist = 'cd' if kvs['md'] in ['mpnet', 'bge'] else 'l2'
    embed_type = re.sub(r'[+]', '_', kvs['emb'])
    encode_fn_type='input' if kvs['md'] in ['mpnet', 'bge'] else 'sft'
    d = get_lm_output(dataset, model_name, encode_fn_type=encode_fn_type, return_text_embedding=True)
    X = d[embed_type]
    Y = np.zeros(X.shape[0])
    S = note_pruning_clustering.semdedup(X, Y, dist=dist, device='cpu')

    return {sort_by: S}


def compute_ranking_dppmap(sort_by, dataset):
    """
        ```
        sort_by = 'dppmap_nc=200_k=lin_kmd=mpnet'
        sort_by = 'dppmap_nc=200_k=vmf_gamma=3.0_kmd=mpnet'
        sort_by = 'dppmap_nc=200_theta=.3_k=vmf_gamma=3.0_kmd=mpnet_q=ifd_qmd=llama7b+lima'
        ```
    """
    if 'gamma=auto' in sort_by:
        S, info = parse_sort_by_and_compute_dppmap_autotune_gamma(sort_by, dataset)
        if S is None and info is None:
            return None, None
    else:
        S, info = parse_sort_by_and_compute_dppmap(sort_by, dataset)
    return {sort_by: S}, {'info': info}


def compute_ranking_dppmapbd(sort_by, dataset):

    kvs = parse_kv_from_string(sort_by)
    model_name = get_full_model_name(kvs['kmd'])
    if kvs['k'] == 'vmf':
        kernel_kwargs = {'gamma': kvs['gamma']}
    elif kvs['k'] == 'rbf':
        kernel_kwargs = {'gamma': kvs['gamma']}
    elif kvs['k'] == 'lin':
        kernel_kwargs = {'gamma': kvs.get('gamma', 1.)}
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
    encode_fn_type= 'input' if kvs['kmd'] in ['mpnet', 'bge'] else 'sft'
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

    return {sort_by: S}, {'info': output}


def compute_ranking_rho(sort_by, dataset, model_name):
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
    Sd = {}
    for k in ks:
        S0 = ds[0][k]
        S1 = ds[1][k]
        # handle nan entries properly.
        nan_mask = np.logical_or(np.isnan(S0), np.isnan(S1))
        S = np.subtract(S0, S1)
        S[nan_mask] = np.nan
        S = S.squeeze()
        Sd[f'{sort_by}_{k}'] = S

    return Sd



def compute_ranking_numtoks(sort_by, dataset, model_name):
    tokenizer_name_or_path = get_tokenizer_name_or_path(model_name)
    tokenizer = get_fast_tokenizer(tokenizer_name_or_path)
    ds = get_dataset_token_lengths(dataset, tokenizer)

    Sd = {}
    for k in ['input', 'output', 'total']:
        S = ds[f'numtoks_{k}']
        Sd[f'{sort_by}_{k}'] = S

    return Sd


sort_by_that_overwrites_model_name = [
    'kmeans',
    'semdedup',
    'dedup',
    'dppmap',
]


def get_final_model_name(model_name, sort_by):
    """overwrite `model_name` in cases where `sort_by` specifies `md`. """
    if any(sort_by.startswith(x) for x in sort_by_that_overwrites_model_name):
        kvs = parse_kv_from_string(sort_by)
        if 'md' in kvs:
            md = kvs['md']
        elif 'kmd' in kvs:
            md = kvs['kmd']
        else:
            raise ValueError(f"kvs={kvs} should contain 'md' or 'kmd'.")
        model_name = get_full_model_name(md)
    return model_name


def check_md_and_model_name_match(sort_by, model_name):
    """Ensures that the model_name matches the model_name in the sort_by string.
        so that in dppmap will output results to the correct directory, i.e., 
        directory with model that is used to generate embeddings for similarity kernel."""
    if any(sort_by.startswith(x) for x in sort_by_that_overwrites_model_name):
        kvs = parse_kv_from_string(sort_by)
        if 'md' in kvs:
            md = kvs['md']
        elif 'kmd' in kvs:
            md = kvs['kmd']
        else:
            raise ValueError(f"kvs={kvs} should contain 'md' or 'kmd'.")
        if md_to_model_name[md] != model_name:
            raise ValueError(f'md={md} does not match with model_name={model_name}')


    
def main(dataset, sort_by, save_dir, model_name):

    check_md_and_model_name_match(sort_by, model_name)

    pkl_extra = {}

    t0 = time.time()
    if any(sort_by.startswith(x) for x in [
            'log_prob', 
            'el2n',  # el2n_agg={l2n|mean}
            'logit_margin', 
            'grad',  # grad_{loraB|qkv|all|last}_l2n
        ]):
        Sd = compute_ranking_copyover(sort_by, dataset, model_name)
    elif sort_by.startswith('random'):
        Sd = compute_ranking_random(sort_by, dataset, model_name)
    elif sort_by.startswith(('ifd', 'log_pmi')):
        Sd = compute_ranking_ifd_and_pmi(dataset, model_name)
    elif sort_by.startswith('rho'):
        Sd = compute_ranking_rho(sort_by, dataset, model_name)
    elif sort_by.startswith('numtoks'):
        Sd = compute_ranking_numtoks(sort_by, dataset, model_name)
    elif sort_by.startswith('kmeans'):
        Sd, pkl_extra = compute_ranking_kmeans_dist_to_centroids(sort_by, dataset)
    elif sort_by.startswith('semdedup'):
        Sd = compute_ranking_semdedup(sort_by, dataset)
    elif sort_by.startswith('dedup'):
        Sd = compute_ranking_dedup(sort_by, dataset)
    elif sort_by.startswith('dppmap_'):
        Sd, pkl_extra = compute_ranking_dppmap(sort_by, dataset)
    elif sort_by.startswith('dppmapbd'):
        Sd, pkl_extra = compute_ranking_dppmapbd(sort_by, dataset)
    else:
        raise ValueError(f'sort_by={sort_by} not implemented.')
    t1 = time.time()
    print(f'Rank datapoints with {sort_by} took {t1-t0:.2f} seconds.')

    if Sd is not None:
        for k, S in Sd.items():
            save_prune_results(save_dir, S, pkl_extra, k, model_name, dataset)



if __name__ == '__main__':
    """
    python note_explore_data_pruning.py --dataset lima --sort_by log_prob
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lima")
    parser.add_argument("--sort_by", type=str, default="prob")
    parser.add_argument("--model_name", type=str, default="llama-7b")
    parser.add_argument("--save_dir", type=str, default="/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/")
    args = parser.parse_args()

    print(json.dumps(vars(args), indent=2))
    
    os.makedirs(args.save_dir, exist_ok=True)
    main(**vars(args))