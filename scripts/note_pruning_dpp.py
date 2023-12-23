import re
import os
import time
import json
import glob
import pickle
from functools import partial
from dataclasses import dataclass, field
from tqdm import tqdm

from numpy.random import RandomState
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
import pandas as pd
from dppy.finite_dpps import FiniteDPP

import pyarrow # add before importing torch
import torch
from transformers import AutoTokenizer

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/fast-map-dpp")
from dpp import dpp

from note_pruning_analysis import get_lm_output, get_dataset, get_full_model_name, scripts_dir



def get_ifd_and_pmi(dataset, model_name):
    """Assumes ran both `sft` and `output` encode_fn_type when generating model output.
        Here `log_prob` is length-normalized negative log prob: - 1/(|x|) Σₜ log p(x_t|x_<t)
            - encode_fn_type="sft":     -1/|y| Σₜ log p(y_t|y_<t,x) = -1/|y| log p(y|x) = log_PPL(y|x)
            - encode_fn_tyee="output":  -1/|y| Σₜ log p(y_t|y_<t)   = -1/|y| log p(y)   = log_PPL(y)
        IFD = log_PPL(y|x) / log_PPL(y) = log p(y|x) / log p(y)
        PMI = p(y|x) / p(y) = exp( log p(y|x) - log p(y) )
            that is perfectly negatively rank correlated with point-wise mutual information.
    """ 
    if model_name.startswith('llama-7b'):
        model_name_or_path = os.path.join(
            scripts_dir, 'results', 'baselines', 'huggyllama/llama-7b')
    elif model_name.startswith('codellama-7b'):
        model_name_or_path = os.path.join(
            scripts_dir, 'results', 'baselines', 'codellama/CodeLlama-7b-hf')
    elif model_name.startswith('mistral-7b'):
        model_name_or_path = os.path.join(
            scripts_dir, 'results', 'baselines', 'mistralai/Mistral-7B-Instruct-v0.1')
    elif model_name.startswith('llama2-7b'):
        model_name_or_path = os.path.join(
            scripts_dir, 'results', 'baselines', 'NousResearch/Llama-2-7b-hf')
    else:
        raise ValueError(f'Cannot find corresponding `model_name_or_path` for model_name: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    ds = get_dataset(dataset)
    def count_numtoks(example):
        output = [x['content'] for i, x in enumerate(example['messages']) if i%2==1]
        output = '\n'.join(output)
        numtoks_output = len(tokenizer(output, max_length=2048, truncation=True)['input_ids'])
        return {'numtoks_output': numtoks_output}
    ds = ds.map(count_numtoks, num_proc=64)
    numtoks_output = ds['numtoks_output']
    numtoks_output = np.maximum(numtoks_output, 1) # in case numtoks=0

    outputs = {}
    for x in ['sft', 'output']:
        outputs[x] = get_lm_output(dataset, model_name, encode_fn_type=x, return_text_embedding=False)
    log_ppl_yx = outputs['sft']['log_prob'].squeeze()
    log_ppl_y = outputs['output']['log_prob'].squeeze()

    ifd = log_ppl_yx / (log_ppl_y+1e-8)
    log_pyx = (-log_ppl_yx * numtoks_output)
    log_py = (-log_ppl_y * numtoks_output)
    log_pmi = log_pyx - log_py
    return {'log_ppl_yx': log_ppl_yx,
            'log_ppl_y': log_ppl_y,
            'ifd': ifd,
            'log_pmi': log_pmi,}


def plt_ifd_simulation():
    import scipy
    import matplotlib.pyplot as plt

    pyx = .5
    py = np.linspace(0+0.01, 1-0.01, 100)

    v1 = np.log(pyx) / np.log(py)
    v2 = pyx/py
    print(scipy.stats.spearmanr(v1,v2).statistic)

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(py, np.log(pyx) / np.log(py), label='ifd')
    ax.plot(py, pyx / py, label='p(y|x)/p(y)')
    ax.plot(py, pyx / (1-py), label='pyx / (1-py)')
    ax.set_yscale('log')
    ax.set_xlabel('p(y)')
    ax.legend()



def sq_distances(X,Y=None):
    """ Returns K where Kij = ||X_i - Y_j||^2 """
    assert(X.ndim==2)
    # Compute pairwise distance matrix. 
    #     Don't use explicit loops, but the above scipy functions
    # if X=Y, use more efficient pdist call which exploits symmetry
    if Y is None:
        sq_dists = squareform(pdist(X, metric='sqeuclidean'))
    else:
        assert(Y.ndim==2)
        assert(X.shape[1]==Y.shape[1])
        sq_dists = cdist(X, Y, metric='sqeuclidean')

    return sq_dists


def gauss_kernel(X, Y=None, sigma=1.0):
    """ Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2 * sigma**2))
        Returns the kernel matrix
    """
    sq_dists = sq_distances(X,Y)
    S = sq_dists
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K


def vmf_kernel(X, Y, gamma=1.0):
    """Computes exponentiated inner product kernel k(x,y)=exp(γ·x^Ty) """
    S = X@Y.T
    # ensure similarity >=0 (avoid the case where S<0 and yields a larger K)
    # range: [-1,1] -> [-1,0] so that max kernel value is 1 for exp(gamma*0)=1
    S = (S-1)/2
    K = np.exp(gamma*S)
    return K

def linear_kernel(X, Y):
    return np.dot(X, Y.T)


def cholesky_jitter(K, jitter=1e-5):
    K[np.diag_indices_from(K)] += jitter
    return K


def torch_vmf_kernel(X, Y, gamma=1.0):
    """Computes exponentiated inner product kernel k(x,y)=exp(γ·x^Ty)
        Assuems `X` and `Y` have unit norm. """
    S = X@Y.T
    S = (S-1)/2 # ensures S \in [-1,0]
    K = torch.exp(gamma*S)
    S = K
    return K


def torch_rbf_kernel(X, Y=None, sigma=1.0):
    """ Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2·σ2)). """
    sq_dists = torch.cdist(X, Y, p=2)**2
    K = torch.exp(-sq_dists / (2 * sigma**2))
    return K


def get_X(N, d, dist='rand'):
    np.random.seed(0)
    if dist == 'rand':
        X = np.random.rand(N, d)-.5
        return {'X': X}
    elif dist == 'randsp':
        θ = np.random.rand(N)*np.pi*2
        X = np.stack((np.sin(θ), np.cos(θ))).T
        return {'X': X}
    elif dist == 'blobsp':
        from sklearn.datasets import make_blobs
        θ, Y, θC = make_blobs(
            n_samples=N,
            n_features=1, 
            centers=np.linspace(-np.pi, np.pi, 3, endpoint=False).reshape(-1,1),
            center_box=(-np.pi, np.pi),
            cluster_std=[.2, .2, .5],
            random_state=0,
            return_centers=True)
        θ, Y, θC = θ.squeeze(), Y.squeeze(), θC.squeeze()
        X = np.stack((np.sin(θ), np.cos(θ))).T
        C = np.stack((np.sin(θC), np.cos(θC))).T
        return {'X': X, 'Y': Y, 'C': C}
    else:
        raise ValueError(f'dist={dist} not implemented.')


def get_L(X, kernel_type):
    if kernel_type.startswith('cd'):
        L = (X@X.T+1)/2
    if kernel_type.startswith('vmf'):
        gamma = float(re.search(r'gamma=([0-9.]+)', kernel_type).group(1))
        L = vmf_kernel(X, X, gamma=gamma)
    elif kernel_type.startswith('rbf'):
        sigma = float(re.search(r'sigma=([0-9.]+)', kernel_type).group(1))
        L = gauss_kernel(X, sigma=sigma)
    else:
        raise ValueError(f'X ({X.shape}) with kernel_type={kernel_type} cannot compute L.')
    return L



def get_subset(L, subset_type, k):
    rng = RandomState(0)
    if subset_type == 'random':
        np.random.seed(0)
        inds = np.random.choice(len(L), k).tolist()
    elif subset_type == 'dpp_exactk':
        DPP = FiniteDPP(kernel_type='likelihood', projection=False,  L=L)
        inds = DPP.sample_exact_k_dpp(size=k, random_state=rng)
    elif subset_type == 'dpp_sample':
        DPP = FiniteDPP(kernel_type='likelihood', projection=False,  L=L)
        inds = DPP.sample_exact(mode='Chol', random_state=rng)
    elif subset_type == 'dppmap':
        inds = dpp(L, max_length=len(L))
    else:
        raise ValueError(f'subset_type={subset_type} not implemented.')
    return inds


def plt_subsets(Is, data):
    X = data['X']
    naxs = len(Is)
    fig, axs = plt.subplots(2,naxs,figsize=(3*naxs,5), sharey='row',
                            gridspec_kw={'height_ratios': [3, 1]})
    for i in range(naxs):
        subset_type, inds = list(Is.items())[i]
        Xs = X[inds]
        ax = axs[0, i]
        if 'Y' not in data:
            ax.scatter(X[:,0], X[:,1], alpha=.3, cmap='Pastel1')
        else:
            Y = data['Y']
            for y in np.unique(Y):
                ax.scatter(X[Y==y,0], X[Y==y,1], alpha=.3, cmap='Pastel1')
        ax.scatter(Xs[:,0], Xs[:,1], color='r', s=50, label=subset_type)
        ax.set_title(f'{subset_type} (M={len(inds)})', fontsize=10)
        ax.set_aspect('equal')
        ax = axs[1, i]
        θ = np.arctan(Xs[:,1]/Xs[:,0])
        bins = 10
        hist, _, _ = ax.hist(θ, bins=bins, density=True)
        ax.axhline(y=1/np.pi, color='red', linestyle='--', label=r'$\frac{1}{\pi}$')
        # ax.set_ylim((0,1))
        H_θ = -np.sum(hist*np.log(hist+1e-10)*np.pi/bins) # integrate over [-pi/2, pi/2]
        H_uniform = np.log(np.pi)
        ax.set_xlabel(f'H(θ)={H_θ:.3f} (H(Unif(θ))={H_uniform:.3f})')

    fig.tight_layout()
    return fig, axs 


def dppmapbd(X, Y, kernel_type, epsilon=1E-10):
    """dppmap with block-diagonal kernel
        `kernel_matrix_list` is kernel matrix for data points within each cluster
    """
    clusters = sorted(list(np.unique(Y)))
    I = []
    for i in clusters:
        mask = (Y==i)
        inds_global = np.where(mask)[0]
        Xi = X[mask]
        Li = get_L(Xi, kernel_type)
        inds = dpp(Li, max_length=len(Li), epsilon=epsilon)
        I.append(inds_global[inds])
    I = np.hstack(I)
    return I.tolist()


def torch_linear_kernel(X, Y, gamma=1.0):
    """ linear kernel k(x,y)=γ·(x^Ty+1)/2. """
    S = X@Y.T
    K = gamma*(S+1)/2 # ensures K \in [0,1]
    return K


def torch_vmf_kernel(X, Y, gamma=1.0):
    """Computes exponentiated inner product kernel k(x,y)=exp(γ·x^Ty)"""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    S = X@Y.T
    S = (S-1)/2 # ensures S \in [-1,0]
    K = torch.exp(gamma*S) # ensures K \in [0,1]
    return K


def torch_rbf_kernel(X, Y=None, gamma=1.0):
    """ Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2·σ2)). """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    sq_dists = torch.cdist(X, Y, p=2)**2
    K = torch.exp(-gamma*sq_dists)
    return K


def matmul_mem_efficient(a, b, device='cuda', split_dim=1, gpu_mem_budget=.1):
    """assume `b` is more memory intensive
        put `a` into gpu memory by default.
        and put chunks of `b` into memory. """
    import math
    num_chunks = math.ceil(((a.numel() + b.numel())*4 / (1024**3)) / gpu_mem_budget)
    split_size = math.ceil(b.shape[split_dim] / num_chunks)
    c = []
    a = a.to(device)
    b_split = torch.split(b, split_size, dim=split_dim)
    for bi in b_split:
        bi = bi.to(device,  non_blocking=True)
        ci = a@bi
        c.append(ci.to('cpu'))
    c = torch.hstack(c)
    return c


def plt_torch_dppmap_run(state):
    marginal_gains = state.marginal_gains
    di2sL = state.di2sL # need to add this to the run when plotting.
    
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.plot(marginal_gains, label='marginal gains')

    for q in [.5]:
        xs = np.linspace(0,len(di2sL)-1,20).round().astype(np.int32)
        di2s_q = [np.quantile(di2sL[i][np.isfinite(di2sL[i])], q) for i in xs]
        ax.plot(xs, di2s_q, linewidth=2, label=f'di2s quantile={q}')
        
    inds = [2,]
    for ind in inds:
        ax.plot(di2sL[:,ind], '--', label=f'di2s(ind={ind})')

    ax.legend()
    # ax.set_xlim((0, 600))
    # ax.set_title(dataset+'_'+sort_by)
    ax.set_xlabel('Steps')
    ax.set_ylabel('di2 values')
    ax.set_yscale('log')


@dataclass
class DppMapState:
    inds: list[int] = field(default_factory=list)
    cis: torch.Tensor = field(default=None)
    di2s: torch.Tensor = field(default=None)
    marginal_gains: list = field(default_factory=list) # extra, keep track of objective
        

def torch_dppmap_memefficient(Ki_fn,
                              Kd_fn,
                              N,
                              M,
                              epsilon=1e-10,
                              verbose=True,
                              save_dir=None,
                              save_freq=None):
    """dpp map inference 
            - https://arxiv.org/pdf/1709.05135.pdf
    """
    if save_dir is not None:
        state_save_path = os.path.join(save_dir, 'DppMapState.pkl')
    else:
        state_save_path = ''

    if not os.path.isfile(state_save_path):
        state = DppMapState()
        state.cis = torch.zeros((M, N), dtype=torch.float32)
        state.di2s = Kd_fn()
        j = torch.argmax(state.di2s).item()
        state.inds.append(j)
        state.marginal_gains.append(state.di2s[j].item())
        start_iteration = 0
    else:
        with open(state_save_path, 'rb') as f:
            state = pickle.load(f)
        j = state.inds[-1]
        start_iteration = len(state.inds)-1
        print(f'Continuing from start_iteration={start_iteration}')

    pbar = tqdm(range(start_iteration, M-1), initial=start_iteration)
    for _ in pbar:
        k = len(state.inds) - 1
        # (k,)
        ci_optimal = state.cis[:k, j]
        di_optimal = torch.sqrt(state.di2s[j])
        # Kj = K[j, :]
        Kj = Ki_fn(j)
        # (N,) - (k,)@(k,N) / (,) -> (N,)
        eis = (Kj - ci_optimal@state.cis[:k, :]) / di_optimal
        # update to k are updated.
        state.cis[k, :] = eis
        state.di2s -= torch.square(eis)
        state.di2s[j] = -float('inf')
        j = torch.argmax(state.di2s).item()

        if state.di2s[j] < epsilon:
            if verbose:
                print(f'Stop on dᵢ^2 = {state.di2s[j]}. len(inds)={len(state.inds)} / {N}')
            break
        state.inds.append(j)
        state.marginal_gains.append(state.di2s[j].item())

        if save_dir is not None and save_freq is not None and (k+1) % save_freq == 0:
            print(f'Save DppMapState at iteration {k} to {state_save_path}')
            with open(state_save_path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        pbar.set_postfix({'di^2: ': state.marginal_gains[-1]})

                
    if save_dir is not None:
        if os.path.isfile(state_save_path):
            os.remove(state_save_path)
            
    return state.inds, state.marginal_gains


def Ki_fn_full(i, kernel_fn, X, Q):
    Si = kernel_fn(X[i], X)
    Ki = Q[i] * Si * Q
    Ki = Ki.reshape(-1).to('cpu')
    return Ki


def Kd_fn_full(kernel_fn, X, Q):
    Kd = torch.stack([kernel_fn(X[i], X[i]) for i in range(len(X))])
    Kd = Kd.reshape(-1)
    Kd = Kd * Q**2
    Kd = Kd.reshape(-1).to('cpu')
    return Kd


def torch_dppmap(dppmap_type, X, Q, kernel_fn, max_length, Y=None, save_dir=None):
    N = len(X)
    max_length = min(N, max_length)
    if dppmap_type == 'dppmap':
        Ki_fn = partial(Ki_fn_full, kernel_fn=kernel_fn, X=X, Q=Q)
        Kd_fn = partial(Kd_fn_full, kernel_fn=kernel_fn, X=X, Q=Q)
        output = torch_dppmap_memefficient(Ki_fn, Kd_fn, N, max_length, verbose=True, save_dir=save_dir, save_freq=int(max_length//10))
    elif dppmap_type == 'dppmapbd': # block diagonal
        clusters = sorted(list(torch.unique(Y).cpu().numpy()))
        inds_list = []
        marginal_gains_list = []
        for i in clusters:
            print(f'dppmapbd: cluster = {i} / {len(clusters)}')
            mask = (Y == i)
            inds_global = np.nonzero(mask.reshape(-1).to('cpu').numpy())[0]
            Xi = X[mask]
            Qi = Q[mask]
            Ni = Xi.shape[0]
            if Ni <= 10: continue
            Mi = min(Ni, max_length) # max_length, just take all 
            Ki_fn = partial(Ki_fn_full, kernel_fn=kernel_fn, X=Xi, Q=Qi)
            Kd_fn = partial(Kd_fn_full, kernel_fn=kernel_fn, X=Xi, Q=Qi)
            inds, marginal_gains = torch_dppmap_memefficient(
                Ki_fn, Kd_fn, Ni, Mi, verbose=True)
            inds = inds_global[np.array(inds)].tolist()
            inds_list += inds
            marginal_gains_list += marginal_gains
            if len(inds) != len(marginal_gains):
                raise ValueError(f'cluster={i}: len(inds)={len(inds)} != len(marginal_gains)={len(marginal_gains)}')
            num_inf = np.sum(np.isinf(marginal_gains))
            num_nan = np.sum(np.isnan(marginal_gains))
            if num_inf!=0 or num_nan!=0:
                raise ValueError(f'cluster={i}: marginal_gains has num_inf={num_inf}, num_nan={num_nan}')
            # if i == 2: break
        ## Use `marginal_gains` to rank points cross clusters 
        # so that points with higher marginal gains appears at the front of the list
        inds, marginal_gains = inds_list, marginal_gains_list
        if len(inds)!=len(marginal_gains):
            raise ValueError(f'len(inds)={len(inds)} != len(marginal_gains)={len(marginal_gains)}\n\n inds={inds}\n\n marginal_gains={marginal_gains}')
        inds, marginal_gains = zip(*[(i, v) for i, v in sorted(
            zip(inds, marginal_gains), key=lambda x: x[1], reverse=True)])
        inds, marginal_gains = list(inds), list(marginal_gains)
        if not all([x==y for x, y in zip(marginal_gains, sorted(marginal_gains, reverse=True))]):
            raise ValueError(f'marginal_gains not sorted\n\n marginal_gains={marginal_gains}')
        output = inds, marginal_gains
    else:
        raise ValueError(f'dppmap_type={dppmap_type} not implemented.')
    return output


def normalize_to_unit_interval(Q, quantile_to_0=0.01, quantile_to_1=0.99):
    q_lower, q_mid, q_upper = np.quantile(Q, quantile_to_0), \
                              np.quantile(Q, .5), \
                              np.quantile(Q, quantile_to_1)
    Q = (Q-q_lower) / (q_upper-q_lower)
    print(f'linearly transform scores s.t. prev value='
          f'[{q_lower:.2f}, {q_mid:.2f}, {q_upper:.2f}] -> [0, {np.mean(Q):.2f}, 1]\n')
    return Q


def compute_dppmap(
    dppmap_type,
    dataset,
    kernel_type,
    kernel_embed_model,
    kernel_embed_type,
    kernel_kwargs,
    quality_score_type,
    quality_score_embed_model,
    theta,
    max_length,
    device,
    run_name,
    Y=None, # cluster assignments
):  
    """
        Memory usage:
        ```
        gpu_mem_fn = lambda N, D: N*(D+1)*4/(1024**3)
        cpu_mem_fn = lambda N, M, D: N*(M+1)*4/(1024**3)
        for N, M, D in [
            (100_000,  10_000, 4096),
            (1500_000, 10_000, 4096),
            (1500_000, 20_000, 4096),
        ]:
            print(f'[D={D}, {N:7} ->{M:6}]\t'
                f'gpu_mem = {gpu_mem_fn(N, D):5.2f}GB\tcpu_mem = {cpu_mem_fn(N, M, D):5.2f}GB')
        ```
    """
    
    if dppmap_type not in ['dppmap', 'dppmapbd']:
        raise ValueError(f'dppmap_type={dppmap_type} not supported.')

    if kernel_type == 'vmf':
        kernel_fn = partial(torch_vmf_kernel, **kernel_kwargs)
    elif kernel_type == 'rbf':
        kernel_fn = partial(torch_rbf_kernel, **kernel_kwargs)
    elif kernel_type == 'lin':
        kernel_fn = partial(torch_linear_kernel, **kernel_kwargs)
    else:
        raise ValueError(f'kernel_type={kernel_type} not supported.')
    
    valid_embed_model = ['mpnet', 'bge', 'llama7b', 'llama2:7b', 'mistral7b', 'llama7b+lima']
        
    if kernel_embed_model not in valid_embed_model:
        raise ValueError(f'kernel_embed_model={kernel_embed_model} not supported.')
    if theta != 0. and \
        quality_score_embed_model not in valid_embed_model + ['llama7b+lima']:
        # if theta!=0, then we're using quality score and so need to specify model
        raise ValueError(f'quality_score_embed_model={quality_score_embed_model} not supported.')

    ## parse `text_embedding_rp256` to `text_embedding` and `rand_proj` dimensions
    match = re.search(r'rp(\d+)', kernel_embed_type)
    rand_proj = int(match.group(1)) if match else None
    kernel_embed_type = re.sub(r'_rp(\d+)', '', kernel_embed_type)
    if kernel_embed_type not in ['text_embedding', 'grad_rp_loraB']:
        raise ValueError(f'kernel_embed_type={kernel_embed_type} not supported.')
        

    dk = get_lm_output(
        dataset, 
        get_full_model_name(kernel_embed_model),
        encode_fn_type='input' if kernel_embed_model in ['mpnet', 'bge'] else 'sft', 
        return_text_embedding=True)
    X = dk[kernel_embed_type]
    if any(x in kernel_embed_model for x in ['mpnet', 'bge']) or kernel_type in ['vmf', 'lin']:
        X = X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-8) # possibly divide by zero.
    if rand_proj is not None:
        from sklearn.random_projection import GaussianRandomProjection
        jl_proj = GaussianRandomProjection(n_components=rand_proj, random_state=0)
        print(f'Start random projection from {X.shape[1]} -> {rand_proj} ...')
        X = jl_proj.fit_transform(X)
        print('Done random projection.')

    X = torch.from_numpy(X).to(device)

    if quality_score_type is None:
        Q = torch.ones([1], device=device).expand(len(X))
    elif quality_score_type.startswith('ifd'):
        Q = get_ifd_and_pmi(dataset, get_full_model_name(quality_score_embed_model))['ifd']
        if 'neg' in quality_score_type:
            Q = -Q
        Q = normalize_to_unit_interval(Q)
        Q = torch.from_numpy(Q).to(device)
    elif quality_score_type.startswith('log_pmi'): # higher is better!
        Q = get_ifd_and_pmi(dataset, get_full_model_name(quality_score_embed_model))['log_pmi']
        if 'neg' in quality_score_type:
            Q = -Q
        Q = normalize_to_unit_interval(Q)
        Q = torch.from_numpy(Q).to(device)
    else:
        dq = get_lm_output(
            dataset, 
            get_full_model_name(quality_score_embed_model),
            encode_fn_type='input' if quality_score_embed_model in ['mpnet', 'bge'] else 'sft', 
            return_text_embedding=True)
        if quality_score_type == 'prob':
            Q = dq['log_prob']
            Q = np.exp(Q)
        else:
            if quality_score_type not in dq:
                raise ValueError(f'quality_score_type={quality_score_type} not pre-computed {dq.keys()}')
            Q = dq[quality_score_type]
        Q = torch.from_numpy(Q).to(device)
    Q = Q.reshape(-1)
    alpha = theta / (2*(max(1-theta, 1e-5)))
    Q = torch.exp(alpha*Q)

    if Y is not None:
        Y = torch.from_numpy(Y.reshape(-1)).to(device)

    N = X.shape[0]
    if not X.shape[0] == Q.numel():
        raise ValueError(f'X ({X.shape}) and Q ({Q.shape}) does not match')

    info = {
        'dppmap_type': dppmap_type,
        'dataset': dataset,
        'kernel_type': kernel_type,
        'kernel_embed_model': kernel_embed_model,
        'kernel_embed_type': kernel_embed_type,
        'kernel_kwargs': kernel_kwargs,
        'quality_score_type': quality_score_type,
        'quality_score_embed_model': quality_score_embed_model,
        'theta': theta,
        'max_length': max_length,
        'rand_proj': rand_proj,
    }
    save_dir = os.path.join(scripts_dir, 'dpp', dataset, run_name)
    os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()
    inds, marginal_gains = torch_dppmap(dppmap_type, X, Q, kernel_fn, max_length, Y=Y, save_dir=save_dir)
    info['M'] = len(inds)
    info["time_elapsed"] = time.time()-t0

    data = {}
    data['marginal_gains'] = marginal_gains
    data['inds'] = inds

    with open(os.path.join(save_dir, 'info.json'), 'w') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
        
    with open(os.path.join(save_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    plt_dppmap_results(N, Y, inds, marginal_gains, save_dir, run_name, max_length)
    plt_subset_size_vs_kernel_params(dataset) # update the subset size vs. kernel hyperparameter trend plot.


    ## convert `inds` to scores `S`
    # assign random data points to the rest of the slots, after `max_length`
    inds_left = set(np.arange(N).tolist()) - set(inds)
    np.random.seed(0)
    inds_left = np.random.permutation(list(inds_left)).tolist()
    inds += inds_left
    if not len(inds) == N:
        raise ValueError(f'len(inds)={len(inds)} != N={N}')
    # points ranked higher in `inds` will have a smaller score
    S = np.zeros(N, dtype=np.int64)
    for i, ind in enumerate(inds):
        S[ind] = i

    ## release gpu memory manually to avoid repeated call to this function giving oom
    if device == 'cuda':
        del X
        del Q
        if Y is not None: del Y
        torch.cuda.empty_cache()

    return S, info



def plt_dppmap_results(N, Y, inds, marginal_gains, save_dir, run_name, max_length):

    inds = inds.copy()
    marginal_gains = marginal_gains.copy()

    M = len(inds)

    ## plot how marginal gain decays vs. iterations
    spacings = 1000
    fig, axs = plt.subplots(2,1,figsize=(12,6), sharex=True)
    ys = marginal_gains
    xs = list(range(0, len(marginal_gains), spacings))
    ys = marginal_gains[::spacings]
    for i in range(2):
        ax = axs[i]
        ax.plot(xs, ys, label=run_name)
        ax.set_xlim((0, N))
        if i == 1:
            ax.set_yscale('log')
        ax.set_ylabel('Marginal Gain (dᵢ^2)')
        ax.set_xlabel('Iterations')
    fig.suptitle(run_name)
    fig.tight_layout()
    save_path = os.path.join(save_dir, f'fig_dppmap_marginal_gain_vs_iterations.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

    ## subsequent plots requires `Y` to be provided
    if Y is None: 
        return

    ## Plot dpp map subset size w.r.t. clusters as a function of M, the kept set size.
    # 
    if isinstance(Y, torch.Tensor):
        Y = Y.to('cpu').numpy()
        
    def first_n_cluster_sizes(n):
        """Returns number of points belonging to each cluster 
            for first `n` data points' in dppmap results """
        Y_first_n = Y[inds[:n]]
        cluster_sizes = {k: v for k, v in zip(*np.unique(Y_first_n, return_counts=True))}
        return cluster_sizes

    cluster_sizes = {k: v for k, v in zip(*np.unique(Y, return_counts=True))}
    xs = list(cluster_sizes.keys())
    cluster_sizes = np.array(list(cluster_sizes.values()))

    pct = [0.03, 0.05, 0.1, 0.2, 0.3]
    first_n_list = [int(x*N) for x in pct] + [M]
    nrows = len(first_n_list)
    fig, axs = plt.subplots(nrows, 1, figsize=(12,4*nrows), sharex=True)

    for i, first_n in enumerate(first_n_list):
        ax = np.array(axs)[i]
        ax.bar(xs, cluster_sizes, label='Cluster Size')
        if max_length < cluster_sizes.max():
            ax.axhline(y = max_length, color='r', linestyle='--', label='max_length')
        first_n_sizes = first_n_cluster_sizes(first_n)
        ys  = np.array([first_n_sizes[i] if i in first_n_sizes else 0 for i in xs])
        ax.bar(xs, ys, label=f'dppmap subset')
        ax.legend()
        ax.set_ylabel(f'M={first_n}, pct={round(first_n/N, 3)}', fontsize=20)

    fig.suptitle(f"dppmap subset from each cluster N={N}, M={M}, max_length={max_length}", y=1)
    fig.tight_layout()
    save_path = os.path.join(save_dir, f'fig_dppmap_subset_wrt_clusters.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

    ## plot marginal gain decays for each cluster.
    #  can get a better idea if different clusters marginal gain decay at different rate.
    def get_first_n_marginal_gains_wrt_clusters(n):
        clusters = np.unique(Y)
        Y_first_n = Y[inds[:n]]
        marginal_gains_first_n = np.array(marginal_gains)[:n]
        marginal_gains_wrt_clusters = {i: marginal_gains_first_n[Y_first_n==i] for i in clusters}
        return marginal_gains_wrt_clusters
    marginal_gains_wrt_clusters = get_first_n_marginal_gains_wrt_clusters(M)
    fig, axs = plt.subplots(2,1,figsize=(12,6), sharex=True)
    for axi in range(2):
        ax = axs[axi]
        for i, ys in marginal_gains_wrt_clusters.items():
            ax.plot(ys, label=f'Cluster {i}')
        ax.set_ylabel('Marginal Gain (dᵢ^2)')
        if axi == 1:
            ax.set_yscale('log')
    axs[-1].set_xlabel('dppmap subset size (w.r.t clusters)')
    fig.suptitle(f'Marginal gains decay in each cluster ({run_name})')
    fig.tight_layout()
    save_path = os.path.join(save_dir, f'fig_dppmap_marginal_gain_clusterwise.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()



def plt_subset_size_vs_kernel_params(dataset, save_fig=True, filter_fn=None, sort_by='dppmap_*', hlines=[1_000, 2_000, 10_000]):
    """Plot how subse size varies with kernel hyperparameters.
        ```
        from note_pruning_dpp import plt_subset_size_vs_kernel_params
        dataset = 'sharegptv2'
        filter_fn = lambda row: 'llama2' not in row['sort_by'] and row['M'] < 20_000
        dataset = 'wizardlm'
        filter_fn = lambda row: 'theta' not in row['sort_by'] and row['M'] < 20_000
        fig, axs = plt_subset_size_vs_kernel_params(dataset, save_fig=False, filter_fn=filter_fn)

        ## cross-dataset comparison
        datasets = [
            'stanford_alpaca', 
            'sharegptv2',
            'wizardlm',
            'tulu_v2',
            'open_orca_slim',
        ]
        M = 1200; hlines = [1_000]
        M = 50_000; hlines = [1_000, 2_000, 10_000]
        sort_by='dppmap_k=rbf*kemb=text+embedding'
        sort_by='dppmap_*mpnet*'
        sort_by = 'dppmap_k=vmf*kemb=grad+rp+loraB'
        fig, axs = plt_subset_size_vs_kernel_params(
            datasets, 
            save_fig=False, 
            filter_fn=lambda row: row['M'] <M and 'theta' not in row['sort_by'] \
                and 'llama2' not in row['sort_by'],
            hlines=hlines,
            sort_by=sort_by,
        )
        ```
    """
    if isinstance(dataset, list):
        df_list = []
        for x in dataset:
            df = get_dppmap_run_info(sort_by, x)
            df['sort_by'] = df.apply(lambda row: f"{x}:{row['sort_by']}", axis=1)
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
        dataset = ';'.join([x[:3] for x in dataset])
    else:
        df = get_dppmap_run_info(sort_by, dataset)
    df = df[df.max_length > df.M]
    if filter_fn is not None:
        df = df[df.apply(filter_fn, axis=1)]
    df = df.reset_index(drop=True)

    info = df.groupby('sort_by').apply(lambda g: g[['gamma', 'M', 'time_elapsed']].to_dict(orient='list')).to_dict()

    nrows = len(info) + 2

    fig, axs = plt.subplots(nrows, 1, figsize=(12,4*nrows))

    ax = axs[0]
    for y in hlines:
        ax.axhline(y=y, color='gray', linestyle='--')
    for i, (sort_by, d) in enumerate(info.items()):
        ax.plot(d['gamma'], d['M'], 'o-', label=sort_by)
    ax.set_ylabel('subset size or rank(L) (larger -> less redundant)')
    ax.set_xlabel(r'$\sqrt{\gamma}$ (larger -> repulsive effect more local)')
    ax.set_xscale('function', functions=(lambda x: x**0.5, lambda x: x**2))
    ax.set_xlim(left=np.finfo(float).eps)
    ax.legend(fontsize=8)

    ax = axs[1]
    for i, (sort_by, d) in enumerate(info.items()):
        ax.plot(d['gamma'], d['M'], 'o-', label=sort_by)
    ax.set_ylabel('subset size')
    ax.set_xlabel(r'$\gamma$')
    # ax.set_yscale('log')
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)

    for i, (sort_by, d) in enumerate(info.items()):
        ax = axs[i+2]
        xs, ys = d['gamma'], d['M']
        ax.plot(xs, ys, 'x', markersize=10, label=sort_by)
        coeff = np.polyfit(xs, ys, 1)
        poly1d_fn = np.poly1d(coeff)
        xs_extrap = np.linspace(0, 1.05*np.max(xs), 100)
        ax.plot(xs_extrap, poly1d_fn(xs_extrap), '--k', label=f'Fitted linear fn (m={coeff[0]:.1f},b={coeff[1]:.1f})')
        ys_target = np.array([1_000, 5_000, 10_000, 50_000])
        xs_target = (ys_target - coeff[1]) / coeff[0]
        for x, y in zip(xs_target, ys_target):
            ax.scatter(x, y, color='k', label=f'({x:.8f}, {y})')
        for x, y in zip(xs_target, ys_target):
            ymin, ymax = ax.get_ylim(); yrel = (y-ymin)/(ymax-ymin)
            xmin, xmax = ax.get_xlim(); xrel = (x-xmin)/(xmax-xmin)
            ax.axvline(x=x, ymax=yrel, color='b', linestyle='--')
            ax.axhline(y=y, xmax=xrel, color='b', linestyle='--')
        ax.set_ylabel('subset size')
        ax.set_xlabel('gamma')
        ax.legend()

    fig.suptitle(dataset)
    fig.tight_layout()


    if save_fig:
        save_path = os.path.join(scripts_dir, 'dpp', dataset, f'fig_dppmap_subset_size_vs_kernel_params.png')
        fig.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()

    return fig, axs
    


def get_dppmap_run_info(filename, dataset):
    """get previous run info that is related to `filename` or `sort_by`. 
            only for `sort_by` with tunable `gamma`. 

        info: gamma, max_length, M, time_elapsed
    """
    filepaths = glob.glob(os.path.join(scripts_dir, 'dpp', dataset, filename))
    filenames = [os.path.basename(x) for x in filepaths]
    if not filenames:
        return pd.DataFrame(columns=[
            'filename',
            'sort_by',
            'gamma',
            'max_length',
            'M',
            'time_elapsed',
        ])

    df = pd.DataFrame({'filename': filenames})
    df = df[df['filename'].str.contains('gamma')]

    def get_sort_by_fn(r):
        match = re.search(r'gamma=([\d.e+-]+)', r['filename'])
        if match:
            return r['filename'].replace(match.group(1), '([\d.e+-]+)')
        else:
            return None
    df['sort_by'] = df.apply(get_sort_by_fn, axis=1)

    def get_info_fn(r):
        info_path = os.path.join(scripts_dir, 'dpp', dataset, r['filename'], 'info.json')
        if not os.path.isfile(info_path):
            cols = {
                'gamma': None, 
                'max_length': None, 
                'M': None, 
                'time_elapsed': None,
            }
        else:
            with open(info_path, 'r') as f:
                d = json.load(f)
                cols = {
                    'gamma': d['kernel_kwargs']['gamma'],
                    'max_length': int(d['max_length']),
                    'M': int(d['M']),
                    'time_elapsed': d['time_elapsed'],
                }
        return pd.Series(cols)
    df = pd.concat([df, df.apply(get_info_fn, axis=1)], axis=1)
    df = df[~df['gamma'].isna()]
    df['max_length'] = df['max_length'].astype(np.int32)
    df['M'] = df['M'].astype(np.int32)
    df = df.sort_values(['sort_by', 'gamma'])
    df = df.reset_index(drop=True)

    return df
