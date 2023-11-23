import re
import time
import numpy as np
from functools import partial

from numpy.random import RandomState
from scipy.spatial.distance import squareform, pdist, cdist
from dppy.finite_dpps import FiniteDPP

import pyarrow # add before importing torch
import torch

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/fast-map-dpp")
from dpp import dpp



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


def torch_linear_kernel(X, Y=None):
    S = X@Y.T
    K = (S+1)/2 # ensures K \in [0,1]
    return K


def torch_vmf_kernel(X, Y, gamma=1.0):
    """Computes exponentiated inner product kernel k(x,y)=exp(γ·x^Ty)"""
    S = X@Y.T
    S = (S-1)/2 # ensures S \in [-1,0]
    K = torch.exp(gamma*S) # ensures K \in [0,1]
    return K


def torch_rbf_kernel(X, Y=None, sigma=1.0):
    """ Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2·σ2)). """
    sq_dists = torch.cdist(X, Y, p=2)**2
    K = torch.exp(-sq_dists / (2 * sigma**2))
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



def torch_dppmap_memefficient(Ki_fn, Kd_fn, N, M, epsilon=1e-10, verbose=True):
    """dpp map inference 
            - https://arxiv.org/pdf/1709.05135.pdf
    """
    from tqdm import tqdm
    # (M, N)
    cis = torch.zeros((M, N), dtype=torch.float32)
    # marginal gain of logdet by including j-th item at each iteration
    marginal_gains = []
    # (N,)
    # di2s = torch.diag(K)
    di2s = Kd_fn()
    # grows to at most length M
    inds = []
    j = torch.argmax(di2s).item()
    inds.append(j)
    marginal_gains.append(di2s[j].item())
    for _ in tqdm(range(M-1)):
        k = len(inds) - 1
        # (k,)
        ci_optimal = cis[:k, j]
        di_optimal = torch.sqrt(di2s[j])
        # Kj = K[j, :]
        Kj = Ki_fn(j)
        # (N,) - (k,)@(k,N) / (,) -> (N,)
        eis = (Kj - ci_optimal@cis[:k, :]) / di_optimal
        # update to k are updated.
        cis[k, :] = eis
        di2s -= torch.square(eis)
        di2s[j] = -float('inf')
        j = torch.argmax(di2s).item()

        if di2s[j] < epsilon:
            if verbose:
                print(f'Stop on dᵢ^2 = {di2s[j]}. len(inds)={len(inds)} / {N}')
            break
        inds.append(j)
        marginal_gains.append(di2s[j].item())

    return inds, marginal_gains


def Ki_fn_full(i, kernel_fn, X, Q):
    Si = kernel_fn(X[i], X)
    Ki = Q[i] * Si * Q
    Ki = Ki.reshape(-1).to('cpu')
    return Ki


def Kd_fn_full(kernel_fn, X, Q):
    Kd = torch.stack([kernel_fn(X[i], X[i]) for i in range(len(X))])
    Kd = Kd * Q**2
    Kd = Kd.reshape(-1).to('cpu')
    return Kd


def torch_dppmap(dppmap_type, X, Q, kernel_fn, max_length, Y=None):
    N = len(X)
    if dppmap_type == 'dppmap':
        Ki_fn = partial(Ki_fn_full, kernel_fn=kernel_fn, X=X, Q=Q)
        Kd_fn = partial(Kd_fn_full, kernel_fn=kernel_fn, X=X, Q=Q)
        output = torch_dppmap_memefficient(Ki_fn, Kd_fn, N, max_length, verbose=True)
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
            inds_list += inds_global[np.array(inds)].tolist()
            marginal_gains_list += marginal_gains
        ## Use `marginal_gains` to rank points cross clusters 
        # so that points with higher marginal gains appears at the front of the list
        inds, marginal_gains = inds_list, marginal_gains_list
        if len(inds)!=len(marginal_gains):
            raise ValueError(f'len(inds)={len(inds)} != len(marginal_gains)={len(marginal_gains)}')
        inds, marginal_gains = zip(*[(i, v) for i, v in sorted(
            zip(inds, marginal_gains), key=lambda x: x[1], reverse=True)])
        inds, marginal_gains = list(inds), list(marginal_gains)
        assert(all([x==y for x, y in zip(marginal_gains, sorted(marginal_gains, reverse=True))]))
        output = inds, marginal_gains
    else:
        raise ValueError(f'dppmap_type={dppmap_type} not implemented.')
    return output


def get_full_model_name(md):
    if md == 'mpnet':
        model_name = 'all-mpnet-base-v2'
    elif md == 'bge':
        model_name = 'bge-large-en-v1.5'
    elif md == 'llama7b':
        model_name = 'llama-7b+lora:r=256:a=256'
    elif md == 'mistral7b':
        model_name = 'mistral-7b+lora:r=256:a=256'
    else:
        raise ValueError(f'Dont know full name for model_name={md}')
    return model_name



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
    Y=None, # cluster assignments
):
    
    from note_pruning_analysis import get_lm_output

    if kernel_type == 'vmf':
        kernel_fn = partial(torch_vmf_kernel, **kernel_kwargs)
    elif kernel_type == 'rbf':
        kernel_fn = partial(torch_rbf_kernel, **kernel_kwargs)
    elif kernel_type == 'lin':
        kernel_fn = torch_linear_kernel
    else:
        raise ValueError(f'kernel_type={kernel_type} not supported.')
        
    if kernel_embed_model not in ['mpnet', 'bge', 'llama7b', 'mistral7b']:
        raise ValueError(f'kernel_embed_model={kernel_embed_model} not supported.')
    if theta != 0. and \
        quality_score_embed_model not in ['mpnet', 'bge', 'llama7b', 'mistral7b']:
        # if theta!=0, then we're using quality score and so need to specify model
        raise ValueError(f'quality_score_embed_model={quality_score_embed_model} not supported.')
    if kernel_embed_type not in ['text_embedding', 'grad_rp_loraB']:
        raise ValueError(f'kernel_embed_type={kernel_embed_type} not supported.')
        

    dk = get_lm_output(
        dataset, 
        get_full_model_name(kernel_embed_model),
        encode_fn_type='input' if kernel_embed_model in ['mpnet', 'bge'] else 'sft', 
        return_text_embedding=True)
    X = dk[kernel_embed_type]
    if any(x in kernel_embed_model for x in ['mpnet', 'bge']):
        X = X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-8) # possibly divide by zero.
    X = torch.from_numpy(X).to(device)

    if quality_score_type is None:
        Q = torch.ones([1], device=device).expand(len(X))
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

    output = {
        'dataset': dataset,
        'kernel_type': kernel_type,
        'kernel_embed_model': kernel_embed_model,
        'kernel_embed_type': kernel_embed_type,
        'kernel_kwargs': kernel_kwargs,
        'quality_score_type': quality_score_type,
        'quality_score_embed_model': quality_score_embed_model,
        'theta': theta,
        'max_length': max_length
    }
    t0 = time.time()
    inds, marginal_gains = torch_dppmap(dppmap_type, X, Q, kernel_fn, max_length, Y=Y)
    output['M'] = len(inds)
    output["time_elapsed"] = time.time()-t0
    output['marginal_gains'] = marginal_gains

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

    return S, output