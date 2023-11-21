import re
import numpy as np
from numpy.random import RandomState
from scipy.spatial.distance import squareform, pdist, cdist
from dppy.finite_dpps import FiniteDPP

import pyarrow # add before importing torch
import torch

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/fast-map-dpp")
from dpp import dpp


def torch_vmf_kernel(X, Y, gamma=1.0):
    """Computes exponentiated inner product kernel k(x,y)=exp(γ·x^Ty)
        Assuems `X` and `Y` have unit norm. """
    S = X@Y.T
    S = (S-1)/2 # ensures S \in [-1,0]
    K = torch.exp(gamma*S)
    return K


def torch_rbf_kernel(X, Y=None, sigma=1.0):
    """ Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2·σ2)). """
    sq_dists = torch.cdist(X, Y, p=2)**2
    K = torch.exp(-sq_dists / (2 * sigma**2))
    return K



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



def torch_vmf_kernel(X, Y, gamma=1.0):
    """Computes exponentiated inner product kernel k(x,y)=exp(γ·x^Ty)"""
    S = X@Y.T
    S = (S-1)/2 # ensures S \in [-1,0]
    K = torch.exp(gamma*S)
    return K


def torch_rbf_kernel(X, Y=None, sigma=1.0):
    """ Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2·σ2)). """
    sq_dists = torch.cdist(X, Y, p=2)**2
    K = torch.exp(-sq_dists / (2 * sigma**2))
    return K



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
