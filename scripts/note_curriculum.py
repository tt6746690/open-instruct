import os
import re
import random
import pickle
import glob
import itertools

import numpy as np


from note_pruning_analysis import curriculum_dir, save_to_pickle, get_sorted_inds



def convert_existing_data_inds_to_curriculum_scores():

    paths = glob.glob('data_inds/*/*/*.pkl')
    paths = [x for x in paths if 'incr' in x and 'pythia' not in x]

    for path in paths:

        pkl_filename = os.path.basename(path)
        sort_by = os.path.splitext(pkl_filename)[0]
        dataset = os.path.basename(os.path.dirname(path))
        model_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        sorted_inds = get_sorted_inds(dataset, model_name, sort_by)
        sorted_inds = {k: np.array(v) if v else None for k, v in sorted_inds.items()}
        if 'S' in sorted_inds and 'inds' in sorted_inds:
            sorted_inds['S'] = sorted_inds['S'][np.argsort(sorted_inds['inds'])]
        S = sorted_inds['S']

        ## use `note_pruning` to generate scores for curriculum learning.
        sort_by = sort_by.split('_incr')[0]
        for pacing_fn in [sort_by, sort_by+'_neg']:
            curriculum_output_dir = os.path.join('curriculum', model_name, dataset, pacing_fn)
            print(curriculum_output_dir)
            os.makedirs(curriculum_output_dir, exist_ok=True)
            save_path = os.path.join(curriculum_output_dir, 'scores.pkl')
            output = {'S': -S if pacing_fn.endswith('_neg') else S}
            save_to_pickle(save_path=save_path, output=output)

    
def np_random_choice_maximize_noreplacement(L, n):
    """Sample `L` without replacement `n` samples.
        if `n>len(L)`, then sample without replacement as much as possible, and
            accumulate the resulting samples until the entire `n` is sampled
    """
    S = []
    while n>0:
        l = np.random.choice(L, size=min(n, len(L)), replace=False)
        n -= l.size
        S.append(l)
    S = np.hstack(S)
    return S


def scores_path_to_attrs(path):
    """Convert `scores.pkl` path to factors that generated the scores."""
    pkl_filename = os.path.basename(path)
    scoring_fn = os.path.basename(os.path.dirname(path))
    dataset = os.path.basename(os.path.dirname(os.path.dirname(path)))
    model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
    output = {
        'model_name': model_name, 
        'dataset': dataset, 
        'scoring_fn': scoring_fn,
    }
    return output


def get_curriculum_scores(path):
    """The scores are pre-computed via `note_pruning.py`. """
    output = scores_path_to_attrs(path)
    with open(path, 'rb') as f:
        scores = pickle.load(f)['S']
    output.update({'scores': scores})
    return output


def get_curriculum(model_name, dataset, scoring_fn, pacing_fn):
    """Get a curriculum that contains both `scores` and `inds`
        once the curriculum is generated.

        ```
        from note_curriculum import get_curriculum
        get_curriculum('llama-7b+lora:r=256:a=256', 
                       'tulu_v1_mix', 
                       'log_prob_neg', 
                       'prune_size=150000_ep=3')
        ```
    
    """
    path = os.path.join(curriculum_dir, model_name, dataset, scoring_fn, 'inds_'+pacing_fn+'.pkl')
    with open(path, 'rb') as f:
        output = pickle.load(f)
    for k, v in [('dataset', dataset), ('model_name', model_name), ('scoring_fn', scoring_fn)]:
        assert(output[k] == v)
    for k in ['scores', 'inds']:
        output[k] = output.pop(k)
    return output




def plt_curriculum(scores, inds_to_inds, inds, N, pacing_fn, fig=None, axs=None):
    import matplotlib.pyplot as plt
    if fig is None or axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(15,3), sharey=False, gridspec_kw={'width_ratios': [2,.5,.5]})

    ys = inds_to_inds/N
    ys = ys[::50]

    ax = axs[0]
    ax.scatter(np.arange(len(ys)), ys, marker='x', alpha=.1)
    ax.set_ylabel('Data Index Ranking', fontsize=15)
    ax.set_xlabel('Training Iterations', fontsize=15)
    ax.set_ylim((0,1))
    ax.set_title(f'{pacing_fn}', fontsize=20)


    ax = axs[1]
    ax.hist(ys, bins=20)
    ax.set_xlabel('Data Index Ranking', fontsize=15)

    ax = axs[2]
    ys = scores[inds][::50]
    ax.hist(ys, bins=20)
    ax.set_xlabel('Scores', fontsize=15)
    ax.set_xlim((0,6))

    ylim = np.vstack([np.array(axs[i].get_ylim()) for i in [1,2]])
    ylim = list(np.max(ylim, axis=0))
    for i in [1,2]:
        axs[i].set_ylim(ylim)
        
    return fig, axs


def generate_curriculum(path, pacing_fn, verbose=False, save_output=True):
    """Generate a data ordering for curriculum learning given 
        `path` to scores of each data point and a pacing function.
    """

    output = get_curriculum_scores(path)
    output['pacing_fn'] = pacing_fn

    scores = output['scores']
    N = scores.size
    inds_sorted = np.argsort(scores).tolist() # increasing `scores`.
    match = re.search(r'_s=([^_]+)', pacing_fn)
    seed = int(match.group(1)) if match else 0

    np.random.seed(seed)

    if pacing_fn.startswith('prune'):

        match = re.search(r'size=([^_]+)', pacing_fn)
        M = int(match.group(1)) # total data points over `num_epochs` epochs.
        match = re.search(r'ep=([^_]+)', pacing_fn)
        num_epochs = int(match.group(1))
            
        inds_to_inds_sorted = np.arange(len(inds_sorted))

        sizes = []
        for _ in range(num_epochs-1):
            sizes.append(int(M//num_epochs))
        sizes.append(int(M-np.sum(sizes)))

        if max(sizes) >= len(inds_sorted):
            raise ValueError(f'len(number of unique data point)={max(sizes)} > len(inds)={len(inds_sorted)}')
        
        inds = []
        for size in sizes:
            inds_kept = list(inds_to_inds_sorted[:size])
            np.random.shuffle(inds_kept)
            inds.append(inds_kept)
        inds = [x for sl in inds for x in sl]
        inds_to_inds = np.array(inds)
        inds = np.array(inds_sorted)[inds_to_inds]

        if len(inds) != M:
            raise ValueError('len(inds) should be equal to size')
    elif pacing_fn.startswith(('singlestep', 'fep')):
        
        if pacing_fn.startswith('singlestep'):
            match = re.search(r'size=([^_]+)', pacing_fn)
            M = int(match.group(1))
            match = re.search(r'startingfrac=([\d.]+)', pacing_fn)
            startingfrac = float(match.group(1))
            nsteps = 2
            inc = 1/startingfrac
        else:
            match = re.search(r'size=([^_]+)', pacing_fn)
            M = int(match.group(1))
            match = re.search(r'startingfrac=([\d.]+)', pacing_fn)
            startingfrac = float(match.group(1))
            match = re.search(r'nsteps=([^_]+)', pacing_fn)
            nsteps = int(match.group(1))
            match = re.search(r'inc=([\d.]+)', pacing_fn)
            inc = float(match.group(1))
            
        inds_to_inds_sorted = np.arange(len(inds_sorted))

        step_lens = []
        for _ in range(nsteps-1):
            step_lens.append(int(M//nsteps))
        step_lens.append(int(M-np.sum(step_lens)))

        stepwise_data_fracs = []
        for i in range(len(step_lens)):
            stepwise_data_fracs.append(min(startingfrac*(inc)**(i), 1))

        if verbose:
            print(f'fixed exponential pacing startingfrac={startingfrac}, inc={inc}, nsteps={nsteps} \n'
                f'Implies: step_lens={step_lens}, stepwise_data_fracs={stepwise_data_fracs}')
        inds = []
        for step_len, stepwise_data_frac in zip(step_lens, stepwise_data_fracs):
            inds_sample_from = list(inds_to_inds_sorted[:int(stepwise_data_frac*N)])
            inds_perstep = np_random_choice_maximize_noreplacement(inds_sample_from, step_len)
            inds.append(inds_perstep)

        inds_to_inds = np.hstack([x.reshape(-1) for x in inds])
        inds = np.array(inds_sorted)[inds_to_inds]

    else:
        raise ValueError(f'{pacing_fn} not implemented.')

    output['inds'] = inds
    save_path = os.path.join(os.path.dirname(path), 'inds_'+pacing_fn+'.pkl')
    if save_output:
        save_to_pickle(save_path, output)

    fn_output = {
        'output': output,
        'scores': scores,
        'inds': inds,
        'inds_to_inds': inds_to_inds,
        'pacing_fn': pacing_fn,
        'N': N,
    }

    if save_output:
        return output
    else:
        return fn_output


def generate_curriculum_forall_scoring_fn(
        model_name_list, 
        dataset_list, 
        pacing_fn_list, 
        verbose=False
    ):
    from rosemary import jpt_in_notebook
    import pandas as pd
    if not isinstance(model_name_list, list):
        model_name_list = [model_name_list]
    if not isinstance(pacing_fn_list, list):
        pacing_fn_list = [pacing_fn_list]
    if not isinstance(dataset_list, list):
        dataset_list = [dataset_list]
        
    paths = glob.glob('curriculum/*/*/*/scores.pkl')
    paths = [x for x in paths if \
                any(y in x for y in model_name_list) and \
                any(y in x for y in dataset_list)]
    data = [scores_path_to_attrs(path) for path in paths]
    df = pd.DataFrame(data)
    if verbose and jpt_in_notebook():
        from IPython.display import display
        display(df)
    output_list = []
    for path, pacing_fn in itertools.product(paths, pacing_fn_list):
        output = generate_curriculum(path, pacing_fn, verbose=verbose)
        output_list.append(output)
    return output_list


if __name__ == '__main__':

    model_name = 'llama-7b'; dataset = 'tulu_v1_mix'; M = 150_000
    # model_name = 'mistral-7b'; dataset = 'ultrachat'; M =  50_000

    pacing_fn_list = [
        f'prune_size={M}_ep=1', # for `scoring_fn=random` baselines
        f'prune_size={M}_ep=3',
        f'singlestep_size={M}_startingfrac=0.2',
        f'singlestep_size={M}_startingfrac=0.1',
    ]

    output_list = generate_curriculum_forall_scoring_fn(
        model_name, dataset, pacing_fn_list, verbose=False)