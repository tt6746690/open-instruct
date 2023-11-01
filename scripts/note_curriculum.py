import os
import re
import random
import pickle
import glob
import itertools

import numpy as np


from note_pruning import save_to_pickle
from note_pruning_analysis import curriculum_dir



def convert_existing_data_inds_to_curriculum_scores():
    from note_pruning_analysis import get_sorted_inds

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


def generate_curriculum(path, pacing_fn, verbose=False):
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

        if M >= len(inds_sorted):
            raise ValueError(f'size={M} > len(inds)={len(inds_sorted)}')

        sizes = []
        for _ in range(num_epochs-1):
            sizes.append(int(M//num_epochs))
        sizes.append(int(M-np.sum(sizes)))
            
        inds = []
        for size in sizes:
            inds_kept = list(inds_sorted[:size])
            np.random.shuffle(inds_kept)
            inds.append(inds_kept)
        inds = [x for sl in inds for x in sl]
        inds = np.array(inds)

        if len(inds) != M:
            raise ValueError('len(inds) should be equal to size')
    elif pacing_fn.startswith('singlestep'):

        match = re.search(r'size=([^_]+)', pacing_fn)
        M = int(match.group(1))
        match = re.search(r'startingfrac=([\d.]+)', pacing_fn)
        startingfrac = float(match.group(1))

        inds_step1 = list(inds_sorted[:int(startingfrac*N)])
        inds_step1_size = int(M//2)
        if verbose:
            print(f'Step 1: sample {inds_step1_size} from first {len(inds_step1)} examples.')
        inds_step1 = np_random_choice_maximize_noreplacement(inds_step1, inds_step1_size)

        inds_step2 = list(inds_sorted)
        inds_step2_size = M-inds_step1_size
        if verbose:
            print(f'Step 2: sample {inds_step2_size} from the rest of {len(inds_step2)} examples.')
        inds_step2 = np_random_choice_maximize_noreplacement(inds_step2, inds_step2_size)

        inds = [inds_step1, inds_step2]
        inds = np.hstack([x.reshape(-1) for x in inds])
        
        if len(inds) != M:
            raise ValueError('len(inds) should be equal to size')
    else:
        raise ValueError(f'{pacing_fn} not implemented.')

    output['inds'] = inds
    save_path = os.path.join(os.path.dirname(path), 'inds_'+pacing_fn+'.pkl')
    save_to_pickle(save_path, output)

    return output


def generate_curriculum_forall_scoring_fn(
        model_name_list, 
        dataset_list, 
        pacing_fn_list, 
        verbose=False
    ):
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