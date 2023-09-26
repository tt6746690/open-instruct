import os
import numpy as np
import time

import pickle

from sklearn.cluster import KMeans


save_dir = '/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts'
save_path = os.path.join(save_dir, 'note_explore_dpp_llama-7b_flan_v2_outputs.pkl')

with open(save_path, 'rb') as f:
    d = pickle.load(f)
        
# some entries are nan.
d['log_probs'] = np.nan_to_num(d['log_probs'], nan=np.nanmean(d['log_probs']))
text_embeddings = d['text_embeddings']
log_probs = d['log_probs'].squeeze()


# n_clusters_list = [10]
# X = text_embeddings[:100,:]

X = text_embeddings
n_clusters_list = [100, 1000, 10000]

for n_clusters in n_clusters_list:

    s = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto', verbose=True)
    kmeans.fit(X)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    D = np.linalg.norm(X - cluster_centers[cluster_labels], axis=1)
    e = time.time()

    print(f'kmeans took {e-s:.2f} seconds')


    ## sort by increasing/decreasing
    save_path = os.path.join(
        save_dir, f'note_explore_dpp_llama-7b_flan_v2_subsets_kmeansl2_{str(n_clusters)}_incr.pkl')
    inds = np.argsort(D).tolist()
    with open(save_path, 'wb') as f:
        pickle.dump({'K': inds, 'D': D[inds].tolist()}, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_path = os.path.join(
        save_dir, f'note_explore_dpp_llama-7b_flan_v2_subsets_kmeansl2_{str(n_clusters)}_decr.pkl')
    inds = inds[::-1]
    with open(save_path, 'wb') as f:
        pickle.dump({'K': inds, 'D': D[inds].tolist()}, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        

