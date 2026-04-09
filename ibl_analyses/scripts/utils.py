# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""
import ray
import pickle

import numpy as np
import jax.numpy as jnp

from netrep.metrics import LinearMetric
from netrep.metrics import GaussianStochasticMetric

from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances

# %%
@ray.remote
def _stochastic_metrics_pair(pair,alpha=2.,niter=1000):
    Xi,Xj = pair

    metric = GaussianStochasticMetric(alpha,niter=niter)
    metric.fit(Xi,Xj)
    dist_neural = metric.score(Xi,Xj)
    return dist_neural

# %%
def ssd(pairs,alpha=2.,niter=1000):
    refs = [
        _stochastic_metrics_pair.remote(pair,alpha,niter)
            for pair in pairs
    ]
    D = ray.get(refs)
    D = np.array(D)

    return D

# %%
@ray.remote
def _deterministic_metrics_pair(pair,alpha=0.):
    Xi,Xj = pair

    metric = LinearMetric(alpha=alpha,center_columns=True,score_method='euclidean')
    metric.fit(Xi,Xj)
    dist_neural = metric.score(Xi,Xj)
    return dist_neural

# %%
def dsd(pairs,alpha=0.):
    refs = [
        _deterministic_metrics_pair.remote(pair,alpha)
            for pair in pairs
    ]
    D = ray.get(refs)
    D = np.array(D)

    return D

# %%
def delay_embedding(data,tau,delta=1):
    K,C,T,N = data.shape
    embedding = np.zeros((K,C,T-(tau-1)*delta,N*tau))

    for d in range(tau):
        embedding[:,:,:,d*N:d*N+N] = data[:,:,(tau-1-d)*delta:T-d*delta]

    return embedding

# %%
def create_adjacency(x):
    idx = rankdata(x, method='dense',axis=0)-1
    dist = pairwise_distances(idx,metric='l1')
    dist[dist != 1] = 0
    return dist

# %%
def filter_eids(QC_json, criteria='PASS'):
    with open(QC_json, 'r') as f:
        import json
        QC_data = json.load(f)
    
    eids = []
    for QC_eid in QC_data:
        if QC_eid['qc_outcome'] == criteria:
            eids.append(QC_eid['eid'])
    return eids

# %%
# Load and preprocess spike trains
def filter_region(ys, cs, r, regions, n_trials, min_neurons=50, min_trials=20):
    ys_aux = []
    cs_aux = []
    valid = []
    for i in range(len(r)):
        region_filter = np.isin(r[i], regions)
        if sum(region_filter) >= min_neurons and ys[i].shape[0] >= min_trials:
            valid.append(i)
            if n_trials is not None:
                n_trials.append(ys[i].shape[0])
            y_masked = ys[i][:, :, region_filter]
            y_mean = y_masked.mean(0)
            neuron_std = y_mean.std(axis=0)
            idx = np.argsort(neuron_std)[:100]
            ys_aux.append(y_masked[:, :, idx].mean(0))
            cs_masked = np.array(cs[i])
            cs_masked = np.where(cs_masked == -1, 0, 1)
            cs_aux.append(cs_masked.mean(0))
            
    return ys_aux, cs_aux, valid, n_trials

def lazy_raw_process(params, dataloader, regions='all', n_folds = 50):
    """ 
    Cross-validation lazy loading
    Lazy loading and processing of the data: Using sessions with more than 20 trials and 100 neurons.
    Seprate regions if specified, and then get the ones with more than 20 trials and 100 neurons.
    After, save processed data and raw behavior in pickle files.
    """
    
    # filter bad neurons/sessions
    min_neurons = 100
    min_trials = 20

    __train_data_folds = []
    __test_data_folds = []
    
    n_trials = []

    for train_data_folds, test_data_folds in dataloader.generate_folds(n_folds):
        _,ys,_,cs,r = train_data_folds  # r = regions 
        _,ys_t,_,cs_t,_ = test_data_folds
        
        # Using all regions
        if regions=='all':
            valid = [i for i in range(len(ys)) if ys[i].shape[2] >= min_neurons and ys[i].shape[0] >= min_trials]
            
            ys = [y[:,:,jnp.argsort(y.mean(0).std(0))[:min_neurons]] for y in ys]

            if len(n_trials) == 0:
                n_trials = [ys[i].shape[0] for i in valid]

            ys = [ys[i].mean(0) for i in valid]

            cs_filter = [np.array(cs[i]) for i in valid]
            cs_filter = [np.where(c == -1, 0, 1).mean(0) for c in cs_filter]
            
        # Filter specific regions
        else:
            if len(n_trials) == 0:
                ys, cs_filter, valid, n_trials = filter_region(ys, cs, r, regions, n_trials, min_trials=10)
            else:
                ys, cs_filter, valid, _ = filter_region(ys, cs, r, regions, None, min_trials=10)

        # Average across trials  
        ys_t = [ys_t[i].mean(0) for i in valid]

        # Compute behavior tuning curve
        # Here, im changing wrong (for each subject) from -1 to 0, and then averaging all trials 
        # This will give me the probability of being correct
        cs_t_filter = [np.array(cs_t[i]) for i in valid]
        cs_t_filter = [np.where(c == -1, 0, 1).mean(0) for c in cs_t_filter]

        S = len(ys)
        __train_data_folds.append([ys,cs_filter])
        __test_data_folds.append([ys_t,cs_t_filter])

    # # save pickle
    with open(f"choice_100_folds_1_time_point_{regions}.pkl","wb") as f:
        pickle.dump([params,__train_data_folds,__test_data_folds],f)

    _,ys,_,cs,_ = dataloader.load_train_data()
    valid_cs = [cs[i] for i in valid]
    with open(f"raw_behavior_{regions}.pkl","wb") as f:
        pickle.dump([n_trials,valid_cs],f)

    return valid
