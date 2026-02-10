# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import numpy as np

import ray
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