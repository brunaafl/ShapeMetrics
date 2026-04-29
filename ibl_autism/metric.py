import ray

import numpy as np
import jax.numpy as jnp

from netrep.metrics import LinearMetric
from netrep.metrics import GaussianStochasticMetric

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


def dsd(pairs,alpha=0.):
    refs = [
        _deterministic_metrics_pair.remote(pair,alpha)
            for pair in pairs
    ]
    D = ray.get(refs)
    D = np.array(D)

    return D
