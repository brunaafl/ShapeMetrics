# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""
import logging

import logging

import ray
import pickle
import tqdm
import gzip
import multiprocessing

import numpy as np
import jax.numpy as jnp
import scipy.stats as sts
import seaborn as sns

from netrep.metrics import LinearMetric
from netrep.metrics import GaussianStochasticMetric

from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
N_CPUS = multiprocessing.cpu_count()

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
            #neuron_std = y_mean.std(axis=0)
            #idx = np.argsort(neuron_std)[:100]
            ys_aux.append(y_masked.mean(0))
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
            
            #ys = [y[:,:,jnp.argsort(y.mean(0).std(0))[:min_neurons]] for y in ys]

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

# %%

def process_and_correlate_fold(fold, w=10, s=1, r='all'):

    all_corrs = [] # to save the correlations for each time bin

    ys_time, cs_time, regions = fold['ys'], fold['cs_t'], fold['r']

    S = len(ys_time)
    mask = np.triu_indices(S,1)

    # Behavioral distance doesnt change across windows

    corr_matrix = np.corrcoef(cs_time)
    dist_cc = 1 - corr_matrix
    dist_cc = dist_cc[mask]

    # Convert ys_time to jax arrays for a bit faster processing in dsd
    ys_time_jax = [jnp.array(ys_time[session]) for session in range(S)]

    # I think this can be optimized with matrix multiplications for gpu parallelization
    # Adding some stride to speed up
    for t in tqdm.tqdm(range(0, len(ys_time[0])-w, s)):  # Stride of s

        # get best min_neurons and current time point + window
        all_sessions = [ys_time_jax[session][t:t+w,:].reshape(-1, ys_time_jax[session].shape[-1]) for session in range(S)]

        try:
            # this will make the process waaay faster
            dist_neural_list = dsd([
                [all_sessions[i],all_sessions[j]]
                for i in range(S) 
                for j in range(i+1,S)]
            )

            # symmetrize the distance matrix
            dist_neural = np.zeros((S,S))
            dist_neural[np.triu_indices(S,1)] = dist_neural_list

            corr = sts.pearsonr(dist_neural[mask], dist_cc)[0]

        except (np.linalg.LinAlgError, ValueError):
            logger.warning(f"{t} linear algebra error/nan")
            # Can occour when the optimization problem does not converge?
            all_corrs.append(np.nan)
            continue
        
        all_corrs.append(corr)

    return all_corrs


import time

def process_fold(gzipped_fold, w=10, s=1):

    t_start = time.time()
    with gzip.open(f"./notebooks/{gzipped_fold}", 'rb') as f:
        fold = pickle.load(f)
    t_decomp = time.time()
    
    result = process_and_correlate_fold(fold, w=w, s=s)
    t_end = time.time()
    
    logger.debug(f"Fold {gzipped_fold}: decomp={t_decomp-t_start:.2f}s, process={t_end-t_decomp:.2f}s")
    return result

#@ray.remote
def load_and_process_fold(region, align_to, w=10, s=1):
    t_total_start = time.time()
    
    fold_path = f"./notebooks/prep_data/{region}/folds_index_{align_to}.pkl"
    with open(f"{fold_path}", "rb") as f:
        folds = pickle.load(f)

    logger.info(f"Loaded {len(folds['files'])} folds for region {region} and alignment {align_to}.")
    
    all_corrs_list = []
    for gzipped_fold in tqdm.tqdm(folds['files']):
        result = process_fold(gzipped_fold, w, s)
        all_corrs_list.append(result)
    
    t_total_end = time.time()
    logger.info(f"Region {region} ({align_to}): completed in {t_total_end-t_total_start:.2f}s")
    return np.array(all_corrs_list)


def plot_time_corrs(all_corrs, ax=None, w=10, s=1, save_path=None):

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5))

    """x_time = np.linspace(-1,1,len(all_corrs[list(all_corrs.keys())[0]][0]))
    x_time += np.diff(x_time)[0]*w"""

    # I think what i was doing is wrong, now im centering the time bins in the middle of the window and accounting for stride
    idx = np.array(list(range(0,150-w,s)))
    x_time = np.linspace(-1,1,150)
    #diff = np.diff(x_time)[0] * w / 2 # Centering time bins to the middle of the window
    diff = np.diff(x_time)[0] * w  # Centering time bins to the end of the window
    x_time = x_time[idx] + diff

    sig = {}

    for align_to, all_corrs_align in all_corrs.items():

        sig_align = (1-np.nanmean(all_corrs_align>0,0))*2 <0.05
        sig[align_to] = sig_align

        ax.plot(x_time,np.nanmean(all_corrs_align,0),color="darkblue" if align_to=='response' else 'darkred')
        ax.fill_between(x_time,[np.nanpercentile(a,2.5) for a in all_corrs_align.T],[np.nanpercentile(a,97.5) for a in all_corrs_align.T],
                        color='darkblue' if align_to=='response' else 'darkred',alpha=0.2,label=f'{align_to} aligned (95% CI.).')

    ax.plot(x_time[sig['stim']],np.ones_like(x_time[sig['stim']])*0.29,'|',color='darkred')
    ax.plot(x_time[sig['response']],np.ones_like(x_time[sig['response']])*0.275,'|',color='darkblue')
    sig_stim_resp = (1-np.nanmean(all_corrs['stim']<np.nanmean(all_corrs['response'],0),0))*2 <0.05
    ax.plot(x_time[sig_stim_resp],np.ones_like(x_time[sig_stim_resp])*0.26,'|',color='k')

    ax.plot(x_time,np.zeros_like(x_time),'--',color='k',alpha=0.5)
    ax.set_xlabel('time from alignment (s)')
    ax.set_ylabel('pearson correlation')
    ax.set_title('neural and behavioral distance\ncorrelation time course')
    ax.legend()
    ax.set_xlim(-1,1)
    ax.set_ylim(-0.1,0.3)
    ax.set_yticks([-0.1,0,0.1,0.2,0.3],['',0,'','','.3'])
    ax.set_xticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1],['-1','','-0.5','','0','','0.5','','1'])
    ax.legend(frameon=False)
    
    # change font color of legend
    leg = ax.get_legend()
    ltext  = leg.get_texts()

    plt.setp(ltext[1], color='darkred')
    plt.setp(ltext[0], color='darkblue')
    sns.despine()
    #fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax

def plot_all_regions(all_corrs, w=10, s=1, save_path=None):
    colors = {'VISa':'#03A6A6', 'CA1':'#88D94E', 'DG':'#4ED97D', 'LP':'#F28D9F', 'PO':'#F2AB8D'}
    
    # Calculate time axis based on actual data length
    # 150 is the number of time bins i sorted data 
    idx = np.array(list(range(0, 150 - w, s)))
    x_time = np.linspace(-1, 1, 150)
    #diff = np.diff(x_time)[0] * w / 2 # Centering time bins to the middle of the window
    diff = np.diff(x_time)[0] * w  # Centering time bins to the end of the window

    x_time = x_time[idx] + diff

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    
    for region, all_corrs_region in all_corrs.items():
        
        color = colors.get(region, '#000000')  # Default to black if region not in colors dict
        
        mean_corr = np.nanmean(all_corrs_region, 0)
        ax.plot(x_time, mean_corr, color=color, linewidth=2, label=region)
        ax.fill_between(x_time, 
                       [np.nanpercentile(a, 5) for a in all_corrs_region.T], 
                       [np.nanpercentile(a, 95) for a in all_corrs_region.T],
                       color=color, alpha=0.2)
    
    ax.plot(x_time, np.zeros_like(x_time), '--', color='k', alpha=0.5)
    ax.set_xlabel('time from alignment (s)')
    ax.set_ylabel('pearson correlation')
    ax.set_title('neural and behavioral distance\ncorrelation time course by region')
    ax.set_xlim(-1,1)
    ax.set_ylim(-0.1,0.35)
    ax.set_yticks([-0.1,0,0.1,0.2,0.35],['',0,'','','.35'])
    ax.set_xticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1],['-1','','-0.5','','0','','0.5','','1'])
    ax.legend(frameon=False, loc='upper left')
    sns.despine()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

#%%
# Get neura and behavioral distances 

def convert_choose_right(cs):
    contrasts = np.array([-100, -25, -12.5, -6.25, 0., 6.25, 12.5, 25., 100.])
    
    percent_right = np.empty((len(cs), len(contrasts)))

    for ii, percent_correct in enumerate(cs):
        # Convert to percent choose right
        percent_right[ii,:] = np.hstack((1-percent_correct[contrasts < 0],
                                percent_correct[contrasts >= 0]))
        
    return percent_right

def fit_psychmetric(cs, n_trials_total=None, dx=0.1):
    # Fit psychometric curves to each session
    contrasts = np.array([-100, -25, -12.5, -6.25, 0., 6.25, 12.5, 25., 100.])
    psym_pars = np.empty((len(cs), 4)) # psychometric parameters
    
    percent_right = np.empty((len(cs), len(contrasts)))

    xx = np.arange(-100, 100, dx)
    fit_curve = np.zeros((len(cs), len(xx)))

    for ii, percent_correct in enumerate(cs):
        """
        # Rempve since im converting to percent correct before
        behavior[behavior == -1] = 0
        percent_correct = np.nanmean(behavior == 1, axis=0)"""

        # Convert to percent choose right
        percent_right[ii,:] = np.hstack((1-percent_correct[contrasts < 0],
                                percent_correct[contrasts >= 0]))

        if n_trials_total is None:
            n_trials = [np.sum(~np.isnan(x)) for x in percent_correct.T]
        else:
            n_trials = [n_trials_total[ii]] * len(contrasts)

        # Format data for mle_fit_psycho
        # Row 1: conditions
        # Row 2: number of trials per condition
        # Row 3: percent left in each condition
        data = np.vstack((
            contrasts,
            n_trials,
            percent_right[ii,:]
        ))

        parmin = np.array([-20, -50, 0, 0])
        parmax = np.array([20, 50, 1, 1])
        psym_pars[ii,:], _ = \
            psy.mle_fit_psycho(data,
                               'erf_psycho_2gammas',
                               nfits=15,
                               parmin = parmin,
                               parmax = parmax)
        
        fit_curve[ii,:] = psy.erf_psycho_2gammas(psym_pars[ii,:], xx)

    return fit_curve, percent_right, psym_pars

def compute_behavioral_distance(cs, n_trials_total):
    fit_curve, percent_right, psycho_params = fit_psychmetric(cs, n_trials_total)

    dist_behavioral = np.array(
        [np.sqrt(np.sum((fit_curve[i,:] - fit_curve[j,:])**2 * 0.1))
        for i in range(len(cs))
        for j in range(len(cs))]).reshape(len(cs), len(cs))

    return dist_behavioral, psycho_params, percent_right


def get_dists(train_fold,test_fold=None, metric="area", n_trials_total=None):

    ys,cs = train_fold
    if test_fold is not None:
        _,cs = test_fold

    S = len(ys)

    #compute distance matrices
    # I can improve speed by only computing upper triangle and then symmetrizing
    dist_neural_list = dsd([
        [ys[i],ys[j]]
        for i in range(len(ys)) 
        for j in range(i+1,len(ys))]
    ) # its a list, the shape is S*(S-1)/2

    # symmetrize the distance matrix
    dist_neural = np.zeros((S,S))
    dist_neural[np.triu_indices(S,1)] = dist_neural_list
    dist_neural = dist_neural + dist_neural.T

    if metric=="sigmoid":
        dist_cc,_,_ = compute_behavioral_distance(cs, n_trials_total=n_trials_total)
    elif metric=="area":
        corr_matrix = np.corrcoef(cs)
        dist_cc = 1 - corr_matrix   
    return dist_neural,dist_cc,cs

def get_kneigh_decreasing(dist_neural,cs,S=10,W=2):
    """
    Compute the correlation between each subject behavior and average of the top K neighbors in neural space.
    cs: list of behavioral functions for each subject
    dist_neural: distance matrix between subjects in neural space
    S: number of subjects
    W: minimum number of neighbors to consider
    returns all_preds_top: (Kmax-W+1, S)
    """

    avg_cc = np.array(cs)
    all_preds_top = []

    for K in range(W,S+1):
        preds_top = []
        for subj in range(S):
            subj_idx = list(range(S))

            # ignore itself
            subj_idx.remove(subj)
            subj_idx = np.array(subj_idx)

            dists = dist_neural[subj][subj_idx]
            sorted_idx  = subj_idx[np.argsort(dists)]

            top_k = avg_cc[sorted_idx[K-W:K]]
            preds_top.append(sts.pearsonr(np.mean(top_k,0),avg_cc[subj])[0])

        all_preds_top.append(preds_top)

    all_preds_top = np.array(all_preds_top)
    return all_preds_top
