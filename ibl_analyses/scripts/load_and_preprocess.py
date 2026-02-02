
import jax.numpy as jnp

import sys
import pickle

import utils
import loader
import tqdm
import imp
imp.reload(loader)
import setup
import scipy.stats as sts
from matplotlib import pyplot as plt

import numpy as np


def preprocess_raw(params, dataloader):

    #n_folds = 1
    n_folds = 100

    train_data_folds, test_data_folds = dataloader.new_folds(n_folds)

    __train_data_folds = []
    __test_data_folds = []

    for i in range(n_folds):
        _,ys,_,cs = train_data_folds[i]
        _,ys_t,_,cs_t = test_data_folds[i]

        # filter bad neurons/sessions
        min_neurons = 100
        min_trials = 20

        valid = [i for i in range(len(ys)) if ys[i].shape[2] >= min_neurons and ys[i].shape[0] >= min_trials]
        ys = [y[:,:,jnp.argsort(y.mean(0).std(0))[:100]] for y in ys]

        ys = [ys[i].mean(0) for i in valid]
        cs = [cs[i].mean(0) for i in valid]

        ys_t = [ys_t[i].mean(0) for i in valid]
        cs_t = [cs_t[i].mean(0) for i in valid]

        S = len(ys)
        __train_data_folds.append([ys,cs])
        __test_data_folds.append([ys_t,cs_t])

    # # save pickle
    with open("100_folds_1_time_point.pkl","wb") as f:
        pickle.dump([params,__train_data_folds,__test_data_folds],f)

    xs,ys,rs,cs = dataloader.load_train_data()

    valid_cs = np.array([cs[i] for i in valid],dtype=object)

    with open("raw_behavior.pkl","wb") as f:
        pickle.dump(cs,f)

params = {
    'file': '../data_new/', #'../../../Data/',
    'tag': '2022_Q2_IBL_et_al_RepeatedSite',
    'probe': 'probe00',
    'sessions': [0,5,6],
    'areas': ['CA1','DG','LP','PO','VISa'],
    'props':{'train':.5,'test':.5,'validation':0},
    'seeds':{'train':0,'test':1,'validation':2},
    'n_neurons': None, # all neurons
    'n_trials': None, # all trials
    'pre_time':0,
    'post_time':0.4,
    'n_bins': 10,
    'align_to': 'response',
    'train_trial_prop':.9, 
    'train_condition_prop':1, 
    'seed':0,
    'verbose': True,
    'bins_as_conds': True,
    'mode':'remote' ## Local or remote. Remote if testing
    }

dataloader = loader.IBLDataLoader(
         params,eids=setup.good_eids
)

preprocess_raw(params, dataloader)
