
import jax.numpy as jnp

import argparse
import sys
import pickle

import utils
import ibl_analyses.scripts.loader as loader
import tqdm
import imp
imp.reload(loader)
import setup
import scipy.stats as sts
from matplotlib import pyplot as plt

import numpy as np

# %%


parser = argparse.ArgumentParser()
parser.add_argument('--align_to', default='response')
parser.add_argument('--mode', default='local')
parser.add_argument('--bin_as_conds', default='False', choices=['True', 'False'])

args = parser.parse_args()

params = {
    'file': '../notebooks/data_andrew/', #'../../../Data/',
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
    'align_to': args.align_to,
    'train_trial_prop':.9, 
    'train_condition_prop':1, 
    'seed':0,
    'verbose': True,
    'bins_as_conds': args.mode == 'True',
    'mode':args.mode, ## Local or remote. Remote if testing
    }


dataloader = loader.IBLDataLoader(
         params,eids=setup.all_eids
)

utils.lazy_raw_process(params, dataloader)
