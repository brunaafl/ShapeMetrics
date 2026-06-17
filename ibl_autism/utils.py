"""
Helper functions for loading given genotype, animal, region
"""

import os
import glob
import scipy.io as sio
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

DATA_DIR = Path('/home/blopes/ShapeMetrics/ibl_autism/data/')
BEHAVIOR_DIR = Path('/home/blopes/ShapeMetrics/ibl_autism/data_behavior/')

def load_animal_region_kernels(genotype, animal_id, region, r2_cutoff=0.01, load_prior=False):
    list_units = glob.glob(os.path.join(DATA_DIR, genotype, region, f'gam_fit_useCoupling0_*_{region}_{animal_id}_*.mat'))
    list_units = sorted(list_units)  
    all_kernels = []
    prior_kernels = []
    
    for path_unit in list_units:
        mat_data = sio.loadmat(path_unit, squeeze_me=True)
        results = mat_data['results']
        
        # Quality filter
        if results[0]['full_pseudo_r2_eval'] < r2_cutoff:
            continue
            
        # Check for NaN mutual_info
        if np.isnan(results[0]['mutual_info']):
            continue

        # Extract contrast kernels (indices 0-8)
        contrast_kernels = results[0:9]
        # kernel or kernel_Hz? Kernel Hz is apparently in firing rate scale, whatever it means
        contrast_kernels = np.array([ck['kernel'] for ck in contrast_kernels])  # shape (9, 106)

        if load_prior:
            # Extract subjective prior kernel
            variables = [results[i]['variable'] for i in range(len(results))]
            idx = np.where(np.array(variables) == 'subjective_prior')[0]

            # Keep unit only when both contrast and prior kernels are available.
            if len(idx) > 0:
                prior_kernel = results[idx[0]]['kernel']  # shape (100,)
                all_kernels.append(contrast_kernels)
                prior_kernels.append(prior_kernel)
        else:
            all_kernels.append(contrast_kernels)

    if len(all_kernels) == 0:
        if load_prior:
            return np.array([]), np.array([])
        return np.array([])
    
    result_kernels = np.array(all_kernels)  # shape (n_units, 9, 106)
    
    if load_prior:
        result_priors = np.array(prior_kernels)  # shape (n_units, 100)
        return result_kernels, result_priors
    else:
        return result_kernels


def load_region_kernels(genotype, region, r2_cutoff=0.01):
    # Load contrast kernels for all animals in a given region with quality filters from matlab code
    list_units = glob.glob(os.path.join(DATA_DIR, genotype, region, f'gam_fit_useCoupling0_*_{region}_*.mat'))
    list_units = sorted(list_units)  

    all_kernels = []
    
    for path_unit in list_units:
        mat_data = sio.loadmat(path_unit, squeeze_me=True)
        results = mat_data['results']
        
        # Quality filter
        if results[0]['full_pseudo_r2_eval'] < r2_cutoff:
            continue
            
        # Validity filter
        if np.isnan(results[0]['mutual_info']):
            continue

        # Extract contrast kernels
        contrast_kernels = results[0:9]
        contrast_kernels = np.array([ck['kernel'] for ck in contrast_kernels])
        
        all_kernels.append(contrast_kernels)

    if len(all_kernels) == 0:
        return np.array([])
    
    result_kernels = np.array(all_kernels)
    
    return result_kernels


def load_animal_kernels(genotype, regions, animal_id, r2_cutoff=0.01, load_prior=False):
    all_kernels = []
    prior_kernels = []

    for region in regions:
        result = load_animal_region_kernels(genotype, animal_id, region, r2_cutoff=r2_cutoff, load_prior=load_prior)

        if load_prior:
            region_kernels, region_priors = result
            if len(region_kernels) > 0:
                all_kernels.append(region_kernels)
            if len(region_priors) > 0:
                prior_kernels.append(region_priors)
        else:
            region_kernels = result
            if len(region_kernels) > 0:
                all_kernels.append(region_kernels)

    if len(all_kernels) == 0:
        return np.array([])
    
    result_kernels = np.concatenate(all_kernels, axis=0)  # shape (n_units, n_contrasts, n_time_bins)
    if load_prior:
        result_priors = np.concatenate(prior_kernels, axis=0)  # shape (n_units, 100)
        return result_kernels, result_priors
    else:
        return result_kernels

#%%

def load_animal_region_prior(genotype, animal_id, region):
    # This loads the pgam's nonlinear transformation of the subjective prior 
    list_units = glob.glob(os.path.join(DATA_DIR, genotype, region, f'gam_fit_useCoupling0_*_{region}_{animal_id}_*.mat'))
    list_units = sorted(list_units)  

    all_kernels = []
    
    for path_unit in list_units:
        mat_data = sio.loadmat(path_unit, squeeze_me=True)
        results = mat_data['results']
        
        variables = [results[i]['variable'] for i in range(len(results))]
        idx = np.where(np.array(variables) == 'subjective_prior')[0]

        # Extract subjective prior kernel
        prior_kernels = results[idx[0]]['kernel']  # shape (100,)
        
        all_kernels.append(prior_kernels)

    if len(all_kernels) == 0:
        return np.array([])
    
    result_kernels = np.array(all_kernels)  # shape (n_units, 100)
    
    return result_kernels

def load_animal_prior(genotype, regions, animal_id, r2_cutoff=0.01):
    all_kernels = []
    print(animal_id)
    for region in regions:
        print(region)
        result = load_animal_region_prior(genotype, animal_id, region, r2_cutoff=r2_cutoff)
        print(result.shape)
        if len(result) > 0:
            all_kernels.append(result)

    if len(all_kernels) == 0:
        return np.array([])
    
    result_kernels = np.concatenate(all_kernels, axis=0)  # shape (n_units, n_contrasts, n_time_bins)
    return result_kernels


#%%
def check_number_of_units(genotype, region):

    # Load predictors of the pGAM for all units of this region and all animals
    list_units = glob.glob(os.path.join(DATA_DIR, genotype, region, f'gam_fit_useCoupling0_*_{region}_*.mat'))
    list_units = sorted(list_units)  

    return len(list_units)

def list_regions(genotype):

    # List all regions for which we have data for this genotype
    list_regions = glob.glob(os.path.join(DATA_DIR, genotype, '*'))
    list_regions = sorted(list_regions)  

    list_regions = [os.path.basename(path) for path in list_regions]

    return list_regions

def list_animals(genotype):

    # List all animals for which we have data for this genotype
    list_animals = glob.glob(os.path.join(DATA_DIR, genotype, '*', f'gam_fit_useCoupling0_*.mat'))
    list_animals = [os.path.basename(path).split('_')[6] for path in list_animals]
    list_animals = list(set(list_animals)) # unique animals
    list_animals = sorted(list_animals)  # sort unique animals

    return list_animals


def load_behavior(animal_id=None):
    # Load the behavior data for all animals, or for a given animal if animal_id is specified
    mat = sio.loadmat(BEHAVIOR_DIR / 'summary_behavior.mat', squeeze_me=True)
    
    # Format everything into a pandas dataframe to acess 
    df = pd.DataFrame({
        'animal': mat['master_animal'],
        'contrast': mat['master_contrast'],
        'choice': mat['master_choice'],
        'feedback': mat['master_feedback'],
        'prob_left': mat['master_probLeft'],
    })

    # Change -1 to 0 for feedback to average afterwards
    df['feedback'] = (df['feedback'] == 1).astype(int) 

    if animal_id is not None:
        if isinstance(animal_id, str):
            animal_id = [animal_id]
        df = df[df['animal'].isin(animal_id)].reset_index(drop=True)

    return df

def add_genotype_info(df):
    # Add a column with the genotype information based on the animal id
    def get_genotype(animal_id):
        if animal_id.startswith('N'):
            return 'N'
        elif animal_id.startswith('C'):
            return 'C'
        elif animal_id.startswith('F'):
            return 'F'
        elif animal_id.startswith('S'):
            return 'S'
        else:
            return 'Unknown'

    df['genotype'] = df['animal'].apply(get_genotype)
    return df

def filter_kernels_outliers(kernels, percentile_upper=95, percentile_lower=5, prior=None):
    # Remove extreme units/outliers
    # Get min and max for each unit
    kernel_max = kernels.max(axis=(1, 2))  # shape (n_units,)
    kernel_min = kernels.min(axis=(1, 2))
    
    # Compute percentile thresholds
    upper_threshold = np.percentile(kernel_max, percentile_upper)
    lower_threshold = np.percentile(kernel_min, percentile_lower)
    
    # Create filtering mask
    mask = (kernel_max <= upper_threshold) & (kernel_min >= lower_threshold)
    
    if prior is not None:
        # Also filter the prior kernels with the same mask
        return kernels[mask], prior[mask]

    return kernels[mask]


def detect_outlier_subjects(kernel_list, mad_factor=3.0):
    # Detect outlier subjects based on the amplitude of their mean kernel
    
    # Mean kernel per subject
    amplitudes = np.array([np.abs(k.mean(axis=0)).max() for k in kernel_list])

    med = np.nanmedian(amplitudes)
    std = np.nanstd(amplitudes)

    amplitude_threshold = med + mad_factor * std
    
    keep_mask = amplitudes <= amplitude_threshold
    return keep_mask, amplitudes
