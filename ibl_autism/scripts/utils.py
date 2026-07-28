"""
Helper functions for loading given genotype, animal, region
"""

import os
import glob
import h5py

import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from scipy.cluster import hierarchy

DATA_DIR = Path('/home/blopes/ShapeMetrics/ibl_autism/data/')
BEHAVIOR_DIR = Path('/home/blopes/ShapeMetrics/ibl_autism/data_behavior/')

#%%
"""
Basic functions to list regions and animals for a given genotype
"""

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


#%%
"""
Load kernels for a given genotype, animal, region
"""

def load_animal_region_kernels(genotype, animal_id, region, r2_cutoff=0.01, load_prior=False):
    """ Given A SPECIFIC ANIMAL ID, A SPECIFIC REGION and its genotype, load the contrast kernels for all units of an specific region"""
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
    """ Given a SPECIFIC REGION and its genotype, list all animals that contain this region and, for each, load the contrast kernels for all units"""
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
    """ Given A LIST OF REGIONS, A SPECIFIC ANIMAL ID and its genotype, load the contrast kernels for all units of an specific region"""

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

def load_animal_region_var(genotype, animal_id, region, var='subjective_prior', r2_cutoff=0.01):
    # This loads the pgam's nonlinear transformation of the subjective prior 
    list_units = glob.glob(os.path.join(DATA_DIR, genotype, region, f'gam_fit_useCoupling0_*_{region}_{animal_id}_*.mat'))
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
        
        variables = [results[i]['variable'] for i in range(len(results))]
        idx = np.where(np.array(variables) == var)[0]

        # Extract subjective prior kernel
        prior_kernels = results[idx[0]]['kernel'] 
        
        all_kernels.append(prior_kernels[:100])

    if len(all_kernels) == 0:
        return np.array([])
    
    result_kernels = np.array(all_kernels)  # shape (n_units, 100)
    
    return result_kernels

def load_animal_var(genotype, regions, animal_id, var='subjective_prior', r2_cutoff=0.01):
    all_kernels = []
    for region in regions:
        result = load_animal_region_var(genotype, animal_id, region, var=var, r2_cutoff=r2_cutoff)
        if len(result) > 0:
            all_kernels.append(result)

    if len(all_kernels) == 0:
        return np.array([])
    
    result_kernels = np.concatenate(all_kernels, axis=0)  # shape (n_units, n_time_bins)
    return result_kernels


#%%

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



#%% 
## Processing functions

def filter_kernels_outliers(kernels, percentile_upper=95, percentile_lower=5, return_mask=False):
    # Remove extreme units/outliers
    # Get min and max for each unit
    kernel_max = kernels.max(axis=(1, 2))  # shape (n_units,)
    kernel_min = kernels.min(axis=(1, 2))
    
    # Compute percentile thresholds
    upper_threshold = np.percentile(kernel_max, percentile_upper)
    lower_threshold = np.percentile(kernel_min, percentile_lower)
    
    # Create filtering mask
    mask = (kernel_max <= upper_threshold) & (kernel_min >= lower_threshold)
    
    if return_mask:
        # Also filter the prior kernels with the same mask
        return kernels[mask], mask

    return kernels[mask]

def filter_distributions_outliers(k_animals, percentile_upper=97.5,  return_mask=False):
    # Filter out subjects with standard deviation across neurons too big
    stds = [np.abs(a).std().mean() for a in k_animals]
    filtered = [a for a, s in zip(k_animals, stds) if s - np.percentile(stds, percentile_upper) < 0.01]
    if return_mask:
        mask = [(s - np.percentile(stds, percentile_upper) < 0.01) for s in stds]
        return filtered, mask
    return filtered

def detect_outlier_subjects(kernel_list, mad_factor=3.0):
    # Detect outlier subjects based on the amplitude of their mean kernel
    
    # Mean kernel per subject
    amplitudes = np.array([np.abs(k.mean(axis=0)).max() for k in kernel_list])

    med = np.nanmedian(amplitudes)
    std = np.nanstd(amplitudes)

    amplitude_threshold = med + mad_factor * std
    
    keep_mask = amplitudes <= amplitude_threshold
    return keep_mask, amplitudes

#%% 
# Plotting functions

def plot_mds_pca(dist_matrix, labels, n_components=50, ax=None, title=False):

    # DIstance matrices are not euclidean so PCA directly is not ideal
    # MDS on the distance matrix
    mds = MDS(n_components=n_components, 
              dissimilarity='precomputed', 
              random_state=66,
              normalized_stress=False,
              n_init=10,)
    
    mds_coords = mds.fit_transform(dist_matrix)

    #mds_coords = (mds_coords - mds_coords.mean(axis=0)) / mds_coords.std(axis=0)

    # PCA on MDS coordinates
    pca_2d = PCA(n_components=2, random_state=42)
    pca_coords = pca_2d.fit_transform(mds_coords)

    colors = {'N':'#0D0D0D', 'F':'#61A656', 'C':'#D99C2B', 'S':'#A67232'}
    map_labels = {'C':'Cntnap2', 'F':'Fmr1', 'N':'C57BL6', 'S':'Shank3'}

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # PCA on MDS plot
    for g in np.unique(labels):
        mask = np.array(labels) == g
        ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                        c=colors[g], label=map_labels[g], s=60, edgecolors='white', linewidths=0.5)
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    if title:
        ax.set_title('PCA on MDS embedding', fontsize=13, fontweight='bold')
    ax.legend()

    return ax, pca_coords

def plot_dendogram(dist_neural, labs, fig=None, save=False, lab_to_color=None, title=False):
    
    S = dist_neural.shape[0]
    
    dist_neural_flat = dist_neural[np.triu_indices(S,1)] #squareform(dist_neural)

    linkage = hierarchy.ward(dist_neural_flat)
    linkage = hierarchy.optimal_leaf_ordering(linkage, dist_neural_flat)

    if fig is None:
        fig = plt.figure(figsize=(4,5))

    # Plot all together in the same figure...
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.02)

    ax_dendro = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1])

    dn = hierarchy.dendrogram(
        linkage, 
        ax=ax_dendro, 
        color_threshold=0, 
        above_threshold_color='k'
    )
    ax_dendro.axis('off')

    leaf_order = hierarchy.leaves_list(
        hierarchy.optimal_leaf_ordering(linkage, dist_neural_flat)
    )
    if lab_to_color is None:
        unique_labs = np.unique(labs)
        cmap = plt.get_cmap('tab10') 
        lab_to_color = {lab: cmap(i / len(unique_labs)) for i, lab in enumerate(unique_labs)}

    x_coords = np.arange(5, S * 10 + 5, 10)

    for i, original_idx in enumerate(leaf_order):
        lab = labs[original_idx]
        color = lab_to_color[lab]
        ax_dendro.scatter(x_coords[i], 0, color=color, s=60, zorder=10, clip_on=False)

    ordered_dist = dist_neural[leaf_order, :][:, leaf_order]
    
    ax_heatmap.imshow(ordered_dist, cmap="gray", aspect='auto', 
                      vmin=np.min(ordered_dist), vmax=np.max(ordered_dist))

    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])

    if title:
        plt.title('Distance matrix with dendrogram ordering', fontsize=12, y=1.2)
    
    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)

    if save:
        plt.savefig(f"distance_dendrogram.png", dpi=300, bbox_inches='tight')
    
    return fig, ax_dendro, ax_heatmap
