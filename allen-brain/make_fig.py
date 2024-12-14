import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import json
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from netrep.metrics import LinearMetric, PermutationMetric
from netrep.multiset import pairwise_distances

# Specify path to data.
DATAPATH = "/mnt/home/awilliams/ceph/netrep-data/allen-brain"

# Constants.
MIN_NEURONS = 50
SIGMA = 20              # gaussian smoothing bandwidth for spike trains.

# Load responses into dictionary.
datadict = dict()
for area, activity in dict(np.load(
            os.path.join(
            DATAPATH,
            "extracted_data",
            "natural_movie_three_responses.npz"
        ))).items():
    n_neurons = activity.shape[0]
    if n_neurons >= MIN_NEURONS:
        datadict[area] = activity.T

# Exclude regions that are not interpretable (e.g. white matter)
for excluded_region in ["MB", "alv", "ccs", "dhc", "fp", "or"]:
    try:
        datadict.pop(excluded_region)
    except:
        pass

# Pre-process spike trains. Normalize spike counts, smooth trains, PCA to MIN_NEURONS
print("Performing PCA preprocessing...")
sorted_keys = np.sort(list(datadict.keys()))
Xs = []
for area in tqdm(sorted_keys):
    _x = datadict[area]
    # _x /= np.linalg.norm(_x, axis=0, keepdims=True)
    _x = gaussian_filter1d(_x, SIGMA, axis=0)
    _x = PCA(MIN_NEURONS).fit_transform(_x) # this will mean-center the columns
    _x = _x / np.linalg.norm(_x.ravel())
    Xs.append(_x)

# Number of brain regions with at least MIN_NEURONS
NUM_REGIONS = len(Xs)

# Compute pairwise distances.
print("Fitting pairwise distances...")
rot_dists = pairwise_distances(
    LinearMetric(alpha=1.0, score_method="euclidean"),
    Xs
)
perm_dists = pairwise_distances(
    PermutationMetric(score_method="euclidean"),
    Xs
)

# Perform multi-dimensional scaling.
for title, dists in zip(("Procrustes", "Permutation"), (rot_dists, perm_dists)):
    print("Performing MDS...")
    embedding = MDS(
        n_components=50,
        dissimilarity="precomputed",
        normalized_stress=False,
        random_state=0,
        n_init=10,
    ).fit_transform(dists)
    x, y = PCA(2).fit_transform(embedding).T

    # Load structure tree.
    from allensdk.core.reference_space_cache import ReferenceSpaceCache
    REF_SPACE_RES = 10
    REF_SPACE_KEY = "annotation/ccf_2017"
    tree = ReferenceSpaceCache(
        REF_SPACE_RES,
        REF_SPACE_KEY,
        manifest=os.path.join(DATAPATH, "rsp_manifest.json")
    ).get_structure_tree(structure_graph_id=1)
    ref_tree_structures = tree.get_structures_by_acronym(sorted_keys)
    c = np.array([s["rgb_triplet"] for s in ref_tree_structures]) / 255

    # Plot result.
    print("Plotting result...")
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    texts = []
    for _x, _y, s in zip(x, y, ref_tree_structures):
        texts.append(
            ax.text(_x, _y, s["acronym"], fontsize=12)
        )
    ax.scatter(x, y, lw=0, s=30*3, c=c)
    ax.axis("off")
    ax.set_title(title)
    fig.savefig(title + "_mds.pdf")
    plt.close(fig)

    # Visualize distance matrices.
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    linkage = hierarchy.ward(squareform(dists))
    idx = hierarchy.leaves_list(
        hierarchy.optimal_leaf_ordering(linkage, squareform(dists))
    )
    ax.imshow(dists[idx][:, idx], cmap="gray", clim=(
        np.min(dists[np.triu_indices_from(dists, 1)]),
        np.max(dists[np.triu_indices_from(dists, 1)])
    ))
    ax.set_title(title)
    fig.savefig(title + "_distmat.pdf")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    dn = hierarchy.dendrogram(
        linkage, ax=ax,
        link_color_func=lambda x: 'k',
        leaf_label_func=lambda x: ref_tree_structures[x]["acronym"]
    )
    x00 = np.array(dn["icoord"]).min()
    x11 = np.array(dn["icoord"]).max()
    ax.scatter(
        np.linspace(x00, x11, NUM_REGIONS),
        np.zeros(NUM_REGIONS)+.02,
        lw=0, s=60, c=c[dn["leaves"]], zorder=100
    )
    ax.set_title(title)
    fig.savefig(title + "_dn.pdf")
    plt.show()
