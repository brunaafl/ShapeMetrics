import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from tqdm import tqdm, trange
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

R = np.load("simulated_data.npz")
K = R["N_ANIMALS"]
D = R["distmat_orth"][:K][:, :K]
idx = np.triu_indices_from(D, 1)
distortions = []
pca_varexp = []
num_dims = np.arange(2, 21)

for d in tqdm(num_dims):
    Z = MDS(
            d, dissimilarity='precomputed', random_state=123,
            n_init=100, normalized_stress="auto"
        ).fit_transform(D)
    D_est = squareform(pdist(Z, "euclidean"))
    distortions.append(
        np.mean(np.maximum(D[idx] / D_est[idx], D_est[idx] / D[idx])) - 1
    )
    pca_varexp.append(
        PCA(2).fit(Z).explained_variance_ratio_.sum()
    )

print("Mean Distortion:", distortions[-1])
print("PCA var explained:", pca_varexp[-1])


cluster_labels = AgglomerativeClustering(
    metric="precomputed", linkage="average"
).fit_predict(R["distmat_perm"])


# plt.plot(num_dims, distortions)
