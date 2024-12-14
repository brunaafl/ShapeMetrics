import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from tqdm import trange

datastore = np.load("./data/noise_sim.npz")

D = datastore['distmat']
fish_infos = datastore['fish_infos']
n_animals = len(fish_infos)

Z = MDS(
    20, dissimilarity='precomputed', n_init=20
).fit_transform(D)
x, y = PCA(2).fit_transform(Z).T

fig, ax = plt.subplots(1, 1)
# ax.scatter(x, y, lw=0, c=np.log(fish_infos))
ax.scatter(x, y, lw=0, c=np.argsort(fish_infos))
ax.set_title("Shape Space, colored by fisher information")

model = KNeighborsRegressor(metric="precomputed", n_neighbors=3)

preds = []
for i in trange(n_animals):
    train_idx = [i for i in range(n_animals) if i != 5]
    model.fit(D[train_idx][:, train_idx], fish_infos[train_idx])
    preds.append(model.predict(D[i][train_idx][None, :]))

fig, ax = plt.subplots(1, 1)
ax.scatter(fish_infos, preds, lw=0, color='k', s=30)
a = np.min(fish_infos)
b = np.max(fish_infos)
ax.plot([a, b], [a, b], "--r")
ax.set_xlabel("true fisher information")
ax.set_ylabel("predicted fisher information")
ax.set_xscale("log")
ax.set_yscale("log")

fig, ax = plt.subplots(1, 1)
ax.plot(np.linalg.svd(Z, compute_uv=False), '.-')
ax.set_ylabel("singular value")
plt.show()
