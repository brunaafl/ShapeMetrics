import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from tqdm import trange

def simple_cmap(colors, name='none'):
    """Create a colormap from a sequence of rgb values.
    cmap = simple_cmap([(1,1,1), (1,0,0)]) # white to red colormap
    cmap = simple_cmap(['w', 'r']) # white to red colormap
    cmap = simple_cmap(['r', 'b', 'r']) # red to blue to red
    """
    # check inputs
    n_colors = len(colors)
    if n_colors <= 1:
        raise ValueError('Must specify at least two colors')
    # convert colors to rgb
    colors = [colorConverter.to_rgb(c) for c in colors]
    # set up colormap
    r, g, b = colors[0]
    cdict = {'red': [(0.0, r, r)], 'green': [(0.0, g, g)], 'blue': [(0.0, b, b)]}
    for i, (r, g, b) in enumerate(colors[1:]):
        idx = (i+1) / (n_colors-1)
        cdict['red'].append((idx, r, r))
        cdict['green'].append((idx, g, g))
        cdict['blue'].append((idx, b, b))
    return LinearSegmentedColormap(name, {k: tuple(v) for k, v in cdict.items()})

R = np.load("simulated_data.npz")
cmap = sns.color_palette('husl', R["N_THETAS"])
rs = np.random.RandomState(111)

### PANELS A + E ###

idx = [
    np.argmax(R["sharpnesses"]),
    np.argmin(R["sharpnesses"]),
    np.argmax(R["concentrations"]),
    np.argmin(R["concentrations"])
]

panel_A = plt.figure(figsize=(1.8, 4))

for j, i in enumerate(idx):
    Q = np.linalg.qr(rs.randn(3, 3))[0]
    ax1 = panel_A.add_subplot(4, 2, 2 * j + 1)
    ax2 = panel_A.add_subplot(4, 2, 2 * j + 2, projection="3d")
    ax1.imshow(R["animals"][i].T, aspect="auto", interpolation="none")
    x, y, z = Q @ PCA(3).fit_transform(R["animals"][i]).T
    ax2.scatter(x, y, zs=z, c=cmap, lw=0, alpha=1)
    zlim = ax2.get_zlim()
    ax2.plot(x, y, zs=zlim[0]-1, lw=3, color='k', alpha=.3)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_linewidth(1)
    ax2.axis('off')

panel_A.subplots_adjust(top=.98, bottom=0, left=.02, right=1)
panel_A.savefig("A.pdf", transparent=True)

panel_D = plt.figure(figsize=(1.8, 4))

for j, i in enumerate(idx):
    Q = np.linalg.qr(rs.randn(3, 3))[0]
    ax1 = panel_D.add_subplot(4, 2, 2 * j + 1)
    ax2 = panel_D.add_subplot(4, 2, 2 * j + 2, projection="3d")
    img = ax1.imshow(
        R["rotated_animals"][i].T, aspect="auto", interpolation="none",
    )
    img.set_clim([-0.9, 0.9])
    x, y, z = Q @ PCA(3).fit_transform(R["animals"][i]).T
    ax2.scatter(x, y, zs=z, c=cmap, lw=0, alpha=1)
    zlim = ax2.get_zlim()
    ax2.plot(x, y, zs=zlim[0]-1, lw=3, color='k', alpha=.3)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_linewidth(1)
    ax2.axis('off')

panel_D.subplots_adjust(top=.98, bottom=0, left=.02, right=1)
panel_D.savefig("E.pdf", transparent=True)
plt.close("all")


### PANELS B + C + D ###
K = R["N_ANIMALS"]
x, y = PCA(2).fit_transform(
    MDS(20, dissimilarity='precomputed', random_state=123).fit_transform(
        R["distmat_orth"][:K][:, :K]
    )
).T
fig, axes = plt.subplots(
    3, 1, figsize=(1.6, 4.6),
    gridspec_kw=dict(
        height_ratios=[1, 1, 1.15]
    )
)
panel_b_scatter = axes[0].scatter(
    x, y, c=R["sharpnesses"], lw=0, s=20,
    cmap=simple_cmap([(.2, .2, .2), (1, .2, .2)])
)
panel_c_scatter = axes[1].scatter(
    x, y, c=R["concentrations"], lw=0, s=20,
    cmap=simple_cmap([(.2, .2, .2), (.2, .2, 1)])
)
axes[0].axis('off')
axes[1].axis('off')

model = KNeighborsRegressor(metric="precomputed", n_neighbors=3)

distmat = R["distmat_orth"][:K][:, :K]
preds, sqresids = [], []
for i in trange(K):
    train_idx = np.array([j for j in range(K) if i != j])
    model.fit(
        distmat[train_idx][:, train_idx],
        R["fish_infos"][train_idx]
    )
    pred = model.predict(distmat[i][train_idx][None, :])
    sqresids.append((pred - R["fish_infos"][i]) ** 2)
    preds.append(pred)

print(
    "Rsquared:",
    1 - np.sum(sqresids) / np.sum((R["fish_infos"][:K] - np.mean(R["fish_infos"][:K])) ** 2)
)

axes[2].scatter(preds, R["fish_infos"][:K], color='k', lw=0, s=10)
axes[2].tick_params("both", labelsize=8, pad=1, length=1)
fig.subplots_adjust(top=.98, bottom=.1, left=.02, right=1, hspace=.6)
bbox = axes[2].get_position()
axes[2].set_position([bbox.xmin + .25, bbox.ymin, bbox.width - .3, bbox.height])
axes[2].set_xlabel("est. information\n(3 nearest neighbors)")
axes[2].set_ylabel("true information\n(fisher)", labelpad=0)


xl, yl = axes[2].get_xlim(), axes[2].get_ylim()
axes[2].plot(
    [0, 2 *np.max(preds)],
    [0, 2 *np.max(preds)],
    "-r", alpha=.75, lw=2
)
axes[2].set_xlim(xl)
axes[2].set_ylim(yl)

fig.savefig("BCD.pdf", transparent=True)
for i in range(4):
    axes[0].text(x[idx[i]], y[idx[i]], str(i + 1), fontsize=20, color="#59A14F")
    axes[1].text(x[idx[i]], y[idx[i]], str(i + 1), fontsize=20, color="#59A14F")
    axes[0].plot(x[idx[i]], y[idx[i]], "o", color="#59A14F")
    axes[1].plot(x[idx[i]], y[idx[i]], "o", color="#59A14F")
fig.savefig("BCD_reference.pdf", transparent=True)

### PANELS F... ###
fig, axes = plt.subplots(
    3, 1, figsize=(1.6, 4.6),
    # gridspec_kw=dict(
    #     height_ratios=[1, 1, 1.15]
    # )
)
x, y = PCA(2).fit_transform(
    MDS(20, dissimilarity='precomputed', random_state=123).fit_transform(
        R["distmat_orth"]
    )
).T 
axes[0].scatter(
    x[:K], y[:K],
    marker="o",
    lw=0,
    s=20,
    color='k'
)
axes[0].scatter(
    x[K:], y[K:],
    marker="x",
    lw=1,
    s=20,
)
axes[0].axis("off")

x, y = PCA(2).fit_transform(
    MDS(20, dissimilarity='precomputed', random_state=123).fit_transform(
        R["distmat_perm"]
    )
).T 
axes[1].scatter(
    x[:K], y[:K],
    marker="o",
    lw=0,
    s=20,
    color='k'
)
axes[1].scatter(
    x[K:], y[K:],
    marker="x",
    lw=1,
    s=20,
)
axes[1].axis("off")

distmat = R["distmat_orth"]
orth_preds = []
perm_preds = []
for i in trange(2 * K):
    train_idx = np.array([j for j in range(K) if i != j])
    model.fit(
        R["distmat_orth"][train_idx][:, train_idx],
        R["fish_infos"][train_idx]
    )
    orth_preds.append(model.predict(R["distmat_orth"][i][train_idx][None, :]))
    model.fit(
        R["distmat_perm"][train_idx][:, train_idx],
        R["fish_infos"][train_idx]
    )
    perm_preds.append(model.predict(R["distmat_perm"][i][train_idx][None, :]))


model = KNeighborsRegressor(metric="precomputed", n_neighbors=3)

distmat = R["distmat_perm"][:K][:, :K]
preds, sqresids = [], []
for i in trange(K):
    train_idx = np.array([j for j in range(K) if i != j])
    model.fit(
        distmat[train_idx][:, train_idx],
        R["fish_infos"][train_idx]
    )
    pred = model.predict(distmat[i][train_idx][None, :])
    sqresids.append((pred - R["fish_infos"][i]) ** 2)
    preds.append(pred)

print(
    "Rsquared:",
    1 - np.sum(sqresids) / np.sum((R["fish_infos"][:K] - np.mean(R["fish_infos"][:K])) ** 2)
)

axes[2].scatter(orth_preds, R["fish_infos"], lw=0, s=10, color='k')
axes[2].scatter(perm_preds, R["fish_infos"], lw=1, s=10, marker='x')
axes[2].tick_params("both", labelsize=8, pad=1, length=1)

fig.subplots_adjust(top=.98, bottom=.1, left=.02, right=1, hspace=.6)
bbox = axes[0].get_position()
axes[0].set_position([bbox.xmin, bbox.ymin - .12, bbox.width, bbox.height])
bbox = axes[1].get_position()
axes[1].set_position([bbox.xmin, bbox.ymin - .075, bbox.width, bbox.height])
bbox = axes[2].get_position()
axes[2].set_position([bbox.xmin + .25, bbox.ymin, bbox.width - .3, bbox.height])
axes[2].set_xlabel("est. information\n(3 nearest neighbors)")
axes[2].set_ylabel("true information\n(fisher)", labelpad=0)

xl, yl = axes[2].get_xlim(), axes[2].get_ylim()
axes[2].plot(
    [0, 2 *np.max(preds)],
    [0, 2 *np.max(preds)],
    "-r", alpha=.75, lw=2
)
axes[2].set_xlim(xl)
axes[2].set_ylim(yl)

fig.savefig("G.pdf")

### COLORBARS ###
fig, ax = plt.subplots(1, 1, figsize=(0.8, .1))
ax.imshow(
    np.tile(np.linspace(0, 1, 100)[None, ::-1], (2, 1)),
    cmap=panel_b_scatter.cmap, aspect='auto'
)
ax.set_xticks([])
ax.set_yticks([])
fig.subplots_adjust(bottom=.1, top=.9, right=.95, left=.05)
fig.savefig("panel_b_colormap.pdf", transparent=True)

fig, ax = plt.subplots(1, 1, figsize=(0.8, .1))
ax.imshow(
    np.tile(np.linspace(0, 1, 100)[None, ::-1], (2, 1)),
    cmap=panel_c_scatter.cmap, aspect='auto'
)
ax.set_xticks([])
ax.set_yticks([])
fig.subplots_adjust(bottom=.1, top=.9, right=.95, left=.05)
fig.savefig("panel_c_colormap.pdf", transparent=True)

fig, ax = plt.subplots(1, 1, figsize=(0.8, .1))
ax.imshow(
    np.tile(np.linspace(0, 1, 100)[None, :], (2, 1)),
    cmap='viridis', aspect='auto'
)
ax.set_xticks([])
ax.set_yticks([])
fig.subplots_adjust(bottom=.1, top=.9, right=.95, left=.05)
fig.savefig("panel_a_colormap.pdf", transparent=True)




fig = plt.figure(figsize=(4, 2))
for j, i in enumerate(idx):
    plt.subplot(4, 1, j+1)
    [plt.plot(R["animals"][i][:,j],clip_on=False,lw=1) for j in range(0,100,20)]
    plt.ylim(-0.5, 1)
    plt.axis('off')

fig.savefig("TCs.pdf", transparent=True)


fig = plt.figure(figsize=(4, 2))
for j, i in enumerate(idx):
    plt.subplot(4, 1, j+1)
    [plt.plot(R["rotated_animals"][i][:,j],clip_on=False,lw=1) for j in range(0,100,20)]
    plt.ylim(-0.5, 1)
    plt.axis('off')

fig.savefig("TCs_rotated.pdf", transparent=True)

plt.show()

# THETAS = np.linspace(-np.pi, np.pi, 300)
# RS = RandomState(1234)
# N_NEURONS = 100
# N_ANIMALS = 80

# def tuning_curve(offset, sharpness, amplitude):
#     # Apply offset modulo 2 * pi
#     theta = (THETAS + offset + np.pi) % (2 * np.pi) - np.pi
#     # Compute unimodal cosine tuning curve
#     curve = np.cos(np.clip(
#         theta * sharpness, -np.pi, np.pi
#     ))
#     # Scale by amplitude
#     return amplitude * (curve + 1) * .5

# def neural_population(sharpness, concentration):
#     offsets = np.sort(
#         RS.vonmises(0.0, concentration, size=N_NEURONS)
#     )
#     return np.column_stack(
#         [tuning_curve(o, sharpness, 1.0) for o in offsets]
#     )


# sharpnesses = RS.uniform(2, 4, size=N_ANIMALS)
# concentrations = RS.uniform(0, 3, size=N_ANIMALS)

# Xs = [neural_population(s, c) for s, c in zip(sharpnesses, concentrations)]

# metric = LinearMetric(alpha=1.0)
# dists = pairwise_distances(metric, Xs)

# # Panel A
# x, y = PCA(2).fit_transform(MDS(20, dissimilarity='precomputed').fit_transform(dists)).T
# fig, axes = plt.subplots(2, 2)
# axes[0, 0].imshow(
#     Xs[np.argmax(sharpnesses)].T, interpolation='none', aspect='auto'
# )
# axes[0, 1].imshow(
#     Xs[np.argmin(sharpnesses)].T, interpolation='none', aspect='auto'
# )
# axes[1, 0].imshow(
#     Xs[np.argmax(concentrations)].T, interpolation='none', aspect='auto'
# )
# axes[1, 1].imshow(
#     Xs[np.argmin(concentrations)].T, interpolation='none', aspect='auto'
# )
# plt.show()
