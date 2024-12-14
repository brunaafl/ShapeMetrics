import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant
from sklearn.decomposition import PCA
import matplotlib.colors as col
import seaborn as sns

n = 500
theta = np.clip(np.linspace(-2*np.pi, 2*np.pi, 500), -np.pi, np.pi)
M = np.roll(circulant((1 + np.cos(theta)) / 2), n // 2, axis=0)

# Tuning curve panel.
fig, axes = plt.subplots(5, 1, figsize=(1.2, 2))
for i, ax in zip(np.arange(40, n, n // 5), axes):
    ax.plot(M[i], color=sns.color_palette('husl', n)[i], lw=2, zorder=np.random.choice(1000), alpha=.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, .5)
    ax.set_yticks([0, 0.5])
    ax.set_yticklabels([])
    ax.set_xticks([])

for ax in axes[:-1]:
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0, color='k', dashes=[2, 2])

axes[-1].set_xticks([0, n//2, n])
axes[-1].set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])
axes[-1].tick_params(labelsize=8)
# axes[-1].set_xlabel(r"$\theta$", fontsize=12)
axes[-1].spines["bottom"].set_bounds(0, n)
fig.tight_layout()
fig.subplots_adjust(hspace=0)
fig.savefig("panel_D1.pdf", transparent=True)


# Matrix heatmap panel.
fig, ax = plt.subplots(1, 1, figsize=(1, 1.3))
ax.imshow(M, aspect='auto')
ax.axis('off')
ax.set_title(r'$\mathbf{X}_k$')
# fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
fig.tight_layout()
fig.savefig("panel_D2.pdf", transparent=True)

# 3D pca panel.
fig = plt.figure(figsize=(1.2, 1.2))
ax = fig.add_subplot(projection='3d')
x, y, z = PCA(3).fit_transform(M).T
# ax.plot(x, y, zs=z, lw=3, color='k')
ax.scatter(
    x[::10], y[::10], zs=z[::10],
    lw=0, s=30, alpha=1, c=np.linspace(0, 1, n//10),
    cmap=col.ListedColormap(sns.color_palette('husl', n//10))
)
zlim = ax.get_zlim()
ax.plot(x, y, zs=zlim[0]-1, lw=3, color='k', alpha=.3)
ax.axis('off')
fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
fig.savefig("panel_D3.pdf", transparent=True)

# Done.
plt.show()
