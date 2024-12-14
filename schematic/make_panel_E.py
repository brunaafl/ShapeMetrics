import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as col
import seaborn as sns

FIGSIZE = (1.2, 1.2)
rs = np.random.RandomState(111)
n = 500
theta = np.clip(np.linspace(-2*np.pi, 2*np.pi, 500), -np.pi, np.pi)

X = np.column_stack(
    [np.roll(1 + np.cos(theta), rs.choice(n)) for _ in range(7)]
)
Y = np.column_stack(
    [np.roll(1 + np.cos(theta), rs.choice(n)) for _ in range(7)]
)
X -= np.mean(X, axis=0)
Y -= np.mean(Y, axis=0)

U, _, Vt = np.linalg.svd(X.T @ Y)
Q = U @ Vt

# 3D pca panel.
for i, M in enumerate([X, Y]):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(projection='3d')
    x, y, z = PCA(3).fit_transform(M).T
    # ax.plot(x, y, zs=z, lw=3, color='k')
    ax.scatter(
        x[::10], y[::10], zs=z[::10],
        lw=0, s=20, alpha=1, c=np.linspace(0, 1, n//10),
        cmap=col.ListedColormap(sns.color_palette('husl', n//10))
    )
    zlim = ax.get_zlim()
    ax.plot(x, y, zs=zlim[0]-1, lw=3, color='k', alpha=.3)
    ax.axis('off')
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    fig.savefig(f"panel_E{i + 1}.pdf", transparent=True)

# Aligned panel
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(projection='3d')
_x, _y, _z = PCA(3).fit_transform(np.row_stack((X @ Q, Y))).T
x1, x2 = np.array_split(_x, 2)
y1, y2 = np.array_split(_y, 2)
z1, z2 = np.array_split(_z, 2)
# ax.plot(x1, y1, zs=z1, lw=1, color='k')
ax.scatter(
    x1[::10], y1[::10], zs=z1[::10],
    lw=0, s=20, alpha=.75, c=np.linspace(0, 1, n//10),
    cmap=col.ListedColormap(sns.color_palette('husl', n//10))
)
# ax.plot(x2, y2, zs=z2, lw=1, color='k')
ax.scatter(
    x2[::10], y2[::10], zs=z2[::10],
    lw=0, s=10, alpha=.75, c=np.linspace(0, 1, n//10),
    cmap=col.ListedColormap(sns.color_palette('husl', n//10))
)
for xx, yy, zz in zip(
        np.column_stack((x1, x2))[::10],
        np.column_stack((y1, y2))[::10],
        np.column_stack((z1, z2))[::10],
    ):
    ax.plot(
        xx, yy, zs=zz,
        lw=1, color='k',
        alpha=.7, #zorder=1000
    )
zlim = ax.get_zlim()
ax.plot(x1, y1, zs=zlim[0]-1, lw=3, color='k', alpha=.3)
ax.plot(x2, y2, zs=zlim[0]-1, lw=3, color='k', alpha=.3)
ax.axis('off')
fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
fig.savefig(f"panel_E3.pdf", transparent=True)

# Done.
plt.show()
