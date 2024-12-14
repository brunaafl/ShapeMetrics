import itertools
from tqdm import tqdm
import numpy as np
from netrep.metrics import GaussianStochasticMetric
from netrep.multiset import pairwise_distances
from numpy.random import RandomState
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

RS = RandomState(1234)
N_THETAS = 300
N_NEURONS = 100
N_ANIMALS = 80
THETAS = np.linspace(-np.pi, np.pi, N_THETAS + 1)[:-1]

def randomly_rotate(X):
    """
    Return a randomly reflected + rotated version of X.
    """
    Q = np.linalg.qr(RS.randn(X.shape[1]))[0]
    return X @ Q


def tuning_curve(offset, sharpness):
    """
    Create a single neuron tuning curve.

    Parameters
    ----------
    offset : float
        Peak value of tuning curve between [-pi, pi).

    sharpness : float
        Larger values create sharper tuning curves.

    Returns
    -------
    curve : ndarray
        Vector, same length as THETAS, holding neural
        firing rate as a function of the stimulus.

    spatial_deriv : ndarray
        Vector, same length as THETAS, holding the
        df / dtheta, where f is the firing rate and
        theta is the stimlus.
    """
    # Apply offset modulo 2 * pi
    theta = (THETAS + offset + np.pi) % (2 * np.pi) - np.pi
    # Compute unimodal cosine tuning curve
    curve = 0.5 + 0.5 * np.cos(np.clip(
        theta * sharpness, -np.pi, np.pi
    ))
    # Compute spatial derivative of the tuning curve
    spatial_deriv = -sharpness * .5 * np.sin(np.clip(
        theta * sharpness, -np.pi, np.pi
    ))
    return curve, spatial_deriv

def neural_population(sharpness, cov_scale, lam, offsets="random"):
    """
    Create population of noisy tuning curves.

    Parameters
    ----------
    sharpeness : float
        Larger values create sharper tuning curves.

    cov_scale : float
        Trace of the covariance matrix at each point

    lam : float
        Scalar greater than negative one, controlling
        the noise correlation structure

    Returns
    -------
    means : ndarray
        (N_THETAS x N_NEURONS) array of mean responses,
        i.e. tuning curves for each neuron.

    covs : ndarray
        (N_THETAS x N_NEURONS x N_NEURONS) array of
        noise covariance matrices for each condition.
    """
    I = np.eye(N_NEURONS)
    if offsets == "random":
        offsets = RS.uniform(-np.pi, np.pi, N_NEURONS)
    elif offsets == "uniform":
        offsets = np.linspace(-np.pi, np.pi, N_NEURONS + 1)[:-1]
    means, vs = zip(*[tuning_curve(o, sharpness) for o in offsets])
    vs = np.column_stack(vs)
    vs /= np.linalg.norm(vs, axis=1, keepdims=True)
    covs = [I + lam * np.outer(v, v) for v in vs]
    covs = [S * (cov_scale / np.trace(S)) for S in covs]
    return np.column_stack(means), np.stack(covs, axis=0)

def fish_info(m1, m2, c1, c2):
    """
    Compute approximate linear Fisher Information.

    Parameters
    ----------
    m1 : ndarray
        N_NEURONS vector specifying mean response
        in condition 1.
    m2 : ndarray
        N_NEURONS vector specifying mean response
        in condition 2.
    c1 : ndarray
        N_NEURONS x N_NEURONS matrix specifying
        noise covariance in condition 1.
    c2 : ndarray
        N_NEURONS x N_NEURONS matrix specifying
        noise covariance in condition 2.

    Returns
    -------
    fisher_information : float
        The Fisher information estimated by
        averaging the covariances and a
        discrete approximation of the derivative
        with respect to the stimulus.
    """
    cov = (c1 + c2) / .5
    df = (m1 - m2) / (THETAS[1] - THETAS[0])
    return df.T @ np.linalg.solve(cov, df)

def fish_info_sanity_test():
    means1, covs1 = neural_population(1, N_NEURONS, -0.9, offsets="uniform")
    means2, covs2 = neural_population(1, N_NEURONS, 0.0, offsets="uniform")
    means3, covs3 = neural_population(1, N_NEURONS, 0.9, offsets="uniform")

    fish1, fish2, fish3 = [], [], []
    for i in range(len(THETAS) - 1):
        fish1.append(
            fish_info(means1[i], means1[i + 1], covs1[i], covs1[i + 1])
        )
        fish2.append(
            fish_info(means2[i], means2[i + 1], covs2[i], covs2[i + 1])
        )
        fish3.append(
            fish_info(means3[i], means3[i + 1], covs3[i], covs3[i + 1])
        )

    assert np.allclose(fish1, np.mean(fish1), rtol=1e-3)
    assert np.allclose(fish2, np.mean(fish2), rtol=1e-3)
    assert np.allclose(fish3, np.mean(fish3), rtol=1e-3)
    assert np.mean(fish1) > np.mean(fish2)
    assert np.mean(fish2) > np.mean(fish3)
    print("Completed sanity test.")

def compute_mean_fish_info(means, covs):
    infos = []
    for i in range(len(THETAS)):
        j = (i + 1) % len(THETAS)
        infos.append(fish_info(means[i], means[j], covs[i], covs[j]))
    return np.mean(infos)

# Run a quick test that the fisher information is behaving as intended
fish_info_sanity_test()

# Sample many animals
sharpnesses = RS.uniform(
    2.0, 4.0, size=N_ANIMALS
)
cov_scales = N_NEURONS * np.exp(
    RS.uniform(0.8, 1.2, size=N_ANIMALS)
)
lams = RS.uniform(
    -0.95, 1.0, size=N_ANIMALS
)
animals = []
for sharp, scale, lam in zip(sharpnesses, cov_scales, lams):
    animals.append(
        neural_population(sharp, scale, lam)
    )

# Create rotated version of each animal
rotated_animals = [randomly_rotate(X) for X in animals]
all_animals = animals + rotated_animals

# Fit pairwise distances
print("Computing distances...")
metric = GaussianStochasticMetric(1.0, niter=2)
distmat = np.zeros((2 * N_ANIMALS, 2 * N_ANIMALS))
pbar = tqdm(total=((2 * N_ANIMALS * (2 * N_ANIMALS - 1)) // 2))
for i, j in itertools.combinations(range(2 * N_ANIMALS), 2):
    metric.fit(all_animals[i], all_animals[j])
    distmat[i, j] = metric.score(all_animals[i], all_animals[j])
    distmat[j, i] = distmat[i, j]
    pbar.update(1)
pbar.close()

# Compute fisher information for each animal.
fish_infos = []
for means, covs in animals:
    fish_infos.append(compute_mean_fish_info(means, covs))

# Save result.
np.savez(
    "simulated_data.npz",
    distmat=distmat,
    sharpnesses=sharpnesses,
    cov_scales=cov_scales,
    lams=lams,
    fish_infos=fish_infos,
    animals=np.stack(animals),
    rotated_animals=np.stack(rotated_animals)
)

