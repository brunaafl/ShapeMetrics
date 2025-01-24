import itertools
from tqdm import tqdm
import numpy as np
from netrep.metrics import LinearMetric, PermutationMetric
from netrep.multiset import pairwise_distances
from numpy.random import RandomState
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

RS = RandomState(123)
N_THETAS = 300
N_NEURONS = 100
N_ANIMALS = 40
THETAS = np.linspace(-np.pi, np.pi, N_THETAS + 1)[:-1]

def randomly_rotate(X):
    """
    Return a randomly reflected + rotated version of X.
    """
    mu = np.mean(X, axis=0, keepdims=True)
    n = X.shape[1]
    Q = np.linalg.qr(RS.randn(n, n))[0]
    return ((X - mu) @ Q) + mu


def make_tuning_curve(offset, sharpness):
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
    curve = np.cos(np.clip(
        THETAS * sharpness, -np.pi, np.pi
    ))
    return curve

import matplotlib.pylab as plt

k

def neural_population(sharpness, concentration):
    offsets = np.sort(
        RS.vonmises(0.0, concentration, size=N_NEURONS)
    )
    return np.column_stack(
        [make_tuning_curve(o, sharpness) for o in offsets]
    )

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

def compute_mean_fish_info(tuning_curves, noise_lev=0.1):
    infos = []
    for i in range(len(THETAS)):
        j = (i + 1) % len(THETAS)
        infos.append(fish_info(
            tuning_curves[i],
            tuning_curves[j],
            noise_lev * np.eye(N_NEURONS),
            noise_lev * np.eye(N_NEURONS)
        ))
    return np.mean(infos)

# Generate data from N_ANIMALS
print("Generating tuning curves...")
sharpnesses = RS.uniform(
    2.0, 4.0, size=N_ANIMALS
)
concentrations = RS.uniform(
    0.0, 3.0, size=N_ANIMALS
)
animals, rotated_animals = [], []
for sharp, conc in zip(sharpnesses, concentrations):
    animals.append(
        neural_population(sharp, conc)
    )
    rotated_animals.append(
        # randomly_rotate(neural_population(sharp, conc))
        randomly_rotate(animals[-1])
    )

# Fit pairwise distances
print("Computing distances...")
distmat_orth = pairwise_distances(
    LinearMetric(alpha=1.0, center_columns=True),
    animals + rotated_animals
)
distmat_perm = pairwise_distances(
    PermutationMetric(center_columns=True),
    animals + rotated_animals
)

# Sanity check on distances.
assert np.allclose(0.0, np.diag(distmat_orth, N_ANIMALS), atol=1e-6)

# Compute fisher information for each animal.
fish_infos = []
for X in (animals + rotated_animals):
    fish_infos.append(compute_mean_fish_info(X))

# Sanity test.
print("Running fisher info sanity test")
for X, Y in zip(animals, rotated_animals):
    f1 = compute_mean_fish_info(X)
    f2 = compute_mean_fish_info(Y)
    assert abs(f1 - f2) < 1e-6
print("Passed sanity test...")


# Save result.
np.savez(
    "simulated_data.npz",
    distmat_orth=distmat_orth,
    distmat_perm=distmat_perm,
    sharpnesses=sharpnesses,
    concentrations=concentrations,
    fish_infos=fish_infos,
    animals=np.stack(animals),
    rotated_animals=np.stack(rotated_animals),
    N_ANIMALS=N_ANIMALS,
    N_NEURONS=N_NEURONS,
    N_THETAS=N_THETAS,
)

