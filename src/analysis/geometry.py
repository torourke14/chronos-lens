"""
geometric latent-space analysis of the JEPA displacement field. Operates 
on the raw displacement vectors (no external metadata). 
Load z_context, z_pred, and labels, compute ||Delta||, PCA, divergence, and 
ICC analysis.

Plotting from these functions done in 'notebooks/*.ipynb'
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.analysis.calc import (
    fit_pca,
    marchenko_pastur_upper
)



SEED = 42
rng  = np.random.default_rng(SEED)

# =============================================================================
# PCA Decomp
# =============================================================================

def fit_pca_stats(delta: np.ndarray, top_k: int, n_samples: int, seed: int = SEED):
    """
    Compute stats on the PCA decomposition of the displacement field
    
    Marchenko-Pastur null upper bound: upper bound of the null hypothesis that 
    the eigenvalues of a covariance matrix are due to noise. The number of 
    signal eigenvalues is the number of eigenvalues above this bound.
    
    Uses n_components=D for PCA so the complete eigenvalue spectrum is available
    
    Parameters
    ----------
    delta : np.ndarray
        Displacement field Δ = z_pred - z_context
    top_k : int
        Maximum number of principal components to compute
    n_samples : int
        Number of samples
    seed : int
        Random seed

    References
    ----------
    [1] Marchenko, V. A., & Pastur, L. A. (1967). Distribution of
    eigenvalues for a multivariate normal distribution. Journal of
    Multivariate Analysis, 6(3), 407-412.
    """
    pca = fit_pca(delta, seed)
    
    projections  = pca.transform(delta)[:, :top_k].astype(np.float64)
    
    D = pca.n_components_
    eigenvalues = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    k = min(top_k, D)

    trace_cov = float(eigenvalues.sum())
    mp_upper  = marchenko_pastur_upper(n_samples, D, trace_cov)
    n_signal  = int((eigenvalues > mp_upper).sum())

    ev_sq_sum = float(np.sum(eigenvalues ** 2))
    d_eff     = (trace_cov ** 2) / ev_sq_sum if ev_sq_sum > 0 else 0

    thresh90 = int(np.searchsorted(cumvar, 0.90)) + 1
    thresh95 = int(np.searchsorted(cumvar, 0.95)) + 1
    
    stats = {
        "n_samples": n_samples,
        "n_components": D,
        "top_k": k,
        "n_signal_components": n_signal,
        "mp_upper_bound": float(mp_upper),
        "effective_dimensionality": float(d_eff),
        "components_for_90pct": thresh90,
        "components_for_95pct": thresh95,
        "top_k_explained_variance": float(evr[:k].sum()),
        "eigenvalues_all": eigenvalues.tolist(),
    }

    return pca, projections, stats

# =============================================================================
# UMAP
# =============================================================================


# =============================================================================
# Divergent pairs analysis
# =============================================================================

def find_divergent_pairs(
    context_sims: np.ndarray,
    pred_dists:   np.ndarray,
    eps:          float = 0.9,
    delta:        float = 0.5,
) -> tuple:
    """
    Upper-triangle pairs where context histories are cos_similar > eps
    but cosine_dist of predictions diverge by > delta.

    Parameters
    ----------
    context_sims : (N, N) cosine similarity matrix
    pred_dists   : (N, N) cosine distance matrix
    eps          : context similarity lower bound
    delta        : prediction distance lower bound

    Returns
    -------
    (row_idx, col_idx) : integer arrays indexing into the N samples
                         (upper triangle only;  row_idx < col_idx)
    """
    N  = context_sims.shape[0]
    iu = np.triu_indices(N, k=1)
    mask = (context_sims[iu] > eps) & (pred_dists[iu] > delta)
    return iu[0][mask], iu[1][mask]


def regress_divergence(
    context_sims: np.ndarray,
    pred_dists:   np.ndarray,
) -> dict:
    """
    OLS regression of prediction cosine distance on context cosine similarity
    over all upper-triangle pairs.

    The continuous version of divergence analysis: instead of hard
    thresholds, study residuals from the regression as the "unexplained"
    divergence signal.

    Parameters
    ----------
    context_sims : (N, N) cosine similarity matrix
    pred_dists   : (N, N) cosine distance matrix

    Returns
    -------
    dict with keys:
      slope, intercept, r, p, stderr — OLS statistics
      residuals                       — (n_pairs,) signed residuals (y − ŷ)
      ctx_sim_flat                    — (n_pairs,) upper-triangle context sims
      pred_dist_flat                  — (n_pairs,) upper-triangle pred dists
      n_pairs                         — number of unique pairs evaluated
    """
    N  = context_sims.shape[0]
    iu = np.triu_indices(N, k=1)
    x  = context_sims[iu]
    y  = pred_dists[iu]

    lr        = stats.linregress(x, y)
    slope     = float(lr.slope) # type: ignore[arg-type]
    intercept = float(lr.intercept) # type: ignore[arg-type]
    residuals = y - (slope * x + intercept)

    return {
        "slope":          slope,
        "intercept":      intercept,
        "r":              float(lr.rvalue),  # type: ignore[arg-type]
        "p":              float(lr.pvalue),  # type: ignore[arg-type]
        "stderr":         float(lr.stderr),  # type: ignore[arg-type]
        "residuals":      residuals,
        "ctx_sim_flat":   x,
        "pred_dist_flat": y,
        "n_pairs":        int(len(x)),
    }
    

def project_pair_divergence_vectors(
    z_pred:         np.ndarray,
    pair_indices:   tuple,
    pca_components: np.ndarray,
) -> np.ndarray:
    """
    - Compute divergence vectors  d_ij = z_pred_i - z_pred_j
    - project onto top-k PCA axes from the geometry step.

    Parameters
    ----------
    z_pred         : (N, D) predicted embeddings
    pair_indices   : (row_idx, col_idx) from find_divergent_pairs
    pca_components : (k, D) from pca.components_[:k]

    Returns
    -------
    projections : (n_pairs, k)  projection of each divergence vector onto PCs
                  Returns shape (0, k) when pair_indices is empty.
    """
    row_idx, col_idx = pair_indices
    k = pca_components.shape[0]
    if len(row_idx) == 0:
        return np.empty((0, k), dtype=np.float64)
    div_vecs = z_pred[row_idx] - z_pred[col_idx]    # (n_pairs, D)
    return div_vecs @ pca_components.T              # (n_pairs, k)


def divergence_variance_comparison(
    z_pred:         np.ndarray,
    pair_indices:   tuple,
    pca_components: np.ndarray,
) -> tuple:
    """
    Per-PC projection variance of divergent pairs vs. a matched random baseline.

    If divergent-pair vectors are concentrated along a PC axis (high variance
    ratio), that axis encodes the divergence structure — not noise.

    Parameters
    ----------
    z_pred          : (N, D) predicted embeddings
    pair_indices    : (row_idx, col_idx) divergent pair indices
    pca_components  : (k, D) PCA components

    Returns
    -------
    (div_pc_var, rand_pc_var) : each (k,) float arrays
                                np.nan entries when pair_indices is empty.
    """
    row_idx, col_idx = pair_indices
    k     = pca_components.shape[0]
    n_div = len(row_idx)

    if n_div == 0:
        return np.full(k, np.nan), np.full(k, np.nan)

    # Divergent pair projections
    div_proj   = project_pair_divergence_vectors(z_pred, pair_indices, pca_components)
    div_pc_var = div_proj.var(axis=0)

    # Random baseline — same count, drawn uniformly from all upper-triangle pairs
    N           = z_pred.shape[0]
    n_total     = N * (N - 1) // 2
    rand_linear = rng.choice(n_total, size=n_div, replace=(n_div > n_total))
    iu          = np.triu_indices(N, k=1)
    rand_proj   = project_pair_divergence_vectors(
        z_pred, (iu[0][rand_linear], iu[1][rand_linear]), pca_components
    )
    rand_pc_var = rand_proj.var(axis=0)

    return div_pc_var, rand_pc_var

# =============================================================================
# ICC Analysis
# =============================================================================