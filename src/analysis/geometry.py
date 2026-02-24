"""
Foundation module for geometric latent-space analysis of the JEPA
displacement field  Δ = z_pred − z_context  (thesis §5.1-5.2).

Functions are split into *computation* (pure, no IO) and *plotting*
(save figures to disk) so downstream steps can reuse fitted objects
without re-running analysis.

Downstream consumers
--------------------
  projections.npy      → Step 3 (Divergence), Step 4 (ICC), Step 5 (LASSO)
  pca.pkl              → Step 3 (Divergence)
  umap_embedding.npy   → Step 5 Tier B (HDBSCAN cluster enrichment)
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import umap as umap_module


SEED = 42
rng  = np.random.default_rng(SEED)


# =============================================================================
# Utilities
# =============================================================================

def displacement_stats(delta: np.ndarray, labels: np.ndarray) -> dict:
    norms  = np.linalg.norm(delta, axis=1)
    mask0  = labels == 0
    mask1  = labels == 1
    norms0 = norms[mask0]
    norms1 = norms[mask1]

    if len(norms0) >= 1 and len(norms1) >= 1:
        mw_stat, mw_p = stats.mannwhitneyu(norms0, norms1, alternative="two-sided")
        mw_stat, mw_p = float(mw_stat), float(mw_p)
    else:
        mw_stat, mw_p = np.nan, np.nan

    return {
        "n_samples":        len(norms),
        "n_label0":         mask0.sum(),
        "n_label1":         mask1.sum(),
        "norm_mean":        float(norms.mean()),
        "norm_mean_label0": float(norms0.mean()),
        "norm_mean_label1": float(norms1.mean()),
        "norm_std":         float(norms.std()),
        "norm_min":         float(norms.min()),
        "norm_median":      float(np.median(norms)),
        "norm_max":         float(norms.max()),
        "mannwhitney_stat": mw_stat,
        "mannwhitney_p":    mw_p,
    }


# =============================================================================
# PCA Decomp
# =============================================================================

def fit_pca(delta: np.ndarray, seed: int = SEED) -> PCA:
    _, D = delta.shape
    pca  = PCA(n_components=D, random_state=seed, svd_solver="full")
    pca.fit(delta)
    return pca


def marchenko_pastur_upper(n: int, p: int, trace: float) -> float:
    sigma_sq = trace / p
    gamma    = p / n
    return sigma_sq * (1.0 + np.sqrt(gamma)) ** 2


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

def fit_umap_2d(
    delta:       np.ndarray,
    n_neighbors: int = 15,
    metric:      str = "cosine",
    seed:        int = SEED,
) -> np.ndarray:
    """
    Fit UMAP on the displacement field.

    "Cosine metric (default) is consistent with the pairwise distance
    measure used throughout the divergence and clustering analyses
    and avoids the curse of dimensionality"

    Parameters
    ----------
    delta       : (N, D) displacement vectors
    n_neighbors : UMAP locality parameter (automatically clamped to N-1)
    metric      : distance metric — 'cosine' recommended; 'euclidean' also valid
    seed        : random state for reproducibility

    Returns
    -------
    umap_embedding : (N, 2) UMAP embedding
    """

    N = delta.shape[0]
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, N - 1),
        metric=metric,
        random_state=seed,
    )
    return np.asarray(reducer.fit_transform(delta), dtype=np.float64)


def plot_umap(
    embedding_2d: np.ndarray,
    projections:  np.ndarray,
    labels:       np.ndarray,
    delta:        np.ndarray,
    out_dir:      Path,
) -> None:
    """
    Two UMAP figures saved to out_dir.

    umap_vs_pca.png — UMAP vs PCA side-by-side coloured by label
                      (linearity-assumption validation, thesis §5.2)
    umap_norm.png   — UMAP coloured by ||Δ||
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    norms = np.linalg.norm(delta, axis=1)

    # -- side-by-side linearity check --
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for lbl, col in [(0, "steelblue"), (1, "tomato")]:
        mask = labels == lbl
        if mask.any():
            axes[0].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                            c=col, label=f"label={lbl}", alpha=0.65, s=20)
            axes[1].scatter(projections[mask, 0], projections[mask, 1],
                            c=col, label=f"label={lbl}", alpha=0.65, s=20)
    axes[0].set_title("UMAP  (cosine metric on Δ)")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].legend(fontsize=8)
    axes[1].set_title("PCA  (same samples)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=8)
    fig.suptitle("UMAP vs PCA on Displacement Field Δ  |  linearity check (§5.2)")
    fig.tight_layout()
    fig.savefig(out_dir / "umap_vs_pca.png", dpi=150)
    plt.close(fig)

    # -- coloured by displacement magnitude --
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                    c=norms, cmap="viridis", alpha=0.7, s=20)
    plt.colorbar(sc, ax=ax, label="||Δ||")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP  (coloured by ||Δ|| — displacement magnitude)")
    fig.tight_layout()
    fig.savefig(out_dir / "umap_norm.png", dpi=150)
    plt.close(fig)