"""
Utility functions operating on the displacement vectors, embeddings, etc.

Only matrix calculations and functions calling external libraries 
imported here.
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import umap as umap_module
import pingouin as pg


SEED = 42
rng  = np.random.default_rng(SEED)


def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """(N, N) pairwise cosine similarity.  Zero-norm rows -> zero similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    Xn    = X / norms
    return Xn @ Xn.T


def _cosine_dist_matrix(X: np.ndarray) -> np.ndarray:
    """(N, N) pairwise cosine distance  (= 1 − cosine_similarity)."""
    return 1.0 - _cosine_sim_matrix(X)


def marchenko_pastur_upper(n: int, p: int, trace: float) -> float:
    sigma_sq = trace / p
    gamma    = p / n
    return sigma_sq * (1.0 + np.sqrt(gamma)) ** 2


def fit_pca(delta: np.ndarray, seed: int = SEED) -> PCA:
    _, D = delta.shape
    pca  = PCA(n_components=D, random_state=seed, svd_solver="full")
    pca.fit(delta)
    return pca


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


# =============================================================================
# ICC Stability Analysis
# =============================================================================

def compute_icc(
    pc_projections:  np.ndarray,
    subject_ids:     np.ndarray,
    top_k:           int,
    min_samples:     int = 3,
    trait_threshold: float = 0.8,
    state_threshold: float = 0.2
) -> dict:
    """
    Intraclass correlation coefficient per top-k PC axis across encounter
    windows within each patient.

    Interpretation
    --------------
    ICC > trait_threshold (0.8) -> axis tracks stable patient-level trait
    ICC < state_threshold (0.2) -> axis reflects dynamic encounter state

    Parameters
    ----------
    pc_projections : (N, >= top_k) PC score matrix from the geometry step
    subject_ids    : (N,) patient identifier per sample
    top_k          : number of PC axes to evaluate (clamped to shape[1])
    min_samples    : patients with fewer encounter samples are excluded

    Returns
    -------
    dict with keys:
      icc_per_pc       : {"PC1": float|None, ...}
      trait_pcs        : list of PC labels with ICC > trait_threshold
      state_pcs        : list of PC labels with ICC < state_threshold
      eligible_patients: count of qualifying patients
      trait_threshold  : TRAIT_THRESHOLD
      state_threshold  : STATE_THRESHOLD
    """
    def _nan_to_none(v):
        return None if (isinstance(v, float) and np.isnan(v)) else v
    
    # ---------------------------------------------------------------- #
    k = min(top_k, pc_projections.shape[1])

    # Group encounter-level PC scores by patient
    unique_sids   = np.unique(subject_ids)
    groups_per_pc = [[] for _ in range(k)]
    eligible_pids = []

    for sid in unique_sids:
        m = subject_ids == sid
        if int(m.sum()) < min_samples:
            continue
        eligible_pids.append(sid)
        for pc_idx in range(k):
            groups_per_pc[pc_idx].append(pc_projections[m, pc_idx])

    n_eligible = len(eligible_pids)
    icc_values = np.full(k, np.nan)

    icc_list = []
    for pc_idx in range(k):
        rows = []
        for pid in eligible_pids:
            m = subject_ids == pid
            for val in pc_projections[m, pc_idx]:
                rows.append({"subject": pid, "score": float(val)})
        if len(rows) < 4:
            icc_list.append(np.nan)
            continue
        df = pd.DataFrame(rows)
        df["rater"] = df.groupby("subject").cumcount()
        try:
            result = pg.intraclass_corr(
                data=df, targets="subject", raters="rater", ratings="score",
                # nan_policy="omit"
            )
            row_31 = result[result["Type"] == "ICC3,1"]
            icc_list.append(
                float(row_31["ICC"].values[0]) if len(row_31) else np.nan
            )
        except Exception:
            icc_list.append(np.nan)

    icc_values = np.array(icc_list)

    trait_pcs = [f"PC{i+1}" for i, v in enumerate(icc_values)
                 if not np.isnan(v) and v > trait_threshold]
    state_pcs = [f"PC{i+1}" for i, v in enumerate(icc_values)
                 if not np.isnan(v) and v < state_threshold]

    return {
        "icc_per_pc":        {f"PC{i+1}": _nan_to_none(float(v))
                               for i, v in enumerate(icc_values)},
        "trait_pcs":         trait_pcs,
        "state_pcs":         state_pcs,
        "eligible_patients": n_eligible,
        "trait_threshold":   float(trait_threshold),
        "state_threshold":   float(state_threshold),
    }