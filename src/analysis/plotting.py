from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import numpy as np
from scipy import stats


def show_or_savefig(
    fig: Figure,
    show: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 150,
    **savefig_kwargs,
):
    """ I'll save your figure, %$&#, I'll even show it for you! """
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", **savefig_kwargs)
        print(f"Saved: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()
        
        
def loss_curve(loss_history: list[float], run_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("JEPA Training Loss")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    curve_path = run_dir / "loss_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print("   Loss curve saved to run directory")

# =============================================================================
# Displacement geometry
# =============================================================================

def displacement_hist_mag_v_label(
    delta: np.ndarray, labels: np.ndarray,
    show: bool = True, save_fn: Path = None
):
    delta_norm  = np.linalg.norm(delta, axis=1)
    colors = {0: "steelblue", 1: "tomato"}

    # -- histogram --
    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, col in colors.items():
        vals = delta_norm[labels == lbl]
        ax.hist(vals, bins=30, alpha=0.6, color=col, label=f"label={lbl}")
    ax.set_xlabel(r"||$\Delta$||  (L2 norm of displacement vector)")
    ax.set_ylabel("Count")
    ax.set_title("Displacement Magnitude by Label")
    ax.legend()
    fig.tight_layout()
    
    show_or_savefig(fig, show, save_fn)
    
    
def displacement_bp_mw(
    delta: np.ndarray, labels: np.ndarray,
    show: bool = True, save_fn: Path | None = None
):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    delta_norm  = np.linalg.norm(delta, axis=1)
    mw_stat, mw_p = stats.mannwhitneyu(x=delta_norm[labels == 0], y=delta_norm[labels == 1], 
                                       alternative="two-sided")
    colors = {0: "steelblue", 1: "tomato"}
    
    
    fig, ax = plt.subplots(figsize=(5, 4))
    grouped = [
        (delta_norm[labels == lbl], lbl) 
        for lbl in [0, 1] if (labels == lbl).any()
    ]

    bp = ax.boxplot([d for d, _ in grouped],
        label=[f"label={lbl}" for _, lbl in grouped],
        patch_artist=True,
    )
    for patch, (_, lbl) in zip(bp["boxes"], grouped):
        patch.set_facecolor(colors[lbl])
        patch.set_alpha(0.7)
    p_str = f"{mw_p:.3f}" if not np.isnan(mw_p) else "n/a"
    ax.set_ylabel(r"||$\Delta$||")
    ax.set_title(f"Displacement Magnitude by Label\n(Mann-Whitney p = {p_str})")
    fig.tight_layout()
    
    show_or_savefig(fig, show, save_fn)
    
    
def displacement_heatmap(
    delta: np.ndarray, labels: np.ndarray,
    show: bool = True, save_fn: Path | None = None
):
    sort_idx     = np.argsort(labels)
    delta_sorted = delta[sort_idx]
    n_neg        = int((labels[sort_idx] == 0).sum())
    vmax = float(np.percentile(np.abs(delta_sorted), 99)) or 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        delta_sorted.T, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax.axvline(n_neg - 0.5, color="black", linewidth=2, linestyle="--",
                label="label boundary")
    ax.set_xlabel("Sample (sorted by label)")
    ax.set_ylabel("Embedding dimension")
    ax.set_title(r"Displacement Field $\Delta$ | left=label 0, right=label 1  (red= +, blue= -)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    
    show_or_savefig(fig, show, save_fn)
        
        
def displacement_dim_profile(
    delta: np.ndarray, labels: np.ndarray,
    show: bool = True, save_fn: Path | None = None
):
    norms = np.linalg.norm(delta, axis=-1)
    safe_norms        = norms[:, np.newaxis].copy()
    safe_norms[safe_norms < 1e-10] = 1e-10
    delta_normed      = delta / safe_norms # unit vectors
    dim_mean          = delta_normed.mean(axis=0)
    dim_std           = delta_normed.std(axis=0)
    D                 = delta.shape[1]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(D)
    ax.bar(x, dim_mean, color="mediumseagreen", alpha=0.75, label="mean direction", width=1.0)
    ax.errorbar(x, dim_mean, yerr=dim_std, fmt="none", ecolor="black",
                elinewidth=1.0, capsize=2, alpha=0.5, label=r"$\pm$1 std")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Embedding dimension", fontsize=10)
    ax.set_xlim(-0.5, D-0.5)
    ax.set_xticks(np.arange(0, D, 2))
    ax.set_ylabel(r"Mean component of $\Delta$/||$\Delta$||")
    ax.set_title(r"Per-dimension displacement direction profile  (unit-normalised $\Delta$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    ax.grid(True, axis="y", alpha=0.4)
    
    show_or_savefig(fig, show, save_fn)
    
    
# =============================================================================
# PCA
# =============================================================================


# =============================================================================
# Plotting - Divergence
# =============================================================================

def plot_context_pred_scatter(
    regression_result: dict,
    context_sims:      np.ndarray,
    pred_dists:        np.ndarray,
    pair_indices:      tuple,
    out_dir:           Path,
    max_scatter:       int = 50000,
) -> None:
    """
    Context similarity vs. prediction distance scatter with OLS regression line.
    Divergent pairs highlighted in red.

    Saves context_pred_scatter.png to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    slope     = regression_result["slope"]
    intercept = regression_result["intercept"]
    r         = regression_result["r"]
    p         = regression_result["p"]
    x         = regression_result["ctx_sim_flat"]
    y         = regression_result["pred_dist_flat"]
    n_total   = len(x)

    row_idx, col_idx = pair_indices
    n_div = len(row_idx)

    # Background subsample (avoid overplotting)
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(n_total, size=min(max_scatter, n_total), replace=False)

    # Values for divergent pairs (index into raw matrices)
    div_x = context_sims[row_idx, col_idx] if n_div > 0 else np.array([])
    div_y = pred_dists[row_idx, col_idx]   if n_div > 0 else np.array([])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x[bg_idx], y[bg_idx],
               alpha=0.2, s=4, color="gray", label=f"all pairs (subsample n={len(bg_idx):,})")
    if n_div > 0:
        ax.scatter(div_x, div_y,
                   alpha=0.8, s=15, color="tomato", zorder=3,
                   label=f"divergent pairs (n={n_div:,})")
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    p_str  = f"{p:.2e}"
    ax.plot(x_line, slope * x_line + intercept, "b-", linewidth=2,
            label=f"OLS  r = {r:.3f},  p = {p_str}")
    ax.set_xlabel("Context cosine similarity")
    ax.set_ylabel("Prediction cosine distance")
    ax.set_title("Context Similarity vs. Prediction Divergence  (§5.3)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "context_pred_scatter.png", dpi=150)
    plt.close(fig)


def plot_divergence_pc_projections(
    div_pc_var:  np.ndarray,
    rand_pc_var: np.ndarray,
    out_dir:     Path,
) -> None:
    """
    Grouped bar chart: divergent vs. random pair projection variance per PC.

    Saves divergence_pc_projections.png to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    k     = len(div_pc_var)
    x_pos = np.arange(k)
    width = 0.35

    safe_div  = np.where(np.isnan(div_pc_var),  0, div_pc_var)
    safe_rand = np.where(np.isnan(rand_pc_var), 0, rand_pc_var)

    fig, ax = plt.subplots(figsize=(max(6, k * 0.9), 4))
    ax.bar(x_pos - width / 2, safe_div,  width,
           label="Divergent pairs", color="tomato",   alpha=0.8)
    ax.bar(x_pos + width / 2, safe_rand, width,
           label="Random pairs",   color="steelblue", alpha=0.8)
    ax.set_xlabel("PC index")
    ax.set_ylabel("Projection variance")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"PC{i+1}" for i in range(k)], rotation=45)
    ax.set_title("Divergence Vectors: Projection Variance per PC  (§5.3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "divergence_pc_projections.png", dpi=150)
    plt.close(fig)


def plot_variance_comparison(
    div_pc_var:  np.ndarray,
    rand_pc_var: np.ndarray,
    out_dir:     Path,
) -> None:
    """
    Variance ratio (divergent / random) per PC.

    Bars above 1.0 → divergent pairs concentrate more variance along that PC
    than random pairs — indicating genuine geometric structure.

    Saves variance_comparison.png to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    k     = len(div_pc_var)
    x_pos = np.arange(k)
    ratio = np.where(rand_pc_var > 1e-12,
                     div_pc_var / rand_pc_var,
                     np.nan)

    bar_colors = [
        "tomato"    if (not np.isnan(r) and r > 1.0) else "steelblue"
        for r in ratio
    ]

    fig, ax = plt.subplots(figsize=(max(6, k * 0.9), 4))
    ax.bar(x_pos, np.where(np.isnan(ratio), 0, ratio), color=bar_colors, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1,
               label="ratio = 1  (no enrichment vs. chance)")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Variance ratio  (divergent / random)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"PC{i+1}" for i in range(k)], rotation=45)
    ax.set_title("Divergent vs. Random Pair Variance Ratio per PC  (§5.3)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "variance_comparison.png", dpi=150)
    plt.close(fig)


# =============================================================================
# ICC Stability
# =============================================================================

def plot_icc_bar(
    icc_result: dict,
    out_dir:    Path,
) -> None:
    """
    ICC per PC bar chart, colour-coded by trait / state threshold.

    Red    → ICC > TRAIT_THRESHOLD  (0.8) — stable across encounter windows
    Blue   → ICC < STATE_THRESHOLD  (0.2) — varies with encounter context
    Silver → intermediate / ambiguous

    Saves icc_per_pc.png to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    icc_dict   = icc_result["icc_per_pc"]
    n_eligible = icc_result["eligible_patients"]
    trait_t    = icc_result["trait_threshold"]
    state_t    = icc_result["state_threshold"]
    method     = icc_result["method"]
    pc_labels  = list(icc_dict.keys())
    k          = len(pc_labels)
    icc_values = np.array([v if v is not None else np.nan
                            for v in icc_dict.values()])

    bar_colors = [
        "tomato"    if (not np.isnan(v) and v > trait_t) else
        "steelblue" if (not np.isnan(v) and v < state_t) else
        "silver"
        for v in icc_values
    ]

    fig, ax = plt.subplots(figsize=(max(7, k * 0.9), 4))
    ax.bar(np.arange(k), np.where(np.isnan(icc_values), 0, icc_values),
           color=bar_colors, alpha=0.85)
    ax.axhline(trait_t, color="tomato",    linestyle="--", linewidth=1.5,
               label=f"Trait  (ICC > {trait_t})")
    ax.axhline(state_t, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"State  (ICC < {state_t})")
    ax.axhline(0.0,     color="black",     linestyle="-",  linewidth=0.5)
    ax.set_xlabel("PC index")
    ax.set_ylabel(f"ICC  ({method})")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(pc_labels, rotation=45)
    ax.set_title(
        f"Intraclass Correlation per PC  |  {n_eligible} patients  "
        f"(§5.4 stability)"
    )
    ax.legend(fontsize=8)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "icc_per_pc.png", dpi=150)
    plt.close(fig)


def plot_spaghetti(
    pc_projections: np.ndarray,
    subject_ids:    np.ndarray,
    mask_positions: np.ndarray,
    out_dir:        Path,
    pc_idx:         int = 0,
    max_patients:   int = 10,
    min_samples:    int = 3,
) -> None:
    """
    PC score vs. mask_position for selected patients ("spaghetti plot").

    Each line is a patient; each point is one masked encounter window.
    High-ICC PCs → roughly flat lines.  Low-ICC PCs → variable lines.

    Parameters
    ----------
    pc_projections : (N, k) PC score matrix
    subject_ids    : (N,) patient IDs
    mask_positions : (N,) integer encounter indices
    out_dir        : output directory
    pc_idx         : which PC to show (0-indexed)
    max_patients   : patients to overlay (largest sample count wins)
    min_samples    : minimum windows per patient for inclusion

    Saves spaghetti_pc{pc_idx+1}.png to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_sids   = np.unique(subject_ids)
    sample_counts = {sid: int((subject_ids == sid).sum()) for sid in unique_sids}
    eligible      = [sid for sid in unique_sids if sample_counts[sid] >= min_samples]
    plot_sids     = sorted(eligible, key=lambda s: -sample_counts[s])[:max_patients]

    if not plot_sids:
        print(f"plot_spaghetti: no patients with >= {min_samples} samples.. figure skipped.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for i, sid in enumerate(plot_sids):
        m         = subject_ids == sid
        positions = mask_positions[m]
        pc_vals   = pc_projections[m, pc_idx]
        order     = np.argsort(positions)
        ax.plot(positions[order], pc_vals[order], "o-",
                color=cmap(i % 10), alpha=0.75, markersize=5,
                label=f"pid {sid}")

    ax.set_xlabel("Mask position (encounter index)")
    ax.set_ylabel(f"PC{pc_idx + 1} projection score")
    ax.set_title(
        f"PC{pc_idx + 1} Score Across Encounter Windows  "
        f"({len(plot_sids)} patients)  —  §5.4 stability"
    )
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / f"spaghetti_pc{pc_idx + 1}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)




