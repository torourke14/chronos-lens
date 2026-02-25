
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.models.sequential_jepa import JEPA



rng = np.random.default_rng(42)



@torch.no_grad()
def calc_embedding_vecs(
    model: JEPA,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """ Calculate context, predicted, and target embeddings for all masked tokens in the dataset.
        Returns:
            z_context: (N, D) array of context embeddings
            z_pred:    (N, D) array of predicted embeddings
            z_target:  (N, D) array of target embeddings
            delta:     (N, D) array of displacement vectors (pred - ctx)
            subject_ids: (N,) array of subject IDs
            mask_positions: (N,) array of masked positions in sequence
            labels: (N,) array of true tokens at masked positions
    """
    model.eval()
    all_z_ctx:  list[np.ndarray] = []
    all_z_pred: list[np.ndarray] = []
    all_z_tgt:  list[np.ndarray] = []
    all_sids:   list[str]        = []
    all_mpos:   list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        z_ctx, z_pred, z_tgt = model(batch_dev)

        all_z_ctx.append(z_ctx.cpu().numpy())
        all_z_pred.append(z_pred.cpu().numpy())
        all_z_tgt.append(z_tgt.cpu().numpy())
        all_sids.extend(batch["subject_ids"])
        all_mpos.append(batch["mask_pos"].numpy())
        all_labels.append(batch["labels"].numpy())

    z_context     = np.concatenate(all_z_ctx,  axis=0)
    z_pred_arr    = np.concatenate(all_z_pred, axis=0)
    z_target      = np.concatenate(all_z_tgt,  axis=0)
    delta         = z_pred_arr - z_context
    subject_ids   = np.array(all_sids)
    mask_positions = np.concatenate(all_mpos,  axis=0)
    labels        = np.concatenate(all_labels, axis=0)

    return z_context, z_pred_arr, z_target, delta, subject_ids, mask_positions, labels


def run_disp_field_checks(
    z_context: np.ndarray,
    z_pred:    np.ndarray,
    z_target:  np.ndarray,
    delta:     np.ndarray,
    frac_nontrivial_thresh: float = 0.5
) -> bool:
    """ Run useful checks on extracted embeddings to ensure
        predictor hasn't collapsed, delta remains statistically significant, etc
    """
    checks: Dict[str, tuple[bool, Any]] = {}

    # Check z_pred has non-trivial variance (std > 0.01 per dimension)
    pred_std        = z_pred.std(axis=0)
    frac_nontrivial = float((pred_std > 0.01).mean())
    z_pred_ok = frac_nontrivial > frac_nontrivial_thresh
    
    checks['nontrivial_variance'] = (z_pred_ok, (
        f"z_pred min_std={pred_std.min():.4f}, "
        f"z_pred frac_dims(>0.01)={frac_nontrivial:.1%}"
    ))

    # Check displacement field ||Delta|| hasn't collapsed
    delta_norms = np.linalg.norm(delta, axis=-1)
    delt_norm_ok = delta_norms.std() > 1e-4
    checks['delta_norms'] = (delt_norm_ok, (
        f"mean={delta_norms.mean():.4f}, "
        f"std={delta_norms.std():.4f}, "
        f"min={delta_norms.min():.4f}, "
        f"max={delta_norms.max():.4f}"
    ))

    # Check the predictor beats a random baseline scaled to z_target's distribution
    rand_pred = rng.standard_normal(z_pred.shape).astype(np.float32) * z_target.std() + z_target.mean()
    pred_dist = float(np.linalg.norm(z_pred    - z_target, axis=-1).mean())
    rand_dist = float(np.linalg.norm(rand_pred - z_target, axis=-1).mean()) 
    checks['predictor_v_random'] = (pred_dist < rand_dist, (
        f"||z_pred-z_tgt||={pred_dist:.4f} vs "
        f"||rand-z_tgt||={rand_dist:.4f}"
    ))

    for k, (ok, msg) in checks.items():
        if not ok:
            print(f"="*10 + "EMBEDDING STATE FAILURE" + "="*10)
            for k, (ok, msg) in checks.items():
                print(f"{k:20s} : {'PASS' if ok else 'FAIL'}  ({msg})")
            return False
    return True


def extract_embedding_vecs(
    model: JEPA,
    loader: DataLoader,
    device: torch.device,
    out_fn: Path
):
    """ Fetch context, predicted, and target embeddings for all masked tokens,
        and save to a .npz file for later analysis.
    """
    model.eval()

    z_context, z_pred, z_target, delta, subject_ids, mask_positions, labels = \
        calc_embedding_vecs(model, loader, device)

    # don't save if predictor collapsed
    all_passed = run_disp_field_checks(z_context, z_pred, z_target, delta)
    if not all_passed:
        print("Embeddings NOT saved: one or more sanity checks failed.")
        return

    np.savez(out_fn,
             z_context = z_context, z_pred = z_pred, z_target = z_target,
             delta = delta,
             subject_ids = subject_ids,
             mask_positions = mask_positions,
             labels = labels,)