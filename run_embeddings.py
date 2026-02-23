#!/usr/bin/env python3
"""
Standalone embedding extraction from a frozen JEPA checkpoint.

Loads a saved model_checkpoint.pt and vocab.json, runs a full forward pass
over every (patient x masked-encounter) sample, and writes embeddings.npz.

This is the freeze boundary: everything downstream is pure numpy.

CLI
---
  python src/extract_embeddings.py \\
      --checkpoint runs/undertrained/model_checkpoint.pt \\
      --vocab      runs/undertrained/vocab.json \\
      --data       sequences.jsonl \\
      --output     runs/undertrained/embeddings.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config.io import (
    load_sequences, 
    PROCESSED_DIR,
    MODEL_RUNS_BASE_DIR)
from src.training.dataset import JEPADataset, collate_fn
from src.models.sequential_jepa import JEPA


# =============================================================================
# Embedding extraction
# =============================================================================

@torch.no_grad()
def extract_embeddings(
    model: JEPA,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    print("\nExtracting embeddings...")
    
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


# -----------------------------------------------------------------------------
# Sanity checks (no loss history - pure geometry on the embedding arrays)
# -----------------------------------------------------------------------------

def run_sanity_checks(
    z_context:  np.ndarray,
    z_pred:     np.ndarray,
    z_target:   np.ndarray,
    delta:      np.ndarray,
) -> bool:
    print("\n" + "=" * 62)
    print("SANITY CHECKS")
    print("=" * 62)

    results: list[tuple[bool, str]] = []

    # 1. Consistent N_samples across all arrays
    n_samples = {
        "z_context": z_context.shape[0],
        "z_pred":    z_pred.shape[0],
        "z_target":  z_target.shape[0],
        "delta":     delta.shape[0],
    }
    ok1 = len(set(n_samples.values())) == 1
    results.append((ok1, f"Consistent N_samples: {n_samples}"))

    # 2. z_pred has non-trivial variance (std > 0.01 per dimension)
    pred_std        = z_pred.std(axis=0)
    frac_nontrivial = float((pred_std > 0.01).mean())
    ok2             = frac_nontrivial > 0.5
    results.append((ok2, (
        f"z_pred non-trivial variance: "
        f"min_std={pred_std.min():.4f}, "
        f"frac_dims(>0.01)={frac_nontrivial:.1%}"
    )))

    # 3. ||Delta|| distribution stats
    delta_norms = np.linalg.norm(delta, axis=-1)
    ok3 = delta_norms.std() > 1e-4
    results.append((ok3, (
        f"||Δ|| distribution: "
        f"mean={delta_norms.mean():.4f}, "
        f"std={delta_norms.std():.4f}, "
        f"min={delta_norms.min():.4f}, "
        f"max={delta_norms.max():.4f}"
    )))

    # 4. Predictor beats a random baseline scaled to z_target's distribution
    rng       = np.random.default_rng(42)
    rand_pred = rng.standard_normal(z_pred.shape).astype(np.float32) \
                * z_target.std() + z_target.mean()
    pred_dist = float(np.linalg.norm(z_pred    - z_target, axis=-1).mean())
    rand_dist = float(np.linalg.norm(rand_pred - z_target, axis=-1).mean())
    ok4       = pred_dist < rand_dist
    results.append((ok4, (
        f"Predictor beats random: "
        f"||z_pred-z_tgt||={pred_dist:.4f} vs "
        f"||rand-z_tgt||={rand_dist:.4f}"
    )))

    for idx, (ok, msg) in enumerate(results, start=1):
        print(f"  [{'PASS' if ok else 'FAIL'}] {idx}. {msg}")

    n_pass = sum(ok for ok, _ in results)
    print(f"\n  {n_pass}/{len(results)} checks passed.")
    print("=" * 62 + "\n")
    return n_pass == len(results)



def extract(
    model: JEPA,
    loader: DataLoader,
    device: torch.device,
    out_fn: Path,
):    
    model.eval()

    # -- Extract -----------------------------------------------------------
    z_context, z_pred, z_target, delta, subject_ids, mask_positions, labels = \
        extract_embeddings(model, loader, device)

    # don't save if embeddings not statistically sane (e.g. predictor collapsed)
    all_passed = run_sanity_checks(z_context, z_pred, z_target, delta)

    if not all_passed:
        print("Embeddings NOT saved: one or more sanity checks failed.")
        return

    np.savez(
        out_fn,
        z_context      = z_context,
        z_pred         = z_pred,
        z_target       = z_target,
        delta          = delta,
        subject_ids    = subject_ids,
        mask_positions = mask_positions,
        labels         = labels,
    )
    print(f"Embeddings saved to {out_fn}")
    print(f"  z_context      : {z_context.shape}")
    print(f"  z_pred         : {z_pred.shape}")
    print(f"  z_target       : {z_target.shape}")
    print(f"  delta          : {delta.shape}")
    print(f"  subject_ids    : {subject_ids.shape}")
    print(f"  mask_positions : {mask_positions.shape}")
    print(f"  labels         : {labels.shape}")


def extract_from_checkpoint(
    model_tag:     str,
    ckpt_name:     str,
    n_patients:    int | None = None,
    batch_size:    int = 64,
):
    artifact_dir = MODEL_RUNS_BASE_DIR / model_tag
    assert artifact_dir.exists(), f"[extract_from_checkpoint] Model run folder not found: {artifact_dir}"
    
    ckpt_fn = f"{ckpt_name.replace('.pt', '')}.pt"
    checkpoint_path = artifact_dir / ckpt_fn
    assert checkpoint_path.exists(), f"[extract_from_checkpoint] Checkpoint file not found: {checkpoint_path}"
    
    vocab_path = artifact_dir / "vocab.json"
    assert vocab_path.exists(), f"[extract_from_checkpoint] Vocab file not found: {vocab_path}"
    
    sequences_path = PROCESSED_DIR / "sequences.jsonl"
    assert sequences_path.exists(), f"[extract_from_checkpoint] Sequences file not found: {sequences_path}"
    
    # -- Reconstruct model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # --- load model
    model_p = checkpoint["model_params"]
    model = JEPA(**model_p).to(device)
    model.load_state_dict(checkpoint["model_sd"])
    print(f"[extract_from_checkpoint] Model loaded from checkpoint successfully.")
    
    # --- Build vocab/dataset/dataloader -----------------------------------------------------
    with open(vocab_path, encoding="utf-8") as fh:
        vocab = json.load(fh)
    assert vocab is not None and len(vocab) > 0, "Vocab loading failed."
    print(f"[extract_from_checkpoint]vocab loaded vocab")
    
    # --- load patients
    patients = load_sequences(sequences_path)
    if n_patients is not None:
        patients = patients[: n_patients]
    print(f"[extract_from_checkpoint] Loaded {len(patients)} patient sequences")
    
    # ---
    dataset = JEPADataset(patients, vocab)
    assert len(dataset) > 0, "No samples produced. Check that patients have >=2 encounters."
    print(f"[extract_from_checkpoint] Loaded dataset with {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn)
    
    ep_str = f"_ep{checkpoint['epoch']}" if "epoch" in checkpoint else ""
    extract(model, loader=loader, device=device, 
            out_fn=artifact_dir / f"embeddings_{model_tag}{ep_str}.npz")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract JEPA embeddings from a frozen checkpoint")
    
    parser.add_argument("--model_tag",  type=str, required=True,
                        help="model tag (name of run folder in artifacts/runs/)")
    parser.add_argument("--checkpoint_name",  type=str, required=True,
                        help="checkpoint filename (e.g. model_checkpoint.pt)")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for embedding passover (default: 64)")
    parser.add_argument("--n-patients", type=int, default=None, 
                        help="Max patients to process (default: all)")
    args = parser.parse_args()

    extract_from_checkpoint(
        model_tag=args.model_tag,
        ckpt_name=args.checkpoint_name,
        n_patients=args.n_patients,
        batch_size=args.batch_size,
    )
    
    
    
    
    
    
    