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

from src.config.io import load_checkpoint, resolve_path
from src.training.dataset import JEPADataset, collate_fn
from src.models.sequential_jepa import JEPA
from src.train import (
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    FFN_DIM,
    MAX_SEQ_LEN,
    PREDICTOR_HIDDEN,
    TAU,
    BATCH_SIZE,
)


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



def main(
    model: JEPA,
    loader: DataLoader,
    device: torch.device,
    artifacts_dir: Path
):    
    model.eval()

    # -- Extract -----------------------------------------------------------
    z_context, z_pred, z_target, delta, subject_ids, mask_positions, labels = \
        extract_embeddings(model, loader, device)

    # -- Sanity checks (before save) ---------------------------------------
    all_passed = run_sanity_checks(z_context, z_pred, z_target, delta)

    if not all_passed:
        print("Embeddings NOT saved: one or more sanity checks failed.")
        return

    # -- Save --------------------------------------------------------------
    out_path = artifacts_dir / "embeddings.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        z_context      = z_context,
        z_pred         = z_pred,
        z_target       = z_target,
        delta          = delta,
        subject_ids    = subject_ids,
        mask_positions = mask_positions,
        labels         = labels,
    )
    print(f"Embeddings saved to {out_path}")
    print(f"  z_context      : {z_context.shape}")
    print(f"  z_pred         : {z_pred.shape}")
    print(f"  z_target       : {z_target.shape}")
    print(f"  delta          : {delta.shape}")
    print(f"  subject_ids    : {subject_ids.shape}")
    print(f"  mask_positions : {mask_positions.shape}")
    print(f"  labels         : {labels.shape}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract JEPA embeddings from a frozen checkpoint")
    
    parser.add_argument(
        "--model_tag",  type=str, required=True,
        help="model tag (name of run folder in artifacts/runs/)")
    parser.add_argument(
        "--data_dir",   default="data/processed/sequences.jsonl",
        help="Path to sequences.jsonl")
    parser.add_argument(
        "--n-patients", type=int, default=None, 
        help="Max patients to process (default: all)")
    args = parser.parse_args()

    # --- parse paths
    data_dir = resolve_path(args.data_dir)
    ckpt_path = resolve_path(f"artifacts/runs/{args.model_tag}/model_checkpoint.pt")
    vocab_path = resolve_path(f"artifacts/runs/{args.model_tag}/vocab.json")
    artifacts_dir = resolve_path(f"artifacts/runs/{args.model_tag}")
    
    # --- Build dataset/dataloader -----------------------------------------------------
    # --- load vocab
    with open(vocab_path, encoding="utf-8") as fh:
        vocab = json.load(fh)
    assert vocab is not None, "Vocab loading failed."
    print(f"Loaded vocab from:      {vocab_path}")
    
    # --- load patients
    patients: list[dict] = []
    with open(data_dir, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                patients.append(json.loads(line))
    if args.n_patients is not None:
        patients = patients[: args.n_patients]
    print(f"Loaded patients from:   {data_dir}")
    
    # ---
    dataset = JEPADataset(patients, vocab)
    print(f"Loaded dataset with {len(dataset)} samples")
    assert len(dataset) > 0, "No samples produced. Check that patients have >=2 encounters."

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)
    
    # -- Reconstruct model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_checkpoint(str(ckpt_path.absolute()), device)
    
    main(model, loader=loader, device=device, artifacts_dir=artifacts_dir)
    