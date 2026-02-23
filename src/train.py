#!/usr/bin/env python3

# Odd non-breaking Windows issue where PyTorch's MKL/Intel OpenMP gets initialized during the save operation
from os import environ
from typing import Dict

from .training.helper import init_optimizers
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Minimal JEPA training pipeline for longitudinal patient sequences.
Architecture
------------
  token_embedding  : nn.Embedding(vocab_size, 64), mean-pooled per encounter
  context_encoder  : hand-rolled Transformer (2 layers, 2 heads, dim=64) → z_context
  target_encoder   : EMA copy of context_encoder (τ=0.996), no backprop → z_target
  predictor        : MLP(z_context ⊕ pos_emb → z_pred, hidden=128)
  loss             : MSE(z_pred, z_target)
"""

import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg") # Set rendering backend for saving plots w/o a display
import matplotlib.pyplot as plt

from src.models.sequential_jepa import JEPA
from src.training.dataset import JEPADataset, collate_fn
from src.config.io import (
    MODEL_RUNS_BASE_DIR, 
    PROCESSED_DIR, 
    load_sequences)




SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# Vocab
# =============================================================================

def build_vocab(patients: list[dict], pad_idx: int) -> dict[str, int]:
    """Map every unique ICD code and med name to a positive integer index.
    Index 0 is reserved for [PAD].
    """
    tokens: set[str] = set()
    for p in patients:
        for enc in p.get("encounters", []):
            tokens.update(enc.get("icd_codes", []))
            tokens.update(enc.get("meds", []))
    vocab: dict[str, int] = {"[PAD]": pad_idx}
    for i, tok in enumerate(sorted(tokens), start=1):
        vocab[tok] = i
    return vocab



def main(device: torch.device, params: Dict) -> None:
    # --- Params passed from config file ---
    # Purposefully set to bad defaults to avoid silent errors
    # --- data settings
    data_p = params['data']
    batch_size = data_p.get('batch_size', 0)
    n_patients = data_p.get('n_patients', 0)
    pad_idx = data_p.get('pad_idx', -1)
    use_bfloat16 = data_p.get('use_bfloat16', False)
    # --- model hypers
    model_p = params['model']
    embed_dim = model_p.get('embed_dim', 0)
    num_heads = model_p.get('num_heads', 0)
    num_layers = model_p.get('num_layers', 0)
    ffn_dim = model_p.get('ffn_dim', 0)
    max_seq_len = model_p.get('max_seq_len', 0)
    predictor_hidden = model_p.get('predictor_hidden', 0)
    # --- optimization
    optimization_p = params['optimization']
    epochs = optimization_p.get('epochs', 0)
    tau = optimization_p.get('tau', 0.0)
    # --- artifact settings
    artifact_p = params['artifacts']
    model_tag = artifact_p.get('model_tag', '')
    checkpoint = artifact_p.get('checkpoint', None)
    checkpoint_every = artifact_p.get('checkpoint_every', epochs)
    log_emb_vecs = artifact_p.get('log_emb_vecs', True)
    log_emb_vecs_every = artifact_p.get('log_emb_vecs_every', epochs)
    
    # --- path resolution ---
    artifact_folder = MODEL_RUNS_BASE_DIR / model_tag
    artifact_folder.mkdir(parents=True, exist_ok=True)

    # --- init PATIENT SEQUENCES, VOCAB ---
    patients = load_sequences(PROCESSED_DIR / "sequences.jsonl")
    patients = patients[: n_patients]

    vocab = build_vocab(patients, pad_idx)
    vocab_out_path = artifact_folder / "vocab.json"
    with open(vocab_out_path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2)
    print(f"   vocab saved artifact folder")

    # --- init MODEL ---
    model = JEPA(
        **model_p,
        vocab_size=len(vocab),
        tau=tau,
        pad_idx=pad_idx,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {n_params:,}")
    
    
    # -- init dataset ---
    dataset = JEPADataset(patients, vocab)
    assert len(dataset) > 0, "No training samples produced. Ensure patients have at least 2 encounters."
    
    # -- init data loader ---
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False)
    
    # --- init optimizer/scheduler ---
    optimizer, scheduler, scaler = init_optimizers(
        model, 
        opt_params=optimization_p, 
        ipe=len(loader), 
        num_epochs=epochs, 
        use_bfloat16=use_bfloat16)

    # --- (Optionally) init from checkpoint ---
    start_epoch = 1
    loss_history: list[float] = []
    
    if checkpoint is not None:
        from src.config.io import load_checkpoint
        model, optimizer, scaler, start_epoch, loss_history = load_checkpoint(
            artifact_folder / f"{checkpoint.removesuffix('.pt')}.pt",
            device=device, 
            opt=optimizer, 
            scaler=None)
        
    def save_checkpoint(epoch: int, loss_history: list[float]) -> None:
        ckpt_path = artifact_folder / f"checkpoint_ep{epoch}.pt"
        torch.save({
            "epoch":        epoch,
            "loss_history": loss_history,
            "model_sd":     model.state_dict(),
            "optimizer_sd": optimizer.state_dict(),
            "scaler_sd":    None if scaler is None else scaler.state_dict(),
            "model_params": {
                "embed_dim":    embed_dim,
                "num_heads":    num_heads,
                "num_layers":   num_layers,
                "ffn_dim":      ffn_dim,
                "max_seq_len":  max_seq_len,
                "predictor_hidden": predictor_hidden,
                
                "vocab_size":   len(vocab),
                "tau":          tau,
                "pad_idx":      pad_idx,
            },
        }, ckpt_path)
    
    log_emb_vecs_fn = lambda m, ep: None
    if log_emb_vecs:
        from run_embeddings import extract_embedding_vecs as eev_fn
        log_emb_vecs_fn = lambda m, ep: eev_fn(
            model=m, loader=loader, device=device,
            out_fn=artifact_folder / f"embeddings_ep{ep}.npz")

    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    print(f"Training for {len(loader)} batches (size: {batch_size}) for {epochs} epochs")
    
    model.train()
    for epoch in range(start_epoch, epochs + 1):
        epoch_losses: list[float] = []
        # epoch_grad_norms: list[float] = []

        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            
            if use_bfloat16 and scaler is not None:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16): # type: ignore # pylance
                    _, z_pred, z_target = model(batch_dev)
                    loss = F.mse_loss(z_pred, z_target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                scaler.step(optimizer)
                scaler.update()
            else:
                _, z_pred, z_target = model(batch_dev)
                loss = F.mse_loss(z_pred, z_target)
                loss.backward()
                # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                optimizer.step()
                
            if scheduler is not None:
                scheduler.step()

            model.update_target_encoder()
            epoch_losses.append(loss.item())
            # epoch_grad_norms.append(grad_norm.item())

        mean_loss = float(np.mean(epoch_losses))
        # mean_grad_norm = float(np.mean(epoch_grad_norms))
        loss_history.append(mean_loss)
        
        # print(f"  Epoch {epoch:3d}/{epochs}  loss={mean_loss:.6f}  grad_norm={mean_grad_norm:.4f}")
        print(f"  Epoch {epoch:3d}/{epochs}  loss={mean_loss:.6f}")
        
        if checkpoint_every is not None and epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(epoch, loss_history)
            print(f"Checkpoint saved to artifact folder at epoch {epoch}")
        
        if log_emb_vecs and epoch % log_emb_vecs_every == 0 or epoch == epochs:
            log_emb_vecs_fn(model, epoch)
            print(f"Embedding vecs saved to artifact folder at epoch {epoch}")
    
    # ------------------------------------------------------------------
    # --- POST-TRAINING ------------------------------------------------
    # ------------------------------------------------------------------

    # --- loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("JEPA Training Loss")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    curve_path = artifact_folder / "loss_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to artifact folder")