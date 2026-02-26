#!/usr/bin/env python3
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
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ^^ Odd non-breaking Windows issue where PyTorch's MKL/Intel OpenMP gets initialized during the save operation

from typing import Dict
from pathlib import Path
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
from src.training.optimizers import init_optimizers
from src.training.helper import build_vocab
from src.analysis.displacement import save_embedding_vecs
from src.utils.io import load_sequences
from src.analysis.plotting import loss_curve



SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def main(params: Dict, run_dir: Path, device: torch.device) -> None:
    model_tag = run_dir.name
    vocab_out_path = run_dir / "vocab.json"
    
    # --- Params passed from config file ---
    # Purposefully set to bad defaults to avoid silent errors
    # --- model hypers
    model_p =       params['model']
    embed_dim =     model_p.get('embed_dim', 0)
    num_heads =     model_p.get('num_heads', 0)
    num_layers =    model_p.get('num_layers', 0)
    ffn_dim =       model_p.get('ffn_dim', 0)
    max_seq_len =   model_p.get('max_seq_len', 0)
    predictor_hidden = model_p.get('predictor_hidden', 0)
    use_bfloat16 =  model_p.get('use_bfloat16', False)
    
    # --- optimization
    opt_params =    params['optimization']
    epochs =        opt_params.get('epochs', 0)
    tau =           opt_params.get('tau', 0.0)
    
    # --- data settings
    data_p =        params['data']
    batch_size =    data_p.get('batch_size', 0)
    n_patients =    data_p.get('n_patients', 0)
    pad_idx =       data_p.get('pad_idx', 0)
    
    # --- artifact settings
    artifact_p =    params['artifacts']
    checkpoint =    artifact_p.get('checkpoint', None)
    checkpoint_every = artifact_p.get('checkpoint_every', epochs)
    log_emb_vecs =  artifact_p.get('log_emb_vecs', True)
    log_emb_vecs_every = artifact_p.get('log_emb_vecs_every', epochs)

    # --- sequences, vocab, and dataset ---
    patients = load_sequences(n=n_patients)
    vocab = build_vocab(patients, pad_idx)
    with open(vocab_out_path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2)
        
    dataset = JEPADataset(patients, vocab)

    # --- MODEL ---
    model = JEPA(
        **model_p,
        vocab_size=len(vocab), tau=tau, pad_idx=pad_idx,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {(n_params / 1e6):2f}M")
    
    # -- data loader ---
    loader = DataLoader(
        dataset, batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=False)
    
    # --- optimizer/scheduler/scaler ---
    optimizer, scheduler, scaler = init_optimizers(
        model, opt_params, 
        ipe=len(loader), 
        num_epochs=epochs, 
        use_bfloat16=use_bfloat16)

    # --- (Optionally) load from checkpoint ---
    start_epoch = 1
    loss_history: list[float] = []
    
    if params.get("resume_from"):
        from src.utils.io import load_checkpoint
        start_epoch, loss_history = load_checkpoint(
            model, optimizer, scaler,
            ckpt_path = run_dir / params["resume_from"],
            device=device,
            resume_optimizer=params.get("resume_optimizer", False))
        
    def save_checkpoint(epoch: int, loss_history: list[float]) -> None:
        ckpt_path = run_dir / f"epoch_{epoch}.pt"
        torch.save({
            "epoch":        epoch,
            "loss_history": loss_history,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    None if scaler is None else scaler.state_dict(),
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
    
    # Disabling tf32 matmul when using bfloat16 avoids stacking two levels of reduced precision
    if device.type == "cuda" and use_bfloat16:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True

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
                with torch.amp.autocast('cuda', dtype=torch.bfloat16): # type: ignore # (pylance)
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
        print(f"-- Epoch {epoch:3d}/{epochs}  loss={mean_loss:.6f}")
        
        if checkpoint_every is not None and epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(epoch, loss_history)
            print(f"   Checkpoint saved to run directory at epoch {epoch}")
        
        if log_emb_vecs and epoch % log_emb_vecs_every == 0 or epoch == epochs:
            save_embedding_vecs(model, loader, device=device,
                                out_fn=run_dir / f"embeddings_ep{epoch}.npz", 
                                log=True)
            print(f"   Embedding vectors saved to run directory at epoch {epoch}")
    
    # ------------------------------------------------------------------
    # --- POST-TRAINING ------------------------------------------------
    # ------------------------------------------------------------------
    print(f"Training finished with loss={loss_history[-1]:.6f}")

    # --- loss curve
    loss_curve(loss_history, run_dir)