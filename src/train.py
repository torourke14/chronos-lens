#!/usr/bin/env python3

# Odd non-breaking Windows issue where PyTorch's MKL/Intel OpenMP gets initialized during the save operation
from os import environ
from typing import Dict
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

import argparse

import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg") # Set rendering backend for saving plots w/o a display
import matplotlib.pyplot as plt

from src.config.io import MODEL_RUNS_BASE_DIR, PROCESSED_DIR, resolve_path
from src.training.dataset import JEPADataset, collate_fn
from src.models.sequential_jepa import JEPA

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




EMBED_DIM        = 64
NUM_HEADS        = 2
NUM_LAYERS       = 2
FFN_DIM          = 256      # 4 × EMBED_DIM
MAX_SEQ_LEN      = 128      # positional embedding capacity
PREDICTOR_HIDDEN = 128
TAU              = 0.996    # EMA momentum
LR               = 1e-3
BATCH_SIZE       = 8
PAD_IDX          = 0        # token index reserved for padding


# =============================================================================
# Vocab
# =============================================================================

def build_vocab(patients: list[dict]) -> dict[str, int]:
    """Map every unique ICD code and med name to a positive integer index.
    Index 0 is reserved for [PAD].
    """
    tokens: set[str] = set()
    for p in patients:
        for enc in p.get("encounters", []):
            tokens.update(enc.get("icd_codes", []))
            tokens.update(enc.get("meds", []))
    vocab: dict[str, int] = {"[PAD]": PAD_IDX}
    for i, tok in enumerate(sorted(tokens), start=1):
        vocab[tok] = i
    return vocab


# =============================================================================
# Training loop
# =============================================================================

def train(
    model: JEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
) -> list[float]:
    model.train()
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []

        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            _, z_pred, z_target = model(batch_dev)
            loss = F.mse_loss(z_pred, z_target)
            loss.backward()
            optimizer.step()

            model.update_target_encoder()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)
        print(f"  Epoch {epoch:3d}/{epochs}  loss={mean_loss:.6f}")

    return loss_history



def main(device: torch.device, params: Dict) -> None:
    # --- Params passed from config file ---
    # --- data settings
    batch_size = params.get('data', {}).get('batch_size', 0)
    n_patients = params.get('data', {}).get('n_patients', 0)
    pad_idx = params.get('data', {}).get('pad_idx', 0)
    data_dir = params.get('data', {}).get('data_dir', '')
    # --- artifact settings
    output_dir = params.get('artifacts', {}).get('output_dir', '')
    model_tag = params.get('artifacts', {}).get('tag', '')
    run_emb_extraction = params.get('artifacts', {}).get('run_emb_extraction_post', True)
    # --- model hypers
    embed_dim = params.get('model', {}).get('embed_dim', 0)
    num_heads = params.get('model', {}).get('num_heads', 0)
    num_layers = params.get('model', {}).get('num_layers', 0)
    ffn_dim = params.get('model', {}).get('ffn_dim', 0)
    max_seq_len = params.get('model', {}).get('max_seq_len', 0)
    predictor_hidden = params.get('model', {}).get('predictor_hidden_dim', 0)
    # --- optimization
    epochs = params.get('optimization', {}).get('epochs', 0)
    tau = params.get('optimization', {}).get('tau', 0.0)
    lr = params.get('optimization', {}).get('lr', 0.0)
    
    
    # --- path resolution and sanity checks ---
    data_dir = resolve_path(data_dir, PROCESSED_DIR)
    out_dir = resolve_path(output_dir, dflt=MODEL_RUNS_BASE_DIR) / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)


    # -- LOAD DATA
    print(f"Loading data from: {data_dir}")
    patients: list[dict] = []
    with open(data_dir / "sequences.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                patients.append(json.loads(line))
    patients = patients[: n_patients]
    print(f"Patients loaded:   {len(patients)}")


    # -- BUILD VOCAB
    vocab = build_vocab(patients)
    print(f"Vocabulary size:   {len(vocab)}")

    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2)
    print(f"Vocab saved →      {vocab_path}")


    # -- MODEL
    model = JEPA(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        ffn_dim=ffn_dim,
        predictor_hidden=predictor_hidden,
        tau=tau,
        pad_idx=pad_idx,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params:  {n_params:,}")
    
    
    # -- init dataset
    # TODO LATER: init_dataset function - build vocab, create dataset object, return dataloader
    # Note: dataset constructor will skip patients with <2 encounters, since they can't be used for the JEPA objective.
    dataset = JEPADataset(patients, vocab)
    print(f"Training samples:  {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("No training samples produced. Ensure that patients have at least 2 encounters.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    
    # --- init optimizer/scheduler
    # TODO LATER: init_opt function - initialize optimizer, scheduler, grad scalar if using bfloat16
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    # --- TRAINING LOOP
    print(f"\nTraining for {epochs} epochs (batch_size={batch_size}, lr={lr}) …")
    loss_history = train(model, loader, optimizer, epochs, device)



    # --- save checkpoint
    # TODO LATER:
    # - move above train loop
    # - loading model checkpoint, save_checkpoint every N epochs
    
    ckpt_path = out_dir / "model_checkpoint.pt"
    torch.save({
        "model_sd": model.state_dict(),
        "optimizer_sd": optimizer.state_dict(),
        "vocab_size":       len(vocab),
        "loss_history":     loss_history,
        "model_params": {
            "embed_dim":    embed_dim,
            "num_heads":    num_heads,
            "num_layers":   num_layers,
            "ffn_dim":      ffn_dim,
            "max_seq_len":  max_seq_len,
            "predictor_hidden": predictor_hidden,
            "tau":          tau,
        }
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")
    
    
    # --- optionally run embedding extraction after training
    if run_emb_extraction:
        print(f"\nRunning embedding extraction from checkpoint …")
        from run_embeddings import main as extract_main
        extract_main(model, loader=loader, device=device, artifacts_dir=out_dir)

    # --- loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("JEPA Training Loss")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    curve_path = out_dir / "loss_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {curve_path}")