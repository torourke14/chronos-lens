#!/usr/bin/env python3

import argparse
import json

import torch
from torch.utils.data import DataLoader

from src.utils.io import (
    load_sequences, 
    PROCESSED_DIR,
    MODEL_RUNS_BASE_DIR)
from src.training.dataset import JEPADataset, collate_fn
from src.models.sequential_jepa import JEPA
from src.analysis.displacement import extract_embedding_vecs


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
    
    # --- reconstruct model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_p = checkpoint["model_params"]
    model = JEPA(**model_p).to(device)
    model.load_state_dict(checkpoint["model_sd"])
    
    # --- Build vocab/dataset/data loader
    with open(vocab_path, encoding="utf-8") as fh:
        vocab = json.load(fh)
    
    patients = load_sequences(sequences_path)
    if n_patients is not None:
        patients = patients[: n_patients]
        
    dataset = JEPADataset(patients, vocab)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn)
    
    print(f"[extract_from_checkpoint] Loaded {ckpt_fn} for extraction",
          f"   vocab: {len(vocab)} tokens",
          f"   patients: {len(patients)} sequences",
          f"   dataset: {len(dataset)} samples")
    
    ep_str = f"_ep{checkpoint['epoch']}" if "epoch" in checkpoint else ""
    extract_embedding_vecs(model, loader=loader, device=device, 
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
    
    
    
    
    
    
    