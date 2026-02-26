#!/usr/bin/env python3

import argparse
import json

import torch
from torch.utils.data import DataLoader

from src.utils.io import load_sequences
from src.training.dataset import JEPADataset, collate_fn
from src.utils.io import load_sequences, load_checkpoint, EXPERIMENTS_DIR
from src.analysis.displacement import save_embedding_vecs


def save_embedding_vecs_from_checkpoint(
    model_tag:     str,
    ckpt_name:     str,
    n_patients:    int | None = None,
    batch_size:    int = 64,
):
    """ Utility to save embedding vectors from a frozen checkpoint.
        Possibly improve later to run full analysis from checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    artifact_dir = EXPERIMENTS_DIR / model_tag
    assert artifact_dir.exists(), f"[extract_from_checkpoint] Model run folder not found: {artifact_dir}"
    
    # --- reconstruct model from checkpoint
    chkpt_dir = artifact_dir / "checkpoints" / ckpt_name
    model, checkpoint = load_checkpoint(chkpt_dir, None, None, chkpt_dir, device, resume_optimizer=False)
    
    # --- Build vocab/dataset/data loader
    vocab_path = artifact_dir / "vocab.json"
    assert vocab_path.exists(), f"[extract_from_checkpoint] Vocab file not found: {vocab_path}"
    
    with open(vocab_path, encoding="utf-8") as fh:
        vocab = json.load(fh)
        
    patients = load_sequences(n=n_patients)
    dataset = JEPADataset(patients, vocab)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn)
    
    print(f"[extract_from_checkpoint] Loaded {chkpt_dir.name} for extraction",
          f"   vocab: {len(vocab)} tokens",
          f"   patients: {len(patients)} sequences",
          f"   dataset: {len(dataset)} samples")
    
    ep_str = f"_ep{checkpoint['epoch']}" if "epoch" in checkpoint else ""
    save_embedding_vecs(model, loader=loader, device=device, 
            out_fn=artifact_dir / f"embeddings_{model_tag}{ep_str}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract JEPA embeddings from a frozen checkpoint")
    
    parser.add_argument("--model_tag",  type=str, required=True,
                        help="model tag (name of run folder in 'artifacts')")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="full path to checkpoint (e.g. 'jepa_64-2-2/checkpoint_ep10')")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for embedding passover (default: 32)")
    parser.add_argument("--n-patients", type=int, default=None, 
                        help="Max patients to process (default: all)")
    args = parser.parse_args()

    save_embedding_vecs_from_checkpoint(
        model_tag=args.model_tag,
        ckpt_name=args.checkpoint_name,
        n_patients=args.n_patients,
        batch_size=args.batch_size,
    )