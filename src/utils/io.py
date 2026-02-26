from pathlib import Path
import json
import yaml
# import pickle

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent.parent

PARQUET_DIR = ROOT / "data/parquet"
PROCESSED_DIR = ROOT / "data/processed"
EXPERIMENTS_DIR = ROOT / "experiments"


def resolve_run_dir(prefix: str) -> Path:
    existing = sorted(EXPERIMENTS_DIR.glob(f"{prefix}_v*"))
    if not existing:
        return EXPERIMENTS_DIR / f"{prefix}_v001"
    last_num = int(existing[-1].name.split("_v")[-1])
    return EXPERIMENTS_DIR / f"{prefix}_v{last_num + 1:03d}"


def create_run(model: str) -> tuple[Path, dict]:
    run_dir = EXPERIMENTS_DIR / model
    
    if run_dir.exists():
        run_dir = resolve_run_dir(model)
        run_dir.mkdir(parents=True, exist_ok=True)
    
    for sub in ["checkpoints", "logs", "metrics"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
        
    params: dict = {}
    with open(run_dir / "config.yaml", 'r') as y_file:
        params = yaml.safe_load(y_file)
    
    return (run_dir, params)


def load_sequences_dict(path: Path) -> dict:
    """Load sequences from JSONL into dict of dicts"""
    patients = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p
    return patients
    

def load_sequences(n=None) -> list[dict]:
    """Load sequences from JSONL into list of dicts, parse ISO datetime strings back to datetime objects"""
    sequences = []
    try:
        with open(PROCESSED_DIR / "sequences.jsonl") as f:
            for line in f:
                record = json.loads(line)
                for enc in record["encounters"]:
                    enc["admittime"] = pd.Timestamp(enc["admittime"])
                    enc["dischtime"] = pd.Timestamp(enc["dischtime"])
                sequences.append(record)
        
        if n is None or n == 0:
            return sequences
        if n < 0:
            raise ValueError("n must be non-negative")
        return sequences[:n]
    except Exception as e:
        raise FileNotFoundError(f"[load_sequences] Error loading .jsonl sequences from '{PROCESSED_DIR}/sequences.jsonl': {e}")
    # pickle
    # with open(path, "rb") as f:
    #     sequences = pickle.load(f)
    

# --- Model/Training ----------------------------------------------------------------
def load_checkpoint(model, opt, scaler, ckpt_path, device, resume_optimizer = False):
    ckpt_path = Path(ckpt_path).with_suffix(".pt")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # validate architecture matches
    saved = checkpoint["model_params"]
    for k in ["embed_dim", "num_heads", "num_layers", "ffn_dim"]:
        cur = getattr(model, k, None)
        if cur is not None and cur != saved[k]:
            raise ValueError(f"Config Mismatch: {k}: config={cur}, checkpoint={saved[k]}")
    
    # if loading outside of training loop
    if model is None:
        from src.models.sequential_jepa import JEPA
        model = JEPA(saved).to(device)
        model.load_state_dict(checkpoint["model_sd"])
        return model, checkpoint
        
    model.load_state_dict(checkpoint["model_sd"])
    
    if resume_optimizer and opt is not None:
        opt.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint['scaler'])
    
    start_epoch = checkpoint.get("epoch", 0) + 1
    loss_history = checkpoint.get("loss_history", [])
        
    return start_epoch, loss_history
    
    