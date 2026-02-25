from pathlib import Path
import json
# import pickle

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent.parent

PARQUET_DIR = ROOT / "data/parquet"
PROCESSED_DIR = ROOT / "data/processed"
MODEL_RUNS_BASE_DIR = ROOT / "artifacts/runs"


def resolve_path(p: str, dflt: Path=None) -> Path:
    """Resolve path string to relative or absolute Path, fallback to default"""
    path = Path(ROOT / p) if p != "" else dflt
    if path is None or not path.exists():
        path = Path(p)
        if path is None or not path.exists():
            if dflt is not None and dflt.exists():
                return dflt
            raise FileNotFoundError(f"Path {p} does not exist.")
    return path


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
    

def load_sequences(path: Path) -> list[dict]:
    """Load sequences from JSONL into list of dicts, parse ISO datetime strings back to datetime objects"""
    print(f"[load_sequences] Loading sequences..")
    sequences = []
    try:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                for enc in record["encounters"]:
                    enc["admittime"] = pd.Timestamp(enc["admittime"])
                    enc["dischtime"] = pd.Timestamp(enc["dischtime"])
                sequences.append(record)
        print(f"   loaded {len(sequences)} sequences")
        return sequences
    except Exception as e:
        raise FileNotFoundError(f"   Error loading .jsonl sequences: {e}")
    # # try pickle
    # with open(path, "rb") as f:
    #     sequences = pickle.load(f)

# --- Model/Training ----------------------------------------------------------------
def load_checkpoint(
    ckpt_path: Path, 
    device: torch.device,
    opt, 
    scaler=None,
    loss_history: list[float]=[]
):
    from src.models.sequential_jepa import JEPA
    
    print(f"[load_checkpoint] Loading checkpoint '{ckpt_path.name}'")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # --- load model
    model_p = checkpoint["model_params"]
    model = JEPA(**model_p).to(device)
    model.load_state_dict(checkpoint["model_sd"])
    print(f"-- Model loaded")
    
    # --- load optimizer
    opt.load_state_dict(checkpoint["optimizer_sd"])
    print(f"   Optimizer loaded")
    
    # --- load scaler if using bfloat16
    if scaler is not None and "scaler_sd" in checkpoint and checkpoint["scaler_sd"]:
        scaler.load_state_dict(checkpoint['scaler_sd'])
        print(f"   Grad scaler loaded")
        
    # --- load other state
    epoch = checkpoint.get("epoch", 1)
    if "loss_history" in checkpoint and checkpoint["loss_history"]:
        loss_history.extend(checkpoint["loss_history"])
        print(f"   Loss history loaded ({len(checkpoint['loss_history'])} entries)")
        
    return model, opt, scaler, epoch, loss_history
    
    