from pathlib import Path
import json
import pickle

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


def load_sequences(path: Path) -> list[dict]:
    """Load sequences from JSONL, parse ISO datetime strings back to datetime objects"""
    print(f"[load_sequences] Loading from {path}..")
    sequences = []
    try:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                for enc in record["encounters"]:
                    enc["admittime"] = pd.Timestamp(enc["admittime"])
                    enc["dischtime"] = pd.Timestamp(enc["dischtime"])
                sequences.append(record)
        print(f"[load_sequences] loaded {len(sequences)} sequences")
        return sequences
    except Exception as e:
        print(f"[load_sequences] Error loading .jsonl sequences: {e}")
        pass
    try:
        # try pickle
        with open(path, "rb") as f:
            sequences = pickle.load(f)
        print(f"[load_sequences] loaded {len(sequences)} sequences from pickle fallback")
        return sequences
    except Exception as e:
        raise ValueError(f"[load_sequences] Failed to load sequences from {path} as JSONL or pickle.")
        


# --- Data Extraction ---------------------------------------------------------------
def save_parquets(admissions, patients, diagnoses, prescriptions) -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "admissions":    admissions,
        "patients":      patients,
        "diagnoses":     diagnoses,
        "prescriptions": prescriptions,
    }
    print(f"Saving parquet cache to {PARQUET_DIR} ...")
    for name, df in tables.items():
        path = PARQUET_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {name:20s} {len(df):>8,} rows -> {path.name}")


def load_parquets(data_dir: Path) -> tuple:
    print(f"Loading from parquet ({data_dir})...")

    file_map = {
        "admissions":    "admissions.parquet",
        "patients":      "patients.parquet",
        "diagnoses":     "diagnoses.parquet",
        "prescriptions": "prescriptions.parquet",
    }

    dfs = {}
    for name, filename in file_map.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Expected files: {list(file_map.values())}")
        dfs[name] = pd.read_parquet(path)
        print(f"  {name:20s} {len(dfs[name]):>8,} rows")

    return dfs["admissions"], dfs["patients"], dfs["diagnoses"], dfs["prescriptions"]


# --- Model/Training ----------------------------------------------------------------
def load_checkpoint(
    ckpt_path: Path, 
    device: torch.device,
    opt, 
    scaler=None,
    loss_history: list[float]=[]
):
    from src.models.sequential_jepa import JEPA
    
    print(f"Loading checkpoint from:  {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # --- load model
    model_p = checkpoint["model_params"]
    model = JEPA(**model_p).to(device)
    model.load_state_dict(checkpoint["model_sd"])
    print(f"[Load Checkpoint] Model loaded successfully.")
    
    # --- load optimizer
    opt.load_state_dict(checkpoint["optimizer_sd"])
    print(f"[Load Checkpoint] Optimizer loaded successfully.")
    
    # --- load scaler if using bfloat16
    if scaler is not None and "scaler_sd" in checkpoint and checkpoint["scaler_sd"]:
        scaler.load_state_dict(checkpoint['scaler'])
        print(f"[Load Checkpoint] Grad scaler loaded successfully.")
        
    # --- load other state
    epoch = checkpoint.get("epoch", 1)
    if "loss_history" in checkpoint and checkpoint["loss_history"]:
        loss_history.extend(checkpoint["loss_history"])
        print(f"[Load Checkpoint] Loss history loaded with {len(checkpoint['loss_history'])} entries.")
        
    return model, opt, scaler, epoch, loss_history
    
    