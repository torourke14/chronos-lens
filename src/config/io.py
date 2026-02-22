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


def load_sequences_pkl(path: str) -> list[dict]:
    """Load sequences from pickle file."""
    with open(path, "rb") as f:
        sequences = pickle.load(f)
    print(f"Loaded {len(sequences)} sequences from {path}")
    return sequences


def load_sequences_jsonl(path: str) -> list[dict]:
    """Load sequences from JSONL, parse ISO datetime strings back to datetime objects"""
    sequences = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            for enc in record["encounters"]:
                enc["admittime"] = pd.Timestamp(enc["admittime"])
                enc["dischtime"] = pd.Timestamp(enc["dischtime"])
            sequences.append(record)
    print(f"Loaded {len(sequences)} sequences from {path}")
    return sequences


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
    ckpt_path: str, device: torch.device,
    opt=None, scaler=None
):
    from src.models.sequential_jepa import JEPA
    
    print(f"Loading checkpoint from:  {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # --- load model 
    model = JEPA(
        vocab_size       = checkpoint["vocab_size"],
        embed_dim        = checkpoint["model_params"]["embed_dim"],
        num_heads        = checkpoint["model_params"]["num_heads"],
        num_layers       = checkpoint["model_params"]["num_layers"],
        max_seq_len      = checkpoint["model_params"]["max_seq_len"],
        ffn_dim          = checkpoint["model_params"]["ffn_dim"],
        predictor_hidden = checkpoint["model_params"]["predictor_hidden"],
        tau              = checkpoint["model_params"]["tau"],
    ).to(device)
    model.load_state_dict(checkpoint["model_sd"])
    print(f"[Load Checkpoint] Model loaded successfully.")
    
    # --- load optimizer
    if opt is not None and "optimizer_sd" in checkpoint:
        opt.load_state_dict(checkpoint["optimizer_sd"])
        print(f"[Load Checkpoint] Optimizer loaded successfully.")
    
    # --- load scaler if using bfloat16
    if scaler is not None and "scaler_sd" in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        print(f"[Load Checkpoint] Grad scaler loaded successfully.")
    
    
    return model, opt, scaler
    
    