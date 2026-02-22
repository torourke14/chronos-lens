from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent.parent

PARQUET_DIR = ROOT / "data/parquet"
PROCESSED_DIR = ROOT / "data/processed"
MODEL_RUNS_BASE_DIR = ROOT / "artifacts/runs"
MODELS_DIR = ROOT / "artifacts/models"


def save_checkpoint(model, vocab, embed_dim, loss_history, checkpoint_dir):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": len(vocab),
        "embed_dim": embed_dim,
        "loss_history": loss_history,
    }, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")


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
  

def save_dataset(
    sequences: list[dict], 
    out_dir: Path,
    min_encounters: int,
    readm_window_days: int,
    depression_prefixes: tuple[str, ...]
):
    """
    Save sequences in (model-ready format)
      PROCESSED_DIR/
        sequences.jsonl    - one JSON object p/patient (primary format)
        sequences.pkl      - pickle backup (preserves datetime objects)
        metadata.json      - cohort stats, schema, label distribution
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- JSONL ---
    jsonl_path = out_dir / "sequences.jsonl"
    with open(jsonl_path, "w") as f:
        for seq in sequences:
            f.write(json.dumps({
                "subject_id": seq["subject_id"],
                "label": seq["label"],
                "encounters": [
                    {
                        "hadm_id": enc["hadm_id"],
                        "admittime": enc["admittime"].isoformat(),
                        "dischtime": enc["dischtime"].isoformat(),
                        "icd_codes": enc["icd_codes"],
                        "meds": enc["meds"],
                    }
                    for enc in seq["encounters"]
                ],
            }) + "\n")

    # --- pickle (backup) ---
    pkl_path = out_dir / "sequences.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(sequences, f)

    # --- data is so meta ---
    enc_counts = [len(s["encounters"]) for s in sequences]
    n_pos = sum(1 for s in sequences if s["label"] == 1)
    all_icd = set()
    all_meds = set()
    for s in sequences:
        for enc in s["encounters"]:
            all_icd.update(enc["icd_codes"])
            all_meds.update(enc["meds"])

    metadata = {
        "n_patients": len(sequences),
        "n_positive": n_pos,
        "n_negative": len(sequences) - n_pos,
        "positive_rate": round(n_pos / len(sequences), 4),
        "encounters_per_patient": {
            "mean": round(np.mean(enc_counts), 2),
            "median": int(np.median(enc_counts)),
            "min": int(min(enc_counts)),
            "max": int(max(enc_counts)),
        },
        "vocab_size_icd": len(all_icd),
        "vocab_size_meds": len(all_meds),
        "cohort_filters": {
            "depression_codes": list(depression_prefixes),
            "min_encounters": min_encounters,
            "READM_WINDOW_DAYS": readm_window_days,
            "exclude_deceased": True,
        },
        "schema": {
            "subject_id": "str",
            "label": "int (0 or 1)",
            "encounters[].hadm_id": "int",
            "encounters[].admittime": "ISO datetime string",
            "encounters[].dischtime": "ISO datetime string",
            "encounters[].icd_codes": "list[str] - ICD codes with dot notation",
            "encounters[].meds": "list[str] - lowercase drug names active at admission",
        },
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to {out_dir}/")
    print(f"  {jsonl_path.name:20s} {jsonl_path.stat().st_size / 1024:.0f} KB  (primary — for JEPA dataloader)")
    print(f"  {pkl_path.name:20s} {pkl_path.stat().st_size / 1024:.0f} KB  (backup — preserves datetime)")
    print(f"  {meta_path.name:20s} {meta_path.stat().st_size / 1024:.0f} KB  (cohort stats & schema)")


def load_sequences(path: str) -> list[dict]:
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


def resolve_path(p: str, dflt: Path) -> Path:
    """Resolve path string to relative or absolute Path, fallback to default"""
    path = Path(ROOT / p) if p != "" else dflt
    if not path.exists():
        path = ROOT / path
        if not path.exists():
            if dflt is not None and dflt.exists():
                return dflt
            raise FileNotFoundError(f"Path {p} does not exist.")
    return path