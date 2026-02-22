from typing import Tuple
import json
import tempfile
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.config.io import load_sequences_jsonl


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


def validate_sequences(sequences: list[dict]):
    """Run all structural checks on built sequences."""
    print("\nSEQUENCE VALIDATION")
    print("=" * 60)

    assert len(sequences) > 0, "FAIL: no sequences produced"
    print(f"  Pipeline produced {len(sequences)} patient sequences")

    # Min encounters
    min_enc = min(len(s["encounters"]) for s in sequences)
    assert min_enc >= 3, f"FAIL: found sequence with {min_enc} encounters"
    print(f"  All sequences have >= 3 encounters (min={min_enc})")

    # F32/F33 in every patient
    for seq in sequences:
        has_dep = any(
            code.upper().replace(".", "").startswith(("F32", "F33"))
            for enc in seq["encounters"]
            for code in enc["icd_codes"]
        )
        assert has_dep, f"FAIL: patient {seq['subject_id']} has no F32/F33"
    print("  All patients have at least one F32/F33 diagnosis")

    # Time ordering
    for seq in sequences:
        times = [enc["admittime"] for enc in seq["encounters"]]
        assert times == sorted(times), f"FAIL: patient {seq['subject_id']} not sorted"
    print("  ✓ All encounter sequences are time-ordered")

    # Binary labels
    labels = set(s["label"] for s in sequences)
    assert labels.issubset({0, 1}), f"FAIL: unexpected labels {labels}"
    print("  ✓ Labels are binary (0/1)")

    n_pos = sum(1 for s in sequences if s["label"] == 1)
    n_neg = len(sequences) - n_pos
    print(f"  ✓ Label distribution: {n_pos} positive, {n_neg} negative")

    # Schema
    for seq in sequences:
        assert isinstance(seq["subject_id"], str)
        assert isinstance(seq["label"], int)
        assert isinstance(seq["encounters"], list)
        for enc in seq["encounters"]:
            assert isinstance(enc["hadm_id"], int)
            assert isinstance(enc["icd_codes"], list)
            assert isinstance(enc["meds"], list)
            assert hasattr(enc["admittime"], "strftime")
    print("  ✓ Schema matches target specification")

    # Meds normalized
    for seq in sequences[:50]:
        for enc in seq["encounters"]:
            for med in enc["meds"]:
                assert med == med.lower().strip(), f"FAIL: med '{med}' not normalized"
    print("  ✓ Medication names are lowercase and stripped")


def validate_save_load(sequences: list[dict], min_encounters: int, 
                       readm_window_days: int, depression_prefixes: Tuple
):
    """Verify save_dataset output and JSONL round-trip."""
    print("\nSAVE / LOAD VALIDATION")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dataset(sequences, Path(tmpdir), min_encounters, readm_window_days, depression_prefixes)

        assert (Path(tmpdir) / "sequences.jsonl").exists()
        assert (Path(tmpdir) / "sequences.pkl").exists()
        assert (Path(tmpdir) / "metadata.json").exists()
        print("  All output files present")

        with open(Path(tmpdir) / "metadata.json") as f:
            meta = json.load(f)
        assert meta["n_patients"] == len(sequences)
        assert meta["vocab_size_icd"] > 0
        assert meta["vocab_size_meds"] > 0
        print(f"  Metadata valid (vocab: {meta['vocab_size_icd']} ICD, {meta['vocab_size_meds']} meds)")

        loaded = load_sequences_jsonl(str(Path(tmpdir) / "sequences.jsonl"))
        assert len(loaded) == len(sequences)
        assert loaded[0]["subject_id"] == sequences[0]["subject_id"]
        assert isinstance(loaded[0]["encounters"][0]["admittime"], pd.Timestamp)
        print("  JSONL round-trip preserves data and datetimes")