import os
os.environ["GRPC_VERBOSITY"] = "NONE" # suppress gRPC/abseil C++ log spam
os.environ["GRPC_TRACE"] = ""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse

from src.mimic.mimic import (
  build_patient_sequences,
  load_tables)
from src.mimic.helper import (
  save_dataset,
  validate_sequences, 
  validate_save_load)
from src.mimic.baseline import run_baseline
from src.config.io import (
  resolve_path,
  PROCESSED_DIR)


BQ_PROJECT_ID = "aihc-463505"
BQ_PROJECT_NAME = "mimic-aihc"
MIMIC_BQ_DATASET = "physionet-data.mimiciv_3_1_hosp"

# F32/F33 depression codes for cohort inclusion
DEPRESSION_ICD10_PREFIXES = ("F32", "F33")
# Any F-code = psychiatric diagnosis (for readmission label)
PSYCH_ICD10_PREFIX = "F"
# Minimum encounters per patient
DFLT_MIN_ENCOUNTERS = 3
# Lookback period for readmission label (in days)
DFLT_READM_WINDOW_DAYS = 90


def print_sample_sequences(sequences: list[dict], n: int = 3):
    """Pretty-print sample sequences for sanity checking."""
    print(f"\n{'=' * 60}")
    print(f"Sample Sequences (n={n})")
    print(f"{'=' * 60}")

    # Show mix of positive and negative
    pos = [s for s in sequences if s["label"] == 1]
    neg = [s for s in sequences if s["label"] == 0]
    samples = []
    if pos:
        samples.append(pos[0])
    if neg:
        samples.append(neg[0])
    remaining = [s for s in sequences if s not in samples]
    samples.extend(remaining[: n - len(samples)])

    for seq in samples[:n]:
        print(f"\n  Patient: {seq['subject_id']}  |  Label: {seq['label']}  "
              f"|  Encounters: {len(seq['encounters'])}")
        
        for i, enc in enumerate(seq["encounters"][:5]):
            icd_preview = enc["icd_codes"][:5]
            if len(enc["icd_codes"]) > 5:
                icd_preview = icd_preview + [f"...+{len(enc['icd_codes']) - 5}"]
            med_preview = enc["meds"][:3]
            if len(enc["meds"]) > 3:
                med_preview = med_preview + [f"...+{len(enc['meds']) - 3}"]
                
            print(f"    [{i}] {enc['admittime'].strftime('%Y-%m-%d')} → "
                  f"{enc['dischtime'].strftime('%Y-%m-%d')}  "
                  f"ICD: {icd_preview}  Meds: {med_preview}")
        if len(seq["encounters"]) > 5:
            print(f"\n    ... +{len(seq['encounters']) - 5} more encounters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""MIMIC-IV Patient Sequence extraction pipeline
      - Running requires either a PhysioNet-linked BigQuery project set up, or admissions.parquet, 
        diagnoses.partquet, medications.parquet, and prescriptions.parquet downloaded and specified 
        with --data-dir.
      - By default, runs data validation tests and a logistic regression baseline. Use --skip-tests and 
        '--skip-baseline' to bypass these.
      - Specify output directory for sequences and metadata with --output-dir (defaults to src/datasets/mimic-out)
      - Example usage:
        python src/extract.py --output-dir data/mimic-out-1234 --min-encounters 4 --readm-window-days 60
        python src/extract.py --data-dir data/mimic-parquet-1234 --output-dir data/mimic-out-1234

      Happy hacking!""")
    parser.add_argument("--output-dir",         default="", type=str,
                        help="sequences + metadata output directory")
    parser.add_argument("--min-encounters",     default=DFLT_MIN_ENCOUNTERS, type=int,
                        help=f"Minimum encounters per patient")
    parser.add_argument("--readm-window-days",  default=DFLT_READM_WINDOW_DAYS, type=int,
                        help=f"Readmission window in days")
    parser.add_argument("--skip-tests",         default=False, action="store_true")
    parser.add_argument("--dry_run",            default=False, action="store_true")
    args = parser.parse_args()
    out_dir = resolve_path(args.output_dir, dflt=PROCESSED_DIR)

    admissions, patients, diagnoses, prescriptions = load_tables(MIMIC_BQ_DATASET, BQ_PROJECT_ID)

    sequences = build_patient_sequences(
        admissions, patients, diagnoses, prescriptions,
        min_encounters=args.min_encounters,
        readm_window_days=args.readm_window_days,
        depression_prefixes=DEPRESSION_ICD10_PREFIXES,
        psych_prefix=PSYCH_ICD10_PREFIX)
    
    print_sample_sequences(sequences)
    if args.dry_run and not args.skip_tests:
        validate_sequences(sequences)
        validate_save_load(sequences, args.min_encounters, 
                           args.readm_window_days, DEPRESSION_ICD10_PREFIXES)
        run_baseline(sequences)
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    
    if not args.dry_run:
        save_dataset(sequences, out_dir, args.min_encounters, 
                     args.readm_window_days, DEPRESSION_ICD10_PREFIXES)