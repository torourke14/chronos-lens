"""
MIMIC-IV Patient Sequence Feature Extraction Pipeline functions
===================================================
Build patient-level temporal sequences from MIMIC-IV v3.1 (BigQuery) 
for 90-day psychiatric readmission prediction.

Schema per patient:
{
  "subject_id": str,
  "encounters": [
    {
      "hadm_id": int,
      "admittime": datetime,
      "dischtime": datetime,
      "icd_codes": ["F32.1", "I10", ...],
      "meds": ["sertraline", "metformin", ...]
    }, ...
  ],
  "label": int  # 1 if any subsequent admission has F-code within 90 days of dischtime
}

Cohort filters:
  - At least one admission with F32/F33 as primary or secondary diagnosis
  - Minimum 3 encounters per patient
  - Exclude deceased during admission (hospital_expire_flag = 0, deathtime is null)

BigQuery auth setup:
  gcloud auth application-default login
  gcloud config set project aihc-463505
"""

from typing import Tuple
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import google.auth
from google.cloud import bigquery

from src.utils.io import (
    load_parquets, 
    save_parquets,
    PARQUET_DIR)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def _authenticate():
    print(f"Authenticating...")
    try:
        credentials, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        assert credentials is not None, "No credentials found"
        assert project is not None, "No project found in credentials"
        
        print(f"  Authenticated (project: {project})")
        return credentials, project
    except Exception:
        raise RuntimeError("BigQuery auth failed. Run 'gcloud auth application-default login'")
    

def load_tables(dataset: str, project_id: str = None) -> tuple:
    """ Load MIMIC tables from parquet cache if available, otherwise from BigQuery.
        tables are saved to PARQUET_SUBDIR on BigQuery fetch so subsequent runs skip 
        the call entirely (save $$$).
    """
    _PARQUET_FILES = ["admissions.parquet", "patients.parquet",
                      "diagnoses.parquet", "prescriptions.parquet"]
    
    if all((PARQUET_DIR / f).exists() for f in _PARQUET_FILES):
        return load_parquets(PARQUET_DIR)

    credentials, detected_project = _authenticate()
    if project_id is None:
        project_id = detected_project
    client = bigquery.Client(project=project_id, credentials=credentials)
    print(f"Loading from BigQuery ({dataset}) from {project_id}...")
    
    bq_tables = {
        "admissions":    f"{dataset}.admissions",
        "patients":      f"{dataset}.patients",
        "diagnoses":     f"{dataset}.diagnoses_icd",
        "prescriptions": f"{dataset}.prescriptions",
    }
    dfs = {}
    for name, table in bq_tables.items():
        df = client.query(f"SELECT * FROM `{table}`").to_dataframe()
        dfs[name] = df
        print(f"  {name:20s} {len(df):>8,} rows")

    save_parquets(dfs["admissions"], dfs["patients"],
                 dfs["diagnoses"], dfs["prescriptions"])

    return dfs["admissions"], dfs["patients"], dfs["diagnoses"], dfs["prescriptions"]


def clean_admissions(admissions: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to alive-at-discharge admissions and parse datetimes.

    Note: patients table is NOT merged here. Demographics are deliberately
    excluded from encounter sequences. patients table is only needed if 
    demographics are later added as an experiment.
    """
    adm = admissions[["subject_id", "hadm_id", "admittime", "dischtime",
                       "deathtime", "hospital_expire_flag"]].copy()

    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])

    n_before = len(adm)

    # alive at discharge, valid time range
    adm = adm[
        (adm["hospital_expire_flag"] == 0)
        & (adm["deathtime"].isna())
        & (adm["admittime"] < adm["dischtime"])
    ].copy()
    
    adm = adm[["subject_id", "hadm_id", "admittime", "dischtime"]].reset_index(drop=True)

    print(f"  Clean admissions: {len(adm):,} / {n_before:,} ({adm['subject_id'].nunique():,} patients)")
    return adm


def build_admission_icd_codes(diagnoses: pd.DataFrame, depr_prefixes: Tuple, psych_prefix: str) -> pd.DataFrame:
    """
    Group all billed ICD codes per admission.

    Returns DataFrame with columns:
        hadm_id, icd_codes (list[str]), has_depression (bool), has_fcode (bool)

    ICD-10 codes format: dot notation (e.g., F32.1).
    ICD-9 codes format: as-is (e.g., 296.00).
    """
    dx = diagnoses[["hadm_id", "icd_code", "icd_version"]].copy()

    # strip whitespace, uppercase, remove dots for matching
    dx["icd_clean"] = dx["icd_code"].astype(str).str.strip().str.replace(".", "", regex=False).str.upper()

    # Display format: insert dot after 3rd char for ICD-10 codes > 3 chars
    is_icd10_long = (dx["icd_version"] == 10) & (dx["icd_clean"].str.len() > 3)
    dx["icd_display"] = dx["icd_code"]  # default: keep original
    dx.loc[is_icd10_long, "icd_display"] = (
        dx.loc[is_icd10_long, "icd_clean"].str[:3] + "." + dx.loc[is_icd10_long, "icd_clean"].str[3:]
    )

    dx["is_depression"] = dx["icd_clean"].str.startswith(depr_prefixes)
    dx["is_fcode"] = (dx["icd_version"] == 10) & dx["icd_clean"].str.startswith(psych_prefix)

    adm_dx = (dx.groupby("hadm_id")
        .agg(
            icd_codes=("icd_display", list),
            has_depression=("is_depression", "any"),
            has_fcode=("is_fcode", "any"),
        ).reset_index()
    )

    print(f"  Admissions with diagnoses: {len(adm_dx):,}")
    print(f"  Admissions with F32/F33:   {adm_dx['has_depression'].sum():,}")
    return adm_dx


def build_admission_active_meds(
    prescriptions: pd.DataFrame,
    adm_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each admission, find medications active at admission time:
        starttime <= admittime <= stoptime

    Returns DataFrame with columns:
        hadm_id, meds (list[str] — lwc drug names)
    """
    rx = prescriptions[["hadm_id", "drug", "starttime", "stoptime"]].copy()
    rx["starttime"] = pd.to_datetime(rx["starttime"], errors="coerce")
    rx["stoptime"] = pd.to_datetime(rx["stoptime"], errors="coerce")

    # Drop rows with missing times
    rx = rx.dropna(subset=["starttime", "stoptime"])

    # Merge admittime
    admittime_lookup = adm_clean[["hadm_id", "admittime"]].drop_duplicates("hadm_id")
    rx = rx.merge(admittime_lookup, on="hadm_id", how="inner")

    # Filter to active at admission (starttime <= admittime <= stoptime)
    rx = rx[(rx["starttime"] <= rx["admittime"]) & (rx["stoptime"] >= rx["admittime"])]

    # lwc drug names, deduplicate per admission
    rx["drug_clean"] = rx["drug"].str.lower().str.strip()
    rx = rx.drop_duplicates(subset=["hadm_id", "drug_clean"])

    adm_meds = (rx.groupby("hadm_id")["drug_clean"]
        .apply(list).reset_index()
        .rename(columns={"drug_clean": "meds"}))

    print(f"  Admissions with active meds: {len(adm_meds):,}")
    return adm_meds


def build_patient_sequences(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    diagnoses: pd.DataFrame,
    prescriptions: pd.DataFrame,
    min_encounters: int,
    readm_window_days: int,
    depression_prefixes: Tuple,
    psych_prefix: str
) -> list[dict]:
    """
    Main extraction file: builds patient sequences w/[readm_window_days]-day psychiatric readmission labels.

    Args:
        admissions:  MIMIC-IV admissions table
        patients:    MIMIC-IV patients table (reserved for future demographics experiment)
        diagnoses:   MIMIC-IV diagnoses_icd table
        prescriptions: MIMIC-IV prescriptions table
        min_encounters: minimum encounters per patient for inclusion
        READM_WINDOW_DAYS: window for psychiatric readmission label

    Returns:
        List of dicts matching the target schema.
    """
    print("=" * 60)
    print("MIMIC-IV Patient Sequence Extraction Pipeline")
    print("=" * 60)

    print("\n[1/5] Cleaning admissions..")
    adm_clean = clean_admissions(admissions)

    print("\n[2/5] Building per-admission ICD codes..")
    adm_dx = build_admission_icd_codes(diagnoses, depression_prefixes, psych_prefix)

    print("\n[3/5] Building per-admission active medications...")
    adm_meds = build_admission_active_meds(prescriptions, adm_clean)

    print("\n[4/5] Building encounters & applying cohort filters...")
    # adm_clean (subject_id, hadm_id, times) + dx + meds
    encounters = (
        adm_clean
        .merge(adm_dx, on="hadm_id", how="left")
        .merge(adm_meds, on="hadm_id", how="left")
    )

    # Fill missing lists (admissions with no diagnoses or no active meds)
    encounters["icd_codes"] = encounters["icd_codes"].apply(
        lambda x: x if isinstance(x, list) else [])
    encounters["meds"] = encounters["meds"].apply(
        lambda x: x if isinstance(x, list) else [])
    encounters["has_depression"] = encounters["has_depression"].fillna(False)
    encounters["has_fcode"] = encounters["has_fcode"].fillna(False)
    
    # Sort by patient and time
    encounters = encounters.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    print(f"  Total encounters: {len(encounters):,}")

    # filter > patients with at least one F32/F33 admission
    depression_patients = set(encounters.loc[encounters["has_depression"], "subject_id"])
    print(f"  Patients with F32/F33: {len(depression_patients):,}")
    encounters = encounters[encounters["subject_id"].isin(depression_patients)]

    # filter > minimum encounters per patient
    enc_per_patient = encounters.groupby("subject_id").size()
    qualifying = set(enc_per_patient[enc_per_patient >= min_encounters].index)
    print(f"  Patients with >= {min_encounters} encounters: {len(qualifying):,}")

    encounters = encounters[encounters["subject_id"].isin(qualifying)].copy()
    print(f"  Final: {len(encounters):,} encounters, "
          f"{encounters['subject_id'].nunique():,} patients")

    # --- Compute labels and assemble sequences ---
    print(f"\n[5/5] Computing {readm_window_days}-day readmission labels...")

    sequences = []
    for subject_id, group in encounters.groupby("subject_id"):
        rows = group.sort_values("admittime").to_dict("records")
        n = len(rows)
        patient_label = 0

        enc_list = []
        for i, row in enumerate(rows):
            enc_list.append({
                "hadm_id": int(row["hadm_id"]),
                "admittime": row["admittime"],
                "dischtime": row["dischtime"],
                "icd_codes": row["icd_codes"],
                "meds": row["meds"],
            })

            # Check for x-day psych readmission after this discharge
            if patient_label == 0 and i < n - 1:
                window_end = row["dischtime"] + timedelta(days=readm_window_days)
                for j in range(i + 1, n):
                    if rows[j]["admittime"] > window_end:
                        break
                    if rows[j]["has_fcode"]:
                        patient_label = 1
                        break

        sequences.append({
            "subject_id": str(subject_id),
            "encounters": enc_list,
            "label": patient_label,
        })

    # --- Summary ---
    n_pos = sum(1 for s in sequences if s["label"] == 1)
    n_neg = len(sequences) - n_pos
    enc_counts = [len(s["encounters"]) for s in sequences]

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete!")
    print(f"  Patients:        {len(sequences):,}")
    print(f"  Label=1 (readm): {n_pos:,} ({100 * n_pos / len(sequences):.1f}%)")
    print(f"  Label=0:         {n_neg:,} ({100 * n_neg / len(sequences):.1f}%)")
    print(f"  Encounters/pt:   mean={np.mean(enc_counts):.1f}, "
          f"median={np.median(enc_counts):.0f}, "
          f"min={min(enc_counts)}, max={max(enc_counts)}")
    print(f"{'=' * 60}")

    return sequences