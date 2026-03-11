"""
MediGuard AI — FWA Detection Pipeline
File: fwa_data_pipeline.py

Layer 1–3: Data Ingestion → Normalization → Rule Engine
Author: MediGuard POC | Healthcare FWA Detection
Requirements: pip install pandas numpy scikit-learn requests faker

Run:
    python fwa_data_pipeline.py

Outputs:
    claims_normalized.csv       — cleaned + enriched claims
    claims_flagged.csv          — rule engine violations
    provider_benchmarks.csv     — provider vs peer benchmarks
"""

import pandas as pd
import numpy as np
import json
import os
import hashlib
import warnings
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────
# REFERENCE DATA — CPT / ICD / NCCI
# ──────────────────────────────────────────────

# Sample CPT code descriptions (real codes, simplified descriptions)
CPT_REFERENCE = {
    "99211": {"desc": "Office visit, minimal complexity", "avg_cost": 25,  "specialty": ["Internal Medicine", "Family Medicine", "Pediatrics"]},
    "99212": {"desc": "Office visit, low complexity",     "avg_cost": 50,  "specialty": ["Internal Medicine", "Family Medicine", "Pediatrics"]},
    "99213": {"desc": "Office visit, moderate complexity","avg_cost": 85,  "specialty": ["Internal Medicine", "Family Medicine", "Pediatrics"]},
    "99214": {"desc": "Office visit, high complexity",    "avg_cost": 130, "specialty": ["Internal Medicine", "Family Medicine", "Cardiology"]},
    "99215": {"desc": "Office visit, highest complexity", "avg_cost": 175, "specialty": ["Internal Medicine", "Cardiology", "Neurology"]},
    "27447": {"desc": "Total knee arthroplasty",          "avg_cost": 8500,"specialty": ["Orthopedic Surgery"]},
    "27130": {"desc": "Total hip arthroplasty",           "avg_cost": 9000,"specialty": ["Orthopedic Surgery"]},
    "93000": {"desc": "Electrocardiogram (EKG)",          "avg_cost": 55,  "specialty": ["Cardiology", "Internal Medicine"]},
    "93306": {"desc": "Echocardiography",                 "avg_cost": 850, "specialty": ["Cardiology"]},
    "36415": {"desc": "Venipuncture",                     "avg_cost": 12,  "specialty": ["Internal Medicine", "Family Medicine", "Pathology"]},
    "36416": {"desc": "Capillary blood draw",             "avg_cost": 8,   "specialty": ["Internal Medicine", "Family Medicine"]},
    "85025": {"desc": "Complete blood count (CBC)",       "avg_cost": 28,  "specialty": ["Internal Medicine", "Family Medicine", "Hematology"]},
    "80053": {"desc": "Comprehensive metabolic panel",    "avg_cost": 35,  "specialty": ["Internal Medicine", "Family Medicine"]},
    "45378": {"desc": "Colonoscopy, diagnostic",          "avg_cost": 1200,"specialty": ["Gastroenterology"]},
    "45380": {"desc": "Colonoscopy with biopsy",          "avg_cost": 1500,"specialty": ["Gastroenterology"]},
    "70553": {"desc": "MRI brain with contrast",          "avg_cost": 1800,"specialty": ["Neurology", "Radiology"]},
    "29881": {"desc": "Knee arthroscopy",                 "avg_cost": 3200,"specialty": ["Orthopedic Surgery"]},
    "97110": {"desc": "Therapeutic exercise",             "avg_cost": 65,  "specialty": ["Physical Therapy"]},
    "97035": {"desc": "Ultrasound therapy",               "avg_cost": 45,  "specialty": ["Physical Therapy"]},
    "90837": {"desc": "Psychotherapy, 60 min",            "avg_cost": 150, "specialty": ["Psychiatry", "Psychology"]},
}

# ICD-10 reference: code → description, valid for which CPTs
ICD_REFERENCE = {
    "M17.11": {"desc": "Primary osteoarthritis, right knee",    "valid_cpts": ["27447", "29881", "97110"]},
    "M17.12": {"desc": "Primary osteoarthritis, left knee",     "valid_cpts": ["27447", "29881", "97110"]},
    "M16.11": {"desc": "Primary osteoarthritis, right hip",     "valid_cpts": ["27130"]},
    "M25.361": {"desc": "Stiffness of right knee",              "valid_cpts": ["97110", "97035", "99213"]},
    "I10":    {"desc": "Essential hypertension",                "valid_cpts": ["99213", "99214", "93000", "80053"]},
    "I25.10": {"desc": "Atherosclerotic heart disease",         "valid_cpts": ["93000", "93306", "99214", "99215"]},
    "Z12.11": {"desc": "Encounter for colorectal cancer screen","valid_cpts": ["45378", "45380"]},
    "K57.30": {"desc": "Diverticulosis of large intestine",     "valid_cpts": ["45378", "45380"]},
    "G43.909": {"desc": "Migraine, unspecified",               "valid_cpts": ["70553", "99214", "99215"]},
    "F32.1":  {"desc": "Major depressive disorder, moderate",  "valid_cpts": ["90837", "99214"]},
    "E11.9":  {"desc": "Type 2 diabetes without complications", "valid_cpts": ["99213", "99214", "80053", "85025"]},
    "J06.9":  {"desc": "Acute upper respiratory infection",     "valid_cpts": ["99212", "99213"]},
    "R00.0":  {"desc": "Tachycardia, unspecified",             "valid_cpts": ["93000", "99213", "99214"]},
}

# NCCI bundling rules: if CPT_A is billed with CPT_B on same day → violation
# Format: (code_to_keep, code_that_is_bundled_into_it)
NCCI_BUNDLES = [
    ("36415", "36416"),   # venipuncture subsumes capillary draw
    ("45380", "45378"),   # colonoscopy w/ biopsy subsumes diagnostic colonoscopy
    ("99215", "99211"),   # high complexity subsumes minimal
    ("99215", "99212"),
    ("99214", "99212"),
    ("93306", "93000"),   # echo subsumes EKG in same encounter
]

# OIG Medically Unlikely Edits: max units per day per CPT
MUE_LIMITS = {
    "93000": 1,   # One EKG per day
    "27447": 1,   # One knee replacement per day
    "27130": 1,
    "45378": 1,
    "90837": 4,   # Max 4 psychotherapy sessions per day
    "97110": 8,   # Max 8 therapy units per day
    "99215": 1,   # One highest-level E&M per day per provider
}

SPECIALTIES = list({sp for v in CPT_REFERENCE.values() for sp in v["specialty"]})
STATES = ["TX", "CA", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]
PLACE_OF_SERVICE = {11: "Office", 21: "Inpatient Hospital", 22: "Outpatient Hospital", 23: "ER", 31: "SNF"}


# ──────────────────────────────────────────────
# LAYER 1: DATA GENERATION (Synthetic Claims)
# ──────────────────────────────────────────────

def generate_npi(seed_val: int) -> str:
    """Generate a realistic 10-digit NPI."""
    np.random.seed(seed_val)
    return "1" + "".join([str(np.random.randint(0, 10)) for _ in range(9)])

def generate_synthetic_claims(n_claims: int = 500, inject_fraud: bool = True) -> pd.DataFrame:
    """
    Generate synthetic healthcare claims with realistic patterns.
    Optionally inject known FWA patterns for testing.
    """
    print(f"\n[1/6] Generating {n_claims} synthetic claims...")

    providers = []
    for i in range(30):
        specialty = np.random.choice(SPECIALTIES)
        # Some providers will be "fraudulent" — assigned later
        providers.append({
            "npi": generate_npi(i * 100),
            "name": f"Dr. Provider_{i+1:03d}",
            "specialty": specialty,
            "state": np.random.choice(STATES),
            "is_fraud_provider": False
        })

    # Mark 3 providers as fraudulent for injection
    fraud_provider_indices = [2, 7, 15]
    for idx in fraud_provider_indices:
        providers[idx]["is_fraud_provider"] = True

    cpt_codes = list(CPT_REFERENCE.keys())
    icd_codes = list(ICD_REFERENCE.keys())

    claims = []
    base_date = datetime(2024, 1, 1)

    for i in range(n_claims):
        provider = np.random.choice(providers)
        specialty = provider["specialty"]

        # Pick CPT codes appropriate for this specialty
        valid_cpts = [c for c, v in CPT_REFERENCE.items() if specialty in v["specialty"]]
        if not valid_cpts:
            valid_cpts = ["99213"]
        cpt = np.random.choice(valid_cpts)

        # Pick matching ICD
        valid_icds = [c for c, v in ICD_REFERENCE.items() if cpt in v["valid_cpts"]]
        if not valid_icds:
            valid_icds = list(ICD_REFERENCE.keys())[:3]
        icd = np.random.choice(valid_icds)

        dos = base_date + timedelta(days=np.random.randint(0, 365))
        avg_cost = CPT_REFERENCE[cpt]["avg_cost"]
        billed = round(avg_cost * np.random.uniform(0.8, 1.3), 2)
        units = 1
        pos = np.random.choice(list(PLACE_OF_SERVICE.keys()))
        age = np.random.randint(18, 90)
        gender = np.random.choice(["M", "F"])

        claim = {
            "claim_id": f"CLM-2024-{i+1:05d}",
            "patient_id": f"PAT-{np.random.randint(1000, 9999)}",
            "patient_age": age,
            "patient_gender": gender,
            "provider_npi": provider["npi"],
            "provider_name": provider["name"],
            "provider_specialty": specialty,
            "provider_state": provider["state"],
            "date_of_service": dos.strftime("%Y-%m-%d"),
            "cpt_code": cpt,
            "cpt_description": CPT_REFERENCE[cpt]["desc"],
            "icd_primary": icd,
            "icd_description": ICD_REFERENCE[icd]["desc"],
            "billed_amount": billed,
            "units": units,
            "place_of_service": pos,
            "pos_description": PLACE_OF_SERVICE[pos],
            "_is_fraud_provider": provider["is_fraud_provider"],
            "fraud_label": "CLEAN"
        }
        claims.append(claim)

    df = pd.DataFrame(claims)

    # ── INJECT FRAUD PATTERNS ──────────────────
    if inject_fraud:
        df = inject_fwa_patterns(df, providers)

    print(f"    ✓ Generated {len(df)} claims across {len(providers)} providers")
    fraud_count = len(df[df["fraud_label"] != "CLEAN"])
    print(f"    ✓ Injected {fraud_count} fraudulent/suspicious claims ({fraud_count/len(df)*100:.1f}%)")
    return df


def inject_fwa_patterns(df: pd.DataFrame, providers: list) -> pd.DataFrame:
    """Inject realistic FWA patterns into the dataset."""
    print("    → Injecting FWA patterns...")

    fraud_records = []
    base_date = datetime(2024, 3, 1)

    # Pattern 1: UPCODING — provider bills 99215 for everything
    fraud_npi = [p["npi"] for p in providers if p["is_fraud_provider"]][0]
    for i in range(50):
        dos = base_date + timedelta(days=np.random.randint(0, 180))
        fraud_records.append({
            "claim_id": f"CLM-FRAUD-UP-{i+1:04d}",
            "patient_id": f"PAT-{np.random.randint(5000, 6000)}",
            "patient_age": np.random.randint(30, 70),
            "patient_gender": np.random.choice(["M", "F"]),
            "provider_npi": fraud_npi,
            "provider_name": "Dr. Fraud_Upcoder",
            "provider_specialty": "Internal Medicine",
            "provider_state": "TX",
            "date_of_service": dos.strftime("%Y-%m-%d"),
            "cpt_code": "99215",
            "cpt_description": CPT_REFERENCE["99215"]["desc"],
            "icd_primary": "J06.9",  # Simple cold → highest complexity visit
            "icd_description": ICD_REFERENCE["J06.9"]["desc"],
            "billed_amount": round(CPT_REFERENCE["99215"]["avg_cost"] * np.random.uniform(1.0, 1.2), 2),
            "units": 1,
            "place_of_service": 11,
            "pos_description": "Office",
            "_is_fraud_provider": True,
            "fraud_label": "UPCODING"
        })

    # Pattern 2: UNBUNDLING — billing 36415 + 36416 together
    fraud_npi2 = [p["npi"] for p in providers if p["is_fraud_provider"]][1]
    for i in range(30):
        dos = base_date + timedelta(days=np.random.randint(0, 180))
        base_claim = {
            "patient_id": f"PAT-{np.random.randint(6000, 7000)}",
            "patient_age": np.random.randint(40, 80),
            "patient_gender": np.random.choice(["M", "F"]),
            "provider_npi": fraud_npi2,
            "provider_name": "Dr. Fraud_Unbundler",
            "provider_specialty": "Internal Medicine",
            "provider_state": "FL",
            "date_of_service": dos.strftime("%Y-%m-%d"),
            "icd_primary": "I10",
            "icd_description": ICD_REFERENCE["I10"]["desc"],
            "units": 1,
            "place_of_service": 11,
            "pos_description": "Office",
            "_is_fraud_provider": True,
            "fraud_label": "UNBUNDLING"
        }
        for cpt in ["36415", "36416"]:
            r = base_claim.copy()
            r["claim_id"] = f"CLM-FRAUD-UNB-{i+1:04d}-{cpt}"
            r["cpt_code"] = cpt
            r["cpt_description"] = CPT_REFERENCE[cpt]["desc"]
            r["billed_amount"] = round(CPT_REFERENCE[cpt]["avg_cost"] * 1.1, 2)
            fraud_records.append(r)

    # Pattern 3: ICD-CPT MISMATCH — knee surgery billed without orthopedic ICD
    fraud_npi3 = [p["npi"] for p in providers if p["is_fraud_provider"]][2]
    for i in range(20):
        dos = base_date + timedelta(days=np.random.randint(0, 180))
        fraud_records.append({
            "claim_id": f"CLM-FRAUD-MIS-{i+1:04d}",
            "patient_id": f"PAT-{np.random.randint(7000, 8000)}",
            "patient_age": np.random.randint(50, 80),
            "patient_gender": np.random.choice(["M", "F"]),
            "provider_npi": fraud_npi3,
            "provider_name": "Dr. Fraud_Mismatch",
            "provider_specialty": "Orthopedic Surgery",
            "provider_state": "CA",
            "date_of_service": dos.strftime("%Y-%m-%d"),
            "cpt_code": "27447",
            "cpt_description": CPT_REFERENCE["27447"]["desc"],
            "icd_primary": "M25.361",  # Knee stiffness — NOT sufficient for total knee replacement
            "icd_description": ICD_REFERENCE["M25.361"]["desc"],
            "billed_amount": round(CPT_REFERENCE["27447"]["avg_cost"] * np.random.uniform(0.9, 1.1), 2),
            "units": 1,
            "place_of_service": 21,
            "pos_description": "Inpatient Hospital",
            "_is_fraud_provider": True,
            "fraud_label": "ICD_CPT_MISMATCH"
        })

    # Pattern 4: DUPLICATE CLAIMS — same claim slightly varied
    for i in range(20):
        dos = (base_date + timedelta(days=np.random.randint(0, 180))).strftime("%Y-%m-%d")
        pat = f"PAT-{np.random.randint(8000, 9000)}"
        base_amt = CPT_REFERENCE["93000"]["avg_cost"]
        for j, amt_delta in enumerate([0, np.random.uniform(1, 5)]):
            fraud_records.append({
                "claim_id": f"CLM-FRAUD-DUP-{i+1:04d}-{j+1}",
                "patient_id": pat,
                "patient_age": 62,
                "patient_gender": "M",
                "provider_npi": providers[5]["npi"],
                "provider_name": providers[5]["name"],
                "provider_specialty": "Cardiology",
                "provider_state": "NY",
                "date_of_service": dos,
                "cpt_code": "93000",
                "cpt_description": CPT_REFERENCE["93000"]["desc"],
                "icd_primary": "I25.10",
                "icd_description": ICD_REFERENCE["I25.10"]["desc"],
                "billed_amount": round(base_amt + amt_delta, 2),
                "units": 1,
                "place_of_service": 11,
                "pos_description": "Office",
                "_is_fraud_provider": False,
                "fraud_label": "DUPLICATE" if j == 1 else "CLEAN"
            })

    fraud_df = pd.DataFrame(fraud_records)
    return pd.concat([df, fraud_df], ignore_index=True)


# ──────────────────────────────────────────────
# LAYER 2: NORMALIZATION & FEATURE ENGINEERING
# ──────────────────────────────────────────────

def normalize_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw claims and engineer features for downstream FWA detection.
    """
    print("\n[2/6] Normalizing and enriching claims...")

    # Standardize data types
    df["date_of_service"] = pd.to_datetime(df["date_of_service"])
    df["billed_amount"] = pd.to_numeric(df["billed_amount"], errors="coerce").fillna(0)
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(1).astype(int)
    df["cpt_code"] = df["cpt_code"].astype(str).str.strip()
    df["icd_primary"] = df["icd_primary"].astype(str).str.strip()
    df["provider_npi"] = df["provider_npi"].astype(str).str.strip()

    # Feature: Day of week (weekend billing flag)
    df["day_of_week"] = df["date_of_service"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Feature: ICD-CPT compatibility flag
    df["icd_cpt_compatible"] = df.apply(
        lambda r: 1 if (
            r["icd_primary"] in ICD_REFERENCE and
            r["cpt_code"] in ICD_REFERENCE.get(r["icd_primary"], {}).get("valid_cpts", [])
        ) else 0, axis=1
    )

    # Feature: CPT expected cost benchmark
    df["cpt_benchmark_cost"] = df["cpt_code"].map(
        lambda c: CPT_REFERENCE.get(c, {}).get("avg_cost", 100)
    )
    df["cost_ratio_vs_benchmark"] = (df["billed_amount"] / df["cpt_benchmark_cost"]).round(3)

    # Feature: CPT specialty mismatch
    df["specialty_mismatch"] = df.apply(
        lambda r: 0 if (
            r["cpt_code"] in CPT_REFERENCE and
            r["provider_specialty"] in CPT_REFERENCE.get(r["cpt_code"], {}).get("specialty", [])
        ) else 1, axis=1
    )

    # Feature: Provider-level claim frequency
    provider_freq = df.groupby("provider_npi").size().reset_index(name="provider_claim_count")
    df = df.merge(provider_freq, on="provider_npi", how="left")

    # Feature: Provider avg billed amount per CPT
    provider_cpt_avg = df.groupby(["provider_npi", "cpt_code"])["billed_amount"].mean().reset_index()
    provider_cpt_avg.columns = ["provider_npi", "cpt_code", "provider_cpt_avg"]
    df = df.merge(provider_cpt_avg, on=["provider_npi", "cpt_code"], how="left")

    # Feature: Provider vs specialty benchmark ratio
    specialty_cpt_avg = df.groupby(["provider_specialty", "cpt_code"])["billed_amount"].mean().reset_index()
    specialty_cpt_avg.columns = ["provider_specialty", "cpt_code", "specialty_cpt_avg"]
    df = df.merge(specialty_cpt_avg, on=["provider_specialty", "cpt_code"], how="left")
    df["provider_vs_specialty_ratio"] = (df["provider_cpt_avg"] / df["specialty_cpt_avg"]).round(3)

    # Feature: E&M upcoding proxy (% of 99215 per provider)
    em_codes = ["99211", "99212", "99213", "99214", "99215"]
    em_claims = df[df["cpt_code"].isin(em_codes)].copy()
    if not em_claims.empty:
        em_total = em_claims.groupby("provider_npi").size().reset_index(name="em_total")
        em_top = em_claims[em_claims["cpt_code"] == "99215"].groupby("provider_npi").size().reset_index(name="em_99215")
        em_ratio = em_total.merge(em_top, on="provider_npi", how="left").fillna(0)
        em_ratio["pct_99215"] = (em_ratio["em_99215"] / em_ratio["em_total"] * 100).round(1)
        df = df.merge(em_ratio[["provider_npi", "pct_99215"]], on="provider_npi", how="left")
    else:
        df["pct_99215"] = 0

    df["pct_99215"] = df["pct_99215"].fillna(0)

    # Feature: unique patients per provider
    unique_pts = df.groupby("provider_npi")["patient_id"].nunique().reset_index(name="unique_patients")
    df = df.merge(unique_pts, on="provider_npi", how="left")

    # Feature: weekend billing rate per provider
    weekend_rate = df.groupby("provider_npi")["is_weekend"].mean().reset_index(name="weekend_billing_rate")
    df = df.merge(weekend_rate, on="provider_npi", how="left")
    df["weekend_billing_rate"] = (df["weekend_billing_rate"] * 100).round(1)

    # Feature: claim hash for duplicate detection
    df["claim_hash"] = df.apply(
        lambda r: hashlib.md5(
            f"{r['patient_id']}{r['provider_npi']}{r['cpt_code']}{r['date_of_service'].date()}".encode()
        ).hexdigest()[:12], axis=1
    )
    hash_counts = df.groupby("claim_hash").size().reset_index(name="hash_duplicate_count")
    df = df.merge(hash_counts, on="claim_hash", how="left")

    print(f"    ✓ Enriched {len(df)} claims with {len(df.columns)} features")
    return df


# ──────────────────────────────────────────────
# LAYER 3: RULE ENGINE (OIG / NCCI / MUE)
# ──────────────────────────────────────────────

class FWARuleEngine:
    """
    Hard-coded rule engine based on CMS OIG guidelines, NCCI edits, and MUEs.
    Returns flags for each claim.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.flags = defaultdict(list)  # claim_id → list of rule violations

    def run_all_rules(self) -> pd.DataFrame:
        print("\n[3/6] Running OIG/NCCI rule engine...")
        self._rule_icd_cpt_mismatch()
        self._rule_ncci_unbundling()
        self._rule_mue_limits()
        self._rule_upcoding_proxy()
        self._rule_duplicate_claims()
        self._rule_specialty_mismatch()
        self._rule_weekend_outlier()
        self._rule_cost_outlier()
        return self._compile_results()

    def _rule_icd_cpt_mismatch(self):
        """Flag claims where ICD-10 doesn't justify the CPT procedure."""
        flagged = self.df[self.df["icd_cpt_compatible"] == 0]
        for _, row in flagged.iterrows():
            self.flags[row["claim_id"]].append({
                "rule": "ICD_CPT_MISMATCH",
                "severity": "HIGH",
                "detail": f"ICD {row['icd_primary']} ({row['icd_description']}) does not justify CPT {row['cpt_code']} ({row['cpt_description']})"
            })
        print(f"    → ICD-CPT mismatch: {len(flagged)} claims")

    def _rule_ncci_unbundling(self):
        """Flag claims where bundled CPT codes are billed separately on same day."""
        grouped = self.df.groupby(["patient_id", "provider_npi", "date_of_service"])["cpt_code"].apply(list)
        unbundled_patients = set()

        for (patient, npi, dos), cpts in grouped.items():
            cpt_set = set(cpts)
            for keep_code, bundled_code in NCCI_BUNDLES:
                if keep_code in cpt_set and bundled_code in cpt_set:
                    unbundled_patients.add((patient, npi, str(dos.date())))
                    # Flag the bundled (redundant) claim
                    mask = (
                        (self.df["patient_id"] == patient) &
                        (self.df["provider_npi"] == npi) &
                        (self.df["date_of_service"] == dos) &
                        (self.df["cpt_code"] == bundled_code)
                    )
                    for _, row in self.df[mask].iterrows():
                        self.flags[row["claim_id"]].append({
                            "rule": "NCCI_UNBUNDLING",
                            "severity": "HIGH",
                            "detail": f"CPT {bundled_code} is bundled into CPT {keep_code} per NCCI edits. Cannot bill separately."
                        })

        print(f"    → NCCI unbundling: {len(unbundled_patients)} patient-provider-DOS groups")

    def _rule_mue_limits(self):
        """Flag claims exceeding CMS Medically Unlikely Edit (MUE) unit limits."""
        daily_units = self.df.groupby(["patient_id", "provider_npi", "date_of_service", "cpt_code"])["units"].sum()
        count = 0
        for (patient, npi, dos, cpt), total_units in daily_units.items():
            if cpt in MUE_LIMITS and total_units > MUE_LIMITS[cpt]:
                mask = (
                    (self.df["patient_id"] == patient) &
                    (self.df["provider_npi"] == npi) &
                    (self.df["date_of_service"] == dos) &
                    (self.df["cpt_code"] == cpt)
                )
                for _, row in self.df[mask].iterrows():
                    self.flags[row["claim_id"]].append({
                        "rule": "MUE_EXCEEDED",
                        "severity": "MEDIUM",
                        "detail": f"CPT {cpt} billed {total_units} units. CMS MUE limit is {MUE_LIMITS[cpt]} unit(s) per day."
                    })
                count += 1
        print(f"    → MUE violations: {count} CPT-day combinations")

    def _rule_upcoding_proxy(self):
        """Flag providers with >80% E&M visits at highest complexity level."""
        flagged_providers = self.df[self.df["pct_99215"] > 80]["provider_npi"].unique()
        mask = (self.df["provider_npi"].isin(flagged_providers)) & (self.df["cpt_code"] == "99215")
        count = 0
        for _, row in self.df[mask].iterrows():
            self.flags[row["claim_id"]].append({
                "rule": "UPCODING_PROXY",
                "severity": "HIGH",
                "detail": f"Provider bills 99215 for {row['pct_99215']:.0f}% of E&M visits vs ~18% national avg. Pattern consistent with systematic upcoding."
            })
            count += 1
        print(f"    → Upcoding proxy flags: {count} claims across {len(flagged_providers)} providers")

    def _rule_duplicate_claims(self):
        """Flag exact or near-duplicate claims (same patient/provider/CPT/DOS)."""
        mask = self.df["hash_duplicate_count"] > 1
        count = 0
        for _, row in self.df[mask].iterrows():
            self.flags[row["claim_id"]].append({
                "rule": "DUPLICATE_CLAIM",
                "severity": "HIGH",
                "detail": f"Claim hash matches {row['hash_duplicate_count']-1} other claim(s) for same patient/provider/CPT/DOS. Potential duplicate billing."
            })
            count += 1
        print(f"    → Duplicate claim flags: {count} claims")

    def _rule_specialty_mismatch(self):
        """Flag claims where provider specialty doesn't match CPT expectations."""
        mask = self.df["specialty_mismatch"] == 1
        count = 0
        for _, row in self.df[mask].iterrows():
            expected = CPT_REFERENCE.get(row["cpt_code"], {}).get("specialty", ["Unknown"])
            self.flags[row["claim_id"]].append({
                "rule": "SPECIALTY_MISMATCH",
                "severity": "MEDIUM",
                "detail": f"CPT {row['cpt_code']} typically billed by {expected}. Provider specialty: {row['provider_specialty']}."
            })
            count += 1
        print(f"    → Specialty mismatch flags: {count} claims")

    def _rule_weekend_outlier(self):
        """Flag providers with unusually high weekend/holiday billing rates."""
        mask = self.df["weekend_billing_rate"] > 40  # >40% weekend billing is anomalous
        flagged = self.df[mask]["provider_npi"].unique()
        count = 0
        for npi in flagged:
            npi_claims = self.df[(self.df["provider_npi"] == npi) & (self.df["is_weekend"] == 1)]
            rate = self.df[self.df["provider_npi"] == npi]["weekend_billing_rate"].iloc[0]
            for _, row in npi_claims.iterrows():
                self.flags[row["claim_id"]].append({
                    "rule": "WEEKEND_OUTLIER",
                    "severity": "LOW",
                    "detail": f"Provider has {rate:.1f}% weekend billing rate vs ~15% national avg. High weekend billing may indicate falsified dates."
                })
                count += 1
        print(f"    → Weekend outlier flags: {count} claims across {len(flagged)} providers")

    def _rule_cost_outlier(self):
        """Flag claims billed >3x the CPT benchmark."""
        mask = self.df["cost_ratio_vs_benchmark"] > 3.0
        count = 0
        for _, row in self.df[mask].iterrows():
            self.flags[row["claim_id"]].append({
                "rule": "COST_OUTLIER",
                "severity": "MEDIUM",
                "detail": f"Billed ${row['billed_amount']:.2f} vs CPT benchmark ${row['cpt_benchmark_cost']:.2f} ({row['cost_ratio_vs_benchmark']:.1f}x). Significant overbilling."
            })
            count += 1
        print(f"    → Cost outlier flags: {count} claims")

    def _compile_results(self) -> pd.DataFrame:
        """Compile all flags into the main dataframe."""
        self.df["rule_flags"] = self.df["claim_id"].map(
            lambda cid: json.dumps(self.flags.get(cid, []))
        )
        self.df["rule_flag_count"] = self.df["claim_id"].map(
            lambda cid: len(self.flags.get(cid, []))
        )
        self.df["rule_flag_severity"] = self.df["claim_id"].map(
            lambda cid: (
                "HIGH" if any(f["severity"] == "HIGH" for f in self.flags.get(cid, [])) else
                "MEDIUM" if any(f["severity"] == "MEDIUM" for f in self.flags.get(cid, [])) else
                "LOW" if self.flags.get(cid) else "CLEAN"
            )
        )

        flagged_count = len(self.df[self.df["rule_flag_count"] > 0])
        print(f"\n    ✓ Rule engine complete: {flagged_count} claims flagged ({flagged_count/len(self.df)*100:.1f}%)")
        return self.df


# ──────────────────────────────────────────────
# LAYER 4: ANOMALY SCORING (Statistical)
# ──────────────────────────────────────────────

def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolation Forest anomaly detection on provider-level features.
    Assigns anomaly score to each claim.
    """
    print("\n[4/6] Running anomaly detection (Isolation Forest)...")

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        features = [
            "cost_ratio_vs_benchmark",
            "provider_vs_specialty_ratio",
            "pct_99215",
            "weekend_billing_rate",
            "specialty_mismatch",
            "icd_cpt_compatible",
            "hash_duplicate_count"
        ]

        feature_df = df[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df)

        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.15,  # Expect ~15% anomalies
            random_state=RANDOM_SEED
        )
        df["anomaly_score_raw"] = iso_forest.fit_predict(X_scaled)
        df["anomaly_decision_score"] = iso_forest.decision_function(X_scaled)

        # Normalize to 0–100 risk score (higher = more anomalous)
        min_score = df["anomaly_decision_score"].min()
        max_score = df["anomaly_decision_score"].max()
        df["ml_risk_score"] = (
            100 - ((df["anomaly_decision_score"] - min_score) / (max_score - min_score) * 100)
        ).round(1)

        # Composite risk: combine rule flags + ML score
        df["composite_risk_score"] = (
            df["ml_risk_score"] * 0.5 +
            df["rule_flag_count"] * 15 +
            df["specialty_mismatch"] * 10 +
            (1 - df["icd_cpt_compatible"]) * 20
        ).clip(0, 100).round(1)

        anomaly_count = len(df[df["anomaly_score_raw"] == -1])
        print(f"    ✓ Isolation Forest complete: {anomaly_count} anomalies detected")

    except ImportError:
        print("    ⚠ scikit-learn not installed. Run: pip install scikit-learn")
        df["ml_risk_score"] = 50.0
        df["composite_risk_score"] = (df["rule_flag_count"] * 15).clip(0, 100)

    return df


# ──────────────────────────────────────────────
# LAYER 5: PROVIDER BENCHMARKS
# ──────────────────────────────────────────────

def build_provider_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-provider summary for audit prioritization."""
    print("\n[5/6] Building provider benchmark report...")

    provider_summary = df.groupby(["provider_npi", "provider_name", "provider_specialty", "provider_state"]).agg(
        total_claims=("claim_id", "count"),
        total_billed=("billed_amount", "sum"),
        avg_billed=("billed_amount", "mean"),
        unique_patients=("patient_id", "nunique"),
        pct_99215=("pct_99215", "first"),
        weekend_billing_rate=("weekend_billing_rate", "first"),
        high_risk_claims=("composite_risk_score", lambda x: (x >= 70).sum()),
        flagged_claims=("rule_flag_count", lambda x: (x > 0).sum()),
        avg_composite_risk=("composite_risk_score", "mean"),
        max_composite_risk=("composite_risk_score", "max"),
    ).reset_index()

    provider_summary["flag_rate_pct"] = (
        provider_summary["flagged_claims"] / provider_summary["total_claims"] * 100
    ).round(1)
    provider_summary["avg_billed"] = provider_summary["avg_billed"].round(2)
    provider_summary["total_billed"] = provider_summary["total_billed"].round(2)
    provider_summary["avg_composite_risk"] = provider_summary["avg_composite_risk"].round(1)

    provider_summary = provider_summary.sort_values("avg_composite_risk", ascending=False)

    print(f"    ✓ Provider benchmarks built for {len(provider_summary)} providers")
    return provider_summary


# ──────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, provider_df: pd.DataFrame):
    print("\n[6/6] Saving outputs...")

    # Full normalized claims
    claims_out = df.drop(columns=["_is_fraud_provider"], errors="ignore")
    claims_path = os.path.join(OUTPUT_DIR, "claims_normalized.csv")
    claims_out.to_csv(claims_path, index=False)
    print(f"    ✓ claims_normalized.csv ({len(claims_out)} rows)")

    # Flagged claims only (for LangChain AI layer)
    flagged = df[df["rule_flag_count"] > 0].drop(columns=["_is_fraud_provider"], errors="ignore")
    flagged_path = os.path.join(OUTPUT_DIR, "claims_flagged.csv")
    flagged.to_csv(flagged_path, index=False)
    print(f"    ✓ claims_flagged.csv ({len(flagged)} rows — input to AI reasoning layer)")

    # Provider benchmarks
    prov_path = os.path.join(OUTPUT_DIR, "provider_benchmarks.csv")
    provider_df.to_csv(prov_path, index=False)
    print(f"    ✓ provider_benchmarks.csv ({len(provider_df)} providers)")

    # High-priority claims for AI review (risk > 70)
    ai_queue = df[df["composite_risk_score"] >= 70].sort_values("composite_risk_score", ascending=False)
    ai_queue_cols = [
        "claim_id", "patient_id", "patient_age", "patient_gender",
        "provider_npi", "provider_name", "provider_specialty", "provider_state",
        "date_of_service", "cpt_code", "cpt_description",
        "icd_primary", "icd_description", "billed_amount",
        "units", "pos_description", "composite_risk_score",
        "rule_flag_count", "rule_flag_severity", "rule_flags",
        "pct_99215", "weekend_billing_rate", "provider_vs_specialty_ratio",
        "fraud_label"
    ]
    ai_queue = ai_queue[[c for c in ai_queue_cols if c in ai_queue.columns]]
    ai_path = os.path.join(OUTPUT_DIR, "ai_review_queue.csv")
    ai_queue.to_csv(ai_path, index=False)
    print(f"    ✓ ai_review_queue.csv ({len(ai_queue)} high-priority claims → feed to LangChain layer)")

    return claims_path, flagged_path, prov_path, ai_path


def print_summary(df: pd.DataFrame, provider_df: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  MEDIAGUARD AI — PIPELINE SUMMARY")
    print("═" * 60)
    print(f"  Total claims processed  : {len(df):,}")
    print(f"  Rule-flagged claims     : {len(df[df['rule_flag_count'] > 0]):,} ({len(df[df['rule_flag_count'] > 0])/len(df)*100:.1f}%)")
    print(f"  High-risk (score ≥70)   : {len(df[df['composite_risk_score'] >= 70]):,}")
    print(f"  Providers analyzed      : {len(provider_df):,}")
    print(f"  Avg composite risk      : {df['composite_risk_score'].mean():.1f}/100")

    print("\n  TOP 5 HIGH-RISK PROVIDERS:")
    top5 = provider_df.head(5)[["provider_name", "provider_specialty", "flag_rate_pct", "avg_composite_risk"]]
    for _, row in top5.iterrows():
        print(f"    {row['provider_name']:<25} | {row['provider_specialty']:<22} | "
              f"Flag rate: {row['flag_rate_pct']:.0f}% | Avg risk: {row['avg_composite_risk']:.0f}/100")

    print("\n  FLAG TYPE BREAKDOWN:")
    all_flags = []
    for flags_json in df["rule_flags"]:
        try:
            all_flags.extend(json.loads(flags_json))
        except Exception:
            pass
    from collections import Counter
    rule_counts = Counter(f["rule"] for f in all_flags)
    for rule, count in rule_counts.most_common():
        print(f"    {rule:<30} : {count:,} claims")

    print("\n  OUTPUT FILES → ./output/")
    print("    claims_normalized.csv     — full enriched claims")
    print("    claims_flagged.csv        — rule-violated claims")
    print("    provider_benchmarks.csv   — provider risk summary")
    print("    ai_review_queue.csv       — HIGH PRIORITY → feed to LangChain AI layer")
    print("═" * 60)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║  MEDIAGUARD AI — FWA Data Pipeline       ║")
    print("║  Healthcare Fraud, Waste & Abuse POC     ║")
    print("╚══════════════════════════════════════════╝")

    # Layer 1: Generate / Load data
    df = generate_synthetic_claims(n_claims=500, inject_fraud=True)

    # Layer 2: Normalize + feature engineering
    df = normalize_and_enrich(df)

    # Layer 3: Rule engine (OIG / NCCI / MUE)
    engine = FWARuleEngine(df)
    df = engine.run_all_rules()

    # Layer 4: Anomaly detection (Isolation Forest)
    df = run_anomaly_detection(df)

    # Layer 5: Provider benchmarks
    provider_df = build_provider_benchmarks(df)

    # Layer 6: Save all outputs
    save_outputs(df, provider_df)

    # Print summary
    print_summary(df, provider_df)

    print("\n✅ Pipeline complete. Run fwa_langchain_reasoning.py next for AI explanations.")
