"""
╔══════════════════════════════════════════════════════════════════╗
║  MEDIAGUARD AI — CMS SynPUF Carrier Claims Loader                ║
║  Converts CMS DE-SynPUF format → MediGuard pipeline CSV          ║
╚══════════════════════════════════════════════════════════════════╝

PURPOSE:
  Reads the CMS Medicare Claims Synthetic Public Use File (DE-SynPUF)
  Carrier Claims CSV and converts it into the normalized format that
  fwa_data_pipeline.py expects — same columns as fhir_converted_claims.csv.

  This is Phase 2 data — real CMS Medicare claim structure with
  synthetic (de-identified) beneficiary data. No BAA required.

INPUT FILE:
  data/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A
  Source: data.cms.gov/medicare-claims-synthetic-public-use-files

SYNPUF FORMAT NOTES:
  - Wide format: up to 13 line items per claim row (HCPCS_CD_1..13)
  - ICD-9 codes (not ICD-10) — file is from 2008-2010
  - Each row = one claim with multiple procedure lines
  - We expand each non-empty HCPCS line into a separate pipeline row
  - First HCPCS line = primary CPT, others = additional_cpt

OUTPUT:
  output/synpuf_converted_claims.csv  → feeds into fwa_data_pipeline.py

USAGE:
  python cms_synpuf_loader.py

  Or import into fwa_data_pipeline.py:
    from cms_synpuf_loader import load_synpuf_claims
    df = load_synpuf_claims(max_claims=5000)

SYNPUF COLUMN MAPPING:
  DESYNPUF_ID          → patient_id
  CLM_ID               → claim_id
  CLM_FROM_DT          → service_date
  ICD9_DGNS_CD_1       → icd_code  (ICD-9 — noted in conversion)
  HCPCS_CD_1           → cpt_code  (primary procedure)
  HCPCS_CD_2..13       → additional_cpt (comma-separated)
  PRF_PHYSN_NPI_1      → provider_npi
  LINE_NCH_PMT_AMT_1   → billed_amount (Medicare payment amount)
  LINE_ALOWD_CHRG_AMT_1→ allowed_amount
"""

import os
import pandas as pd
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
_DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

SYNPUF_FILE    = os.path.join(_DATA_DIR, "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.CSV")
OUTPUT_CSV     = os.path.join(_OUTPUT_DIR, "synpuf_converted_claims.csv")

# Number of HCPCS/NPI/amount line columns in SynPUF (up to 13)
MAX_LINES = 13

# Output columns — matches fhir_converted_claims.csv exactly
OUTPUT_COLUMNS = [
    "claim_id", "patient_id", "provider_npi", "provider_name",
    "provider_specialty", "provider_state", "payer_id", "payer_name",
    "plan_name", "cpt_code", "cpt_description", "icd_code", "icd_description",
    "additional_cpt", "billed_amount", "service_date", "place_of_service",
    "pos_description", "pos_source", "claim_type", "source_format",
    "converted_at", "raw_fhir_id", "conversion_notes", "cpt_expected_specialty",
    "fraud_type"
]


# ── DATE PARSER ───────────────────────────────────────────────────────────────

def parse_synpuf_date(date_val):
    """
    Convert SynPUF date format (YYYYMMDD integer) to YYYY-MM-DD string.
    SynPUF stores dates as integers e.g. 20090725 → '2009-07-25'
    """
    try:
        date_str = str(int(float(str(date_val)))).strip()
        if len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    except (ValueError, TypeError):
        pass
    return ""


# ── PLACE OF SERVICE INFERENCE ────────────────────────────────────────────────

def infer_pos_from_cpt(cpt_code):
    """
    Infer Place of Service from CPT code range.
    SynPUF uses HCPCS codes which overlap with CPT ranges.
    Same logic as fhir_converter.py Layer 2.
    """
    try:
        cpt_int = int(str(cpt_code).replace(".", "").strip())
    except (ValueError, TypeError):
        return 11, "Office", "default (non-numeric)"

    if 99201 <= cpt_int <= 99215:
        return 11, "Office", "CPT range: office E&M"
    if 99221 <= cpt_int <= 99239:
        return 21, "Inpatient Hospital", "CPT range: inpatient care"
    if 99241 <= cpt_int <= 99255:
        return 22, "Outpatient Hospital", "CPT range: outpatient consult"
    if 99281 <= cpt_int <= 99288:
        return 23, "Emergency Room", "CPT range: ED"
    if 99304 <= cpt_int <= 99318:
        return 31, "Skilled Nursing Facility", "CPT range: SNF"
    if 10000 <= cpt_int <= 29999:
        return 22, "Outpatient Hospital", "CPT range: outpatient surgery"
    if 30000 <= cpt_int <= 69999:
        return 21, "Inpatient Hospital", "CPT range: major surgery"
    if 70000 <= cpt_int <= 79999:
        return 22, "Outpatient Hospital", "CPT range: radiology"
    if 80000 <= cpt_int <= 89999:
        return 11, "Office", "CPT range: lab/pathology"
    if 90000 <= cpt_int <= 99199:
        return 11, "Office", "CPT range: medicine"

    return 11, "Office", "default (unmatched range)"


# ── MAIN LOADER ───────────────────────────────────────────────────────────────

def load_synpuf_claims(
    filepath=None,
    max_claims=5000,
    skip_empty_cpt=True,
    sample_only=False
):
    """
    Load CMS SynPUF Carrier Claims and convert to MediGuard pipeline format.

    Args:
        filepath:       Path to SynPUF CSV file. Defaults to data/ folder.
        max_claims:     Maximum number of OUTPUT rows to produce.
                        SynPUF has 500K+ rows — use this to control size.
                        Each input row can produce up to 13 output rows
                        (one per HCPCS line). Default: 5000.
        skip_empty_cpt: Skip rows where HCPCS_CD_1 is empty. Default: True.
        sample_only:    If True, only read first 1000 input rows (for testing).

    Returns:
        pd.DataFrame: Converted claims in MediGuard pipeline format.
        Also saves to output/synpuf_converted_claims.csv.

    Pipeline usage:
        from cms_synpuf_loader import load_synpuf_claims
        df = load_synpuf_claims(max_claims=5000)
        # Then pass df into fwa_data_pipeline.py normalize step
    """
    if filepath is None:
        filepath = SYNPUF_FILE

    if not os.path.exists(filepath):
        print(f"  ❌ SynPUF file not found: {filepath}")
        print(f"     Download from: data.cms.gov/medicare-claims-synthetic-public-use-files")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    print(f"  [SynPUF] Loading: {os.path.basename(filepath)}")

    # ── Read CSV ──────────────────────────────────────────────────────────────
    try:
        nrows = 1000 if sample_only else None
        raw = pd.read_csv(
            filepath,
            dtype=str,           # read everything as string first
            nrows=nrows,
            on_bad_lines="skip", # skip malformed rows
            low_memory=False
        )
        print(f"  [SynPUF] Raw rows loaded: {len(raw):,}")
    except Exception as e:
        print(f"  ❌ Error reading SynPUF file: {e}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    converted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    skipped = 0

    # ── Process each row ──────────────────────────────────────────────────────
    for _, raw_row in raw.iterrows():

        if len(rows) >= max_claims:
            break

        # ── Core claim fields ──────────────────────────────────────────────
        patient_id   = str(raw_row.get("DESYNPUF_ID", "")).strip()
        claim_id_raw = str(raw_row.get("CLM_ID", "")).strip()
        service_date = parse_synpuf_date(raw_row.get("CLM_FROM_DT", ""))

        # Clean up CLM_ID — SynPUF stores as scientific notation float
        try:
            claim_id = f"SYNPUF-{int(float(claim_id_raw))}"
        except (ValueError, TypeError):
            claim_id = f"SYNPUF-{claim_id_raw}"

        # ── ICD-9 diagnosis (primary) ──────────────────────────────────────
        # SynPUF uses ICD-9 (2008-2010 data) — noted in conversion_notes
        icd_code = str(raw_row.get("ICD9_DGNS_CD_1", "")).strip()
        if icd_code in ("nan", "None", ""):
            icd_code = ""

        # ── Provider NPI ───────────────────────────────────────────────────
        provider_npi = str(raw_row.get("PRF_PHYSN_NPI_1", "")).strip()
        if provider_npi in ("nan", "None", ""):
            provider_npi = ""

        # ── Collect all non-empty HCPCS codes for this claim ──────────────
        hcpcs_codes = []
        amounts     = []

        for i in range(1, MAX_LINES + 1):
            code = str(raw_row.get(f"HCPCS_CD_{i}", "")).strip()
            amt  = str(raw_row.get(f"LINE_NCH_PMT_AMT_{i}", "0")).strip()

            if code and code not in ("nan", "None", ""):
                hcpcs_codes.append(code.zfill(5))
                try:
                    amounts.append(float(amt) if amt not in ("nan", "None", "") else 0.0)
                except ValueError:
                    amounts.append(0.0)

        # Skip if no HCPCS codes found
        if skip_empty_cpt and not hcpcs_codes:
            skipped += 1
            continue

        # ── Primary CPT = first HCPCS code ────────────────────────────────
        cpt_code       = hcpcs_codes[0] if hcpcs_codes else ""
        billed_amount  = amounts[0] if amounts else 0.0
        additional_cpt = ";".join(hcpcs_codes[1:]) if len(hcpcs_codes) > 1 else ""

        # ── POS inference from CPT range ───────────────────────────────────
        pos_code, pos_desc, pos_source = infer_pos_from_cpt(cpt_code)

        # ── Build notes ────────────────────────────────────────────────────
        notes = []
        notes.append("ICD-9 (2008-2010 SynPUF data)")
        if additional_cpt:
            notes.append(f"additional CPTs: {additional_cpt}")
        if len(hcpcs_codes) > 1:
            notes.append(f"{len(hcpcs_codes)} HCPCS lines on claim")

        rows.append({
            "claim_id":             claim_id,
            "patient_id":           patient_id,
            "provider_npi":         provider_npi,
            "provider_name":        "",          # not in SynPUF carrier file
            "provider_specialty":   "",          # not in SynPUF carrier file
            "provider_state":       "",          # not in SynPUF carrier file
            "payer_id":             "CMS_MEDICARE",
            "payer_name":           "Medicare (CMS SynPUF)",
            "plan_name":            "Medicare Part B",
            "cpt_code":             cpt_code,
            "cpt_description":      "",          # no desc in SynPUF
            "icd_code":             icd_code,
            "icd_description":      "",          # no desc in SynPUF
            "additional_cpt":       additional_cpt,
            "billed_amount":        round(billed_amount, 2),
            "service_date":         service_date,
            "place_of_service":     pos_code,
            "pos_description":      pos_desc,
            "pos_source":           f"layer2-cpt-range ({pos_source})",
            "claim_type":           "professional",
            "source_format":        "CMS_SYNPUF_CARRIER",
            "converted_at":         converted_at,
            "raw_fhir_id":          claim_id_raw,
            "conversion_notes":     "; ".join(notes),
            "cpt_expected_specialty": "",        # not in SynPUF carrier file
            "fraud_type":           "NONE"       # SynPUF has no injected fraud labels
        })

    # ── Build DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # ── Save CSV ───────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"  [SynPUF] ✅ Converted {len(df):,} claims")
    print(f"  [SynPUF]    Skipped {skipped:,} rows (empty HCPCS)")
    print(f"  [SynPUF]    Output → {OUTPUT_CSV}")
    print(f"  [SynPUF]    Note: ICD-9 codes (2008-2010 data)")
    print(f"  [SynPUF]    Note: Provider name/specialty not in carrier file")

    return df


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MediGuard AI — CMS SynPUF Carrier Claims Loader Test            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    df = load_synpuf_claims(max_claims=1000, sample_only=True)

    if df.empty:
        print("  ❌ No data loaded.")
    else:
        print()
        print("  ── Sample Output (first 5 rows) ──────────────────────────────")
        for _, row in df.head(5).iterrows():
            print(f"  Claim : {row['claim_id']}")
            print(f"    CPT : {row['cpt_code']}")
            print(f"    ICD : {row['icd_code']}")
            print(f"    POS : {row['place_of_service']} ({row['pos_description']})")
            print(f"    Amt : ${row['billed_amount']:,.2f}")
            if row['additional_cpt']:
                print(f"    +CPT: {row['additional_cpt']}")
            print(f"    Note: {row['conversion_notes']}")
            print()

        print(f"  ── Stats ──────────────────────────────────────────────────────")
        print(f"  Total claims    : {len(df):,}")
        print(f"  Unique patients : {df['patient_id'].nunique():,}")
        print(f"  Unique CPTs     : {df['cpt_code'].nunique():,}")
        print(f"  Unique ICDs     : {df['icd_code'].nunique():,}")
        print(f"  Avg billed amt  : ${df['billed_amount'].mean():,.2f}")
        print(f"  Multi-CPT claims: {(df['additional_cpt'] != '').sum():,}")
        print()
        print(f"  ✅ SynPUF loader test complete.")
        print(f"  To use in pipeline: set DATA_SOURCE = 'synpuf' in fwa_data_pipeline.py")
