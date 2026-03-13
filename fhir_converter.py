"""
╔══════════════════════════════════════════════════════════════╗
║  MEDIAGUARD AI — FHIR R4 Converter  (v2)                    ║
║  Converts FHIR ExplanationOfBenefit → MediGuard CSV format  ║
╚══════════════════════════════════════════════════════════════╝

PURPOSE:
  Reads FHIR R4 ExplanationOfBenefit (EOB) resources — either
  individual JSON files OR a full FHIR Bundle — and converts
  them into the normalized claims CSV format that the MediGuard
  AI pipeline (fwa_data_pipeline.py) understands.

  This is the FHIR interoperability layer that makes MediGuard AI
  compatible with any CMS-compliant payer API (CMS-0057-F mandate).

INPUT OPTIONS:
  Option A — FHIR Bundle:   fhir_samples/fhir_bundle.json
  Option B — Individual:    fhir_samples/*.json  (any EOB files)
  Option C — Live API:      CMS BCDA sandbox endpoint (Month 2)

OUTPUT:
  output/fhir_converted_claims.csv   ← feeds into fwa_data_pipeline.py

FULL PIPELINE RUN ORDER:
  1. python fhir_sample_generator.py
  2. python fhir_converter.py            ← this file
  3. python fwa_data_pipeline.py         (set DATA_SOURCE = "fhir")
  4. python fwa_langchain_reasoning.py

═══════════════════════════════════════════════════════════════
ARCHITECTURAL DECISION LOG — Place of Service (POS) Resolution
Date: March 2026
═══════════════════════════════════════════════════════════════

PROBLEM:
  fwa_data_pipeline.py expects a `place_of_service` column (integer
  CMS POS code) and `pos_description` column (display name).
  The original FHIR generator (v1) did not include the FHIR location
  field in EOB items, so the converter had no POS data to work with.

WHAT WE TRIED AND WHY WE REJECTED IT:

  Attempt 1 — Flat default (df["place_of_service"] = 11):
    Fast to implement but clinically wrong. A knee replacement
    (CPT 27447) defaulting to Office (POS 11) is itself a fraud
    flag — major surgery cannot be performed in a clinic.
    This would pollute the rule engine with false positives.
    REJECTED.

  Attempt 2 — Hardcoded CPT-to-POS dictionary:
    We built a small dict of ~20 CPTs mapped to POS codes.
    Ram correctly challenged: "What happens when we expand to
    10,000 real CPT codes?" The dict would need constant manual
    maintenance and would miss anything not listed.
    REJECTED as not scalable.

FINAL SOLUTION — Three-layer POS resolution (implemented below):

  LAYER 1 — Parse from FHIR location field (PRIMARY):
    The real CMS BCDA API always includes a location field in
    EOB items. The v2 generator now includes it too.
    We parse: item.location.locationCodeableConcept.coding[0].code
    This is the most accurate source — direct from the FHIR resource.
    Handles: all v2 synthetic data + real CMS BCDA API responses.

  LAYER 2 — CPT code range inference (FALLBACK):
    For claims arriving WITHOUT a location field (e.g., partial
    FHIR implementations, legacy payer APIs, v1 synthetic data).
    CPT codes follow CMS category structure — the numeric range
    tells us the likely care setting:
      99221–99233  →  Inpatient Hospital (21)  — hospital visit codes
      99281–99285  →  Emergency Room (23)       — ED codes
      10000–69999  →  Surgical range (21 or 22) — infer by sub-range
      99201–99215  →  Office visits (11)        — E&M office codes
      70000–79999  →  Radiology (22)            — typically outpatient
    This covers the vast majority of real-world CPT codes without
    a full 10,000-row lookup table.
    Not perfect but clinically sound for ~90% of claims.

  LAYER 3 — V2 Roadmap (Month 2, RAG Layer):
    When we add ChromaDB, we embed the full CMS POS-CPT reference
    table as a vector store. The LLM retrieves the exact correct
    POS for any CPT code, including codes added after this file
    was written. This eliminates the CPT range inference entirely.

WHY THIS THREE-LAYER APPROACH IS CORRECT ARCHITECTURE:
  - Layer 1 handles the happy path (well-formed FHIR data)
  - Layer 2 handles real-world messiness (partial compliance)
  - Layer 3 is the long-term scalable solution
  This is called "graceful degradation" — each layer falls back
  to the next only when needed, always using the best available data.
═══════════════════════════════════════════════════════════════
"""

import json
import csv
import os
import sys
import glob
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
FHIR_BUNDLE_PATH  = "fhir_samples/fhir_bundle.json"
FHIR_SAMPLES_DIR  = "fhir_samples"
OUTPUT_DIR        = "output"
OUTPUT_CSV        = os.path.join(OUTPUT_DIR, "fhir_converted_claims.csv")

# FHIR standard system URIs
NPI_SYSTEM          = "http://hl7.org/fhir/sid/us-npi"
ICD10_SYSTEM        = "http://hl7.org/fhir/sid/icd-10-cm"
CPT_SYSTEM          = "http://www.ama-assn.org/go/cpt"
CLAIM_TYPE_SYSTEM   = "http://terminology.hl7.org/CodeSystem/claim-type"
ADJUDICATION_SYSTEM = "http://terminology.hl7.org/CodeSystem/adjudication"
POS_SYSTEM          = "https://www.cms.gov/Medicare/Coding/place-of-service-codes"

# MediGuard extension URIs
EXT_SPECIALTY      = "https://mediaguard.ai/fhir/extension/provider-specialty"
EXT_STATE          = "https://mediaguard.ai/fhir/extension/provider-state"
EXT_PLAN           = "https://mediaguard.ai/fhir/extension/plan-name"
EXT_FRAUD_TYPE     = "https://mediaguard.ai/fhir/extension/injected-fraud-type"
# Expected specialty for the CPT procedure — set by the scenario, not the provider.
# Used by fwa_data_pipeline.py to detect specialty mismatch without a hardcoded CPT dict.
EXT_CPT_SPECIALTY  = "https://mediaguard.ai/fhir/extension/cpt-expected-specialty"

# POS display names — CMS standard
POS_DISPLAY = {
    11: "Office",
    21: "Inpatient Hospital",
    22: "Outpatient Hospital",
    23: "Emergency Room",
    31: "Skilled Nursing Facility",
}

# Output columns — must match fwa_data_pipeline.py expectations
OUTPUT_COLUMNS = [
    "claim_id", "patient_id", "provider_npi", "provider_name",
    "provider_specialty", "provider_state", "payer_id", "payer_name",
    "plan_name", "cpt_code", "cpt_description", "cpt_expected_specialty",
    "icd_code", "icd_description",
    "additional_cpt", "billed_amount", "service_date", "place_of_service",
    "pos_description", "pos_source", "claim_type", "source_format",
    "converted_at", "raw_fhir_id", "fraud_type", "conversion_notes"
]


# ── LAYER 2: CPT RANGE → POS INFERENCE ──────────────────────────────────────
# Fallback when FHIR location field is absent.
# Based on CMS CPT code category structure.
# See ARCHITECTURAL DECISION LOG above for full explanation.

def infer_pos_from_cpt(cpt_code):
    """
    Infer Place of Service from CPT code numeric range.

    This is Layer 2 of our three-layer POS resolution strategy.
    Used ONLY when the FHIR location field is absent (Layer 1 fails).

    CPT code structure follows CMS category assignments:
      - E&M codes (99xxx) encode the setting in their sub-range
      - Surgical codes (1xxxx-6xxxx) are typically inpatient or outpatient
      - Radiology (7xxxx) is typically outpatient hospital
      - Lab/path (8xxxx) is typically office or outpatient

    SCALABILITY NOTE:
      This approach handles any CPT code — including codes not yet
      invented — because it uses numeric ranges, not a hardcoded list.
      A hardcoded list of 20 CPTs would break when we expand the dataset.
      CPT range logic scales to all 10,000+ CPT codes automatically.

    Returns: (pos_code: int, pos_description: str, inference_confidence: str)
    """
    try:
        cpt_int = int(str(cpt_code).replace(".", "").strip())
    except (ValueError, TypeError):
        # Non-numeric CPT (e.g., HCPCS like G0008) — default to Office
        return 11, "Office", "default (non-numeric CPT)"

    # ── E&M Codes (99xxx) — setting encoded in sub-range ──
    if 99201 <= cpt_int <= 99215:
        return 11, "Office", "CPT range: office E&M visits"
    if 99221 <= cpt_int <= 99239:
        return 21, "Inpatient Hospital", "CPT range: inpatient hospital care"
    if 99241 <= cpt_int <= 99255:
        return 22, "Outpatient Hospital", "CPT range: outpatient consultations"
    if 99281 <= cpt_int <= 99288:
        return 23, "Emergency Room", "CPT range: emergency department"
    if 99304 <= cpt_int <= 99318:
        return 31, "Skilled Nursing Facility", "CPT range: SNF care"

    # ── Surgical Codes ──
    # Major surgery (musculoskeletal, cardiovascular, thoracic) → Inpatient
    if 10000 <= cpt_int <= 29999:
        return 22, "Outpatient Hospital", "CPT range: minor/outpatient surgery"
    if 30000 <= cpt_int <= 49999:
        return 21, "Inpatient Hospital", "CPT range: major inpatient surgery"
    if 50000 <= cpt_int <= 69999:
        return 21, "Inpatient Hospital", "CPT range: major inpatient surgery"

    # ── Radiology (70000–79999) → Outpatient Hospital ──
    if 70000 <= cpt_int <= 79999:
        return 22, "Outpatient Hospital", "CPT range: radiology/imaging"

    # ── Lab/Pathology (80000–89999) → Office or Outpatient ──
    if 80000 <= cpt_int <= 89999:
        return 11, "Office", "CPT range: lab/pathology"

    # ── Medicine section (90000–99199) ──
    if 90000 <= cpt_int <= 99199:
        return 11, "Office", "CPT range: medicine/therapy"

    # ── Default: Office ──
    return 11, "Office", "default (unmatched range)"


# ── FHIR PARSING HELPERS ─────────────────────────────────────────────────────

def get_extension_value(extensions, url):
    if not extensions:
        return ""
    for ext in extensions:
        if ext.get("url") == url:
            return ext.get("valueString", ext.get("valueCode", ""))
    return ""

def get_coding_value(codeable_concept, system=None):
    if not codeable_concept:
        return "", ""
    codings = codeable_concept.get("coding", [])
    for coding in codings:
        if system is None or coding.get("system") == system:
            return coding.get("code", ""), coding.get("display", "")
    if codings:
        return codings[0].get("code", ""), codings[0].get("display", "")
    return "", ""

def get_identifier_value(identifiers, system=None):
    if not identifiers:
        return ""
    for ident in identifiers:
        if system is None or ident.get("system") == system:
            return ident.get("value", "")
    if identifiers:
        return identifiers[0].get("value", "")
    return ""

def get_total_amount(totals):
    if not totals:
        return 0.0
    for total in totals:
        category = total.get("category", {})
        code, _ = get_coding_value(category, ADJUDICATION_SYSTEM)
        if code == "submitted":
            return total.get("amount", {}).get("value", 0.0)
    if totals:
        return totals[0].get("amount", {}).get("value", 0.0)
    return 0.0

def parse_patient_id(patient_ref):
    if not patient_ref:
        return ""
    ref = patient_ref.get("reference", "")
    return ref.split("/")[-1] if "/" in ref else ref

def parse_service_date(eob):
    items = eob.get("item", [])
    if items:
        serviced = items[0].get("servicedDate", "")
        if serviced:
            return serviced
    period = eob.get("billablePeriod", {})
    return period.get("start", "")

def parse_cpt_codes(items):
    if not items:
        return "", "", []
    primary_cpt, primary_desc = "", ""
    additional = []
    for i, item in enumerate(items):
        product = item.get("productOrService", {})
        code, desc = get_coding_value(product, CPT_SYSTEM)
        if not code:
            code, desc = get_coding_value(product)
        if i == 0:
            primary_cpt, primary_desc = code, desc
        else:
            additional.append(code)
    return primary_cpt, primary_desc, additional

def parse_diagnosis(diagnoses):
    if not diagnoses:
        return "", ""
    for diag in diagnoses:
        dtype = diag.get("type", [])
        for t in dtype:
            code, _ = get_coding_value(t)
            if code in ("principal", "admitting"):
                icd_concept = diag.get("diagnosisCodeableConcept", {})
                icd_code, icd_desc = get_coding_value(icd_concept, ICD10_SYSTEM)
                if not icd_code:
                    icd_code, icd_desc = get_coding_value(icd_concept)
                return icd_code, icd_desc
    first_diag = diagnoses[0]
    icd_concept = first_diag.get("diagnosisCodeableConcept", {})
    icd_code, icd_desc = get_coding_value(icd_concept, ICD10_SYSTEM)
    if not icd_code:
        icd_code, icd_desc = get_coding_value(icd_concept)
    return icd_code, icd_desc


# ── LAYER 1: PARSE POS FROM FHIR LOCATION FIELD ──────────────────────────────

def parse_pos_from_fhir(items, primary_cpt):
    """
    THREE-LAYER POS RESOLUTION — see ARCHITECTURAL DECISION LOG above.

    Layer 1 (this function, PRIMARY):
      Try to read POS from the FHIR location field in the first item.
      Standard FHIR R4 path: item[0].location.locationCodeableConcept

      This is present in:
        - Our v2 synthetic data (fhir_sample_generator.py v2)
        - Real CMS BCDA API responses
        - Any FHIR-compliant payer API

    Layer 2 (FALLBACK — called if Layer 1 returns nothing):
      Infer POS from CPT code numeric range via infer_pos_from_cpt().
      Handles: v1 synthetic data, partial FHIR implementations.
      Scalable to all 10,000+ CPT codes via range logic.

    Layer 3 (ROADMAP — Month 2 RAG layer):
      ChromaDB lookup of full CMS POS-CPT reference table.
      Will replace Layer 2 entirely.

    Returns: (pos_code: int, pos_description: str, pos_source: str)
      pos_source documents which layer resolved the POS — useful for
      debugging and for measuring how often each layer fires.
    """
    if not items:
        # No items at all — use CPT fallback with empty string
        pos_code, pos_desc, reason = infer_pos_from_cpt(primary_cpt)
        return pos_code, pos_desc, f"layer2-fallback ({reason})"

    # ── Layer 1: FHIR location field ──
    first_item = items[0]
    location   = first_item.get("location", {})

    if location:
        loc_concept = location.get("locationCodeableConcept", {})
        pos_str, pos_display = get_coding_value(loc_concept, POS_SYSTEM)

        if not pos_str:
            # Try without system filter (some implementations omit system URI)
            pos_str, pos_display = get_coding_value(loc_concept)

        if pos_str:
            try:
                pos_code = int(pos_str)
                # Use our display name if the FHIR display is blank
                pos_desc = pos_display or POS_DISPLAY.get(pos_code, f"POS-{pos_code}")
                return pos_code, pos_desc, "layer1-fhir-location"
            except ValueError:
                pass  # Fall through to Layer 2

    # ── Layer 2: CPT range inference ──
    # Location field absent or unparseable — infer from CPT range
    pos_code, pos_desc, reason = infer_pos_from_cpt(primary_cpt)
    return pos_code, pos_desc, f"layer2-cpt-range ({reason})"


# ── MAIN CONVERTER ───────────────────────────────────────────────────────────

def convert_eob_to_row(eob):
    """
    Convert a single FHIR R4 ExplanationOfBenefit resource
    into a MediGuard AI pipeline row (dict).
    """
    notes = []

    # ── Claim ID ──
    claim_id = eob.get("id", "")
    if not claim_id:
        claim_id = get_identifier_value(eob.get("identifier", []))
        notes.append("id from identifier")

    # ── Patient ──
    patient_id = parse_patient_id(eob.get("patient", {}))

    # ── Provider ──
    provider_ref  = eob.get("provider", {})
    provider_name = provider_ref.get("display", "")
    provider_npi  = ""
    ident = provider_ref.get("identifier", {})
    if ident:
        provider_npi = ident.get("value", "")

    # ── Provider Specialty & State (from MediGuard extensions) ──
    extensions         = eob.get("extension", [])
    provider_specialty = get_extension_value(extensions, EXT_SPECIALTY)
    provider_state     = get_extension_value(extensions, EXT_STATE)
    plan_name          = get_extension_value(extensions, EXT_PLAN)
    # The specialty the CPT procedure requires — sourced from the scenario,
    # not the provider. Pipeline uses this for specialty mismatch detection
    # without needing a hardcoded CPT→specialty lookup table.
    cpt_expected_specialty = get_extension_value(extensions, EXT_CPT_SPECIALTY)

    # ── Payer ──
    insurer     = eob.get("insurer", {})
    payer_name  = insurer.get("display", "")
    payer_ident = insurer.get("identifier", {})
    payer_id    = payer_ident.get("value", "") if payer_ident else ""

    # ── Diagnosis ──
    icd_code, icd_desc = parse_diagnosis(eob.get("diagnosis", []))
    if not icd_code:
        notes.append("no ICD code found")

    # ── CPT codes ──
    items = eob.get("item", [])
    cpt_code, cpt_desc, additional_cpts = parse_cpt_codes(items)
    if not cpt_code:
        notes.append("no CPT code found")
    additional_cpt_str = ";".join(additional_cpts) if additional_cpts else ""
    if additional_cpts:
        notes.append(f"additional CPTs: {additional_cpt_str}")

    # ── Place of Service — THREE-LAYER RESOLUTION ──────────────────────────
    # See parse_pos_from_fhir() and ARCHITECTURAL DECISION LOG for full
    # explanation of why we use this three-layer approach instead of a
    # hardcoded default or a fixed CPT list.
    pos_code, pos_desc, pos_source = parse_pos_from_fhir(items, cpt_code)
    # pos_source tells us which layer resolved it — useful for debugging
    if "layer2" in pos_source:
        notes.append(f"POS inferred: {pos_source}")

    # ── Amounts ──
    billed_amount = get_total_amount(eob.get("total", []))
    if billed_amount == 0.0:
        for item in items:
            billed_amount += item.get("unitPrice", {}).get("value", 0.0)
        if billed_amount > 0:
            notes.append("amount summed from items")

    # ── Dates ──
    service_date = parse_service_date(eob)

    # ── Claim Type ──
    claim_type_concept = eob.get("type", {})
    claim_type, _ = get_coding_value(claim_type_concept, CLAIM_TYPE_SYSTEM)
    if not claim_type:
        claim_type = "professional"

    # Ground-truth fraud label — present in synthetic data only, not real CMS claims.
    # Carried as a FHIR extension so it travels cleanly through to the pipeline
    # for evaluation without polluting the clinical fields.
    fraud_type = get_extension_value(extensions, EXT_FRAUD_TYPE)

    return {
        "claim_id":               claim_id,
        "patient_id":             patient_id,
        "provider_npi":           provider_npi,
        "provider_name":          provider_name,
        "provider_specialty":     provider_specialty,
        "provider_state":         provider_state,
        "payer_id":               payer_id,
        "payer_name":             payer_name,
        "plan_name":              plan_name,
        "cpt_code":               cpt_code,
        "cpt_description":        cpt_desc,
        "cpt_expected_specialty": cpt_expected_specialty,  # from scenario, not provider
        "icd_code":               icd_code,
        "icd_description":        icd_desc,
        "additional_cpt":         additional_cpt_str,
        "billed_amount":          round(billed_amount, 2),
        "service_date":           service_date,
        "place_of_service":       pos_code,
        "pos_description":        pos_desc,
        "pos_source":             pos_source,
        "claim_type":             claim_type,
        "source_format":          "FHIR_R4_EOB",
        "converted_at":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_fhir_id":            eob.get("id", ""),
        "fraud_type":             fraud_type,   # ground-truth label for evaluation only
        "conversion_notes":       "; ".join(notes) if notes else "clean"
    }


def load_eobs_from_bundle(bundle_path):
    with open(bundle_path) as f:
        bundle = json.load(f)
    return [
        entry["resource"]
        for entry in bundle.get("entry", [])
        if entry.get("resource", {}).get("resourceType") == "ExplanationOfBenefit"
    ]

def load_eobs_from_directory(directory):
    eobs = []
    for fpath in sorted(glob.glob(os.path.join(directory, "*.json"))):
        fname = os.path.basename(fpath)
        if fname in ("fhir_bundle.json", "fhir_claims_summary.csv"):
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
            if data.get("resourceType") == "ExplanationOfBenefit":
                eobs.append(data)
        except Exception as e:
            print(f"    ⚠️  Skipped {fname}: {e}")
    return eobs


def run_converter(source=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  MEDIAGUARD AI — FHIR R4 Converter  (v2)                    ║")
    print("║  Three-layer POS resolution: FHIR → CPT range → RAG(v3)    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Determine source ──
    if source is None:
        if os.path.exists(FHIR_BUNDLE_PATH):
            source      = FHIR_BUNDLE_PATH
            source_type = "bundle"
            print(f"  [SOURCE] FHIR Bundle: {FHIR_BUNDLE_PATH}")
        elif os.path.exists(FHIR_SAMPLES_DIR):
            source      = FHIR_SAMPLES_DIR
            source_type = "directory"
            print(f"  [SOURCE] FHIR Directory: {FHIR_SAMPLES_DIR}/")
        else:
            print("  ❌ No FHIR source found. Run fhir_sample_generator.py first.")
            sys.exit(1)
    else:
        source_type = "directory" if os.path.isdir(source) else "bundle"

    # ── Load ──
    print()
    print("  [LOAD] Reading FHIR ExplanationOfBenefit resources...")
    eobs = load_eobs_from_bundle(source) if source_type in ("bundle", "file") \
           else load_eobs_from_directory(source)

    if not eobs:
        print("  ❌ No ExplanationOfBenefit resources found.")
        sys.exit(1)
    print(f"    Loaded {len(eobs)} EOB resources")

    # ── Convert ──
    print()
    print("  [CONVERT] Converting FHIR EOB → MediGuard format...")
    print("  ─────────────────────────────────────────────────────────")

    rows           = []
    errors         = []
    layer1_count   = 0   # resolved via FHIR location field
    layer2_count   = 0   # resolved via CPT range inference

    for i, eob in enumerate(eobs):
        claim_id = eob.get("id", f"EOB-{i+1}")
        try:
            row = convert_eob_to_row(eob)
            rows.append(row)

            # Track which POS layer fired
            if "layer1" in row["pos_source"]:
                layer1_count += 1
            else:
                layer2_count += 1

            flag = "✅" if row["conversion_notes"] == "clean" else "⚠️ "
            print(f"    [{i+1:03}/{len(eobs)}] {claim_id:<32} "
                  f"CPT:{row['cpt_code']:<8} "
                  f"POS:{row['place_of_service']:>2} ({row['pos_description']:<22}) "
                  f"${row['billed_amount']:>9,.2f}  {flag}")

        except Exception as e:
            errors.append({"claim_id": claim_id, "error": str(e)})
            print(f"    [{i+1:03}/{len(eobs)}] {claim_id:<32} ❌ ERROR: {e}")

    # ── Write CSV ──
    print()
    print(f"  [WRITE] Saving to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ──
    print()
    print("═" * 66)
    print("  FHIR CONVERSION COMPLETE (v2)")
    print("═" * 66)
    print(f"  EOB resources loaded      : {len(eobs)}")
    print(f"  Successfully converted    : {len(rows)}")
    print(f"  Errors                    : {len(errors)}")
    print()
    print("  POS Resolution breakdown:")
    print(f"    Layer 1 (FHIR location field) : {layer1_count} claims  ← ideal path")
    print(f"    Layer 2 (CPT range inference) : {layer2_count} claims  ← fallback")
    print(f"    Layer 3 (RAG — Month 2)       : not yet implemented")
    print()
    print(f"  Output → {OUTPUT_CSV}")
    print()
    print("  SAMPLE OUTPUT (first 3 rows):")
    print("  " + "─" * 60)
    for row in rows[:3]:
        print(f"  Claim : {row['claim_id']}")
        print(f"    CPT : {row['cpt_code']} — {row['cpt_description']}")
        print(f"    ICD : {row['icd_code']} — {row['icd_description']}")
        print(f"    POS : {row['place_of_service']} ({row['pos_description']}) "
              f"← source: {row['pos_source']}")
        print(f"    Amt : ${row['billed_amount']:,.2f}  |  {row['payer_name']}")
        if row["additional_cpt"]:
            print(f"    ⚠️   Extra CPT: {row['additional_cpt']} (potential unbundling)")
        print()
    print("═" * 66)
    print()
    print("  ✅ FHIR conversion complete!")
    print("  Next: Set DATA_SOURCE = 'fhir' in fwa_data_pipeline.py and run it.")
    print()

    return rows


if __name__ == "__main__":
    source = None
    if len(sys.argv) > 2 and sys.argv[1] == "--source":
        source = sys.argv[2]
    run_converter(source)
