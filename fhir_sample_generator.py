"""
╔══════════════════════════════════════════════════════════════╗
║  MEDIAGUARD AI — FHIR Sample Data Generator  (v2)            ║
║  Generates realistic FHIR R4 ExplanationOfBenefit (EOB)      ║
║  resources that mirror CMS BCDA API output                   ║
╚══════════════════════════════════════════════════════════════╝

PURPOSE:
  Generates synthetic FHIR R4 ExplanationOfBenefit JSON files
  that match the structure of real CMS BCDA API output.
  Used as realistic test data for the MediGuard AI FHIR converter.

OUTPUT:
  fhir_samples/                          ← folder of individual EOB JSONs
  fhir_samples/fhir_bundle.json          ← full FHIR Bundle (all claims)
  fhir_samples/fhir_claims_summary.csv   ← human-readable summary

USAGE:
  python fhir_sample_generator.py

FULL PIPELINE RUN ORDER:
  1. python fhir_sample_generator.py     ← this file
  2. python fhir_converter.py
  3. python fwa_data_pipeline.py         (set DATA_SOURCE = "fhir")
  4. python fwa_langchain_reasoning.py

WHAT IS FHIR?
  FHIR (Fast Healthcare Interoperability Resources) is the modern
  standard for exchanging healthcare data. Under CMS-0057-F rule,
  all Medicare Advantage and Medicaid payers must expose FHIR APIs
  by January 2027. ExplanationOfBenefit (EOB) is the FHIR resource
  that represents a processed insurance claim.

═══════════════════════════════════════════════════════════════
ARCHITECTURAL DECISION LOG — Place of Service (POS)
Date: March 2026
═══════════════════════════════════════════════════════════════

PROBLEM:
  When we built the first version of the FHIR converter, we did
  not include the Place of Service (POS) field in our generated
  EOB resources. The fwa_data_pipeline.py rule engine needs POS
  to score claims correctly.

  Example of why POS matters to FWA scoring:
    - CPT 27447 (total knee replacement) billed in Office (POS 11)
      → itself a red flag — major surgery cannot happen in a clinic
    - CPT 27447 billed in Inpatient Hospital (POS 21)
      → clinically expected — no flag from POS alone
  Without correct POS, the rule engine either misses real fraud
  signals or generates false positives.

ATTEMPT 1 — Flat default (REJECTED):
  We tried: df["place_of_service"] = 11  (always Office)
  Problem: Clinically inaccurate. A knee replacement defaulting
  to Office (11) would generate false fraud flags and distort
  every downstream score. Ram correctly identified this as wrong.

ATTEMPT 2 — Hardcoded CPT-to-POS dictionary (REJECTED):
  We built a small dict mapping specific CPT codes to POS codes.
  Problem: Ram challenged this: "These are only our sample CPTs.
  What happens when we expand to 10,000 real CPT codes?"
  A hardcoded list of 20 CPTs does not scale to production.

FINAL SOLUTION — Three-layer architecture:

  LAYER 1 (this file — fhir_sample_generator.py):
    Embed the correct FHIR `location` field directly in each EOB
    item at generation time, keyed to the scenario's pos_code.
    This is the architecturally correct fix — the real CMS BCDA
    API always includes a location field in EOB items.
    We were simply omitting it in v1.

  LAYER 2 (fhir_converter.py — primary path):
    Parse the `location.locationCodeableConcept` from FHIR when
    present. This handles both our synthetic data AND real CMS
    BCDA API data, since both will carry the location field.

  LAYER 3 (fhir_converter.py — fallback path):
    CPT code range logic for any claim that arrives WITHOUT a
    location field (e.g., partial FHIR implementations or legacy
    payer APIs that don't fully comply yet).
    CPT codes follow CMS category ranges:
      99221–99223  →  Inpatient Hospital (21)
      99281–99285  →  Emergency Room (23)
      10000–69999  →  Surgical procedures (infer from sub-range)
      99201–99215  →  Office visits (11)
    This covers the vast majority of real-world CPT codes without
    requiring a full 10,000-row lookup table.

  V2 ROADMAP (Month 2 — RAG Layer):
    When we add ChromaDB and the CMS reference data layer, we will
    embed the full CMS POS-CPT mapping table as a vector store.
    The LLM will then retrieve the exact correct POS for any CPT
    code, including new codes added after this file was written.

WHY THIS MATTERS FOR MEDIAGUARD AI ACCURACY:
  1. False positives: Wrong POS causes legitimate inpatient
     procedures to look like office fraud.
  2. Missed fraud: Some fraud patterns are only visible when
     POS is correctly matched to the clinical context.
  3. LLM quality: Groq/Llama receives POS as context in the
     claim briefing. Correct POS improves narrative accuracy.
═══════════════════════════════════════════════════════════════
"""

import json
import csv
import uuid
import random
import os
from datetime import datetime, timedelta

# ── CONFIG ──────────────────────────────────────────────────────────────────
OUTPUT_DIR       = "fhir_samples"
NUM_CLEAN_CLAIMS = 30   # normal legitimate claims
NUM_FRAUD_CLAIMS = 20   # injected FWA patterns
RANDOM_SEED      = 42
random.seed(RANDOM_SEED)

# ── PLACE OF SERVICE REFERENCE ───────────────────────────────────────────────
# Standard CMS Place of Service codes used across all US payers.
# Source: CMS Place of Service Code Set
# https://www.cms.gov/medicare/coding-billing/place-of-service-codes
#
# DESIGN NOTE: POS is defined at the SCENARIO level, not the provider level.
# The same provider may bill different POS codes for different procedures —
# e.g., office visits (POS 11) AND surgeries at the hospital (POS 21).
# Defining POS per scenario mirrors real-world billing patterns accurately.

POS_REFERENCE = {
    11: {"display": "Office",                   "description": "Location other than hospital"},
    21: {"display": "Inpatient Hospital",        "description": "Admitted for overnight stay"},
    22: {"display": "Outpatient Hospital",       "description": "Hospital-based, no overnight"},
    23: {"display": "Emergency Room",            "description": "ER/ED services"},
    31: {"display": "Skilled Nursing Facility",  "description": "SNF post-acute care"},
}

# ── REFERENCE DATA ──────────────────────────────────────────────────────────

PROVIDERS = [
    {"npi": "1234567890", "name": "Dr. Sarah Johnson",  "specialty": "Internal Medicine",  "state": "TX"},
    {"npi": "1234567891", "name": "Dr. Michael Chen",   "specialty": "Family Practice",    "state": "FL"},
    {"npi": "1234567892", "name": "Dr. Patricia Moore", "specialty": "Orthopedic Surgery", "state": "CA"},
    {"npi": "1234567893", "name": "Dr. James Williams", "specialty": "Cardiology",         "state": "NY"},
    {"npi": "1234567894", "name": "Dr. Linda Martinez", "specialty": "Neurology",          "state": "OH"},
    {"npi": "1234567895", "name": "Dr. Robert Davis",   "specialty": "Urology",            "state": "GA"},
    {"npi": "1234567896", "name": "Dr. Karen Wilson",   "specialty": "Dermatology",        "state": "IL"},
    {"npi": "1234567897", "name": "Dr. Thomas Brown",   "specialty": "Gastroenterology",   "state": "PA"},
]

PAYERS = [
    {"id": "PAYER001", "name": "BlueCross BlueShield", "plan": "Medicare Advantage Plan A"},
    {"id": "PAYER002", "name": "Aetna",                "plan": "Medicare Advantage Gold"},
    {"id": "PAYER003", "name": "UnitedHealthcare",     "plan": "AARP Medicare Complete"},
    {"id": "PAYER004", "name": "Humana",               "plan": "Medicare Advantage PPO"},
]

# ── LEGITIMATE SCENARIOS ─────────────────────────────────────────────────────
# V2 CHANGE: Each scenario now includes pos_code — the clinically correct
# Place of Service for that CPT code in that clinical context.
# This was absent in v1, causing POS inference problems downstream.
# Adding it here at the source (the generator) is the cleanest architectural
# fix — the converter simply reads what is already in the FHIR resource.

LEGITIMATE_SCENARIOS = [
    {
        "cpt": "99213", "cpt_desc": "Office visit, low-moderate complexity",
        "icd": "J06.9",  "icd_desc": "Acute upper respiratory infection",
        "amount": 120.00, "specialty": "Family Practice",
        "pos_code": 11,  # Office — standard for outpatient E&M visits
    },
    {
        "cpt": "99214", "cpt_desc": "Office visit, moderate complexity",
        "icd": "I10",    "icd_desc": "Essential hypertension",
        "amount": 165.00, "specialty": "Internal Medicine",
        "pos_code": 11,  # Office
    },
    {
        "cpt": "93000", "cpt_desc": "Electrocardiogram (ECG) with interpretation",
        "icd": "I25.10", "icd_desc": "Atherosclerotic heart disease",
        "amount": 85.00,  "specialty": "Cardiology",
        "pos_code": 11,  # Office — EKG commonly performed in cardiology clinic
    },
    {
        "cpt": "27447", "cpt_desc": "Total knee replacement arthroplasty",
        "icd": "M17.11", "icd_desc": "Primary osteoarthritis, right knee",
        "amount": 12500.00, "specialty": "Orthopedic Surgery",
        "pos_code": 21,  # Inpatient Hospital — major surgery requires admission
        # NOTE: If this were POS 11 (Office), that itself would be a fraud flag
    },
    {
        "cpt": "99232", "cpt_desc": "Inpatient hospital care, subsequent",
        "icd": "N39.0",  "icd_desc": "Urinary tract infection",
        "amount": 210.00, "specialty": "Internal Medicine",
        "pos_code": 21,  # Inpatient Hospital — CPT 99232 by definition requires inpatient
    },
    {
        "cpt": "36415", "cpt_desc": "Venipuncture for blood collection",
        "icd": "E11.9",  "icd_desc": "Type 2 diabetes mellitus",
        "amount": 18.00,  "specialty": "Family Practice",
        "pos_code": 11,  # Office — routine blood draw in clinic
    },
    {
        "cpt": "80053", "cpt_desc": "Comprehensive metabolic panel",
        "icd": "E11.9",  "icd_desc": "Type 2 diabetes mellitus",
        "amount": 38.00,  "specialty": "Internal Medicine",
        "pos_code": 11,  # Office — lab panel ordered during office visit
    },
    {
        "cpt": "71046", "cpt_desc": "Chest X-ray, 2 views",
        "icd": "J18.9",  "icd_desc": "Pneumonia, unspecified",
        "amount": 95.00,  "specialty": "Internal Medicine",
        "pos_code": 22,  # Outpatient Hospital — radiology typically hospital-based
    },
    {
        "cpt": "99215", "cpt_desc": "Office visit, high complexity",
        "icd": "C34.11", "icd_desc": "Malignant neoplasm upper lobe lung",
        "amount": 225.00, "specialty": "Internal Medicine",
        "pos_code": 11,  # Office — oncology follow-up visit in clinic
    },
    {
        "cpt": "45378", "cpt_desc": "Colonoscopy, diagnostic",
        "icd": "Z12.11", "icd_desc": "Colorectal cancer screening",
        "amount": 850.00, "specialty": "Gastroenterology",
        "pos_code": 22,  # Outpatient Hospital — endoscopy suite, no overnight admission
    },
]

# ── FWA SCENARIOS ─────────────────────────────────────────────────────────────
# FWA scenarios also carry pos_code.
#
# IMPORTANT DESIGN NOTE:
# Correct POS is critical for catching the RIGHT fraud signal.
# Pattern 3 (ICD-CPT mismatch) is a good example:
#   - The procedure (knee replacement) IS in an inpatient setting (POS 21) ✓
#   - The DIAGNOSIS (hip stiffness) does NOT justify it ✗
#   - Fraud signal = wrong diagnosis, NOT wrong setting
# If we had defaulted POS to Office (11), the rule engine would fire for
# the wrong reason (impossible surgery in office) and mask the actual
# clinical fraud signal (diagnosis doesn't justify procedure).
# Correct POS lets the rule engine fire on the RIGHT rule.

FWA_SCENARIOS = [
    # Pattern 1: Upcoding — billing 99215 for a simple cold
    # POS is Office (11) — setting is correct, complexity code is inflated
    {
        "fraud_type": "UPCODING",
        "cpt": "99215", "cpt_desc": "Office visit, high complexity (UPCODED)",
        "icd": "J06.9",  "icd_desc": "Acute upper respiratory infection",
        "amount": 225.00,
        "pos_code": 11,       # Office — fraud is the code level, not the setting
        "specialty": "Internal Medicine",  # Upcoding by internist — clinically plausible
        "note": "High complexity visit billed for simple cold — typical upcoding pattern"
    },
    # # Pattern 2: Unbundling — billing 36415 + 36416 together on same day
    # # NCCI says 36415 (venipuncture) already includes 36416 (capillary draw)
    # {
    #     "fraud_type": "UNBUNDLING",
    #     "cpt": "36415", "cpt_desc": "Venipuncture (should not appear with 36416)",
    #     "icd": "E11.9",  "icd_desc": "Type 2 diabetes mellitus",
    #     "amount": 18.00,
    #     "additional_cpt": "36416",
    #     "additional_desc": "Capillary blood draw (bundled per NCCI — violation)",
    #     "pos_code": 11,       # Office — both codes are office-based procedures
    #     "specialty": "Internal Medicine",  # Blood draws are common in Internal Medicine
    #     "note": "CPT 36415 and 36416 billed together — NCCI unbundling violation"
    # },

    # Pattern 2: Unbundling — billing anesthesia + office visit same day
    # CPT 00100 + 99215 is an ACTIVE HARD NCCI edit (confirmed in CMS ccipra-v321r0-f1.xlsx)
    # Rationale: "Standard preparation/monitoring services for anesthesia"
    # Anesthesia services include all patient E&M and monitoring — 99215 cannot
    # be billed separately on the same day as anesthesia. Hard edit: modifier not allowed.
    {
    "fraud_type": "UNBUNDLING",
    "cpt": "00100", "cpt_desc": "Anesthesia for salivary gland surgery",
    "icd": "K11.5",  "icd_desc": "Sialolithiasis (salivary gland stone)",
    "amount": 450.00,
    "additional_cpt": "99215",
    "additional_desc": "Office visit high complexity (bundled — NCCI hard edit violation)",
    "pos_code": 22,  # Outpatient Hospital — surgery setting
    "cpt_expected_specialty": "Anesthesiology",
    "note": "CPT 00100 and 99215 billed same day — anesthesia includes all E&M (NCCI hard edit)"
    },
    
    # Pattern 3: ICD-CPT mismatch — knee replacement but wrong diagnosis
    # POS is Inpatient (21) — the SETTING is correct for knee replacement.
    # The DIAGNOSIS (hip stiffness M25.361) does not justify knee surgery.
    {
        "fraud_type": "ICD_CPT_MISMATCH",
        "cpt": "27447", "cpt_desc": "Total knee replacement",
        "icd": "M25.361", "icd_desc": "Stiffness of right hip (wrong body part)",
        "amount": 12500.00,
        "pos_code": 21,       # Inpatient Hospital — setting correct, diagnosis wrong
        "specialty": "Orthopedic Surgery",  # Only ortho surgeons do knee replacements
        "note": "Knee replacement billed but diagnosis is hip stiffness — ICD-CPT mismatch"
    },
    # Pattern 4: Medically unnecessary — venipuncture with no associated lab
    {
        "fraud_type": "MEDICALLY_UNNECESSARY",
        "cpt": "36415", "cpt_desc": "Venipuncture for blood collection",
        "icd": "Z00.00", "icd_desc": "General adult medical exam",
        "amount": 18.00,
        "pos_code": 11,       # Office
        "specialty": "Family Practice",    # Wellness exams are Family Practice territory
        "note": "Blood draw billed during wellness exam with no associated lab test ordered"
    },
]


# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def random_date(start_year=2024, end_year=2024):
    """Generate a random date within the given year range."""
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")

def random_patient_id():
    return f"PAT-{random.randint(10000, 99999)}"

def random_member_id():
    return f"MBR{random.randint(100000000, 999999999)}"

def random_claim_id(prefix="CLM"):
    return f"{prefix}-FHIR-{random.randint(10000, 99999)}"


# ── FHIR EOB BUILDER ────────────────────────────────────────────────────────

def build_eob(claim_id, patient_id, member_id, provider, payer,
              scenario, service_date, fraud_type=None, fraud_note=None):
    """
    Build a FHIR R4 ExplanationOfBenefit resource.
    Mirrors the structure returned by the CMS BCDA API.

    V2 KEY CHANGE — location field added to each item:
      The FHIR `location.locationCodeableConcept` field carries the
      CMS Place of Service code. The real CMS BCDA API always includes
      this field. We were omitting it in v1, forcing the converter to
      guess POS from CPT codes — an approach that doesn't scale.
      Now we embed POS from the scenario definition, exactly as the
      real API would provide it.
    """
    pos_code    = scenario.get("pos_code", 11)
    pos_display = POS_REFERENCE.get(pos_code, {}).get("display", "Office")

    items = [
        {
            "sequence": 1,
            "productOrService": {
                "coding": [{
                    "system":  "http://www.ama-assn.org/go/cpt",
                    "code":    scenario["cpt"],
                    "display": scenario["cpt_desc"]
                }]
            },
            # ── FHIR location field — V2 addition ──────────────────────────
            # Carries the CMS Place of Service code for this claim item.
            # Standard FHIR R4 path: item.location.locationCodeableConcept
            # This is the primary source of POS in fhir_converter.py.
            # Fallback (CPT range inference) only used if this is absent.
            "location": {
                "locationCodeableConcept": {
                    "coding": [{
                        "system":  "https://www.cms.gov/Medicare/Coding/place-of-service-codes",
                        "code":    str(pos_code),
                        "display": pos_display
                    }]
                }
            },
            "servicedDate": service_date,
            "unitPrice": {
                "value":    scenario["amount"],
                "currency": "USD"
            },
            "net": {
                "value":    scenario["amount"],
                "currency": "USD"
            }
        }
    ]

    # Add second CPT item for unbundling fraud pattern
    if scenario.get("additional_cpt"):
        items.append({
            "sequence": 2,
            "productOrService": {
                "coding": [{
                    "system":  "http://www.ama-assn.org/go/cpt",
                    "code":    scenario["additional_cpt"],
                    "display": scenario["additional_desc"]
                }]
            },
            "location": {
                "locationCodeableConcept": {
                    "coding": [{
                        "system":  "https://www.cms.gov/Medicare/Coding/place-of-service-codes",
                        "code":    str(pos_code),
                        "display": pos_display
                    }]
                }
            },
            "servicedDate": service_date,
            "unitPrice": {"value": 12.00, "currency": "USD"},
            "net":        {"value": 12.00, "currency": "USD"}
        })

    total_amount = sum(item["unitPrice"]["value"] for item in items)

    eob = {
        "resourceType": "ExplanationOfBenefit",
        "id": claim_id,
        "meta": {
            "lastUpdated": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "profile": [
                "http://hl7.org/fhir/us/carin-bb/StructureDefinition/"
                "C4BB-ExplanationOfBenefit-Professional-NonClinician"
            ]
        },
        "identifier": [
            {"system": "https://www.cms.gov/mediaguard/claim-id",  "value": claim_id},
            {"system": "https://www.cms.gov/mediaguard/member-id", "value": member_id}
        ],
        "status": "active",
        "type": {
            "coding": [{
                "system":  "http://terminology.hl7.org/CodeSystem/claim-type",
                "code":    "professional",
                "display": "Professional"
            }]
        },
        "use": "claim",
        "patient":       {"reference": f"Patient/{patient_id}"},
        "billablePeriod": {"start": service_date, "end": service_date},
        "created":       datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "insurer": {
            "identifier": {
                "system": "https://www.cms.gov/mediaguard/payer-id",
                "value":  payer["id"]
            },
            "display": payer["name"]
        },
        "provider": {
            "identifier": {
                "system": "http://hl7.org/fhir/sid/us-npi",
                "value":  provider["npi"]
            },
            "display": provider["name"]
        },
        "outcome": "complete",
        "diagnosis": [
            {
                "sequence": 1,
                "diagnosisCodeableConcept": {
                    "coding": [{
                        "system":  "http://hl7.org/fhir/sid/icd-10-cm",
                        "code":    scenario["icd"],
                        "display": scenario["icd_desc"]
                    }]
                },
                "type": [{
                    "coding": [{
                        "system":  "http://terminology.hl7.org/CodeSystem/ex-diagnosistype",
                        "code":    "principal",
                        "display": "Principal Diagnosis"
                    }]
                }]
            }
        ],
        "item":  items,
        "total": [
            {
                "category": {
                    "coding": [{
                        "system":  "http://terminology.hl7.org/CodeSystem/adjudication",
                        "code":    "submitted",
                        "display": "Submitted Amount"
                    }]
                },
                "amount": {"value": total_amount, "currency": "USD"}
            }
        ],
        "extension": [
            {
                "url":         "https://mediaguard.ai/fhir/extension/provider-specialty",
                "valueString": provider["specialty"]
            },
            {
                "url":         "https://mediaguard.ai/fhir/extension/provider-state",
                "valueString": provider["state"]
            },
            {
                "url":         "https://mediaguard.ai/fhir/extension/plan-name",
                "valueString": payer["plan"]
            },
            # The specialty the CPT procedure requires — comes from the scenario,
            # not the provider. This travels through FHIR → converter → pipeline
            # so the rule engine compares against data, not a hardcoded CPT dict.
            {
                "url":         "https://mediaguard.ai/fhir/extension/cpt-expected-specialty",
                "valueString": scenario.get("specialty", "")
            }
        ]
    }

    # Ground-truth fraud labels for evaluation (not present in real data)
    if fraud_type:
        eob["extension"].append({
            "url":         "https://mediaguard.ai/fhir/extension/injected-fraud-type",
            "valueString": fraud_type
        })
        eob["extension"].append({
            "url":         "https://mediaguard.ai/fhir/extension/fraud-note",
            "valueString": fraud_note or ""
        })

    return eob


# ── MAIN GENERATOR ──────────────────────────────────────────────────────────

def generate_fhir_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bundle_entries = []
    summary_rows   = []
    generated      = 0

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  MEDIAGUARD AI — FHIR R4 Sample Data Generator  (v2)        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Generating {NUM_CLEAN_CLAIMS} clean + {NUM_FRAUD_CLAIMS} FWA claims...")
    print(f"  v2: POS embedded per scenario (see ARCHITECTURAL DECISION LOG)")
    print()

    # ── CLEAN CLAIMS ──
    print("  [1/2] Generating clean legitimate claims...")
    for i in range(NUM_CLEAN_CLAIMS):
        claim_id   = random_claim_id("CLM-CLEAN")
        patient_id = random_patient_id()
        member_id  = random_member_id()
        payer      = random.choice(PAYERS)
        scenario   = random.choice(LEGITIMATE_SCENARIOS)
        svc_date   = random_date()

        # FIX v3: Match provider specialty to scenario specialty.
        # Previously we picked a random provider regardless of specialty,
        # which caused 86% SPECIALTY_MISMATCH flags — the rule engine correctly
        # detected that e.g. a Gastroenterologist was billing knee replacements.
        # Now we filter providers to those whose specialty matches the scenario.
        # Fallback to any provider if no specialty match exists (should not happen
        # with our current provider/scenario set, but guards against future additions).
        matching_providers = [p for p in PROVIDERS if p["specialty"] == scenario["specialty"]]
        provider = random.choice(matching_providers) if matching_providers else random.choice(PROVIDERS)

        eob = build_eob(claim_id, patient_id, member_id,
                        provider, payer, scenario, svc_date)

        fpath = os.path.join(OUTPUT_DIR, f"{claim_id}.json")
        with open(fpath, "w") as f:
            json.dump(eob, f, indent=2)

        bundle_entries.append({"fullUrl": f"urn:uuid:{claim_id}", "resource": eob})
        summary_rows.append({
            "claim_id":     claim_id,
            "patient_id":   patient_id,
            "provider":     provider["name"],
            "specialty":    provider["specialty"],
            "state":        provider["state"],
            "payer":        payer["name"],
            "cpt_code":     scenario["cpt"],
            "cpt_desc":     scenario["cpt_desc"],
            "icd_code":     scenario["icd"],
            "icd_desc":     scenario["icd_desc"],
            "amount":       scenario["amount"],
            "pos_code":     scenario["pos_code"],
            "pos_desc":     POS_REFERENCE[scenario["pos_code"]]["display"],
            "service_date": svc_date,
            "fraud_type":   "NONE",
            "is_fraud":     "NO"
        })
        generated += 1

    print(f"    ✅ {NUM_CLEAN_CLAIMS} clean claims generated")

    # ── FRAUD CLAIMS ──
    print("  [2/2] Injecting FWA claims...")
    fraud_per_pattern = NUM_FRAUD_CLAIMS // len(FWA_SCENARIOS)

    for scenario in FWA_SCENARIOS:
        for _ in range(fraud_per_pattern):
            claim_id   = random_claim_id("CLM-FWA")
            patient_id = random_patient_id()
            member_id  = random_member_id()
            payer      = random.choice(PAYERS)
            svc_date   = random_date()

            # FIX v3: Same specialty-matching logic as clean claims.
            # For FWA scenarios this is especially important — upcoding by
            # an Internal Medicine doctor is clinically plausible and harder
            # to detect. A Gastroenterologist billing 99215 for a cold would
            # fire a specialty mismatch flag that masks the real upcoding signal.
            # FWA scenarios carry a specialty field for exactly this reason.
            fwa_specialty = scenario.get("specialty", None)
            if fwa_specialty:
                matching_providers = [p for p in PROVIDERS if p["specialty"] == fwa_specialty]
                provider = random.choice(matching_providers) if matching_providers else random.choice(PROVIDERS)
            else:
                provider = random.choice(PROVIDERS)

            eob = build_eob(claim_id, patient_id, member_id,
                            provider, payer, scenario, svc_date,
                            fraud_type=scenario["fraud_type"],
                            fraud_note=scenario["note"])

            fpath = os.path.join(OUTPUT_DIR, f"{claim_id}.json")
            with open(fpath, "w") as f:
                json.dump(eob, f, indent=2)

            bundle_entries.append({"fullUrl": f"urn:uuid:{claim_id}", "resource": eob})
            summary_rows.append({
                "claim_id":     claim_id,
                "patient_id":   patient_id,
                "provider":     provider["name"],
                "specialty":    provider["specialty"],
                "state":        provider["state"],
                "payer":        payer["name"],
                "cpt_code":     scenario["cpt"],
                "cpt_desc":     scenario["cpt_desc"],
                "icd_code":     scenario["icd"],
                "icd_desc":     scenario["icd_desc"],
                "amount":       scenario["amount"],
                "pos_code":     scenario["pos_code"],
                "pos_desc":     POS_REFERENCE[scenario["pos_code"]]["display"],
                "service_date": svc_date,
                "fraud_type":   scenario["fraud_type"],
                "is_fraud":     "YES"
            })
            generated += 1

    print(f"    ✅ {NUM_FRAUD_CLAIMS} FWA claims injected ({len(FWA_SCENARIOS)} patterns)")

    # ── FHIR BUNDLE ──
    bundle = {
        "resourceType": "Bundle",
        "id":           str(uuid.uuid4()),
        "meta":         {"lastUpdated": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")},
        "type":         "searchset",
        "total":        generated,
        "link": [{"relation": "self",
                  "url": "https://sandbox.bcda.cms.gov/api/v2/ExplanationOfBenefit"}],
        "entry": bundle_entries
    }

    bundle_path = os.path.join(OUTPUT_DIR, "fhir_bundle.json")
    with open(bundle_path, "w") as f:
        json.dump(bundle, f, indent=2)

    # ── CSV SUMMARY ──
    csv_path = os.path.join(OUTPUT_DIR, "fhir_claims_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    # ── SUMMARY ──
    print()
    print("═" * 62)
    print("  FHIR SAMPLE DATA GENERATION COMPLETE (v2)")
    print("═" * 62)
    print(f"  Total EOB resources : {generated}")
    print(f"  Clean claims        : {NUM_CLEAN_CLAIMS}")
    print(f"  FWA claims          : {NUM_FRAUD_CLAIMS}")
    print(f"  POS field           : ✅ embedded in each item (v2 fix)")
    print()
    print("  FWA Pattern Breakdown:")
    for s in FWA_SCENARIOS:
        pos_desc = POS_REFERENCE[s["pos_code"]]["display"]
        print(f"    {s['fraud_type']:<28} : {fraud_per_pattern} claims  (POS: {pos_desc})")
    print()
    print(f"  Output → {OUTPUT_DIR}/")
    print("═" * 62)
    print()
    print("  Next: Run fhir_converter.py")
    print()


if __name__ == "__main__":
    generate_fhir_dataset()
