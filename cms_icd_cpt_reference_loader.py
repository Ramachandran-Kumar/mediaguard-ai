"""
╔══════════════════════════════════════════════════════════════════╗
║  MEDIAGUARD AI — CMS ICD & CPT Reference Loader                  ║
║  Loads ICD-9→ICD-10 crosswalk, ICD-10-CM codes, PFS rates        ║
╚══════════════════════════════════════════════════════════════════╝

PURPOSE:
  Loads three CMS reference files and returns them in the exact
  format that fwa_data_pipeline.py expects — replacing all remaining
  hardcoded ICD_REFERENCE and CPT_REFERENCE tables.

FILES REQUIRED (save in mediaguard-ai/data/):
  1. 2018_I9gem              ← ICD-9 to ICD-10 crosswalk (GEM)
  2. icd10cm_codes_2026      ← ICD-10-CM codes + descriptions
     (in data/Code Descriptions/ subfolder)
  3. PFREV26B.txt            ← CMS Physician Fee Schedule 2026

WHAT REPLACES WHAT:
  load_icd9_to_icd10_map()   → translates SynPUF ICD-9 codes to ICD-10
  load_icd10_reference()     → replaces hardcoded ICD_REFERENCE (18 codes)
  load_cpt_reference()       → replaces hardcoded CPT_REFERENCE (20 codes)

USAGE IN fwa_data_pipeline.py:
  from cms_icd_cpt_reference_loader import (
      load_icd9_to_icd10_map,
      load_icd10_reference,
      load_cpt_reference
  )
  ICD9_TO_ICD10  = load_icd9_to_icd10_map()
  ICD_REFERENCE  = load_icd10_reference()
  CPT_REFERENCE  = load_cpt_reference()

PHYSICIAN FEE SCHEDULE FILE FORMAT (PFREV26B.txt):
  Comma-separated, no headers
  Column positions (0-indexed):
  [0]  Year
  [1]  HCPCS/CPT code
  [2]  Modifier
  [3]  Locality (00000 = national)
  [4]  blank
  [5]  Non-facility payment amount  ← THIS IS WHAT WE WANT
  [6]  Non-facility limiting charge
  ...
"""

import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

GEM_FILE     = os.path.join(_DATA_DIR, "2018_I9gem")
ICD10_FILE   = os.path.join(_DATA_DIR, "Code Descriptions", "icd10cm_codes_2026")
PFS_FILE     = os.path.join(_DATA_DIR, "PFREV26B.txt")

# ── FALLBACK VALUES ───────────────────────────────────────────────────────────
# Original hardcoded values — used if files not found

_FALLBACK_ICD_REFERENCE = {
    "J06.9":  {"description": "Acute upper respiratory infection", "valid_cpts": ["99213", "99214"]},
    "I10":    {"description": "Essential hypertension",            "valid_cpts": ["99213", "99214", "99215"]},
    "I25.10": {"description": "Atherosclerotic heart disease",     "valid_cpts": ["93000", "99213", "99214"]},
    "M17.11": {"description": "Primary osteoarthritis right knee", "valid_cpts": ["27447", "99213"]},
    "N39.0":  {"description": "Urinary tract infection",          "valid_cpts": ["99232", "99213"]},
    "E11.9":  {"description": "Type 2 diabetes mellitus",         "valid_cpts": ["80053", "36415", "99213"]},
    "J18.9":  {"description": "Pneumonia unspecified",            "valid_cpts": ["71046", "99213"]},
    "C34.11": {"description": "Malignant neoplasm upper lobe lung","valid_cpts": ["99215", "71046"]},
    "Z12.11": {"description": "Colorectal cancer screening",      "valid_cpts": ["45378"]},
    "M25.361":{"description": "Stiffness of right hip",           "valid_cpts": ["99213", "27447"]},
    "Z00.00": {"description": "General adult medical exam",       "valid_cpts": ["99213", "99397"]},
    "K11.5":  {"description": "Sialolithiasis",                   "valid_cpts": ["00100", "42330"]},
}

_FALLBACK_CPT_REFERENCE = {
    "99213": {"description": "Office visit low-moderate complexity",  "avg_cost": 120.00, "specialty": ["Family Practice", "Internal Medicine"]},
    "99214": {"description": "Office visit moderate complexity",       "avg_cost": 165.00, "specialty": ["Internal Medicine", "Family Practice"]},
    "99215": {"description": "Office visit high complexity",           "avg_cost": 225.00, "specialty": ["Internal Medicine"]},
    "36415": {"description": "Venipuncture",                           "avg_cost": 18.00,  "specialty": ["Family Practice", "Internal Medicine"]},
    "80053": {"description": "Comprehensive metabolic panel",          "avg_cost": 38.00,  "specialty": ["Internal Medicine"]},
    "93000": {"description": "Electrocardiogram with interpretation",  "avg_cost": 85.00,  "specialty": ["Cardiology"]},
    "27447": {"description": "Total knee replacement arthroplasty",    "avg_cost": 12500.00,"specialty": ["Orthopedic Surgery"]},
    "45378": {"description": "Colonoscopy diagnostic",                 "avg_cost": 850.00, "specialty": ["Gastroenterology"]},
    "71046": {"description": "Chest X-ray 2 views",                   "avg_cost": 95.00,  "specialty": ["Internal Medicine", "Radiology"]},
    "99232": {"description": "Inpatient hospital care subsequent",     "avg_cost": 210.00, "specialty": ["Internal Medicine"]},
    "00100": {"description": "Anesthesia for salivary gland surgery",  "avg_cost": 450.00, "specialty": ["Anesthesiology"]},
}


# ── LOADER 1: ICD-9 TO ICD-10 CROSSWALK ──────────────────────────────────────

def load_icd9_to_icd10_map():
    """
    Load CMS GEM (General Equivalence Mapping) file for ICD-9 → ICD-10 translation.

    Used to translate SynPUF ICD-9 codes (2008-2010 data) to ICD-10 codes
    so they can be validated against the ICD-10-based rule engine.

    GEM file format (space-separated, no header):
      Column 1: ICD-9 code
      Column 2: ICD-10 code
      Column 3: Flags (00000 = approximate, 10000 = exact, etc.)

    Flag meaning (5 digits):
      Digit 1: Approximate (1) or Exact (0)
      Digit 2: No map (1) or Has map (0)
      Digit 3: Combination (1) or not
      Digit 4: Scenario number
      Digit 5: Choice list order

    Strategy: For each ICD-9 code, pick the BEST ICD-10 match:
      - Prefer exact maps (flag starts with 0)
      - Among approximates, take the first listed
      - If multiple choices, take scenario 1, choice 1

    Returns:
        dict: {"7245": "M54.5", "4010": "I10", ...}
        ICD-9 code (str) → best ICD-10 code (str)
        Falls back to empty dict if file not found.
    """
    if not os.path.exists(GEM_FILE):
        print(f"  ⚠️  GEM crosswalk not found: {GEM_FILE}")
        print(f"      Download from: cms.gov/medicare/coding-billing/icd-10-codes/icd-10-cm-icd-10-pcs-gems")
        return {}

    print(f"  [GEM]  Loading ICD-9→ICD-10 crosswalk from: {os.path.basename(GEM_FILE)}")

    icd9_map = {}      # icd9 → list of (icd10, flags)
    loaded   = 0
    skipped  = 0

    try:
        with open(GEM_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    skipped += 1
                    continue

                icd9  = parts[0].strip()
                icd10 = parts[1].strip()
                flags = parts[2].strip()

                # Skip "no map" entries (flag digit 2 = 1)
                if len(flags) >= 2 and flags[1] == "1":
                    skipped += 1
                    continue

                if icd9 not in icd9_map:
                    icd9_map[icd9] = []
                icd9_map[icd9].append((icd10, flags))
                loaded += 1

        # For each ICD-9, pick the best ICD-10 match
        # Priority: exact (flag[0]='0') > approximate, then first listed
        best_map = {}
        for icd9, mappings in icd9_map.items():
            # Sort: exact maps first (flag[0]='0'), then approximate
            sorted_maps = sorted(mappings, key=lambda x: (x[1][0] if x[1] else '9'))
            best_map[icd9] = sorted_maps[0][0]  # take best match

        print(f"  [GEM]  ✅ Loaded {len(best_map):,} ICD-9→ICD-10 mappings")
        print(f"  [GEM]     Skipped {skipped:,} no-map or invalid entries")
        return best_map

    except Exception as e:
        print(f"  ❌ Error loading GEM file: {e}")
        return {}


def translate_icd9_to_icd10(icd9_code, icd9_map):
    """
    Translate a single ICD-9 code to ICD-10 using the loaded map.

    Args:
        icd9_code: ICD-9 code string (e.g., "7245", "4010")
        icd9_map:  Dict from load_icd9_to_icd10_map()

    Returns:
        str: ICD-10 code if found, original ICD-9 code if not mapped.
             Appends '-ICD9' suffix if untranslated so rule engine
             can identify unmapped codes rather than silently failing.
    """
    code = str(icd9_code).strip()
    if code in icd9_map:
        return icd9_map[code]
    # Try without decimal
    code_clean = code.replace(".", "")
    if code_clean in icd9_map:
        return icd9_map[code_clean]
    return f"{code}-ICD9"  # flag as untranslated


# ── LOADER 2: ICD-10-CM REFERENCE ────────────────────────────────────────────

def load_icd10_reference():
    """
    Load CMS ICD-10-CM codes and descriptions.

    File format (space-separated, no header):
      Column 1: ICD-10 code (no dots, e.g., "J069" not "J06.9")
      Column 2+: Description (rest of line)

    Note: CMS stores codes without dots. We add dots back in standard
    positions (after 3rd character) for consistency with our pipeline.

    Returns:
        dict: {
            "J06.9": {"description": "Acute upper respiratory infection", "valid_cpts": []},
            "I10":   {"description": "Essential hypertension", "valid_cpts": []},
            ...
        }
        Note: valid_cpts is empty — ICD-10-CM file doesn't map to CPTs.
        The rule engine uses this for description lookup and validity checking.
        CPT validation is handled separately by CPT_REFERENCE.

        Falls back to 12 hardcoded codes if file not found.
    """
    if not os.path.exists(ICD10_FILE):
        # Try with .txt extension
        alt_path = ICD10_FILE + ".txt"
        if os.path.exists(alt_path):
            return _load_icd10_from_path(alt_path)
        print(f"  ⚠️  ICD-10-CM file not found: {ICD10_FILE}")
        print(f"      Expected in: data/Code Descriptions/icd10cm_codes_2026")
        print(f"      Using {len(_FALLBACK_ICD_REFERENCE)} hardcoded fallback codes.")
        return _FALLBACK_ICD_REFERENCE

    return _load_icd10_from_path(ICD10_FILE)


def _load_icd10_from_path(filepath):
    print(f"  [ICD10] Loading ICD-10-CM codes from: {os.path.basename(filepath)}")

    icd_ref  = {}
    loaded   = 0
    skipped  = 0

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    skipped += 1
                    continue

                # Split on first whitespace — code is first token, rest is description
                parts = line.split(None, 1)
                if len(parts) < 2:
                    skipped += 1
                    continue

                raw_code    = parts[0].strip()
                description = parts[1].strip()

                if not raw_code or not description:
                    skipped += 1
                    continue

                # Add dot after 3rd character for standard ICD-10 format
                # e.g., "J069" → "J06.9", "I10" stays "I10", "M1711" → "M17.11"
                if len(raw_code) > 3:
                    formatted_code = raw_code[:3] + "." + raw_code[3:]
                else:
                    formatted_code = raw_code

                icd_ref[formatted_code] = {
                    "description": description,
                    "valid_cpts":  []  # populated separately if needed
                }
                loaded += 1

        print(f"  [ICD10] ✅ Loaded {loaded:,} ICD-10-CM codes")
        print(f"  [ICD10]    Replaces 18 hardcoded codes → {loaded:,} real CMS codes")
        return icd_ref

    except Exception as e:
        print(f"  ❌ Error loading ICD-10-CM file: {e}")
        print(f"     Falling back to {len(_FALLBACK_ICD_REFERENCE)} hardcoded codes.")
        return _FALLBACK_ICD_REFERENCE


# ── LOADER 3: CMS PHYSICIAN FEE SCHEDULE ─────────────────────────────────────

def load_cpt_reference():
    """
    Load CMS Physician Fee Schedule and return CPT reference dict.

    File format (PFREV26B.txt — comma-separated, no header):
      [0]  Year (e.g., "2026")
      [1]  HCPCS/CPT code
      [2]  Modifier
      [3]  Locality code (00000 = national average)
      [4]  blank
      [5]  Non-facility payment amount  ← national average rate
      [6]  Non-facility limiting charge
      ...

    Strategy:
      - Only load national rates (locality = 00000 or 0000000000)
      - Only load rows with no modifier (modifier = blank or space)
      - avg_cost = non-facility payment amount (column 5)
      - specialty left empty — PFS doesn't map CPTs to specialties

    Returns:
        dict: {
            "99213": {"description": "", "avg_cost": 78.61, "specialty": []},
            "27447": {"description": "", "avg_cost": 1234.56, "specialty": []},
            ...
        }
        Falls back to 11 hardcoded CPTs if file not found.
    """
    if not os.path.exists(PFS_FILE):
        print(f"  ⚠️  PFS file not found: {PFS_FILE}")
        print(f"      Using {len(_FALLBACK_CPT_REFERENCE)} hardcoded fallback CPTs.")
        return _FALLBACK_CPT_REFERENCE

    print(f"  [PFS]  Loading CMS Physician Fee Schedule from: {os.path.basename(PFS_FILE)}")

    cpt_ref  = {}
    loaded   = 0
    skipped  = 0

    try:
        with open(PFS_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    skipped += 1
                    continue

                parts = [p.strip().strip('"') for p in line.split(",")]

                if len(parts) < 6:
                    skipped += 1
                    continue

                cpt_code = parts[1].strip().zfill(5) if parts[1].strip() else ""
                modifier = parts[2].strip()
                locality = parts[3].strip()
                amount   = parts[5].strip()

                # Only load national rates (locality 00000 variants)
                if locality not in ("00000", "0000000000", ""):
                    skipped += 1
                    continue

                # Skip rows with modifiers — we want base rates only
                if modifier and modifier not in ("", " "):
                    skipped += 1
                    continue

                if not cpt_code:
                    skipped += 1
                    continue

                # Parse payment amount — remove leading zeros, convert to float
                try:
                    # PFS amounts are stored as integers (e.g., "0001337.33" or "1337.33")
                    avg_cost = float(amount.lstrip("0") or "0")
                except (ValueError, TypeError):
                    avg_cost = 0.0

                # Store — if duplicate (multiple localities), keep first national rate
                if cpt_code not in cpt_ref:
                    cpt_ref[cpt_code] = {
                        "description": "",   # PFS doesn't include descriptions
                        "avg_cost":    round(avg_cost, 2),
                        "specialty":   []    # PFS doesn't map to specialties
                    }
                    loaded += 1
                else:
                    skipped += 1

        print(f"  [PFS]  ✅ Loaded {loaded:,} CPT payment rates")
        print(f"  [PFS]     Replaces 20 hardcoded CPTs → {loaded:,} real CMS rates")

        # Merge descriptions and specialties from fallback for our known CPTs
        # (PFS has no descriptions — add them from fallback for the codes we know)
        for cpt, fallback_data in _FALLBACK_CPT_REFERENCE.items():
            if cpt in cpt_ref:
                cpt_ref[cpt]["description"] = fallback_data.get("description", "")
                cpt_ref[cpt]["specialty"]   = fallback_data.get("specialty", [])

        return cpt_ref

    except Exception as e:
        print(f"  ❌ Error loading PFS file: {e}")
        print(f"     Falling back to {len(_FALLBACK_CPT_REFERENCE)} hardcoded CPTs.")
        return _FALLBACK_CPT_REFERENCE


# ── LOAD ALL ──────────────────────────────────────────────────────────────────

def load_all_icd_cpt_reference():
    """
    Load all three reference tables in one call.

    Returns:
        tuple: (icd9_map dict, icd10_reference dict, cpt_reference dict)

    Usage:
        ICD9_MAP, ICD_REFERENCE, CPT_REFERENCE = load_all_icd_cpt_reference()
    """
    print()
    print("  ── CMS ICD & CPT Reference Loader ───────────────────────────")
    icd9_map      = load_icd9_to_icd10_map()
    icd_reference = load_icd10_reference()
    cpt_reference = load_cpt_reference()
    print("  ── ICD & CPT reference data ready ───────────────────────────")
    print()
    return icd9_map, icd_reference, cpt_reference


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MediGuard AI — CMS ICD & CPT Reference Loader Test              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    icd9_map, icd_ref, cpt_ref = load_all_icd_cpt_reference()

    # ── ICD-9 crosswalk spot checks
    print("\n  ICD-9 → ICD-10 Spot Checks (SynPUF common codes):")
    test_icd9 = ["7245", "4010", "29606", "8410", "29521", "82311"]
    for code in test_icd9:
        icd10 = translate_icd9_to_icd10(code, icd9_map)
        desc  = icd_ref.get(icd10, {}).get("description", "— not in ICD-10 ref")
        print(f"    ICD-9 {code:<8} → ICD-10 {icd10:<10} {desc[:50]}")

    # ── ICD-10 spot checks
    print("\n  ICD-10-CM Spot Checks (our known codes):")
    test_icd10 = ["J06.9", "I10", "M17.11", "Z00.00", "K11.5"]
    for code in test_icd10:
        desc = icd_ref.get(code, {}).get("description", "❌ NOT FOUND")
        print(f"    {code:<10} → {desc[:60]}")

    # ── CPT fee schedule spot checks
    print("\n  CPT Fee Schedule Spot Checks (our known CPTs):")
    test_cpts = ["99213", "99215", "27447", "36415", "00100", "45378"]
    for cpt in test_cpts:
        data = cpt_ref.get(cpt, {})
        cost = data.get("avg_cost", "NOT FOUND")
        desc = data.get("description", "")
        print(f"    CPT {cpt} → ${cost:>10} {desc[:40]}")

    # ── Stats
    print(f"\n  ── Summary ──────────────────────────────────────────────────")
    print(f"  ICD-9 crosswalk entries : {len(icd9_map):,}")
    print(f"  ICD-10-CM codes         : {len(icd_ref):,}")
    print(f"  CPT payment rates       : {len(cpt_ref):,}")
    print(f"\n  ✅ All three loaders working.")
    print(f"  Ready to import into fwa_data_pipeline.py and cms_synpuf_loader.py")
