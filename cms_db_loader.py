"""
╔══════════════════════════════════════════════════════════════════╗
║  MEDIAGUARD AI — CMS Reference Database Loader                   ║
║  Fast SQLite queries — replaces all flat file loaders            ║
╚══════════════════════════════════════════════════════════════════╝

PURPOSE:
  Queries the SQLite database built by cms_db_setup.py.
  Called by fwa_data_pipeline.py and cms_synpuf_loader.py.
  Millisecond lookups vs seconds of flat file reading every run.

PREREQUISITE:
  Run cms_db_setup.py first to build the database.

USAGE:
  from cms_db_loader import load_all_from_db, translate_icd9, get_icd_description

  # Load all reference tables at pipeline startup (fast — DB query)
  NCCI_BUNDLES, MUE_LIMITS, ICD_REFERENCE, CPT_REFERENCE, ICD9_MAP, ICD_VALID_CPTS = load_all_from_db()

  # Translate a single ICD-9 code on the fly
  icd10 = translate_icd9("7245")   # returns "M54.5"

  # Look up ICD description
  desc = get_icd_description("J06.9")  # returns "Acute upper respiratory infection"
"""

import os
import sqlite3

# ── CONFIG ────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH   = os.path.join(_DATA_DIR, "mediaguard_reference.db")

# ── FALLBACKS (if DB not built yet) ──────────────────────────────────────────
_FALLBACK_NCCI = {
    ("36415", "36416"), ("99213", "99201"), ("99214", "99201"),
    ("99214", "99202"), ("99215", "99201"), ("99215", "99202"),
}
_FALLBACK_MUE = {
    "99213": 1, "99214": 1, "99215": 1,
    "36415": 1, "80053": 1, "93000": 1, "27447": 1,
}
_FALLBACK_ICD = {
    "J06.9":  {"description": "Acute upper respiratory infection", "valid_cpts": []},
    "I10":    {"description": "Essential hypertension",            "valid_cpts": []},
    "M17.11": {"description": "Primary osteoarthritis right knee", "valid_cpts": []},
    "Z00.00": {"description": "General adult medical exam",        "valid_cpts": []},
    "K11.5":  {"description": "Sialolithiasis",                   "valid_cpts": []},
}
_FALLBACK_CPT = {
    "99213": {"description": "Office visit low-moderate complexity", "avg_cost": 120.00, "specialty": ["Family Practice"]},
    "99215": {"description": "Office visit high complexity",          "avg_cost": 225.00, "specialty": ["Internal Medicine"]},
    "27447": {"description": "Total knee replacement",                "avg_cost": 12500.00,"specialty": ["Orthopedic Surgery"]},
    "00100": {"description": "Anesthesia for salivary gland surgery", "avg_cost": 450.00, "specialty": ["Anesthesiology"]},
}


# ── CONNECTION ────────────────────────────────────────────────────────────────

def _get_conn():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_exists():
    return os.path.exists(DB_PATH)


# ── LOADERS ───────────────────────────────────────────────────────────────────

def load_ncci_from_db():
    """Load NCCI bundling pairs as a set of (col1, col2) tuples."""
    conn = _get_conn()
    if not conn:
        print("  ⚠️  DB not found — using fallback NCCI pairs. Run cms_db_setup.py first.")
        return _FALLBACK_NCCI
    try:
        rows = conn.execute("SELECT col1_cpt, col2_cpt FROM ncci_bundles").fetchall()
        result = {(r["col1_cpt"], r["col2_cpt"]) for r in rows}
        print(f"  [DB] NCCI bundles  : {len(result):,} pairs loaded")
        return result
    except Exception as e:
        print(f"  ⚠️  NCCI DB query failed: {e} — using fallback")
        return _FALLBACK_NCCI
    finally:
        conn.close()


def load_mue_from_db():
    """Load MUE limits as dict: cpt_code → max_units."""
    conn = _get_conn()
    if not conn:
        print("  ⚠️  DB not found — using fallback MUE limits. Run cms_db_setup.py first.")
        return _FALLBACK_MUE
    try:
        rows = conn.execute("SELECT cpt_code, mue_value FROM mue_limits").fetchall()
        result = {r["cpt_code"]: r["mue_value"] for r in rows}
        print(f"  [DB] MUE limits    : {len(result):,} CPT codes loaded")
        return result
    except Exception as e:
        print(f"  ⚠️  MUE DB query failed: {e} — using fallback")
        return _FALLBACK_MUE
    finally:
        conn.close()


def load_icd10_from_db():
    """Load ICD-10-CM codes as dict: code → {description, valid_cpts}."""
    conn = _get_conn()
    if not conn:
        print("  ⚠️  DB not found — using fallback ICD codes. Run cms_db_setup.py first.")
        return _FALLBACK_ICD
    try:
        rows = conn.execute("SELECT icd10_code, description FROM icd10_codes").fetchall()
        result = {
            r["icd10_code"]: {"description": r["description"], "valid_cpts": []}
            for r in rows
        }
        print(f"  [DB] ICD-10 codes  : {len(result):,} codes loaded")
        return result
    except Exception as e:
        print(f"  ⚠️  ICD-10 DB query failed: {e} — using fallback")
        return _FALLBACK_ICD
    finally:
        conn.close()


def load_cpt_from_db():
    """Load CPT rates as dict: cpt_code → {description, avg_cost, specialty}."""
    conn = _get_conn()
    if not conn:
        print("  ⚠️  DB not found — using fallback CPT rates. Run cms_db_setup.py first.")
        return _FALLBACK_CPT
    try:
        rows = conn.execute("SELECT cpt_code, avg_cost, description, specialty FROM cpt_rates").fetchall()
        result = {}
        for r in rows:
            specialty = r["specialty"].split(",") if r["specialty"] else []
            result[r["cpt_code"]] = {
                "description": r["description"] or "",
                "avg_cost":    r["avg_cost"],
                "specialty":   [s for s in specialty if s]
            }
        print(f"  [DB] CPT rates     : {len(result):,} CPT codes loaded")
        return result
    except Exception as e:
        print(f"  ⚠️  CPT DB query failed: {e} — using fallback")
        return _FALLBACK_CPT
    finally:
        conn.close()


def load_icd_cpt_rules_from_db():
    """
    Load ICD-CPT validation rules as dict: icd10_code → list of valid CPT codes.

    These are domain-curated rules stored in the icd_cpt_rules table.
    Used by the rule engine to detect ICD_CPT_MISMATCH violations.

    Returns:
        dict: {"J06.9": ["99212", "99213"], "M17.11": ["27447", "29881", "97110"], ...}
        Empty dict if DB not found.
    """
    conn = _get_conn()
    if not conn:
        print("  ⚠️  DB not found — ICD-CPT rules unavailable. Run cms_db_setup.py first.")
        return {}
    try:
        rows = conn.execute(
            "SELECT icd10_code, cpt_code FROM icd_cpt_rules ORDER BY icd10_code"
        ).fetchall()

        result = {}
        for r in rows:
            icd = r["icd10_code"]
            cpt = r["cpt_code"]
            if icd not in result:
                result[icd] = []
            result[icd].append(cpt)

        print(f"  [DB] ICD-CPT rules : {len(rows):,} pairs across {len(result):,} ICD codes loaded")
        return result
    except Exception as e:
        print(f"  ⚠️  ICD-CPT rules DB query failed: {e}")
        return {}
    finally:
        conn.close()



    """Load ICD-9→ICD-10 crosswalk as dict: icd9_code → icd10_code."""
    conn = _get_conn()
    if not conn:
        print("  ⚠️  DB not found — ICD-9 translation unavailable. Run cms_db_setup.py first.")
        return {}
    try:
        rows = conn.execute("SELECT icd9_code, icd10_code FROM icd9_crosswalk").fetchall()
        result = {r["icd9_code"]: r["icd10_code"] for r in rows}
        print(f"  [DB] ICD-9 crosswalk: {len(result):,} mappings loaded")
        return result
    except Exception as e:
        print(f"  ⚠️  ICD-9 crosswalk DB query failed: {e}")
        return {}
    finally:
        conn.close()


# ── SINGLE-CODE LOOKUPS (for use in synpuf_loader per-row translation) ────────

def load_icd9_map_from_db():
    """Load ICD-9→ICD-10 crosswalk as dict: icd9_code → icd10_code."""
    conn = _get_conn()
    if not conn:
        print("  ⚠️  DB not found — ICD-9 translation unavailable. Run cms_db_setup.py first.")
        return {}
    try:
        rows = conn.execute("SELECT icd9_code, icd10_code FROM icd9_crosswalk").fetchall()
        result = {r["icd9_code"]: r["icd10_code"] for r in rows}
        print(f"  [DB] ICD-9 crosswalk: {len(result):,} mappings loaded")
        return result
    except Exception as e:
        print(f"  ⚠️  ICD-9 crosswalk DB query failed: {e}")
        return {}
    finally:
        conn.close()


def translate_icd9(icd9_code):
    """
    Translate a single ICD-9 code to ICD-10.
    Queries DB directly — use for per-row translation in synpuf_loader.

    Returns:
        str: ICD-10 code if found, original ICD-9 with '-ICD9' suffix if not.
    """
    conn = _get_conn()
    if not conn:
        return f"{icd9_code}-ICD9"
    try:
        code = str(icd9_code).strip()
        row  = conn.execute(
            "SELECT icd10_code FROM icd9_crosswalk WHERE icd9_code = ?", (code,)
        ).fetchone()
        if row:
            return row["icd10_code"]
        # Try without leading zeros
        row = conn.execute(
            "SELECT icd10_code FROM icd9_crosswalk WHERE icd9_code = ?", (code.lstrip("0"),)
        ).fetchone()
        return row["icd10_code"] if row else f"{code}-ICD9"
    except Exception:
        return f"{icd9_code}-ICD9"
    finally:
        conn.close()


def get_icd_description(icd10_code):
    """Look up ICD-10 description from DB. Returns empty string if not found."""
    conn = _get_conn()
    if not conn:
        return ""
    try:
        row = conn.execute(
            "SELECT description FROM icd10_codes WHERE icd10_code = ?", (icd10_code,)
        ).fetchone()
        return row["description"] if row else ""
    except Exception:
        return ""
    finally:
        conn.close()


def get_cpt_avg_cost(cpt_code):
    """Look up CPT average cost from DB. Falls back to _FALLBACK_CPT, then 0.0."""
    conn = _get_conn()
    if not conn:
        return _FALLBACK_CPT.get(cpt_code, {}).get("avg_cost", 0.0)
    try:
        row = conn.execute(
            "SELECT avg_cost FROM cpt_rates WHERE cpt_code = ?", (cpt_code,)
        ).fetchone()
        if row and row["avg_cost"]:
            return row["avg_cost"]
        return _FALLBACK_CPT.get(cpt_code, {}).get("avg_cost", 0.0)
    except Exception:
        return _FALLBACK_CPT.get(cpt_code, {}).get("avg_cost", 0.0)
    finally:
        conn.close()


def is_valid_ncci_pair(cpt1, cpt2):
    """Check if a CPT pair is an active NCCI hard edit. Fast single-query lookup."""
    conn = _get_conn()
    if not conn:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM ncci_bundles WHERE col1_cpt = ? AND col2_cpt = ?",
            (str(cpt1).zfill(5), str(cpt2).zfill(5))
        ).fetchone()
        return row is not None
    except Exception:
        return False
    finally:
        conn.close()


# ── LOAD ALL ──────────────────────────────────────────────────────────────────

def load_all_from_db():
    """
    Load all six reference tables in one call.
    Replaces all individual CMS flat file loaders in fwa_data_pipeline.py.

    Returns:
        tuple: (ncci_bundles, mue_limits, icd_reference, cpt_reference, icd9_map, icd_valid_cpts)

    Usage in fwa_data_pipeline.py:
        from cms_db_loader import load_all_from_db
        NCCI_BUNDLES, MUE_LIMITS, ICD_REFERENCE, CPT_REFERENCE, ICD9_MAP, ICD_VALID_CPTS = load_all_from_db()
    """
    if not db_exists():
        print("  ❌ Reference database not found.")
        print("     Run: python cms_db_setup.py")
        print("     Then re-run the pipeline.")
        print("     Falling back to hardcoded values for now.")
        return _FALLBACK_NCCI, _FALLBACK_MUE, _FALLBACK_ICD, _FALLBACK_CPT, {}, {}

    print()
    print("  ── CMS Reference Database ────────────────────────────────────")
    ncci         = load_ncci_from_db()
    mue          = load_mue_from_db()
    icd_ref      = load_icd10_from_db()
    cpt_ref      = load_cpt_from_db()
    icd9_map     = load_icd9_map_from_db()
    icd_valid    = load_icd_cpt_rules_from_db()
    print("  ── Reference data ready ──────────────────────────────────────")
    print()
    return ncci, mue, icd_ref, cpt_ref, icd9_map, icd_valid


# ── DB STATUS ─────────────────────────────────────────────────────────────────

def print_db_status():
    """Print current database status — table counts and last load times."""
    if not db_exists():
        print("  ❌ Database not found. Run: python cms_db_setup.py")
        return

    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT table_name, record_count, loaded_at, source_file FROM db_metadata ORDER BY table_name"
        ).fetchall()
        print(f"\n  MediGuard AI Reference Database — {DB_PATH}")
        print(f"  {'Table':<20} {'Records':>10}  {'Loaded At':<22}  Source")
        print(f"  {'─'*20} {'─'*10}  {'─'*22}  {'─'*30}")
        for r in rows:
            print(f"  {r['table_name']:<20} {r['record_count']:>10,}  {r['loaded_at']:<22}  {r['source_file']}")
        db_size = os.path.getsize(DB_PATH) / (1024*1024)
        print(f"\n  Database size: {db_size:.1f} MB")
    except Exception as e:
        print(f"  Error reading DB status: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MediGuard AI — CMS Reference Database Status                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print_db_status()

    if db_exists():
        print("\n  Quick lookup tests:")
        print(f"  ICD-9 '7245'   → {translate_icd9('7245')}")
        print(f"  ICD-9 '4010'   → {translate_icd9('4010')}")
        print(f"  ICD-10 'J06.9' → {get_icd_description('J06.9')}")
        print(f"  CPT 99215 avg  → ${get_cpt_avg_cost('99215'):.2f}")
        print(f"  NCCI 00100+99215 → {is_valid_ncci_pair('00100','99215')}")
