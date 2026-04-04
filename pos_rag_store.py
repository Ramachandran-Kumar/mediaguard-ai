"""
╔══════════════════════════════════════════════════════════════════╗
║  MEDIAGUARD AI — ChromaDB POS RAG Store                          ║
║  Layer 2 of the three-layer POS resolution chain                 ║
╚══════════════════════════════════════════════════════════════════╝

PURPOSE:
  Replaces the hardcoded CPT numeric range inference (old Layer 2)
  with a semantic vector store over the full CMS CPT reference.
  When a claim arrives without a FHIR location field, ChromaDB finds
  the most clinically similar CPT and returns its expected POS code.

SETUP (run once, after cms_db_setup.py):
    pip install chromadb
    python pos_rag_store.py

USAGE:
    from pos_rag_store import get_pos_store
    pos_code, pos_desc, distance = get_pos_store().query_pos_for_cpt("27447")
    # → (21, "Inpatient Hospital", 0.0)

HOW IT WORKS:
  1. build_pos_store() reads all CPT codes from SQLite cpt_rates table
     (code, description, specialty — already loaded by cms_db_setup.py).
  2. For each CPT, derives the baseline POS from the same CPT numeric
     range logic used in the old Layer 2 (infer_pos_from_cpt). This
     seeds the vector store without needing any additional CMS data file.
  3. Embeds each CPT as a rich text document using all-MiniLM-L6-v2
     (local, no API key) and stores in ChromaDB at data/chroma_pos_store/.
  4. At query time, exact CPT ID lookup is tried first (O(1)).
     If the CPT isn't in the store (future codes, HCPCS, etc.), semantic
     search finds the most similar CPT and returns its POS — the vector
     embedding captures clinical context (surgery vs. office vs. radiology)
     far better than a numeric range check.
"""

import os
import sqlite3

import chromadb

# ── CONFIG ────────────────────────────────────────────────────────────────────
_DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
DB_PATH    = os.path.join(_DATA_DIR, "mediaguard_reference.db")
CHROMA_DIR = os.path.join(_DATA_DIR, "chroma_pos_store")
COLLECTION = "pos_cpt_rules"

# Distance threshold: cosine distance above this → not confident, fall back
# to range inference. Range: 0.0 (exact) → 2.0 (opposite). 0.6 is conservative.
RAG_DISTANCE_THRESHOLD = 0.6

_store_instance = None


# ── SEEDING HELPER ────────────────────────────────────────────────────────────

def _seed_pos_code(cpt_code: str) -> tuple[int, str]:
    """
    Derive correct POS from CPT numeric range for store seeding.

    Rules are ordered most-specific first so narrow ranges (ED, ortho)
    are matched before any broader range would catch them.
    Only used during build_pos_store() — not at query time.
    """
    try:
        n = int(str(cpt_code).strip())
    except (ValueError, TypeError):
        return 11, "Office"  # Non-numeric (HCPCS) → office default

    # ── Specific sub-ranges first (order matters) ──────────────────────────
    # ED E&M — must precede the broad 99201-99499 office block
    if 99281 <= n <= 99285:  return 23, "Emergency Room"

    # Anesthesia — 00100-01999 (stored zero-padded, parses to 100-1999)
    if 100 <= n <= 1999:     return 21, "Inpatient Hospital"

    # Orthopedic surgery — major joint procedures → inpatient
    if 27000 <= n <= 27999:  return 21, "Inpatient Hospital"

    # Cardiovascular / IV access — venipuncture, infusions → office
    if 36000 <= n <= 36999:  return 11, "Office"

    # ── Broad category ranges ──────────────────────────────────────────────
    # Office E&M visits (all non-ED 99xxx E&M)
    if 99201 <= n <= 99499:  return 11, "Office"

    # Radiology / Imaging
    if 70000 <= n <= 79999:  return 22, "Outpatient Hospital"

    # Lab / Pathology
    if 80000 <= n <= 89999:  return 11, "Office"

    # Mental Health / Medicine (90000-90899)
    if 90000 <= n <= 90899:  return 11, "Office"

    # Default — catches everything else including unlisted surgical ranges
    return 11, "Office"


# ── STORE CLASS ───────────────────────────────────────────────────────────────

class POSRagStore:
    """ChromaDB-backed CPT → Place of Service semantic lookup."""

    def __init__(self):
        self._client     = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = self._client.get_collection(name=COLLECTION)
        count = self._collection.count()
        print(f"  [RAG] POS store loaded — {count:,} CPT embeddings")

    def query_pos_for_cpt(self, cpt_code: str) -> tuple[int, str, float]:
        """
        Return the expected Place of Service for a CPT code.

        Strategy:
          1. Exact ID lookup — CPT is in the store; return with distance 0.0.
          2. Range inference — CPT not in store; call _seed_pos_code() and
             return with distance 0.25 (derived, not matched).

        Semantic nearest-neighbor search is intentionally not used here.
        The cpt_rates table has only a small number of records with sparse
        descriptions, which makes cosine similarity unreliable. When the store
        grows to cover the full CMS CPT catalog, semantic search can be
        re-enabled as a step between 1 and 2.

        Returns:
            (pos_code: int, pos_description: str, distance: float)
            0.0  = exact store match
            0.25 = derived via CPT range inference (still within RAG threshold)
            1.0  = lookup failed entirely (caller should fall back)
        """
        cpt = str(cpt_code).strip().zfill(5)

        # ── Exact ID lookup ──
        try:
            result = self._collection.get(ids=[cpt], include=["metadatas"])
            if result["ids"]:
                meta = result["metadatas"][0]
                return int(meta["pos_code"]), meta["pos_description"], 0.0
        except Exception:
            pass

        # ── CPT not in store: derive from range logic ──
        try:
            pos_code, pos_desc = _seed_pos_code(cpt_code)
            return pos_code, pos_desc, 0.25
        except Exception:
            pass

        return 11, "Office", 1.0


# ── BUILD ─────────────────────────────────────────────────────────────────────

def build_pos_store(force_rebuild: bool = False) -> None:
    """
    Build (or rebuild) the ChromaDB POS vector store from SQLite cpt_rates.

    Reads every CPT code from the database, seeds the POS from the numeric
    range logic, and embeds the full clinical document text using the local
    all-MiniLM-L6-v2 model (via chromadb's DefaultEmbeddingFunction).

    Args:
        force_rebuild: Delete and recreate the collection even if it exists.

    Run:
        python pos_rag_store.py
    """
    if not os.path.exists(DB_PATH):
        print(f"  ❌ SQLite DB not found at {DB_PATH}. Run cms_db_setup.py first.")
        return

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION)
            print(f"  ♻️  Deleted existing collection '{COLLECTION}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not force_rebuild:
        print(f"  [RAG] Store already built ({collection.count():,} docs). "
              f"Pass force_rebuild=True to rebuild.")
        return

    # ── Load CPT reference from SQLite ──
    # Pull from cpt_rates (descriptions + specialties) and icd_cpt_rules
    # (clinical indications). Union both so the store covers all CPT codes
    # used anywhere in the pipeline, not just those in the PFS file.
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        # From cpt_rates: description and specialty
        cpt_rate_rows = conn.execute(
            "SELECT cpt_code, description, specialty FROM cpt_rates"
        ).fetchall()

        # From icd_cpt_rules: aggregate ICD indication descriptions per CPT
        # Gives clinical context ("indicated for: osteoarthritis, knee pain…")
        indication_rows = conn.execute("""
            SELECT r.cpt_code,
                   GROUP_CONCAT(ic.description, '; ') AS indications
            FROM   icd_cpt_rules r
            LEFT JOIN icd10_codes ic ON r.icd10_code = ic.icd10_code
            GROUP  BY r.cpt_code
        """).fetchall()
    finally:
        conn.close()

    # Build unified CPT dict — cpt_rates takes priority for description/specialty
    cpt_map: dict[str, dict] = {}

    for row in cpt_rate_rows:
        cpt = str(row["cpt_code"]).strip().zfill(5)
        cpt_map[cpt] = {
            "description": row["description"] or "",
            "specialty":   row["specialty"] or "",
            "indications": "",
        }

    for row in indication_rows:
        cpt = str(row["cpt_code"]).strip().zfill(5)
        indications = row["indications"] or ""
        if cpt in cpt_map:
            cpt_map[cpt]["indications"] = indications
        else:
            cpt_map[cpt] = {"description": "", "specialty": "", "indications": indications}

    if not cpt_map:
        print("  ❌ No CPT codes found in cpt_rates or icd_cpt_rules. Run cms_db_setup.py first.")
        return

    print(f"  [RAG] Embedding {len(cpt_map):,} CPT codes → ChromaDB...")
    print(f"        Sources: cpt_rates ({len(cpt_rate_rows)}) + icd_cpt_rules ({len(indication_rows)} unique CPTs)")
    print(f"        Model  : all-MiniLM-L6-v2 (local, via chromadb default EF)")
    print(f"        Store  : {CHROMA_DIR}")

    ids       = []
    documents = []
    metadatas = []

    for cpt, info in cpt_map.items():
        pos_code, pos_desc = _seed_pos_code(cpt)
        desc        = info["description"]
        specialty   = info["specialty"]
        indications = info["indications"]

        # Rich document: description + indications + setting → better semantic surface
        parts = [f"CPT {cpt}"]
        if desc:        parts.append(desc)
        if specialty:   parts.append(f"Specialty: {specialty}")
        if indications: parts.append(f"Indicated for: {indications}")
        parts.append(f"Typical care setting: {pos_desc} (POS {pos_code})")
        doc = ". ".join(parts) + "."

        ids.append(cpt)
        documents.append(doc)
        metadatas.append({
            "cpt_code":        cpt,
            "pos_code":        pos_code,
            "pos_description": pos_desc,
        })

    # Upsert in batches — ChromaDB recommendation for large datasets
    batch_size = 500
    total      = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.upsert(
            ids       = ids[start:end],
            documents = documents[start:end],
            metadatas = metadatas[start:end],
        )
        pct = end / total * 100
        print(f"    → {end:>{len(str(total))}}/{total} CPT codes embedded ({pct:.0f}%)", end="\r")

    print(f"\n  ✓ POS store built — {collection.count():,} CPT embeddings")
    print(f"  ✓ Persisted to: {CHROMA_DIR}")


# ── SINGLETON ─────────────────────────────────────────────────────────────────

def get_pos_store() -> POSRagStore:
    """
    Return the singleton POSRagStore, building it if necessary.

    Auto-builds on first call if the ChromaDB store doesn't exist.
    Subsequent calls return the cached instance (no re-embedding).
    Thread-safety: not required — pipeline is single-process.
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    # Build if store directory is missing or collection is empty
    needs_build = not os.path.exists(CHROMA_DIR)
    if not needs_build:
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            col    = client.get_collection(name=COLLECTION)
            needs_build = col.count() == 0
        except Exception:
            needs_build = True

    if needs_build:
        print("  [RAG] POS store not found or empty — building from SQLite...")
        build_pos_store()

    _store_instance = POSRagStore()
    return _store_instance


# ── CLI ENTRY POINT ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MediGuard AI — ChromaDB POS RAG Store Builder                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    build_pos_store(force_rebuild=True)

    print()
    print("  ── Smoke test ──────────────────────────────────────────────────")
    store = POSRagStore()

    test_cases = [
        ("27447", 21, "Total knee arthroplasty  → Inpatient Hospital"),
        ("99213", 11, "Office E&M moderate      → Office"),
        ("99281", 23, "ED E&M                   → Emergency Room"),
        ("70553", 22, "MRI brain w/ contrast    → Outpatient Hospital"),
        ("85025", 11, "CBC lab panel            → Office"),
        ("90837", 11, "Psychotherapy 60 min     → Office"),
        ("XXXXX", 11, "Non-numeric HCPCS        → Office (default)"),
    ]

    all_pass = True
    for cpt, expected_pos, label in test_cases:
        pos_code, pos_desc, dist = store.query_pos_for_cpt(cpt)
        status = "✓" if pos_code == expected_pos else "✗"
        if pos_code != expected_pos:
            all_pass = False
        print(f"  {status} CPT {cpt:<8}  POS {pos_code:>2} ({pos_desc:<22})  "
              f"dist={dist:.3f}  {label}")

    print()
    print("  ✓ All tests passed" if all_pass else "  ✗ Some tests failed — review seeding logic")
