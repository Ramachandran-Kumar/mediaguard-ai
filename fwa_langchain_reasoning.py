"""
MediGuard AI — LangChain Gen AI Reasoning Layer
File: fwa_langchain_reasoning.py

Layer 4 (Gen AI): Takes high-priority claims from the data pipeline
and uses an LLM (BioMistral via Ollama or Claude/OpenAI via API)
to generate clinical reasoning, fraud explanations, and recommendations.

Requirements:
    pip install langchain langchain-community pandas chromadb requests

For local model (free, recommended for POC):
    1. Install Ollama: https://ollama.ai
    2. Pull model: ollama pull biomistral  (or: ollama pull mistral)
    3. Set LLM_PROVIDER = "ollama" below

For Groq API (FREE, fastest — recommended for development):
    1. Sign up at console.groq.com (free, no credit card)
    2. Generate API key
    3. Set LLM_PROVIDER = "groq" below
    4. pip install groq
    5. Set GROQ_API_KEY environment variable
       Windows:      set GROQ_API_KEY=gsk_your_key_here
       macOS/Linux:  export GROQ_API_KEY=gsk_your_key_here

For Claude API (best reasoning quality, small cost):
    1. Set LLM_PROVIDER = "claude"
    2. Set ANTHROPIC_API_KEY environment variable

Run:
    python fwa_langchain_reasoning.py
"""

import os
import json
import time
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field, asdict

# ──────────────────────────────────────────────
# CONFIG — Change these for your setup
# ──────────────────────────────────────────────

LLM_PROVIDER = "groq"           # "groq" | "ollama" | "claude" | "mock"
                                # ↑ RECOMMENDED: Groq is free, fastest, no install needed

# Groq settings (FREE — get key at console.groq.com)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Best reasoning on Groq free tier
                                         # Alt: "mixtral-8x7b-32768" | "gemma2-9b-it"

# Ollama settings (free local model — requires Ollama installed)
OLLAMA_MODEL = "mistral"        # "biomistral" | "meditron" | "mistral" | "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

# API keys for cloud providers
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

INPUT_FILE = "output/ai_review_queue.csv"
OUTPUT_FILE = "output/fwa_ai_report.csv"
MAX_CLAIMS_TO_ANALYZE = 20      # Set to 1 for debug run — change back to 20 after fix
#MIN_RISK_SCORE = 70             # Only analyze claims above this threshold
# Queue already pre-filtered by pipeline (score ≥ 70 OR HIGH severity flag).
# Set to 0 here so all queued claims reach Groq — filtering happened upstream.
MIN_RISK_SCORE = 0

ENABLE_VECTOR_SEARCH = False    # Set True if ChromaDB installed (pip install chromadb)


# ──────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class ClaimRecord:
    claim_id: str
    patient_age: int
    patient_gender: str
    provider_npi: str
    provider_name: str
    provider_specialty: str
    provider_state: str
    date_of_service: str
    cpt_code: str
    cpt_description: str
    icd_primary: str
    icd_description: str
    billed_amount: float
    units: int
    pos_description: str
    composite_risk_score: float
    rule_flag_count: int
    rule_flag_severity: str
    rule_flags: list
    pct_99215: float = 0.0
    weekend_billing_rate: float = 0.0
    provider_vs_specialty_ratio: float = 1.0
    fraud_label: str = "UNKNOWN"


@dataclass
class FWAAnalysisResult:
    claim_id: str
    risk_score: int = 0
    fraud_category: str = ""
    plausible: bool = True
    confidence: int = 0
    narrative: str = ""
    red_flags: list = field(default_factory=list)
    recommendation: str = ""
    llm_provider: str = ""
    analysis_time_sec: float = 0.0
    raw_response: str = ""
    error: Optional[str] = None


# ──────────────────────────────────────────────
# PROMPT TEMPLATES
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior US healthcare claims auditor with 15+ years of FWA detection experience.

YOUR PRIMARY EVIDENCE IS THE RULE ENGINE OUTPUT:
The claim you will receive has already been processed by an automated rule engine that checked
OIG compliance rules, NCCI bundling edits, ICD-CPT medical necessity, and statistical benchmarks.
The rule engine flags listed in the prompt are your starting point — treat them as established
findings, not suggestions. Your job is to confirm, explain, and score them using clinical reasoning.

PIPELINE PATTERN LABEL — HIGHEST PRIORITY:
If the prompt contains a line "★ CONFIRMED FRAUD PATTERN: <label>", that label IS the fraud_category
to use in your response. Do not override it with the rule flag name. The pipeline has already
resolved the specific pattern — your job is to explain it clinically in the narrative.

FRAUD CATEGORY SELECTION — use in this priority order:
  1. If "★ CONFIRMED FRAUD PATTERN" is present → use that exact category label
  2. Otherwise map the rule engine finding:
       ICD_CPT_MISMATCH fired      → "MEDICALLY_UNNECESSARY"
       NCCI_UNBUNDLING fired       → "UNBUNDLING"
       UPCODING_PROXY fired        → "UPCODING"
       SPECIALTY_MISMATCH fired    → "SPECIALTY_MISMATCH"
       COST_OUTLIER fired          → "STATISTICAL_OUTLIER"
       No flags, low risk          → "CLEAN"

RECOMMENDATION THRESHOLDS:
  risk_score 85–100  → "REFER_TO_SIU"   (clear fraud pattern, multiple rule violations)
  risk_score 70–84   → "DENY"           (strong rule violation, clinically indefensible)
  risk_score 40–69   → "REVIEW"         (flagged but needs human review before action)
  risk_score 0–39    → "APPROVE"        (minor or no flags, clinically plausible)

CLEAN CLAIM GUIDANCE:
  A routine venipuncture (CPT 36415) ordered during a diabetes follow-up is standard care.
  A chest X-ray (CPT 71046) for a pneumonia diagnosis is clinically expected.
  An inpatient hospital visit E&M (CPT 99232) is routine for admitted patients.
  Do NOT flag these as high-risk unless a confirmed fraud pattern is explicitly stated.

CRITICAL: Respond with ONLY a valid JSON object. No preamble, no markdown, no explanation outside JSON.
{
  "plausible": <true or false — is the claim clinically defensible?>,
  "confidence": <integer 0-100 — how certain are you of your finding?>,
  "risk_score": <integer 0-100 — overall FWA risk>,
  "fraud_category": <"CLEAN" | "UPCODING" | "UNBUNDLING" | "PHANTOM_BILLING" | "MEDICALLY_UNNECESSARY" | "DUPLICATE" | "SPECIALTY_MISMATCH" | "STATISTICAL_OUTLIER">,
  "narrative": <string: 2-3 sentences. Name the specific violation. Explain the clinical reason it is or is not fraud.>,
  "red_flags": <array of strings — each flag names a specific clinical or billing concern>,
  "recommendation": <"APPROVE" | "REVIEW" | "DENY" | "REFER_TO_SIU">
}"""


def build_claim_prompt(claim: ClaimRecord) -> str:
    """
    Build the user-turn prompt sent to Groq for each claim.

    DESIGN NOTE — why rule flags come first:
    LLMs read top-to-bottom and anchor on early information.
    Putting the rule engine findings at the top means Groq starts
    from what the automated system already found and explains it
    clinically, rather than re-deriving the fraud category from scratch.

    DESIGN NOTE — confirmed fraud pattern section:
    When the pipeline has already identified the specific fraud type
    (UNBUNDLING, UPCODING, etc.), we surface it as a separate starred
    section above the rule flags. This prevents Groq from overriding
    a known UNBUNDLING label with MEDICALLY_UNNECESSARY just because
    the rule flag text says ICD_CPT_MISMATCH.
    """

    # Format the rule engine flags into a readable block.
    # These are the OIG/NCCI violations the automated pipeline detected.
    rule_flags_text = ""
    if claim.rule_flags:
        flags = claim.rule_flags if isinstance(claim.rule_flags, list) else []
        for f in flags:
            rule_flags_text += f"\n  ✦ [{f.get('severity','?')}] {f.get('rule','?')}: {f.get('detail','')}"

    # Known FWA patterns the pipeline has already confirmed.
    # For these, Groq must use the exact category — it maps directly to the
    # fraud_category field in the JSON response.
    known_fwa_types = {"UPCODING", "UNBUNDLING", "ICD_CPT_MISMATCH",
                       "MEDICALLY_UNNECESSARY", "SPECIALTY_MISMATCH",
                       "DUPLICATE", "PHANTOM_BILLING"}

    # Map pipeline labels to the fraud_category values Groq should return.
    # ICD_CPT_MISMATCH in the pipeline becomes MEDICALLY_UNNECESSARY in output
    # because that is the correct clinical category name for this violation.
    category_map = {
        "ICD_CPT_MISMATCH":      "MEDICALLY_UNNECESSARY",
        "UNBUNDLING":            "UNBUNDLING",
        "UPCODING":              "UPCODING",
        "MEDICALLY_UNNECESSARY": "MEDICALLY_UNNECESSARY",
        "SPECIALTY_MISMATCH":    "SPECIALTY_MISMATCH",
        "DUPLICATE":             "DUPLICATE",
        "PHANTOM_BILLING":       "PHANTOM_BILLING",
    }

    # Build the confirmed pattern block — only shown when a known fraud type
    # is present. The ★ star and ALL-CAPS make it visually dominant.
    confirmed_pattern_block = ""
    if claim.fraud_label in known_fwa_types:
        mapped_category = category_map.get(claim.fraud_label, claim.fraud_label)
        confirmed_pattern_block = (
            f"\n★ CONFIRMED FRAUD PATTERN: {mapped_category}"
            f"\n  Use \"{mapped_category}\" as the fraud_category in your JSON response."
            f"\n  Explain this specific pattern in the narrative field.\n"
        )

    return f"""Analyze this healthcare claim. The rule engine findings below are your PRIMARY evidence.
{confirmed_pattern_block}
═══ RULE ENGINE FINDINGS ({claim.rule_flag_count} violation(s), severity: {claim.rule_flag_severity}) ═══{rule_flags_text if rule_flags_text else chr(10) + "  No rule violations detected"}
  Composite risk score : {claim.composite_risk_score:.0f}/100

═══ CLAIM ═══
Claim ID         : {claim.claim_id}
Date of Service  : {claim.date_of_service}
Place of Service : {claim.pos_description}

═══ PATIENT ═══
Age              : {claim.patient_age}
Gender           : {claim.patient_gender}

═══ PROVIDER ═══
Name             : {claim.provider_name}
Specialty        : {claim.provider_specialty}
State            : {claim.provider_state}
NPI              : {claim.provider_npi}

═══ BILLING ═══
Procedure (CPT)  : {claim.cpt_code} — {claim.cpt_description}
Diagnosis (ICD)  : {claim.icd_primary} — {claim.icd_description}
Billed Amount    : ${claim.billed_amount:,.2f}
Units Billed     : {claim.units}

═══ PROVIDER BENCHMARKS ═══
% of E&M visits billed as 99215 (highest complexity): {claim.pct_99215:.0f}%  (national avg: ~18%)
Weekend billing rate                                 : {claim.weekend_billing_rate:.1f}%  (national avg: ~15%)
Avg cost vs specialty peers                          : {claim.provider_vs_specialty_ratio:.2f}x

Now provide your FWA assessment as JSON. Use the rule engine findings above as your primary evidence."""


# ──────────────────────────────────────────────
# LLM CLIENTS
# ──────────────────────────────────────────────

class OllamaClient:
    """Local LLM via Ollama — free, runs on your laptop."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url

    def call(self, system: str, user: str) -> str:
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512}
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}. Is Ollama running? Try: ollama serve")


class ClaudeClient:
    """Claude API via Anthropic — best reasoning quality."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Export it: export ANTHROPIC_API_KEY=sk-ant-...")
        self.api_key = api_key

    def call(self, system: str, user: str) -> str:
        try:
            import requests
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 1024,
                    "system": system,
                    "messages": [{"role": "user", "content": user}]
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"].strip()
        except Exception as e:
            raise RuntimeError(f"Claude API error: {e}")


class GroqClient:
    """
    Groq API — FREE, ultra-fast inference (~0.5s/claim).
    Best choice for development and testing.
    Get free API key at: console.groq.com (no credit card needed)
    Install: pip install groq
    """

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "  1. Sign up free at console.groq.com\n"
                "  2. Generate an API key\n"
                "  Windows:     set GROQ_API_KEY=gsk_your_key\n"
                "  macOS/Linux: export GROQ_API_KEY=gsk_your_key"
            )
        self.api_key = api_key
        self.model = model

    def call(self, system: str, user: str) -> str:
        # temperature=0.1 keeps responses deterministic and consistent.
        # Higher values make the LLM more creative but less reliable for JSON output.
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},  # auditor persona + JSON rules
                    {"role": "user",   "content": user}     # the individual claim details
                ],
                temperature=0.1,   # low = consistent, high = creative but unreliable
                max_tokens=1024    # enough for a full JSON response + narrative
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            raise RuntimeError(
                "groq package not installed. Run: pip install groq"
            )
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}")



class MockClient:
    """Mock LLM for testing without any API/model — returns deterministic results."""

    def call(self, system: str, user: str) -> str:
        # Deterministic mock based on keywords in prompt
        is_mismatch = "M25.361" in user and "27447" in user
        is_upcoding = "99215" in user and float(
            next((w for w in user.split() if w.replace('.','').isdigit() and len(w) > 3), "18")
        ) > 60 if "99215" in user else False

        risk = 88 if is_mismatch else (82 if is_upcoding else 45)
        category = "MEDICALLY_UNNECESSARY" if is_mismatch else ("UPCODING" if is_upcoding else "CLEAN")
        plausible = not (is_mismatch or is_upcoding)

        return json.dumps({
            "plausible": plausible,
            "confidence": 87,
            "risk_score": risk,
            "fraud_category": category,
            "narrative": f"Mock analysis: Claim shows {'significant FWA indicators' if risk > 70 else 'no major concerns'}. "
                         f"{'Diagnosis does not support procedure billed.' if is_mismatch else ''}"
                         f"{'Upcoding pattern detected in provider billing history.' if is_upcoding else ''}",
            "red_flags": [
                "ICD-CPT mismatch detected by rule engine" if is_mismatch else "Provider billing within normal range",
                f"Risk score: {risk}/100"
            ],
            "recommendation": "REFER_TO_SIU" if risk >= 80 else ("REVIEW" if risk >= 60 else "APPROVE")
        })


def get_llm_client(provider: str):
    """Factory function to get the appropriate LLM client."""
    if provider == "groq":
        print(f"  Using: Groq API ({GROQ_MODEL}) — ultra-fast free inference")
        print(f"  Tip: Get free key at console.groq.com if not set")
        return GroqClient(GROQ_API_KEY, GROQ_MODEL)
    elif provider == "ollama":
        print(f"  Using: Ollama ({OLLAMA_MODEL}) at {OLLAMA_BASE_URL}")
        print(f"  Tip: Run 'ollama pull {OLLAMA_MODEL}' if not already downloaded")
        return OllamaClient(OLLAMA_MODEL, OLLAMA_BASE_URL)
    elif provider == "claude":
        print("  Using: Claude claude-sonnet-4-5 (Anthropic API — best quality)")
        return ClaudeClient(ANTHROPIC_API_KEY)
    elif provider == "mock":
        print("  Using: Mock LLM (for testing — no real AI inference)")
        return MockClient()
    else:
        print("  Unknown provider. Falling back to mock.")
        return MockClient()


# ──────────────────────────────────────────────
# VECTOR STORE (Optional — ChromaDB)
# ──────────────────────────────────────────────

class FWAVectorStore:
    """
    Store past FWA findings as embeddings for similarity search.
    Helps find 'similar past fraud cases' for each new claim.
    Optional — requires: pip install chromadb sentence-transformers
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self._init()

    def _init(self):
        try:
            import chromadb
            from chromadb.config import Settings
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
            self.collection = self.client.create_collection("fwa_cases")
            self._seed_known_cases()
            print("    ✓ ChromaDB vector store initialized with known FWA cases")
        except ImportError:
            print("    ⚠ ChromaDB not installed. Skipping vector search. (pip install chromadb)")

    def _seed_known_cases(self):
        """Seed with known FWA patterns as reference cases."""
        known_cases = [
            {
                "id": "ref_001",
                "text": "Provider billed CPT 99215 for 94% of patient encounters. ICD codes were simple URIs and routine follow-ups not warranting highest complexity E&M. Systematic upcoding confirmed.",
                "category": "UPCODING"
            },
            {
                "id": "ref_002",
                "text": "CPT 27447 total knee arthroplasty billed with ICD M25.361 knee stiffness. No prior conservative treatment history. Diagnosis does not meet CMS medical necessity for surgical intervention.",
                "category": "MEDICALLY_UNNECESSARY"
            },
            {
                "id": "ref_003",
                "text": "CPT 36415 venipuncture and CPT 36416 capillary blood draw billed on same date of service for same patient. NCCI edit violation. 36416 is bundled into 36415.",
                "category": "UNBUNDLING"
            },
            {
                "id": "ref_004",
                "text": "Identical claims for CPT 93000 EKG submitted twice on same date with billed amounts differing by $3. Near-duplicate claim pattern to evade exact-match filters.",
                "category": "DUPLICATE"
            },
            {
                "id": "ref_005",
                "text": "Physical therapist billing CPT 93306 echocardiography. Procedure is outside scope of practice for physical therapy specialty.",
                "category": "SPECIALTY_MISMATCH"
            }
        ]
        if self.collection:
            self.collection.add(
                documents=[c["text"] for c in known_cases],
                metadatas=[{"category": c["category"]} for c in known_cases],
                ids=[c["id"] for c in known_cases]
            )

    def find_similar(self, claim_text: str, n: int = 2) -> list:
        """Find similar past FWA cases for a given claim description."""
        if not self.collection:
            return []
        try:
            results = self.collection.query(query_texts=[claim_text], n_results=n)
            return [
                {"case": doc, "category": meta["category"]}
                for doc, meta in zip(
                    results["documents"][0],
                    results["metadatas"][0]
                )
            ]
        except Exception:
            return []


# ──────────────────────────────────────────────
# CORE ANALYZER
# ──────────────────────────────────────────────

class FWAReasoningEngine:
    """
    Main Gen AI reasoning engine.
    Processes claims queue through LLM, parses responses, and compiles report.
    """

    def __init__(self, llm_provider: str = "ollama"):
        print("\n[AI LAYER] Initializing FWA Reasoning Engine...")
        self.llm = get_llm_client(llm_provider)
        self.vector_store = FWAVectorStore() if ENABLE_VECTOR_SEARCH else None
        self.results: list[FWAAnalysisResult] = []

    def _parse_llm_response(self, raw: str, claim_id: str) -> dict:
        """
        Safely parse the JSON that Groq returns.

        Groq is instructed to return pure JSON, but occasionally wraps it in
        markdown code fences (```json ... ```). This method strips those first,
        then parses. If parsing still fails, a regex fallback tries to extract
        any JSON object from the response. If that also fails, we return safe
        defaults so the batch doesn't crash on one bad response.
        """
        clean = raw.strip()

        # Strip markdown fences — Groq sometimes adds these despite instructions
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: extract the first {...} block found anywhere in the response
            import re
            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass

            # If all parsing fails, return safe defaults — batch continues
            print(f"      ⚠ JSON parse failed for {claim_id}. Using defaults.")
            return {
                "plausible":      True,
                "confidence":     0,
                "risk_score":     50,
                "fraud_category": "REVIEW_NEEDED",
                "narrative":      f"LLM response could not be parsed. Raw: {raw[:200]}",
                "red_flags":      ["Parse error — manual review recommended"],
                "recommendation": "REVIEW"
            }

    def analyze_claim(self, claim: ClaimRecord) -> FWAAnalysisResult:
        """
        Run one claim through the full LLM reasoning pipeline.

        Flow:
          1. Optionally enrich prompt with similar past cases from ChromaDB (Month 2)
          2. Build the structured claim prompt (rule flags first, then clinical details)
          3. Send to Groq via GroqClient.call()
          4. Parse the JSON response
          5. Return a FWAAnalysisResult dataclass
        """
        start = time.time()

        # Step 1: Vector store similarity search (optional — requires ChromaDB)
        # Finds past confirmed FWA cases that resemble this claim.
        # Disabled by default (ENABLE_VECTOR_SEARCH = False) — Month 2 feature.
        similar_context = ""
        if self.vector_store:
            claim_text = f"{claim.cpt_description} {claim.icd_description} {claim.provider_specialty}"
            similar = self.vector_store.find_similar(claim_text)
            if similar:
                similar_context = "\n═══ SIMILAR PAST CASES ═══\n"
                for s in similar:
                    similar_context += f"[{s['category']}] {s['case']}\n"

        # Step 2: Build prompt — rule flags appear first so Groq anchors to them
        user_prompt = build_claim_prompt(claim)
        if similar_context:
            user_prompt += similar_context  # append past cases at the end if available

        # Step 3 & 4: Send to LLM and parse response
        try:
            raw_response = self.llm.call(SYSTEM_PROMPT, user_prompt)
            parsed       = self._parse_llm_response(raw_response, claim.claim_id)

            result = FWAAnalysisResult(
                claim_id           = claim.claim_id,
                risk_score         = parsed.get("risk_score",     50),
                fraud_category     = parsed.get("fraud_category", "UNKNOWN"),
                plausible          = parsed.get("plausible",      True),
                confidence         = parsed.get("confidence",     0),
                narrative          = parsed.get("narrative",      ""),
                red_flags          = parsed.get("red_flags",      []),
                recommendation     = parsed.get("recommendation", "REVIEW"),
                llm_provider       = LLM_PROVIDER,
                analysis_time_sec  = round(time.time() - start, 2),
                raw_response       = raw_response
            )
        except Exception as e:
            # Don't crash the whole batch — log the error and continue
            result = FWAAnalysisResult(
                claim_id          = claim.claim_id,
                error             = str(e),
                narrative         = f"Analysis failed: {str(e)}",
                recommendation    = "REVIEW",
                analysis_time_sec = round(time.time() - start, 2)
            )

        return result

    def run_batch(self, claims: list[ClaimRecord]) -> list[FWAAnalysisResult]:
        """
        Process a full batch of claims through the reasoning engine.

        Each claim is analyzed independently — Groq has no memory between calls.
        All claim context (rule flags, clinical data, benchmarks) must be in the
        prompt itself, which is why build_claim_prompt() is thorough.
        """
        print(f"\n  Processing {len(claims)} claims through {LLM_PROVIDER.upper()} LLM...")
        print("  " + "─" * 50)

        results = []
        for i, claim in enumerate(claims, 1):
            print(f"  [{i:02d}/{len(claims)}] Analyzing {claim.claim_id} "
                  f"(CPT {claim.cpt_code}, risk {claim.composite_risk_score:.0f}/100)...", end=" ")

            result = self.analyze_claim(claim)
            results.append(result)

            if result.error:
                print(f"ERROR: {result.error}")
            else:
                # 🔴 = high risk (≥80), 🟡 = medium (≥60), 🟢 = low (<60)
                icon = "🔴" if result.risk_score >= 80 else "🟡" if result.risk_score >= 60 else "🟢"
                print(f"{icon} {result.fraud_category} | Risk: {result.risk_score}/100 | "
                      f"→ {result.recommendation} ({result.analysis_time_sec}s)")

            # Rate limiting — Claude API has stricter limits than Groq
            if LLM_PROVIDER == "claude" and i < len(claims):
                time.sleep(0.5)

        self.results = results
        return results


# ──────────────────────────────────────────────
# REPORT BUILDER
# ──────────────────────────────────────────────

def build_report(claims_df: pd.DataFrame, results: list[FWAAnalysisResult]) -> pd.DataFrame:
    """Merge AI analysis results back with claim data for final report."""
    print("\n  Building final FWA report...")

    results_rows = []
    for r in results:
        results_rows.append({
            "claim_id": r.claim_id,
            "ai_risk_score": r.risk_score,
            "ai_fraud_category": r.fraud_category,
            "ai_plausible": r.plausible,
            "ai_confidence": r.confidence,
            "ai_narrative": r.narrative,
            "ai_red_flags": " | ".join(r.red_flags) if r.red_flags else "",
            "ai_recommendation": r.recommendation,
            "ai_provider": r.llm_provider,
            "ai_analysis_time_sec": r.analysis_time_sec,
            "ai_error": r.error or ""
        })

    results_df = pd.DataFrame(results_rows)
    report = claims_df.merge(results_df, on="claim_id", how="left")
    report = report.sort_values("ai_risk_score", ascending=False)

    return report


def print_ai_summary(results: list[FWAAnalysisResult]):
    """Print a formatted summary of AI analysis results."""
    valid = [r for r in results if not r.error]
    if not valid:
        print("  ⚠ No valid results to summarize.")
        return

    print("\n" + "═" * 65)
    print("  MEDIAGUARD AI — GEN AI ANALYSIS SUMMARY")
    print("═" * 65)
    print(f"  Claims analyzed         : {len(valid)}")
    print(f"  Avg analysis time       : {sum(r.analysis_time_sec for r in valid)/len(valid):.1f}s/claim")
    print(f"  Avg AI risk score       : {sum(r.risk_score for r in valid)/len(valid):.0f}/100")

    # Recommendation breakdown
    from collections import Counter
    recs = Counter(r.recommendation for r in valid)
    print("\n  RECOMMENDATIONS:")
    for rec, count in recs.most_common():
        bar = "█" * count
        print(f"    {rec:<20} : {bar} ({count})")

    # Category breakdown
    cats = Counter(r.fraud_category for r in valid)
    print("\n  FWA CATEGORIES DETECTED:")
    for cat, count in cats.most_common():
        print(f"    {cat:<30} : {count}")

    # Highest risk claims
    top_risk = sorted(valid, key=lambda r: r.risk_score, reverse=True)[:5]
    print("\n  TOP 5 HIGHEST RISK CLAIMS:")
    for r in top_risk:
        print(f"    {r.claim_id:<22} | Risk: {r.risk_score:3d}/100 | "
              f"{r.fraud_category:<25} | → {r.recommendation}")

    # Sample narrative
    siu_cases = [r for r in valid if r.recommendation == "REFER_TO_SIU"]
    if siu_cases:
        sample = siu_cases[0]
        print(f"\n  SAMPLE SIU REFERRAL — {sample.claim_id}:")
        print(f"    Category    : {sample.fraud_category}")
        print(f"    Risk Score  : {sample.risk_score}/100 (confidence: {sample.confidence}%)")
        print(f"    Narrative   : {sample.narrative}")
        if sample.red_flags:
            print("    Red Flags   :")
            for flag in sample.red_flags[:3]:
                print(f"      • {flag}")

    print("═" * 65)


# ──────────────────────────────────────────────
# DATA LOADER
# ──────────────────────────────────────────────

def load_claims_for_ai(filepath: str, max_claims: int, min_risk: float) -> tuple[pd.DataFrame, list[ClaimRecord]]:
    """Load and prepare claims from pipeline output for AI analysis."""
    print(f"\n[LOAD] Reading {filepath}...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"'{filepath}' not found.\n"
            "Run fwa_data_pipeline.py first to generate the input data."
        )

    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} claims from pipeline output")

    # Filter to high-risk claims
    #df_filtered = df[df["composite_risk_score"] >= min_risk].head(max_claims)
    # Accept all claims the pipeline queued — they were already filtered by
    # score ≥ 70 OR HIGH severity rule flag. No second filter needed here.
    df_filtered = df[df["composite_risk_score"] >= min_risk].head(max_claims) \
              if min_risk > 0 else df.head(max_claims)
    print(f"  Filtered to {len(df_filtered)} claims (risk ≥ {min_risk}, max {max_claims})")

    claims = []
    for _, row in df_filtered.iterrows():
        # Parse rule flags
        try:
            rule_flags = json.loads(row.get("rule_flags", "[]"))
        except Exception:
            rule_flags = []

        claims.append(ClaimRecord(
            claim_id=str(row.get("claim_id", "")),
            patient_age=int(row.get("patient_age", 0)),
            patient_gender=str(row.get("patient_gender", "")),
            provider_npi=str(row.get("provider_npi", "")),
            provider_name=str(row.get("provider_name", "")),
            provider_specialty=str(row.get("provider_specialty", "")),
            provider_state=str(row.get("provider_state", "")),
            date_of_service=str(row.get("date_of_service", "")),
            cpt_code=str(row.get("cpt_code", "")),
            cpt_description=str(row.get("cpt_description", "")),
            icd_primary=str(row.get("icd_primary", "")),
            icd_description=str(row.get("icd_description", "")),
            billed_amount=float(row.get("billed_amount", 0)),
            units=int(row.get("units", 1)),
            pos_description=str(row.get("pos_description", "")),
            composite_risk_score=float(row.get("composite_risk_score", 0)),
            rule_flag_count=int(row.get("rule_flag_count", 0)),
            rule_flag_severity=str(row.get("rule_flag_severity", "CLEAN")),
            rule_flags=rule_flags,
            pct_99215=float(row.get("pct_99215", 0)),
            weekend_billing_rate=float(row.get("weekend_billing_rate", 0)),
            provider_vs_specialty_ratio=float(row.get("provider_vs_specialty_ratio", 1)),
            fraud_label=str(row.get("fraud_label", "UNKNOWN"))
        ))

    return df_filtered, claims


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║  MEDIAGUARD AI — LangChain Reasoning     ║")
    print("║  Gen AI FWA Explanation Engine           ║")
    print("╚══════════════════════════════════════════╝")
    print(f"\n  LLM Provider  : {LLM_PROVIDER.upper()}")
    print(f"  Max claims    : {MAX_CLAIMS_TO_ANALYZE}")
    print(f"  Min risk score: {MIN_RISK_SCORE}/100")

    # Auto-fallback logic
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            print(f"\n  ⚠ GROQ_API_KEY not set.")
            print("  → Get a free key at console.groq.com (takes 2 min, no credit card)")
            print("  → Windows:     set GROQ_API_KEY=gsk_your_key")
            print("  → macOS/Linux: export GROQ_API_KEY=gsk_your_key")
            print("  → Falling back to MOCK mode for now\n")
            LLM_PROVIDER = "mock"

    elif LLM_PROVIDER == "ollama":
        try:
            import requests
            requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        except Exception:
            print(f"\n  ⚠ Ollama not running at {OLLAMA_BASE_URL}")
            print("  → Falling back to MOCK mode for demo")
            print("  → To use real AI: Install Ollama (ollama.ai) then run: ollama pull mistral")
            LLM_PROVIDER = "mock"

    # Load claims
    claims_df, claims = load_claims_for_ai(INPUT_FILE, MAX_CLAIMS_TO_ANALYZE, MIN_RISK_SCORE)

    if not claims:
        print("\n  No claims found matching criteria. Exiting.")
        exit(0)

    # Run AI reasoning
    engine = FWAReasoningEngine(llm_provider=LLM_PROVIDER)
    results = engine.run_batch(claims)

    # Build and save report
    report = build_report(claims_df, results)
    report.to_csv(OUTPUT_FILE, index=False)

    # Print summary
    print_ai_summary(results)

    print(f"\n✅ AI analysis complete.")
    print(f"   Report saved → {OUTPUT_FILE}")
    print(f"   {len([r for r in results if r.recommendation == 'REFER_TO_SIU'])} claims referred to SIU")
    print(f"   {len([r for r in results if r.recommendation == 'DENY'])} claims recommended for denial")
    print(f"\n   Next step: Run streamlit_dashboard.py for interactive UI")
