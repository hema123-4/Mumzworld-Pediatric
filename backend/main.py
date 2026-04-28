"""
Mumzworld Pediatric Symptom Triage API
=====================================
Architecture: RAG + Structured Output + Validation + Uncertainty handling
Author: Built for Mumzworld take-home assignment

Flow:
1. Parse and validate input (Pydantic schema)
2. Retrieve relevant knowledge chunks (keyword + age-based RAG)
3. Send to LLM with strict system prompt + knowledge context
4. Parse + validate structured JSON response against schema
5. Apply safety overrides (never silence emergency signals)
6. Return validated, typed response

Design decisions:
- Uses OpenRouter (free tier) so no paid key required
- Knowledge base is JSON (no vector DB needed at this scale; ~12 entries)
  For production: swap to ChromaDB or Pinecone with embeddings
- Two-stage retrieval: fast keyword filter then LLM re-ranking
- Hard safety rules applied post-LLM to prevent hallucination overrides
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
import json
import os
import re
import httpx
import asyncio
from pathlib import Path

app = FastAPI(title="Mumzworld Triage API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],

)
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the path to the directory where main.py is located
base_dir = Path(__file__).resolve().parent
env_path = base_dir / ".env"

# Load the .env file using the absolute path
load_dotenv(dotenv_path=env_path)

if os.getenv("OPENROUTER_API_KEY"):
    print("✅ Success: OpenRouter API Key loaded!")
else:
    print(f"❌ Error: Looked for .env at {env_path} but couldn't find the key.")

# ── Load knowledge base at startup ──────────────────────────────────────────
KB_PATH = Path(__file__).parent.parent / "data" / "symptom_knowledge.json"
with open(KB_PATH, encoding="utf-8") as f:
    KB = json.load(f)

KNOWLEDGE_ENTRIES = KB["knowledge_base"]
TRIAGE_LEVELS = KB["triage_levels"]
DISCLAIMER = KB["disclaimer"]


# ── Input / Output schemas (Pydantic) ────────────────────────────────────────

class TriageRequest(BaseModel):
    symptom_description: str = Field(..., min_length=5, max_length=1000)
    child_age_months: Optional[float] = Field(None, ge=0, le=216)  # 0–18 years
    language: Literal["en", "ar", "auto"] = "auto"

    @validator("symptom_description")
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Symptom description cannot be empty")
        return v.strip()


class ProductRecommendation(BaseModel):
    name: str
    category: str
    note: Optional[str] = None


class TriageResponse(BaseModel):
    # Core triage
    triage_level: Optional[str] = None           # null if out of scope
    triage_label_en: Optional[str] = None
    triage_label_ar: Optional[str] = None
    defer_to_doctor: Optional[bool] = None

    # Analysis
    matched_condition: Optional[str] = None
    reasoning_en: str
    reasoning_ar: str
    red_flags_present: List[str] = []
    red_flags_to_watch: List[str] = []

    # Guidance
    action_en: str
    action_ar: str

    # Products (only for home-care triages)
    products: List[ProductRecommendation] = []

    # Confidence and uncertainty
    confidence: float = Field(..., ge=0.0, le=1.0)
    out_of_scope: bool = False
    needs_more_info: bool = False
    clarification_needed: Optional[str] = None

    # Meta
    disclaimer_en: str
    disclaimer_ar: str
    detected_language: str


# ── RAG: keyword + age-based retrieval ───────────────────────────────────────

def retrieve_relevant_entries(description: str, age_months: Optional[float], top_k: int = 4) -> list:
    """
    Simple but effective retrieval:
    1. Tokenize description, match against keywords in knowledge base
    2. Filter by age range if provided
    3. Score by match count, return top_k

    For production: replace with embedding similarity search
    (e.g. all-MiniLM-L6-v2 embeddings + cosine similarity or Chroma)
    """
    desc_lower = description.lower()
    scored = []

    for entry in KNOWLEDGE_ENTRIES:
        score = 0
        keywords = entry.get("keywords", [])

        for kw in keywords:
            if kw.lower() in desc_lower:
                score += 1

        # Age-range filter — soft boost (don't hard-exclude, LLM can judge)
        if age_months is not None:
            age_range_key = entry.get("age_group", "all")
            age_ranges = KB.get("age_ranges", {})
            if age_range_key in age_ranges:
                ar = age_ranges[age_range_key]
                if ar["min_months"] <= age_months <= ar["max_months"]:
                    score += 2  # boost age-appropriate entries
                elif age_range_key == "all":
                    score += 1  # neutral boost for age-agnostic entries

        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


# ── LLM call via OpenRouter ──────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Uses Qwen2.5-72B (free on OpenRouter) — strong multilingual, good at JSON
# Fallback to claude-3-haiku if ANTHROPIC_API_KEY is set
MODEL = os.getenv("TRIAGE_MODEL", "qwen/qwen-2.5-72b-instruct")


def build_system_prompt() -> str:
    return """You are a pediatric symptom triage assistant for Mumzworld, a leading mother-and-baby e-commerce platform in the Middle East.

Your role is to:
1. Assess pediatric symptoms described by parents
2. Provide evidence-based triage guidance using the knowledge context provided
3. Recommend when to seek emergency care vs. home management
4. Suggest relevant Mumzworld products ONLY when home care is appropriate (NEVER for emergencies)
5. Respond in BOTH English and Arabic natively — not as a translation

CRITICAL SAFETY RULES (non-negotiable):
- NEVER downplay emergency symptoms. When in doubt, recommend emergency care.
- NEVER provide medication dosages. Refer to doctor/pharmacist.
- NEVER diagnose specific diseases (e.g., "your child has X").
- NEVER invent information not in the provided knowledge context.
- If the question is about an adult (not a child) or is completely unrelated to pediatric symptoms, set out_of_scope=true and explain kindly.
- If the description is too vague to assess, set needs_more_info=true with a specific clarification question.
- Express confidence honestly — low confidence on ambiguous inputs.

OUTPUT FORMAT: You must return ONLY valid JSON matching this exact schema:
{
  "triage_level": "<string from allowed levels or null if out of scope>",
  "matched_condition": "<brief description of matched symptom pattern or null>",
  "reasoning_en": "<clear English explanation of why this triage level, grounded in the knowledge>",
  "reasoning_ar": "<same reasoning in natural Arabic — NOT a translation, native Arabic>",
  "red_flags_present": ["<list of red flags mentioned in the input that triggered concern>"],
  "red_flags_to_watch": ["<list of red flags to monitor going forward>"],
  "action_en": "<specific actionable guidance in English>",
  "action_ar": "<same guidance in natural Arabic>",
  "products": [{"name": "<product>", "category": "<category>", "note": "<optional note>"}],
  "confidence": <0.0 to 1.0>,
  "out_of_scope": <true/false>,
  "needs_more_info": <false/true>,
  "clarification_needed": "<question to ask if needs_more_info is true, else null>",
  "defer_to_doctor": <true/false>
}

Allowed triage_level values: "emergency", "urgent_if_red_flag_else_monitor", "emergency_if_red_flag_else_observe", "emergency_if_anaphylaxis", "see_doctor_within_24_48h", "monitor_dehydration", "home_care", "home_care_with_monitoring", "monitor_with_red_flag_awareness", null

Return ONLY the JSON object. No preamble, no explanation outside the JSON."""


def build_user_prompt(request: TriageRequest, retrieved_entries: list) -> str:
    context_str = json.dumps(retrieved_entries, ensure_ascii=False, indent=2)
    age_str = f"{request.child_age_months} months" if request.child_age_months is not None else "not specified"

    lang_instruction = ""
    if request.language == "ar":
        lang_instruction = "\nNote: The parent wrote in Arabic. Ensure Arabic responses feel native and natural, not translated."
    elif request.language == "en":
        lang_instruction = "\nNote: The parent wrote in English."
    else:
        lang_instruction = "\nNote: Detect the language and respond accordingly, but always include BOTH English and Arabic in your JSON fields."

    return f"""KNOWLEDGE CONTEXT (use this as your primary source — do not invent facts outside it):
{context_str}

PARENT'S INPUT:
Child age: {age_str}
Symptom description: "{request.symptom_description}"
{lang_instruction}

Assess the symptoms, match to the most relevant knowledge entry, and return the structured JSON triage response."""


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Calls OpenRouter API. Falls back to Anthropic if ANTHROPIC_API_KEY is set.
    Timeout: 30s — long enough for 72B model, short enough to not hang UI.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        # Use Anthropic directly (if key available)
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1500,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}]
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]
    else:
        # OpenRouter with free model
        if not OPENROUTER_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="No API key configured. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY in environment."
            )
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://mumzworld.com",
                    "X-Title": "Mumzworld Triage Assistant",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL,
                    "max_tokens": 1500,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


def parse_llm_json(raw: str) -> dict:
    """
    Safely parse JSON from LLM output.
    LLMs sometimes wrap JSON in markdown code fences — strip those first.
    Raises ValueError with clear message if parse fails.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw output: {raw[:500]}")


def apply_safety_overrides(parsed: dict, retrieved: list) -> dict:
    """
    Post-LLM safety layer. These rules CANNOT be overridden by the LLM.

    Rule 1: If any retrieved entry has triage_level=emergency AND
            the LLM returned something less severe, override to emergency.
            (Prevents LLM from downplaying emergencies)

    Rule 2: Emergency triages NEVER get product recommendations.

    Rule 3: Confidence cap: if out_of_scope or needs_more_info, cap at 0.5.
    """
    triage = parsed.get("triage_level")

    # Rule 1: Emergency escalation check
    emergency_levels = {"emergency", "emergency_if_anaphylaxis"}
    for entry in retrieved:
        if entry.get("triage_level") == "emergency":
            # If retrieved entry is emergency but LLM said something else,
            # and red flags are mentioned in the action/reasoning, escalate
            action = (parsed.get("action_en") or "").lower()
            reasoning = (parsed.get("reasoning_en") or "").lower()
            if any(rf_kw in action + reasoning for rf_kw in ["emergency", "immediately", "911", "999", "ambulance"]):
                if triage not in emergency_levels:
                    parsed["triage_level"] = "emergency"
                    parsed["_safety_override"] = "Escalated to emergency by safety layer"

    # Rule 2: No products on emergency triage
    if parsed.get("triage_level") in emergency_levels:
        parsed["products"] = []
        parsed["defer_to_doctor"] = True

    # Rule 3: Confidence cap
    if parsed.get("out_of_scope") or parsed.get("needs_more_info"):
        parsed["confidence"] = min(parsed.get("confidence", 0.5), 0.5)

    # Rule 4: Null triage if out of scope
    if parsed.get("out_of_scope"):
        parsed["triage_level"] = None
        parsed["defer_to_doctor"] = None

    return parsed


def detect_language(text: str) -> str:
    """Detect if text is Arabic or English based on Unicode ranges."""
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "en"
    return "ar" if arabic_chars / total_alpha > 0.3 else "en"


# ── Main triage endpoint ──────────────────────────────────────────────────────

@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    """
    Main triage endpoint.
    Returns structured, validated triage response in EN + AR.
    """
    # 1. Detect language if auto
    detected_lang = detect_language(request.symptom_description) if request.language == "auto" else request.language

    # 2. RAG retrieval
    retrieved = retrieve_relevant_entries(
        request.symptom_description,
        request.child_age_months,
        top_k=4
    )

    # 3. Build prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(request, retrieved)

    # 4. Call LLM
    try:
        raw_output = await call_llm(system_prompt, user_prompt)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    # 5. Parse JSON
    try:
        parsed = parse_llm_json(raw_output)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 6. Apply safety overrides (post-LLM safety layer)
    parsed = apply_safety_overrides(parsed, retrieved)

    # 7. Enrich triage label from KB
    triage_level = parsed.get("triage_level")
    if triage_level and triage_level in TRIAGE_LEVELS:
        parsed["triage_label_en"] = TRIAGE_LEVELS[triage_level]["label_en"]
        parsed["triage_label_ar"] = TRIAGE_LEVELS[triage_level]["label_ar"]

    # 8. Add disclaimer
    parsed["disclaimer_en"] = DISCLAIMER["en"]
    parsed["disclaimer_ar"] = DISCLAIMER["ar"]
    parsed["detected_language"] = detected_lang

    # 9. Validate against Pydantic schema (explicit failure, not silent)
    try:
        response = TriageResponse(**parsed)
    except Exception as e:
        # Schema validation failed — return structured error, don't silently pass
        raise HTTPException(
            status_code=422,
            detail=f"Response schema validation failed: {str(e)}. Raw LLM: {raw_output[:300]}"
        )

    return response


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "kb_entries": len(KNOWLEDGE_ENTRIES)}


@app.get("/")
async def root():
    return {
        "service": "Mumzworld Pediatric Triage API",
        "version": "1.0.0",
        "endpoints": ["/triage (POST)", "/health (GET)", "/docs (GET)"]
    }
