"""
Mumzworld Pediatric Symptom Triage API
=====================================
Architecture: RAG + Structured Output + Validation + Uncertainty handling
Author: Built for Mumzworld take-home assignment

API Priority Order:
1. Groq       — Free, fast, reliable (RECOMMENDED)
2. Gemini     — Free tier but has quota limits
3. Anthropic  — Paid but most reliable
4. OpenRouter — Free models available
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
import json
import os
import re
import httpx
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ───────────────────────────────────────────────
base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── Print which provider will be used ────────────────────────────────────────
if GROQ_API_KEY:
    print("✅ GROQ API Key loaded! Using Groq (llama-3.3-70b-versatile) — free & fast")
elif GEMINI_API_KEY:
    print(f"✅ Gemini API Key loaded! Using {GEMINI_MODEL}")
elif ANTHROPIC_API_KEY:
    print("✅ Anthropic API Key loaded! Using Claude Haiku")
elif OPENROUTER_API_KEY:
    print("✅ OpenRouter API Key loaded!")
else:
    print("❌ No API key found!")
    print("   Add one of these to backend/.env:")
    print("   GROQ_API_KEY=gsk_...       (free - console.groq.com)")
    print("   GEMINI_API_KEY=AIza...     (free - aistudio.google.com)")
    print("   ANTHROPIC_API_KEY=sk-ant.. (paid - console.anthropic.com)")

# ── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="Mumzworld Triage API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load knowledge base ───────────────────────────────────────────────────────
KB_PATH = Path(__file__).parent.parent / "data" / "symptom_knowledge.json"
with open(KB_PATH, encoding="utf-8") as f:
    KB = json.load(f)

KNOWLEDGE_ENTRIES = KB["knowledge_base"]
TRIAGE_LEVELS     = KB["triage_levels"]
DISCLAIMER        = KB["disclaimer"]


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TriageRequest(BaseModel):
    symptom_description: str = Field(..., min_length=5, max_length=1000)
    child_age_months: Optional[float] = Field(None, ge=0, le=216)
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
    triage_level: Optional[str] = None
    triage_label_en: Optional[str] = None
    triage_label_ar: Optional[str] = None
    defer_to_doctor: Optional[bool] = None
    matched_condition: Optional[str] = None
    reasoning_en: str
    reasoning_ar: str
    red_flags_present: List[str] = []
    red_flags_to_watch: List[str] = []
    action_en: str
    action_ar: str
    products: List[ProductRecommendation] = []
    confidence: float = Field(..., ge=0.0, le=1.0)
    out_of_scope: bool = False
    needs_more_info: bool = False
    clarification_needed: Optional[str] = None
    disclaimer_en: str
    disclaimer_ar: str
    detected_language: str


# ── RAG retrieval ─────────────────────────────────────────────────────────────

def retrieve_relevant_entries(description: str, age_months: Optional[float], top_k: int = 4) -> list:
    desc_lower = description.lower()
    scored = []
    for entry in KNOWLEDGE_ENTRIES:
        score = 0
        for kw in entry.get("keywords", []):
            if kw.lower() in desc_lower:
                score += 1
        if age_months is not None:
            age_range_key = entry.get("age_group", "all")
            age_ranges = KB.get("age_ranges", {})
            if age_range_key in age_ranges:
                ar = age_ranges[age_range_key]
                if ar["min_months"] <= age_months <= ar["max_months"]:
                    score += 2
                elif age_range_key == "all":
                    score += 1
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


# ── Prompts ───────────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    return """You are a pediatric symptom triage assistant for Mumzworld, a leading mother-and-baby platform in the Middle East.

Your role:
1. Assess pediatric symptoms described by parents
2. Provide evidence-based triage guidance using ONLY the knowledge context provided
3. Recommend emergency care vs. home management
4. Suggest Mumzworld products ONLY for home care (NEVER for emergencies)
5. Respond in BOTH English and Arabic natively (not as a translation)

CRITICAL SAFETY RULES:
- NEVER downplay emergency symptoms. When in doubt, recommend emergency care.
- NEVER provide medication dosages. Always refer to doctor/pharmacist.
- NEVER diagnose specific diseases (never say "your child has X").
- NEVER invent information not in the knowledge context.
- If about an adult or unrelated topic: set out_of_scope=true
- If description is too vague: set needs_more_info=true with a specific question
- Be honest about confidence — low confidence for ambiguous inputs

You MUST return ONLY a valid JSON object. No extra text before or after. No markdown. No explanation outside the JSON.

JSON schema to follow exactly:
{
  "triage_level": "<one of the allowed values or null>",
  "matched_condition": "<matched symptom pattern or null>",
  "reasoning_en": "<English explanation grounded in knowledge>",
  "reasoning_ar": "<Arabic explanation — native Arabic, not translated>",
  "red_flags_present": ["<red flags found in the input>"],
  "red_flags_to_watch": ["<red flags to monitor>"],
  "action_en": "<specific English guidance>",
  "action_ar": "<specific Arabic guidance>",
  "products": [{"name": "<product name>", "category": "<category>", "note": "<optional>"}],
  "confidence": <number between 0.0 and 1.0>,
  "out_of_scope": <true or false>,
  "needs_more_info": <true or false>,
  "clarification_needed": "<question string or null>",
  "defer_to_doctor": <true or false>
}

Allowed triage_level values (use exactly as written):
"emergency"
"urgent_if_red_flag_else_monitor"
"emergency_if_red_flag_else_observe"
"emergency_if_anaphylaxis"
"see_doctor_within_24_48h"
"monitor_dehydration"
"home_care"
"home_care_with_monitoring"
"monitor_with_red_flag_awareness"
null (only when out_of_scope is true)"""


def build_user_prompt(request: TriageRequest, retrieved_entries: list) -> str:
    context_str = json.dumps(retrieved_entries, ensure_ascii=False, indent=2)
    age_str = f"{request.child_age_months} months" if request.child_age_months is not None else "not specified"

    if request.language == "ar":
        lang_note = "Parent wrote in Arabic. Make Arabic fields feel native, not translated."
    elif request.language == "en":
        lang_note = "Parent wrote in English. Still include Arabic in all Arabic fields."
    else:
        lang_note = "Always include both English and Arabic in all respective fields."

    return f"""KNOWLEDGE CONTEXT (your only source of truth — do not invent facts):
{context_str}

PARENT INPUT:
Child age: {age_str}
Symptoms: "{request.symptom_description}"
Language note: {lang_note}

Return ONLY the JSON object. No other text."""


# ── LLM API calls ─────────────────────────────────────────────────────────────

async def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Tries providers in this order:
    1. Groq       (free, fast, no quota issues)
    2. Gemini     (free tier, may have quota limits)
    3. Anthropic  (paid)
    4. OpenRouter (free models available)
    """

    # ── 1. GROQ ──────────────────────────────────────────────────────────────
    if GROQ_API_KEY:
        print("🤖 Calling Groq: llama-3.3-70b-versatile")
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "max_tokens": 1500,
                    "temperature": 0.3,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt}
                    ]
                }
            )
            if resp.status_code != 200:
                print(f"❌ Groq error {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
            print("✅ Groq responded OK")
            return resp.json()["choices"][0]["message"]["content"]

    # ── 2. GEMINI ────────────────────────────────────────────────────────────
    elif GEMINI_API_KEY:
        print(f"🤖 Calling Gemini: {GEMINI_MODEL}")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models"
            f"/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 1500,
                        "temperature": 0.3,
                        "responseMimeType": "application/json"
                    }
                }
            )
            if resp.status_code != 200:
                print(f"❌ Gemini error {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
            print("✅ Gemini responded OK")
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    # ── 3. ANTHROPIC ─────────────────────────────────────────────────────────
    elif ANTHROPIC_API_KEY:
        print("🤖 Calling Anthropic Claude Haiku")
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
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
            if resp.status_code != 200:
                print(f"❌ Anthropic error {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
            print("✅ Anthropic responded OK")
            return resp.json()["content"][0]["text"]

    # ── 4. OPENROUTER ────────────────────────────────────────────────────────
    elif OPENROUTER_API_KEY:
        model = os.getenv("TRIAGE_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        print(f"🤖 Calling OpenRouter: {model}")
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://mumzworld.com",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "max_tokens": 1500,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt}
                    ]
                }
            )
            if resp.status_code != 200:
                print(f"❌ OpenRouter error {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
            print("✅ OpenRouter responded OK")
            return resp.json()["choices"][0]["message"]["content"]

    # ── No key found ──────────────────────────────────────────────────────────
    else:
        raise HTTPException(
            status_code=500,
            detail=(
                "No API key configured. Add one of these to backend/.env:\n"
                "  GROQ_API_KEY=gsk_...    (free - console.groq.com)\n"
                "  GEMINI_API_KEY=AIza...  (free - aistudio.google.com)\n"
                "  ANTHROPIC_API_KEY=...   (paid - console.anthropic.com)"
            )
        )


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_llm_json(raw: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw output: {raw[:500]}")


# ── Post-LLM safety overrides ─────────────────────────────────────────────────

def apply_safety_overrides(parsed: dict, retrieved: list) -> dict:
    """
    Deterministic safety rules applied AFTER the LLM responds.
    These cannot be overridden by the model.
    """
    triage = parsed.get("triage_level")
    emergency_levels = {"emergency", "emergency_if_anaphylaxis"}

    # Rule 1: Escalate to emergency if evidence warrants it
    for entry in retrieved:
        if entry.get("triage_level") == "emergency":
            action    = (parsed.get("action_en") or "").lower()
            reasoning = (parsed.get("reasoning_en") or "").lower()
            emergency_words = ["emergency", "immediately", "911", "999", "ambulance", "er ", "e.r"]
            if any(w in action + reasoning for w in emergency_words):
                if triage not in emergency_levels:
                    parsed["triage_level"] = "emergency"
                    parsed["_safety_override"] = "Escalated by safety layer"

    # Rule 2: No products on emergency triage
    if parsed.get("triage_level") in emergency_levels:
        parsed["products"] = []
        parsed["defer_to_doctor"] = True

    # Rule 3: Cap confidence when uncertain
    if parsed.get("out_of_scope") or parsed.get("needs_more_info"):
        parsed["confidence"] = min(parsed.get("confidence", 0.5), 0.5)

    # Rule 4: Null triage level when out of scope
    if parsed.get("out_of_scope"):
        parsed["triage_level"] = None
        parsed["defer_to_doctor"] = None

    return parsed


# ── Language detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    total_alpha  = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "en"
    return "ar" if arabic_chars / total_alpha > 0.3 else "en"


# ── Main triage endpoint ──────────────────────────────────────────────────────

@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):

    # 1. Detect language
    detected_lang = (
        detect_language(request.symptom_description)
        if request.language == "auto"
        else request.language
    )

    # 2. RAG retrieval
    retrieved = retrieve_relevant_entries(
        request.symptom_description,
        request.child_age_months,
        top_k=4
    )

    # 3. Build prompts
    system_prompt = build_system_prompt()
    user_prompt   = build_user_prompt(request, retrieved)

    # 4. Call LLM
    try:
        raw_output = await call_llm(system_prompt, user_prompt)
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM API error: {e.response.status_code} — {e.response.text[:300]}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    # 5. Parse JSON
    try:
        parsed = parse_llm_json(raw_output)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 6. Safety overrides
    parsed = apply_safety_overrides(parsed, retrieved)

    # 7. Enrich triage labels from knowledge base
    triage_level = parsed.get("triage_level")
    if triage_level and triage_level in TRIAGE_LEVELS:
        parsed["triage_label_en"] = TRIAGE_LEVELS[triage_level]["label_en"]
        parsed["triage_label_ar"] = TRIAGE_LEVELS[triage_level]["label_ar"]

    # 8. Add disclaimer and detected language
    parsed["disclaimer_en"]     = DISCLAIMER["en"]
    parsed["disclaimer_ar"]     = DISCLAIMER["ar"]
    parsed["detected_language"] = detected_lang

    # 9. Validate against Pydantic schema — explicit failure, never silent
    try:
        response = TriageResponse(**parsed)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Schema validation failed: {str(e)}\nRaw LLM output: {raw_output[:300]}"
        )

    return response


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    if GROQ_API_KEY:
        provider = "groq"
        model    = "llama-3.3-70b-versatile"
    elif GEMINI_API_KEY:
        provider = "gemini"
        model    = GEMINI_MODEL
    elif ANTHROPIC_API_KEY:
        provider = "anthropic"
        model    = "claude-haiku-4-5"
    elif OPENROUTER_API_KEY:
        provider = "openrouter"
        model    = os.getenv("TRIAGE_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    else:
        provider = "none"
        model    = "none"

    return {
        "status": "ok" if provider != "none" else "no_api_key",
        "provider": provider,
        "model": model,
        "kb_entries": len(KNOWLEDGE_ENTRIES)
    }


@app.get("/")
async def root():
    return {
        "service": "Mumzworld Pediatric Triage API",
        "version": "1.0.0",
        "docs": "/docs"
    }
