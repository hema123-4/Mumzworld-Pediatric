
import React, { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:8000";

const TRIAGE_COLORS = {
  emergency: { bg: "#FCEBEB", border: "#E24B4A", text: "#501313", icon: "🚨" },
  emergency_if_anaphylaxis: { bg: "#FCEBEB", border: "#E24B4A", text: "#501313", icon: "🚨" },
  emergency_if_red_flag_else_observe: { bg: "#FAEEDA", border: "#EF9F27", text: "#412402", icon: "⚠️" },
  urgent_if_red_flag_else_monitor: { bg: "#FAEEDA", border: "#EF9F27", text: "#412402", icon: "⚠️" },
  see_doctor_within_24_48h: { bg: "#FAEEDA", border: "#BA7517", text: "#412402", icon: "📋" },
  monitor_dehydration: { bg: "#E6F1FB", border: "#378ADD", text: "#042C53", icon: "💧" },
  home_care: { bg: "#EAF3DE", border: "#639922", text: "#173404", icon: "✅" },
  home_care_with_monitoring: { bg: "#EAF3DE", border: "#639922", text: "#173404", icon: "✅" },
  monitor_with_red_flag_awareness: { bg: "#EAF3DE", border: "#639922", text: "#173404", icon: "✅" },
};

const CATEGORY_LABELS = {
  medicine: "Medicine",
  health_device: "Health Device",
  skincare: "Skincare",
  hygiene: "Hygiene",
  comfort: "Comfort",
  nutrition: "Nutrition",
  hydration: "Hydration",
  health: "Health",
  first_aid: "First Aid",
  safety: "Safety",
  gear: "Baby Gear",
  warning: "⚠️ Caution",
};

function ConfidenceMeter({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? "#639922" : pct >= 60 ? "#BA7517" : "#E24B4A";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{ fontSize: 12, color: "var(--color-text-secondary)", minWidth: 80 }}>
        Confidence
      </span>
      <div style={{
        flex: 1, height: 6, background: "var(--color-background-secondary)",
        borderRadius: 3, overflow: "hidden"
      }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 3, transition: "width 0.6s ease" }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 500, color, minWidth: 32 }}>{pct}%</span>
    </div>
  );
}

function RedFlagPill({ text, type }) {
  const isPresent = type === "present";
  return (
    <span style={{
      display: "inline-block",
      fontSize: 12, padding: "3px 10px",
      borderRadius: 20,
      background: isPresent ? "#FCEBEB" : "#FAEEDA",
      color: isPresent ? "#791F1F" : "#633806",
      border: `0.5px solid ${isPresent ? "#F09595" : "#FAC775"}`,
      marginRight: 6, marginBottom: 6,
    }}>
      {isPresent ? "⚠ " : "◎ "}{text}
    </span>
  );
}

function ProductCard({ product }) {
  const isWarning = product.category === "warning";
  return (
    <div style={{
      background: isWarning ? "#FCEBEB" : "var(--color-background-primary)",
      border: `0.5px solid ${isWarning ? "#F09595" : "var(--color-border-tertiary)"}`,
      borderRadius: "var(--border-radius-md)",
      padding: "10px 12px",
      display: "flex", flexDirection: "column", gap: 4,
    }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8 }}>
        <span style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)", lineHeight: 1.3 }}>
          {product.name}
        </span>
        <span style={{
          fontSize: 11, padding: "2px 8px",
          borderRadius: 12,
          background: isWarning ? "#F7C1C1" : "var(--color-background-secondary)",
          color: isWarning ? "#501313" : "var(--color-text-secondary)",
          whiteSpace: "nowrap", flexShrink: 0,
        }}>
          {CATEGORY_LABELS[product.category] || product.category}
        </span>
      </div>
      {product.note && (
        <span style={{ fontSize: 12, color: "var(--color-text-secondary)", fontStyle: "italic" }}>
          {product.note}
        </span>
      )}
    </div>
  );
}

function LangToggle({ lang, onChange }) {
  return (
    <div style={{ display: "flex", gap: 4 }}>
      {["en", "ar"].map(l => (
        <button
          key={l}
          onClick={() => onChange(l)}
          style={{
            padding: "4px 12px", fontSize: 13,
            border: `0.5px solid ${lang === l ? "var(--color-border-primary)" : "var(--color-border-tertiary)"}`,
            borderRadius: "var(--border-radius-md)",
            background: lang === l ? "var(--color-background-secondary)" : "transparent",
            color: "var(--color-text-primary)",
            cursor: "pointer", fontWeight: lang === l ? 500 : 400,
          }}
        >
          {l === "en" ? "English" : "العربية"}
        </button>
      ))}
    </div>
  );
}

function TriageResultCard({ result, displayLang }) {
  const isAr = displayLang === "ar";
  const triage = result.triage_level;
  const colors = TRIAGE_COLORS[triage] || { bg: "#F1EFE8", border: "#888780", text: "#2C2C2A", icon: "ℹ️" };

  const section = (label, content) => content ? (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
        {label}
      </div>
      {content}
    </div>
  ) : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      {/* Triage Banner */}
      {(result.triage_label_en || result.out_of_scope) && (
        <div style={{
          background: result.out_of_scope ? "#F1EFE8" : colors.bg,
          border: `1.5px solid ${result.out_of_scope ? "#B4B2A9" : colors.border}`,
          borderRadius: "var(--border-radius-lg)",
          padding: "16px 20px",
          marginBottom: 20,
          direction: isAr ? "rtl" : "ltr",
        }}>
          <div style={{ fontSize: 18, fontWeight: 500, color: result.out_of_scope ? "#444441" : colors.text, marginBottom: 4 }}>
            {result.out_of_scope
              ? (isAr ? "خارج النطاق" : "Out of scope")
              : (isAr ? result.triage_label_ar : result.triage_label_en)}
          </div>
          {result.matched_condition && (
            <div style={{ fontSize: 13, color: result.out_of_scope ? "#888780" : colors.text, opacity: 0.8 }}>
              {result.matched_condition}
            </div>
          )}
        </div>
      )}

      {/* Confidence */}
      <div style={{ marginBottom: 16 }}>
        <ConfidenceMeter value={result.confidence} />
      </div>

      {/* Out-of-scope or needs-more-info message */}
      {result.out_of_scope && section(
        isAr ? "ملاحظة" : "Note",
        <p style={{ fontSize: 14, color: "var(--color-text-secondary)", margin: 0, direction: isAr ? "rtl" : "ltr" }}>
          {isAr ? result.reasoning_ar : result.reasoning_en}
        </p>
      )}

      {result.needs_more_info && section(
        isAr ? "نحتاج مزيداً من المعلومات" : "More info needed",
        <div style={{
          background: "#E6F1FB", border: "0.5px solid #85B7EB",
          borderRadius: "var(--border-radius-md)", padding: "10px 14px",
          fontSize: 14, color: "#042C53", direction: isAr ? "rtl" : "ltr"
        }}>
          {result.clarification_needed}
        </div>
      )}

      {/* Reasoning */}
      {!result.out_of_scope && section(
        isAr ? "التقييم" : "Assessment",
        <p style={{ fontSize: 14, color: "var(--color-text-primary)", margin: 0, lineHeight: 1.65, direction: isAr ? "rtl" : "ltr" }}>
          {isAr ? result.reasoning_ar : result.reasoning_en}
        </p>
      )}

      {/* Red flags present */}
      {result.red_flags_present?.length > 0 && section(
        isAr ? "علامات الخطر الموجودة" : "Red flags in your description",
        <div style={{ direction: isAr ? "rtl" : "ltr" }}>
          {result.red_flags_present.map((rf, i) => <RedFlagPill key={i} text={rf} type="present" />)}
        </div>
      )}

      {/* Red flags to watch */}
      {result.red_flags_to_watch?.length > 0 && section(
        isAr ? "علامات يجب مراقبتها" : "Watch for these red flags",
        <div style={{ direction: isAr ? "rtl" : "ltr" }}>
          {result.red_flags_to_watch.map((rf, i) => <RedFlagPill key={i} text={rf} type="watch" />)}
        </div>
      )}

      {/* Action guidance */}
      {!result.out_of_scope && section(
        isAr ? "ماذا تفعلين الآن" : "What to do now",
        <div style={{
          background: "var(--color-background-secondary)",
          border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-md)",
          padding: "12px 16px",
          fontSize: 14, lineHeight: 1.65,
          color: "var(--color-text-primary)",
          direction: isAr ? "rtl" : "ltr"
        }}>
          {isAr ? result.action_ar : result.action_en}
        </div>
      )}

      {/* Product recommendations */}
      {result.products?.length > 0 && section(
        isAr ? "منتجات Mumzworld المقترحة" : "Mumzworld suggested products",
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 8 }}>
          {result.products.map((p, i) => <ProductCard key={i} product={p} />)}
        </div>
      )}

      {/* Disclaimer */}
      <div style={{
        marginTop: 8,
        padding: "10px 14px",
        background: "var(--color-background-secondary)",
        borderRadius: "var(--border-radius-md)",
        borderLeft: "2px solid var(--color-border-secondary)",
        fontSize: 12, color: "var(--color-text-secondary)",
        lineHeight: 1.6, direction: isAr ? "rtl" : "ltr"
      }}>
        {isAr ? result.disclaimer_ar : result.disclaimer_en}
      </div>
    </div>
  );
}

const EXAMPLE_INPUTS = [
  { en: "My 6-week-old has a fever of 38.5°C", ar: "طفلي عمره 6 أسابيع وعنده حرارة 38.5 درجة", age: 1.5 },
  { en: "My 8-month-old has runny nose and mild cough, no fever, still eating well", ar: "طفلتي 8 أشهر عندها زكام وسعال خفيف بدون حمى وتأكل عادي", age: 8 },
  { en: "My 9-month-old lips look blue and she's breathing very fast", ar: "شفاه طفلتي 9 أشهر تبدو مزرقة وتتنفس بسرعة كبيرة", age: 9 },
  { en: "My 2-year-old has diaper rash for 3 days, no fever", ar: "طفلتي 2 سنة عندها احمرار في منطقة الحفاض من 3 أيام بدون حمى", age: 24 },
  { en: "My baby is sick", ar: "طفلي مريض", age: 6 },
];

export default function App() {
  const [symptom, setSymptom] = useState("");
  const [ageMonths, setAgeMonths] = useState("");
  const [inputLang, setInputLang] = useState("en");
  const [displayLang, setDisplayLang] = useState("en");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking");
  const resultRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(() => setApiStatus("ok"))
      .catch(() => setApiStatus("offline"));
  }, []);

  const handleSubmit = async () => {
    if (!symptom.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/triage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symptom_description: symptom,
          child_age_months: ageMonths ? parseFloat(ageMonths) : null,
          language: inputLang,
        }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      setResult(data);
      setDisplayLang(inputLang === "ar" ? "ar" : "en");
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const loadExample = (ex) => {
    setSymptom(inputLang === "ar" ? ex.ar : ex.en);
    setAgeMonths(String(ex.age));
    setResult(null);
    setError(null);
  };

  return (
    <div style={{ maxWidth: 680, margin: "0 auto", padding: "1.5rem 1rem" }}>
      <h2 style={{ sr: "only" }}>Mumzworld Pediatric Symptom Triage Assistant</h2>

      {/* Header */}
      <div style={{ marginBottom: 24, display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <span style={{ fontSize: 22, fontWeight: 500, color: "var(--color-text-primary)" }}>
              Pediatric Triage
            </span>
            <span style={{
              fontSize: 11, padding: "2px 8px",
              background: "#EAF3DE", color: "#3B6D11",
              border: "0.5px solid #97C459",
              borderRadius: 12,
            }}>Mumzworld</span>
          </div>
          <p style={{ fontSize: 14, color: "var(--color-text-secondary)", margin: 0 }}>
            Describe your child's symptoms for evidence-based triage guidance — in English or Arabic
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>API</span>
          <span style={{
            width: 8, height: 8, borderRadius: "50%",
            background: apiStatus === "ok" ? "#639922" : apiStatus === "offline" ? "#E24B4A" : "#EF9F27",
            display: "inline-block",
          }} />
        </div>
      </div>

      {/* Input form */}
      <div style={{
        background: "var(--color-background-primary)",
        border: "0.5px solid var(--color-border-tertiary)",
        borderRadius: "var(--border-radius-lg)",
        padding: "1.25rem",
        marginBottom: 16,
      }}>
        {/* Input language + display language */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>Input language:</span>
            <LangToggle lang={inputLang} onChange={setInputLang} />
          </div>
        </div>

        {/* Symptom textarea */}
        <textarea
          value={symptom}
          onChange={e => setSymptom(e.target.value)}
          placeholder={inputLang === "ar"
            ? "صفي أعراض طفلك هنا... مثال: طفلي عمره 6 أشهر عنده حمى وسعال"
            : "Describe your child's symptoms here... e.g. My 8-month-old has had a fever of 38.5°C since this morning"}
          dir={inputLang === "ar" ? "rtl" : "ltr"}
          rows={3}
          style={{
            width: "100%", boxSizing: "border-box",
            fontSize: 14, lineHeight: 1.6,
            padding: "10px 12px",
            resize: "vertical",
            fontFamily: "var(--font-sans)",
            direction: inputLang === "ar" ? "rtl" : "ltr",
          }}
        />

        {/* Age input */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 12 }}>
          <label style={{ fontSize: 13, color: "var(--color-text-secondary)", whiteSpace: "nowrap" }}>
            Child's age (months):
          </label>
          <input
            type="number"
            value={ageMonths}
            onChange={e => setAgeMonths(e.target.value)}
            placeholder="e.g. 8"
            min="0" max="216" step="0.5"
            style={{ width: 80, fontSize: 14, padding: "6px 10px" }}
          />
          <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
            (optional but recommended)
          </span>
        </div>

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={loading || !symptom.trim()}
          style={{
            marginTop: 14, width: "100%",
            padding: "10px 0", fontSize: 14, fontWeight: 500,
            opacity: loading || !symptom.trim() ? 0.5 : 1,
            cursor: loading || !symptom.trim() ? "default" : "pointer",
          }}
        >
          {loading ? "Analyzing…" : "Get triage guidance ↗"}
        </button>
      </div>

      {/* Example inputs */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 8 }}>
          Try an example:
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {EXAMPLE_INPUTS.map((ex, i) => (
            <button
              key={i}
              onClick={() => loadExample(ex)}
              style={{
                fontSize: 12, padding: "4px 10px",
                border: "0.5px solid var(--color-border-tertiary)",
                borderRadius: "var(--border-radius-md)",
                background: "var(--color-background-secondary)",
                color: "var(--color-text-secondary)",
                cursor: "pointer", textAlign: "left", maxWidth: 280,
              }}
            >
              {inputLang === "ar" ? ex.ar.slice(0, 50) : ex.en.slice(0, 55)}
              {(inputLang === "ar" ? ex.ar : ex.en).length > (inputLang === "ar" ? 50 : 55) ? "…" : ""}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          padding: "12px 16px", marginBottom: 16,
          background: "#FCEBEB", border: "0.5px solid #F09595",
          borderRadius: "var(--border-radius-md)",
          fontSize: 14, color: "#501313"
        }}>
          Error: {error}
          {apiStatus === "offline" && " — The API server is not running. See README for setup."}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div style={{
          background: "var(--color-background-secondary)",
          border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)",
          padding: "1.5rem",
          animation: "pulse 1.5s ease-in-out infinite",
        }}>
          <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }`}</style>
          {[180, 120, 200, 100].map((w, i) => (
            <div key={i} style={{
              height: i === 0 ? 24 : 14,
              width: `${w}px`, maxWidth: "90%",
              background: "var(--color-border-tertiary)",
              borderRadius: 4, marginBottom: i === 0 ? 16 : 10,
            }} />
          ))}
        </div>
      )}

      {/* Result */}
      {result && !loading && (
        <div ref={resultRef}>
          {/* Display language toggle for result */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
              Showing response in:
            </span>
            <LangToggle lang={displayLang} onChange={setDisplayLang} />
          </div>

          <div style={{
            background: "var(--color-background-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-lg)",
            padding: "1.25rem",
          }}>
            <TriageResultCard result={result} displayLang={displayLang} />
          </div>
        </div>
      )}
    </div>
  );
}
