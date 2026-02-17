import csv
import io
import json
import re
import time
import html as html_lib
from typing import Dict, List, Tuple, Set, Any, Literal

import streamlit as st
from pydantic import BaseModel, Field

# ----------------------------
# CONFIG
# ----------------------------
DEFAULT_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0

ALLOWLIST = [
    "celery",
    "cereals containing gluten",
    "crustaceans",
    "eggs",
    "fish",
    "lupin",
    "milk",
    "molluscs",
    "mustard",
    "nuts",
    "peanuts",
    "sesame",
    "soy",
    "sulphites",
]

AllowedCategory = Literal[
    "celery",
    "cereals containing gluten",
    "crustaceans",
    "eggs",
    "fish",
    "lupin",
    "milk",
    "molluscs",
    "mustard",
    "nuts",
    "peanuts",
    "sesame",
    "soy",
    "sulphites",
]

SYSTEM_PROMPT = (
    "You are a compliance auditor.\n"
    "You will be given JSON with:\n"
    "- sku, sku_name\n"
    "- ingredients_marked: a single string where bolded text is wrapped as [[B]]...[[/B]].\n"
    "- candidate_evidence: a JSON list of ONLY the allergen synonyms that Python actually found.\n\n"
    "Your task: decide which of the 14 regulated allergen CATEGORIES are mentioned in a NON-precautionary way\n"
    "with an UNBOLDED mention that is NOT properly covered.\n\n"
    "Critical rules:\n"
    "1) You MUST ONLY use candidate_evidence to decide (do not invent or search for other allergens).\n"
    "2) Ignore precautionary statements (may contain / traces / facility handles / made in a site that handles / similar).\n"
    "3) Component coverage rule:\n"
    "   - candidate_evidence includes component_id.\n"
    "   - If a category has an unbolded synonym in a component BUT that SAME component also has a bolded synonym for that category,\n"
    "     treat it as compliant (do NOT flag).\n"
    "4) Do not flag gluten-free phrases; do not treat sulphate/sulfate as sulphites.\n"
    "5) If the product is clearly topical/cosmetic, set is_topical=true and return no allergens.\n\n"
    "Return only the structured JSON output."
)

# ----------------------------
# Candidate detection
# ----------------------------
def _word_regex(words: List[str]) -> re.Pattern:
    escaped = [re.escape(w) for w in words]
    pattern = r"(?i)(?<![A-Za-z0-9])(" + "|".join(escaped) + r")(?![A-Za-z0-9])"
    return re.compile(pattern)

CANDIDATE_TERMS = [
    "celery",
    "wheat", "rye", "barley", "oat", "oats",
    "crustacean", "crustaceans", "shrimp", "prawn", "prawns", "crab", "lobster",
    "mollusc", "molluscs", "mussel", "mussels", "oyster", "oysters", "squid", "octopus", "clam", "clams",
    "egg", "eggs", "albumen",
    "fish", "salmon", "tuna", "cod", "anchovy", "anchovies",
    "lupin", "lupine",
    "milk", "whey", "casein", "lactose", "butter", "cream", "cheese", "yoghurt", "yogurt",
    "mustard",
    "almond", "almonds", "hazelnut", "hazelnuts", "walnut", "walnuts", "cashew", "cashews",
    "pecan", "pecans", "brazil nut", "brazil nuts", "pistachio", "pistachios", "macadamia", "macadamias",
    "peanut", "peanuts", "groundnut", "groundnuts",
    "sesame",
    "soy", "soya", "soja",
    "sulphite", "sulphites", "sulfite", "sulfites", "so2", "sulfur dioxide", "sulphur dioxide",
    "e220", "e221", "e222", "e223", "e224", "e225", "e226", "e227", "e228",
]
SULPHATE_EXCLUSIONS = re.compile(r"(?i)(?<![A-Za-z0-9])(sulphate|sulphates|sulfate|sulfates)(?![A-Za-z0-9])")
CANDIDATE_REGEX = _word_regex(CANDIDATE_TERMS)

def is_candidate(ingredients: str) -> Tuple[bool, List[str]]:
    if not ingredients:
        return False, []
    text = ingredients.strip()
    matches = [m.group(1) for m in CANDIDATE_REGEX.finditer(text)]
    if not matches:
        return False, []
    filtered = []
    for term in matches:
        if SULPHATE_EXCLUSIONS.search(term):
            continue
        filtered.append(term)
    return (len(filtered) > 0), sorted(set(filtered), key=str.lower)

# ----------------------------
# HTML -> marked string [[B]]...[[/B]]
# ----------------------------
BOLD_OPEN = re.compile(r"(?i)<\s*(strong|b)\b[^>]*>")
BOLD_CLOSE = re.compile(r"(?i)<\s*/\s*(strong|b)\s*>")
BLOCK_TAGS = re.compile(r"(?i)<\s*/?\s*(p|div|br|li|ul|ol|tr|td|th)\b[^>]*>")

def html_to_marked_text(html: str) -> str:
    if not html:
        return ""
    s = html
    s = BOLD_OPEN.sub("[[B]]", s)
    s = BOLD_CLOSE.sub("[[/B]]", s)
    s = BLOCK_TAGS.sub("\n", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = html_lib.unescape(s)
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def iter_marked_segments(marked: str):
    i = 0
    bold = 0
    buf: List[str] = []
    while i < len(marked):
        if marked.startswith("[[B]]", i):
            if buf:
                yield ("".join(buf), bold)
                buf = []
            bold += 1
            i += 5
            continue
        if marked.startswith("[[/B]]", i):
            if buf:
                yield ("".join(buf), bold)
                buf = []
            bold = max(0, bold - 1)
            i += 6
            continue
        buf.append(marked[i])
        i += 1
    if buf:
        yield ("".join(buf), bold)

# ----------------------------
# Evidence extraction
# ----------------------------
PRECAUTION_RE = re.compile(
    r"(?i)(may\s+contain|traces?\s+of|manufactured\s+in|made\s+in|facility\s+that\s+handles|factory\s+that\s+handles|site\s+that\s+handles)"
)
GLUTEN_FREE_RE = re.compile(r"(?i)\b(gluten[-\s]?free|free\s+from\s+gluten)\b")
SULPHATE_RE = re.compile(r"(?i)\b(sulphate|sulphates|sulfate|sulfates)\b")

SYNONYM_TO_CATEGORY = {
    "wheat": "cereals containing gluten",
    "rye": "cereals containing gluten",
    "barley": "cereals containing gluten",
    "oat": "cereals containing gluten",
    "oats": "cereals containing gluten",

    "milk": "milk",
    "milk powder": "milk",
    "skimmed milk": "milk",
    "whey": "milk",
    "whey powder": "milk",
    "casein": "milk",
    "lactose": "milk",
    "butter": "milk",
    "cream": "milk",
    "cheese": "milk",
    "yoghurt": "milk",
    "yogurt": "milk",

    "soy": "soy",
    "soya": "soy",
    "soja": "soy",

    "almond": "nuts",
    "almonds": "nuts",
    "hazelnut": "nuts",
    "hazelnuts": "nuts",
    "walnut": "nuts",
    "walnuts": "nuts",
    "cashew": "nuts",
    "cashews": "nuts",
    "pecan": "nuts",
    "pecans": "nuts",
    "brazil nut": "nuts",
    "brazil nuts": "nuts",
    "pistachio": "nuts",
    "pistachios": "nuts",
    "macadamia": "nuts",
    "macadamias": "nuts",

    "peanut": "peanuts",
    "peanuts": "peanuts",
    "groundnut": "peanuts",
    "groundnuts": "peanuts",

    "egg": "eggs",
    "eggs": "eggs",
    "albumen": "eggs",

    "fish": "fish",
    "salmon": "fish",
    "tuna": "fish",
    "cod": "fish",
    "anchovy": "fish",
    "anchovies": "fish",

    "lupin": "lupin",
    "lupine": "lupin",

    "mustard": "mustard",
    "celery": "celery",
    "sesame": "sesame",

    "sulphite": "sulphites",
    "sulphites": "sulphites",
    "sulfite": "sulphites",
    "sulfites": "sulphites",
    "so2": "sulphites",
    "sulfur dioxide": "sulphites",
    "sulphur dioxide": "sulphites",
    "e220": "sulphites",
    "e221": "sulphites",
    "e222": "sulphites",
    "e223": "sulphites",
    "e224": "sulphites",
    "e225": "sulphites",
    "e226": "sulphites",
    "e227": "sulphites",
    "e228": "sulphites",
}

def _compile_term(term: str) -> re.Pattern:
    esc = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"(?i)(?<![A-Za-z0-9])({esc})(?![A-Za-z0-9])")

TERM_PATTERNS = [(t, _compile_term(t)) for t in sorted(SYNONYM_TO_CATEGORY.keys(), key=len, reverse=True)]

def compute_component_bounds(text: str) -> List[Tuple[int, int]]:
    bounds: List[Tuple[int, int]] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and ch in [",", ";", "\n"]:
            bounds.append((start, i))
            start = i + 1
    bounds.append((start, len(text)))
    return bounds

def compute_component_id_map(text: str) -> List[int]:
    ids = [0] * len(text)
    depth = 0
    comp = 0
    for i, ch in enumerate(text):
        ids[i] = comp
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and ch in [",", ";", "\n"]:
            comp += 1
    return ids

def extract_candidate_evidence(marked: str) -> List[Dict[str, Any]]:
    segments = list(iter_marked_segments(marked))
    plain = "".join(seg for seg, _ in segments)
    comp_bounds = compute_component_bounds(plain)
    comp_id_map = compute_component_id_map(plain) if plain else []
    evidence: List[Dict[str, Any]] = []
    cursor = 0

    for seg_text, bold_level in segments:
        seg_start = cursor
        seg_end = cursor + len(seg_text)

        for term, pat in TERM_PATTERNS:
            for m in pat.finditer(seg_text):
                found = m.group(1)
                abs_start = seg_start + m.start(1)
                abs_end = seg_start + m.end(1)

                cat = SYNONYM_TO_CATEGORY[term]
                is_bolded = bold_level > 0

                window = plain[max(0, abs_start - 30):min(len(plain), abs_end + 30)]
                if cat == "cereals containing gluten" and GLUTEN_FREE_RE.search(window):
                    continue
                if cat == "sulphites" and SULPHATE_RE.search(window):
                    continue

                clause = plain[max(0, abs_start - 80):min(len(plain), abs_end + 120)]
                is_precaution_like = bool(PRECAUTION_RE.search(clause))

                component_id = comp_id_map[abs_start] if comp_id_map and abs_start < len(comp_id_map) else 0
                cb_start, cb_end = comp_bounds[component_id] if component_id < len(comp_bounds) else (0, len(plain))
                component_text = plain[cb_start:cb_end].strip()
                context = plain[max(0, abs_start - 50):min(len(plain), abs_end + 50)].strip()

                evidence.append({
                    "category": cat,
                    "term": found,
                    "is_bolded": is_bolded,
                    "is_precaution_like": is_precaution_like,
                    "component_id": component_id,
                    "component_text": component_text,
                    "context": context,
                })

        cursor = seg_end

    # De-dupe
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for e in evidence:
        key = (e["category"], e["term"].lower(), e["is_bolded"], e["is_precaution_like"], e["component_id"], e["context"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq

def uncovered_categories_from_evidence(evidence: List[Dict[str, Any]]) -> Set[str]:
    bold_comps: Dict[str, Set[int]] = {}
    unbold_hits: Dict[str, List[Dict[str, Any]]] = {}

    for e in evidence:
        cat = e["category"]
        if e.get("is_precaution_like"):
            continue
        if e.get("is_bolded"):
            bold_comps.setdefault(cat, set()).add(int(e.get("component_id", 0)))
        else:
            unbold_hits.setdefault(cat, []).append(e)

    uncovered: Set[str] = set()
    for cat, hits in unbold_hits.items():
        covered = bold_comps.get(cat, set())
        if any(int(h.get("component_id", 0)) not in covered for h in hits):
            uncovered.add(cat)
    return uncovered

# ----------------------------
# Structured output model
# ----------------------------
class AuditResult(BaseModel):
    unbolded_allergens: List[AllowedCategory] = Field(default_factory=list)
    debug_matches: List[str] = Field(default_factory=list)
    is_topical: bool = False

# ----------------------------
# OpenAI call
# ----------------------------
def openai_check(client, model: str, row: Dict[str, str]) -> Dict[str, Any]:
    ingredients_html = row.get("ingredients", "") or ""
    marked = html_to_marked_text(ingredients_html)
    evidence = extract_candidate_evidence(marked)

    if not evidence:
        return {"unbolded_allergens": [], "debug_matches": [], "is_topical": False}

    payload = {
        "sku": row.get("sku", "") or "",
        "sku_name": row.get("sku_name", "") or "",
        "ingredients_marked": marked,
        "candidate_evidence": evidence,
    }
    user_msg = json.dumps(payload, ensure_ascii=False)

    resp = client.responses.parse(
        model=model,
        temperature=TEMPERATURE,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        text_format=AuditResult,
    )

    parsed: AuditResult = resp.output_parsed
    data = parsed.model_dump()

    # Post-filter safety
    uncovered = uncovered_categories_from_evidence(evidence)
    model_cats = [c for c in data.get("unbolded_allergens", []) if c in ALLOWLIST]
    final_cats = [c for c in ALLOWLIST if c in model_cats and c in uncovered]

    if data.get("is_topical", False):
        final_cats = []

    data["unbolded_allergens"] = final_cats
    return data

# ----------------------------
# CSV helpers
# ----------------------------
def read_csv_upload(uploaded) -> Tuple[List[Dict[str, str]], List[str]]:
    raw = uploaded.getvalue().decode("utf-8-sig", errors="replace")
    f = io.StringIO(raw)
    reader = csv.DictReader(f)
    rows = list(reader)
    headers = reader.fieldnames or []
    return rows, headers

def to_csv_bytes(rows: List[Dict[str, Any]], fieldnames: List[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Allergen Bold Audit", layout="wide")
st.title("Allergen Bold Audit (CSV → OpenAI)")

st.markdown("**Input CSV must contain columns:** `sku`, `sku_name`, `ingredients`")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API key", type="password", help="Used only for this session, not saved.")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    run_button = st.button("Run audit")

uploaded = st.file_uploader("Upload input CSV", type=["csv"])

if uploaded and run_button:
    if not api_key.strip():
        st.error("Please enter your API key in the sidebar.")
        st.stop()

    # Lazy import so app loads even if openai not installed yet
    try:
        from openai import OpenAI
    except Exception:
        st.error("Missing dependency: openai. Install with `pip install openai`.")
        st.stop()

    client = OpenAI(api_key=api_key.strip())

    rows, headers = read_csv_upload(uploaded)
    required = {"sku", "sku_name", "ingredients"}
    if not required.issubset(set(h.lower() for h in headers)):
        st.error(f"CSV headers found: {headers}\n\nRequired: sku, sku_name, ingredients")
        st.stop()

    # Normalize column keys to expected names (case-insensitive)
    # Build a mapping from lowercase -> actual
    header_map = {h.lower(): h for h in headers}
    def norm_row(r: Dict[str, str]) -> Dict[str, str]:
        return {
            "sku": r.get(header_map["sku"], ""),
            "sku_name": r.get(header_map["sku_name"], ""),
            "ingredients": r.get(header_map["ingredients"], ""),
        }

    output_rows: List[Dict[str, Any]] = []
    progress = st.progress(0)
    status = st.empty()

    total = len(rows)
    for i, raw_row in enumerate(rows, start=1):
        row = norm_row(raw_row)
        sku = (row.get("sku") or "").strip()
        cand, hits = is_candidate(row.get("ingredients", ""))

        out = {
            "sku": row.get("sku", ""),
            "sku_name": row.get("sku_name", ""),
            "ingredients": row.get("ingredients", ""),
            "candidate": "Y" if cand else "N",
            "candidate_hits": ", ".join(hits),
            "unbolded_allergens": "",
            "debug_matches_json": "[]",
            "is_topical": "N",
            "status": "skipped_non_candidate",
            "error": "",
        }

        if cand:
            status.text(f"[{i}/{total}] SKU={sku or '(blank)'} calling OpenAI…")
            try:
                result = openai_check(client, model, row)
                cats = result.get("unbolded_allergens", []) or []
                out["unbolded_allergens"] = ", ".join(cats)
                out["debug_matches_json"] = json.dumps(result.get("debug_matches", []), ensure_ascii=False)
                out["is_topical"] = "Y" if result.get("is_topical", False) else "N"
                out["status"] = "ok"
            except Exception as e:
                out["status"] = "error"
                out["error"] = str(e)

        output_rows.append(out)
        progress.progress(i / total)

    status.text("Done.")

    fieldnames = [
        "sku", "sku_name", "ingredients",
        "candidate", "candidate_hits",
        "unbolded_allergens", "debug_matches_json",
        "is_topical",
        "status", "error",
    ]

    st.success("Audit complete.")
    st.dataframe(output_rows, use_container_width=True)
    st.download_button(
        "Download output CSV",
        data=to_csv_bytes(output_rows, fieldnames),
        file_name="output.csv",
        mime="text/csv",
    )

elif uploaded and not run_button:
    st.info("Upload looks good. Enter your API key in the sidebar, then click **Run audit**.")
