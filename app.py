"""
Hiberno-English Lexical Extraction — Streamlit review app
Comparing Rule-Based vs LLM extraction on A Dictionary of Hiberno-English (Dolan, 2006).
"""

import json
import re
import unicodedata
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="Hiberno-English Lexical Extraction",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "comparison.json")

CONTENT_FIELDS = [
    "headword", "headword_raw", "homonym_index", "variant_forms",
    "pronunciation", "pronunciation_2", "pronunciation_3",
    "part_of_speech", "grammatical_labels", "definition",
    "etymology", "cross_references", "examples", "region_mentions",
]

STRATA = [
    "stratum_high_confidence", "stratum_medium_confidence", "stratum_low_confidence",
    "stratum_numbered_senses", "stratum_who_adds", "stratum_multi_pronunciation",
    "stratum_variant_forms", "stratum_with_etymology", "stratum_multi_region",
    "stratum_long_definition",
]

MISCLASS_FLAGS = [
    "misclass_etym_in_definition",
    "misclass_examples_in_definition",
    "misclass_chained_etym_collapsed",
    "misclass_examples_dropped",
    "misclass_numbered_senses_not_split",
    "misclass_cross_ref_missed",
]

MISCLASS_DESCRIPTIONS = {
    "misclass_etym_in_definition": (
        "LLM placed etymology content (< language marker) inside "
        "the definition field instead of the etymology field."
    ),
    "misclass_examples_in_definition": (
        "LLM definition contains a single-quoted string while the "
        "rule-based parser has examples. Many of these are legitimate "
        "— definitions like 'in the phrase X' are normal for this "
        "dictionary. Manual review needed to distinguish true errors."
    ),
    "misclass_chained_etym_collapsed": (
        "Rule-based etymology has multiple < markers (chained sources) "
        "but LLM only captured the first. e.g. '< Ir. < Ir a mhuirnín.' "
        "collapsed to '< Ir.'"
    ),
    "misclass_examples_dropped": (
        "Rule-based has examples but LLM returned null. Often occurs "
        "when examples are embedded in EDD citations or 'in such "
        "phrases as' constructions."
    ),
    "misclass_numbered_senses_not_split": (
        "Rule-based definition is a list (numbered senses) but LLM "
        "returned a single string."
    ),
    "misclass_cross_ref_missed": (
        "Rule-based has cross_references but LLM returned null. "
        "Often when 'See X.' appears mid-entry rather than at the end."
    ),
}

# Colour palette
COLOR_RB    = "#1f77b4"
COLOR_LLM   = "#ff7f0e"
COLOR_MATCH = "#2ca02c"
COLOR_NOMATCH = "#d62728"

STOPWORDS = {
    "see", "also", "the", "and", "of", "in", "to", "for", "or",
    "as", "is", "an", "on", "at", "by", "it", "its", "from",
    "with", "this", "that", "was", "be", "are", "used", "who",
    "adds", "one", "he", "she", "his", "her", "they", "them",
    "their", "not", "no", "but", "if", "so", "up",
}

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise(value) -> str:
    """Flatten any field value to a comparable lowercase string."""
    try:
        if value is None:
            s = ""
        elif isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(" ".join(str(v) for v in item.values()))
                else:
                    parts.append(str(item))
            s = " ".join(sorted(parts))
        elif isinstance(value, dict):
            s = " ".join(str(v) for v in value.values())
        else:
            s = str(value)

        # Normalise fullwidth Unicode
        s = "".join(
            chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c
            for c in s
        )
        # Strip diacritics
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    except Exception:
        return ""


def get_tokens(value) -> set:
    """Return meaningful tokens from a field value."""
    return {
        t for t in normalise(value).split()
        if len(t) > 2 and not t.isdigit()
    }


def display_value(value) -> str:
    """Human-readable display of a field value."""
    if value is None:
        return "—"
    if isinstance(value, list):
        if not value:
            return "—"
        if value and isinstance(value[0], dict):
            # region_mentions style
            parts = []
            for item in value:
                informant = item.get("informant", "")
                county = item.get("county", "")
                parts.append(f"{informant} ({county})" if county else informant)
            return "; ".join(parts)
        return ", ".join(str(v) for v in value)
    return str(value)


def strip_prefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s.replace("_", " ")


def fmt_stratum(s: str) -> str:
    return strip_prefix(s, "stratum_")


def fmt_misclass(s: str) -> str:
    return strip_prefix(s, "misclass_")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    entries = data["entries"]

    # Build headword index sorted alphabetically
    hw_index = sorted(
        [
            {
                "para_id": e["para_id"],
                "headword": e["rb"].get("headword") or e["llm"].get("headword") or "",
                "headword_raw": e["rb"].get("headword_raw") or "",
            }
            for e in entries
        ],
        key=lambda x: x["headword"].lower() if x["headword"] else "",
    )

    # Build a para_id → entry dict for fast lookup
    entry_map = {e["para_id"]: e for e in entries}

    return data, hw_index, entry_map


data, hw_index, entry_map = load_data()
aggregate = data["aggregate"]
entries = data["entries"]
classification_disagreements = data.get("classification_disagreements", [])

EMBED_PATH = os.path.join(os.path.dirname(__file__), "data", "embeddings.json")


@st.cache_data
def load_embeddings():
    if not os.path.exists(EMBED_PATH):
        return None
    with open(EMBED_PATH, encoding="utf-8") as f:
        return json.load(f)


embed_data = load_embeddings()


# ─────────────────────────────────────────────────────────────────────────────
# Token annotation (Page 5)
# ─────────────────────────────────────────────────────────────────────────────

def annotate_tokens(full_text: str, rb: dict, llm: dict) -> list:
    """Annotate each whitespace-delimited token in full_text with a colour."""
    # Build lookup sets from all content fields
    rb_tokens: set = set()
    llm_tokens: set = set()
    for f in CONTENT_FIELDS:
        rb_tokens |= get_tokens(rb.get(f))
        llm_tokens |= get_tokens(llm.get(f))

    result = []
    for token in full_text.split():
        norm = normalise(token)
        # A source token like "left-handed" normalises to "left handed" (two words).
        # Membership in rb_tokens/llm_tokens must use a set-intersection check on
        # the individual sub-tokens, not a string-equality check on the joined form.
        norm_parts = {t for t in norm.split() if len(t) > 2 and not t.isdigit()}
        is_stop   = norm in STOPWORDS or len(norm) <= 2 or norm.isdigit() or not norm_parts
        rb_found  = bool(norm_parts & rb_tokens)  if not is_stop else False
        llm_found = bool(norm_parts & llm_tokens) if not is_stop else False

        if is_stop:
            colour = "stop"
        elif rb_found and llm_found:
            colour = "both"
        elif rb_found:
            colour = "rb"
        elif llm_found:
            colour = "llm"
        else:
            colour = "miss"

        result.append({"token": token, "norm": norm, "colour": colour})
    return result


TOKEN_STYLES = {
    "both": "background:#d4edda;color:#155724;padding:1px 4px;border-radius:3px;",
    "rb":   "background:#cce5ff;color:#004085;padding:1px 4px;border-radius:3px;",
    "llm":  "background:#fff3cd;color:#856404;padding:1px 4px;border-radius:3px;",
    "miss": "background:#f8d7da;color:#721c24;padding:1px 4px;border-radius:3px;",
    "stop": "color:#999;font-size:0.85em;",
}

LEGEND_HTML = (
    '<span style="background:#d4edda;color:#155724;padding:1px 6px;border-radius:3px;margin-right:6px;">■ Both systems</span>'
    '<span style="background:#cce5ff;color:#004085;padding:1px 6px;border-radius:3px;margin-right:6px;">■ Rule-based only</span>'
    '<span style="background:#fff3cd;color:#856404;padding:1px 6px;border-radius:3px;margin-right:6px;">■ LLM only</span>'
    '<span style="background:#f8d7da;color:#721c24;padding:1px 6px;border-radius:3px;margin-right:6px;">■ Neither</span>'
    '<span style="color:#999;font-size:0.85em;">· stopword</span>'
)


def render_tokens_html(annotated: list) -> str:
    parts = []
    for item in annotated:
        style = TOKEN_STYLES[item["colour"]]
        parts.append(f'<span style="{style}">{item["token"]}</span>')
    return '<p style="line-height:2.2;word-wrap:break-word;">' + " ".join(parts) + "</p>"


def render_field_tokens_html(value, rb_tokens: set, llm_tokens: set) -> str:
    """Render a field value with token colouring based on cross-system overlap."""
    text = display_value(value)
    if text == "—":
        return '<span style="color:#999;">—</span>'
    parts = []
    for token in text.split():
        norm = normalise(token)
        norm_parts = {t for t in norm.split() if len(t) > 2 and not t.isdigit()}
        is_stop   = norm in STOPWORDS or len(norm) <= 2 or norm.isdigit() or not norm_parts
        rb_found  = bool(norm_parts & rb_tokens)  if not is_stop else False
        llm_found = bool(norm_parts & llm_tokens) if not is_stop else False
        if is_stop:
            colour = "stop"
        elif rb_found and llm_found:
            colour = "both"
        elif rb_found:
            colour = "rb"
        elif llm_found:
            colour = "llm"
        else:
            colour = "miss"
        style = TOKEN_STYLES[colour]
        parts.append(f'<span style="{style}">{token}</span>')
    return '<span style="line-height:2.2;">' + " ".join(parts) + "</span>"


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

if "selected_para_id" not in st.session_state:
    st.session_state.selected_para_id = None
if "p5_index" not in st.session_state:
    st.session_state.p5_index = 0
if "current_page" not in st.session_state:
    st.session_state.current_page = "semantic"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="padding:0 0 16px 0;">
  <div style="font-size:1.1rem;font-weight:700;color:#1a1a2e;
              letter-spacing:-0.3px;line-height:1.2;">
    Hiberno-English
  </div>
  <div style="font-size:0.75rem;color:#666;margin-top:2px;
              letter-spacing:0.3px;text-transform:uppercase;">
    Lexical Extraction
  </div>
</div>
""", unsafe_allow_html=True)

NAV_PAGES = [
    ("semantic",   "Semantic Space",      "◉"),
    ("overview",   "Overview",            "○"),
    ("browser",    "Entry Browser",       "≡"),
    ("deepdive",   "Field Deep-Dive",     "⊞"),
    ("misclass",   "Misclassification",   "⚑"),
    ("annotation", "Source Annotation",   "◈"),
]

# Inject CSS to style the nav buttons
st.sidebar.markdown("""
<style>
/* Nav button base */
div[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    text-align: left;
    padding: 7px 12px;
    margin: 1px 0;
    border-radius: 6px;
    border: 1px solid transparent;
    background: transparent;
    color: #444;
    font-size: 0.875rem;
    font-weight: 400;
    line-height: 1.4;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
}
div[data-testid="stSidebar"] div.stButton > button:hover {
    background: #f0f0f5;
    color: #1a1a2e;
    border-color: #e0e0e8;
}
</style>
""", unsafe_allow_html=True)

for key, label, icon in NAV_PAGES:
    is_active = st.session_state.current_page == key
    if is_active:
        # Active item rendered as styled HTML (not a button)
        st.sidebar.markdown(
            f'<div style="background:#1a1a2e;color:#fff;padding:7px 12px;'
            f'border-radius:6px;font-size:0.875rem;font-weight:600;'
            f'margin:1px 0;cursor:default;">'
            f'{icon}&nbsp;&nbsp;{label}</div>',
            unsafe_allow_html=True,
        )
    else:
        if st.sidebar.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.current_page = key
            st.rerun()

st.sidebar.markdown("<hr style='margin:12px 0;border-color:#e8e8e8;'>", unsafe_allow_html=True)

page = st.session_state.current_page

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────

if page == "overview":
    st.title("Overview")

    ea = aggregate["exact_agreement_pct"]
    cov = aggregate["coverage"]
    n_fields = len(CONTENT_FIELDS)
    fields_ge90 = sum(1 for v in ea.values() if v >= 90)
    fields_lt80 = sum(1 for v in ea.values() if v < 80)

    # Row 1: metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entries Evaluated", f"{len(entries):,}")
    c2.metric("Mean LLM Token Coverage", f"{cov['mean_llm_coverage_pct']:.1f}%")
    c3.metric(f"Fields ≥90% Exact Agreement", f"{fields_ge90} / {n_fields}")
    c4.metric(f"Fields <80% Exact Agreement", f"{fields_lt80} / {n_fields}")

    st.markdown("---")

    # Row 2: Exact agreement bar chart
    st.subheader("Exact Agreement by Field")
    sorted_fields = sorted(CONTENT_FIELDS, key=lambda f: ea.get(f, 0))
    bar_vals = [ea.get(f, 0) for f in sorted_fields]
    bar_labels = [f.replace("_", " ") for f in sorted_fields]
    bar_colors = [
        COLOR_MATCH if v >= 90 else ("#ff7f0e" if v >= 80 else COLOR_NOMATCH)
        for v in bar_vals
    ]

    fig_ea = go.Figure(go.Bar(
        x=bar_vals,
        y=bar_labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in bar_vals],
        textposition="outside",
    ))
    fig_ea.add_vline(x=80, line_dash="dash", line_color=COLOR_NOMATCH,
                     annotation_text="80%", annotation_position="top right")
    fig_ea.update_layout(
        height=420,
        margin=dict(l=10, r=60, t=20, b=20),
        xaxis=dict(range=[0, 108], title="Exact Agreement %"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    st.plotly_chart(fig_ea, use_container_width=True)

    st.markdown("---")

    # Row 3: Coverage + Misclassification
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Source Coverage")
        fig_cov = go.Figure()
        fig_cov.add_trace(go.Bar(
            name="Rule-Based",
            x=["Token Coverage", "Char Coverage"],
            y=[cov["mean_rb_coverage_pct"], cov["mean_rb_char_pct"]],
            marker_color=COLOR_RB,
            text=[f"{cov['mean_rb_coverage_pct']:.2f}%", f"{cov['mean_rb_char_pct']:.2f}%"],
            textposition="outside",
        ))
        fig_cov.add_trace(go.Bar(
            name="LLM",
            x=["Token Coverage", "Char Coverage"],
            y=[cov["mean_llm_coverage_pct"], cov["mean_llm_char_pct"]],
            marker_color=COLOR_LLM,
            text=[f"{cov['mean_llm_coverage_pct']:.2f}%", f"{cov['mean_llm_char_pct']:.2f}%"],
            textposition="outside",
        ))
        fig_cov.update_layout(
            barmode="group",
            yaxis=dict(range=[92, 101.5], title="Coverage %"),
            height=340,
            margin=dict(l=10, r=10, t=20, b=20),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_cov, use_container_width=True)

    with col_right:
        st.subheader("Misclassification Counts")
        mc = aggregate["misclassification_counts"]
        mc_items = [(k, v) for k, v in mc.items() if k != "misclass_any"]
        mc_items_sorted = sorted(mc_items, key=lambda x: x[1], reverse=True)
        mc_labels = [fmt_misclass(k) for k, _ in mc_items_sorted]
        mc_vals   = [v for _, v in mc_items_sorted]
        fig_mc = go.Figure(go.Bar(
            x=mc_vals,
            y=mc_labels,
            orientation="h",
            marker_color=COLOR_NOMATCH,
            text=mc_vals,
            textposition="outside",
        ))
        fig_mc.update_layout(
            height=340,
            margin=dict(l=10, r=40, t=20, b=20),
            xaxis=dict(title="Count"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        st.caption(
            "**Note:** *examples in definition* (419) — many are legitimate "
            "(definition includes quoted phrases). Manual review needed."
        )

    st.markdown("---")

    # Row 4: Stratum × field heatmap
    st.subheader("Stratum × Field Agreement Heatmap")
    heatmap_fields = [
        "definition", "etymology", "examples", "part_of_speech",
        "region_mentions", "cross_references",
    ]
    stratum_agg = aggregate["stratum_agreement"]
    # Build matrix: rows = strata, cols = fields
    # We use exact_agreement_pct from aggregate, broken down by stratum
    # The stratum_agreement only has overall n/exact_agreement_pct/mean_token_similarity
    # We need per-stratum per-field — compute from entries
    # Build {stratum: {field: [values]}}
    stratum_field_agree: dict = {s: {f: [] for f in heatmap_fields} for s in STRATA}
    for e in entries:
        comp = e["comparison"]
        strata_flags = comp.get("strata", {})
        ea_entry = comp.get("exact_agreement", {})
        for s in STRATA:
            if strata_flags.get(s, False):
                for f in heatmap_fields:
                    val = ea_entry.get(f)
                    if val is not None:
                        stratum_field_agree[s][f].append(1 if val else 0)

    z_vals = []
    z_text = []
    for s in STRATA:
        row = []
        row_text = []
        for f in heatmap_fields:
            vals = stratum_field_agree[s][f]
            pct = (sum(vals) / len(vals) * 100) if vals else 0
            row.append(round(pct, 1))
            row_text.append(f"{pct:.1f}%")
        z_vals.append(row)
        z_text.append(row_text)

    fig_hm = go.Figure(go.Heatmap(
        z=z_vals,
        x=[f.replace("_", " ") for f in heatmap_fields],
        y=[fmt_stratum(s) for s in STRATA],
        colorscale=[[0, "#d62728"], [0.5, "#ffff00"], [1, "#2ca02c"]],
        zmin=0, zmax=100,
        text=z_text,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Agreement %"),
    ))
    fig_hm.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis=dict(title="Field"),
        yaxis=dict(title="Stratum"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    # Classification disagreements
    st.subheader("Classification Disagreements")
    st.info(
        f"**{len(classification_disagreements)} entries** were classified differently by the two systems. "
        "The LLM is correct in all cases — these are 'See X.' pointer entries that the rule-based "
        "parser incorrectly treated as full dictionary entries."
    )
    if classification_disagreements:
        cd_rows = []
        for cd in classification_disagreements:
            cd_rows.append({
                "para_id": cd["para_id"],
                "headword": cd.get("headword", ""),
                "full_text": (cd.get("full_text", "") or "")[:120],
                "rb_type": cd.get("rb_type", ""),
                "llm_type": cd.get("llm_type", ""),
            })
        st.dataframe(pd.DataFrame(cd_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Entry Browser
# ─────────────────────────────────────────────────────────────────────────────

elif page == "browser":
    st.title("Entry Browser")

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Filters")

    hw_list = [h["headword"] for h in hw_index if h["headword"]]

    jump_hw = st.sidebar.text_input("Jump to headword", key="p2_jump")
    search_hw = st.sidebar.text_input("Search headword contains", key="p2_search")

    strata_choices = [fmt_stratum(s) for s in STRATA]
    sel_strata = st.sidebar.multiselect("Strata", strata_choices, key="p2_strata")

    flag_choices = [fmt_misclass(f) for f in MISCLASS_FLAGS]
    sel_flags = st.sidebar.multiselect("Misclassification flags", flag_choices, key="p2_flags")

    min_agree = st.sidebar.slider("Min agree %", 0, 100, 0, key="p2_min_agree")
    max_agree = st.sidebar.slider("Max agree %", 0, 100, 100, key="p2_max_agree")
    max_llm_cov = st.sidebar.slider("Max LLM coverage %", 0, 100, 100, key="p2_llm_cov")

    sort_opt = st.sidebar.selectbox(
        "Sort by",
        [
            "agree_pct ascending (worst first)",
            "agree_pct descending (best first)",
            "llm_coverage_pct ascending",
            "para_id ascending",
        ],
        key="p2_sort",
    )

    # ── Filter entries ────────────────────────────────────────────────────────
    sel_strata_raw = [STRATA[strata_choices.index(s)] for s in sel_strata]
    sel_flags_raw  = [MISCLASS_FLAGS[flag_choices.index(f)] for f in sel_flags]

    filtered = []
    for e in entries:
        comp = e["comparison"]
        hw   = e["rb"].get("headword") or e["llm"].get("headword") or ""

        if search_hw and search_hw.lower() not in hw.lower():
            continue
        ap = comp.get("agree_pct", 0) or 0
        if not (min_agree <= ap <= max_agree):
            continue
        lc = comp.get("llm_coverage_pct", 100) or 100
        if lc > max_llm_cov:
            continue
        if sel_strata_raw:
            strata_flags = comp.get("strata", {})
            if not all(strata_flags.get(s, False) for s in sel_strata_raw):
                continue
        if sel_flags_raw:
            mf = comp.get("misclassification_flags", {})
            if not all(mf.get(f, False) for f in sel_flags_raw):
                continue
        filtered.append(e)

    # Sort
    if sort_opt.startswith("agree_pct ascending"):
        filtered.sort(key=lambda e: e["comparison"].get("agree_pct", 0) or 0)
    elif sort_opt.startswith("agree_pct descending"):
        filtered.sort(key=lambda e: e["comparison"].get("agree_pct", 0) or 0, reverse=True)
    elif sort_opt.startswith("llm_coverage_pct"):
        filtered.sort(key=lambda e: e["comparison"].get("llm_coverage_pct", 100) or 100)
    else:
        filtered.sort(key=lambda e: e["para_id"])

    # Handle jump-to-headword
    if jump_hw:
        jump_hw_lower = jump_hw.lower()
        match = next(
            (h for h in hw_index if h["headword"].lower().startswith(jump_hw_lower)),
            None,
        )
        if match:
            st.session_state.selected_para_id = match["para_id"]

    # ── Build results table ───────────────────────────────────────────────────
    rows = []
    for e in filtered:
        comp = e["comparison"]
        strata_active = [
            fmt_stratum(s)
            for s in STRATA
            if comp.get("strata", {}).get(s, False)
        ]
        flags_active = [
            fmt_misclass(f)
            for f in MISCLASS_FLAGS
            if comp.get("misclassification_flags", {}).get(f, False)
        ]
        rows.append({
            "para_id":  e["para_id"],
            "headword": e["rb"].get("headword") or e["llm"].get("headword") or "",
            "agree_pct": round(comp.get("agree_pct", 0) or 0, 1),
            "rb_cov":   round(comp.get("rb_coverage_pct", 0) or 0, 1),
            "llm_cov":  round(comp.get("llm_coverage_pct", 0) or 0, 1),
            "strata":   ", ".join(strata_active),
            "flags":    ", ".join(flags_active),
        })

    df = pd.DataFrame(rows)
    st.markdown(f"**{len(df)} entries** match the current filters.")

    if df.empty:
        st.warning("No entries match the current filters.")
        st.stop()

    sel_state = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Resolve selected entry
    selected_entry = None
    if sel_state.selection and sel_state.selection.get("rows"):
        row_idx = sel_state.selection["rows"][0]
        para_id = df.iloc[row_idx]["para_id"]
        st.session_state.selected_para_id = para_id
        selected_entry = entry_map.get(para_id)
    elif st.session_state.selected_para_id:
        selected_entry = entry_map.get(st.session_state.selected_para_id)

    # ── Entry detail ──────────────────────────────────────────────────────────
    if selected_entry:
        e = selected_entry
        rb   = e["rb"]
        llm  = e["llm"]
        comp = e["comparison"]
        hw   = rb.get("headword") or llm.get("headword") or ""

        st.markdown(f"### Entry: **{hw}** (para_id {e['para_id']})")
        st.caption(
            f"Agreement: {comp.get('agree_pct', 0):.1f}%  |  "
            f"RB coverage: {comp.get('rb_coverage_pct', 0):.1f}%  |  "
            f"LLM coverage: {comp.get('llm_coverage_pct', 0):.1f}%"
        )

        with st.expander("Source text", expanded=True):
            st.markdown(
                f'<div style="background:#f8f9fa;padding:12px;border-radius:6px;'
                f'font-family:monospace;white-space:pre-wrap;">{e["full_text"]}</div>',
                unsafe_allow_html=True,
            )

        ea_map  = comp.get("exact_agreement", {})
        ts_map  = comp.get("token_similarity", {})

        for field in CONTENT_FIELDS:
            rb_val  = rb.get(field)
            llm_val = llm.get(field)
            if rb_val is None and llm_val is None:
                continue

            is_match = ea_map.get(field, False)
            ts = ts_map.get(field, {})
            ord_sim = ts.get("ordered", None)
            unord_sim = ts.get("unordered", None)

            col_name, col_rb, col_llm = st.columns([2, 4, 4])
            with col_name:
                st.markdown(f"**{field.replace('_', ' ')}**")
                if ord_sim is not None:
                    st.caption(f"seq: {ord_sim:.2f}  f1: {unord_sim:.2f}")
            with col_rb:
                bg = "#d4edda" if is_match else "#f8d7da"
                st.markdown(
                    f'<div style="background:{bg};padding:8px;border-radius:4px;">'
                    f'<small><b>Rule-based</b></small><br>{display_value(rb_val)}</div>',
                    unsafe_allow_html=True,
                )
            with col_llm:
                bg = "#d4edda" if is_match else "#f8d7da"
                st.markdown(
                    f'<div style="background:{bg};padding:8px;border-radius:4px;">'
                    f'<small><b>LLM</b></small><br>{display_value(llm_val)}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Field Deep-Dive
# ─────────────────────────────────────────────────────────────────────────────

elif page == "deepdive":
    st.title("Field Deep-Dive")

    field = st.selectbox(
        "Choose field",
        CONTENT_FIELDS,
        format_func=lambda f: f.replace("_", " "),
        key="p3_field",
    )

    ea_pct = aggregate["exact_agreement_pct"].get(field, 0)
    ts_agg = aggregate["token_similarity"].get(field, {})
    fp     = aggregate["field_presence"].get(field, {})

    # Metric cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Exact Agreement %", f"{ea_pct:.1f}%")
    c2.metric("Mean Ordered Similarity", f"{ts_agg.get('ordered', 0):.3f}")
    c3.metric("Mean Unordered Similarity", f"{ts_agg.get('unordered', 0):.3f}")

    # Field presence
    st.markdown("#### Field Presence")
    p_cols = st.columns(5)
    p_cols[0].metric("RB present %", f"{fp.get('rb_pct', 0):.1f}%")
    p_cols[1].metric("LLM present %", f"{fp.get('llm_pct', 0):.1f}%")
    p_cols[2].metric("Both present", fp.get("both", 0))
    p_cols[3].metric("RB only", fp.get("rb_only", 0))
    p_cols[4].metric("LLM only", fp.get("llm_only", 0))

    st.markdown("---")
    st.subheader(f"Disagreements on '{field.replace('_', ' ')}'")

    # Build disagreements table
    disagree_rows = []
    for e in entries:
        comp = e["comparison"]
        ea   = comp.get("exact_agreement", {})
        if ea.get(field, True):  # skip agreements
            continue
        ts   = comp.get("token_similarity", {}).get(field, {})
        rb_v = display_value(e["rb"].get(field))
        llm_v = display_value(e["llm"].get(field))
        disagree_rows.append({
            "para_id":     e["para_id"],
            "headword":    e["rb"].get("headword") or e["llm"].get("headword") or "",
            "rb_value":    rb_v[:100] if rb_v else "—",
            "llm_value":   llm_v[:100] if llm_v else "—",
            "ordered_sim": round(ts.get("ordered", 0) or 0, 3),
            "unordered_sim": round(ts.get("unordered", 0) or 0, 3),
            "_rb_full":    rb_v,
            "_llm_full":   llm_v,
            "_full_text":  e["full_text"],
        })

    disagree_rows.sort(key=lambda r: r["ordered_sim"])

    df3 = pd.DataFrame(disagree_rows)
    st.markdown(f"**{len(df3)} disagreements**")

    display_cols = ["para_id", "headword", "rb_value", "llm_value", "ordered_sim", "unordered_sim"]
    sel3 = st.dataframe(
        df3[display_cols],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Download
    csv3 = df3[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        csv3,
        file_name=f"disagreements_{field}.csv",
        mime="text/csv",
    )

    # Row detail
    if sel3.selection and sel3.selection.get("rows"):
        idx3 = sel3.selection["rows"][0]
        row3 = disagree_rows[idx3]
        with st.expander("Full entry detail", expanded=True):
            st.markdown(
                f'<div style="background:#f8f9fa;padding:12px;border-radius:6px;'
                f'font-family:monospace;white-space:pre-wrap;">{row3["_full_text"]}</div>',
                unsafe_allow_html=True,
            )
            col_rb3, col_llm3 = st.columns(2)
            with col_rb3:
                st.markdown("**Rule-based value:**")
                st.write(row3["_rb_full"])
            with col_llm3:
                st.markdown("**LLM value:**")
                st.write(row3["_llm_full"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Misclassification Inspector
# ─────────────────────────────────────────────────────────────────────────────

elif page == "misclass":
    st.title("Misclassification Inspector")

    flag_display = {f: fmt_misclass(f) for f in MISCLASS_FLAGS}
    sel_flag = st.selectbox(
        "Misclassification type",
        MISCLASS_FLAGS,
        format_func=lambda f: flag_display[f],
        key="p4_flag",
    )

    st.info(MISCLASS_DESCRIPTIONS[sel_flag])

    mc_count = aggregate["misclassification_counts"].get(sel_flag, 0)
    st.metric("Flagged entries", mc_count)

    # Determine which field to show
    field_map = {
        "misclass_etym_in_definition":          ("definition", "etymology"),
        "misclass_examples_in_definition":       ("definition", "examples"),
        "misclass_chained_etym_collapsed":       ("etymology", "etymology"),
        "misclass_examples_dropped":             ("examples", "examples"),
        "misclass_numbered_senses_not_split":    ("definition", "definition"),
        "misclass_cross_ref_missed":             ("cross_references", "cross_references"),
    }
    rb_field, llm_field = field_map.get(sel_flag, ("definition", "definition"))

    flagged_rows = []
    for e in entries:
        comp = e["comparison"]
        mf   = comp.get("misclassification_flags", {})
        if not mf.get(sel_flag, False):
            continue
        rb_v  = display_value(e["rb"].get(rb_field))
        llm_v = display_value(e["llm"].get(llm_field))
        flagged_rows.append({
            "para_id":   e["para_id"],
            "headword":  e["rb"].get("headword") or e["llm"].get("headword") or "",
            "full_text": (e["full_text"] or "")[:120],
            f"rb_{rb_field}":  rb_v[:120] if rb_v else "—",
            f"llm_{llm_field}": llm_v[:120] if llm_v else "—",
            "_rb_full":  rb_v,
            "_llm_full": llm_v,
            "_full_text": e["full_text"],
        })

    display_cols4 = ["para_id", "headword", "full_text", f"rb_{rb_field}", f"llm_{llm_field}"]
    df4 = pd.DataFrame(flagged_rows)
    if df4.empty:
        st.warning("No entries flagged for this type.")
    else:
        sel4 = st.dataframe(
            df4[display_cols4],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        csv4 = df4[display_cols4].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download as CSV",
            csv4,
            file_name=f"misclass_{sel_flag}.csv",
            mime="text/csv",
        )

        if sel4.selection and sel4.selection.get("rows"):
            idx4 = sel4.selection["rows"][0]
            row4 = flagged_rows[idx4]
            with st.expander("Full entry detail", expanded=True):
                st.markdown(
                    f'<div style="background:#f8f9fa;padding:12px;border-radius:6px;'
                    f'font-family:monospace;white-space:pre-wrap;">{row4["_full_text"]}</div>',
                    unsafe_allow_html=True,
                )
                col_rb4, col_llm4 = st.columns(2)
                with col_rb4:
                    st.markdown(f"**Rule-based `{rb_field}`:**")
                    st.write(row4["_rb_full"])
                with col_llm4:
                    st.markdown(f"**LLM `{llm_field}`:**")
                    st.write(row4["_llm_full"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Source Annotation
# ─────────────────────────────────────────────────────────────────────────────

elif page == "annotation":

    st.sidebar.markdown("### Source Annotation Filters")

    jump_hw5 = st.sidebar.text_input("Jump to headword", key="p5_jump")

    sort_opt5 = st.sidebar.selectbox(
        "Sort / filter by",
        [
            "Worst LLM coverage (ascending)",
            "Biggest RB-LLM coverage gap (descending)",
            "Most uncovered tokens (descending)",
            "Worst agree_pct (ascending)",
            "para_id (ascending)",
        ],
        key="p5_sort",
    )

    max_llm_cov5 = st.sidebar.slider("Max LLM coverage %", 0, 100, 100, key="p5_cov")
    min_miss5    = st.sidebar.slider("Min uncovered token count", 0, 50, 0, key="p5_miss")

    # Pre-filter by coverage
    prefiltered5 = [
        e for e in entries
        if (e["comparison"].get("llm_coverage_pct") or 100) <= max_llm_cov5
    ]

    # For miss count filter we need to annotate — do it only if min_miss5 > 0
    if min_miss5 > 0:
        filtered5 = []
        for e in prefiltered5:
            ann = annotate_tokens(e["full_text"], e["rb"], e["llm"])
            miss_count = sum(1 for a in ann if a["colour"] == "miss")
            if miss_count >= min_miss5:
                filtered5.append(e)
    else:
        filtered5 = prefiltered5

    # Sort
    if sort_opt5.startswith("Worst LLM"):
        filtered5 = sorted(filtered5, key=lambda e: e["comparison"].get("llm_coverage_pct") or 100)
    elif sort_opt5.startswith("Biggest RB-LLM"):
        filtered5 = sorted(
            filtered5,
            key=lambda e: (e["comparison"].get("rb_coverage_pct") or 0) - (e["comparison"].get("llm_coverage_pct") or 0),
            reverse=True,
        )
    elif sort_opt5.startswith("Most uncovered"):
        annotated_cache = {}
        for e in filtered5:
            ann = annotate_tokens(e["full_text"], e["rb"], e["llm"])
            annotated_cache[e["para_id"]] = ann
        filtered5 = sorted(
            filtered5,
            key=lambda e: sum(1 for a in annotated_cache.get(e["para_id"], []) if a["colour"] == "miss"),
            reverse=True,
        )
    elif sort_opt5.startswith("Worst agree"):
        filtered5 = sorted(filtered5, key=lambda e: e["comparison"].get("agree_pct") or 0)
    else:
        filtered5 = sorted(filtered5, key=lambda e: e["para_id"])

    total5 = len(filtered5)
    if total5 == 0:
        st.warning("No entries match the current filters.")
        st.stop()

    # Handle jump-to
    if jump_hw5:
        jump_lower5 = jump_hw5.lower()
        match5 = next(
            (h for h in hw_index if h["headword"].lower().startswith(jump_lower5)),
            None,
        )
        if match5:
            pid = match5["para_id"]
            for i, e in enumerate(filtered5):
                if e["para_id"] == pid:
                    st.session_state.p5_index = i
                    break

    # Clamp index
    idx5 = max(0, min(st.session_state.p5_index, total5 - 1))

    # ── Navigation buttons ────────────────────────────────────────────────────
    nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 8])
    with nav_c1:
        if st.button("← Prev", key="p5_prev") and idx5 > 0:
            st.session_state.p5_index = idx5 - 1
            st.rerun()
    with nav_c2:
        if st.button("Next →", key="p5_next") and idx5 < total5 - 1:
            st.session_state.p5_index = idx5 + 1
            st.rerun()

    idx5 = max(0, min(st.session_state.p5_index, total5 - 1))
    entry5 = filtered5[idx5]
    st.session_state.selected_para_id = entry5["para_id"]

    rb5   = entry5["rb"]
    llm5  = entry5["llm"]
    comp5 = entry5["comparison"]
    hw5   = rb5.get("headword") or llm5.get("headword") or ""

    # ── Header card ───────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:#1a1a2e;color:white;padding:12px 20px;
            border-radius:8px;margin-bottom:16px;
            display:flex;align-items:center;justify-content:space-between;">
  <div>
    <span style="font-size:1.4rem;font-weight:700;">{hw5}</span>
    <span style="font-size:0.8rem;opacity:0.6;margin-left:12px;">
      para_id {entry5['para_id']}
    </span>
  </div>
  <div style="font-size:0.8rem;opacity:0.7;">
    Entry {idx5 + 1} of {total5}
  </div>
</div>
""", unsafe_allow_html=True)

    # Build per-system token sets for field rendering
    rb_all_tokens: set = set()
    llm_all_tokens: set = set()
    for f in CONTENT_FIELDS:
        rb_all_tokens  |= get_tokens(rb5.get(f))
        llm_all_tokens |= get_tokens(llm5.get(f))

    # Annotate source text
    ann5 = annotate_tokens(entry5["full_text"], rb5, llm5)

    # ── Source text card ──────────────────────────────────────────────────────
    source_html = render_tokens_html(ann5)
    st.markdown(f"""
<div style="background:#f8f9fa;border:1px solid #e0e0e0;
            border-radius:8px;padding:16px 20px;margin-bottom:16px;">
  <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;
              text-transform:uppercase;color:#888;margin-bottom:10px;">
    SOURCE TEXT
  </div>
  {LEGEND_HTML}
  <div style="margin-top:10px;">
    {source_html}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Field breakdown — single HTML table ───────────────────────────────────
    st.markdown("""
<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;
            text-transform:uppercase;color:#888;margin:20px 0 10px;">
  FIELD BREAKDOWN
</div>
""", unsafe_allow_html=True)

    rows_html = ""
    for field in CONTENT_FIELDS:
        rb_val5  = rb5.get(field)
        llm_val5 = llm5.get(field)
        if rb_val5 is None and llm_val5 is None:
            continue

        is_match5 = comp5.get("exact_agreement", {}).get(field, False)
        ts5       = comp5.get("token_similarity", {}).get(field, {})
        ord5      = ts5.get("ordered", 0) or 0
        unord5    = ts5.get("unordered", 0) or 0

        rb_html5  = render_field_tokens_html(rb_val5,  rb_all_tokens, llm_all_tokens)
        llm_html5 = render_field_tokens_html(llm_val5, rb_all_tokens, llm_all_tokens)

        match_bg    = "#d4edda" if is_match5 else "#f8d7da"
        match_color = "#155724" if is_match5 else "#721c24"
        match_label = "✓" if is_match5 else "✗"
        cell_bg     = "#f8fff8" if is_match5 else "#fff"

        rows_html += f"""
<tr style="border-bottom:1px solid #f0f0f0;">
  <td style="width:120px;padding:8px 12px;vertical-align:top;
             border-right:1px solid #f0f0f0;">
    <div style="font-size:0.75rem;font-weight:600;color:#444;
                text-transform:uppercase;letter-spacing:0.04em;">
      {field.replace('_', ' ')}
    </div>
    <div style="margin-top:4px;">
      <span style="background:{match_bg};color:{match_color};
                   padding:1px 6px;border-radius:3px;font-size:0.7rem;">
        {match_label}
      </span>
    </div>
    <div style="font-size:0.65rem;color:#999;margin-top:3px;">
      seq {ord5:.2f} · f1 {unord5:.2f}
    </div>
  </td>
  <td style="width:44%;padding:8px 12px;vertical-align:top;
             border-right:1px solid #f0f0f0;background:{cell_bg};">
    <div style="font-size:0.65rem;font-weight:600;color:#1f77b4;
                text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">
      Rule-based
    </div>
    <div style="line-height:1.9;font-size:0.875rem;">
      {rb_html5}
    </div>
  </td>
  <td style="width:44%;padding:8px 12px;vertical-align:top;background:{cell_bg};">
    <div style="font-size:0.65rem;font-weight:600;color:#ff7f0e;
                text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">
      LLM
    </div>
    <div style="line-height:1.9;font-size:0.875rem;">
      {llm_html5}
    </div>
  </td>
</tr>"""

    st.markdown(f"""
<div style="border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">
  <table style="width:100%;border-collapse:collapse;">
    {rows_html}
  </table>
</div>
""", unsafe_allow_html=True)

    # ── Coverage summary card ─────────────────────────────────────────────────
    rb_cov5  = comp5.get("rb_coverage_pct", 0) or 0
    llm_cov5 = comp5.get("llm_coverage_pct", 0) or 0
    gap5     = rb_cov5 - llm_cov5

    miss_tokens = [a["token"] for a in ann5 if a["colour"] == "miss"]
    miss_html = " ".join(
        f'<span style="background:#f8d7da;color:#721c24;padding:1px 5px;'
        f'border-radius:3px;font-size:0.8rem;">{t}</span>'
        for t in miss_tokens
    ) if miss_tokens else '<span style="color:#155724;">None ✓</span>'

    gap_color = "#d62728" if gap5 > 5 else "#2ca02c"

    st.markdown(f"""
<div style="background:#f8f9fa;border:1px solid #e0e0e0;
            border-radius:8px;padding:14px 20px;margin-top:16px;">
  <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;
              text-transform:uppercase;color:#888;margin-bottom:10px;">
    COVERAGE SUMMARY
  </div>
  <div style="display:flex;gap:24px;margin-bottom:10px;">
    <div>
      <div style="font-size:0.7rem;color:#666;">Rule-based</div>
      <div style="font-size:1.3rem;font-weight:700;color:#1f77b4;">
        {rb_cov5:.1f}%
      </div>
    </div>
    <div>
      <div style="font-size:0.7rem;color:#666;">LLM</div>
      <div style="font-size:1.3rem;font-weight:700;color:#ff7f0e;">
        {llm_cov5:.1f}%
      </div>
    </div>
    <div>
      <div style="font-size:0.7rem;color:#666;">Gap</div>
      <div style="font-size:1.3rem;font-weight:700;color:{gap_color};">
        {gap5:.1f}%
      </div>
    </div>
  </div>
  <div style="font-size:0.75rem;color:#666;margin-top:6px;">
    <strong>Uncovered tokens:</strong> {miss_html}
  </div>
</div>
""", unsafe_allow_html=True)


elif page == "semantic":

    if embed_data is None:
        st.title("Semantic Space")
        st.info("""
**embeddings.json not found.**

Run `embed.py` once locally to generate pre-computed UMAP and PCA coordinates:

```
pip install sentence-transformers umap-learn scipy scikit-learn
python embed.py
```

Place the output `data/embeddings.json` in the `data/` folder, then reload the app.
        """)
        st.stop()

    emb_entries = embed_data["entries"]
    emb_meta    = embed_data.get("meta", {})

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.markdown("### Display")

    # Detect whether this file has PCA coords (new format)
    _has_pca = bool(embed_data["entries"] and "rb_pca" in embed_data["entries"][0])

    projection = st.sidebar.radio(
        "Projection",
        ["UMAP", "PCA"] if _has_pca else ["UMAP"],
        horizontal=True,
        key="sem_proj",
    )

    show_llm   = st.sidebar.toggle("Show LLM points",  value=True,  key="sem_show_llm")
    show_drift = st.sidebar.toggle("Show drift lines", value=False, key="sem_drift",
                                   disabled=not show_llm)

    colour_by = st.sidebar.selectbox(
        "Colour points by",
        ["agree_pct", "drift", "stratum", "entry_type"],
        key="sem_colour",
    )

    st.sidebar.markdown("### Filter")

    all_strata_sem = sorted({s for e in emb_entries for s in e.get("strata", [])})
    sel_strata_sem = st.sidebar.multiselect("Strata", all_strata_sem, key="sem_strata")

    min_agree_sem  = st.sidebar.slider("Min agree_pct", 0, 100, 0, key="sem_agree")
    drift_pct_range = st.sidebar.slider("Drift percentile range", 0, 100, (0, 100),
                                        key="sem_drift_range")
    search_sem = st.sidebar.text_input("Highlight headword", key="sem_search")

    if show_drift:
        st.sidebar.markdown("### Drift line colour")
        drift_line_colour = st.sidebar.selectbox(
            "Colour lines by",
            ["drift magnitude", "agree_pct", "uniform grey"],
            key="sem_line_colour",
        )
    else:
        drift_line_colour = "drift magnitude"

    # ── Filter ────────────────────────────────────────────────────────────────
    filtered_sem = emb_entries

    if sel_strata_sem:
        filtered_sem = [
            e for e in filtered_sem
            if any(s in e.get("strata", []) for s in sel_strata_sem)
        ]

    filtered_sem = [
        e for e in filtered_sem
        if e.get("agree_pct", 100) >= min_agree_sem
    ]

    # Use projection-specific drift percentile for the slider filter
    _drift_pct_key = "drift_pca_pct" if projection == "PCA" else "drift_pct"

    dlo = drift_pct_range[0] / 100
    dhi = drift_pct_range[1] / 100
    filtered_sem = [
        e for e in filtered_sem
        if dlo <= e.get(_drift_pct_key, 0) <= dhi
    ]

    if not filtered_sem:
        st.warning("No entries match the current filters.")
        st.stop()

    # ── Projection-specific coordinate and drift keys ─────────────────────────
    # New format: rb_umap / rb_pca.  Old format (UMAP only): rb_xyz fallback.
    if projection == "PCA" and _has_pca:
        _rb_key    = "rb_pca"
        _llm_key   = "llm_pca"
        _drift_key = "drift_pca"
    else:
        # Prefer explicit rb_umap; fall back to rb_xyz for old embeddings.json
        _first = filtered_sem[0]
        _rb_key    = "rb_umap"  if "rb_umap"  in _first else "rb_xyz"
        _llm_key   = "llm_umap" if "llm_umap" in _first else "llm_xyz"
        _drift_key = "drift"

    # ── Colour scale logic ────────────────────────────────────────────────────
    def get_point_colours(ents, cb):
        if cb == "agree_pct":
            vals = [e.get("agree_pct", 100) for e in ents]
            return vals, "RdYlGn", (0, 100), "agree %"
        elif cb == "drift":
            vals = [e.get(_drift_key, 0) for e in ents]
            mn, mx = min(vals), max(vals)
            return vals, "RdYlGn_r", (mn, mx), "drift"
        elif cb == "entry_type":
            type_map = {"full": 1, "cross_ref": 0}
            vals = [type_map.get(e.get("entry_type", "full"), 1) for e in ents]
            return vals, "RdBu", (0, 1), "type"
        else:  # stratum count
            vals = [len(e.get("strata", [])) for e in ents]
            mx = max(vals) if vals else 1
            return vals, "Viridis", (0, mx), "strata count"

    # ── Build arrays ──────────────────────────────────────────────────────────
    rb_x  = [e[_rb_key][0]  for e in filtered_sem]
    rb_y  = [e[_rb_key][1]  for e in filtered_sem]
    rb_z  = [e[_rb_key][2]  for e in filtered_sem]
    llm_x = [e[_llm_key][0] for e in filtered_sem]
    llm_y = [e[_llm_key][1] for e in filtered_sem]
    llm_z = [e[_llm_key][2] for e in filtered_sem]

    headwords = [e.get("headword", "")       for e in filtered_sem]
    para_ids  = [e.get("para_id", 0)         for e in filtered_sem]
    agrees    = [e.get("agree_pct", 100)     for e in filtered_sem]
    drifts    = [e.get(_drift_key, 0)        for e in filtered_sem]
    rb_defs   = [e.get("rb_def",  "")[:80]  for e in filtered_sem]
    llm_defs  = [e.get("llm_def", "")[:80]  for e in filtered_sem]

    col_vals, col_scale, col_range, col_label = get_point_colours(filtered_sem, colour_by)

    highlight_mask = [
        search_sem.lower() in hw.lower() if search_sem else False
        for hw in headwords
    ]
    sizes_rb  = [10 if h else 4 for h in highlight_mask]
    sizes_llm = [10 if h else 4 for h in highlight_mask]

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = go.Figure()

    # Trace 1 — Rule-based points
    fig.add_trace(go.Scatter3d(
        x=rb_x, y=rb_y, z=rb_z,
        mode="markers",
        name="Rule-based",
        marker=dict(
            size=sizes_rb,
            color=col_vals,
            colorscale=col_scale,
            cmin=col_range[0],
            cmax=col_range[1],
            colorbar=dict(title=col_label, thickness=12),
            opacity=0.85,
            symbol="circle",
        ),
        customdata=list(zip(para_ids, headwords, rb_defs, agrees, drifts)),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "para_id: %{customdata[0]}<br>"
            "agree: %{customdata[3]:.1f}%<br>"
            "drift: %{customdata[4]:.4f}<br>"
            "def: %{customdata[2]}<extra>Rule-based</extra>"
        ),
    ))

    # Trace 2 — LLM points
    if show_llm:
        fig.add_trace(go.Scatter3d(
            x=llm_x, y=llm_y, z=llm_z,
            mode="markers",
            name="LLM",
            marker=dict(
                size=sizes_llm,
                color=col_vals,
                colorscale=col_scale,
                cmin=col_range[0],
                cmax=col_range[1],
                opacity=0.65,
                symbol="diamond",
                showscale=False,
            ),
            customdata=list(zip(para_ids, headwords, llm_defs, agrees, drifts)),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "para_id: %{customdata[0]}<br>"
                "agree: %{customdata[3]:.1f}%<br>"
                "drift: %{customdata[4]:.4f}<br>"
                "def: %{customdata[2]}<extra>LLM</extra>"
            ),
        ))

    # Trace 3 — Drift lines
    if show_drift and show_llm:
        lx, ly, lz = [], [], []
        for i in range(len(filtered_sem)):
            lx += [rb_x[i], llm_x[i], None]
            ly += [rb_y[i], llm_y[i], None]
            lz += [rb_z[i], llm_z[i], None]

        mean_d = sum(drifts) / len(drifts) if drifts else 0
        max_d  = max(drifts) if drifts else 1
        r = int(255 * mean_d / max_d)
        g = int(255 * (1 - mean_d / max_d))
        line_col = f"rgba({r},{g},0,0.35)"

        if drift_line_colour == "agree_pct":
            mean_a = sum(agrees) / len(agrees) if agrees else 100
            r2 = int(255 * (1 - mean_a / 100))
            g2 = int(255 * mean_a / 100)
            line_col = f"rgba({r2},{g2},0,0.35)"
        elif drift_line_colour == "uniform grey":
            line_col = "rgba(128,128,128,0.3)"

        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode="lines",
            name="Drift",
            line=dict(color=line_col, width=1),
            hoverinfo="skip",
        ))

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
            bgcolor="rgba(248,249,250,1)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=dict(
            text=(
                f"Semantic space [{projection}] — {len(filtered_sem):,} entries"
                + (f" · {emb_meta.get('model', '')}" if emb_meta.get("model") else "")
            ),
            font=dict(size=13),
            x=0,
        ),
    )

    # PCA variance annotation (shown only in PCA mode)
    if projection == "PCA" and "pca_params" in emb_meta:
        pca_p = emb_meta["pca_params"]
        evr   = pca_p.get("explained_variance_ratio", [])
        if evr:
            st.caption(
                f"PCA explained variance — "
                f"PC1: {evr[0]*100:.1f}%  "
                f"PC2: {evr[1]*100:.1f}%  "
                f"PC3: {evr[2]*100:.1f}%  "
                f"(total {pca_p.get('total_explained_variance', 0)*100:.1f}%)"
            )

    st.plotly_chart(fig, use_container_width=True)

    # ── Stats row ─────────────────────────────────────────────────────────────
    drift_label = f"Mean drift ({projection})"
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Entries shown", f"{len(filtered_sem):,}")
    sc2.metric(drift_label,
               f"{sum(drifts)/len(drifts):.4f}" if drifts else "—")
    sc3.metric(f"Max drift ({projection})",
               f"{max(drifts):.4f}" if drifts else "—")
    sc4.metric("Model", emb_meta.get("model", "—"))

    # ── Entry inspector ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Enter a para_id to inspect an entry:**")

    sel_pid_sem = st.number_input(
        "para_id", min_value=1, max_value=9999, value=1, step=1, key="sem_pid",
    )

    if sel_pid_sem and sel_pid_sem in entry_map:
        sel_entry = entry_map[sel_pid_sem]
        sel_emb   = next(
            (e for e in emb_entries if e["para_id"] == sel_pid_sem), None
        )
        hw_sel = sel_entry["rb"].get("headword") or ""

        st.markdown(f"""
<div style="background:#1a1a2e;color:white;padding:12px 20px;
            border-radius:8px;margin-bottom:12px;">
  <span style="font-size:1.2rem;font-weight:700;">{hw_sel}</span>
  <span style="opacity:0.5;font-size:0.8rem;margin-left:12px;">
    para_id {sel_pid_sem}
  </span>
</div>
""", unsafe_allow_html=True)

        col_src, col_meta = st.columns([2, 1])

        with col_src:
            st.markdown("**Source text**")
            st.markdown(
                f'<div style="background:#f8f9fa;border:1px solid #e0e0e0;'
                f'border-radius:6px;padding:12px;font-size:0.875rem;">'
                f'{sel_entry["full_text"]}</div>',
                unsafe_allow_html=True,
            )

        with col_meta:
            st.markdown("**Semantic metrics**")
            if sel_emb:
                st.markdown(f"**Drift:** `{sel_emb.get('drift', 0):.4f}`")
                st.markdown(
                    f"**Drift percentile:** "
                    f"`{sel_emb.get('drift_pct', 0) * 100:.1f}th`"
                )
                st.markdown("**RB definition:**")
                st.info(sel_emb.get("rb_def", "—") or "—")
                st.markdown("**LLM definition:**")
                st.warning(sel_emb.get("llm_def", "—") or "—")

            st.markdown(
                f"**Agree:** `{sel_entry['comparison']['agree_pct']:.1f}%`"
            )
            active_strata = [
                fmt_stratum(k)
                for k, v in sel_entry["comparison"]["strata"].items()
                if v
            ]
            if active_strata:
                st.markdown("**Strata:** " + ", ".join(active_strata))
