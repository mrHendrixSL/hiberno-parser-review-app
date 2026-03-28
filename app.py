import html
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# =========================================================
# Config
# =========================================================
DEFAULT_JSONL_PATH = Path("data/normalized_withhallucinationFlags.jsonl")
if not DEFAULT_JSONL_PATH.exists():
    DEFAULT_JSONL_PATH = Path(
        r"D:\Experiments\Hiberno_English_Dictonary\final_merged_normalized_withhallucinationFlags.jsonl"
    )

st.set_page_config(
    page_title="Parser Output Qualitative Review",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# Constants
# =========================================================
MATCH_FIELDS = [
    "source_text_match",
    "pos_match_norm",
    "region_mentions_match_norm",
]

FILTER_FIELDS = [
    "_merge",
    "source_text_match",
    "pos_match_norm",
    "region_mentions_match_norm",
    "rule_possible_hallucination",
    "genai_possible_hallucination",
    "examples_presence_status",
    "etymology_presence_status",
    "cross_references_presence_status",
    "region_mentions_presence_status",
]

PRESENCE_STATUS_FIELDS = [
    "headword_presence_status",
    "pronunciations_presence_status",
    "part_of_speech_presence_status",
    "definition_presence_status",
    "examples_presence_status",
    "etymology_presence_status",
    "cross_references_presence_status",
    "region_mentions_presence_status",
]

DIAGNOSTIC_FIELDS = [
    "source_char_len",
    "rule_extracted_char_len",
    "genai_extracted_char_len",
    "rule_char_ratio",
    "genai_char_ratio",
]

COMPARISON_ROWS = [
    ("headword_raw", "headword_raw_rule", "headword_raw_genai"),
    ("headword", "headword_rule", "headword_genai"),
    ("variant_forms_raw", "variant_forms_raw_rule", "variant_forms_raw_genai"),
    ("variant_forms", "variant_forms_rule", "variant_forms_genai"),
    ("pronunciations", "pronunciations_rule", "pronunciations_genai"),
    ("part_of_speech", "part_of_speech_rule", "part_of_speech_genai"),
    ("part_of_speech_norm", "part_of_speech_rule_norm", "part_of_speech_genai_norm"),
    ("definition", "definition_rule", "definition_genai"),
    ("examples", "examples_rule", "examples_genai"),
    ("etymology", "etymology_rule", "etymology_genai"),
    ("cross_references", "cross_references_rule", "cross_references_genai"),
    ("region_mentions", "region_mentions_rule", "region_mentions_genai"),
    ("region_mentions_norm", "region_mentions_rule_norm", "region_mentions_genai_norm"),
]

GENAI_HIGHLIGHT_FIELDS = {
    "headword_raw_genai",
    "headword_genai",
    "variant_forms_raw_genai",
    "variant_forms_genai",
    "pronunciations_genai",
    "part_of_speech_genai",
    "part_of_speech_genai_norm",
    "definition_genai",
    "examples_genai",
    "etymology_genai",
    "cross_references_genai",
    "region_mentions_genai",
    "region_mentions_genai_norm",
}

# =========================================================
# Small UI helpers
# =========================================================
def render_badge(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        txt, bg = "missing", "#6b7280"
    else:
        txt = str(value)
        lower = txt.strip().lower()
        if lower in {"true", "match", "matched", "both_present", "present", "ok", "yes"}:
            bg = "#15803d"
        elif lower in {"partial", "one_sided", "one-sided", "rule_only", "genai_only", "left_only", "right_only"}:
            bg = "#ca8a04"
        elif lower in {"false", "mismatch", "different", "hallucination", "flagged", "error", "conflict"}:
            bg = "#b91c1c"
        elif lower in {"both_missing", "missing", "none", "null", ""}:
            bg = "#6b7280"
            txt = "missing" if lower == "" else txt
        else:
            bg = "#374151"

    return (
        f"<span style='display:inline-block;padding:0.15rem 0.45rem;border-radius:999px;"
        f"background:{bg};color:white;font-size:0.75rem;font-weight:600;white-space:nowrap;'>"
        f"{html.escape(str(txt))}</span>"
    )


def panel_start(title: str) -> str:
    return f"""
    <div style="
        border:1px solid #e5e7eb;
        border-radius:10px;
        padding:0.7rem 0.8rem;
        background:white;
        height:100%;
    ">
        <div style="font-size:0.82rem;font-weight:700;color:#111827;margin-bottom:0.45rem;">
            {html.escape(title)}
        </div>
    """


def panel_end() -> str:
    return "</div>"


# =========================================================
# State
# =========================================================
def init_state() -> None:
    defaults = {
        "current_pos": 0,
        "jump_entry_id_input": "",
        "search_text_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================================================
# Data loading
# =========================================================
@st.cache_data(show_spinner=True)
def load_jsonl(path_str: str) -> pd.DataFrame:
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("The JSONL file was loaded, but it contains no rows.")

    return df


# =========================================================
# Generic helpers
# =========================================================
def value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return False


def pretty_value(value: Any) -> str:
    if value_is_missing(value):
        return "∅"

    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, indent=2, ensure_ascii=False)
        except Exception:
            return str(value)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (
            (stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))
        ):
            try:
                parsed = json.loads(stripped)
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            except Exception:
                return value
        return value

    return str(value)


def normalize_scalar_for_search(value: Any) -> str:
    if value_is_missing(value):
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False).lower()
        except Exception:
            return str(value).lower()
    return str(value).lower()


def value_equal(a: Any, b: Any) -> bool:
    if value_is_missing(a) and value_is_missing(b):
        return True
    return pretty_value(a) == pretty_value(b)


def is_false_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    if value_is_missing(value):
        return False
    v = str(value).strip().lower()
    return v in {"false", "0", "no", "mismatch", "different", "not_match", "not matched"}


# =========================================================
# Search / filtering
# =========================================================
def search_mask(df: pd.DataFrame, query: str) -> pd.Series:
    if not query.strip():
        return pd.Series(True, index=df.index)

    query_l = query.lower().strip()
    search_cols = [
        "source_text_rule",
        "source_text_genai",
        "headword_rule",
        "headword_genai",
    ]
    existing_cols = [c for c in search_cols if c in df.columns]
    if not existing_cols:
        return pd.Series(True, index=df.index)

    combined = pd.Series("", index=df.index, dtype="object")
    for col in existing_cols:
        combined = combined + " " + df[col].apply(normalize_scalar_for_search)

    return combined.str.contains(query_l, na=False, regex=False)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    with st.sidebar:
        st.markdown("### Filters")

        for field in FILTER_FIELDS:
            if field not in filtered.columns:
                continue

            values = filtered[field].dropna().astype(str).unique().tolist()
            values = sorted(values)
            selected = st.multiselect(
                field,
                options=values,
                default=[],
                key=f"filter_{field}",
            )
            if selected:
                filtered = filtered[filtered[field].astype(str).isin(selected)]

        only_false_matches = st.checkbox(
            "Only rows where any match flag is false",
            value=False,
        )
        if only_false_matches:
            existing = [c for c in MATCH_FIELDS if c in filtered.columns]
            if existing:
                mask = pd.Series(False, index=filtered.index)
                for col in existing:
                    mask = mask | filtered[col].apply(is_false_like)
                filtered = filtered[mask]

    search_text = st.session_state.get("search_text_input", "").strip()
    filtered = filtered[search_mask(filtered, search_text)]

    return filtered


# =========================================================
# Navigation
# =========================================================
def clamp_current_pos(total_rows: int) -> None:
    if total_rows <= 0:
        st.session_state.current_pos = 0
        return
    st.session_state.current_pos = max(0, min(st.session_state.current_pos, total_rows - 1))


def jump_to_entry_id(filtered_df: pd.DataFrame, entry_id_value: str) -> Optional[int]:
    if "entry_id" not in filtered_df.columns:
        return None
    if not entry_id_value.strip():
        return None

    matches = filtered_df.index[
        filtered_df["entry_id"].astype(str).str.strip() == entry_id_value.strip()
    ].tolist()

    if not matches:
        return None

    target_index = matches[0]
    return list(filtered_df.index).index(target_index)


# =========================================================
# Highlighting
# =========================================================
def tokenize_for_alignment(text: str) -> List[str]:
    return re.findall(r"\w+|\W+", text, flags=re.UNICODE)


def normalize_token(token: str) -> str:
    return token.lower()


def build_hallucination_highlight_html(source_text: Any, target_text: Any) -> str:
    if value_is_missing(target_text):
        return "∅"

    source_text = "" if value_is_missing(source_text) else str(source_text)
    target_text = str(target_text)

    src_tokens = tokenize_for_alignment(source_text)
    tgt_tokens = tokenize_for_alignment(target_text)

    src_norm = [normalize_token(t) for t in src_tokens]
    tgt_norm = [normalize_token(t) for t in tgt_tokens]

    matcher = SequenceMatcher(a=src_norm, b=tgt_norm, autojunk=False)

    matched_target_positions = set()
    for block in matcher.get_matching_blocks():
        for idx in range(block.b, block.b + block.size):
            matched_target_positions.add(idx)

    rendered = []
    for i, token in enumerate(tgt_tokens):
        escaped = html.escape(token)
        if i in matched_target_positions:
            rendered.append(escaped)
        else:
            if re.search(r"\w", token, flags=re.UNICODE):
                rendered.append(
                    "<span style='background:#fee2e2;color:#991b1b;padding:0 2px;border-radius:3px;'>"
                    f"{escaped}</span>"
                )
            else:
                rendered.append(escaped)

    return "".join(rendered)


def render_json_like_html(value: Any, source_text: Any, highlight_non_source: bool) -> str:
    if value_is_missing(value):
        return "∅"

    def render(v: Any, indent: int = 0) -> str:
        spacer = "&nbsp;" * 2 * indent

        if value_is_missing(v):
            return "null"

        if isinstance(v, str):
            escaped = (
                build_hallucination_highlight_html(source_text, v)
                if highlight_non_source
                else html.escape(v)
            )
            return f'"{escaped}"'

        if isinstance(v, bool):
            return "true" if v else "false"

        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return html.escape(str(v))

        if isinstance(v, list):
            if not v:
                return "[]"
            parts = ["["]
            for item in v:
                parts.append(f"{spacer}&nbsp;&nbsp;{render(item, indent + 1)}")
            parts.append(f"{spacer}]")
            return "<br>".join(parts)

        if isinstance(v, dict):
            if not v:
                return "{}"
            parts = ["{"]
            for k, item in v.items():
                key_html = html.escape(str(k))
                val_html = render(item, indent + 1)
                parts.append(f"{spacer}&nbsp;&nbsp;\"{key_html}\": {val_html}")
            parts.append(f"{spacer}}}")
            return "<br>".join(parts)

        return html.escape(str(v))

    return render(value)


def render_cell_html(value: Any, source_text: Any, highlight_non_source: bool) -> str:
    if value_is_missing(value):
        return "∅"

    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (
            (stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))
        ):
            try:
                parsed = json.loads(stripped)
                return render_json_like_html(parsed, source_text, highlight_non_source)
            except Exception:
                pass

    if isinstance(value, (dict, list)):
        return render_json_like_html(value, source_text, highlight_non_source)

    if isinstance(value, str):
        if highlight_non_source:
            return build_hallucination_highlight_html(source_text, value)
        return html.escape(value)

    return html.escape(pretty_value(value))


# =========================================================
# Render blocks
# =========================================================
def render_top_strip(row: pd.Series, filtered_position: int, filtered_total: int, raw_index: Any) -> None:
    left, middle, right = st.columns([1.2, 1.4, 1.1], gap="small")

    with left:
        st.markdown("##### Navigation & Search")
        st.button("Previous row", key="prev_row_top", use_container_width=True)
        st.button("Next row", key="next_row_top", use_container_width=True)
        st.text_input("Jump by entry_id", key="jump_entry_id_input", label_visibility="visible")
        st.text_input("Search text", key="search_text_input", label_visibility="visible")
        st.caption(f"{filtered_total} rows in current subset")

    with middle:
        summary_html = (
            panel_start("Row Summary")
            + f"""
            <div style="font-size:0.82rem;line-height:1.55;">
                <div><b>Position</b> {filtered_position + 1} / {filtered_total}</div>
                <div><b>Raw index</b> {raw_index}</div>
                <div><b>entry_id</b> <code>{html.escape(str(row.get("entry_id", "N/A")))}</code></div>
            </div>

            <div style="margin-top:0.55rem;display:flex;flex-wrap:wrap;gap:0.35rem;">
                {render_badge(row.get("_merge"))}
                {render_badge(row.get("source_text_match"))}
                {render_badge(row.get("pos_match_norm"))}
                {render_badge(row.get("region_mentions_match_norm"))}
                {render_badge(row.get("rule_possible_hallucination"))}
                {render_badge(row.get("genai_possible_hallucination"))}
            </div>
            """
            + panel_end()
        )
        st.markdown(summary_html, unsafe_allow_html=True)

    with right:
        diag_html = (
            panel_start("Diagnostics")
            + "<div style='font-size:0.8rem;line-height:1.55;'>"
            + "".join(
                f"<div><b>{html.escape(field)}</b>: {html.escape(str(row.get(field, 'N/A')))}</div>"
                for field in DIAGNOSTIC_FIELDS
                if field in row.index
            )
            + "</div>"
            + panel_end()
        )
        st.markdown(diag_html, unsafe_allow_html=True)


def render_source_text(row: pd.Series) -> None:
    st.markdown("#### Source text")

    rule_text = row.get("source_text_rule")
    genai_text = row.get("source_text_genai")

    if value_equal(rule_text, genai_text):
        st.code(pretty_value(rule_text), language=None)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("source_text_rule")
            st.code(pretty_value(rule_text), language=None)
        with c2:
            st.caption("source_text_genai")
            st.code(pretty_value(genai_text), language=None)


def render_comparison_table(row: pd.Series) -> None:
    st.markdown("#### Comparison")

    source_text = row.get("source_text_rule")
    if value_is_missing(source_text):
        source_text = row.get("source_text_genai")

    rows_html = ""

    for field_label, rule_col, genai_col in COMPARISON_ROWS:
        rule_val = row.get(rule_col)
        genai_val = row.get(genai_col)

        row_diff = not value_equal(rule_val, genai_val)

        rule_html = render_cell_html(rule_val, source_text, highlight_non_source=False)
        genai_html = render_cell_html(
            genai_val,
            source_text,
            highlight_non_source=(genai_col in GENAI_HIGHLIGHT_FIELDS),
        )

        bg = "#fef2f2" if row_diff else "white"

        rows_html += f"""
        <tr>
            <td style="font-weight:600; background:#fafafa; width:15%; border:1px solid #ddd; padding:8px; vertical-align:top;">
                {html.escape(field_label)}
            </td>
            <td style="background:{bg}; padding:8px; vertical-align:top; border:1px solid #ddd;">
                <div style="white-space:pre-wrap; max-height:220px; overflow:auto; font-family:monospace; font-size:0.84rem; line-height:1.42;">
                    {rule_html}
                </div>
            </td>
            <td style="background:{bg}; padding:8px; vertical-align:top; border:1px solid #ddd;">
                <div style="white-space:pre-wrap; max-height:220px; overflow:auto; font-family:monospace; font-size:0.84rem; line-height:1.42;">
                    {genai_html}
                </div>
            </td>
        </tr>
        """

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; margin: 0; padding: 0;">
        <div style="font-size:0.85rem; color:#666; margin-bottom:10px;">
            Red text in the GenAI column = text not grounded in source.
        </div>
        <table style="width:100%; border-collapse:collapse; table-layout:fixed;">
            <thead>
                <tr style="background:#f3f4f6;">
                    <th style="text-align:left; padding:8px; border:1px solid #ddd; width:15%;">Field</th>
                    <th style="text-align:left; padding:8px; border:1px solid #ddd; width:42.5%;">Rule-based</th>
                    <th style="text-align:left; padding:8px; border:1px solid #ddd; width:42.5%;">GenAI</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </body>
    </html>
    """

    components.html(full_html, height=980, scrolling=True)


def render_presence_flags(row: pd.Series) -> None:
    present = [f for f in PRESENCE_STATUS_FIELDS if f in row.index]
    if not present:
        return

    st.markdown("#### Presence flags")
    html_block = (
        "<div style='display:flex;flex-wrap:wrap;gap:0.45rem;'>"
        + "".join(
            f"<div style='display:flex;flex-direction:column;gap:0.2rem;'>"
            f"<div style='font-size:0.74rem;color:#666;'>{html.escape(field)}</div>"
            f"{render_badge(row.get(field))}</div>"
            for field in present
        )
        + "</div>"
    )
    st.markdown(html_block, unsafe_allow_html=True)


# =========================================================
# Main
# =========================================================
def main() -> None:
    init_state()

    st.title("Parser Output Qualitative Review")

    file_path = st.text_input(
        "JSONL path",
        value=str(DEFAULT_JSONL_PATH),
        help="Local path or repo-relative path to the merged JSONL.",
    )

    try:
        df = load_jsonl(file_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    df = df.reset_index(drop=False).rename(columns={"index": "__original_index__"})

    filtered = apply_filters(df)
    filtered_total = len(filtered)

    if filtered_total == 0:
        st.warning("No rows match the current filters/search.")
        st.stop()

    jump_value = st.session_state.get("jump_entry_id_input", "").strip()
    if jump_value:
        target_pos = jump_to_entry_id(filtered, jump_value)
        if target_pos is not None:
            st.session_state.current_pos = target_pos

    clamp_current_pos(filtered_total)

    if st.session_state.get("prev_row_top"):
        st.session_state.current_pos = max(0, st.session_state.current_pos - 1)
    if st.session_state.get("next_row_top"):
        st.session_state.current_pos = min(filtered_total - 1, st.session_state.current_pos + 1)

    clamp_current_pos(filtered_total)

    selected_row = filtered.iloc[st.session_state.current_pos]
    raw_index = selected_row["__original_index__"]

    render_top_strip(
        row=selected_row,
        filtered_position=st.session_state.current_pos,
        filtered_total=filtered_total,
        raw_index=raw_index,
    )

    st.divider()
    render_source_text(selected_row)

    st.divider()
    render_comparison_table(selected_row)

    st.divider()
    render_presence_flags(selected_row)


if __name__ == "__main__":
    main()
