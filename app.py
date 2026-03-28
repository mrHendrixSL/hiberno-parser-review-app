import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pandas as pd
import streamlit as st

# =========================================================
# Config
# =========================================================
DEFAULT_JSONL_PATH = "data/normalized_withhallucinationFlags.jsonl"

st.set_page_config(
    page_title="Parser Output Review",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Helpers / field groups
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

SMALL_COMPARISON_FIELDS = [
    ("headword_raw_rule", "headword_raw_genai"),
    ("headword_rule", "headword_genai"),
    ("variant_forms_raw_rule", "variant_forms_raw_genai"),
    ("variant_forms_rule", "variant_forms_genai"),
    ("pronunciations_rule", "pronunciations_genai"),
    ("part_of_speech_rule", "part_of_speech_genai"),
    ("part_of_speech_rule_norm", "part_of_speech_genai_norm"),
    ("cross_references_rule", "cross_references_genai"),
    ("region_mentions_rule", "region_mentions_genai"),
    ("region_mentions_rule_norm", "region_mentions_genai_norm"),
]

LARGE_COMPARISON_FIELDS = [
    ("definition_rule", "definition_genai"),
    ("examples_rule", "examples_genai"),
    ("etymology_rule", "etymology_genai"),
]

DIAGNOSTIC_FIELDS = [
    "source_char_len",
    "rule_extracted_char_len",
    "genai_extracted_char_len",
    "rule_char_ratio",
    "genai_char_ratio",
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
# Value helpers
# =========================================================
def value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


def normalize_scalar_for_search(value: Any) -> str:
    if value_is_missing(value):
        return ""

    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False).lower()
        except Exception:
            return str(value).lower()

    return str(value).lower()


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
                pass
        return value

    return str(value)


def value_equal(a: Any, b: Any) -> bool:
    if value_is_missing(a) and value_is_missing(b):
        return True
    return a == b


def is_false_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    if value_is_missing(value):
        return False
    v = str(value).strip().lower()
    return v in {"false", "0", "no", "mismatch", "different", "not_match", "not matched"}


# =========================================================
# UI badges / blocks
# =========================================================
def status_badge(value: Any) -> str:
    if value_is_missing(value):
        label = "missing"
        color = "#6b7280"
    else:
        text = str(value).strip()
        lower = text.lower()

        green_values = {"true", "match", "matched", "both_present", "present", "ok", "yes"}
        yellow_values = {
            "partial",
            "one_sided",
            "one-sided",
            "rule_only",
            "genai_only",
            "only_rule",
            "only_genai",
            "left_only",
            "right_only",
        }
        red_values = {"false", "mismatch", "different", "hallucination", "flagged", "error", "conflict"}
        grey_values = {"both_missing", "missing", "none", "null", ""}

        if lower in green_values:
            label = text
            color = "#15803d"
        elif lower in yellow_values:
            label = text
            color = "#ca8a04"
        elif lower in red_values:
            label = text
            color = "#b91c1c"
        elif lower in grey_values:
            label = text if text else "missing"
            color = "#6b7280"
        elif isinstance(value, bool):
            label = text
            color = "#15803d" if value else "#b91c1c"
        else:
            if "match" in lower or "both_present" in lower:
                label = text
                color = "#15803d"
            elif "partial" in lower or "only" in lower or "one" in lower:
                label = text
                color = "#ca8a04"
            elif "mismatch" in lower or "halluc" in lower or lower == "false":
                label = text
                color = "#b91c1c"
            elif "missing" in lower:
                label = text
                color = "#6b7280"
            else:
                label = text
                color = "#374151"

    return (
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;"
        f"border-radius:999px;background:{color};color:white;font-size:0.82rem;"
        f"font-weight:600;white-space:nowrap;'>{label}</span>"
    )


def render_value_block(
    label: str,
    value: Any,
    *,
    is_different: bool = False,
    max_height: int = 320,
    font_size: str = "0.9rem",
) -> None:
    border = "#dc2626" if is_different else "#d1d5db"
    background = "#fef2f2" if is_different else "#f9fafb"

    st.markdown(f"**{label}**")
    st.markdown(
        f"""
        <div style="
            border:1px solid {border};
            background:{background};
            border-radius:8px;
            padding:0.65rem;
            max-height:{max_height}px;
            overflow:auto;
            white-space:pre-wrap;
            font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size:{font_size};
            line-height:1.4;
        ">{pretty_value(value)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_small_value_block(label: str, value: Any, *, is_different: bool = False) -> None:
    render_value_block(
        label=label,
        value=value,
        is_different=is_different,
        max_height=140,
        font_size="0.85rem",
    )


def render_large_value_block(label: str, value: Any, *, is_different: bool = False) -> None:
    render_value_block(
        label=label,
        value=value,
        is_different=is_different,
        max_height=300,
        font_size="0.92rem",
    )


def render_badge_row(data: pd.Series, fields: Iterable[str]) -> None:
    field_list = [f for f in fields if f in data.index]
    if not field_list:
        return

    cols = st.columns(len(field_list))
    for col, field in zip(cols, field_list):
        with col:
            st.caption(field)
            st.markdown(status_badge(data[field]), unsafe_allow_html=True)


# =========================================================
# Filtering / navigation
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

    st.sidebar.header("Filters")

    for field in FILTER_FIELDS:
        if field not in filtered.columns:
            continue

        raw_values = filtered[field].dropna().astype(str).unique().tolist()
        raw_values = sorted(raw_values)

        selected = st.sidebar.multiselect(
            field,
            options=raw_values,
            default=[],
            key=f"filter_{field}",
        )

        if selected:
            filtered = filtered[filtered[field].astype(str).isin(selected)]

    only_false_matches = st.sidebar.checkbox(
        "Show only rows where any match flag is false",
        value=False,
        help="Checks source_text_match, pos_match_norm, and region_mentions_match_norm.",
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


def clamp_current_pos(total_rows: int) -> None:
    if total_rows <= 0:
        st.session_state.current_pos = 0
        return
    st.session_state.current_pos = max(0, min(st.session_state.current_pos, total_rows - 1))


def go_previous() -> None:
    st.session_state.current_pos = max(0, st.session_state.current_pos - 1)


def go_next(total_rows: int) -> None:
    st.session_state.current_pos = min(total_rows - 1, st.session_state.current_pos + 1)


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
    positions = list(filtered_df.index)
    return positions.index(target_index)


# =========================================================
# Render sections
# =========================================================
def render_header(row: pd.Series, filtered_position: int, filtered_total: int, raw_index: Any) -> None:
    st.subheader("Row Summary")
    st.markdown(
        f"""
        **Filtered position:** {filtered_position + 1} / {filtered_total}  
        **Original row index:** {raw_index}
        """
    )

    info_cols = st.columns(3)

    with info_cols[0]:
        st.markdown(f"**entry_id**: `{row.get('entry_id', 'N/A')}`")
        st.markdown(f"**_merge**: {status_badge(row.get('_merge'))}", unsafe_allow_html=True)

    with info_cols[1]:
        st.markdown(
            f"**source_text_match**: {status_badge(row.get('source_text_match'))}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**pos_match_norm**: {status_badge(row.get('pos_match_norm'))}",
            unsafe_allow_html=True,
        )

    with info_cols[2]:
        st.markdown(
            f"**region_mentions_match_norm**: {status_badge(row.get('region_mentions_match_norm'))}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**rule_possible_hallucination**: {status_badge(row.get('rule_possible_hallucination'))}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**genai_possible_hallucination**: {status_badge(row.get('genai_possible_hallucination'))}",
            unsafe_allow_html=True,
        )


def render_source_text(row: pd.Series) -> None:
    st.subheader("Source Text")

    source_value = row.get("source_text_rule")
    render_large_value_block("source_text", source_value, is_different=False)


def render_small_comparison(row: pd.Series) -> None:
    st.subheader("Structured Fields")

    header_left, header_right = st.columns(2)
    with header_left:
        st.markdown("### Rule-based")
    with header_right:
        st.markdown("### GenAI")

    for left_field, right_field in SMALL_COMPARISON_FIELDS:
        val_left = row.get(left_field)
        val_right = row.get(right_field)
        different = not value_equal(val_left, val_right)

        col1, col2 = st.columns(2)
        with col1:
            render_small_value_block(left_field, val_left, is_different=different)
        with col2:
            render_small_value_block(right_field, val_right, is_different=different)


def render_large_comparison(row: pd.Series) -> None:
    st.subheader("Detailed Fields")

    header_left, header_right = st.columns(2)
    with header_left:
        st.markdown("### Rule-based")
    with header_right:
        st.markdown("### GenAI")

    for left_field, right_field in LARGE_COMPARISON_FIELDS:
        val_left = row.get(left_field)
        val_right = row.get(right_field)
        different = not value_equal(val_left, val_right)

        col1, col2 = st.columns(2)
        with col1:
            render_large_value_block(left_field, val_left, is_different=different)
        with col2:
            render_large_value_block(right_field, val_right, is_different=different)


def render_diagnostics(row: pd.Series) -> None:
    st.subheader("Row-Level Diagnostics")

    diag_fields_present = [f for f in DIAGNOSTIC_FIELDS if f in row.index]
    if diag_fields_present:
        diag_cols = st.columns(len(diag_fields_present))
        for col, field in zip(diag_cols, diag_fields_present):
            with col:
                st.caption(field)
                st.code(pretty_value(row.get(field, "N/A")), language=None)

    st.markdown("#### Presence Flags")
    existing_presence_fields = [f for f in PRESENCE_STATUS_FIELDS if f in row.index]
    if existing_presence_fields:
        render_badge_row(row, existing_presence_fields)
    else:
        st.info("No presence status fields available in this row.")


# =========================================================
# App
# =========================================================
def main() -> None:
    init_state()

    st.title("Parser Output Qualitative Review")
    st.caption("Streamlit app for auditing rule-based vs GenAI parser output from a JSONL file.")

    file_path = st.text_input(
        "JSONL path",
        value=DEFAULT_JSONL_PATH,
        help="Path to the merged JSONL used for auditing.",
    )

    try:
        df = load_jsonl(file_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    df = df.reset_index(drop=False).rename(columns={"index": "__original_index__"})

    st.subheader("Navigation & Search")

    nav1, nav2, nav3, nav4 = st.columns([1.1, 1.1, 2, 3])

    with nav1:
        prev_top = st.button("⬅ Previous", use_container_width=True)

    with nav2:
        next_top = st.button("Next ➡", use_container_width=True)

    with nav3:
        st.text_input(
            "Jump by entry_id",
            key="jump_entry_id_input",
            placeholder="Enter exact entry_id",
        )

    with nav4:
        st.text_input(
            "Search text",
            key="search_text_input",
            placeholder="Search source text or headwords",
        )

    filtered = apply_filters(df)
    filtered_total = len(filtered)

    st.markdown(f"**Rows after filtering:** {filtered_total} / {len(df)}")

    if filtered_total == 0:
        st.warning("No rows match the current filters/search.")
        st.stop()

    if prev_top:
        go_previous()
    if next_top:
        go_next(filtered_total)

    jump_value = st.session_state.get("jump_entry_id_input", "").strip()
    if jump_value:
        target_pos = jump_to_entry_id(filtered, jump_value)
        if target_pos is not None:
            st.session_state.current_pos = target_pos
        else:
            st.info("No matching entry_id found within the current filtered subset.")

    nav_buttons = st.columns([1, 1, 4])
    with nav_buttons[0]:
        prev_bottom = st.button("Previous row", key="prev_row_bottom")
    with nav_buttons[1]:
        next_bottom = st.button("Next row", key="next_row_bottom")

    if prev_bottom:
        go_previous()
    if next_bottom:
        go_next(filtered_total)

    clamp_current_pos(filtered_total)

    current_pos = st.session_state.current_pos
    selected_pos = st.number_input(
        "Row selector within filtered subset",
        min_value=0,
        max_value=max(filtered_total - 1, 0),
        value=current_pos,
        step=1,
        help="This index is relative to the filtered subset, not the raw file.",
    )

    if selected_pos != current_pos:
        st.session_state.current_pos = int(selected_pos)

    clamp_current_pos(filtered_total)
    current_pos = st.session_state.current_pos

    selected_row = filtered.iloc[current_pos]
    raw_index = selected_row["__original_index__"]

    render_header(
        row=selected_row,
        filtered_position=current_pos,
        filtered_total=filtered_total,
        raw_index=raw_index,
    )

    st.divider()
    render_source_text(selected_row)

    st.divider()
    render_small_comparison(selected_row)

    st.divider()
    render_large_comparison(selected_row)

    st.divider()
    render_diagnostics(selected_row)


if __name__ == "__main__":
    main()
