"""
hallucination.py — compute LLM hallucination rate from comparison.json

A token is "novel" (hallucinated) if it appears in LLM output fields
but does NOT appear anywhere in the source full_text.
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Windows consoles default to cp1252; force UTF-8 so IPA/diacritic chars print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_PATH = Path("data/comparison.json")
OUT_PATH  = Path("outputs/hallucination_report.json")

LLM_FIELDS = [
    "headword", "headword_raw", "variant_forms",
    "pronunciation", "pronunciation_2", "pronunciation_3",
    "part_of_speech", "grammatical_labels", "definition",
    "etymology", "cross_references", "examples", "region_mentions",
]


# ── Normalisation ─────────────────────────────────────────────────────────────

def flatten(value) -> str:
    """Flatten any field value to a single string."""
    if value is None:
        return ""
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(" ".join(str(v) for v in item.values()))
            else:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values())
    return str(value)


def normalise(text: str) -> list[str]:
    """Apply the full normalisation pipeline and return tokens."""
    # Fullwidth Unicode → ASCII
    s = "".join(
        chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c
        for c in text
    )
    # Strip diacritics
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = [t for t in s.split() if len(t) > 2 and not t.isdigit()]
    return tokens


def field_tokens(value) -> set[str]:
    return set(normalise(flatten(value)))


def source_tokens(full_text: str) -> set[str]:
    return set(normalise(full_text))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading comparison.json...")
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    entries = data["entries"]

    # Filter: full entries where both systems are present
    eligible = [
        e for e in entries
        if e.get("rb") and e.get("llm")
        # entry_type lives on the rb object; default to "full" if absent
        and e["rb"].get("entry_type", "full") == "full"
    ]
    n = len(eligible)
    print(f"  Eligible entries (full, both present): {n}")

    # ── Per-entry computation ─────────────────────────────────────────────────
    per_entry   = []
    novel_freq  = Counter()

    for e in eligible:
        src_tok = source_tokens(e["full_text"])

        llm_tok: set[str] = set()
        for field in LLM_FIELDS:
            llm_tok |= field_tokens(e["llm"].get(field))

        novel = llm_tok - src_tok
        total = len(llm_tok)
        nc    = len(novel)
        rate  = nc / total if total > 0 else 0.0

        novel_freq.update(novel)

        per_entry.append({
            "para_id":      e["para_id"],
            "headword":     e["rb"].get("headword") or "",
            "novel_count":  nc,
            "total_llm":    total,
            "novel_rate":   rate,
            "novel_tokens": sorted(novel),
            "full_text":    e["full_text"],
        })

    # ── Aggregates ────────────────────────────────────────────────────────────
    total_llm_all   = sum(r["total_llm"]   for r in per_entry)
    total_novel_all = sum(r["novel_count"] for r in per_entry)
    overall_rate    = total_novel_all / total_llm_all if total_llm_all > 0 else 0.0
    mean_rate       = sum(r["novel_rate"] for r in per_entry) / n if n > 0 else 0.0

    any_novel       = sum(1 for r in per_entry if r["novel_count"] > 0)
    gt1pct          = sum(1 for r in per_entry if r["novel_rate"] > 0.01)
    gt5pct          = sum(1 for r in per_entry if r["novel_rate"] > 0.05)

    # ── Top entries ───────────────────────────────────────────────────────────
    worst = sorted(per_entry, key=lambda r: r["novel_count"], reverse=True)[:20]

    # ── Print report ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  HALLUCINATION RATE REPORT")
    print("=" * 60)
    print(f"  Entries evaluated: {n:,}")
    print()
    print("  AGGREGATE:")
    print(f"    Total LLM output tokens:     {total_llm_all:,}")
    print(f"    Total novel tokens:          {total_novel_all:,}")
    print(f"    Overall novel rate:          {overall_rate*100:.2f}%")
    print(f"    Mean per-entry novel rate:   {mean_rate*100:.2f}%")
    print()
    print("  PREVALENCE:")
    print(f"    Entries with ANY novel token:   {any_novel:,}  ({any_novel/n*100:.1f}%)")
    print(f"    Entries with novel rate >  1%:  {gt1pct:,}  ({gt1pct/n*100:.1f}%)")
    print(f"    Entries with novel rate >  5%:  {gt5pct:,}  ({gt5pct/n*100:.1f}%)")
    print()
    print("  TOP 20 ENTRIES BY NOVEL TOKEN COUNT:")
    header = f"  {'para_id':>8}  {'headword':<28}  {'novel':>6}  {'total':>6}  {'rate%':>7}  novel_tokens"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in worst:
        tokens_preview = " ".join(r["novel_tokens"][:8])
        if len(r["novel_tokens"]) > 8:
            tokens_preview += f" … (+{len(r['novel_tokens'])-8})"
        print(
            f"  {r['para_id']:>8}  {r['headword']:<28}  "
            f"{r['novel_count']:>6}  {r['total_llm']:>6}  "
            f"{r['novel_rate']*100:>6.1f}%  {tokens_preview}"
        )
    print()
    print("  NOVEL TOKEN FREQUENCY (top 30):")
    for token, count in novel_freq.most_common(30):
        print(f"    {token:<30} {count:>5}")
    print()

    # ── Save JSON report ──────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(exist_ok=True)
    report = {
        "meta": {
            "n_entries": n,
            "scope": "entry_type==full, both systems present",
        },
        "aggregate": {
            "total_llm_tokens":         total_llm_all,
            "total_novel_tokens":       total_novel_all,
            "overall_novel_rate_pct":   round(overall_rate * 100, 4),
            "mean_entry_novel_rate_pct": round(mean_rate * 100, 4),
            "entries_with_any_novel":   any_novel,
            "entries_with_novel_gt1pct": gt1pct,
        },
        "top_novel_tokens": [
            {"token": t, "count": c}
            for t, c in novel_freq.most_common(100)
        ],
        "worst_entries": [
            {
                "para_id":      r["para_id"],
                "headword":     r["headword"],
                "novel_count":  r["novel_count"],
                "total_llm_tokens": r["total_llm"],
                "novel_rate_pct": round(r["novel_rate"] * 100, 4),
                "novel_tokens": r["novel_tokens"],
                "full_text":    r["full_text"],
            }
            for r in worst
        ],
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"  Report saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
