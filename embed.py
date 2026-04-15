"""
embed.py — run once locally to generate data/embeddings.json

pip install sentence-transformers umap-learn scipy scikit-learn
python embed.py
"""

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

DATA_PATH = Path("data/comparison.json")
OUT_PATH  = Path("data/embeddings.json")


def flatten_definition(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


def main():
    print("Loading comparison.json...")
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entries = data["entries"]
    n = len(entries)
    print(f"  {n} entries")

    rb_defs  = [flatten_definition(e["rb"].get("definition")) for e in entries]
    llm_defs = [
        flatten_definition(e["llm"].get("definition")) if e.get("llm") else ""
        for e in entries
    ]

    # ── Encode ────────────────────────────────────────────────────────────────
    print("Loading sentence transformer (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding all definitions...")
    batch = rb_defs + llm_defs
    embeddings = model.encode(
        batch,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    rb_emb  = embeddings[:n]
    llm_emb = embeddings[n:]
    all_emb = np.vstack([rb_emb, llm_emb])

    # ── UMAP 3-D ──────────────────────────────────────────────────────────────
    print("Running UMAP 3D reduction...")
    import umap
    umap_reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    umap_coords     = umap_reducer.fit_transform(all_emb)
    rb_umap_coords  = umap_coords[:n]
    llm_umap_coords = umap_coords[n:]

    # ── PCA 3-D ───────────────────────────────────────────────────────────────
    print("Running PCA 3D reduction...")
    from sklearn.decomposition import PCA
    pca_reducer = PCA(n_components=3, random_state=42)
    pca_coords     = pca_reducer.fit_transform(all_emb)
    rb_pca_coords  = pca_coords[:n]
    llm_pca_coords = pca_coords[n:]

    pca_var = pca_reducer.explained_variance_ratio_
    print(f"  PCA explained variance: "
          f"PC1={pca_var[0]:.3f}  PC2={pca_var[1]:.3f}  PC3={pca_var[2]:.3f}  "
          f"total={pca_var.sum():.3f}")

    # ── Drift scores ──────────────────────────────────────────────────────────
    print("Computing drift scores...")
    from scipy.stats import rankdata

    umap_drifts   = np.linalg.norm(rb_umap_coords  - llm_umap_coords,  axis=1)
    pca_drifts    = np.linalg.norm(rb_pca_coords   - llm_pca_coords,   axis=1)
    umap_drift_pct = rankdata(umap_drifts) / n
    pca_drift_pct  = rankdata(pca_drifts)  / n

    # ── Build output ──────────────────────────────────────────────────────────
    print("Building output...")
    out_entries = []
    for i, e in enumerate(entries):
        cmp = e.get("comparison", {})
        active_strata = [
            k.replace("stratum_", "")
            for k, v in cmp.get("strata", {}).items()
            if v
        ]
        out_entries.append({
            "para_id":    e["para_id"],
            "headword":   e["rb"].get("headword") or "",
            "rb_def":     rb_defs[i],
            "llm_def":    llm_defs[i],
            # UMAP coordinates
            "rb_umap":    [round(float(x), 4) for x in rb_umap_coords[i]],
            "llm_umap":   [round(float(x), 4) for x in llm_umap_coords[i]],
            # PCA coordinates
            "rb_pca":     [round(float(x), 4) for x in rb_pca_coords[i]],
            "llm_pca":    [round(float(x), 4) for x in llm_pca_coords[i]],
            # Backward-compat aliases (UMAP)
            "rb_xyz":     [round(float(x), 4) for x in rb_umap_coords[i]],
            "llm_xyz":    [round(float(x), 4) for x in llm_umap_coords[i]],
            # Drift (both projections)
            "drift":      round(float(umap_drifts[i]),   4),
            "drift_pct":  round(float(umap_drift_pct[i]), 4),
            "drift_pca":  round(float(pca_drifts[i]),    4),
            "drift_pca_pct": round(float(pca_drift_pct[i]), 4),
            # Agreement / metadata
            "agree_pct":  cmp.get("agree_pct", 100),
            "strata":     active_strata,
            "entry_type": e["rb"].get("entry_type", "full") if e.get("rb") else "full",
        })

    output = {
        "meta": {
            "model":      "all-MiniLM-L6-v2",
            "n_entries":  n,
            "umap_params": {
                "n_components": 3,
                "n_neighbors":  15,
                "min_dist":     0.1,
                "metric":       "cosine",
                "random_state": 42,
            },
            "pca_params": {
                "n_components": 3,
                "random_state": 42,
                "explained_variance_ratio": [round(float(v), 4) for v in pca_var],
                "total_explained_variance": round(float(pca_var.sum()), 4),
            },
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        "entries": out_entries,
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"\nDone.")
    print(f"  Entries:         {n}")
    print(f"  UMAP mean drift: {umap_drifts.mean():.4f}  max: {umap_drifts.max():.4f}")
    print(f"  PCA  mean drift: {pca_drifts.mean():.4f}   max: {pca_drifts.max():.4f}")
    print(f"  Output:          {OUT_PATH} ({size_mb:.1f} MB)")

    top5 = sorted(enumerate(umap_drifts), key=lambda x: -x[1])[:5]
    print("  Top 5 UMAP-drift entries:")
    for i, d in top5:
        print(f"    para_id={entries[i]['para_id']}  "
              f"hw={entries[i]['rb'].get('headword', '')}  drift={d:.4f}")


if __name__ == "__main__":
    main()
