#!/usr/bin/env python3
"""
visualize_advanced.py

Build a single interactive HTML report exploring main- vs light-verb
embeddings produced from examples.csv.

Sections in the report
----------------------
1. Summary stats per verb (counts, cosine distance between centroids).
2. Global UMAP scatter (all verbs in one chart; hover = Urdu sentence).
3. Per-verb UMAP grid (UMAP refit per verb; hover = Urdu sentence).
4. Per-verb pairwise cosine-similarity histograms
   (main↔main, light↔light, main↔light).
5. Sentence-by-sentence similarity-matrix heatmap.
6. Logistic-regression probe per verb + table of misclassified sentences
   (good candidates for re-labelling).

Usage
-----
    pip install umap-learn plotly scikit-learn pandas numpy
    pip install torch transformers   # only needed when no cache exists

    python visualize_advanced.py \\
        --input examples.csv \\
        --output urdu_lightverb_report.html \\
        --embeddings-cache embeddings_cache.npz

The `--embeddings-cache` .npz is signature-checked against the CSV, so as
long as the CSV does not change, BERT runs only once.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME_DEFAULT = "../UrduBert/urdu-bert-64k-17epochs"

COLOR_MAIN  = "#2563eb"   # vivid blue
COLOR_LIGHT = "#dc2626"   # vivid red
COLOR_MIX   = "#6b7280"   # neutral grey for main↔light


# ---------------------------------------------------------------------------
# Embeddings (with on-disk cache keyed by CSV content)
# ---------------------------------------------------------------------------

def _csv_signature(df: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{r.verb}\t{r.usage}\t{r.sentence}"
        for r in df.itertuples(index=False)
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def attach_embeddings(df: pd.DataFrame, cache_path: str | None,
                     model_name: str) -> pd.DataFrame:
    """Return a NEW dataframe (subset of df) with an 'embedding' column."""
    sig = _csv_signature(df)
    cache = Path(cache_path) if cache_path else None

    if cache and cache.exists():
        cached = np.load(cache, allow_pickle=True)
        if str(cached.get("signature", "")) == sig:
            embeddings = cached["embeddings"]
            keep_idx = cached["keep_idx"]
            print(f"Loaded {len(embeddings)} cached embeddings from {cache}")
            out = df.iloc[keep_idx].reset_index(drop=True).copy()
            out["embedding"] = list(embeddings)
            return out
        print(f"Cache signature mismatch in {cache}; recomputing.")

    print(f"Loading BERT model: {model_name}")
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    vecs, keep_idx = [], []
    df_reset = df.reset_index(drop=True)
    for i, row in df_reset.iterrows():
        sentence, target = row["sentence"], row["verb"]
        encoded = tokenizer(
            sentence, return_tensors="pt",
            return_offsets_mapping=True, truncation=True, max_length=128,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]
        start = sentence.find(target)
        if start == -1:
            continue
        end = start + len(target)
        idxs = [
            j for j, (s, e) in enumerate(offsets)
            if s != e and max(s, start) < min(e, end)
        ]
        if not idxs:
            continue
        vecs.append(hidden[idxs].mean(dim=0).numpy())
        keep_idx.append(i)
        if (i + 1) % 25 == 0 or (i + 1) == len(df_reset):
            print(f"  embedded {i + 1}/{len(df_reset)}")

    if not vecs:
        raise RuntimeError("No embeddings could be extracted.")
    embeddings = np.stack(vecs)
    if cache:
        np.savez(cache, embeddings=embeddings, signature=sig,
                 keep_idx=np.array(keep_idx))
        print(f"Saved embeddings cache to {cache}")

    out = df_reset.iloc[keep_idx].reset_index(drop=True).copy()
    out["embedding"] = list(embeddings)
    return out


# ---------------------------------------------------------------------------
# Section builders (each returns an HTML string)
# ---------------------------------------------------------------------------

def section_summary_table(df: pd.DataFrame):
    from sklearn.metrics.pairwise import cosine_similarity
    rows = []
    for verb in df["verb"].unique():
        sub = df[df["verb"] == verb]
        n_main = int((sub["usage"] == "main").sum())
        n_light = int((sub["usage"] == "light").sum())
        cos_dist = np.nan
        if n_main and n_light:
            vm = np.stack(sub[sub["usage"] == "main"]["embedding"]).mean(axis=0)
            vl = np.stack(sub[sub["usage"] == "light"]["embedding"]).mean(axis=0)
            cos_dist = 1.0 - float(cosine_similarity([vm], [vl])[0, 0])
        rows.append({
            "verb": verb,
            "main": n_main,
            "light": n_light,
            "total": n_main + n_light,
            "cosine_distance": cos_dist,
        })
    summary = (pd.DataFrame(rows)
               .sort_values("cosine_distance", ascending=False, na_position="last")
               .reset_index(drop=True))
    return summary, summary.to_html(index=False, classes="table",
                                    float_format=lambda x: f"{x:.3f}")


def section_global_umap(df: pd.DataFrame) -> str:
    try:
        import umap
    except ImportError:
        return ("<p><em>UMAP not installed. "
                "Run <code>pip install umap-learn</code>.</em></p>")
    import plotly.express as px

    X = np.stack(df["embedding"].values)
    n_neighbors = max(2, min(15, len(df) - 1))
    reducer = umap.UMAP(n_components=2, random_state=42,
                        n_neighbors=n_neighbors, min_dist=0.1)
    coords = reducer.fit_transform(X)
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    fig = px.scatter(
        plot_df, x="x", y="y",
        color="verb", symbol="usage",
        hover_data={"sentence": True, "verb": True, "usage": True,
                    "x": False, "y": False},
        title="Global UMAP — colour = verb, symbol = main/light",
        width=950, height=620,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))
    fig.update_layout(legend=dict(itemsizing="constant"))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def section_per_verb_umap(df: pd.DataFrame) -> str:
    try:
        import umap
    except ImportError:
        return ""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    verbs = [v for v in df["verb"].unique()
             if ((df[df["verb"] == v]["usage"] == "main").any()
                 and (df[df["verb"] == v]["usage"] == "light").any())]
    if not verbs:
        return "<p><em>No verb has both main and light examples.</em></p>"

    n_cols = min(3, len(verbs))
    n_rows = (len(verbs) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=verbs,
        horizontal_spacing=0.08, vertical_spacing=0.14,
    )

    legend_shown = {"main": False, "light": False}
    for i, verb in enumerate(verbs):
        row, col = i // n_cols + 1, i % n_cols + 1
        sub = df[df["verb"] == verb].reset_index(drop=True)
        if len(sub) < 4:
            continue
        X = np.stack(sub["embedding"].values)
        n_neighbors = max(2, min(15, len(sub) - 1))
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=n_neighbors, min_dist=0.1)
        coords = reducer.fit_transform(X)
        sub["x"] = coords[:, 0]
        sub["y"] = coords[:, 1]

        for usage, color, symbol in [
            ("main",  COLOR_MAIN,  "circle"),
            ("light", COLOR_LIGHT, "x"),
        ]:
            sub_u = sub[sub["usage"] == usage]
            if len(sub_u) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub_u["x"], y=sub_u["y"],
                    mode="markers",
                    marker=dict(color=color, size=10, symbol=symbol,
                                line=dict(color="white", width=0.5)),
                    text=sub_u["sentence"],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    name=usage,
                    legendgroup=usage,
                    showlegend=not legend_shown[usage],
                ),
                row=row, col=col,
            )
            legend_shown[usage] = True

    fig.update_layout(
        height=340 * n_rows + 80,
        width=340 * n_cols,
        title_text="Per-Verb UMAP (hover any point to read the sentence)",
        showlegend=True,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def section_similarity_histograms(df: pd.DataFrame) -> str:
    from sklearn.metrics.pairwise import cosine_similarity
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    verbs = [v for v in df["verb"].unique()
             if ((df[df["verb"] == v]["usage"] == "main").any()
                 and (df[df["verb"] == v]["usage"] == "light").any())]
    if not verbs:
        return ""

    n_cols = min(3, len(verbs))
    n_rows = (len(verbs) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=verbs,
        horizontal_spacing=0.08, vertical_spacing=0.16,
    )

    def _pairwise(A, B=None):
        if B is None:
            sim = cosine_similarity(A)
            iu = np.triu_indices_from(sim, k=1)
            return sim[iu]
        return cosine_similarity(A, B).flatten()

    legend_shown = {"main↔main": False, "light↔light": False, "main↔light": False}
    for i, verb in enumerate(verbs):
        row, col = i // n_cols + 1, i % n_cols + 1
        sub = df[df["verb"] == verb]
        main_X  = np.stack(sub[sub["usage"] == "main"]["embedding"])
        light_X = np.stack(sub[sub["usage"] == "light"]["embedding"])
        sim_mm = _pairwise(main_X)  if len(main_X)  >= 2 else np.array([])
        sim_ll = _pairwise(light_X) if len(light_X) >= 2 else np.array([])
        sim_ml = _pairwise(main_X, light_X)

        for name, vals, color in [
            ("main↔main",  sim_mm, COLOR_MAIN),
            ("light↔light", sim_ll, COLOR_LIGHT),
            ("main↔light", sim_ml, COLOR_MIX),
        ]:
            if len(vals) == 0:
                continue
            fig.add_trace(
                go.Histogram(
                    x=vals, name=name, marker_color=color,
                    opacity=0.6, nbinsx=20,
                    legendgroup=name, showlegend=not legend_shown[name],
                ),
                row=row, col=col,
            )
            legend_shown[name] = True

    fig.update_layout(
        barmode="overlay",
        height=320 * n_rows + 80,
        width=340 * n_cols,
        title_text=("Per-verb pairwise cosine similarities. "
                    "Clean separation → grey curve sits to the left of blue/red."),
    )
    fig.update_xaxes(title_text="cosine similarity", range=[0, 1])
    return fig.to_html(full_html=False, include_plotlyjs=False)


def section_similarity_heatmap(df: pd.DataFrame) -> str:
    from sklearn.metrics.pairwise import cosine_similarity
    import plotly.graph_objects as go

    sorted_df = df.sort_values(["verb", "usage"]).reset_index(drop=True)
    X = np.stack(sorted_df["embedding"].values)
    sim = cosine_similarity(X)
    labels = [f"{r.verb}/{r.usage}"
              for r in sorted_df.itertuples(index=False)]

    fig = go.Figure(data=go.Heatmap(
        z=sim, x=labels, y=labels,
        colorscale="RdBu_r", zmin=0.0, zmax=1.0, zmid=0.5,
        colorbar=dict(title="cosine sim"),
        hovertemplate="<b>%{y}</b><br>vs <b>%{x}</b><br>sim = %{z:.3f}"
                      "<extra></extra>",
    ))
    fig.update_layout(
        title="Cosine similarity between every sentence (sorted by verb, then usage)",
        width=950, height=950,
        xaxis=dict(tickangle=-90, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def section_probe(df: pd.DataFrame) -> str:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, f1_score

    rows, misclassified = [], []
    for verb in df["verb"].unique():
        sub = df[df["verb"] == verb].reset_index(drop=True)
        n_main  = int((sub["usage"] == "main").sum())
        n_light = int((sub["usage"] == "light").sum())
        if n_main < 2 or n_light < 2:
            continue
        X = np.stack(sub["embedding"].values)
        y = (sub["usage"] == "light").astype(int).values
        n_splits = min(5, min(n_main, n_light))
        if n_splits < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            proba = cross_val_predict(
                clf, X, y, cv=cv, method="predict_proba"
            )[:, 1]
        except Exception as exc:
            print(f"Probe failed for {verb}: {exc}", file=sys.stderr)
            continue
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "verb": verb,
            "n": int(len(y)),
            "baseline (majority)": round(max(y.mean(), 1 - y.mean()), 3),
            "accuracy": round(accuracy_score(y, pred), 3),
            "f1": round(f1_score(y, pred, zero_division=0), 3),
        })
        for i in range(len(y)):
            if y[i] != pred[i]:
                misclassified.append({
                    "verb": verb,
                    "true": "light" if y[i] else "main",
                    "predicted": "light" if pred[i] else "main",
                    "p(light)": round(float(proba[i]), 3),
                    "sentence": sub.iloc[i]["sentence"],
                })

    probe_df = (pd.DataFrame(rows).sort_values("f1", ascending=False)
                if rows else pd.DataFrame())
    miscls_df = (pd.DataFrame(misclassified)
                 .sort_values(["verb", "p(light)"], ascending=[True, False])
                 if misclassified else pd.DataFrame())

    html = ["<h3>Per-verb probe (5-fold cross-validation)</h3>"]
    if probe_df.empty:
        html.append("<p>Not enough data to train any probe.</p>")
    else:
        html.append(probe_df.to_html(index=False, classes="table"))
    if not miscls_df.empty:
        html.append(
            f"<h3>Sentences the probe disagrees with ({len(miscls_df)})</h3>"
        )
        html.append('<p style="color:#666;font-size:0.92em;">'
                    "These are candidates for re-labelling or removal: the "
                    "BERT embedding looks more like the <em>other</em> class "
                    "than the label says.</p>")
        html.append(miscls_df.to_html(index=False, classes="table"))
    elif not probe_df.empty:
        html.append("<p>No disagreements — every label is consistent with "
                    "the embedding-space classifier.</p>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
       Helvetica, Arial, sans-serif; max-width: 1180px; margin: 2em auto;
       padding: 0 1em; color: #1f2937; line-height: 1.5; }
h1 { border-bottom: 2px solid #2563eb; padding-bottom: 0.3em; }
h2 { margin-top: 2.4em; color: #2563eb; border-bottom: 1px solid #e5e7eb;
     padding-bottom: 0.2em; }
h3 { margin-top: 1.5em; color: #374151; }
p  { max-width: 80ch; }
.table { border-collapse: collapse; margin: 1em 0; font-size: 0.92em; }
.table th, .table td { border: 1px solid #e5e7eb; padding: 5px 9px;
                      text-align: left; }
.table th { background: #f3f4f6; }
.toc { background: #f9fafb; padding: 1em 1.5em; border-radius: 6px;
       border: 1px solid #e5e7eb; }
.toc a { display: block; padding: 0.18em 0; color: #2563eb;
         text-decoration: none; }
.toc a:hover { text-decoration: underline; }
code { background: #f3f4f6; padding: 0.1em 0.4em; border-radius: 3px;
       font-size: 0.92em; }
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a single interactive HTML report exploring "
                    "main vs light verb embeddings."
    )
    parser.add_argument("--input", "-i", default="examples.csv",
                        help="Input CSV with columns verb,usage,sentence.")
    parser.add_argument("--output", "-o",
                        default="urdu_lightverb_report.html",
                        help="Output HTML report path.")
    parser.add_argument("--embeddings-cache", default="embeddings_cache.npz",
                        help="Path to .npz cache of embeddings. Skips BERT "
                             "when the cache is in sync with the CSV.")
    parser.add_argument("--model", default=MODEL_NAME_DEFAULT,
                        help="BERT model name or local path.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")
    required = {"verb", "usage", "sentence"}
    if not required <= set(df.columns):
        print(f"Input CSV must have columns: {sorted(required)}",
              file=sys.stderr)
        return 2

    df["verb"]     = df["verb"].astype(str).str.strip()
    df["usage"]    = df["usage"].astype(str).str.strip().str.lower()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df = df[df["usage"].isin(["main", "light"])].reset_index(drop=True)
    df = df[(df["verb"] != "") & (df["sentence"] != "")].reset_index(drop=True)
    print(f"Loaded {len(df)} sentences from {args.input}")

    df = attach_embeddings(df, args.embeddings_cache, args.model)
    print(f"Embedded {len(df)} sentences. Building report ...")

    summary_df, summary_html = section_summary_table(df)
    print("  global UMAP ...")
    global_umap_html = section_global_umap(df)
    print("  per-verb UMAP ...")
    per_verb_umap_html = section_per_verb_umap(df)
    print("  similarity histograms ...")
    hist_html = section_similarity_histograms(df)
    print("  similarity heatmap ...")
    heatmap_html = section_similarity_heatmap(df)
    print("  logistic probe ...")
    probe_html = section_probe(df)

    body = f"""
<h1>Urdu Main vs Light Verb Analysis</h1>
<p>Embeddings: <code>{args.model}</code><br>
   Source: <code>{args.input}</code> ({len(df)} sentences after
   embedding extraction).</p>

<div class="toc">
  <strong>Contents</strong>
  <a href="#summary">1. Summary stats per verb</a>
  <a href="#umap-global">2. Global UMAP (interactive, hover = sentence)</a>
  <a href="#umap-perverb">3. Per-verb UMAP grid</a>
  <a href="#hist">4. Per-verb similarity histograms</a>
  <a href="#heatmap">5. Similarity matrix heatmap</a>
  <a href="#probe">6. Logistic-regression probe + misclassified sentences</a>
</div>

<h2 id="summary">1. Summary stats per verb</h2>
<p>Cosine distance between the main- and light-verb centroids (sorted highest
   to lowest separation).</p>
{summary_html}

<h2 id="umap-global">2. Global UMAP</h2>
<p>All verbs in one 2-D projection. Colour = verb, shape = main/light.
   <strong>Hover any point to see the Urdu sentence</strong> — great for
   spotting outliers that may be label errors.</p>
{global_umap_html}

<h2 id="umap-perverb">3. Per-verb UMAP</h2>
<p>One panel per verb, UMAP refit only on that verb so the layout shows
   <em>within-verb</em> main vs light structure. Blue circle = main, red × =
   light. Hover to read the sentence.</p>
{per_verb_umap_html}

<h2 id="hist">4. Per-verb similarity histograms</h2>
<p>Pairwise cosine similarities within main (blue), within light (red), and
   <em>between</em> main and light (grey). When a verb separates cleanly,
   the grey distribution sits to the <em>left</em> of the two within-group
   distributions.</p>
{hist_html}

<h2 id="heatmap">5. Similarity matrix heatmap</h2>
<p>Pairwise cosine similarity between every sentence, sorted by verb and then
   usage. A strong block-diagonal pattern means embeddings cluster by verb;
   sub-blocks within a verb mean main and light are also separable.</p>
{heatmap_html}

<h2 id="probe">6. Logistic-regression probe</h2>
<p>Per verb, a logistic regression is trained on the BERT embeddings to
   predict main vs light, using 5-fold stratified cross-validation. The
   probe gives a quantitative measure of how separable the two classes are,
   and exposes the sentences whose label disagrees with the embedding-space
   neighbourhood — those are the most useful ones to re-check by hand.</p>
{probe_html}
"""

    html = (
        "<!doctype html>\n"
        "<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
        "<title>Urdu Light-Verb Analysis</title>\n"
        f"<style>{CSS}</style>\n"
        "</head><body>\n"
        f"{body}\n"
        "</body></html>\n"
    )
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
