# pip install torch transformers pandas numpy scikit-learn matplotlib

import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from matplotlib import font_manager as fm
from bidi.algorithm import get_display
import arabic_reshaper

URDU_FONT_PATH = "./NotoSansArabic-Regular.ttf"
try:
    urdu_font = fm.FontProperties(fname=URDU_FONT_PATH)
    print("Using font:", urdu_font.get_name())
except Exception as e:
    urdu_font = None
    print("Font not loaded, continuing without Urdu font. Reason:", e)

def reshape_urdu(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

MODEL_NAME = "../UrduBert/urdu-bert-64k-17epochs"

#MODEL_NAME = "bert-base-multilingual-cased"

# Default path to the CSV file containing the example sentences.
# Override on the command line:  python lightverbs.py --input my_examples.csv
DEFAULT_EXAMPLES_CSV = "examples.csv"

# ----------------------------
# 0. Parse CLI arguments
# ----------------------------

parser = argparse.ArgumentParser(
    description="Compare main-verb vs light-verb uses of Urdu verbs using BERT embeddings."
)
parser.add_argument(
    "--input", "-i",
    default=DEFAULT_EXAMPLES_CSV,
    help=f"Path to the input CSV file (default: {DEFAULT_EXAMPLES_CSV}). "
         "Must contain columns: verb, usage, sentence. "
         "`usage` must be either 'main' or 'light'.",
)
parser.add_argument(
    "--results-csv",
    default="urdu_mbert_lightverb_results.csv",
    help="Where to write the per-verb cosine distance results.",
)
parser.add_argument(
    "--pca-png",
    default="urdu_mbert_lightverb_pca.png",
    help="Where to save the per-verb PCA grid (one subplot per verb).",
)
parser.add_argument(
    "--bar-png",
    default="urdu_mbert_lightverb_bars.png",
    help="Where to save the per-verb cosine-distance + example-count bar chart.",
)
parser.add_argument(
    "--global-pca-png",
    default=None,
    help="Optional: also save a single global PCA over ALL verbs to this path. "
         "Off by default (the per-verb grid is usually more readable).",
)
args = parser.parse_args()

# ----------------------------
# 1. Load Urdu mini-corpus from CSV
# ----------------------------
# Expected columns:
#   verb     : the verb being studied (e.g., "دیا")
#   usage    : either "main" or "light"
#   sentence : the example Urdu sentence containing the verb

df = pd.read_csv(args.input, encoding="utf-8")

required_cols = {"verb", "usage", "sentence"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(
        f"Input CSV {args.input!r} is missing required columns: {sorted(missing)}. "
        f"Found columns: {list(df.columns)}"
    )

# Normalise text columns and drop empty rows.
df["verb"]     = df["verb"].astype(str).str.strip()
df["usage"]    = df["usage"].astype(str).str.strip().str.lower()
df["sentence"] = df["sentence"].astype(str).str.strip()
df = df[(df["verb"] != "") & (df["sentence"] != "")].reset_index(drop=True)

# Warn about any rows whose `usage` is not main / light.
valid_usage = {"main", "light"}
bad_usage = df[~df["usage"].isin(valid_usage)]
if not bad_usage.empty:
    print(
        f"Warning: dropping {len(bad_usage)} row(s) whose 'usage' is not in {valid_usage}. "
        f"Unique bad values: {sorted(bad_usage['usage'].unique())}"
    )
    df = df[df["usage"].isin(valid_usage)].reset_index(drop=True)

print(f"Loaded {len(df)} sentences from {args.input!r}")
print("Counts per verb / usage:")
print(df.groupby(["verb", "usage"]).size().unstack(fill_value=0))

# ----------------------------
# 2. Load mBERT
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# ----------------------------
# 3. Extract target embedding
# ----------------------------

def get_target_embedding(sentence, target):
    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=128
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    hidden = outputs.hidden_states[-1][0]

    start = sentence.find(target)
    if start == -1:
        return None

    end = start + len(target)

    token_indices = []
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if max(s, start) < min(e, end):
            token_indices.append(i)

    if not token_indices:
        return None

    return hidden[token_indices].mean(dim=0).numpy()

df["embedding"] = df.apply(
    lambda row: get_target_embedding(row["sentence"], row["verb"]),
    axis=1
)

missing_emb = df["embedding"].isnull().sum()
if missing_emb:
    print(f"Warning: could not extract an embedding for {missing_emb} sentence(s); they will be skipped.")

df = df[df["embedding"].notnull()].reset_index(drop=True)

# ----------------------------
# 4. Main vs light comparison
# ----------------------------

results = []

for verb in df["verb"].unique():
    sub = df[df["verb"] == verb]

    main_sub  = sub[sub["usage"] == "main"]
    light_sub = sub[sub["usage"] == "light"]

    if len(main_sub) == 0 or len(light_sub) == 0:
        print(
            f"Skipping verb {verb!r}: need at least one 'main' and one 'light' example "
            f"(found main={len(main_sub)}, light={len(light_sub)})."
        )
        continue

    main_vecs  = np.stack(main_sub["embedding"])
    light_vecs = np.stack(light_sub["embedding"])

    main_centroid  = main_vecs.mean(axis=0)
    light_centroid = light_vecs.mean(axis=0)

    similarity = cosine_similarity([main_centroid], [light_centroid])[0][0]
    distance = 1 - similarity

    results.append({
        "verb": verb,
        "main_examples": len(main_vecs),
        "light_examples": len(light_vecs),
        "cosine_similarity": similarity,
        "cosine_distance": distance
    })

results_df = pd.DataFrame(results).sort_values(
    "cosine_distance",
    ascending=False
).reset_index(drop=True)

print("\nMain vs Light Verb Distance")
print(results_df)

results_df.to_csv(args.results_csv, index=False)

# ----------------------------
# 5. Visualisation
# ----------------------------
# 5a. Per-verb PCA grid: one subplot per verb (fit PCA only on that verb's
#     examples so inter-verb variance does not dominate the layout).
# 5b. Cosine-distance bar chart + main/light example counts (two panels).
# 5c. (Optional) Global PCA across all verbs (legacy view).

# Vivid blue / red so the contrast is unmistakable even on a small monitor.
COLOR_MAIN  = "#2563eb"   # vivid blue
COLOR_LIGHT = "#dc2626"   # vivid red

verbs_with_both = results_df["verb"].tolist()
n_verbs = len(verbs_with_both)

# Stable per-verb colour map (used by the per-verb PCA titles and the
# separation bar chart) so the same verb always gets the same colour.
_cmap_name = "tab10" if n_verbs <= 10 else "tab20"
_cmap = plt.get_cmap(_cmap_name)
verb_color_map = {
    v: _cmap(i % _cmap.N) for i, v in enumerate(verbs_with_both)
}

# ---- 5a. Per-verb PCA grid ----
if n_verbs > 0:
    n_cols = min(3, n_verbs)
    n_rows = (n_verbs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    for ax_idx, verb in enumerate(verbs_with_both):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]
        sub = df[df["verb"] == verb].copy()

        if len(sub) < 2:
            ax.text(0.5, 0.5, "not enough points",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        X_verb = np.stack(sub["embedding"].values)
        pca_verb = PCA(n_components=2)
        coords_verb = pca_verb.fit_transform(X_verb)
        sub["pc1"] = coords_verb[:, 0]
        sub["pc2"] = coords_verb[:, 1]

        for usage, color, marker in [
            ("main",  COLOR_MAIN,  "o"),
            ("light", COLOR_LIGHT, "X"),
        ]:
            sub_u = sub[sub["usage"] == usage]
            if len(sub_u) == 0:
                continue
            ax.scatter(
                sub_u["pc1"], sub_u["pc2"],
                c=color, marker=marker, alpha=0.6, s=60,
                edgecolors="white", linewidths=0.5,
                label=f"{usage} (n={len(sub_u)})",
            )
            # Big star centroid per group.
            cx = sub_u["pc1"].mean()
            cy = sub_u["pc2"].mean()
            ax.scatter([cx], [cy], c=color, marker="*", s=320,
                       edgecolors="black", linewidths=1.2, zorder=5)

        # Title: Urdu verb + cosine distance for this verb.
        row_res = results_df[results_df["verb"] == verb].iloc[0]
        cos_dist = row_res["cosine_distance"]
        try:
            verb_display = reshape_urdu(verb)
        except Exception:
            verb_display = verb
        ax.set_title(
            f"{verb_display}    cos_dist = {cos_dist:.3f}",
            fontproperties=urdu_font if urdu_font else None,
            fontsize=13,
            color=verb_color_map.get(verb, "black"),
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    # Hide any unused subplots in the trailing row.
    for ax_idx in range(n_verbs, n_rows * n_cols):
        axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

    fig.suptitle("Per-Verb PCA: Main vs Light Usage", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.pca_png, dpi=200)
    print(f"Saved per-verb PCA grid to {args.pca_png}")
    plt.show()

# ---- 5b. Cosine-distance bar chart + example-count grouped bars ----
if not results_df.empty:
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(15, max(4.0, 0.7 * len(results_df) + 1.5)),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 2]},
    )

    verb_labels = [
        (reshape_urdu(v) if urdu_font else v) for v in results_df["verb"]
    ]
    bar_colors = [verb_color_map[v] for v in results_df["verb"]]

    # --- Left panel: separation (cosine distance) ---
    y_pos = np.arange(len(results_df))
    ax1.barh(
        y_pos,
        results_df["cosine_distance"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        height=0.7,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(
        verb_labels,
        fontproperties=urdu_font if urdu_font else None,
        fontsize=13,
    )
    for tick_lbl, c in zip(ax1.get_yticklabels(), bar_colors):
        tick_lbl.set_color(c)
        tick_lbl.set_fontweight("bold")

    ax1.invert_yaxis()
    ax1.set_xlabel("Cosine distance between main and light centroids")
    ax1.set_title("Per-Verb Main vs Light Separation", fontsize=13)
    x_max = results_df["cosine_distance"].max()
    for i, val in enumerate(results_df["cosine_distance"]):
        ax1.text(
            val + x_max * 0.01, i, f"{val:.3f}",
            va="center", fontsize=10, fontweight="bold",
        )
    ax1.set_xlim(0, x_max * 1.18)
    ax1.grid(True, axis="x", alpha=0.3)

    # --- Right panel: main vs light example counts ---
    bar_h = 0.38
    ax2.barh(
        y_pos - bar_h / 2,
        results_df["main_examples"],
        height=bar_h,
        color=COLOR_MAIN,
        label="main (n)",
        edgecolor="black",
        linewidth=0.6,
    )
    ax2.barh(
        y_pos + bar_h / 2,
        results_df["light_examples"],
        height=bar_h,
        color=COLOR_LIGHT,
        label="light (n)",
        edgecolor="black",
        linewidth=0.6,
    )
    ax2.set_xlabel("Number of examples")
    ax2.set_title("Example Counts per Verb", fontsize=13)
    ax2.legend(loc="lower right", fontsize=11, frameon=True)
    ax2.grid(True, axis="x", alpha=0.3)

    count_max = max(
        results_df["main_examples"].max(),
        results_df["light_examples"].max(),
    )
    for i, row in results_df.iterrows():
        ax2.text(
            row["main_examples"] + count_max * 0.015, i - bar_h / 2,
            str(int(row["main_examples"])),
            va="center", fontsize=10, fontweight="bold", color=COLOR_MAIN,
        )
        ax2.text(
            row["light_examples"] + count_max * 0.015, i + bar_h / 2,
            str(int(row["light_examples"])),
            va="center", fontsize=10, fontweight="bold", color=COLOR_LIGHT,
        )
    ax2.set_xlim(0, count_max * 1.18)

    fig.suptitle(
        "Per-Verb Main vs Light: Separation and Sample Sizes",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.bar_png, dpi=200, bbox_inches="tight")
    print(f"Saved bar chart to {args.bar_png}")
    plt.show()

# ---- 5c. Optional global PCA across all verbs ----
if args.global_pca_png:
    X = np.stack(df["embedding"].values)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(11, 8))
    unique_verbs = list(df["verb"].unique())
    global_color_map = {
        v: verb_color_map.get(v, _cmap(i % _cmap.N))
        for i, v in enumerate(unique_verbs)
    }

    for usage, marker in [("main", "o"), ("light", "X")]:
        for v in unique_verbs:
            sub = df[(df["verb"] == v) & (df["usage"] == usage)]
            if len(sub) == 0:
                continue
            label = f"{reshape_urdu(v) if urdu_font else v} ({usage})"
            ax.scatter(
                sub["pc1"], sub["pc2"],
                c=[global_color_map[v]], marker=marker, alpha=0.5, s=45,
                label=label,
            )

    ax.set_title("Global PCA across all verbs (main = circle, light = X)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, ncol=2,
              prop=urdu_font if urdu_font else None)
    plt.tight_layout()
    plt.savefig(args.global_pca_png, dpi=200)
    print(f"Saved global PCA to {args.global_pca_png}")
    plt.show()
