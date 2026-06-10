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
    help="Where to save the PCA scatter plot.",
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
)

print("\nMain vs Light Verb Distance")
print(results_df)

results_df.to_csv(args.results_csv, index=False)

# ----------------------------
# 5. PCA visualization
# ----------------------------

X = np.stack(df["embedding"].values)

pca = PCA(n_components=2)
coords = pca.fit_transform(X)

df["pc1"] = coords[:, 0]
df["pc2"] = coords[:, 1]

plt.figure(figsize=(10, 7))

for usage in ["main", "light"]:
    sub = df[df["usage"] == usage]
    plt.scatter(sub["pc1"], sub["pc2"], label=usage, alpha=0.7)

for _, row in df.iterrows():
    plt.text(row["pc1"], row["pc2"], reshape_urdu(row["verb"]), fontsize=8)

plt.title("mBERT Embeddings: Urdu Main Verb vs Light Verb Uses")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(args.pca_png, dpi=300)
plt.show()
