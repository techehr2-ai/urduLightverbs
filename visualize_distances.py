# pip install torch transformers pandas numpy scikit-learn scipy matplotlib arabic-reshaper python-bidi
"""
visualize_distances.py
======================

Statistical separability analysis: Urdu main vs light verb uses.

For each verb in `examples.csv` this script computes:

  1.  Cosine distance  between main / light centroids
  2.  Cohen's d        on the LDA-projected 1D axis
  3.  Mahalanobis      between centroids (Ledoit-Wolf shrunk covariance)
  4.  Probe AUC        cross-validated linear classifier (main vs light)
  5.  Permutation p    is the observed separation > what shuffled labels give?

All five metrics ship with 95% bootstrap confidence intervals.

Outputs (in --out-dir, default `viz_distances_out/`):

  master_table.csv               every metric + CI per verb
  01_forest_cohens_d.png         forest plot ranked by effect size
  02_lda_kde.png                 per-verb 1D KDEs on LDA axis + overlap coef
  03_pca_contours.png            per-verb 2D KDE contours (PCA space)
  04_within_between_violin.png   within-class vs between-class cosine sims
  05_roc_curves.png              one ROC per verb, AUC annotated
  06_permutation_null.png        per-verb null distribution + observed value
  conclusion.txt                 auto-ranked summary with significance stars

Cache:
  By default the script first tries to load `embeddings_cache.npz`
  (a `np.savez` archive with keys: embeddings, verbs, usages, sentences).
  If the cache is missing, stale, or `--force-recompute` is passed,
  it loads UrduBERT and re-extracts target-verb embeddings, then writes
  a fresh cache for next time.  BERT is therefore only loaded once.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm

from scipy.stats import gaussian_kde

from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------
# Urdu rendering (same setup as lightverbs.py)
# ----------------------------------------------------------------------
URDU_FONT_PATH = "./NotoSansArabic-Regular.ttf"
try:
    urdu_font = fm.FontProperties(fname=URDU_FONT_PATH)
    print("Using font:", urdu_font.get_name())
except Exception as exc:
    urdu_font = None
    print("Font not loaded, continuing without Urdu font. Reason:", exc)

try:
    from bidi.algorithm import get_display
    import arabic_reshaper
    HAS_BIDI = True
except ImportError:
    HAS_BIDI = False
    print("arabic_reshaper / python-bidi not installed — Urdu labels may render LTR.")


def reshape_urdu(text: str) -> str:
    if HAS_BIDI:
        return get_display(arabic_reshaper.reshape(text))
    return text


def verb_label(verb: str) -> str:
    return reshape_urdu(verb) if urdu_font is not None else verb


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
DEFAULT_CSV = "examples.csv"
DEFAULT_CACHE = "embeddings_cache.npz"
DEFAULT_MODEL = "../UrduBert/urdu-bert-64k-17epochs"
DEFAULT_OUT = "viz_distances_out"

N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
RANDOM_SEED = 42


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Statistical separability of Urdu main vs light verbs."
)
parser.add_argument("--input", "-i", default=DEFAULT_CSV,
                    help=f"Input CSV (default: {DEFAULT_CSV})")
parser.add_argument("--cache", default=DEFAULT_CACHE,
                    help=f"Embeddings cache npz (default: {DEFAULT_CACHE})")
parser.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF model path (default: {DEFAULT_MODEL})")
parser.add_argument("--out-dir", default=DEFAULT_OUT,
                    help=f"Output directory (default: {DEFAULT_OUT})")
parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
parser.add_argument("--seed", type=int, default=RANDOM_SEED)
parser.add_argument("--force-recompute", action="store_true",
                    help="Ignore cache and rebuild embeddings from BERT.")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Load CSV
# ----------------------------------------------------------------------
df = pd.read_csv(args.input, encoding="utf-8")
required = {"verb", "usage", "sentence"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV {args.input!r} missing columns {sorted(missing)}")

df["verb"] = df["verb"].astype(str).str.strip()
df["usage"] = df["usage"].astype(str).str.strip().str.lower()
df["sentence"] = df["sentence"].astype(str).str.strip()
df = df[(df["verb"] != "") & (df["sentence"] != "")].reset_index(drop=True)
df = df[df["usage"].isin({"main", "light"})].reset_index(drop=True)

print(f"\nLoaded {len(df)} sentences from {args.input!r}")
print("Counts per verb / usage:")
print(df.groupby(["verb", "usage"]).size().unstack(fill_value=0))


# ----------------------------------------------------------------------
# Embedding extraction (only if cache is missing / stale)
# ----------------------------------------------------------------------
def compute_embeddings_from_bert(frame: pd.DataFrame) -> list:
    """Compute target-verb embeddings using UrduBERT. Mirrors lightverbs.py."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    def get_target_embedding(sentence: str, target: str):
        encoded = tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=128,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]

        start = sentence.find(target)
        if start == -1:
            return None
        end = start + len(target)
        token_idx = [
            i for i, (s, e) in enumerate(offsets)
            if s != e and max(s, start) < min(e, end)
        ]
        if not token_idx:
            return None
        return hidden[token_idx].mean(dim=0).numpy()

    embs = []
    for i, row in frame.iterrows():
        embs.append(get_target_embedding(row["sentence"], row["verb"]))
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(frame)}")
    return embs


def try_load_cache(path: str, expected_n: int):
    """Return X (np.ndarray) if cache is loadable & matches row count, else None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        cache = np.load(p, allow_pickle=True)
    except Exception as e:
        print(f"Could not read cache {path!r}: {e}")
        return None

    # accept several common key names
    for key in ("embeddings", "X", "arr_0"):
        if key in cache.files:
            X = cache[key]
            break
    else:
        print(f"Cache {path!r} has no recognised embedding key. Found: {cache.files}")
        return None

    if len(X) != expected_n:
        print(f"Cache size ({len(X)}) != CSV size ({expected_n}). Recomputing.")
        return None
    return np.asarray(X, dtype=np.float64)


X = None if args.force_recompute else try_load_cache(args.cache, len(df))

if X is not None:
    print(f"Loaded {len(X)} cached embeddings from {args.cache} "
          f"(dim={X.shape[1]}) — BERT will NOT be loaded.")
    df["embedding"] = list(X)
else:
    print("\nCache miss — computing embeddings with UrduBERT...")
    embs = compute_embeddings_from_bert(df)
    df["embedding"] = embs
    n_missing = df["embedding"].isnull().sum()
    if n_missing:
        print(f"Warning: dropped {n_missing} row(s) whose target token could not be located.")
    df = df[df["embedding"].notnull()].reset_index(drop=True)
    X = np.stack(df["embedding"].values)
    np.savez(
        args.cache,
        embeddings=X,
        verbs=df["verb"].values,
        usages=df["usage"].values,
        sentences=df["sentence"].values,
    )
    print(f"Saved cache → {args.cache}  (shape={X.shape})")


# ----------------------------------------------------------------------
# Metric primitives
# ----------------------------------------------------------------------
def m_cosine_distance(A: np.ndarray, B: np.ndarray) -> float:
    sim = cosine_similarity(A.mean(0, keepdims=True), B.mean(0, keepdims=True))[0, 0]
    return float(1 - sim)


def m_cohens_d_lda(A: np.ndarray, B: np.ndarray) -> float:
    """Cohen's d of LDA-projected scalar scores. Robust for high-dim BERT vectors."""
    X_ = np.vstack([A, B])
    y_ = np.array([0] * len(A) + [1] * len(B))
    lda = LinearDiscriminantAnalysis(n_components=1, solver="svd")
    try:
        z = lda.fit_transform(X_, y_).ravel()
    except Exception:
        return float("nan")
    za, zb = z[y_ == 0], z[y_ == 1]
    if len(za) < 2 or len(zb) < 2:
        return float("nan")
    pooled_var = ((len(za) - 1) * za.var(ddof=1) + (len(zb) - 1) * zb.var(ddof=1)) \
                 / (len(za) + len(zb) - 2)
    if pooled_var <= 0:
        return float("nan")
    return float(abs(za.mean() - zb.mean()) / np.sqrt(pooled_var))


def m_mahalanobis(A: np.ndarray, B: np.ndarray) -> float:
    """Mahalanobis distance between centroids with Ledoit-Wolf shrinkage."""
    pooled = np.vstack([A, B])
    if pooled.shape[0] < 3:
        return float("nan")
    try:
        lw = LedoitWolf().fit(pooled)
        diff = A.mean(0) - B.mean(0)
        val = diff @ lw.precision_ @ diff
        if val < 0:
            return float("nan")
        return float(np.sqrt(val))
    except Exception:
        return float("nan")


def m_probe_auc(A: np.ndarray, B: np.ndarray, seed: int = 42) -> float:
    """Cross-validated logistic-regression probe AUC."""
    X_ = np.vstack([A, B])
    y_ = np.array([0] * len(A) + [1] * len(B))
    n_splits = min(5, len(A), len(B))
    if n_splits < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X_, y_):
        if len(np.unique(y_[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_[tr], y_[tr])
        scores = clf.decision_function(X_[te])
        fpr, tpr, _ = roc_curve(y_[te], scores)
        aucs.append(auc(fpr, tpr))
    return float(np.mean(aucs)) if aucs else float("nan")


def bootstrap_ci(A, B, metric_fn, n=N_BOOTSTRAP, rng=None, ci=(2.5, 97.5)):
    if rng is None:
        rng = np.random.default_rng()
    vals = []
    for _ in range(n):
        ai = rng.integers(0, len(A), len(A))
        bi = rng.integers(0, len(B), len(B))
        try:
            v = metric_fn(A[ai], B[bi])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, ci[0])), float(np.percentile(vals, ci[1]))


def permutation_test(A, B, metric_fn, n=N_PERMUTATIONS, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    observed = metric_fn(A, B)
    combined = np.vstack([A, B])
    labels = np.array([0] * len(A) + [1] * len(B))
    null = []
    for _ in range(n):
        perm = rng.permutation(labels)
        a_p, b_p = combined[perm == 0], combined[perm == 1]
        try:
            v = metric_fn(a_p, b_p)
            if np.isfinite(v):
                null.append(v)
        except Exception:
            continue
    null = np.asarray(null) if null else np.array([0.0])
    p = (np.sum(null >= observed) + 1) / (len(null) + 1)
    return float(observed), float(p), null


# ----------------------------------------------------------------------
# Per-verb computation
# ----------------------------------------------------------------------
verbs = df["verb"].unique().tolist()
records = []
per_verb = {}

print("\nComputing per-verb metrics (bootstrap + permutation)...")
for verb in verbs:
    sub = df[df["verb"] == verb]
    main = np.stack(sub[sub["usage"] == "main"]["embedding"].values)
    light = np.stack(sub[sub["usage"] == "light"]["embedding"].values)

    if len(main) < 2 or len(light) < 2:
        print(f"  Skipping {verb!r}: need ≥2 of each (main={len(main)}, light={len(light)})")
        continue

    print(f"  · {verb}  main={len(main):<3d} light={len(light):<3d}")

    cos_d = m_cosine_distance(main, light)
    cos_lo, cos_hi = bootstrap_ci(main, light, m_cosine_distance,
                                  n=args.n_bootstrap, rng=rng)

    d_val = m_cohens_d_lda(main, light)
    d_lo, d_hi = bootstrap_ci(main, light, m_cohens_d_lda,
                              n=args.n_bootstrap, rng=rng)

    mah = m_mahalanobis(main, light)
    mah_lo, mah_hi = bootstrap_ci(main, light, m_mahalanobis,
                                  n=args.n_bootstrap, rng=rng)

    aucv = m_probe_auc(main, light, seed=args.seed)
    auc_lo, auc_hi = bootstrap_ci(
        main, light,
        lambda A, B: m_probe_auc(A, B, seed=args.seed),
        n=args.n_bootstrap, rng=rng,
    )

    _, perm_p, perm_null = permutation_test(
        main, light, m_cosine_distance,
        n=args.n_permutations, rng=rng,
    )

    records.append({
        "verb": verb,
        "n_main": len(main), "n_light": len(light),
        "cosine_distance": cos_d, "cosine_ci_lo": cos_lo, "cosine_ci_hi": cos_hi,
        "cohens_d": d_val, "cohens_d_ci_lo": d_lo, "cohens_d_ci_hi": d_hi,
        "mahalanobis": mah, "mahalanobis_ci_lo": mah_lo, "mahalanobis_ci_hi": mah_hi,
        "probe_auc": aucv, "probe_auc_ci_lo": auc_lo, "probe_auc_ci_hi": auc_hi,
        "perm_pvalue_cosine": perm_p,
    })
    per_verb[verb] = {
        "main": main, "light": light,
        "perm_null": perm_null, "perm_obs": cos_d,
    }

results_df = pd.DataFrame(records).sort_values("cohens_d", ascending=False)
csv_path = out_dir / "master_table.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nMaster table → {csv_path}")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------
def grid_shape(n: int, ncols: int = 3):
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


# ----------------------------------------------------------------------
# Plot 1 — Forest plot of Cohen's d
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, max(4, 0.7 * len(results_df))))
y_pos = np.arange(len(results_df))
d_vals = results_df["cohens_d"].values
err_lo = d_vals - results_df["cohens_d_ci_lo"].values
err_hi = results_df["cohens_d_ci_hi"].values - d_vals

ax.errorbar(d_vals, y_pos, xerr=[err_lo, err_hi],
            fmt="o", color="steelblue", capsize=4, markersize=9, linewidth=1.5)
for thr, ls, lbl in [(0.2, ":", "small (0.2)"),
                     (0.5, "--", "medium (0.5)"),
                     (0.8, "-", "large (0.8)")]:
    ax.axvline(thr, color="gray", linestyle=ls, alpha=0.6, label=lbl)

ax.set_yticks(y_pos)
ax.set_yticklabels([verb_label(v) for v in results_df["verb"]],
                   fontproperties=urdu_font, fontsize=13)
ax.invert_yaxis()
ax.set_xlabel("Cohen's d  (on LDA axis)")
ax.set_title("Main vs Light separability — Cohen's d with 95% bootstrap CI\n"
             "(ranked, larger = more separable)")
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "01_forest_cohens_d.png", dpi=300)
plt.close()
print("Saved: 01_forest_cohens_d.png")


# ----------------------------------------------------------------------
# Plot 2 — LDA 1D KDE projection per verb
# ----------------------------------------------------------------------
def overlap_coefficient(z1, z2, n_grid=400):
    lo = min(z1.min(), z2.min())
    hi = max(z1.max(), z2.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    try:
        k1 = gaussian_kde(z1)(grid)
        k2 = gaussian_kde(z2)(grid)
    except Exception:
        return float("nan")
    return float(np.trapz(np.minimum(k1, k2), grid))


n_verbs = len(per_verb)
nrows, ncols = grid_shape(n_verbs, ncols=3)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)

for ax, (verb, dat) in zip(axes.flat, per_verb.items()):
    main, light = dat["main"], dat["light"]
    Xa = np.vstack([main, light])
    ya = np.array([0] * len(main) + [1] * len(light))
    lda = LinearDiscriminantAnalysis(n_components=1, solver="svd")
    z = lda.fit_transform(Xa, ya).ravel()
    zm, zl = z[ya == 0], z[ya == 1]

    pad = 0.5 * (z.max() - z.min() + 1e-9) * 0.1
    grid = np.linspace(z.min() - pad, z.max() + pad, 400)
    km = gaussian_kde(zm)(grid)
    kl = gaussian_kde(zl)(grid)

    ax.fill_between(grid, km, alpha=0.5, color="steelblue", label="main")
    ax.fill_between(grid, kl, alpha=0.5, color="firebrick", label="light")
    ax.plot(grid, np.minimum(km, kl), color="purple", lw=1, alpha=0.7)

    ovl = overlap_coefficient(zm, zl)
    ax.set_title(f"{verb_label(verb)}    overlap = {ovl:.2f}",
                 fontproperties=urdu_font, fontsize=12)
    ax.set_xlabel("LDA axis"); ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper right")

for ax in axes.flat[n_verbs:]:
    ax.axis("off")

fig.suptitle("LDA 1D projections — main (blue) vs light (red) with overlap coefficient",
             y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(out_dir / "02_lda_kde.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 02_lda_kde.png")


# ----------------------------------------------------------------------
# Plot 3 — PCA 2D density contours per verb
# ----------------------------------------------------------------------
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
for ax, (verb, dat) in zip(axes.flat, per_verb.items()):
    main, light = dat["main"], dat["light"]
    Xa = np.vstack([main, light])
    pca = PCA(n_components=2, random_state=args.seed)
    coords = pca.fit_transform(Xa)
    cm, cl = coords[:len(main)], coords[len(main):]

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    xg, yg = np.mgrid[x_min - pad_x:x_max + pad_x:100j,
                      y_min - pad_y:y_max + pad_y:100j]
    grid_pts = np.vstack([xg.ravel(), yg.ravel()])

    for pts, color, label in [(cm, "steelblue", "main"),
                              (cl, "firebrick", "light")]:
        ax.scatter(pts[:, 0], pts[:, 1], color=color, alpha=0.35,
                   s=18, edgecolor="none", label=label)
        if len(pts) >= 3:
            try:
                kde = gaussian_kde(pts.T)
                zg = kde(grid_pts).reshape(xg.shape)
                ax.contour(xg, yg, zg, levels=5, colors=color,
                           alpha=0.8, linewidths=1.2)
            except Exception:
                pass

    ax.set_title(verb_label(verb), fontproperties=urdu_font, fontsize=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8, loc="best")

for ax in axes.flat[n_verbs:]:
    ax.axis("off")

fig.suptitle("PCA 2D density contours — main (blue) vs light (red)",
             y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(out_dir / "03_pca_contours.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 03_pca_contours.png")


# ----------------------------------------------------------------------
# Plot 4 — Within-vs-between cosine similarity violins
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(max(8, 1.6 * n_verbs), 5.5))
positions, data_list, colors_list, xtick_pos, xtick_lbl = [], [], [], [], []

for i, (verb, dat) in enumerate(per_verb.items()):
    main, light = dat["main"], dat["light"]
    sm = cosine_similarity(main); within_m = sm[np.triu_indices_from(sm, k=1)]
    sl = cosine_similarity(light); within_l = sl[np.triu_indices_from(sl, k=1)]
    between = cosine_similarity(main, light).ravel()

    base = i * 4
    positions += [base, base + 1, base + 2]
    data_list += [within_m, within_l, between]
    colors_list += ["steelblue", "firebrick", "gray"]
    xtick_pos.append(base + 1)
    xtick_lbl.append(verb_label(verb))

vp = ax.violinplot(data_list, positions=positions, widths=0.85,
                   showmeans=True, showextrema=False)
for body, c in zip(vp["bodies"], colors_list):
    body.set_facecolor(c); body.set_alpha(0.6); body.set_edgecolor("black")
if "cmeans" in vp:
    vp["cmeans"].set_color("black")

ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lbl, fontproperties=urdu_font, fontsize=12)
ax.set_ylabel("Cosine similarity")
ax.set_title("Within-class vs between-class cosine similarity")
ax.grid(axis="y", alpha=0.3)
ax.legend(handles=[
    mpatches.Patch(color="steelblue", alpha=0.6, label="within main"),
    mpatches.Patch(color="firebrick", alpha=0.6, label="within light"),
    mpatches.Patch(color="gray", alpha=0.6, label="between main↔light"),
], loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / "04_within_between_violin.png", dpi=300)
plt.close()
print("Saved: 04_within_between_violin.png")


# ----------------------------------------------------------------------
# Plot 5 — Probe ROC curves
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 7.5))
colormap = plt.cm.tab10(np.linspace(0, 1, max(10, n_verbs)))

for i, (verb, dat) in enumerate(per_verb.items()):
    main, light = dat["main"], dat["light"]
    Xa = np.vstack([main, light])
    ya = np.array([0] * len(main) + [1] * len(light))
    n_splits = min(5, len(main), len(light))
    if n_splits < 2:
        continue
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    for tr, te in skf.split(Xa, ya):
        if len(np.unique(ya[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000).fit(Xa[tr], ya[tr])
        scores = clf.decision_function(Xa[te])
        fpr, tpr, _ = roc_curve(ya[te], scores)
        t = np.interp(mean_fpr, fpr, tpr); t[0] = 0.0
        tprs.append(t); aucs.append(auc(fpr, tpr))
    if not tprs:
        continue
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    ax.plot(mean_fpr, mean_tpr, color=colormap[i], lw=2,
            label=f"{verb_label(verb)}  AUC = {mean_auc:.2f}")

ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="random")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Linear probe ROC curves (5-fold CV)\nmain vs light per verb")
ax.legend(loc="lower right", prop=urdu_font, fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "05_roc_curves.png", dpi=300)
plt.close()
print("Saved: 05_roc_curves.png")


# ----------------------------------------------------------------------
# Plot 6 — Permutation null histograms
# ----------------------------------------------------------------------
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
for ax, (verb, dat) in zip(axes.flat, per_verb.items()):
    null = dat["perm_null"]; obs = dat["perm_obs"]
    ax.hist(null, bins=30, color="lightgray", edgecolor="gray")
    ax.axvline(obs, color="crimson", lw=2.5, label=f"observed = {obs:.3f}")
    p = (np.sum(null >= obs) + 1) / (len(null) + 1)
    ax.set_title(f"{verb_label(verb)}     p = {p:.3f}",
                 fontproperties=urdu_font, fontsize=12)
    ax.set_xlabel("cosine distance (shuffled labels)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

for ax in axes.flat[n_verbs:]:
    ax.axis("off")

fig.suptitle("Permutation null distributions — is observed > random?",
             y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(out_dir / "06_permutation_null.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 06_permutation_null.png")


# ----------------------------------------------------------------------
# Auto conclusion summary
# ----------------------------------------------------------------------
def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "ns "


ranked = results_df.sort_values("cohens_d", ascending=False).reset_index(drop=True)
lines = []
lines.append("=" * 78)
lines.append(" OVERALL CONCLUSION  —  Urdu main vs light verb separability")
lines.append("=" * 78)
lines.append(f"  Verbs analysed : {len(ranked)}")
lines.append(f"  Examples (n)   : {len(df)}")
lines.append(f"  Bootstrap reps : {args.n_bootstrap}")
lines.append(f"  Permutations   : {args.n_permutations}")
lines.append("")
lines.append(" Verbs ranked by Cohen's d (effect size on LDA axis):")
lines.append("")
lines.append(f"  {'#':>2}  {'verb':<10}  {'Cohen d':>8}  {'95% CI':<18}  "
             f"{'AUC':>5}  {'p (perm)':>8}  sig")
lines.append("  " + "-" * 70)
for i, row in enumerate(ranked.itertuples(), 1):
    ci = f"[{row.cohens_d_ci_lo:.2f}, {row.cohens_d_ci_hi:.2f}]"
    lines.append(
        f"  {i:>2}. {row.verb:<10}  "
        f"{row.cohens_d:>8.2f}  {ci:<18}  "
        f"{row.probe_auc:>5.2f}  {row.perm_pvalue_cosine:>8.3f}  {stars(row.perm_pvalue_cosine)}"
    )
lines.append("")
lines.append("  Effect sizes:  0.2 small | 0.5 medium | 0.8 large | >1.0 very large")
lines.append("  Probe AUC:     0.50 random | 0.70 good | 0.90 excellent | 1.00 perfect")
lines.append("  Significance:  ***  p<.001    **  p<.01    *  p<.05    ns  not significant")
lines.append("=" * 78)

summary = "\n".join(lines)
print("\n" + summary)
(out_dir / "conclusion.txt").write_text(summary, encoding="utf-8")
print(f"\nSaved: conclusion.txt")
print(f"\nAll outputs written to: {out_dir}/")
