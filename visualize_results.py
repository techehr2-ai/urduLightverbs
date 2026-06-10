#!/usr/bin/env python3
"""
visualize_results.py

Standalone visualisation for a pre-computed `urdu_mbert_lightverb_results.csv`
(produced by lightverbs.py). Useful when you want to iterate on the chart
layout *without* re-running BERT.

Expected input columns:
    verb, main_examples, light_examples, cosine_similarity, cosine_distance

Produces a single PNG with two side-by-side panels that share the y-axis:

    Left panel  : per-verb cosine distance between main- and light-verb
                  centroids. Each verb gets its own colour from tab10 so
                  every bar is visually distinct.
    Right panel : grouped bars showing the number of `main` (blue) and
                  `light` (red) examples for each verb. Lets you tell at a
                  glance whether a low/high cosine distance is supported by
                  lots of examples or only a few.

Usage
-----
    python visualize_results.py
    python visualize_results.py --input urdu_mbert_lightverb_results.csv \
                                --output urdu_mbert_lightverb_bars.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional Urdu shaping - if not available, fall back to raw text.
try:
    from matplotlib import font_manager as fm
    from bidi.algorithm import get_display
    import arabic_reshaper

    URDU_FONT_PATH = "./NotoSansArabic-Regular.ttf"
    try:
        urdu_font = fm.FontProperties(fname=URDU_FONT_PATH)
    except Exception:
        urdu_font = None

    def reshape_urdu(text: str) -> str:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
except ImportError:
    urdu_font = None

    def reshape_urdu(text: str) -> str:
        return text


# Vivid, high-contrast colours so red is unmistakable on screen and in PNG.
COLOR_MAIN  = "#2563eb"   # vivid blue
COLOR_LIGHT = "#dc2626"   # vivid red


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualise a pre-computed urdu_mbert_lightverb_results.csv."
    )
    parser.add_argument(
        "--input", "-i",
        default="urdu_mbert_lightverb_results.csv",
        help="Path to the results CSV (default: urdu_mbert_lightverb_results.csv).",
    )
    parser.add_argument(
        "--output", "-o",
        default="urdu_mbert_lightverb_bars.png",
        help="Where to save the bar chart PNG.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not call plt.show() (useful in headless environments).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = {"verb", "main_examples", "light_examples", "cosine_distance"}
    missing = required - set(df.columns)
    if missing:
        print(f"Input CSV is missing required columns: {sorted(missing)}.",
              file=sys.stderr)
        return 2

    df = df.sort_values("cosine_distance", ascending=False).reset_index(drop=True)
    n_verbs = len(df)
    if n_verbs == 0:
        print("No rows in input CSV.", file=sys.stderr)
        return 1

    # Per-verb stable colours.
    cmap = plt.get_cmap("tab10" if n_verbs <= 10 else "tab20")
    verb_colors = [cmap(i % cmap.N) for i in range(n_verbs)]

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(15, max(4.0, 0.7 * n_verbs + 1.5)),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 2]},
    )

    y_pos = np.arange(n_verbs)
    verb_labels = [reshape_urdu(v) for v in df["verb"]]

    # ---- Left panel: cosine distance ----
    ax1.barh(
        y_pos,
        df["cosine_distance"],
        color=verb_colors,
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
    for lbl, c in zip(ax1.get_yticklabels(), verb_colors):
        lbl.set_color(c)
        lbl.set_fontweight("bold")

    ax1.invert_yaxis()
    ax1.set_xlabel("Cosine distance between main and light centroids")
    ax1.set_title("Per-Verb Main vs Light Separation", fontsize=13)
    x_max = df["cosine_distance"].max()
    for i, val in enumerate(df["cosine_distance"]):
        ax1.text(
            val + x_max * 0.01, i, f"{val:.3f}",
            va="center", fontsize=10, fontweight="bold",
        )
    ax1.set_xlim(0, x_max * 1.18)
    ax1.grid(True, axis="x", alpha=0.3)

    # ---- Right panel: example counts (grouped main vs light) ----
    bar_h = 0.38
    ax2.barh(
        y_pos - bar_h / 2,
        df["main_examples"],
        height=bar_h,
        color=COLOR_MAIN,
        edgecolor="black",
        linewidth=0.6,
        label="main (n)",
    )
    ax2.barh(
        y_pos + bar_h / 2,
        df["light_examples"],
        height=bar_h,
        color=COLOR_LIGHT,
        edgecolor="black",
        linewidth=0.6,
        label="light (n)",
    )
    ax2.set_xlabel("Number of examples")
    ax2.set_title("Example Counts per Verb", fontsize=13)
    ax2.legend(loc="lower right", fontsize=11, frameon=True)
    ax2.grid(True, axis="x", alpha=0.3)

    count_max = max(df["main_examples"].max(), df["light_examples"].max())
    for i, row in df.iterrows():
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
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
