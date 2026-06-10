# urduLightverbs

Tools for studying Urdu **main verb vs. light verb** uses with BERT-family
embeddings.

## Files

| File | Purpose |
|------|---------|
| `build_examples.py` | Build an `examples.csv`-style file from a raw Urdu text corpus by (a) filtering sentences that end in a target verb and are ≤ 20 words long, (b) taking a balanced first-1000 mix across all target verbs, and (c) asking an LLM whether each occurrence is a *main* verb, a *light* verb, or a *complex predicate* that should be skipped. |
| `lightverbs.py` | Load `examples.csv`, embed each sentence with an Urdu/multilingual BERT model, and produce: (i) a per-verb cosine-distance CSV, (ii) a per-verb PCA grid, (iii) a per-verb cosine-distance + example-count bar chart, and (iv) optionally a single global PCA across every verb. |
| `visualize_results.py` | Re-render the per-verb separation + example-count bar chart from the results CSV alone, without re-running BERT. Useful for iterating on the chart layout. |
| `visualize_advanced.py` | Build a single **interactive HTML report** combining (1) summary stats, (2) global UMAP, (3) per-verb UMAP grid, (4) per-verb similarity histograms, (5) sentence-by-sentence similarity heatmap and (6) a logistic-regression probe that flags label mistakes. Hover any UMAP point to see the original Urdu sentence. Embeddings are cached on disk so re-runs are instant. |
| `examples.csv` | Hand-curated seed corpus with columns `verb,usage,sentence` (`usage` ∈ {`main`, `light`}). |

## Pipeline

```text
raw Urdu .txt
   │
   │  build_examples.py
   ▼
filtered_sentences.txt              (≤ 20 words, ends in target verb)
   │
   │  LLM classification (main / light / skip)
   ▼
examples_llm.csv                    (verb, usage, sentence)
   │
   ├── lightverbs.py ──────────── urdu_mbert_lightverb_results.csv
   │                                  urdu_mbert_lightverb_pca.png
   │                                  urdu_mbert_lightverb_bars.png
   │
   │  (the same .csv can also feed:)
   │
   ├── visualize_results.py ───── urdu_mbert_lightverb_bars.png  (no BERT needed)
   │
   └── visualize_advanced.py ──── urdu_lightverb_report.html
                                      embeddings_cache.npz (signature-checked cache)
```

## Quick start

```bash
pip install openai pandas numpy scikit-learn matplotlib \
            torch transformers arabic-reshaper python-bidi \
            umap-learn plotly

# 1. Build a CSV from a raw Urdu corpus
export OPENAI_API_KEY=sk-...
python build_examples.py \
    --input-text urdu_corpus.txt \
    --output-csv examples_llm.csv

# 2. Run the embedding analysis on the resulting CSV (matplotlib charts)
python lightverbs.py --input examples_llm.csv

# 3. Build a full interactive HTML report (UMAP, histograms, heatmap, probe)
python visualize_advanced.py \
    --input examples_llm.csv \
    --output urdu_lightverb_report.html

# 4. Re-render just the bar chart from the existing results CSV (no BERT)
python visualize_results.py
```

## `build_examples.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--input-text`, `-i` | *(required)* | UTF-8 text file containing Urdu content. |
| `--verbs` | `دیا,گیا,بیٹھا,اٹھا,پڑا,آیا` | Comma-separated target verbs. |
| `--max-words` | `20` | Maximum sentence length (in whitespace-separated words). |
| `--total` | `1000` | How many sentences to send to the LLM (balanced across verbs). |
| `--filtered-output` | `filtered_sentences.txt` | Where to dump every filtered sentence. |
| `--output-csv`, `-o` | `examples_llm.csv` | Where to write the classified CSV. |
| `--model` | `gpt-4o-mini` | OpenAI-compatible model name. |
| `--api-key` | `$OPENAI_API_KEY` | API key. |
| `--api-base` | `$OPENAI_BASE_URL` | Optional base URL (Azure / local OpenAI-compatible server). |
| `--dry-run` | `False` | Only build the filtered text file; skip all LLM calls. |
| `--sleep` | `0.0` | Seconds to sleep between LLM calls (basic rate limiting). |
| `--keep-skipped` | `False` | Also write rows the LLM tagged as `skip`. |

## `lightverbs.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | `examples.csv` | Input CSV with columns `verb,usage,sentence`. |
| `--results-csv` | `urdu_mbert_lightverb_results.csv` | Per-verb cosine-distance results CSV. |
| `--pca-png` | `urdu_mbert_lightverb_pca.png` | Per-verb PCA grid (one subplot per verb). |
| `--bar-png` | `urdu_mbert_lightverb_bars.png` | Per-verb cosine-distance + example-count bar chart. |
| `--global-pca-png` | *(off)* | If set, also save a single global PCA across all verbs to this path. |

## `visualize_advanced.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | `examples.csv` | Input CSV with columns `verb,usage,sentence`. |
| `--output`, `-o` | `urdu_lightverb_report.html` | Single self-contained HTML report. |
| `--embeddings-cache` | `embeddings_cache.npz` | On-disk cache of embeddings, keyed by an MD5 of the CSV content. As long as the CSV does not change, BERT runs only once. |
| `--model` | same as `lightverbs.py` | BERT model name or local path. |

The report contains six sections:

1. **Summary table** — per-verb counts plus the cosine distance between
   main- and light-verb centroids.
2. **Global UMAP** — all verbs in one interactive scatter (Plotly).
   Colour = verb, shape = main/light. Hover any point to read the Urdu
   sentence.
3. **Per-verb UMAP grid** — one UMAP per verb, refit on just that verb’s
   embeddings so the layout shows main vs light structure within a single
   verb. Hover to read the sentence; outliers are often label mistakes.
4. **Per-verb similarity histograms** — pairwise cosine similarities within
   main (blue), within light (red) and between main and light (grey). When a
   verb separates cleanly, the grey distribution sits to the *left* of the
   two within-group distributions.
5. **Similarity matrix heatmap** — N×N cosine similarity between every
   sentence, sorted by verb and then usage. A strong block-diagonal pattern
   means embeddings cluster by verb; sub-blocks within a verb mean main and
   light are also separable.
6. **Logistic-regression probe** — per verb, trains a logistic regression on
   the BERT embeddings to predict main vs light using 5-fold stratified
   cross-validation. Reports accuracy and F1 versus the majority-class
   baseline, and lists every sentence the probe disagrees with — those are
   the best candidates for re-labelling or removal.

## Classification rules used in the LLM prompt

* **main** – the word **immediately before** the target verb is a **noun**, and
  the target verb expresses the real, concrete action.
  *e.g.* `اس نے مجھے قلم دیا` → `دیا` is **main** (قلم is a noun).
* **light** – the word **immediately before** the target verb is **another
  verb stem** (V1 + V2 compound); the target verb only adds aspect / completion
  / suddenness.
  *e.g.* `وہ اچانک ہنس پڑا` → `پڑا` is **light** (ہنس is a verb).
* **skip** – the construction is a **complex predicate** where a noun + verb
  fuse to form a new concept (e.g. `کام کرنا`, `فیصلہ کرنا`, `یاد آنا`,
  `نظر آنا`), or the structure is ambiguous. These rows are dropped from the
  CSV by default.
