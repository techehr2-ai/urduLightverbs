# urduLightverbs

Tools for studying Urdu **main verb vs. light verb** uses with BERT-family
embeddings.

## Files

| File | Purpose |
|------|---------|
| `build_examples.py` | Build an `examples.csv`-style file from a raw Urdu text corpus by (a) filtering sentences that end in a target verb and are ≤ 20 words long, (b) taking a balanced first-1000 mix across all target verbs, and (c) asking an LLM whether each occurrence is a *main* verb, a *light* verb, or a *complex predicate* that should be skipped. |
| `lightverbs.py` | Load `examples.csv`, embed each sentence with an Urdu/multilingual BERT model, and produce: (i) a per-verb cosine-distance CSV, (ii) a **per-verb PCA grid** (one subplot per verb, main vs. light shown in different colours + centroid stars), (iii) a **cosine-distance bar chart** across all verbs, and (iv) optionally a single global PCA across every verb. |
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
   │  lightverbs.py
   ▼
urdu_mbert_lightverb_results.csv    (per-verb cosine distances)
urdu_mbert_lightverb_pca.png        (per-verb PCA grid — main vs light)
urdu_mbert_lightverb_bars.png       (per-verb separation bar chart)
```

## Quick start

```bash
pip install openai pandas numpy scikit-learn matplotlib \
            torch transformers arabic-reshaper python-bidi

# 1. Build a CSV from a raw Urdu corpus
export OPENAI_API_KEY=sk-...
python build_examples.py \
    --input-text urdu_corpus.txt \
    --output-csv examples_llm.csv

# 2. Run the embedding analysis on the resulting CSV
python lightverbs.py --input examples_llm.csv

# Optional: also write a single global PCA across all verbs
python lightverbs.py --input examples_llm.csv \
                    --global-pca-png urdu_mbert_lightverb_global_pca.png
```

### `build_examples.py` options

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

### `lightverbs.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | `examples.csv` | Input CSV with columns `verb,usage,sentence`. |
| `--results-csv` | `urdu_mbert_lightverb_results.csv` | Per-verb cosine-distance results CSV. |
| `--pca-png` | `urdu_mbert_lightverb_pca.png` | Per-verb PCA grid (one subplot per verb). |
| `--bar-png` | `urdu_mbert_lightverb_bars.png` | Per-verb cosine-distance bar chart. |
| `--global-pca-png` | *(off)* | If set, also save a single global PCA across all verbs to this path. |

### Visualisations

* **Per-verb PCA grid** (`--pca-png`) — a grid of subplots, one per verb. Each
  subplot fits PCA on just that verb's embeddings, so the layout is not
  dominated by inter-verb variance. *Main* uses are drawn as blue circles,
  *light* uses as red X's, with a large star marking each group's centroid.
  The subplot title shows the verb and its main-vs-light cosine distance.
* **Cosine-distance bar chart** (`--bar-png`) — one horizontal bar per verb
  showing how separated the main and light centroids are in embedding space,
  annotated with the number of *main* and *light* examples per verb.
* **Global PCA** (`--global-pca-png`, opt-in) — a single scatter across all
  verbs (colour by verb, shape by usage). Useful for a high-level overview
  but can be cluttered when there are many verbs.

### Classification rules used in the LLM prompt

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
