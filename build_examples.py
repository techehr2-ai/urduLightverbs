#!/usr/bin/env python3
"""
build_examples.py

Pipeline to build an `examples.csv`-style file from a raw Urdu text corpus.

Steps
-----
1. Read a UTF-8 text file containing Urdu content.
2. Split it into sentences (using ۔ ؟ ! and newlines as separators).
3. Keep only sentences that
      * end with one of the TARGET VERBS (e.g. دیا، گیا، بیٹھا، اٹھا، پڑا، آیا), AND
      * are <= MAX_WORDS words long (default 20).
4. Write all filtered sentences to a text file (one per line) for manual review.
5. Take the FIRST `--total` sentences (default 1000) as a *balanced mixture*
   of all target verbs (round-robin across verbs).
6. Ask an LLM, for each sentence, whether the target verb is being used as
      * "main"  -> a NOUN appears immediately before the verb
                   (the verb expresses the real action)
      * "light" -> another VERB stem appears immediately before the verb
                   (V1+V2 compound; the target verb only adds aspect)
      * "skip"  -> the sentence is a COMPLEX PREDICATE (noun+verb fused into a
                   new concept, e.g. کام کرنا، فیصلہ کرنا، یاد آنا)
                   OR the structure is ambiguous.
7. Write the result to a CSV with columns `verb, usage, sentence` — the exact
   schema that `lightverbs.py` already consumes.

Usage
-----
    pip install openai
    export OPENAI_API_KEY=sk-...
    python build_examples.py --input-text urdu_corpus.txt \
                             --output-csv examples_llm.csv

Use `--dry-run` to only generate `filtered_sentences.txt` without calling the LLM.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_VERBS = ["دیا", "گیا", "بیٹھا", "اٹھا", "پڑا", "آیا"]
DEFAULT_MAX_WORDS = 20
DEFAULT_TOTAL = 1000
DEFAULT_FILTERED_TXT = "filtered_sentences.txt"
DEFAULT_OUTPUT_CSV = "examples_llm.csv"
DEFAULT_MODEL = "gpt-4o-mini"

# Urdu sentence terminators
SENT_SPLIT_RE = re.compile(r"[۔؟!\n]+")

# Punctuation to strip from the trailing token before comparing it to a verb
TRAILING_PUNCT = "۔؟!.,،؛;:\"'()[]{}…«»“”‘’"


# ---------------------------------------------------------------------------
# Step 1 & 2: read text and filter sentences ending with a target verb
# ---------------------------------------------------------------------------

def iter_sentences(text: str) -> Iterable[str]:
    """Yield non-empty sentences from raw text using Urdu/Latin terminators."""
    for chunk in SENT_SPLIT_RE.split(text):
        s = chunk.strip()
        if s:
            yield s


def last_word(sentence: str) -> str:
    """Return the last whitespace-separated token, stripped of trailing punctuation."""
    tokens = sentence.split()
    if not tokens:
        return ""
    return tokens[-1].strip(TRAILING_PUNCT)


def filter_sentences(
    text: str,
    verbs: List[str],
    max_words: int,
) -> List[Tuple[str, str]]:
    """
    Return a list of (verb, sentence) tuples where:
      * sentence has 1..max_words whitespace-separated words
      * sentence's final word matches one of the target verbs
    """
    verb_set = set(verbs)
    results: List[Tuple[str, str]] = []
    for sentence in iter_sentences(text):
        words = sentence.split()
        if not (1 <= len(words) <= max_words):
            continue
        lw = last_word(sentence)
        if lw in verb_set:
            # Normalise inner whitespace and use the stripped last word for safety.
            normalised = " ".join(words[:-1] + [lw]) if words else sentence
            results.append((lw, normalised))
    return results


# ---------------------------------------------------------------------------
# Step 5: balanced sample of N sentences across all verbs
# ---------------------------------------------------------------------------

def balanced_sample(
    pairs: List[Tuple[str, str]],
    total: int,
) -> List[Tuple[str, str]]:
    """
    Interleave examples so the first `total` items are a mixture of all verbs
    (round-robin). If one verb runs out, keep cycling through the others.
    """
    buckets: Dict[str, deque] = defaultdict(deque)
    for verb, sent in pairs:
        buckets[verb].append(sent)

    out: List[Tuple[str, str]] = []
    order = list(buckets.keys())
    while len(out) < total and any(buckets[v] for v in order):
        for v in order:
            if not buckets[v]:
                continue
            out.append((v, buckets[v].popleft()))
            if len(out) >= total:
                break
    return out


# ---------------------------------------------------------------------------
# Step 6: ask an LLM whether the verb is main / light / skip
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """You are an expert in Urdu linguistics analysing verb usage.

The TARGET VERB appears at the END of the sentence below.

  target verb : {verb}
  sentence    : {sentence}

Classify the role of the target verb "{verb}" into EXACTLY ONE label:

1) main  — The word IMMEDIATELY BEFORE "{verb}" is a NOUN (or a noun phrase),
           and "{verb}" expresses the real, concrete action.
   Examples:
     • اس نے مجھے قلم دیا        (قلم = noun -> دیا is main)
     • وہ کل لاہور گیا            (لاہور = noun -> گیا is main)
     • وہ کرسی پر بیٹھا           (کرسی پر = noun phrase -> بیٹھا is main)

2) light — The word IMMEDIATELY BEFORE "{verb}" is ANOTHER VERB stem
           (a V1+V2 compound), and "{verb}" only adds aspect / completion /
           suddenness. It does NOT carry the main lexical meaning.
   Examples:
     • وہ اچانک ہنس پڑا            (ہنس = verb -> پڑا is light)
     • بچہ سو گیا                   (سو = verb -> گیا is light)
     • وہ راز بتا بیٹھا              (بتا = verb -> بیٹھا is light)

3) skip  — The sentence uses a COMPLEX PREDICATE: a NOUN and a VERB fused
           together to form a new concept (e.g. کام کرنا، فیصلہ کرنا،
           یاد آنا، نظر آنا، شروع ہونا). Also use "skip" when the structure is
           ambiguous, or when "{verb}" is only an auxiliary / tense marker.

Respond with ONLY one word on a single line: main, light, or skip.
"""


def classify_with_llm(client, model: str, verb: str, sentence: str) -> str:
    """Call the LLM and return one of 'main', 'light', 'skip'."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=4,
        messages=[
            {
                "role": "system",
                "content": "You answer with exactly one word: main, light, or skip.",
            },
            {
                "role": "user",
                "content": CLASSIFY_PROMPT.format(verb=verb, sentence=sentence),
            },
        ],
    )
    raw = (resp.choices[0].message.content or "").strip().lower()
    if not raw:
        return "skip"
    label = raw.split()[0].strip(".,:;\"'()[]")
    if label not in {"main", "light", "skip"}:
        return "skip"
    return label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a verb/usage/sentence CSV from a raw Urdu text file."
    )
    parser.add_argument(
        "--input-text", "-i", required=True,
        help="Path to a UTF-8 text file containing Urdu content.",
    )
    parser.add_argument(
        "--verbs", default=",".join(DEFAULT_VERBS),
        help=f"Comma-separated list of target verbs "
             f"(default: {','.join(DEFAULT_VERBS)}).",
    )
    parser.add_argument(
        "--max-words", type=int, default=DEFAULT_MAX_WORDS,
        help=f"Maximum sentence length in words (default: {DEFAULT_MAX_WORDS}).",
    )
    parser.add_argument(
        "--total", type=int, default=DEFAULT_TOTAL,
        help=f"How many sentences to classify with the LLM "
             f"(default: {DEFAULT_TOTAL}).",
    )
    parser.add_argument(
        "--filtered-output", default=DEFAULT_FILTERED_TXT,
        help=f"Where to write the filtered sentences "
             f"(default: {DEFAULT_FILTERED_TXT}).",
    )
    parser.add_argument(
        "--output-csv", "-o", default=DEFAULT_OUTPUT_CSV,
        help=f"Where to write the classified CSV (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--api-key", default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (defaults to $OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--api-base", default=os.getenv("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible base URL "
             "(e.g. for Azure / local llama.cpp / Ollama).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip the LLM step; only build the filtered sentences file.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="Seconds to sleep between LLM calls (simple rate limiting).",
    )
    parser.add_argument(
        "--keep-skipped", action="store_true",
        help="Also write 'skip' rows to the CSV (default: drop them).",
    )

    args = parser.parse_args()

    verbs = [v.strip() for v in args.verbs.split(",") if v.strip()]
    if not verbs:
        print("No target verbs provided.", file=sys.stderr)
        return 2

    # ---- Steps 1+2+3: read & filter ----
    text = Path(args.input_text).read_text(encoding="utf-8")
    pairs = filter_sentences(text, verbs, args.max_words)

    per_verb: Dict[str, int] = defaultdict(int)
    for v, _ in pairs:
        per_verb[v] += 1

    print(f"Filtered {len(pairs)} sentences "
          f"(<= {args.max_words} words, ending in a target verb).")
    for v in verbs:
        print(f"  {v}: {per_verb.get(v, 0)}")

    # ---- Step 4: write filtered text file ----
    Path(args.filtered_output).write_text(
        "\n".join(s for _, s in pairs) + ("\n" if pairs else ""),
        encoding="utf-8",
    )
    print(f"Wrote {args.filtered_output}")

    # ---- Step 5: balanced first N ----
    sample = balanced_sample(pairs, args.total)
    print(f"Selected {len(sample)} sentences for LLM classification "
          f"(target {args.total}).")

    if args.dry_run:
        print("Dry run: skipping LLM classification.")
        return 0

    # ---- Step 6: LLM ----
    try:
        from openai import OpenAI
    except ImportError:
        print("Please `pip install openai` to use the LLM step "
              "(or pass --dry-run).", file=sys.stderr)
        return 1

    client_kwargs = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    client = OpenAI(**client_kwargs)

    # ---- Step 7: write CSV incrementally so partial runs are not lost ----
    out_path = Path(args.output_csv)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["verb", "usage", "sentence"])

        kept = 0
        skipped = 0
        for i, (verb, sentence) in enumerate(sample, start=1):
            try:
                label = classify_with_llm(client, args.model, verb, sentence)
            except Exception as exc:  # noqa: BLE001 - log and continue
                print(f"[{i}/{len(sample)}] LLM error on "
                      f"'{sentence}': {exc}", file=sys.stderr)
                label = "skip"

            if label == "skip" and not args.keep_skipped:
                skipped += 1
            else:
                writer.writerow([verb, label, sentence])
                fh.flush()
                kept += 1

            if i % 25 == 0 or i == len(sample):
                print(f"  [{i}/{len(sample)}] kept={kept} skipped={skipped}")

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote {kept} rows to {args.output_csv} (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
