"""
Build character-trigram language profiles from HistCorp year_to_doc files.

Input:  one or more files named <language>_year_to_doc.json, each shaped
        {"1500-1600": {"document.txt": "text ...", ...}, "1600-1700": {...}, ...}
        (the centuries present can differ between languages).
Output: one <code>.json.gz profile per language, used by the scanner to judge
        whether a decoded output reads like a language, and which one.

Decisions:
- One "all" profile per language. To stop a century with lots of text (e.g. a
  30M-character 1600s collection) from dominating, each century contributes at
  most --max-chars-per-century characters, taken from whole documents in random
  (seeded) order. Centuries with less text contribute everything they have.
- Text is NFKC-normalised and lowercased; every non-letter (punctuation, digits,
  symbols) becomes a word gap. Two variants are counted:
    "spaced":   letters and single spaces between words, used when a key encodes word gaps;
    "unspaced": letters only, used when it does not.
- A small held-out sample (from documents not used for counting) is stored so
  the scanner can measure how well real text of the language scores.

Usage:
    python build_language_profiles.py path/to/swedish_year_to_doc.json path/to/german_year_to_doc.json ...
    python build_language_profiles.py path/to/folder_with_json_files --out language_profiles
Then copy the .json.gz files into  src/keybridge/language_profiles/  in the repository.
"""
import argparse
import datetime
import glob
import gzip
import json
import os
import random
import re
import unicodedata
from collections import Counter

LANGUAGES = {
    "czech": ("cz", "Czech"), "dutch": ("nl", "Dutch"), "english": ("en", "English"),
    "french": ("fr", "French"), "german": ("de", "German"), "greek": ("el", "Greek"),
    "hungarian": ("hu", "Hungarian"), "icelandic": ("is", "Icelandic"), "italian": ("it", "Italian"),
    "latin": ("la", "Latin"), "polish": ("pl", "Polish"), "russian": ("ru", "Russian"),
    "slovene": ("sl", "Slovene"), "slovenian": ("sl", "Slovene"), "spanish": ("es", "Spanish"),
    "swedish": ("sv", "Swedish"),
}

NON_LETTERS = re.compile(r"[^\w]|[\d_]")


def normalise(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = NON_LETTERS.sub(" ", text)
    return " ".join(text.split())


def count_trigrams(texts, pad):
    counts = Counter()
    for t in texts:
        padded = f" {t} " if pad else t
        for i in range(len(padded) - 2):
            counts[padded[i:i + 3]] += 1
    return counts


def build(path, out_dir, max_chars, held_out_chars, seed):
    stem = os.path.basename(path).split("_year_to_doc")[0].lower()
    code, name = LANGUAGES.get(stem, (stem, stem.capitalize()))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)
    per_century_held_out = held_out_chars // max(1, len(data))
    used, held_out, centuries = [], [], {}
    for century in sorted(data):
        docs = [(n, normalise(t)) for n, t in data[century].items()]
        docs = [(n, t) for n, t in docs if t]
        rng.shuffle(docs)
        # With 3+ documents, the first (in shuffled order) is held out and never counted.
        if len(docs) >= 3:
            held_out.append(docs.pop(0)[1][:per_century_held_out])
        taken = 0
        for _, clean in docs:
            if taken >= max_chars:
                break
            piece = clean[:max_chars - taken]
            used.append(piece)
            taken += len(piece)
        centuries[century] = taken
        print(f"  {century}: {taken:,} characters counted from {len(docs)} documents")

    spaced = count_trigrams(used, pad=True)
    unspaced = count_trigrams([t.replace(" ", "") for t in used], pad=False)
    sample = " ".join(held_out)[:held_out_chars]
    profile = {
        "format": "language-profile", "version": 1,
        "language": code, "name": name, "source": "HistCorp",
        "built_from": os.path.basename(path),
        "built": datetime.date.today().isoformat(),
        "max_chars_per_century": max_chars, "seed": seed,
        "centuries": centuries,
        "spaced": {"trigrams": dict(spaced), "held_out": sample},
        "unspaced": {"trigrams": dict(unspaced), "held_out": sample.replace(" ", "")},
    }
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{code}.json.gz")
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False)
    total = sum(centuries.values())
    print(f"  -> {out}  ({total:,} characters counted, {len(spaced):,} distinct trigrams, "
          f"{os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="year_to_doc .json files, or folders containing them")
    ap.add_argument("--out", default="language_profiles", help="output folder (default: language_profiles)")
    ap.add_argument("--max-chars-per-century", type=int, default=3_000_000)
    ap.add_argument("--held-out-chars", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    files = []
    for p in args.inputs:
        files += sorted(glob.glob(os.path.join(p, "*_year_to_doc.json"))) if os.path.isdir(p) else [p]
    if not files:
        raise SystemExit("No *_year_to_doc.json files found.")
    for path in files:
        print(f"Building profile from {os.path.basename(path)}")
        build(path, args.out, args.max_chars_per_century, args.held_out_chars, args.seed)


if __name__ == "__main__":
    main()
