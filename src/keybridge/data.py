"""Turning plaintext into aligned (cipher symbols, plaintext units) training examples."""
import re
from collections import Counter

import torch
from torch.utils.data import Dataset

from .common import (DOUBLE_UNIT, EOS, NULL_UNIT, SOS, SPACE_UNIT, SPECIAL_TOKENS, UNK,
                     read_text_file, say, stop)
from .config import DEFAULT_PLAINTEXT_LANGUAGE, PLAINTEXT_DATASETS, PLAINTEXT_FIELD


# -----------------------------------------------------------------------------
# Plaintext
# -----------------------------------------------------------------------------
def parse_replacements(text):
    """'J=I, V=U' -> {'J': 'I', 'V': 'U'}"""
    out = {}
    for part in re.split(r"[,;]", text or ""):
        if "=" in part:
            a, b = part.split("=", 1)
            a, b = a.strip().upper(), b.strip().upper()
            if a:
                out[a] = b
    return out


def normalise_plaintext(text, replacements):
    text = text.upper()
    for a, b in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(a, b)
    return " ".join(text.split())


def load_plaintext_files(paths, replacements):
    docs = []
    for p in paths:
        docs += [normalise_plaintext(par, replacements)
                 for par in re.split(r"\n\s*\n", read_text_file(p)) if par.strip()]
    return docs


def resolve_plaintext_language(name, log=print):
    """Returns a language name from PLAINTEXT_DATASETS; English if empty or unknown."""
    if not name or not str(name).strip():
        return DEFAULT_PLAINTEXT_LANGUAGE
    for lang in PLAINTEXT_DATASETS:
        if str(name).strip().lower() in (lang.lower(), PLAINTEXT_DATASETS[lang].lower()):
            return lang
    say(log, f"Note: no built-in plaintext for {name!r}, so {DEFAULT_PLAINTEXT_LANGUAGE} is used. "
             f"Available: {', '.join(PLAINTEXT_DATASETS)}.")
    return DEFAULT_PLAINTEXT_LANGUAGE


def load_builtin_plaintext(language, replacements, needed_chars, split="train", log=print):
    """Reads plaintexts from the language's Hugging Face dataset (ds[split]["plaintext"]),
    streaming only as many rows as needed."""
    language = resolve_plaintext_language(language, log)
    repo = PLAINTEXT_DATASETS[language]
    try:
        from datasets import load_dataset
        rows = load_dataset(repo, split=split, streaming=True)
    except Exception as e:
        stop(f"The built-in {language} plaintext ({repo}, split '{split}') could not be loaded "
             f"({type(e).__name__}: {e}).\nCheck your internet connection, or use your own plaintext files.")
    docs, total = [], 0
    for row in rows:
        text = row.get(PLAINTEXT_FIELD) if isinstance(row, dict) else None
        if not text:
            continue
        docs.append(normalise_plaintext(text, replacements))
        total += len(text)
        if total >= needed_chars:
            break
    if not docs:
        stop(f"No plaintext was found in {repo} (split '{split}', column '{PLAINTEXT_FIELD}').")
    return docs


# -----------------------------------------------------------------------------
# Plaintext -> key units
# -----------------------------------------------------------------------------
class Tokeniser:
    """Greedy longest match (the rule ChronoFidelius uses when encrypting), except that
    whole-word units only match whole words, and characters with no key entry are
    reported instead of being copied into the ciphertext."""

    def __init__(self, key):
        self.key = key
        textual = [u for u in key.units if u not in (NULL_UNIT, DOUBLE_UNIT)]
        self.by_first = {}
        for u in sorted(textual, key=len, reverse=True):
            self.by_first.setdefault(u[0], []).append(u)

    def __call__(self, text, uncovered):
        out, i, n = [], 0, len(text)
        while i < n:
            ch = text[i]
            match = None
            for u in self.by_first.get(ch, ()):
                if text.startswith(u, i):
                    if u in self.key.whole_word:
                        if not ((i == 0 or text[i - 1] == " ") and (i + len(u) == n or text[i + len(u)] == " ")):
                            continue
                    match = u
                    break
            if match:
                out.append(match)
                i += len(match)
            elif ch == " ":
                out.append(SPACE_UNIT)   # word gap; dropped later if the key has no <SPACE>
                i += 1
            else:
                uncovered[ch] += 1
                i += 1
        return out


def build_unit_stream(documents, key, rng, word_spaces="keep real word boundaries", null_rate=0.05,
                      double_rate=0.5, rare_unit_rate=0.0, custom_edit=None):
    """Turns plaintext documents into one long list of key units, plus a report."""
    tok = Tokeniser(key)
    uncovered = Counter()
    stream, inner_gaps = [], 0
    for doc in documents:
        units = tok(doc, uncovered)
        if units:
            inner_gaps += units.count(SPACE_UNIT)
            stream.extend(units)
            stream.append(SPACE_UNIT)   # boundary between documents
    counts = Counter(stream)
    report = {"uncovered": uncovered, "characters": sum(len(d) for d in documents)}

    if rare_unit_rate > 0:
        rare = [u for u in key.letter_units if counts[u] < 20]
        report["injected_units"] = rare
        if rare:
            injected, has_gaps = [], SPACE_UNIT in counts
            for u in stream:
                injected.append(u)
                if (u == SPACE_UNIT or not has_gaps) and rng.random() < rare_unit_rate:
                    injected.append(rng.choice(rare))
                    if has_gaps:
                        injected.append(SPACE_UNIT)
            stream = injected

    if (key.has(SPACE_UNIT) and word_spaces == "keep real word boundaries"
            and inner_gaps < 0.01 * max(1, len(stream))):
        word_spaces = "insert at random"
        report["word_spaces_note"] = ("The plaintext has (almost) no word gaps, so gaps were inserted at "
                                      "random instead of keeping real word boundaries.")
    if not key.has(SPACE_UNIT) or word_spaces == "remove":
        stream = [u for u in stream if u != SPACE_UNIT]
    elif word_spaces == "insert at random":
        bare = [u for u in stream if u != SPACE_UNIT]
        stream = []
        for u in bare:
            if stream and stream[-1] != SPACE_UNIT and rng.random() < 0.15:
                stream.append(SPACE_UNIT)
            stream.append(u)
    else:  # keep real boundaries, collapsing repeated gaps
        collapsed = []
        for u in stream:
            if u == SPACE_UNIT and (not collapsed or collapsed[-1] == SPACE_UNIT):
                continue
            collapsed.append(u)
        stream = collapsed

    if key.has(DOUBLE_UNIT):
        out = []
        for u in stream:
            prev = out[-1] if out else None
            if (prev is not None and prev != DOUBLE_UNIT and u == prev and u != SPACE_UNIT
                    and len(u) == 1 and rng.random() < double_rate):
                out.append(DOUBLE_UNIT)
            else:
                out.append(u)
        stream = out

    if key.has(NULL_UNIT):
        out = []
        for u in stream:
            if rng.random() < null_rate:
                out.append(NULL_UNIT)
            out.append(u)
        stream = out

    if custom_edit is not None:
        stream = list(custom_edit(stream))
        bad = sorted({u for u in stream if u not in key.units})
        if bad:
            stop("Your custom edit produced units that are not in the key: " + ", ".join(map(repr, bad[:20])))

    report["unit_counts"] = Counter(stream)
    report["stream_length"] = len(stream)
    return stream, report


def describe_coverage(report, key, log=print):
    """Prints and returns notes on plaintext characters and key units the data does not cover."""
    notes = []
    if report.get("word_spaces_note"):
        notes.append(report["word_spaces_note"])
        say(log, "\n" + notes[-1])
    unc = report["uncovered"]
    if unc:
        share = sum(unc.values()) / max(1, report["characters"])
        letters = ", ".join(f"{c!r} ×{n}" for c, n in unc.most_common(15))
        notes.append(f"{share:.1%} of plaintext characters have no entry in the key and were left out: {letters}")
        say(log, "\n" + notes[-1])
        if share > 0.02:
            say(log, "  If these are letters, map them with letter replacements (e.g. J=I, V=U), "
                     "or add them to the key.")
    counts = report["unit_counts"]
    never = sorted(u for u in key.units if counts[u] == 0)
    rare = sorted(u for u in key.letter_units if 0 < counts[u] < 20)
    if never:
        notes.append("Key units that never occur in the training data, so the model cannot learn them: "
                     + ", ".join(repr(u) for u in never))
        say(log, "\n" + notes[-1])
    if rare:
        notes.append("Key units that occur fewer than 20 times: " + ", ".join(repr(u) for u in rare))
        say(log, notes[-1])
    if never or rare:
        say(log, "  This is common for nomenclature (whole-word symbols). Expect these to be missed "
                 "when reading. See 'Rare units' in the guide.")
    return notes


def chunk(stream, length):
    chunks = [stream[i:i + length] for i in range(0, len(stream), length)]
    return [c for c in chunks if len(c) >= max(5, length // 2) and any(u != SPACE_UNIT for u in c)]


def encrypt(units, key, rng):
    return [rng.choice(key.units[u]) for u in units]


def readable(units):
    """Turn plaintext units into reading text (nulls dropped, double sign expanded)."""
    out = []
    for u in units:
        if u == NULL_UNIT:
            continue
        if u == DOUBLE_UNIT:
            out.append(next((p for p in reversed(out) if p != " "), ""))
            continue
        out.append("?" if u in SPECIAL_TOKENS else u)
    return "".join(out)


# -----------------------------------------------------------------------------
# Vocabularies and datasets
# -----------------------------------------------------------------------------
def make_vocab(tokens):
    vocab = list(SPECIAL_TOKENS) + sorted(set(tokens) - set(SPECIAL_TOKENS))
    return {t: i for i, t in enumerate(vocab)}


def encode(seq, vocab):
    unk = vocab[UNK]
    return [vocab[SOS]] + [vocab.get(t, unk) for t in seq] + [vocab[EOS]]


class PairDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs, self.targets = inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]


def make_dataset(cipher_seqs, unit_seqs, in_vocab, out_vocab):
    return PairDataset([encode(c, in_vocab) for c in cipher_seqs],
                       [encode(u, out_vocab) for u in unit_seqs])


def collate(batch):
    width = max(len(x) for x, _ in batch)
    xs = torch.full((len(batch), width), 0, dtype=torch.long)   # 0 == PAD in both vocabularies
    ys = torch.full((len(batch), width), 0, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        xs[i, :len(x)] = torch.tensor(x)
        ys[i, :len(y)] = torch.tensor(y)
    return xs, ys

