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
# Plaintext -> (cipher symbols, key units), enciphered with ChronoFidelius
# -----------------------------------------------------------------------------
# The key is turned into a ChronoFidelius encryption dictionary on the fly (the key file itself does
# not change):
#   - every unit's text maps to its symbols; each option of an [A|B] unit also maps to that unit's
#     symbols, so "A" can come out as a symbol of A or of [A|IT|L];
#   - word gaps, the double-letter sign, rare units inserted for training, and the cancel signs are
#     written into the text as private placeholder characters that map to exactly those units' symbols;
#   - nulls are inserted by ChronoFidelius (null_codes).
# The label of every symbol is then the unit the key gives that symbol.
GAP_CHAR = "\U000F0001"         # a word gap (<SPACE>)
DOUBLE_CHAR = "\U000F0002"      # the "repeat the previous letter" sign (<DOUBLE>)
UNIT_CHAR_BASE = 0xF0100         # one placeholder per unit, for rare-unit insertion
CANCEL_UNIT = "[-]"
DEL_START, DEL_END, DEL_BOTH = "[DEL-START]", "[DEL-END]", "[DEL-START|DEL-END]"
CANCEL_MARK, SPAN_MARKS = "\u2190", ("\u27e6", "\u27e7")     # ChronoFidelius's default cancel markers
NOT_TEXT = {"?", "-", "DEL-START", "DEL-END", "<NULL>", "<SPACE>", "<DOUBLE>"}


def unit_options(unit):
    """'[A|IT|L]' -> ['A', 'IT', 'L']; any other unit -> [unit]."""
    core = unit[1:-1] if len(unit) > 2 and unit.startswith("[") and unit.endswith("]") else unit
    if "|" in core and unit not in SPECIAL_TOKENS:
        return [p for p in core.split("|") if p]
    return [unit]


def options_match(a, b):
    """True if two units are the same or share an option ([REY|REG] and REG)."""
    return a == b or bool(set(unit_options(a)) & set(unit_options(b)))


class KeyDictionary:
    """The key as a ChronoFidelius encryption dictionary, plus the placeholders keybridge uses."""

    def __init__(self, key):
        self.key = key
        d, self.unit_char = {}, {}
        for i, (u, syms) in enumerate(key.units.items()):
            if u == NULL_UNIT:
                continue
            if u == SPACE_UNIT:
                d.setdefault(GAP_CHAR, []).extend(syms)
                continue
            if u == DOUBLE_UNIT:
                d.setdefault(DOUBLE_CHAR, []).extend(syms)
                continue
            if u in (CANCEL_UNIT, DEL_START, DEL_END, DEL_BOTH):
                marks = {CANCEL_UNIT: [CANCEL_MARK], DEL_START: [SPAN_MARKS[0]], DEL_END: [SPAN_MARKS[1]],
                         DEL_BOTH: list(SPAN_MARKS)}[u]
                for m in marks:
                    d.setdefault(m, []).extend(syms)
                continue
            for opt in unit_options(u):
                if opt not in NOT_TEXT and not opt.startswith("<"):
                    d.setdefault(opt, []).extend(syms)
            ch = chr(UNIT_CHAR_BASE + i)
            self.unit_char[u] = ch
            d[ch] = list(syms)
        self.dictionary = {k: list(dict.fromkeys(v)) for k, v in d.items()}
        self.null_codes = list(key.units.get(NULL_UNIT, []))
        self.cancel_types = []
        if CANCEL_MARK in self.dictionary:
            self.cancel_types.append("cancelled")
        if SPAN_MARKS[0] in self.dictionary and SPAN_MARKS[1] in self.dictionary:
            self.cancel_types.append("cancelled_span")


def _prepare_text(doc, kd, rng, gaps, double_rate, rare_chars, rare_rate):
    """Plaintext -> text for ChronoFidelius: word gaps, double signs and rare units as placeholders.
    gaps: 'keep' (real gaps -> <SPACE>), 'random' (gaps inserted at random), or 'none'."""
    key = kd.key
    if gaps == "keep":
        text = doc.replace(" ", GAP_CHAR)
    else:
        text = doc.replace(" ", "")
        if gaps == "random":
            out = []
            for ch in text:
                if out and out[-1] != GAP_CHAR and rng.random() < 0.15:
                    out.append(GAP_CHAR)
                out.append(ch)
            text = "".join(out)
    if key.has(DOUBLE_UNIT) and double_rate > 0:
        out = []
        for ch in text:
            if out and ch == out[-1] and key.has(ch) and rng.random() < double_rate:
                out.append(DOUBLE_CHAR)
            else:
                out.append(ch)
        text = "".join(out)
    if rare_chars and rare_rate > 0:
        out, has_gaps = [], GAP_CHAR in text
        for ch in text:
            out.append(ch)
            # at word gaps when the text has them; otherwise anywhere (e.g. HistCiph, which has no gaps)
            if (ch == GAP_CHAR or not has_gaps) and rng.random() < rare_rate:
                out.append(rng.choice(rare_chars))
                if has_gaps:
                    out.append(GAP_CHAR)
        text = "".join(out)
    return text


def _encipher(texts, kd, seed, null_rate, multi_char_rate, cancel_rate):
    """Runs ChronoFidelius on the prepared texts. Returns ([codes per text], unmatched counts)."""
    try:
        from chronofidelius import ChronoFidelius
    except ImportError:
        stop("keybridge needs ChronoFidelius (version 1.3.0 or later) to make training data: pip install -U ChronoFidelius")
    errors = bool(kd.cancel_types) and cancel_rate > 0
    texts = [t for t in texts if t]
    if not texts:
        return [], Counter()
    try:
        cf = ChronoFidelius(texts, set_seed=seed, include_errors=errors,
                            error_type=kd.cancel_types if errors else None,
                            error_frequency=cancel_rate if errors else 0.05,
                            include_spacing=True, max_length=10 ** 12, set_punctuation="", set_case="no_change",
                            cancel_marker=CANCEL_MARK, cancel_span_markers=SPAN_MARKS)
        cf.encrypt_homophonic(encryption_dict=kd.dictionary, null_codes=kd.null_codes or None,
                              null_frequency=null_rate if kd.null_codes else 0.05,
                              multi_char_frequency=multi_char_rate, unmatched="drop")
    except TypeError as e:
        stop(f"keybridge needs ChronoFidelius 1.3.0 or later ({e}). Update it: pip install -U ChronoFidelius")
    docs = sorted((k for k in cf.pt_ct_dict if k.isdigit()), key=int)
    return [list(cf.pt_ct_dict[k]["ciphertext"]) for k in docs], Counter(cf.unmatched)


def _apply_unit_edit(codes, units, custom_edit, key, rng):
    """custom_edit_training_units: edits the list of units; symbols are kept where the unit is unchanged
    and chosen at random from the key where the edit changed or inserted a unit."""
    from difflib import SequenceMatcher
    new_units = list(custom_edit(list(units)))
    bad = sorted({u for u in new_units if u not in key.units})
    if bad:
        stop("Your custom edit produced units that are not in the key: " + ", ".join(map(repr, bad[:20])))
    new_codes = [None] * len(new_units)
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, units, new_units, autojunk=False).get_opcodes():
        if tag == "equal":
            new_codes[j1:j2] = codes[i1:i2]
    return [c if c is not None else rng.choice(key.units[u]) for c, u in zip(new_codes, new_units)], new_units


def build_examples(documents, key, rng, seed, word_spaces="keep real word boundaries", null_rate=0.05,
                   double_rate=0.5, rare_unit_rate=0.02, multi_char_rate=0.7, cancel_rate=0.05,
                   custom_edit=None, custom_edit_text=None, rare_units=None):
    """Turns plaintext documents into one long list of (cipher symbol, key unit) pairs, enciphered with
    ChronoFidelius. Returns (codes, units, report). rare_units: units to insert (found by a first pass
    when None)."""
    kd = KeyDictionary(key)
    if custom_edit_text is not None:
        documents = [custom_edit_text(d) for d in documents]
    report = {"characters": sum(len(d) for d in documents)}
    inner = sum(d.count(" ") for d in documents)
    if not key.has(SPACE_UNIT) or word_spaces == "remove":
        gaps = "none"
    elif word_spaces == "insert at random":
        gaps = "random"
    elif inner < 0.01 * max(1, report["characters"]):
        gaps = "random"
        report["word_spaces_note"] = ("The plaintext has (almost) no word gaps, so gaps were inserted at "
                                      "random instead of keeping real word boundaries.")
    else:
        gaps = "keep"

    if rare_units is None and rare_unit_rate > 0:
        # first pass, without insertion, to find the units that are rare in this plaintext
        texts = [_prepare_text(d, kd, rng, gaps, double_rate, [], 0) for d in documents]
        coded, _ = _encipher(texts, kd, seed, null_rate, multi_char_rate, cancel_rate)
        counts = Counter(key.symbol_to_unit[c] for cs in coded for c in cs if c in key.symbol_to_unit)
        rare_units = [u for u in key.letter_units if counts[u] < 20 and u in kd.unit_char]
    rare_units = rare_units or []
    report["injected_units"] = rare_units
    rare_chars = [kd.unit_char[u] for u in rare_units if u in kd.unit_char]

    texts = [_prepare_text(d, kd, rng, gaps, double_rate, rare_chars, rare_unit_rate) for d in documents]
    coded, unmatched = _encipher(texts, kd, seed, null_rate, multi_char_rate, cancel_rate)
    codes, units = [], []
    for cs in coded:
        for c in cs:
            u = key.symbol_to_unit.get(c)
            if u is None:          # cannot happen with a well-formed dictionary; skipped to be safe
                continue
            codes.append(c)
            units.append(u)
    if custom_edit is not None:
        codes, units = _apply_unit_edit(codes, units, custom_edit, key, rng)
    report["uncovered"] = Counter({k: v for k, v in unmatched.items()
                                   if k not in (GAP_CHAR, DOUBLE_CHAR) and ord(k[0]) < UNIT_CHAR_BASE})
    report["unit_counts"] = Counter(units)
    report["stream_length"] = len(units)
    report["rare_units"] = rare_units
    return codes, units, report


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
        say(log, "  This is common for nomenclature (whole-word symbols). Expect these to be missed more "
                 "often when reading" + ("" if report.get("injected_units") else
                 " (the rare unit rate is 0, so they were not inserted)") + ". See 'Rare units' in the guide.")
    return notes


def chunk(stream, length):
    chunks = [stream[i:i + length] for i in range(0, len(stream), length)]
    return [c for c in chunks if len(c) >= max(5, length // 2) and any(u != SPACE_UNIT for u in c)]


def chunk_pairs(codes, units, length):
    """Cuts the (symbol, unit) stream into examples of `length` units: [(codes, units), ...]."""
    out = []
    for i in range(0, len(units), length):
        u = units[i:i + length]
        if len(u) >= max(5, length // 2) and any(x != SPACE_UNIT for x in u):
            out.append((codes[i:i + length], u))
    return out


def encrypt(units, key, rng):
    """Simple re-enciphering of a list of units (used for the scrambled-key reference check only;
    training data are made with ChronoFidelius, see build_examples)."""
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

