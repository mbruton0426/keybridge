"""Scoring how much a decoded output reads like a language, and which one.

Used ONLY to rank and describe outputs; the decryption itself never uses language data.

Profiles are character-trigram counts built from HistCorp with
tools/build_language_profiles.py, stored as <code>.json.gz in the package's
language_profiles folder (or a folder given with KEYBRIDGE_LANGUAGE_PROFILES).

Each profile is adapted to the key being scored: letters the key cannot produce are
mapped through the key's letter replacements (e.g. J=I) or merged into one "other"
class, and word gaps are used only if the key encodes them.

plausibility = (score of the output - score of random letters)
               / (score of real text in that language - score of random letters)
so about 1 means "reads like real text in this language" and about 0 "like random
letters". It is a rough guide, not a probability.
"""
import functools
import glob
import gzip
import json
import math
import os
import random
import re
import unicodedata
from collections import Counter

from .common import SPACE_UNIT, SPECIAL_UNITS

OTHER = "·"
LAMBDAS = (0.6, 0.3, 0.1)
MIN_LETTERS = 30


def profile_dirs(extra=None):
    dirs = [os.path.join(os.path.dirname(__file__), "language_profiles")]
    env = os.environ.get("KEYBRIDGE_LANGUAGE_PROFILES")
    for d in (env, extra):
        if d:
            dirs.append(d)
    return dirs


@functools.lru_cache(maxsize=None)
def _load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def available_languages(extra_dir=None):
    """{code: (name, path)} for every profile found."""
    found = {}
    for d in profile_dirs(extra_dir):
        for p in sorted(glob.glob(os.path.join(d, "*.json.gz"))):
            code = os.path.basename(p)[:-len(".json.gz")]
            found[code] = (_load(p).get("name", code), p)
    return found


def key_alphabet(key_units):
    """Lowercase letters the key's plaintext units can produce."""
    letters = set()
    for u in key_units:
        if u in SPECIAL_UNITS:
            continue
        for ch in unicodedata.normalize("NFKC", u).lower():
            if ch.isalpha():
                letters.add(ch)
    return frozenset(letters)


def _char_map(alphabet, replacements):
    reps = {a.lower(): b.lower() for a, b in replacements.items() if len(a) == 1 and len(b) == 1}

    def m(ch):
        if ch == " ":
            return " "
        ch = reps.get(ch, ch)
        return ch if ch in alphabet else OTHER
    return m


class AdaptedModel:
    def __init__(self, trigrams, held_out, alphabet, replacements, spaced):
        m = _char_map(alphabet, replacements)
        tri = Counter()
        for t, n in trigrams.items():
            tri["".join(m(c) for c in t)] += n
        self.tri = tri
        self.ctx2, self.bi, self.ctx1, self.uni = Counter(), Counter(), Counter(), Counter()
        for t, n in tri.items():
            self.ctx2[t[:2]] += n
            self.bi[t[1:]] += n
            self.ctx1[t[1]] += n
            self.uni[t[2]] += n
        self.total = sum(self.uni.values())
        self.vocab = len(alphabet) + (2 if spaced else 1)
        self.spaced = spaced
        self.alphabet = alphabet
        sample = "".join(m(c) for c in held_out)
        self.self_score = self.logprob(sample)
        rng = random.Random(0)
        letters = sorted(alphabet)
        rnd = "".join(" " if spaced and rng.random() < 0.17 else rng.choice(letters) for _ in range(5000))
        self.random_score = self.logprob(rnd)

    def logprob(self, text):
        """Mean log2 probability per character, or None if the text is too short."""
        if sum(c != " " for c in text) < MIN_LETTERS:
            return None
        padded = f" {text} " if self.spaced else text
        l3, l2, l1 = LAMBDAS
        total = 0.0
        for i in range(2, len(padded)):
            a, b, c = padded[i - 2:i], padded[i - 1], padded[i]
            p = l1 * (self.uni[c] + 1) / (self.total + self.vocab)
            if self.ctx1[b]:
                p += l2 * self.bi[b + c] / self.ctx1[b]
            if self.ctx2[a]:
                p += l3 * self.tri[a + c] / self.ctx2[a]
            total += math.log2(p)
        return total / (len(padded) - 2)

    def plausibility(self, text):
        lp = self.logprob(text)
        if lp is None or self.self_score is None or self.self_score - self.random_score < 1e-9:
            return None
        return (lp - self.random_score) / (self.self_score - self.random_score)


@functools.lru_cache(maxsize=256)
def _adapted(path, alphabet, replacements_items, spaced):
    prof = _load(path)
    variant = prof["spaced" if spaced else "unspaced"]
    return AdaptedModel(variant["trigrams"], variant["held_out"], alphabet, dict(replacements_items), spaced)


def prepare_output_text(units, alphabet, spaced):
    """Decoded units -> lowercase letters (and single spaces if the key has word gaps)."""
    out = []
    for u in units:
        if u in SPECIAL_UNITS and u != SPACE_UNIT:
            continue
        out.append(u)
    text = unicodedata.normalize("NFKC", "".join(out)).lower()
    text = "".join(c if (c.isalpha() and c in alphabet) else (OTHER if c.isalpha() else " ") for c in text)
    text = re.sub(r"\s+", " " if spaced else "", text).strip()
    return text


def rank_languages(units, key_units, replacements=None, languages=None, extra_dir=None):
    """Returns [(code, name, plausibility)], best first. Empty if no profiles or too little text."""
    alphabet = key_alphabet(key_units)
    if not alphabet:
        return []
    spaced = SPACE_UNIT in key_units
    text = prepare_output_text(units, alphabet, spaced)
    reps = tuple(sorted((replacements or {}).items()))
    results = []
    for code, (name, path) in available_languages(extra_dir).items():
        if languages and code not in languages:
            continue
        score = _adapted(path, alphabet, reps, spaced).plausibility(text)
        if score is not None:
            results.append((code, name, score))
    return sorted(results, key=lambda r: -r[2])


