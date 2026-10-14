"""Reading, checking and writing cipher keys.

Key files are tab-separated (.tsv). Every plaintext entry and every cipher symbol is
written in double quotes, so spaces and punctuation are always unambiguous:

    plaintext	ciphertext	whole_word
    "A"	"J"	"/"
    "SCH"	"Y"
    "Austria"	"z"	yes
    <SPACE>	" "
    <NULL>	"w"	"7"	"u"

- Column 1: what the symbol(s) mean, in quotes, or one of <SPACE>, <NULL>, <DOUBLE>.
- Then one or more cipher symbols, each in its own quotes. A blank space as a symbol is " ".
  A quote mark inside a symbol is written \" and a backslash \\.
- Optionally, the word yes (no quotes) at the end: the unit only stands for whole words.
- Lines starting with # are notes. An unquoted header line starting with "plaintext" is skipped.
Tabs separate the columns, but spaces between quoted items are tolerated too.
"""
import json
import os
import re

from .common import (BLANK_SYMBOL, DOUBLE_UNIT, NULL_UNIT, ONE_CHAR, SPACE_UNIT, SPECIAL_UNIT_NAMES,
                     SPECIAL_UNITS, read_text_file, stop)

TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|(<[A-Za-z]+>)|(\S+)')
WHOLE_WORD_WORDS = {"yes", "y", "ja", "true", "whole_word", "word"}
SPECIAL_NAME_FOR_UNIT = {v: k for k, v in SPECIAL_UNIT_NAMES.items()}


def _unescape(s):
    return re.sub(r"\\(.)", r"\1", s)


def _escape(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


class CipherKey:
    def __init__(self, units, whole_word=(), warnings=(), source=""):
        self.units = units                  # plaintext unit -> [cipher symbols]
        self.whole_word = set(whole_word)   # units that only stand for whole words
        self.warnings = list(warnings)
        self.source = source
        self.blank_converted = False
        self.symbol_to_unit = {s: u for u, syms in units.items() for s in syms}

    @property
    def symbols(self):
        return sorted(self.symbol_to_unit)

    def has(self, unit):
        return unit in self.units

    @property
    def letter_units(self):
        return [u for u in self.units if u not in SPECIAL_UNITS]

    def symbol_type(self):
        """'digits', 'characters' (one character each) or 'named' (longer names)."""
        syms = self.symbols
        if syms and all(s.isdigit() for s in syms):
            return "digits"
        if all(len(s) == 1 for s in syms):
            return "characters"
        return "named"

    def to_json(self):
        return {"units": self.units, "whole_word": sorted(self.whole_word)}

    @classmethod
    def from_json(cls, data):
        return cls({u: list(s) for u, s in data["units"].items()}, data.get("whole_word", ()))

    def summary(self):
        lines = [f"{len(self.units)} plaintext units written with {len(self.symbol_to_unit)} "
                 f"distinct cipher symbols ({self.symbol_type()})."]
        specials = [SPECIAL_NAME_FOR_UNIT[u] for u in (SPACE_UNIT, NULL_UNIT, DOUBLE_UNIT) if u in self.units]
        if specials:
            lines.append("Special units: " + ", ".join(specials))
        multi = sorted(u for u in self.letter_units if len(u) > 1)
        if multi:
            lines.append(f"Multi-letter units / nomenclature ({len(multi)}): " + ", ".join(multi))
        if self.whole_word:
            lines.append("Whole-word only: " + ", ".join(sorted(self.whole_word)))
        if self.blank_converted:
            lines.append(f'The blank symbol " " is written as "{BLANK_SYMBOL}" (standard format).')
        return "\n".join(lines)

    def write_tsv(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("plaintext\tciphertext\twhole_word\n")
            for u, syms in self.units.items():
                first = SPECIAL_NAME_FOR_UNIT.get(u) or f'"{_escape(u)}"'
                cells = [first] + [f'"{_escape(s)}"' for s in syms]
                if u in self.whole_word:
                    cells.append("yes")
                f.write("\t".join(cells) + "\n")


def _normalise_unit(text):
    if text == " ":
        return SPACE_UNIT
    return " ".join(text.split()).upper()


def _parse_tsv(text):
    """Returns (rows, errors); each row is (line_no, unit, [symbols], whole_word)."""
    rows, errors = [], []
    for no, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.strip().lower().startswith("plaintext"):
            continue
        tokens = list(TOKEN.finditer(line))
        first = tokens[0]
        if first.group(1) is not None:
            raw_unit = _unescape(first.group(1))
            if raw_unit == "":
                errors.append(f"Line {no}: the plaintext is empty (\"\").")
                continue
            unit = _normalise_unit(raw_unit)
        elif first.group(2) is not None:
            name = first.group(2).upper()
            if name not in SPECIAL_UNIT_NAMES:
                errors.append(f"Line {no}: {first.group(2)} is not a known special name. "
                              "Use <SPACE>, <NULL> or <DOUBLE>, or put the text in quotes.")
                continue
            unit = SPECIAL_UNIT_NAMES[name]
        else:
            errors.append(f'Line {no}: the plaintext {first.group(3)!r} must be in double quotes, '
                          f'e.g. "{first.group(3)}".')
            continue
        symbols, whole, errors_before = [], False, len(errors)
        for i, t in enumerate(tokens[1:], 1):
            if t.group(1) is not None:
                sym = _unescape(t.group(1))
                if sym == "":
                    errors.append(f'Line {no}: empty symbol "". A blank space as a symbol is written " ".')
                    continue
                symbols.append(sym)
            elif t.group(3) is not None and t.group(3).lower() in WHOLE_WORD_WORDS and i == len(tokens) - 1:
                whole = True
            else:
                bad = t.group(0)
                errors.append(f'Line {no}: the cipher symbol {bad!r} must be in double quotes, e.g. "{bad}". '
                              f"Each symbol needs its own quotes.")
        if not symbols:
            if len(errors) == errors_before:
                errors.append(f"Line {no}: no cipher symbols found after the plaintext.")
            continue
        rows.append((no, unit, symbols, whole))
    return rows, errors


def _rows_from_json(path):
    data = json.loads(read_text_file(path))
    whole = set()
    if isinstance(data, dict) and "units" in data:
        whole = set(data.get("whole_word", ()))
        data = data["units"]
    rows = []
    for no, (unit, syms) in enumerate(data.items(), 1):
        if isinstance(syms, str):
            syms = [syms]
        u = SPECIAL_UNIT_NAMES.get(unit.upper()) or _normalise_unit(unit)
        rows.append((no, u, [str(s) for s in syms], unit in whole or u in whole))
    return rows, []


def load_key(path, symbol_format=None):
    """Read a key (.tsv, or .json for programmatic use) and check it."""
    if not path or not os.path.exists(path):
        stop(f"The key file could not be found: {path!r}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        rows, errors = _rows_from_json(path)
    elif ext in (".tsv", ".txt"):
        rows, errors = _parse_tsv(read_text_file(path))
    else:
        stop(f"The key must be a .tsv file (tab-separated, symbols in double quotes). Got: {os.path.basename(path)}\n"
             "See 'Key file format' in the guide.")
    if errors:
        shown = "\n".join("  • " + e for e in errors[:15])
        more = f"\n  ... and {len(errors) - 15} more" if len(errors) > 15 else ""
        stop(f"The key file has {len(errors)} problem(s):\n{shown}{more}\n\n"
             "Every plaintext entry and every cipher symbol must be in double quotes, "
             'e.g.  "A"<tab>"J"<tab>"/"')

    # A blank space as a cipher symbol is always written as BLANK_SYMBOL ("_").
    has_blank = any(" " in syms for _, _, syms, _ in rows)
    if has_blank:
        if any(BLANK_SYMBOL in syms for _, _, syms, _ in rows):
            stop(f'The key uses both a blank " " and "{BLANK_SYMBOL}" as cipher symbols. A blank symbol is '
                 f'always written as "{BLANK_SYMBOL}", so the two would be confused. Rename the existing '
                 f'"{BLANK_SYMBOL}" symbol (in the key and the transcriptions) and run again.')
        rows = [(no, u, [BLANK_SYMBOL if s == " " else s for s in syms], w) for no, u, syms, w in rows]

    units, whole_word, warnings, owner = {}, set(), [], {}
    for no, unit, symbols, whole in rows:
        if whole:
            whole_word.add(unit)
        for s in symbols:
            if s in owner and owner[s] != unit:
                warnings.append(f"Line {no}: symbol \"{s}\" is listed for both {owner[s]!r} and {unit!r}. "
                                f"Keeping {owner[s]!r}; a symbol can only stand for one thing.")
                continue
            owner[s] = unit
            units.setdefault(unit, [])
            if s not in units[unit]:
                units[unit].append(s)
    if len([u for u in units if u not in SPECIAL_UNITS]) < 5:
        stop("Fewer than 5 plaintext units were found in the key. Check the key file format.")

    if symbol_format == ONE_CHAR:
        long_syms = sorted(s for s in owner if len(s) > 1)
        if long_syms:
            stop("You chose 'one character per symbol', but these key symbols are longer than one "
                 f"character: {long_syms[:12]}\nChoose 'symbols separated by spaces' instead, "
                 "or write each symbol as a single character.")
    key = CipherKey(units, whole_word, warnings, source=os.path.basename(path))
    key.blank_converted = has_blank
    return key


def scrambled_key(key, rng):
    """Same alphabet and homophone counts, symbols reassigned at random.
    Used as a stand-in for 'a ciphertext written with some other key'."""
    pairs = [(u, s) for u, syms in key.units.items() for s in syms]
    symbols = [s for _, s in pairs]
    rng.shuffle(symbols)
    units = {}
    for (u, _), s in zip(pairs, symbols):
        units.setdefault(u, []).append(s)
    return CipherKey(units, key.whole_word)
