"""Shared constants, the friendly error type and small file helpers."""
import re

# Special tokens shared by both vocabularies (same index in both).
PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]

# Special plaintext units a key may use.
SPACE_UNIT = " "            # <SPACE> in a key: the gap between words
NULL_UNIT = "<NULL>"        # meaningless filler symbols
DOUBLE_UNIT = "<DOUBLE>"    # "repeat the previous letter" sign
SPECIAL_UNIT_NAMES = {"<SPACE>": SPACE_UNIT, "<NULL>": NULL_UNIT, "<DOUBLE>": DOUBLE_UNIT}
SPECIAL_UNITS = set(SPECIAL_UNIT_NAMES.values())

# Digit mode (see digits.py): label of a digit that belongs to the code before it.
JOIN_UNIT = "<JOIN>"

# A blank space used as a cipher symbol (e.g. a word divider) is always written as this
# visible symbol, in keys, transcriptions and outputs, so that every key and transcription
# can use the standard "symbols separated by spaces" format.
BLANK_SYMBOL = "_"

# Transcription formats. "symbols separated by spaces" is the standard; "one character per
# symbol" files are read character by character, with blanks turned into BLANK_SYMBOL.
SPACE_SEPARATED = "symbols separated by spaces"
ONE_CHAR = "one character per symbol"
FORMATS = (SPACE_SEPARATED, ONE_CHAR)
FORMAT_ALIASES = {"space": SPACE_SEPARATED, "spaces": SPACE_SEPARATED, "char": ONE_CHAR,
                  "chars": ONE_CHAR, "character": ONE_CHAR}


def resolve_format(name):
    name = FORMAT_ALIASES.get(name, name)
    if name not in FORMATS:
        stop(f"Unknown transcription format {name!r}. Use 'space' ({SPACE_SEPARATED}) "
             f"or 'char' ({ONE_CHAR}).")
    return name


class StopHere(Exception):
    """Raised with a plain-language message when something needs fixing."""


def stop(message):
    raise StopHere("\n\n" + message.strip() + "\n")


def read_text_file(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def split_symbols(text, symbol_format):
    """Split one transcription line into cipher symbols. In 'one character per symbol' text,
    a blank is a symbol and becomes BLANK_SYMBOL."""
    if symbol_format == SPACE_SEPARATED:
        return text.split()
    return [BLANK_SYMBOL if c == " " else c for c in text]


def safe_name(text, default="my_cipher"):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", (text or "").strip()).strip("_") or default


def say(log, *parts):
    if log:
        log(" ".join(str(p) for p in parts))
