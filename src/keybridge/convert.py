"""Converting keys and transcriptions to the standard format.

Standard format: cipher symbols separated by spaces, and a blank space used as a cipher
symbol (e.g. a word divider) written as "_" (BLANK_SYMBOL).
"""
import os

from .common import ONE_CHAR, read_text_file, split_symbols
from .keys import load_key


def standard_path(path, ext=None):
    stem, old_ext = os.path.splitext(path)
    return f"{stem}_standard{ext or old_ext}"


def convert_transcription(path, out_path=None, skip_prefixes=("##",)):
    """Converts a 'one character per symbol' transcription: every character becomes a symbol,
    separated by spaces, with blanks written as '_'. Empty lines and note lines (starting with
    a skip marker) are copied unchanged. Returns (output path, lines converted)."""
    out_path = out_path or standard_path(path)
    out, converted = [], 0
    for raw in read_text_file(path).splitlines():
        line = raw.rstrip("\r\n")
        if not line.strip() or any(line.lstrip().startswith(p) for p in skip_prefixes if p):
            out.append(line)
            continue
        out.append(" ".join(split_symbols(line.strip(), ONE_CHAR)))
        converted += 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return out_path, converted


def convert_key(path, out_path=None):
    """Rewrites a key (.tsv or .json) in the standard .tsv format, with a blank symbol written
    as '_'. Note lines at the top of a .tsv key are kept. Returns (output path, key)."""
    out_path = out_path or standard_path(path, ".tsv")
    key = load_key(path)
    notes = []
    if path.lower().endswith((".tsv", ".txt")):
        for line in read_text_file(path).splitlines():
            if line.lstrip().startswith("#"):
                notes.append(line)
            elif line.strip():
                break
    key.write_tsv(out_path)
    if notes:
        with open(out_path, encoding="utf-8") as f:
            body = f.read()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(notes) + "\n" + body)
    return out_path, key

