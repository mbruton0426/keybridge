"""Reading transcription files."""
from .common import read_text_file, split_symbols, stop


def read_transcription(path, symbol_format, skip_prefixes=("##",), edit_line=None, max_length=None):
    """Returns a list of (line number, [symbols]). Empty lines and lines starting with any of
    `skip_prefixes` are skipped. Lines longer than 2 × max_length are cut into pieces."""
    out = []
    for no, raw in enumerate(read_text_file(path).splitlines(), 1):
        text = raw.strip("\r\n")
        if not text.strip() or any(text.lstrip().startswith(p) for p in skip_prefixes if p):
            continue
        symbols = split_symbols(text.strip(), symbol_format)
        if edit_line is not None:
            symbols = list(edit_line(symbols))
        if not symbols:
            continue
        if max_length and len(symbols) > 2 * max_length:
            for i in range(0, len(symbols), max_length):
                out.append((no, symbols[i:i + max_length]))
        else:
            out.append((no, symbols))
    if not out:
        stop("No ciphertext lines were found in the transcription. Check the 'skip lines' setting "
             "and that the file is not empty.")
    return out


def parse_skip_prefixes(text):
    return tuple(p.strip() for p in (text or "").split(",") if p.strip())
