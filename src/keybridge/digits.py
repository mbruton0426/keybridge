"""Digit ciphers: codes written as running digits, read digit by digit.

In many cipher manuscripts the codes of a digit key (e.g. 12, 7, 03, 5^.7) are written as one
running string of digits, so where one code ends and the next begins is only known once the
text is deciphered. For such keys the model works on single digits ("pieces"): a digit plus any
marks written with it, e.g. 2^. (dot on top), 7^` (accent), 8_. (dot below). Each digit is
labelled with the plaintext unit ("a new code starts here and means E") or <JOIN> ("this digit
belongs to the code before it").

Reading ("model + key"): of all ways to cut the digits into codes of the key, the one the model
finds most likely is chosen (Viterbi search over the model's log-probabilities). The key decides
what each code means; the model decides where the codes are. No language data is used.

A key uses digit mode when every cipher symbol is made of digits (with optional marks that are
not letters or digits) and at least one symbol has more than one digit.
"""
import math
import re

import torch

from .common import EOS, JOIN_UNIT, PAD, SOS, UNK, stop
from .data import encode

UNREAD_PENALTY = -50.0     # log-score of a digit that fits no code (it is left unread)


# -----------------------------------------------------------------------------
# Pieces
# -----------------------------------------------------------------------------
def code_marks(symbols):
    """The non-digit characters used in the key's codes, e.g. '^.`'."""
    return "".join(sorted({c for s in symbols for c in s if not c.isdigit()}))


def piece_regex(marks):
    """A piece = optional marks, one digit, optional marks (the same rule for keys and transcriptions)."""
    if not marks:
        return re.compile(r"\d")
    m = "[" + re.escape(marks) + "]*"
    return re.compile(m + r"\d" + m)


def split_code(code, rx):
    pieces = rx.findall(code)
    return pieces if "".join(pieces) == code and pieces else None


def detect(key):
    """Digit-mode information for a key, or None if the key is not a digit key."""
    syms = key.symbols
    if not syms:
        return None
    marks = code_marks(syms)
    if any(c.isalnum() or c.isspace() for c in marks):
        return None
    rx = piece_regex(marks)
    split = {s: split_code(s, rx) for s in syms}
    if any(p is None for p in split.values()):
        return None
    longest = max(len(p) for p in split.values())
    if longest < 2:
        return None
    pieces = sorted({p for ps in split.values() for p in ps})
    return {"marks": marks, "pieces": pieces, "max_pieces": longest}


def resolve_digit_mode(key, mode="auto"):
    """'auto' uses digit mode for digit keys; 'on' requires it; 'off' never uses it."""
    mode = (mode or "auto").lower()
    if mode == "off":
        return None
    info = detect(key)
    if mode == "on" and info is None:
        stop("Digit mode was requested, but this key is not a digit key: every cipher symbol must be made "
             "of digits (with optional marks such as ^. or ^`), and at least one must have two or more digits.")
    if info is not None and JOIN_UNIT in key.units:
        stop(f"The key uses {JOIN_UNIT} as a plaintext unit, which digit mode needs for itself. Rename it.")
    return info


def tokenize_symbols(symbols, rx):
    """A transcription line (list of space-separated items) -> [(piece, starts_item)].
    Characters that are not part of a digit piece (e.g. '?', '|') become pieces of their own."""
    out = []
    for item in symbols:
        pos, first = 0, True
        while pos < len(item):
            m = rx.match(item, pos)
            if m and m.end() > pos:
                piece, pos = m.group(), m.end()
            else:
                piece, pos = item[pos], pos + 1
            out.append((piece, first))
            first = False
    return out


def expand(codes, units, rx):
    """Codes and their units (one per code) -> digit pieces and labels (unit, then <JOIN>s)."""
    pieces, labels = [], []
    for c, u in zip(codes, units):
        ps = rx.findall(c)
        pieces += ps
        labels += [u] + [JOIN_UNIT] * (len(ps) - 1)
    return pieces, labels


def info_of(checkpoint):
    """Digit-mode information stored with a model, or None for ordinary models."""
    return (checkpoint.get("settings") or {}).get("digit_mode_info")


def ambiguous_codes(key_units, max_pieces, rx):
    """Codes whose digits can also be cut into other codes of the key, e.g. 71 = CHE or 7 + 1 = F A."""
    code_to_unit = {c: u for u, cs in key_units.items() for c in cs}
    pieces = {c: rx.findall(c) for c in code_to_unit}

    def splittable(ps):
        if not ps:
            return True
        return any("".join(ps[:L]) in code_to_unit and splittable(ps[L:])
                   for L in range(1, min(max_pieces, len(ps)) + 1))
    return {c: ps for c, ps in pieces.items()
            if len(ps) > 1 and any("".join(ps[:L]) in code_to_unit and splittable(ps[L:])
                                   for L in range(1, len(ps)))}


# -----------------------------------------------------------------------------
# Reading
# -----------------------------------------------------------------------------
@torch.no_grad()
def window_logprobs(model, pieces, in_vocab, n_out, window, batch_size=64):
    """Log-probabilities (len(pieces) × n_out) for a long stream of pieces, read in overlapping
    windows (each wrapped in start/end tokens, as in training); every piece takes its values from
    the window in which it is most central."""
    model.eval()
    dev = next(model.parameters()).device
    N = len(pieces)
    out = torch.full((N, n_out), -1e9)
    if N == 0:
        return out
    window = max(8, int(window))
    step = max(1, window // 2)
    starts = list(range(0, max(1, N - window) + 1, step))
    if starts[-1] + window < N:
        starts.append(N - window)
    best = [float("inf")] * N
    for b in range(0, len(starts), batch_size):
        S = starts[b:b + batch_size]
        enc = [encode(pieces[s:s + window], in_vocab) for s in S]
        x = torch.zeros(len(S), max(len(e) for e in enc), dtype=torch.long)
        for i, e in enumerate(enc):
            x[i, :len(e)] = torch.tensor(e)
        lp = torch.log_softmax(model(x.to(dev)).float(), dim=-1).cpu()
        for i, s in enumerate(S):
            L = min(window, N - s)
            centre = (L - 1) / 2
            for t in range(L):
                d = abs(t - centre)
                if d < best[s + t]:
                    best[s + t] = d
                    out[s + t] = lp[i, t + 1]       # +1: position 0 is the start token
    return out


def key_decode(pieces, lp, code_to_unit, out_vocab, max_pieces, boundaries=None):
    """Best split of the pieces into key codes. Returns [(start, end, code, unit, confidence)];
    unit is None for a piece that fits no code. With `boundaries` (True where a transcription item
    starts), codes may not run across item boundaries."""
    N = len(pieces)
    join = out_vocab[JOIN_UNIT]
    lpj = lp[:, join].tolist()
    best = [0.0] + [-math.inf] * N
    back = [None] * (N + 1)
    for i in range(N):
        if best[i] == -math.inf:
            continue
        acc_join = 0.0
        for L in range(1, max_pieces + 1):
            j = i + L
            if j > N:
                break
            if L > 1:
                if boundaries is not None and boundaries[j - 1]:
                    break
                acc_join += lpj[j - 1]
            code = "".join(pieces[i:j])
            unit = code_to_unit.get(code)
            if unit is not None and unit in out_vocab:
                s = lp[i, out_vocab[unit]].item() + acc_join
            elif L == 1:
                s, unit = UNREAD_PENALTY, None
            else:
                continue
            if best[i] + s > best[j]:
                best[j] = best[i] + s
                back[j] = (i, code, unit)
    codes, j = [], N
    while j > 0:
        i, code, unit = back[j]
        if unit is None:
            conf = 0.0
        else:
            probs = [math.exp(lp[i, out_vocab[unit]].item())] + [math.exp(lpj[t]) for t in range(i + 1, j)]
            conf = min(probs)
        codes.append((i, j, code, unit, conf))
        j = i
    return codes[::-1]


def model_decode(pieces, lp, id_to_unit):
    """The model's own labels, grouped into codes: [(start, end, code, unit, confidence)]."""
    conf, idx = lp.exp().max(-1)
    labels = [id_to_unit[k] for k in idx.tolist()]
    out, i, N = [], 0, len(pieces)
    while i < N:
        j = i + 1
        while j < N and labels[j] == JOIN_UNIT:
            j += 1
        u = labels[i] if labels[i] != JOIN_UNIT else None
        out.append((i, j, "".join(pieces[i:j]), u, min(conf[i:j].tolist())))
        i = j
    return out, labels, conf.tolist()


def read_stream(model, checkpoint, pieces, boundaries=None, use_separators=False):
    """Reads a stream of pieces. Returns a dict with the model's own labels per piece and the
    model + key codes (the reading keybridge outputs)."""
    info = info_of(checkpoint)
    in_vocab, out_vocab = checkpoint["input_vocab"], checkpoint["output_vocab"]
    key_units = checkpoint["key"]["units"]
    code_to_unit = {c: u for u, cs in key_units.items() for c in cs}
    lp = window_logprobs(model, pieces, in_vocab, len(out_vocab), info.get("window", 80))
    for t in (PAD, SOS, EOS, UNK):
        lp[:, out_vocab[t]] = -1e9
    id_to_unit = {i: u for u, i in out_vocab.items()}
    model_codes, model_labels, model_conf = model_decode(pieces, lp, id_to_unit)
    codes = key_decode(pieces, lp, code_to_unit, out_vocab, info["max_pieces"],
                       boundaries if use_separators else None)
    return {"pieces": pieces, "model_labels": model_labels, "model_confidence": model_conf,
            "model_codes": model_codes, "codes": codes}


def read_digit_lines(model, checkpoint, numbered_lines, use_separators=False):
    """Like pipeline.read_lines, for digit models: the lines are read as one stream (codes may run
    over line ends), and each code is reported on the line where its first digit is. Each reading's
    'symbols' are the codes as split by the model."""
    rx = piece_regex(info_of(checkpoint)["marks"])
    pieces, owner, bounds = [], [], []
    for k, (_, syms) in enumerate(numbered_lines):
        for piece, starts in tokenize_symbols(syms, rx):
            pieces.append(piece)
            owner.append(k)
            bounds.append(starts)
    res = read_stream(model, checkpoint, pieces, bounds, use_separators)
    readings = [{"line": no, "symbols": [], "units": [], "confidence": [], "digits": [], "model_units": []}
                for no, _ in numbered_lines]
    for s, e, code, unit, conf in res["codes"]:
        r = readings[owner[s]]
        r["symbols"].append(code)
        r["units"].append(unit if unit is not None else UNK)
        r["confidence"].append(conf)
    for i, p in enumerate(pieces):
        readings[owner[i]]["digits"].append(p)
        readings[owner[i]]["model_units"].append(res["model_labels"][i])
    return readings
