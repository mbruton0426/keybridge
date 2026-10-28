"""The training and reading workflows, shared by the notebooks and the command line."""
import csv
import datetime
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field

from . import __version__
from .common import (DOUBLE_UNIT, JOIN_UNIT, NULL_UNIT, SPACE_SEPARATED, SPACE_UNIT, SPECIAL_TOKENS, UNK,
                     say, stop)
from .data import (build_unit_stream, chunk, describe_coverage, encrypt, make_dataset, make_vocab,
                   readable)
from .digits import expand, info_of, piece_regex, read_digit_lines, read_stream, resolve_digit_mode
from .keys import scrambled_key
from .model import (DEFAULT_HYPERPARAMETERS, predict, save_model, score, train_model, tune,
                    uncertain_share)


@dataclass
class TrainSettings:
    project: str = "my_cipher"
    symbol_format: str = SPACE_SEPARATED
    plaintext_source: str = "English"
    plaintext_files: list = field(default_factory=list)
    letter_replacements: str = ""
    word_spaces: str = "keep real word boundaries"
    training_examples: int = 5000
    sequence_length: int = 50
    null_rate: float = 0.05
    double_rate: float = 0.5
    rare_unit_rate: float = 0.02
    search_trials: int = 20
    seed: int = 9
    max_epochs: int = 100
    digit_mode: str = "auto"      # "auto": read digit keys digit by digit (see digits.py); "on"; "off"


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def heldout_sizes(training_examples):
    """Numbers of checking and test examples for a given training size."""
    return max(100, training_examples // 10), max(100, min(1000, training_examples // 10))


def plaintext_chars_needed(training_examples, sequence_length):
    """(characters for training, characters for checking + test) to request from a built-in source."""
    n_val, n_test = heldout_sizes(training_examples)
    return int(training_examples * sequence_length * 1.3), int((n_val + n_test) * sequence_length * 1.5)


def _units_to_chunks(documents, key, s, rng, custom_edit):
    stream, report = build_unit_stream(documents, key, rng, word_spaces=s.word_spaces,
                                       null_rate=s.null_rate, double_rate=s.double_rate,
                                       rare_unit_rate=s.rare_unit_rate, custom_edit=custom_edit)
    chunks = chunk(stream, s.sequence_length)
    rng.shuffle(chunks)
    return chunks, report


def prepare_data(key, documents, s: TrainSettings, custom_edit=None, heldout_documents=None, log=print):
    """documents: plaintext strings to train on. heldout_documents (optional): separate plaintext
    for the checking and test examples, e.g. a dataset's validation split. Without it, checking
    and test examples are taken from `documents` (never overlapping the training examples)."""
    rng = random.Random(s.seed)
    chunks, report = _units_to_chunks(documents, key, s, rng, custom_edit)
    n_val, n_test = heldout_sizes(s.training_examples)
    if heldout_documents:
        held, _ = _units_to_chunks(heldout_documents, key, s, rng, custom_edit)
        if len(held) < n_val + n_test:
            stop(f"The held-out plaintext gave only {len(held)} examples; {n_val + n_test} are needed.")
        val_u, test_u, pool = held[:n_val], held[n_val:n_val + n_test], chunks
        if len(pool) < 200:
            stop(f"The plaintext is too short: it gave {len(pool)} training examples of {s.sequence_length} "
                 "symbols, but at least 200 are needed.")
    else:
        if len(chunks) < n_val + n_test + 200:
            stop(f"The plaintext is too short: it gave {len(chunks)} examples of {s.sequence_length} symbols, "
                 f"but at least {n_val + n_test + 200} are needed.\nPlease provide more text "
                 f"(aim for about {s.training_examples * s.sequence_length * 1.3:,.0f} characters).")
        val_u, test_u, pool = chunks[:n_val], chunks[n_val:n_val + n_test], chunks[n_val + n_test:]
    train_u = [pool[i % len(pool)] for i in range(s.training_examples)]

    info = resolve_digit_mode(key, s.digit_mode)
    codes = {name: [encrypt(u, key, rng) for u in units]
             for name, units in (("train", train_u), ("val", val_u), ("test", test_u))}
    units = {"train": train_u, "val": val_u, "test": test_u}
    data = {"train_units": train_u, "val_units": val_u, "test_units": test_u, "rng": rng,
            "test_codes": codes["test"], "digit_info": info}
    if info:
        # Digit mode: each code is split into its digits; the first digit is labelled with the unit,
        # the others with <JOIN>.
        rx = piece_regex(info["marks"])
        for name in ("train", "val", "test"):
            pairs = [expand(c, u, rx) for c, u in zip(codes[name], units[name])]
            data[f"{name}_cipher"] = [p for p, _ in pairs]
            data[f"{name}_labels"] = [l for _, l in pairs]
        lengths = sorted(len(x) for x in data["train_cipher"])
        info["window"] = lengths[len(lengths) // 2]
        data["in_vocab"] = make_vocab(info["pieces"])
        data["out_vocab"] = make_vocab(list(key.units) + [JOIN_UNIT])
    else:
        for name in ("train", "val", "test"):
            data[f"{name}_cipher"] = codes[name]
            data[f"{name}_labels"] = units[name]
        data["in_vocab"], data["out_vocab"] = make_vocab(key.symbols), make_vocab(list(key.units))
    data["train_ds"] = make_dataset(data["train_cipher"], data["train_labels"], data["in_vocab"], data["out_vocab"])
    data["val_ds"] = make_dataset(data["val_cipher"], data["val_labels"], data["in_vocab"], data["out_vocab"])

    say(log, f"Made {len(train_u)} training, {len(val_u)} checking and {len(test_u)} test examples "
             f"of {s.sequence_length} plaintext units each.")
    if len(pool) < s.training_examples:
        say(log, f"  Note: the plaintext only gave {len(pool)} distinct training examples, so some text is "
                 "reused with different symbols. More plaintext would be better.")
    data["notes"] = describe_coverage(report, key, log)
    # Key units the training text never contains: the model cannot learn their symbols, so the
    # fit check leaves those symbols out (see fit_figures).
    data["untrained_units"] = sorted(u for u in key.units if report["unit_counts"][u] == 0)
    if info:
        amb = "" if not info["marks"] else f" (digits with marks: {info['marks']})"
        say(log, f"\nDigit cipher: the model reads single digits{amb} and decides where each code starts. "
                 f"Examples are about {info['window']} digits long.")
        say(log, "Example (digit → label; <JOIN> = belongs to the code before):")
        pairs = zip(data["train_cipher"][0][:18], data["train_labels"][0][:18])
        say(log, "  " + "  ".join(f"{c}→{'␣' if u == ' ' else u}" for c, u in pairs))
    else:
        say(log, "\nExample (cipher symbol → plaintext unit):")
        pairs = zip(data["train_cipher"][0][:14], train_u[0][:14])
        say(log, "  " + "  ".join(f"{'␣' if c == ' ' else c}→{'␣' if u == ' ' else u}" for c, u in pairs))
    say(log, "  Reads as:", readable(train_u[0]))
    return data


def train(key, data, s: TrainSettings, log=print):
    hp = dict(DEFAULT_HYPERPARAMETERS)
    if s.search_trials:
        say(log, f"Searching for good model settings ({s.search_trials} short trials)...")
        found, acc = tune(data["train_ds"], data["val_ds"], data["in_vocab"], data["out_vocab"],
                          s.search_trials, seed=s.seed, log=log)
        hp.update(found)
        if (hp["hidden_size"] * 2) % hp["num_heads"]:
            hp["num_heads"] = 1
        say(log, f"Best trial reached {acc:.1%}. Settings: {hp}\n")

    say(log, "Training (each round, the model is checked on examples it has not trained on):")
    model, history, seconds = train_model(data["train_ds"], data["val_ds"], data["in_vocab"],
                                          data["out_vocab"], hp, max_epochs=s.max_epochs,
                                          seed=s.seed, log=log)
    say(log, f"Finished in {seconds / 60:.1f} minutes.")

    iv, ov = data["in_vocab"], data["out_vocab"]
    test_pred = predict(model, data["test_cipher"], iv, ov)
    acc, f1 = score(test_pred, data["test_labels"])
    rng = data["rng"]
    untrained = data.get("untrained_units", [])
    untrained_symbols = {sym for u in untrained for sym in key.units[u]}
    info = data.get("digit_info")
    extra = {}
    if info:
        ck = {"input_vocab": iv, "output_vocab": ov, "key": key.to_json(), "settings": {"digit_mode_info": info}}
        rx = piece_regex(info["marks"])

        def digit_reading(code_lists, unit_lists):
            pieces, gold = [], []
            for c, u in zip(code_lists, unit_lists):
                p, l = expand(c, u, rx)
                pieces += p
                gold += l
            res = read_stream(model, ck, pieces)
            labels = [JOIN_UNIT] * len(pieces)
            for st, _, _, unit, _ in res["codes"]:
                labels[st] = unit if unit is not None else UNK
            preds = [[(unit, conf) for _, _, _, unit, conf in res["codes"]]]
            syms = [[code if unit is not None else "\u0000" for _, _, code, unit, _ in res["codes"]]]
            return labels, gold, preds, syms

        labels, gold, preds, syms = digit_reading(data["test_codes"], data["test_units"])
        extra["synthetic_test_accuracy_model_plus_key"] = sum(a == b for a, b in zip(labels, gold)) / max(1, len(gold))
        skip = untrained_symbols | {"\u0000"}
        in_key_ref = share_unsure(preds, syms, skip)
        scrambled = []
        for i in range(3):
            sk = scrambled_key(key, random.Random(s.seed + 1 + i))
            _, _, p2, s2 = digit_reading([encrypt(u, sk, rng) for u in data["test_units"]], data["test_units"])
            scrambled.append(share_unsure(p2, s2, skip))
    else:
        in_key_ref = share_unsure(test_pred, data["test_cipher"], untrained_symbols)
        scrambled = []
        for i in range(3):
            sk = scrambled_key(key, random.Random(s.seed + 1 + i))
            cipher = [encrypt(u, sk, rng) for u in data["test_units"]]
            scrambled.append(share_unsure(predict(model, cipher, iv, ov), cipher, untrained_symbols))
    metrics = {"synthetic_test_accuracy": acc, "synthetic_test_f1": f1, **extra,
               "uncertain_share_in_key_reference": in_key_ref,
               "uncertain_share_scrambled_key_reference": sum(scrambled) / len(scrambled),
               "units_not_in_training": untrained,
               "training_minutes": seconds / 60, "epochs": len(history)}
    say(log, f"\nOn {len(data['test_units'])} unseen synthetic examples: accuracy {acc:.1%}, F1 {f1:.3f}"
             + (" (per digit, the model's own labels)" if info else ""))
    if info:
        say(log, f"  Reading with the key (model + key, as keybridge reads): "
                 f"{extra['synthetic_test_accuracy_model_plus_key']:.1%} of digits correct")
    if acc < 0.95:
        say(log, "  This is lower than usual. More training examples or the model-settings search may help.")
    return {"model": model, "hyperparameters": hp, "history": history, "metrics": metrics}


def save_training_outputs(out_dir, s: TrainSettings, key, key_path, data, result, extra_settings=None):
    """Writes model, key copies and training report. Returns the model path."""
    os.makedirs(out_dir, exist_ok=True)
    name = s.project
    record = {**asdict(s), **(extra_settings or {}), "package_version": __version__,
              "digit_mode_info": data.get("digit_info"),
              "created": datetime.datetime.now().isoformat(timespec="seconds"),
              "key_file": os.path.basename(key_path),
              "plaintext_files": [os.path.basename(p) for p in s.plaintext_files]}
    model_path = os.path.join(out_dir, f"{name}_model.pt")
    save_model(model_path, result["model"], result["hyperparameters"], data["in_vocab"],
               data["out_vocab"], key, record, result["metrics"])
    shutil.copy(key_path, os.path.join(out_dir, f"{name}_key{os.path.splitext(key_path)[1]}"))
    key.write_tsv(os.path.join(out_dir, f"{name}_key_as_read.tsv"))
    with open(os.path.join(out_dir, f"{name}_training_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Training report for '{name}'\n\n")
        f.write("Settings\n" + json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")
        f.write("Key\n" + key.summary() + "\n")
        if data.get("digit_info"):
            f.write("Digit cipher: read digit by digit (codes are found by the model; reading = model + key).\n")
        for w in key.warnings:
            f.write("  key note: " + w + "\n")
        f.write("\nTraining data notes\n" + ("\n".join(data["notes"]) or "none") + "\n\n")
        f.write("Model settings\n" + json.dumps(result["hyperparameters"], indent=2) + "\n\n")
        f.write("Results\n" + json.dumps(result["metrics"], indent=2) + "\n\n")
        f.write("Validation accuracy per round\n")
        for h in result["history"]:
            f.write(f"  {h['epoch']:3d}  {h['val_accuracy']:.4f}\n")
    return model_path


def checkpoint_like(key, data, result, s: TrainSettings):
    """The same information a saved model file holds, for reading right after training."""
    return {"input_vocab": data["in_vocab"], "output_vocab": data["out_vocab"], "key": key.to_json(),
            "metrics": result["metrics"], "settings": {**asdict(s), "digit_mode_info": data.get("digit_info")}}


# -----------------------------------------------------------------------------
# Reading a transcription with one model
# -----------------------------------------------------------------------------
def read_lines(model, checkpoint, numbered_lines, use_separators=False):
    """numbered_lines: [(line_no, symbols)]. Returns readings: dicts with line, symbols, units, confidences.
    For digit models (see digits.py) the symbols are the codes as split by the model; spaces in the
    transcription are ignored unless use_separators is True."""
    if info_of(checkpoint):
        return read_digit_lines(model, checkpoint, numbered_lines, use_separators)
    preds = predict(model, [s for _, s in numbered_lines], checkpoint["input_vocab"], checkpoint["output_vocab"])
    return [{"line": no, "symbols": syms, "units": [u for u, _ in p], "confidence": [c for _, c in p]}
            for (no, syms), p in zip(numbered_lines, preds)]


def share_unsure(predictions, symbol_lines, skip_symbols=(), threshold=0.9):
    """Share of symbols the model is unsure about (confidence below `threshold`), leaving out
    positions whose cipher symbol is in `skip_symbols`."""
    vals = [c for line, syms in zip(predictions, symbol_lines)
            for (_, c), sym in zip(line, syms) if sym not in skip_symbols]
    return sum(c < threshold for c in vals) / max(1, len(vals))


def fit_figures(readings, checkpoint):
    """Fit figures for a reading. The 'unsure' share leaves out symbols the model cannot know:
    symbols not in the key, and symbols of key units that never occurred in its training text
    (e.g. punctuation or nomenclature). Older model files do not record the second group."""
    key_units = checkpoint["key"]["units"]
    known = {s for syms in key_units.values() for s in syms}
    symbols = [s for r in readings for s in r["symbols"]]
    unknown = Counter(s for s in symbols if s not in known)
    m = checkpoint.get("metrics", {})
    untrained = m.get("units_not_in_training")          # None for older model files
    untrained_symbols = {s for u in (untrained or []) for s in key_units.get(u, [])}
    untrained_found = Counter(s for s in symbols if s in untrained_symbols)
    preds = [list(zip(r["units"], r["confidence"])) for r in readings]
    return {
        "symbols": len(symbols),
        "unknown_symbols": sum(unknown.values()),
        "unknown_share": sum(unknown.values()) / max(1, len(symbols)),
        "most_common_unknown": unknown.most_common(10),
        "coverage": 1 - sum(unknown.values()) / max(1, len(symbols)),
        "untrained_units": untrained,
        "untrained_symbols": sum(untrained_found.values()),
        "uncertain_share": share_unsure(preds, [r["symbols"] for r in readings],
                                        set(unknown) | untrained_symbols),
        "ref_in_key": m.get("uncertain_share_in_key_reference"),
        "ref_scrambled": m.get("uncertain_share_scrambled_key_reference"),
    }


def fit_score(f):
    """0 = looks like text from another key, 1 = looks like text in this key (rough guide)."""
    a, b = f["ref_in_key"], f["ref_scrambled"]
    if a is None or b is None or b - a < 1e-6:
        return None
    return max(0.0, min(1.0, (b - f["uncertain_share"]) / (b - a)))


def describe_fit(f, log=print):
    say(log, "Does this text fit the key?")
    unk = f"  Symbols not in the key: {f['unknown_symbols']} of {f['symbols']} ({f['unknown_share']:.1%})"
    if f["most_common_unknown"]:
        unk += ": " + ", ".join(f"{s!r}×{n}" for s, n in f["most_common_unknown"])
    say(log, unk)
    if f.get("untrained_units") is None:
        say(log, "  (This model file is from an older version: symbols of key units that never occurred in its\n"
                 "   training text are counted below. Retrain the model to leave them out.)")
    elif f.get("untrained_symbols"):
        say(log, f"  Symbols for key units never seen in training: {f['untrained_symbols']} "
                 f"(units: {', '.join(repr(u) for u in f['untrained_units'])})")
    say(log, "  Share of symbols the model is unsure about (confidence below 0.9; the symbols above are left out):")
    say(log, f"    your text:                              {f['uncertain_share']:6.1%}")
    if f["ref_in_key"] is not None:
        say(log, f"    for comparison, text in this key:       {f['ref_in_key']:6.1%}")
        say(log, f"    for comparison, text in scrambled keys: {f['ref_scrambled']:6.1%}")
    say(log, "  The most reliable test is to read the output: text written with this key comes out mostly\n"
             "  readable, while text from another key comes out as nonsense or repeated letters. The\n"
             "  figures above are a rough guide only and have not yet been validated on real ciphers.")
    if f["unknown_share"] > 0.10:
        say(log, "\n  Many symbols are not in the key. Check that the transcription writes symbols exactly\n"
                 "  as the key does, and that the transcription format is right.")


# Markers used in the aligned reading. Each has a short form and a full tag; the full tag is used
# whenever the short form is itself a plaintext unit of the key (so it could be mistaken for one).
READING_MARKERS = {
    "space": ("_", "<SPACE>"),     # word gap
    "null": ("·", "<NULL>"),       # null (meaningless filler)
    "none": ("?", "<NONE>"),       # the model gave no plaintext unit (internal token)
}
READING_WIDTH = 100                # aligned rows are wrapped at about this many characters


def reading_markers(key_units):
    """Markers for this key: short forms unless they clash with a plaintext unit of the key."""
    units = set(key_units)
    m = {k: (short if short not in units else full) for k, (short, full) in READING_MARKERS.items()}
    # Symbols the model cannot know are shown as [symbol]; <?symbol> if the key has bracketed units.
    m["bracketed"] = not any(u.startswith("[") and u.endswith("]") for u in units)
    return m


def _symbol_notes(checkpoint):
    """(known symbols, symbols of units never seen in training or None) for a checkpoint."""
    if not checkpoint:
        return None, None
    key_units = checkpoint["key"]["units"]
    known = {s for syms in key_units.values() for s in syms}
    untrained = checkpoint.get("metrics", {}).get("units_not_in_training")
    untrained_symbols = None if untrained is None else {s for u in untrained for s in key_units.get(u, [])}
    return known, untrained_symbols


def aligned_reading(reading, checkpoint=None):
    """One entry per cipher symbol: (symbol, shown_as, note, flowing_text_piece).
    note is 'not in key', 'never in training' or ''."""
    key_units = list(checkpoint["key"]["units"]) if checkpoint else []
    m = reading_markers(key_units)
    known, untrained_symbols = _symbol_notes(checkpoint)
    entries, last_letter = [], ""
    for sym, unit in zip(reading["symbols"], reading["units"]):
        note = ""
        if known is not None and sym not in known:
            note = "not in key"
        elif untrained_symbols and sym in untrained_symbols:
            note = "never in training"
        if note:
            shown = f"[{sym}]" if m["bracketed"] else f"<?{sym}>"
            piece = shown
        elif unit == SPACE_UNIT:
            shown, piece = m["space"], " "
        elif unit == NULL_UNIT:
            shown, piece = m["null"], ""
        elif unit == DOUBLE_UNIT:
            shown = piece = last_letter or m["none"]
        elif unit in SPECIAL_TOKENS:
            shown, piece = m["none"], "?"
        else:
            shown = piece = unit
            last_letter = unit
        entries.append((sym, shown, note, piece))
    return entries


def _aligned_rows(entries, width=READING_WIDTH):
    """Cipher and reading rows with each symbol above its reading, wrapped into blocks."""
    blocks, top, bottom, used = [], [], [], 0
    for sym, shown, _, _ in entries:
        w = max(len(sym), len(shown))
        if top and used + w > width:
            blocks.append((top, bottom))
            top, bottom, used = [], [], 0
        top.append(sym.ljust(w))
        bottom.append(shown.ljust(w))
        used += w + 2
    if top:
        blocks.append((top, bottom))
    return [("  ".join(t).rstrip(), "  ".join(b).rstrip()) for t, b in blocks]


def write_reading(out_dir, name, readings, symbol_format, title, checkpoint=None):
    """Writes <name>_decipherment.txt (each cipher symbol aligned above its reading, plus the
    flowing text) and <name>_decipherment_aligned.csv; returns both paths. With `checkpoint`
    (the model's key and metrics), symbols not in the key or of units never seen in training are
    shown as [symbol] instead of the model's guess (the guess stays in the CSV)."""
    os.makedirs(out_dir, exist_ok=True)
    txt = os.path.join(out_dir, f"{name}_decipherment.txt")
    tab = os.path.join(out_dir, f"{name}_decipherment_aligned.csv")
    m = reading_markers(list(checkpoint["key"]["units"]) if checkpoint else [])
    unk = "[symbol]" if m["bracketed"] else "<?symbol>"
    legend = (f"Each cipher symbol is shown above its reading. {m['space']} = word gap, {m['null']} = null, "
              f"{m['none']} = no reading"
              + (f", {unk} = symbol not in the key or of a unit never seen in training (the model's guess "
                 "is in the CSV)" if checkpoint else "")
              + ". The double sign is shown as the repeated letter. 'text:' is the flowing reading."
              + (" Digit cipher: the digits were read as one stream and split into codes by the model, "
                 "using only codes of the key (model + key); each code is shown above its reading."
                 if checkpoint and info_of(checkpoint) else ""))
    with open(txt, "w", encoding="utf-8") as f:
        f.write(title + "\nDraft output for expert review. Confidence is the model's own certainty (0–1).\n"
                + legend + "\n\n")
        for r in readings:
            entries = aligned_reading(r, checkpoint)
            conf = sum(r["confidence"]) / max(1, len(r["confidence"]))
            f.write(f"[line {r['line']}]  confidence {conf:.2f}\n")
            for top, bottom in _aligned_rows(entries):
                f.write(f"  cipher: {top}\n  reads:  {bottom}\n")
            f.write(f"  text:   {''.join(e[3] for e in entries)}\n\n")
    with open(tab, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "position", "cipher_symbol", "plaintext_unit", "shown_as", "confidence", "note"])
        for r in readings:
            for pos, ((s, shown, note, _), u, c) in enumerate(zip(aligned_reading(r, checkpoint),
                                                                    r["units"], r["confidence"]), 1):
                w.writerow([r["line"], pos, s, u, shown, f"{c:.3f}", note])
    return txt, tab


def preview(readings, n=12, log=print):
    for r in readings[:n]:
        conf = sum(r["confidence"]) / max(1, len(r["confidence"]))
        say(log, f"  [{r['line']:>4}] ({conf:.2f})  {readable(r['units'])}")
    if len(readings) > n:
        say(log, f"  ... {len(readings) - n} more lines in the decipherment file.")

