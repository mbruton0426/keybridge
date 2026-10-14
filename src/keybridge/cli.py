"""Command-line interface: the same workflows as the notebooks, run on your own computer.

    keybridge train   --key my_key.tsv --format space --plaintext text1.txt text2.txt --name my_cipher
    keybridge read    --model my_cipher_model.pt --transcription letter.txt
    keybridge scan    --transcription letter.txt --format char --models all
    keybridge convert --key old_key.tsv --transcription letter.txt     (to the standard format)
    keybridge models  list | download
    keybridge languages
    keybridge evaluate --model my_cipher_model.pt --transcription letter.txt   (research: F1 against the key)
    keybridge index-entry my_cipher_model.pt --add-to index.json      (for maintainers)

Run any command with --help for its options.
"""
import argparse
import importlib.util
import json
import os
import sys

from . import __version__
from .common import StopHere, resolve_format, safe_name
from .config import PLAINTEXT_DATASETS


def _load_custom_edits(path):
    if not path:
        return None, None
    spec = importlib.util.spec_from_file_location("custom_edits", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return (getattr(mod, "custom_edit_training_units", None),
            getattr(mod, "custom_edit_transcription_line", None))


def cmd_train(a):
    from .data import (load_builtin_plaintext, load_plaintext_files, parse_replacements,
                       resolve_plaintext_language)
    from .keys import load_key
    from .pipeline import plaintext_chars_needed, TrainSettings, describe_fit, fit_figures, prepare_data, preview, read_lines, \
        save_training_outputs, train, write_reading
    from .transcription import read_transcription
    from .model import load_model

    fmt = resolve_format(a.format)
    name = safe_name(a.name)
    s = TrainSettings(project=name, symbol_format=fmt,
                      plaintext_source="own files" if a.plaintext else "English", plaintext_files=a.plaintext or [],
                      letter_replacements=a.replacements, word_spaces=a.word_spaces,
                      training_examples=a.examples, sequence_length=a.length, null_rate=a.null_rate,
                      double_rate=a.double_rate, rare_unit_rate=a.rare_rate,
                      search_trials=a.search_trials, seed=a.seed, max_epochs=a.max_epochs)
    train_edit, line_edit = _load_custom_edits(a.custom_edits)
    key = load_key(a.key, fmt)
    print(key.summary())
    for w in key.warnings:
        print("  •", w)
    reps = parse_replacements(a.replacements)
    heldout = None
    if a.plaintext:
        docs = load_plaintext_files(a.plaintext, reps)
    else:
        # Built-in plaintext: English unless --language says otherwise.
        lang = resolve_plaintext_language(a.language)
        s.plaintext_source = lang
        print(f"Using the built-in {lang} plaintext.")
        train_chars, heldout_chars = plaintext_chars_needed(a.examples, a.length)
        docs = load_builtin_plaintext(lang, reps, train_chars, "train")
        heldout = load_builtin_plaintext(lang, reps, heldout_chars, "validation")
    data = prepare_data(key, docs, s, custom_edit=train_edit, heldout_documents=heldout)
    result = train(key, data, s)
    out_dir = os.path.join(a.out, name)
    model_path = save_training_outputs(out_dir, s, key, a.key, data, result)
    print(f"\nSaved to {out_dir}")
    if a.transcription:
        model, ck = load_model(model_path)
        lines = read_transcription(a.transcription, fmt, a.skip.split(","), line_edit, a.length)
        readings = read_lines(model, ck, lines)
        write_reading(out_dir, name, readings, fmt, f"Decipherment of {os.path.basename(a.transcription)}",
                      checkpoint=ck)
        preview(readings)
        describe_fit(fit_figures(readings, ck))


def _registry(a):
    from .registry import ModelRegistry
    return ModelRegistry(a.models_from)


def cmd_read(a):
    from .model import load_model
    from .pipeline import describe_fit, fit_figures, preview, read_lines, write_reading
    from .transcription import read_transcription

    path = a.model if os.path.exists(a.model) else None
    if path is None:
        reg = _registry(a)
        path = reg.model_path(reg.entries(ids=[a.model])[0])
    model, ck = load_model(path)
    fmt = resolve_format(a.format) if a.format else ck["settings"].get("symbol_format")
    _, line_edit = _load_custom_edits(a.custom_edits)
    lines = read_transcription(a.transcription, fmt, a.skip.split(","), line_edit,
                               ck["settings"].get("sequence_length"))
    readings = read_lines(model, ck, lines)
    name = safe_name(os.path.splitext(os.path.basename(a.transcription))[0])
    files = write_reading(a.out, name, readings, fmt,
                          f"Decipherment of {os.path.basename(a.transcription)} with {os.path.basename(path)}",
                          checkpoint=ck)
    print(f"Read {len(readings)} lines:\n")
    preview(readings)
    print()
    describe_fit(fit_figures(readings, ck))
    print("\nWritten:", *files, sep="\n  ")


def cmd_scan(a):
    from .scan import describe_ranking, scan, top_group, write_scan_outputs
    from .transcription import read_transcription

    fmt = resolve_format(a.format)
    _, line_edit = _load_custom_edits(a.custom_edits)
    lines = read_transcription(a.transcription, fmt, a.skip.split(","), line_edit, a.max_length)
    entries, reg = [], None
    if a.models and a.models != "none":
        reg = _registry(a)
        if a.models in ("all", "digits", "characters", "named"):
            entries = reg.entries(symbol_type=None if a.models == "all" else a.models)
        else:
            entries = reg.entries(ids=[m.strip() for m in a.models.split(",")])
    if not entries and not a.model_file:
        raise StopHere("No models to test. Give your own with --model-file.")
    print(f"Testing {len(entries) + len(a.model_file or [])} model(s) on {len(lines)} lines...")
    langs = set(a.languages.split(",")) if a.languages else None
    results = scan(lines, fmt, reg, entries, a.model_file or [], langs, a.min_coverage)
    top = top_group(results, a.margin, a.max_close)
    print()
    describe_ranking(results, top)
    name = safe_name(os.path.splitext(os.path.basename(a.transcription))[0])
    all_files, _ = write_scan_outputs(a.out, name, results, top, os.path.basename(a.transcription), a.all_outputs)
    print("\nWritten:", *all_files, sep="\n  ")


def cmd_evaluate(a):
    from .evaluate import describe, evaluate, write_evaluation
    from .keys import load_key
    from .model import load_model
    from .transcription import read_transcription

    path = a.model if os.path.exists(a.model) else None
    if path is None:
        reg = _registry(a)
        path = reg.model_path(reg.entries(ids=[a.model])[0])
    model, ck = load_model(path)
    fmt = resolve_format(a.format) if a.format else ck["settings"].get("symbol_format")
    gold_key = load_key(a.key) if a.key else None
    _, line_edit = _load_custom_edits(a.custom_edits)
    lines = read_transcription(a.transcription, fmt, a.skip.split(","), line_edit,
                               ck["settings"].get("sequence_length"))
    exclude = [u.strip().upper() for u in a.exclude_units.split(",") if u.strip()] if a.exclude_units else []
    result, units, per_line, readings = evaluate(model, ck, lines, gold_key, a.exclude_unknown, exclude,
                                                 a.bootstrap, a.seed)
    describe(result, units)
    name = safe_name(a.name or os.path.splitext(os.path.basename(a.transcription))[0])
    files = write_evaluation(a.out, name, result, units, per_line, readings)
    if a.reference_plaintext:
        from .evaluate import (describe_reference_check, read_reference_plaintext, reference_check,
                               write_reference_check)
        from .keys import CipherKey
        ref_lines = read_reference_plaintext(a.reference_plaintext, a.skip.split(","))
        summary, rows = reference_check(readings, gold_key or CipherKey.from_json(ck["key"]), ref_lines)
        describe_reference_check(summary)
        files += write_reference_check(a.out, name, summary, rows)
    print("\nWritten:", *files, sep="\n  ")


def cmd_convert(a):
    from .convert import convert_key, convert_transcription
    if not a.key and not a.transcription:
        raise StopHere("Give --key and/or --transcription to convert.")
    if a.key:
        out, key = convert_key(a.key, a.key_out)
        print(f"Key written in the standard format: {out}"
              + (' (blank symbol " " written as "_")' if key.blank_converted else ""))
    for path in a.transcription or []:
        out, n = convert_transcription(path, None, a.skip.split(","))
        print(f"Transcription converted ({n} lines): {out}")


def cmd_models(a):
    reg = _registry(a)
    entries = reg.entries()
    if a.action == "list":
        print(f"{len(entries)} model(s) in {reg.source}:")
        for e in entries:
            print(f"  {e['id']:<28} {e.get('symbol_format', ''):<28} {e.get('symbol_type', ''):<10} {e.get('name', '')}")
    else:
        for e in entries:
            print("  downloading", e["id"], "->", reg.model_path(e))
        print("Done. To work offline from now on, set the environment variable HF_HUB_OFFLINE=1.")


def cmd_languages(a):
    from .languages import available_languages
    langs = available_languages()
    if not langs:
        print("No language profiles installed.")
    for code, (name, path) in sorted(langs.items()):
        print(f"  {code:<4} {name:<12} {path}")


def cmd_index_entry(a):
    from .registry import make_index_entry, new_index
    entry = make_index_entry(a.model, a.id)
    if a.add_to:
        index = new_index()
        if os.path.exists(a.add_to):
            with open(a.add_to, encoding="utf-8") as f:
                index = json.load(f)
        index["models"] = [m for m in index["models"] if m["id"] != entry["id"]] + [entry]
        with open(a.add_to, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        print(f"Added '{entry['id']}' to {a.add_to}. Complete the fields marked FILL IN.")
    else:
        print(json.dumps(entry, ensure_ascii=False, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="keybridge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    fmt_help = "transcription format: 'space' (symbols separated by spaces) or 'char' (one character per symbol)"

    t = sub.add_parser("train", help="train a model for a key")
    t.add_argument("--key", required=True, help="key file (.tsv)")
    t.add_argument("--format", required=True, help=fmt_help)
    t.add_argument("--plaintext", nargs="+", help="your own plaintext .txt file(s); without this, built-in plaintext is used")
    t.add_argument("--language", default="English",
                   help="language of the built-in plaintext (default English): " + ", ".join(PLAINTEXT_DATASETS))
    t.add_argument("--name", default="my_cipher")
    t.add_argument("--replacements", default="", help="letter replacements, e.g. 'J=I, V=U'")
    t.add_argument("--word-spaces", default="keep real word boundaries",
                   choices=["keep real word boundaries", "insert at random", "remove"])
    t.add_argument("--examples", type=int, default=5000, help="number of synthetic training examples (default 5000)")
    t.add_argument("--length", type=int, default=50)
    t.add_argument("--null-rate", type=float, default=0.05)
    t.add_argument("--double-rate", type=float, default=0.5)
    t.add_argument("--rare-rate", type=float, default=0.02,
                   help="insert key units that are rare or missing in the plaintext at this rate per word gap (default 0.02; 0 = off)")
    t.add_argument("--search-trials", type=int, default=20,
                   help="short trials of the model-settings search (default 20; 0 = skip the search and use the fixed default settings)")
    t.add_argument("--seed", type=int, default=9)
    t.add_argument("--max-epochs", type=int, default=100, help="upper limit on training rounds")
    t.add_argument("--transcription", help="also read this transcription after training")
    t.add_argument("--skip", default="##", help="skip transcription lines starting with these (comma-separated)")
    t.add_argument("--custom-edits", help="Python file defining custom_edit_training_units / custom_edit_transcription_line")
    t.add_argument("--out", default="outputs")
    t.set_defaults(func=cmd_train)

    for name, func, helptext in (("read", cmd_read, "read a transcription with one model"),
                                 ("scan", cmd_scan, "test a transcription against many models")):
        r = sub.add_parser(name, help=helptext)
        r.add_argument("--transcription", required=True)
        r.add_argument("--skip", default="##")
        r.add_argument("--custom-edits")
        r.add_argument("--out", default="outputs")
        r.add_argument("--models-from", help="Hugging Face repo id or local folder of the model collection")
        r.set_defaults(func=func)
        if name == "read":
            r.add_argument("--model", required=True, help="model file (.pt) or model ID from the collection")
            r.add_argument("--format", help=fmt_help + " (default: the model's own)")
        else:
            r.add_argument("--format", required=True, help=fmt_help)
            r.add_argument("--models", default="all",
                           help="'all', 'digits', 'characters', 'named', comma-separated model IDs, or 'none'")
            r.add_argument("--model-file", action="append", help="your own model file (repeatable)")
            r.add_argument("--languages", help="limit language scoring to these codes, e.g. de,la,it")
            r.add_argument("--min-coverage", type=float, default=0.75)
            r.add_argument("--margin", type=float, default=0.05)
            r.add_argument("--max-close", type=int, default=3)
            r.add_argument("--max-length", type=int, default=50, help="longer lines are read in pieces")
            r.add_argument("--all-outputs", action="store_true", help="write decipherments for every model")

    ev = sub.add_parser("evaluate", help="(research) score a model's reading against the key")
    ev.add_argument("--model", required=True, help="model file (.pt) or model ID from the collection")
    ev.add_argument("--transcription", required=True)
    ev.add_argument("--format", help=fmt_help + " (default: the model's own)")
    ev.add_argument("--key", help="key to derive the reference from (default: the key stored in the model)")
    ev.add_argument("--reference-plaintext", help="the archive's plaintext transcription, line by line, to check "
                                                   "the key-derived reference against")
    ev.add_argument("--skip", default="##")
    ev.add_argument("--custom-edits", help="Python file defining custom_edit_transcription_line")
    ev.add_argument("--exclude-unknown", action="store_true", help="do not score symbols that are not in the key")
    ev.add_argument("--exclude-units", help="comma-separated plaintext units not to score, e.g. 'MASTER,LODGE'")
    ev.add_argument("--bootstrap", type=int, default=1000, help="line-bootstrap resamples for CIs (0: none)")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--name", help="name for the output files (default: transcription file name)")
    ev.add_argument("--models-from")
    ev.add_argument("--out", default="outputs")
    ev.set_defaults(func=cmd_evaluate)

    cv = sub.add_parser("convert", help="convert a key and/or 'one character per symbol' transcriptions "
                                        "to the standard format")
    cv.add_argument("--key", help="key file to rewrite in the standard format")
    cv.add_argument("--key-out", help="output file for the key (default: <name>_standard.tsv)")
    cv.add_argument("--transcription", nargs="+", help="'one character per symbol' transcription(s) to convert "
                                                       "(each written to <name>_standard.txt)")
    cv.add_argument("--skip", default="##", help="note lines starting with these are copied unchanged")
    cv.set_defaults(func=cmd_convert)

    m = sub.add_parser("models", help="list or download the shared models")
    m.add_argument("action", choices=["list", "download"])
    m.add_argument("--models-from")
    m.set_defaults(func=cmd_models)

    sub.add_parser("languages", help="list installed language profiles").set_defaults(func=cmd_languages)

    ie = sub.add_parser("index-entry", help="(maintainers) make an index.json entry for a model")
    ie.add_argument("model")
    ie.add_argument("--id")
    ie.add_argument("--add-to", help="index.json to add the entry to (created if missing)")
    ie.set_defaults(func=cmd_index_entry)

    a = p.parse_args(argv)
    try:
        a.func(a)
    except StopHere as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

