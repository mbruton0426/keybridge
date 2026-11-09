"""Testing one transcription against many models and ranking the results.

For each model whose transcription format matches:
  coverage      share of the transcription's symbols that exist in the model's key
                (checked from the index before downloading anything);
  plausibility  0–1, how much the decoded output reads like real text in its best
                matching language (from character-trigram language profiles);
  repetition    0–1, the letter entropy of the output relative to that of ordinary text
                (REFERENCE_ENTROPY bits); repetitive output such as "EEEFEFDE…" gets a low value;
  score         coverage × plausibility × repetition.
The ranking only suggests which draft decryptions to read first: whether a ciphertext belongs to a key is decided by
reading the drafts. (The ranking CSV also lists the model's own certainty ("fit") as a diagnostic; it is not part of
the score, and is used in plausibility's place only when no language profiles are installed.)
"""
import csv
import gc
import os

import torch

from .common import say
from .data import parse_replacements, readable
from .digits import info_of
from .languages import available_languages, key_alphabet, prepare_output_text, rank_languages, SPACE_UNIT
import math
from collections import Counter
from .model import load_model
from .pipeline import fit_figures, fit_score, read_lines, write_reading
from .registry import entry_symbol_coverage

DEFAULT_MIN_COVERAGE = 0.75
DEFAULT_MARGIN = 0.05
DEFAULT_MAX_CLOSE = 3
DEFAULT_WRITE_TOP = 10      # draft decryptions are written for this many of the best-ranked models
# Letter entropy of ordinary text in European languages is about 4.0-4.2 bits; the reference is set a little
# lower so that real text is not penalised. Not tuned on test letters.
REFERENCE_ENTROPY = 3.8
MIN_LETTERS_FOR_REPETITION = 20


def letter_entropy(units, key_units):
    """Shannon entropy (bits) of the letters of a decoded output, or None if it is too short."""
    alphabet = key_alphabet(key_units)
    text = prepare_output_text(units, alphabet, SPACE_UNIT in key_units).replace(" ", "")
    if len(text) < MIN_LETTERS_FOR_REPETITION:
        return None
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def repetition_factor(entropy):
    return 1.0 if entropy is None else max(0.0, min(1.0, entropy / REFERENCE_ENTROPY))


def _score_one(model, ck, numbered_lines, languages, log, use_separators=False):
    readings = read_lines(model, ck, numbered_lines, use_separators)
    figs = fit_figures(readings, ck)
    fit = fit_score(figs)
    units = [u for r in readings for u in r["units"] + [" "]]
    reps = parse_replacements(ck.get("settings", {}).get("letter_replacements", ""))
    langs = rank_languages(units, list(ck["key"]["units"]), reps, languages)
    plaus = max(0.0, min(1.0, langs[0][2])) if langs else None
    entropy = letter_entropy(units, list(ck["key"]["units"]))
    rep = repetition_factor(entropy)
    # Score: coverage x plausibility x repetition (see the module notes).
    # Without language profiles, the model's own certainty (fit) is used in plausibility's place.
    main = plaus if plaus is not None else (fit if fit is not None else 0.0)
    combined = figs["coverage"] * main * rep
    return readings, {"coverage": figs["coverage"], "uncertain_share": figs["uncertain_share"],
                      "fit": fit, "plausibility": plaus, "letter_entropy": entropy, "repetition": rep,
                      "languages": langs[:3], "score": combined}


def scan(numbered_lines, symbol_format, registry=None, entries=(), local_models=(), languages=None,
         min_coverage=DEFAULT_MIN_COVERAGE, log=print, use_separators=False):
    """entries: index entries (from registry.entries()); local_models: paths to .pt files.
    Returns results sorted best first; each has id, name, status, figures and readings.
    Digit models read the transcription digit by digit; spaces are then only used as code
    boundaries if use_separators is True."""
    results = []
    jobs = [("index", e) for e in entries] + [("file", p) for p in local_models]
    if not available_languages():
        say(log, "Note: no language profiles are installed, so outputs are ranked by fit only.")
    for i, (kind, item) in enumerate(jobs, 1):
        if kind == "index":
            mid, name = item["id"], item.get("name", item["id"])
            cov = entry_symbol_coverage(item.get("symbols", []), numbered_lines, item.get("digit_mode"))
            if cov < min_coverage:
                results.append({"id": mid, "name": name, "status": "symbols do not match the key",
                                "coverage": cov, "score": 0.0})
                say(log, f"  [{i}/{len(jobs)}] {mid}: skipped, only {cov:.0%} of symbols are in this key")
                continue
            path = registry.model_path(item)
        else:
            path = item
            mid = name = os.path.basename(path).replace("_model.pt", "").replace(".pt", "")
        model, ck = load_model(path)
        readings, figs = _score_one(model, ck, numbered_lines, languages, log, use_separators)
        status = "scored"
        if figs["coverage"] < min_coverage:
            status, figs["score"] = "symbols do not match the key", 0.0
        results.append({"id": mid, "name": name, "status": status, "readings": readings,
                        "symbol_format": symbol_format,
                        "checkpoint_info": {"key": ck["key"], "metrics": ck.get("metrics", {}),
                                            "settings": {"digit_mode_info": info_of(ck)}}, **figs})
        lang = f", reads most like {figs['languages'][0][1]}" if figs.get("languages") else ""
        sc = f"{figs['score']:.2f}" if figs["score"] is not None else "n/a"
        say(log, f"  [{i}/{len(jobs)}] {mid}: score {sc}{lang}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results.sort(key=lambda r: -(r.get("score") or 0.0))
    return results


def top_group(results, margin=DEFAULT_MARGIN, max_close=DEFAULT_MAX_CLOSE):
    """The best result plus any within `margin` of it (at most max_close)."""
    scored = [r for r in results if r.get("readings")]
    if not scored:
        return []
    best = scored[0].get("score") or 0.0
    return [r for r in scored if (r.get("score") or 0.0) >= best - margin][:max_close]


def _fmt(x, pct=False):
    if x is None:
        return "n/a"
    return f"{x:.0%}" if pct else f"{x:.2f}"


def describe_ranking(results, top, log=print, n=15):
    say(log, f"{'rank':>4}  {'model':<28} {'score':>5}  {'reads like':<24} {'repetition':>10} {'symbols in key':>14}")
    for i, r in enumerate(results[:n], 1):
        langs = r.get("languages") or []
        lang = f"{langs[0][1]} ({langs[0][2]:.2f})" if langs else ("—" if r.get("readings") else r["status"])
        mark = " ◀" if r in top else ""
        say(log, f"{i:>4}  {r['id'][:28]:<28} {_fmt(r.get('score')):>5}  {lang[:24]:<24} "
                 f"{_fmt(r.get('repetition')):>10} {_fmt(r.get('coverage'), True):>14}{mark}")
    if len(results) > n:
        say(log, f"  ... {len(results) - n} more in the ranking file.")
    if top:
        best = top[0].get("score") or 0
        if best < 0.5:
            say(log, "\nNone of the models looks like a clear fit (best score below 0.5). "
                     "Read the previews before drawing conclusions.")
        elif len(top) > 1:
            say(log, f"\n{len(top)} models score within the margin of the best; compare their outputs.")


def write_scan_outputs(out_dir, name, results, top, transcription_name, write_all=False,
                       write_top=DEFAULT_WRITE_TOP):
    """Writes the ranking files and draft decryptions; returns (all files, files to download first: the
    top group and the write_top best-ranked drafts).
    Decryptions are written for the top group and the write_top best-ranked models that produced a
    reading (every model if write_all): the right key is not always ranked first, so read several."""
    os.makedirs(out_dir, exist_ok=True)
    rank_csv = os.path.join(out_dir, f"{name}_ranking.csv")
    with open(rank_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "model_id", "model_name", "status", "score", "plausibility", "repetition",
                    "coverage", "language_1", "language_1_plausibility",
                    "language_2", "language_2_plausibility", "language_3", "language_3_plausibility",
                    "in_top_group", "letter_entropy", "uncertain_share", "fit_diagnostic"])
        for i, r in enumerate(results, 1):
            langs = (r.get("languages") or []) + [("", "", None)] * 3
            row = [i, r["id"], r["name"], r["status"], r.get("score"), r.get("plausibility"), r.get("repetition"),
                   r.get("coverage")]
            for code, lname, p in langs[:3]:
                row += [lname, "" if p is None else round(p, 3)]
            w.writerow(row + ["yes" if r in top else "", r.get("letter_entropy"), r.get("uncertain_share"),
                              r.get("fit")])
    rank_txt = os.path.join(out_dir, f"{name}_ranking.txt")
    lines = []
    describe_ranking(results, top, log=lines.append, n=len(results))
    with open(rank_txt, "w", encoding="utf-8") as f:
        f.write(f"Scan of {transcription_name}\n")
        f.write("The ranking only suggests which drafts to read first. Read the draft decryptions to decide.\n\n")
        f.write("\n".join(lines) + "\n")
        for r in top:
            f.write(f"\n--- {r['id']}: first lines ---\n")
            for rd in r["readings"][:10]:
                f.write(f"  [{rd['line']}] {readable(rd['units'])}\n")
    all_files, top_files = [rank_csv, rank_txt], [rank_csv, rank_txt]
    best = [r for r in results if r.get("readings") and r.get("status") == "scored"][:max(0, write_top or 0)]
    for r in results:
        if r.get("readings") and (write_all or r in top or r in best):
            files = write_reading(out_dir, f"{name}__{r['id']}", r["readings"], r["symbol_format"],
                                  f"Draft decryption of {transcription_name} with model '{r['id']}'",
                                  checkpoint=r.get("checkpoint_info"))
            all_files += files
            if r in top or r in best:
                top_files += files
    return all_files, top_files