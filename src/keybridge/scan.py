"""Testing one transcription against many models and ranking the results.

For each model whose transcription format matches:
  coverage      share of the transcription's symbols that exist in the model's key
                (checked from the index before downloading anything);
  fit           0–1, where the share of uncertain symbols falls between that model's own
                reference for scrambled keys (0) and for text in its key (1);
  plausibility  0–1, how much the decoded output reads like real text in its best
                matching language (from character-trigram language profiles);
  score         coverage × the mean of fit and plausibility.
All of these are rough guides that still need validating on known ciphertext/key pairs.
"""
import csv
import gc
import os

import torch

from .common import say
from .data import parse_replacements, readable
from .digits import info_of
from .languages import available_languages, rank_languages
from .model import load_model
from .pipeline import fit_figures, fit_score, read_lines, write_reading
from .registry import entry_symbol_coverage

DEFAULT_MIN_COVERAGE = 0.75
DEFAULT_MARGIN = 0.05
DEFAULT_MAX_CLOSE = 3


def _score_one(model, ck, numbered_lines, languages, log, use_separators=False):
    readings = read_lines(model, ck, numbered_lines, use_separators)
    figs = fit_figures(readings, ck)
    fit = fit_score(figs)
    units = [u for r in readings for u in r["units"] + [" "]]
    reps = parse_replacements(ck.get("settings", {}).get("letter_replacements", ""))
    langs = rank_languages(units, list(ck["key"]["units"]), reps, languages)
    plaus = max(0.0, min(1.0, langs[0][2])) if langs else None
    # A model without usable reference figures gets no credit for fit, so that language
    # plausibility alone cannot push it up the ranking. Missing language profiles are simply
    # left out (the ranking then uses fit only).
    parts = [fit if fit is not None else 0.0] + ([plaus] if plaus is not None else [])
    combined = figs["coverage"] * sum(parts) / len(parts)
    return readings, {"coverage": figs["coverage"], "uncertain_share": figs["uncertain_share"],
                      "fit": fit, "plausibility": plaus, "languages": langs[:3], "score": combined}


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
    say(log, f"{'rank':>4}  {'model':<28} {'score':>5}  {'fit':>4}  {'reads like':<24} {'symbols in key':>14}")
    for i, r in enumerate(results[:n], 1):
        langs = r.get("languages") or []
        lang = f"{langs[0][1]} ({langs[0][2]:.2f})" if langs else ("—" if r.get("readings") else r["status"])
        mark = " ◀" if r in top else ""
        say(log, f"{i:>4}  {r['id'][:28]:<28} {_fmt(r.get('score')):>5}  {_fmt(r.get('fit')):>4}  "
                 f"{lang[:24]:<24} {_fmt(r.get('coverage'), True):>14}{mark}")
    if len(results) > n:
        say(log, f"  ... {len(results) - n} more in the ranking file.")
    if top:
        best = top[0].get("score") or 0
        if best < 0.5:
            say(log, "\nNone of the models looks like a clear fit (best score below 0.5). "
                     "Read the previews before drawing conclusions.")
        elif len(top) > 1:
            say(log, f"\n{len(top)} models score within the margin of the best; compare their outputs.")


def write_scan_outputs(out_dir, name, results, top, transcription_name, write_all=False):
    """Writes the ranking files and decipherments; returns (all files, files for the top group)."""
    os.makedirs(out_dir, exist_ok=True)
    rank_csv = os.path.join(out_dir, f"{name}_ranking.csv")
    with open(rank_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "model_id", "model_name", "status", "score", "fit", "plausibility",
                    "coverage", "uncertain_share", "language_1", "language_1_plausibility",
                    "language_2", "language_2_plausibility", "language_3", "language_3_plausibility",
                    "in_top_group"])
        for i, r in enumerate(results, 1):
            langs = (r.get("languages") or []) + [("", "", None)] * 3
            row = [i, r["id"], r["name"], r["status"], r.get("score"), r.get("fit"), r.get("plausibility"),
                   r.get("coverage"), r.get("uncertain_share")]
            for code, lname, p in langs[:3]:
                row += [lname, "" if p is None else round(p, 3)]
            w.writerow(row + ["yes" if r in top else ""])
    rank_txt = os.path.join(out_dir, f"{name}_ranking.txt")
    lines = []
    describe_ranking(results, top, log=lines.append, n=len(results))
    with open(rank_txt, "w", encoding="utf-8") as f:
        f.write(f"Scan of {transcription_name}\n")
        f.write("Scores are rough guides that have not yet been validated. Always read the outputs.\n\n")
        f.write("\n".join(lines) + "\n")
        for r in top:
            f.write(f"\n--- {r['id']}: first lines ---\n")
            for rd in r["readings"][:10]:
                f.write(f"  [{rd['line']}] {readable(rd['units'])}\n")
    all_files, top_files = [rank_csv, rank_txt], [rank_csv, rank_txt]
    for r in results:
        if r.get("readings") and (write_all or r in top):
            files = write_reading(out_dir, f"{name}__{r['id']}", r["readings"], r["symbol_format"],
                                  f"Decipherment of {transcription_name} with model '{r['id']}'",
                                  checkpoint=r.get("checkpoint_info"))
            all_files += files
            if r in top:
                top_files += files
    return all_files, top_files
