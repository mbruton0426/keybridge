"""Scoring a model's reading of an authentic transcription against the key (for research).

The reference ("gold") for each cipher symbol is the plaintext unit the key assigns to it,
as in the paper's real-world experiments: the transcription decoded symbol by symbol with
the key. Symbols that are not in the key get the gold label <UNK>; they count as errors
unless excluded with exclude_unknown. Scores are computed per cipher symbol (one plaintext
unit per symbol, so a nomenclature symbol counts once), as support-weighted F1 like the paper.
"""
import csv
import json
import os
import random
from collections import Counter, defaultdict

from .common import UNK, say
from .keys import CipherKey
from .pipeline import read_lines


def _f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return (f1_score(y_true, y_pred, average="weighted", zero_division=0),
            f1_score(y_true, y_pred, average="macro", zero_division=0))


def evaluate(model, checkpoint, numbered_lines, gold_key=None, exclude_unknown=False,
             exclude_units=(), bootstrap=1000, seed=0, log=print):
    key = gold_key or CipherKey.from_json(checkpoint["key"])
    readings = read_lines(model, checkpoint, numbered_lines)
    excluded = set(exclude_units)
    per_line, y_true, y_pred = [], [], []
    for i, r in enumerate(readings):
        t_line, p_line = [], []
        for sym, pred in zip(r["symbols"], r["units"]):
            gold = key.symbol_to_unit.get(sym, UNK)
            if (exclude_unknown and gold == UNK) or gold in excluded:
                continue
            t_line.append(gold)
            p_line.append(pred)
        r["gold"] = t_line
        r["scored_pred"] = p_line
        if t_line:
            correct = sum(a == b for a, b in zip(t_line, p_line))
            per_line.append({"index": i, "line": r["line"], "positions": len(t_line), "correct": correct,
                             "accuracy": correct / len(t_line)})
        y_true += t_line
        y_pred += p_line

    n = len(y_true)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / max(1, n)
    f1w, f1m = _f1(y_true, y_pred)
    line_acc = sorted(l["accuracy"] for l in per_line)
    result = {
        "positions_scored": n,
        "lines_scored": len(per_line),
        "accuracy": acc, "f1_weighted": f1w, "f1_macro": f1m,
        "median_line_accuracy": line_acc[len(line_acc) // 2] if line_acc else None,
        "lines_fully_correct": sum(a == 1.0 for a in line_acc) / max(1, len(line_acc)),
        "lines_at_least_95": sum(a >= 0.95 for a in line_acc) / max(1, len(line_acc)),
        "unknown_symbols_in_transcription": sum(1 for r in readings for s in r["symbols"]
                                                if s not in key.symbol_to_unit),
        "exclude_unknown": exclude_unknown, "excluded_units": sorted(excluded),
    }

    if bootstrap and per_line:
        rng = random.Random(seed)
        lines = [(r["gold"], r["scored_pred"]) for r in readings if r["gold"]]
        accs, f1s = [], []
        for _ in range(bootstrap):
            sample = [lines[rng.randrange(len(lines))] for _ in lines]
            t = [u for g, _ in sample for u in g]
            p = [u for _, pr in sample for u in pr]
            accs.append(sum(a == b for a, b in zip(t, p)) / max(1, len(t)))
            f1s.append(_f1(t, p)[0])
        accs.sort()
        f1s.sort()
        lo, hi = int(0.025 * bootstrap), int(0.975 * bootstrap) - 1
        result["bootstrap_resamples"] = bootstrap
        result["accuracy_95ci"] = [accs[lo], accs[hi]]
        result["f1_weighted_95ci"] = [f1s[lo], f1s[hi]]

    support, correct, confusions = Counter(y_true), Counter(), defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct[t] += 1
        else:
            confusions[t][p] += 1
    units = [{"unit": u, "support": support[u], "correct": correct[u],
              "accuracy": correct[u] / support[u],
              "top_confusions": confusions[u].most_common(3)} for u in support]
    units.sort(key=lambda d: -(d["support"] - d["correct"]))
    errors = n - sum(correct.values())
    result["errors"] = errors
    return result, units, per_line, readings


def _pct(x):
    return "n/a" if x is None else f"{x:.3f}"


def describe(result, units, log=print, top=15):
    say(log, f"Scored {result['positions_scored']} positions on {result['lines_scored']} lines"
             + (" (symbols not in the key excluded)" if result["exclude_unknown"] else "") + ".")
    ci = result.get("f1_weighted_95ci")
    say(log, f"  F1 (weighted): {_pct(result['f1_weighted'])}"
             + (f"   95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
    ci = result.get("accuracy_95ci")
    say(log, f"  Accuracy:      {_pct(result['accuracy'])}"
             + (f"   95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
    say(log, f"  F1 (macro):    {_pct(result['f1_macro'])}")
    say(log, f"  Lines fully correct: {result['lines_fully_correct']:.1%}   "
             f"lines ≥95% correct: {result['lines_at_least_95']:.1%}   "
             f"median line accuracy: {_pct(result['median_line_accuracy'])}")
    if result["unknown_symbols_in_transcription"]:
        say(log, f"  Symbols in the transcription that are not in the key: "
                 f"{result['unknown_symbols_in_transcription']}")
    say(log, f"\nUnits with the most errors ({result['errors']} errors in total):")
    for u in units[:top]:
        if u["support"] == u["correct"]:
            break
        conf = ", ".join(f"{p!r}×{c}" for p, c in u["top_confusions"])
        say(log, f"  {u['unit']!r:<22} {u['support'] - u['correct']:>5} errors of {u['support']:>6}"
                 f"  (accuracy {u['accuracy']:.2f})  read as: {conf}")


def write_evaluation(out_dir, name, result, units, per_line, readings):
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, f"{name}_evaluation{suffix}") for k, suffix in
             (("json", ".json"), ("txt", ".txt"), ("units", "_units.csv"), ("lines", "_lines.csv"))}
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = []
    describe(result, units, log=lines.append, top=len(units))
    with open(paths["txt"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(paths["units"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "support", "correct", "accuracy", "top_confusions"])
        for u in units:
            w.writerow([u["unit"], u["support"], u["correct"], f"{u['accuracy']:.4f}",
                        "; ".join(f"{p}×{c}" for p, c in u["top_confusions"])])
    with open(paths["lines"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "positions", "correct", "accuracy", "gold", "predicted"])
        for l in per_line:
            r = readings[l["index"]]
            w.writerow([l["line"], l["positions"], l["correct"], f"{l['accuracy']:.4f}",
                        "|".join(r["gold"]), "|".join(r["scored_pred"])])
    return list(paths.values())


# -----------------------------------------------------------------------------
# Checking the key-derived reference against the archive's plaintext transcription
# -----------------------------------------------------------------------------
def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _as_written(text):
    return " ".join(text.upper().split())


def _letters_only(text):
    return "".join(c for c in text.upper() if c.isalpha())


def read_reference_plaintext(path, skip_prefixes=("##",)):
    """The archive's plaintext transcription: one line per ciphertext line, same order."""
    from .common import read_text_file
    lines = []
    for raw in read_text_file(path).splitlines():
        if not raw.strip() or any(raw.lstrip().startswith(p) for p in skip_prefixes if p):
            continue
        lines.append(raw.strip())
    return lines


def reference_check(readings, key, reference_lines):
    """Compares, line by line, the plaintext obtained by decoding the transcription with the key
    (the reference used for scoring) and the model's reading with the archive's plaintext.
    Returns (summary, rows). Character error rate (CER) = edit distance / reference length."""
    from .common import stop
    from .data import readable
    order, gold_units, pred_units = [], {}, {}
    for r in readings:
        if r["line"] not in gold_units:
            order.append(r["line"])
            gold_units[r["line"]], pred_units[r["line"]] = [], []
        gold_units[r["line"]] += [key.symbol_to_unit.get(s, UNK) for s in r["symbols"]]
        pred_units[r["line"]] += r["units"]
    if len(order) != len(reference_lines):
        stop(f"The transcription has {len(order)} ciphertext lines but the reference plaintext has "
             f"{len(reference_lines)} lines. They must correspond line by line (after skipping empty lines "
             "and lines starting with the skip markers). Remove notes, headers or catchwords that only "
             "one of the files has.")
    rows = []
    totals = {k: 0 for k in ("ref_written", "ref_letters", "key_written", "key_letters",
                             "model_written", "model_letters")}
    identical = {"key_written": 0, "key_letters": 0}
    for line_no, ref in zip(order, reference_lines):
        key_text, model_text = readable(gold_units[line_no]), readable(pred_units[line_no])
        rw, rl = _as_written(ref), _letters_only(ref)
        kw, kl = _as_written(key_text), _letters_only(key_text)
        mw, ml = _as_written(model_text), _letters_only(model_text)
        d = {"key_written": _edit_distance(kw, rw), "key_letters": _edit_distance(kl, rl),
             "model_written": _edit_distance(mw, rw), "model_letters": _edit_distance(ml, rl)}
        totals["ref_written"] += len(rw)
        totals["ref_letters"] += len(rl)
        for k in d:
            totals[k] += d[k]
        identical["key_written"] += d["key_written"] == 0
        identical["key_letters"] += d["key_letters"] == 0
        rows.append({"line": line_no, "reference": ref, "key_decoded": key_text, "model": model_text, **d})
    n = len(rows)
    summary = {
        "lines_compared": n,
        "key_vs_reference_cer_as_written": totals["key_written"] / max(1, totals["ref_written"]),
        "key_vs_reference_cer_letters_only": totals["key_letters"] / max(1, totals["ref_letters"]),
        "key_vs_reference_identical_lines_as_written": identical["key_written"] / max(1, n),
        "key_vs_reference_identical_lines_letters_only": identical["key_letters"] / max(1, n),
        "model_vs_reference_cer_as_written": totals["model_written"] / max(1, totals["ref_written"]),
        "model_vs_reference_cer_letters_only": totals["model_letters"] / max(1, totals["ref_letters"]),
    }
    return summary, rows


def describe_reference_check(summary, log=print):
    say(log, f"\nReference check against the archive plaintext ({summary['lines_compared']} lines):")
    say(log, "  Key-decoded transcription (the scoring reference) vs archive plaintext:")
    say(log, f"    character error rate: {summary['key_vs_reference_cer_as_written']:.2%} as written, "
             f"{summary['key_vs_reference_cer_letters_only']:.2%} letters only")
    say(log, f"    identical lines:      {summary['key_vs_reference_identical_lines_as_written']:.1%} as written, "
             f"{summary['key_vs_reference_identical_lines_letters_only']:.1%} letters only")
    say(log, "  Model reading vs archive plaintext:")
    say(log, f"    character error rate: {summary['model_vs_reference_cer_as_written']:.2%} as written, "
             f"{summary['model_vs_reference_cer_letters_only']:.2%} letters only")
    say(log, "  ('as written' compares upper-cased text with spacing normalised; 'letters only' ignores\n"
             "   spaces, punctuation and other non-letters.)")


def write_reference_check(out_dir, name, summary, rows):
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, f"{name}_reference_check.json")
    cpath = os.path.join(out_dir, f"{name}_reference_check_lines.csv")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "reference_plaintext", "key_decoded", "model_reading",
                    "key_vs_reference_distance_as_written", "key_vs_reference_distance_letters_only",
                    "model_vs_reference_distance_as_written", "model_vs_reference_distance_letters_only"])
        for r in rows:
            w.writerow([r["line"], r["reference"], r["key_decoded"], r["model"], r["key_written"],
                        r["key_letters"], r["model_written"], r["model_letters"]])
    return [jpath, cpath]


# -----------------------------------------------------------------------------
# Digit models (see digits.py)
# -----------------------------------------------------------------------------
def evaluate_digits(model, checkpoint, numbered_lines, gold_key=None, exclude_units=(), bootstrap=1000,
                    seed=0, log=print):
    """Scores a digit model. The transcription must have the codes separated by spaces (e.g. from an
    archive decryption); the reference for each code is the unit the key gives it. The model reads
    the same digits with the spaces removed, so it has to find the codes itself.

    Two readings are scored, per digit (a code's first digit carries its unit, the others <JOIN>):
      model        the model's own label for each digit;
      model + key  the split into key codes the model finds most likely (what keybridge outputs).
    Codes that are not in the key (illegible signs, '?', ...) have no reference and are not scored."""
    from .common import JOIN_UNIT
    from .data import readable
    from .digits import ambiguous_codes, info_of, piece_regex, read_stream, tokenize_symbols
    info = info_of(checkpoint)
    key = gold_key or CipherKey.from_json(checkpoint["key"])
    excluded = set(exclude_units)
    rx = piece_regex(info["marks"])

    pieces, line_of, gold, scored, gold_code, code_id = [], [], [], [], [], []
    gold_codes = []                     # (start, end, code, unit or None, line index)
    for li, (no, items) in enumerate(numbered_lines):
        for item in items:
            ps = [p for p, _ in tokenize_symbols([item], rx)]
            unit = key.symbol_to_unit.get(item)
            ok = unit is not None and unit not in excluded
            start = len(pieces)
            for k, p in enumerate(ps):
                pieces.append(p)
                line_of.append(li)
                gold.append((unit if k == 0 else JOIN_UNIT) if unit is not None else UNK)
                scored.append(ok)
                gold_code.append(item)
                code_id.append(len(gold_codes))
            gold_codes.append((start, len(pieces), item, unit, li, ok))
    res = read_stream(model, checkpoint, pieces)
    labels_key = [JOIN_UNIT] * len(pieces)
    for s, e, code, unit, conf in res["codes"]:
        labels_key[s] = unit if unit is not None else UNK
    methods = {"model": res["model_labels"], "model + key": labels_key}
    codes_by_method = {"model": [(s, e, u) for s, e, c, u, _ in res["model_codes"]],
                       "model + key": [(s, e, u) for s, e, c, u, _ in res["codes"]]}

    amb = ambiguous_codes(checkpoint["key"]["units"], info["max_pieces"], rx)
    hard = set()
    for i in range(len(pieces)):
        for ps in amb.values():
            if pieces[i:i + len(ps)] == ps:
                hard.update(range(i, i + len(ps)))

    idx = [i for i in range(len(pieces)) if scored[i]]
    gold_spans = {(s, e, u) for s, e, c, u, li, ok in gold_codes if ok}
    n_lines = len(numbered_lines)

    def line_text(spans, keep_start):
        out = [[] for _ in range(n_lines)]
        for s, e, u in spans:
            if keep_start(s):
                out[line_of[s]].append(u if u is not None else UNK)
        return [readable(x) for x in out]
    gold_text = line_text([(s, e, u) for s, e, c, u, li, ok in gold_codes if ok], lambda s: True)

    result = {"digit_mode": True, "digits": len(pieces), "positions_scored": len(idx),
              "codes_scored": len(gold_spans), "lines": n_lines,
              "codes_not_in_key": sum(1 for g in gold_codes if g[3] is None),
              "hard_spot_digits": sum(1 for i in idx if i in hard),
              "excluded_units": sorted(excluded), "methods": {}}
    per_line = {}
    for m, lab in methods.items():
        yt, yp = [gold[i] for i in idx], [lab[i] for i in idx]
        f1w, f1m = _f1(yt, yp)
        st_t, st_p = [gold[i] != JOIN_UNIT for i in idx], [lab[i] != JOIN_UNIT for i in idx]
        tp = sum(a and b for a, b in zip(st_t, st_p))
        fp = sum(b and not a for a, b in zip(st_t, st_p))
        fn = sum(a and not b for a, b in zip(st_t, st_p))
        hidx = [i for i in idx if i in hard]
        # Reading text: only codes that start on a scored digit, so unread or illegible spots do not
        # count against either side.
        pred_text = line_text(codes_by_method[m], lambda s: scored[s])
        e_w = e_l = n_w = n_l = 0
        for g, p in zip(gold_text, pred_text):
            gw, pw, gl_, pl_ = _as_written(g), _as_written(p), _letters_only(g), _letters_only(p)
            e_w += _edit_distance(gw, pw)
            n_w += len(gw)
            e_l += _edit_distance(gl_, pl_)
            n_l += len(gl_)
        r = {"accuracy": sum(a == b for a, b in zip(yt, yp)) / max(1, len(yt)),
             "f1_weighted": f1w, "f1_macro": f1m,
             "split_accuracy": sum(a == b for a, b in zip(st_t, st_p)) / max(1, len(st_t)),
             "split_f1": 2 * tp / max(1, 2 * tp + fp + fn),
             "codes_fully_right": len(gold_spans & set(codes_by_method[m])) / max(1, len(gold_spans)),
             "hard_spot_accuracy": sum(gold[i] == lab[i] for i in hidx) / max(1, len(hidx)),
             "cer_as_written": e_w / max(1, n_w), "cer_letters_only": e_l / max(1, n_l)}
        lines = defaultdict(lambda: [0, 0])
        for i in idx:
            lines[line_of[i]][0] += 1
            lines[line_of[i]][1] += gold[i] == lab[i]
        accs = sorted(c / n for n, c in lines.values())
        r["lines_fully_correct"] = sum(a == 1.0 for a in accs) / max(1, len(accs))
        r["median_line_accuracy"] = accs[len(accs) // 2] if accs else None
        if bootstrap and lines:
            rng = random.Random(seed)
            by_line = defaultdict(list)
            for i in idx:
                by_line[line_of[i]].append(i)
            keys = list(by_line)
            a_s, f_s = [], []
            for _ in range(bootstrap):
                sample = [i for _ in keys for i in by_line[keys[rng.randrange(len(keys))]]]
                t, p = [gold[i] for i in sample], [lab[i] for i in sample]
                a_s.append(sum(x == y for x, y in zip(t, p)) / max(1, len(t)))
                f_s.append(_f1(t, p)[0])
            a_s.sort()
            f_s.sort()
            lo, hi = int(0.025 * bootstrap), int(0.975 * bootstrap) - 1
            r["accuracy_95ci"] = [a_s[lo], a_s[hi]]
            r["f1_weighted_95ci"] = [f_s[lo], f_s[hi]]
        result["methods"][m] = r
        per_line[m] = pred_text
    if bootstrap:
        result["bootstrap_resamples"] = bootstrap
    main = result["methods"]["model + key"]
    result.update({"accuracy": main["accuracy"], "f1_weighted": main["f1_weighted"], "f1_macro": main["f1_macro"]})

    support, correct, confusions = Counter(), Counter(), defaultdict(Counter)
    for i in idx:
        t, p = gold[i], labels_key[i]
        support[t] += 1
        if t == p:
            correct[t] += 1
        else:
            confusions[t][p] += 1
    units = [{"unit": u, "support": support[u], "correct": correct[u], "accuracy": correct[u] / support[u],
              "top_confusions": confusions[u].most_common(3)} for u in support]
    units.sort(key=lambda d: -(d["support"] - d["correct"]))
    result["errors"] = sum(support.values()) - sum(correct.values())

    details = {"pieces": pieces, "line_of": line_of, "gold": gold, "scored": scored, "gold_code": gold_code,
               "hard": hard, "labels": methods, "confidence": res["model_confidence"],
               "key_codes": res["codes"], "gold_text": gold_text, "pred_text": per_line,
               "line_numbers": [no for no, _ in numbered_lines]}
    # For reference_check: per line, the reference codes and the model + key units of the codes that start there.
    readings = [{"line": no, "symbols": list(items), "units": []} for no, items in numbered_lines]
    for s, e, code, unit, conf in res["codes"]:
        readings[line_of[s]]["units"].append(unit if unit is not None else UNK)
    return result, units, details, readings


def describe_digits(result, units, log=print, top=15):
    say(log, f"Digit model: scored {result['positions_scored']:,} digits ({result['codes_scored']:,} codes) on "
             f"{result['lines']} lines. {result['codes_not_in_key']} codes are not in the key and are not scored.")
    say(log, "The model reads the digits without the spaces, so it has to find the codes itself.")
    say(log, f"'Hard spots' = digits where a code could also be read as shorter codes ({result['hard_spot_digits']:,} digits).\n")
    rows = [("accuracy", "digit label accuracy"), ("f1_weighted", "F1 (weighted)"), ("f1_macro", "F1 (macro)"),
            ("split_accuracy", "split accuracy"), ("split_f1", "split F1"),
            ("codes_fully_right", "codes fully right"), ("hard_spot_accuracy", "hard spots"),
            ("cer_as_written", "CER as written"), ("cer_letters_only", "CER letters only"),
            ("lines_fully_correct", "lines fully correct")]
    names = list(result["methods"])
    say(log, f"{'':24}" + "".join(f"{n:>14}" for n in names))
    for k, label in rows:
        say(log, f"{label:24}" + "".join(f"{result['methods'][n][k]:14.2%}" for n in names))
    for n in names:
        ci = result["methods"][n].get("f1_weighted_95ci")
        if ci:
            say(log, f"  {n}: F1 (weighted) 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    say(log, f"\nLabels with the most errors, model + key ({result['errors']} errors in total; "
             f"{JOIN_LABEL_NOTE}):")
    for u in units[:top]:
        if u["support"] == u["correct"]:
            break
        conf = ", ".join(f"{p!r}×{c}" for p, c in u["top_confusions"])
        say(log, f"  {u['unit']!r:<22} {u['support'] - u['correct']:>5} errors of {u['support']:>6}"
                 f"  (accuracy {u['accuracy']:.2f})  read as: {conf}")


JOIN_LABEL_NOTE = "'<JOIN>' = a digit that belongs to the code before it"


def write_digit_evaluation(out_dir, name, result, units, details):
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, f"{name}_evaluation{suffix}") for k, suffix in
             (("json", ".json"), ("txt", ".txt"), ("units", "_units.csv"), ("lines", "_lines.csv"),
              ("digits", "_digits.csv"))}
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = []
    describe_digits(result, units, log=lines.append, top=len(units))
    with open(paths["txt"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(paths["units"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "support", "correct", "accuracy", "top_confusions"])
        for u in units:
            w.writerow([u["unit"], u["support"], u["correct"], f"{u['accuracy']:.4f}",
                        "; ".join(f"{p}×{c}" for p, c in u["top_confusions"])])
    d = details
    with open(paths["lines"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "reference_reading", "model_plus_key_reading", "model_reading"])
        for li, no in enumerate(d["line_numbers"]):
            w.writerow([no, d["gold_text"][li], d["pred_text"]["model + key"][li], d["pred_text"]["model"][li]])
    with open(paths["digits"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "digit", "reference_code", "reference_label", "scored", "hard_spot",
                    "model_label", "model_confidence", "model_plus_key_label"])
        for i, p in enumerate(d["pieces"]):
            w.writerow([d["line_numbers"][d["line_of"][i]], p, d["gold_code"][i], d["gold"][i],
                        "yes" if d["scored"][i] else "no", "yes" if i in d["hard"] else "",
                        d["labels"]["model"][i], f"{d['confidence'][i]:.3f}", d["labels"]["model + key"][i]])
    return list(paths.values())
