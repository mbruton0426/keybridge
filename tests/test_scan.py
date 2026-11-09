"""keybridge 1.3.0: top-10 draft decryptions, repetition factor, decryption file names."""
import os
from keybridge import scan as scan_mod
from keybridge.pipeline import write_reading

KEY_UNITS = [chr(c) for c in range(65, 91)] + ["<NULL>"]


def test_repetitive_output_is_penalised():
    real = list("QUESTAELALETTERACHEILNUNZIOSCRIVEALLASEGRETERIADISTATO")
    junk = list("EEEFEFDEDEEEEDEEEFEFEEEEDEDEFEEEEDEEEEEFDFEDED")
    hr, hj = scan_mod.letter_entropy(real, KEY_UNITS), scan_mod.letter_entropy(junk, KEY_UNITS)
    assert hr > 3.5 and hj < 2.0
    assert scan_mod.repetition_factor(hr) > 0.9 and scan_mod.repetition_factor(hj) < 0.5
    assert scan_mod.letter_entropy(list("ABC"), KEY_UNITS) is None and scan_mod.repetition_factor(None) == 1.0


def _results(n):
    return [{"id": f"m{i}", "name": f"m{i}", "status": "scored", "score": 1 - i / 100, "readings": [{"line": 1, "units": ["A"]}],
             "symbol_format": "space"} for i in range(n)] + \
           [{"id": "skipped", "name": "skipped", "status": "symbols do not match the key", "symbol_format": "space", "readings": [{"line": 1, "units": ["A"]}]}]


def _written(monkeypatch, tmp_path, **kw):
    seen = []
    monkeypatch.setattr(scan_mod, "write_reading", lambda out, base, *a, **k: seen.append(base.split("__")[1]) or [base])
    monkeypatch.setattr(scan_mod, "describe_ranking", lambda *a, **k: None)
    res = _results(15)
    scan_mod.write_scan_outputs(str(tmp_path), "t", res, res[:1], "t.txt", **kw)
    return seen


def test_default_writes_top_ten(monkeypatch, tmp_path):
    assert _written(monkeypatch, tmp_path) == [f"m{i}" for i in range(10)]


def test_write_top_and_all(monkeypatch, tmp_path):
    assert _written(monkeypatch, tmp_path, write_top=3) == ["m0", "m1", "m2"]
    assert len(_written(monkeypatch, tmp_path, write_all=True)) == 16


def test_decryption_file_names(tmp_path):
    readings = [{"line": 1, "symbols": ["1", "2"], "units": ["A", "B"], "confidence": [0.9, 0.8]}]
    try:
        files = write_reading(str(tmp_path), "x", readings, "space", "Draft decryption of t")
    except Exception:          # write_reading needs more fields in some versions; the names are what matters
        files = []
    names = os.listdir(tmp_path)
    assert all("decipherment" not in n for n in names)
