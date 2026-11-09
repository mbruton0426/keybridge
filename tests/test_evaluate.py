"""Scoring against the key reference: symbols not in the key, and the scores without them."""
import os
import tempfile
import unittest
from unittest import mock

from keybridge import evaluate as ev
from keybridge.common import UNK
from keybridge.keys import load_key

KEY = 'plaintext\tciphertext\n"A"\t"a1"\t"a2"\n"B"\t"b1"\n"C"\t"c1"\n"D"\t"d1"\n"E"\t"e1"\n"F"\t"f1"\n'


def key():
    path = tempfile.mktemp(suffix=".tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(KEY)
    return load_key(path)


def fake_readings(*_):
    # line 1: all symbols in the key, one misread; line 2: one symbol ("zz") not in the key
    return [{"line": 1, "symbols": ["a1", "b1", "c1", "a2"], "units": ["A", "B", "C", "B"]},
            {"line": 2, "symbols": ["a1", "zz", "b1"], "units": ["A", "A", "B"]}]


class KeyReferenceTest(unittest.TestCase):
    def run_eval(self, exclude_unknown):
        with mock.patch.object(ev, "read_lines", fake_readings):
            return ev.evaluate(None, None, [], key(), exclude_unknown=exclude_unknown, bootstrap=20)

    def test_symbols_not_in_the_key_count_as_errors_and_are_also_left_out(self):
        result, units, per_line, readings = self.run_eval(False)
        self.assertEqual(result["positions_scored"], 7)
        self.assertEqual(result["errors"], 2)                     # one misreading + the symbol not in the key
        self.assertIn(UNK, readings[1]["key_reference"])
        k = result["in_key"]
        self.assertEqual(k["positions_scored"], 6)
        self.assertAlmostEqual(k["accuracy"], 5 / 6)
        self.assertAlmostEqual(k["lines_fully_correct"], 0.5)     # line 2 is right without "zz"
        self.assertIn("f1_weighted_95ci", k)
        out = []
        ev.describe(result, units, log=out.append)
        self.assertTrue(any("Without the symbols not in the key (6 positions)" in l for l in out))
        self.assertTrue(any("against the key reference" in l for l in out))

    def test_no_second_line_when_they_are_excluded(self):
        result, units, _, _ = self.run_eval(True)
        self.assertEqual(result["positions_scored"], 6)
        self.assertNotIn("in_key", result)

    def test_lines_table_has_a_key_reference_column(self):
        result, units, per_line, readings = self.run_eval(False)
        d = tempfile.mkdtemp()
        ev.write_evaluation(d, "t", result, units, per_line, readings)
        with open(os.path.join(d, "t_evaluation_lines.csv"), encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        self.assertEqual(header[-2:], ["key_reference", "predicted"])


if __name__ == "__main__":
    unittest.main()
