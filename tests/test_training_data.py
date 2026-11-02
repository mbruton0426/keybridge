"""Tests for the training data (made with ChronoFidelius) and option-aware scoring.
Run from the repository root:  python -m unittest discover tests"""
import os
import random
import tempfile
import unittest
from collections import Counter

from keybridge.data import build_examples, options_match, unit_options, KeyDictionary, GAP_CHAR
from keybridge.evaluate import readable_best, _as_written
from keybridge.keys import load_key

KEY = '''plaintext\tciphertext
"A"\t"12"\t"37"
"[A|IT|L]"\t"51"
"IT"\t"40"
"L"\t"22"
"I"\t"18"
"T"\t"19"
"B"\t"20"
"C"\t"21"
"QUE"\t"90"
"Q"\t"30"
"U"\t"31"
"E"\t"32"
"ZZZ"\t"99"
<NULL>\t"00"
'''


def make_key(text=KEY, extra=""):
    path = tempfile.mktemp(suffix=".tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + extra)
    return load_key(path)


TEXT = ["ITALIAQUEABCLTEI" * 300]


class TestTrainingData(unittest.TestCase):

    def test_labels_are_the_key_units_of_the_codes(self):
        key = make_key()
        codes, units, _ = build_examples(TEXT, key, random.Random(1), 1)
        self.assertEqual(len(codes), len(units))
        for c, u in zip(codes, units):
            self.assertEqual(key.symbol_to_unit[c], u)

    def test_or_bar_unit_codes_are_trained(self):
        key = make_key()
        codes, units, _ = build_examples(TEXT, key, random.Random(1), 1)
        self.assertGreater(codes.count("51"), 100)
        self.assertEqual({u for c, u in zip(codes, units) if c == "51"}, {"[A|IT|L]"})

    def test_bare_bar_units_count_as_options(self):
        self.assertEqual(unit_options("CHE|CHI"), ["CHE", "CHI"])
        self.assertEqual(unit_options("[A|B]"), ["A", "B"])
        self.assertEqual(unit_options("A"), ["A"])

    def test_nulls_from_chronofidelius(self):
        key = make_key()
        _, units, _ = build_examples(TEXT, key, random.Random(1), 1, null_rate=0.1)
        share = units.count("<NULL>") / len(units)
        self.assertTrue(0.05 < share < 0.15, share)

    def test_multi_letter_units_used_about_70_percent(self):
        key = make_key()
        _, units, _ = build_examples(["QUE" * 2000], key, random.Random(1), 1, null_rate=0, rare_unit_rate=0)
        share = units.count("QUE") / 2000
        self.assertTrue(0.6 < share < 0.8, share)

    def test_rare_units_inserted_without_word_gaps(self):
        key = make_key()
        _, units, report = build_examples(TEXT, key, random.Random(1), 1)
        self.assertIn("ZZZ", report["rare_units"])
        self.assertGreater(units.count("ZZZ"), 20)

    def test_word_gaps_inserted_at_random_when_text_has_none(self):
        key = make_key(extra='<SPACE>\t"_"\n')
        _, units, report = build_examples(TEXT, key, random.Random(1), 1)
        self.assertIn("word_spaces_note", report)
        self.assertGreater(units.count(" "), 100)

    def test_cancel_signs(self):
        key = make_key(extra='"[-]"\t"77"\n"[DEL-START|DEL-END]"\t"78"\n')
        kd = KeyDictionary(key)
        self.assertEqual(sorted(kd.cancel_types), ["cancelled", "cancelled_span"])
        codes, units, _ = build_examples(TEXT, key, random.Random(1), 1)
        self.assertGreater(units.count("[-]"), 20)
        self.assertGreater(units.count("[DEL-START|DEL-END]"), 20)

    def test_no_cancel_errors_without_cancel_units(self):
        key = make_key()
        self.assertEqual(KeyDictionary(key).cancel_types, [])

    def test_uncovered_characters_reported(self):
        key = make_key()
        _, _, report = build_examples(["ABCXYXY" * 50], key, random.Random(1), 1)
        self.assertEqual(report["uncovered"]["X"], 100)

    def test_custom_unit_edit_keeps_alignment(self):
        key = make_key()
        edit = lambda us: [("B" if u == "C" else u) for u in us]
        codes, units, _ = build_examples(TEXT, key, random.Random(1), 1, custom_edit=edit)
        self.assertNotIn("C", units)
        for c, u in zip(codes, units):
            self.assertEqual(key.symbol_to_unit[c], u)

    def test_custom_text_edit(self):
        key = make_key()
        _, units, _ = build_examples(TEXT, key, random.Random(1), 1, custom_edit_text=lambda t: t.replace("QUE", ""), rare_unit_rate=0)
        self.assertNotIn("QUE", units)

    def test_quoted_special_entry_is_read_as_special(self):
        key = make_key(extra='"<SPACE>"\t"_"\n')
        self.assertEqual(key.symbol_to_unit["_"], " ")
        self.assertTrue(any("in quotes" in w for w in key.warnings))

    def test_whole_word_marks_are_ignored(self):
        key = make_key(KEY.replace('"QUE"\t"90"', '"QUE"\t"90"\tyes'))
        _, units, _ = build_examples(["AQUEB" * 200], key, random.Random(1), 1)
        self.assertGreater(units.count("QUE"), 50)


class TestOptionAwareScoring(unittest.TestCase):

    def test_options_match(self):
        self.assertTrue(options_match("[REY|REG]", "REG"))
        self.assertTrue(options_match("A", "A"))
        self.assertFalse(options_match("[REY|REG]", "RE"))

    def test_readable_best_picks_the_fitting_option(self):
        self.assertEqual(readable_best(["[BEAU|BELLE]", "S"], "BELLES", _as_written), "BELLES")
        self.assertEqual(readable_best(["[BEAU|BELLE]", "S"], "BEAUS", _as_written), "BEAUS")

    def test_readable_best_keeps_both_when_the_reference_lists_both(self):
        from keybridge.evaluate import _letters_only
        self.assertEqual(_letters_only(readable_best(["[ALTRA|ALTRO]"], "altra/altro", _letters_only)),
                         _letters_only("altra/altro"))


if __name__ == "__main__":
    unittest.main()
