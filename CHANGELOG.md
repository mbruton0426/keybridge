# Changelog

## 1.3.0

- **Scan score.** Models are now ranked by coverage × plausibility × repetition. The new *repetition* figure (letter
  entropy of the output ÷ 3.8 bits, at most 1; outputs under 20 letters get 1) lowers repetitive output such as
  `EEEFEFDE…`, which the letter statistics could mistake for a real language. The model's own certainty ("fit") is
  no longer part of the score: its references come from synthetic text, and on real letters it could fall to 0 even
  for the right key. It is still listed in the ranking CSV as `fit_diagnostic`, and used in plausibility's place only
  when no language profiles are installed. On 312 letters with known keys, the true key is now ranked first for 83%
  of letters and among the first three for 90% (1.2.0: 59% and 72%).
- **More drafts.** `keybridge scan` writes draft decryptions for the 10 best-ranked models by default
  (`--write-top N`; `--all-outputs` still writes every model), because the right key is not always ranked first. The
  scan notebook has a matching *Drafts to write* setting, and the downloaded top results include these drafts.
- **Wording.** Outputs, notebooks and documentation say *decryption* (reading a ciphertext with a known key) instead
  of *decipherment*. Output files are now `…_decryption.txt` and `…_decryption_aligned.csv`. Models trained with
  earlier versions load unchanged.
- **Evaluation: key reference.** `keybridge evaluate` calls what it scores against the *key reference* (the
  transcription read through the key), to keep it apart from a human decryption, which `--reference-plaintext`
  compares with. The per-line table's column `gold` is now `key_reference`.
- **Evaluation: symbols not in the key.** For symbol keys, symbols that are not in the key still count as errors, and
  the report now adds a second line with F1 (and its confidence interval), accuracy and fully correct lines without
  them (also in the JSON summary, as `in_key`). For digit keys they were never scored, as before.
- **Docs.** The transcription format says that codes of several characters (e.g. `z19`) must be written whole, not
  character by character.

## 1.2.0

**Training data are made with ChronoFidelius.** All synthetic training ciphertext is now enciphered by [ChronoFidelius](https://pypi.org/project/ChronoFidelius/) (1.3.0 or later, installed automatically), the library used to build the HistCiph data. The key is turned into a ChronoFidelius encryption dictionary on the fly; the key format does not change.

- Nulls are inserted by ChronoFidelius (`--null-rate`, as before).
- **Multi-letter rate** (`--multi-rate`, notebook *Multi-letter rate*, default 0.7): multi-letter units (syllables, words) are used this share of the time and spelled out otherwise.
- **Cancel signs**: keys may have the units `[-]`, `[DEL-START]`, `[DEL-END]` (or `[DEL-START|DEL-END]`). Cancelled mistakes are put into the training text with these signs by ChronoFidelius (`--cancel-rate`, default 0.05), so the model learns to recognise them.
- **Polyphones**: a symbol with several readings is one unit, `"[CHE|CHI]"`. It is trained as a possible symbol for each reading and read as `[CHE|CHI]`. Units written with a bare bar (`"CHE|CHI"`) are treated the same way.
- **Rare-unit insertion works on plaintext without word gaps** (such as the built-in HistCiph texts): units are inserted at random positions instead of at word gaps. Before, almost nothing was inserted into such texts.
- **Whole-word units are gone**: every unit matches anywhere. The old `yes` mark in keys is accepted but ignored, and the key summary says so.
- New custom edit `custom_edit_training_text(text)` (notebook step ③, `--custom-edits`), applied to each plaintext document before enciphering.
- A special entry written in quotes (`"<SPACE>"`) is now read as the special entry, with a warning; before, it was taken as a text unit.
- `keybridge evaluate`: option-aware accuracy (a reading `[REY|REG]` counts as right when the reference is `REG`), and option-aware character error rates against an archive plaintext (`--reference-plaintext`).

Models trained with earlier versions load and read as before. Models trained with 1.2.0 can differ from earlier ones trained with the same seed, because the training data are made differently.

## 1.1.0

**Digit ciphers written as running digits.** For keys whose codes are all made of digits (with optional marks written with a digit, such as `^.` for a dot on top), keybridge now reads ciphertexts digit by digit and the model finds where each code starts. Transcriptions of such ciphers can be written as running digits, as in the manuscripts, without knowing the code boundaries.

- Training: digit keys are detected automatically (`--digit-mode auto`, the default; `on` / `off` to choose). Each code of the synthetic ciphertext is split into its digits; the first digit is labelled with the plaintext unit, the others with `<JOIN>`. The model architecture and training are unchanged.
- Reading ("model + key"): of all ways to cut the digits into codes of the key, the one the model finds most likely is chosen. The key decides what each code means; the model decides where the codes are. No language data is used. The text is read as one stream, so codes may run over line ends.
- `--use-separators` (notebooks: *Use spaces as code boundaries*): treat spaces in a transcription as code boundaries when the manuscript separates the codes.
- Decipherments show each code as split by the model above its reading.
- Fit check and scan work on digits for digit models; the collection index records the digits a digit model knows, for the coverage check.
- `keybridge evaluate` for digit models: scores against a transcription with the codes separated (e.g. from an archive decryption), reporting both the model's own labels and model + key, with split accuracy, codes fully right, accuracy on "hard spots" (digits that could also be read as shorter codes) and the character error rate; writes a per-digit table.
- Training reports for digit models include the model + key accuracy on the synthetic test examples.

Nothing changes for other keys: existing models, including the shared collection, load and read exactly as before.

## 1.0.0

First public release.