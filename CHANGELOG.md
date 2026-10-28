# Changelog

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
