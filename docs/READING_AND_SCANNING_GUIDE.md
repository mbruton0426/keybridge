# Guide: reading a ciphertext, or testing it against many keys

Notebook ② takes a transcribed ciphertext and runs it through trained models, either from the **shared collection** of known keys, your own model files, or both. It then ranks which keys the ciphertext most likely fits. The same can be done on your own computer with `keybridge scan` ([local version](LOCAL_INSTALL.md)).

Typical uses:
- **Reading:** you know (or suspect) which key was used. Test one model and get a draft decryption.
- **Screening:** you have an unsolved ciphertext. Test it against every key in the collection to see whether any fits, and which language the result reads like.

The results are a guide for expert review, not a verdict. **Always read the outputs yourself.**

## Quick start

1. Open **notebook ②** (link in the [main README](../README.md)). A GPU is not needed.
2. In step ①, choose your *Transcription format* and *Which models*.
3. Choose **Runtime → Run all** and upload your transcription when asked.
4. Read the ranking and the previews of the best results.
5. The best result, or the best few if their scores are close, downloads as one zip file.

The transcription must follow the **[transcription format](TRANSCRIPTION_FORMAT.md)**: the cipher symbols exactly as they appear in the manuscript and in each key, one manuscript line per line, with notes removed.

## Settings

| Setting | What it does |
|---------|--------------|
| Transcription file | Path to your transcription, or leave empty to upload. |
| Transcription format | "symbols separated by spaces" (the standard) or, for older files, "one character per symbol" (converted automatically; blanks become `_`). |
| Skip lines starting with | Lines to ignore (default `##`; a single `#` is not skipped, since `#` can be a cipher symbol). Several markers can be given, separated by commas. |
| Which models | All shared models; only those whose symbols are digit codes, single characters, or named symbols; specific models by ID; or only your own files. |
| Model IDs | For "Specific shared models": IDs as listed in the collection, comma-separated. |
| Use spaces as code boundaries | Only affects digit models. By default the transcription is read digit by digit and spaces are ignored, so running digits work. Tick this if your transcription separates the codes as the manuscript does ([Digit ciphers](TRANSCRIPTION_FORMAT.md#digit-ciphers-running-digits)). |
| Use my own model files | Also test `.pt` files made with notebook ①. |
| Languages | Limit the language comparison to some languages, e.g. `de, la, it`. Empty means all available. |
| Close margin | Models scoring within this margin of the best are all treated as top results (default 0.05). |
| Minimum symbol coverage | Models whose key contains less than this share of your symbols are skipped without being downloaded (default 75%). Set it to 0 to test every model. |
| Auto-download top results | Downloads the best result(s) as one zip at the end. |
| Drafts to write | Draft decryptions are written for this many of the best-ranked models (default 10), because the right key is not always ranked first. |
| Write decryptions for all models | Also writes a draft for every tested model. |
| Save to Google Drive | Copies all results to *My Drive/cipher_scans/…*. |
| Model collection | *Advanced:* a different collection (Hugging Face repository or folder). |

## How models are ranked

For every model tested, three figures are calculated:

| Figure | Meaning |
|--------|---------|
| **Symbols in key** (coverage) | The share of your transcription's symbols that exist in the model's key. This is checked before anything is downloaded; models below the minimum are skipped. For digit models, it is the share of your digits (with their marks) that the key uses. |
| **Reads like** (plausibility, 0–1) | How much the decoded output resembles real text in each available language, based on letter-triple statistics from historical corpora. About 1 means it reads like real text in that language; about 0, like random letters. The best language is shown. |
| **Repetition** (0–1) | The variety of letters in the output (letter entropy ÷ 3.8 bits, at most 1). A model applied to a ciphertext of another key often outputs long runs of a few letters (`EEEFEFDE…`), which can look like a real language to the letter statistics; such output gets a low value. |
| **Score** | coverage × plausibility × repetition. Used for the ranking. |

The ranking is a guide to **which draft decryptions to read first**, not a decision: reading the drafts is the only way to tell whether a ciphertext was written with a key. A ciphertext from the right key comes out largely readable; one from another key comes out as nonsense or as runs of a few letters. For that reason, drafts are written for the ten best-ranked models by default. In tests on 312 letters with known keys, the true key was ranked first for 83% of the letters, among the first three for 90% and among the first ten for 95%. (The ranking CSV also lists the model's own certainty, `fit_diagnostic`; it is not part of the score.)

Language statistics are used **only to rank and describe outputs**. The decryption itself never uses them.

**Important caveats**
- The scores are **rough guides**. A high score is a reason to read the output closely, not proof.
- A model whose key has many words or names (nomenclature) can produce a draft that is a string of real words without sentences. Such drafts can score highly; read them, and look for running text.
- A ciphertext in the right key with many rare nomenclature symbols, or a poor transcription, can score lower than expected.
- Two keys that share many symbols can both score well. Compare their outputs.
- "Reads like" can only name languages that have statistics installed. A text in another language will match its closest available relative.
- Very short transcriptions (a line or two) give unreliable scores.
- If the best score is below 0.5, the notebook says that no model looks like a clear fit.

## Which models are tested

All keys and transcriptions use the same standard format (symbols separated by spaces; a blank symbol written as `_`), so every model can be tested on every transcription. A model is only left out when:
- it does not match your choice under *Which models* (e.g. digit-code models only, or specific IDs); or
- its key contains less than the *Minimum symbol coverage* share of your transcription's symbols (default 75%). This is checked from the collection's index before anything is downloaded. Set it to 0 to test every model regardless of overlap, for example to see how models for other keys behave on your text.

For scanning to work, transcriptions must name symbols the same way the keys do. This is straightforward for letters, digits and other characters. Symbol names (`lam` vs `lambda`) need care, because different transcribers name the same glyph differently.

## Outputs

| File | Contents |
|------|----------|
| `..._ranking.txt` | The ranking, plus the first lines of each top result. |
| `..._ranking.csv` | All figures for every model, including the three best-matching languages. Opens in a spreadsheet. |
| `..._<model>_decryption.txt` | The reading by that model, line by line, with confidences: each cipher symbol aligned above its reading (see [TRAINING_GUIDE.md](TRAINING_GUIDE.md#6-understanding-the-results) for the markers), then the flowing text. Written for the top results and the best-ranked models (10 by default), or for all models if chosen. |
| `..._<model>_decryption_aligned.csv` | One row per symbol: the symbol, what it was read as, how it is shown, the confidence, and a note if the symbol is not in the key or belongs to a unit never seen in training. |

Two zip files are made: one with the **top results** and the best-ranked drafts (downloaded automatically) and one with **everything** (in the Colab file panel, and on Google Drive if chosen).

## Privacy

In Colab, your transcription is processed on Google's servers for the length of the session. Models are downloaded from the shared collection; your transcription is never uploaded to it.

For full privacy, use the [local version](LOCAL_INSTALL.md). After downloading the models once, it can run with no internet connection at all.