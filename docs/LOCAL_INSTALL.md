# Running everything on your own computer

The notebooks and the command-line tool use the same code and give the same results. The local version keeps your keys and transcriptions entirely on your computer. Once the shared models are downloaded, it can run without an internet connection.

## Install

You need **Python 3.9 or newer** ([python.org](https://www.python.org/downloads/)). Then, in a terminal (Command Prompt on Windows):

```
pip install keybridge
```

This also installs PyTorch, which is a large download (around 1–3 GB depending on your system). Check that the install worked:

```
keybridge --version
```

To install the latest development version from GitHub instead, use `pip install git+https://github.com/mbruton0426/keybridge.git`, or download the repository as a zip, unpack it, and run `pip install .` inside the unpacked folder.

A GPU is not required. Reading and scanning are fast on any computer. Training is much slower without a GPU, especially with the default 5,000 examples and the model-settings search, so use the Colab notebook for training if you can.

## Commands

Every command has a `--help` option listing all its settings, e.g. `keybridge train --help`. Results are written to an `outputs` folder in the current directory (change it with `--out`).

The transcription format is given as `space` (symbols separated by spaces, the standard) or `char` (older one-character-per-symbol files; every blank becomes `_`). See [TRANSCRIPTION_FORMAT.md](TRANSCRIPTION_FORMAT.md).

### Train a model for your key

```
keybridge train --key my_key.tsv --format space --plaintext text1.txt text2.txt --name my_cipher
```

Without `--plaintext`, the built-in English plaintext is used (or another language with `--language`). This is downloaded from Hugging Face the first time.

Useful options, matching the notebook settings ([TRAINING_GUIDE.md](TRAINING_GUIDE.md#5-all-settings-explained)):

| Option | Notebook setting |
|--------|------------------|
| `--replacements "J=I, V=U"` | Letter replacements |
| `--word-spaces "insert at random"` | Word spaces |
| `--examples 5000` | Training examples (default 5000) |
| `--length 50` | Sequence length |
| `--null-rate 0.05`, `--double-rate 0.5`, `--rare-rate 0.02` | Null / double / rare unit rates (defaults shown; `--rare-rate 0` turns rare-unit insertion off) |
| `--search-trials 20` | Short trials of the model-settings search (default 20; `0` skips the search and uses the fixed default settings) |
| `--language Swedish` | Built-in plaintext language when `--plaintext` is not given (default English; also Czech, Dutch, French, Hungarian, Icelandic, Italian, Polish, Spanish) |
| `--digit-mode auto` | Digit mode: `auto` (default; digit keys are read digit by digit), `on` or `off`. See [Digit ciphers](TRANSCRIPTION_FORMAT.md#digit-ciphers-running-digits). |
| `--transcription letter.txt` | Also read a transcription after training |
| `--custom-edits my_edits.py` | Custom edits (see below) |

### Read a transcription with one model

```
keybridge read --model outputs/my_cipher/my_cipher_model.pt --transcription letter.txt
```

`--model` also accepts the ID of a model in the shared collection. The transcription format defaults to the model's own.

For digit models, the transcription is read digit by digit and spaces are ignored; add `--use-separators` to treat spaces as code boundaries (also available for `scan` and `train --transcription`).

### Test a transcription against many models

```
keybridge scan --transcription letter.txt --format char --models all
```

| Option | Meaning |
|--------|---------|
| `--models all` | All shared models. Also `digits`, `characters`, `named`, a comma-separated list of IDs, or `none`. |
| `--model-file my_model.pt` | Also test your own model (repeat for several). |
| `--languages de,la,it` | Limit language scoring. |
| `--margin 0.05`, `--max-close 3` | Which results count as top results. |
| `--min-coverage 0.75` | Skip models whose key contains less than this share of your symbols. `0` tests every model. |
| `--all-outputs` | Write a decipherment for every model. |

The ranking is printed and written to `outputs/<transcription>_ranking.txt` and `.csv`. How the scores work is explained in [READING_AND_SCANNING_GUIDE.md](READING_AND_SCANNING_GUIDE.md#how-models-are-ranked).

### Score a reading against the key (research)

```
keybridge evaluate --model my_cipher_model.pt --transcription letter.txt
```

For checking a model on an authentic ciphertext whose key is known, as in the paper's experiments. The reference for each symbol is the plaintext unit the key gives it (the transcription decoded with the key). The command reports:
- support-weighted F1 (as in the paper), accuracy and macro F1, with 95% confidence intervals from a line-level bootstrap;
- the share of lines fully correct and at least 95% correct;
- the units with the most errors, and what they were read as.

It writes a `.json` and `.txt` summary, a per-unit table and a per-line table.

**Digit models.** Give a transcription with the **codes separated by spaces**, as in an archive decryption (e.g. `12^. 1 5 62^. 71 00`). The reference is the key's reading of each code; the model reads the same digits with the spaces removed, so it has to find the codes itself. Two readings are scored, per digit (a code's first digit carries the unit, the others the label `<JOIN>`): *model* (the model's own labels) and *model + key* (the split into key codes that the model finds most likely, which is what keybridge outputs). Besides F1 and accuracy, the command reports split accuracy, the share of codes read fully right, accuracy on "hard spots" (digits where a code could also be read as shorter codes), and the character error rate of the reading. Codes that are not in the key (e.g. `?`) are not scored. It also writes a per-digit table.

| Option | Meaning |
|--------|---------|
| `--reference-plaintext archive_pt.txt` | Also compare the reference with the archive's own plaintext transcription (one line per ciphertext line, same order), and report how much they differ (see below). |
| `--key full_key.tsv` | Derive the reference from this key instead of the one stored in the model. Use the *true* key here to score a model on a ciphertext written with a different key. |
| `--exclude-unknown` | Do not score symbols that are not in the key (by default they count as errors). |
| `--exclude-units "MASTER,LODGE"` | Do not score these plaintext units, e.g. nomenclature absent from training. |
| `--bootstrap 1000` | Number of resamples for the confidence intervals (`0` for none). |
| `--custom-edits my_edits.py` | Normalise the transcription first, e.g. a project's transcription conventions. |

Scores are per cipher symbol, with one plaintext unit per symbol. A nomenclature symbol therefore counts once, not once per letter.

**Checking the reference.** The reference is the key's reading of the transcription, which can differ from the archive's published plaintext: through spelling normalisation, transcription choices, or errors in the key or transcription. With `--reference-plaintext`, both the reference and the model's reading are compared line by line with the archive plaintext. The command reports:
- the character error rate (edit distance divided by the length of the archive text);
- the share of identical lines;

each both "as written" (case ignored, spacing normalised) and "letters only" (spaces and punctuation ignored). A per-line table shows all three texts side by side. The two files must match line by line after skipping empty lines and lines with the skip markers.

### Convert to the standard format

```
keybridge convert --key old_key.tsv --transcription letter1.txt letter2.txt
```

This rewrites a key, and any "one character per symbol" transcriptions, in the standard format: symbols separated by spaces, with a blank symbol written as `_`. The results are saved next to the originals as `old_key_standard.tsv`, `letter1_standard.txt` and so on; the originals are not changed. Note lines starting with `##` (or the markers given with `--skip`) are copied unchanged.

You don't strictly have to convert: blanks in keys are turned into `_` automatically, and older transcriptions can be read with `--format char`. Converting once keeps all your files in one format.

### Work completely offline

1. While online, download every shared model once:
   ```
   keybridge models download
   ```
2. From then on, tell the tool to use only the downloaded copies.
   - Mac/Linux: `export HF_HUB_OFFLINE=1`
   - Windows (Command Prompt): `set HF_HUB_OFFLINE=1`

   Then run `read` or `scan` as usual.

Alternatively, download the whole model collection from Hugging Face as a folder and point the tool at it with `--models-from path/to/folder`.

Other commands:
- `keybridge models list` shows the shared models.
- `keybridge languages` shows the installed language statistics.

### Custom edits

Write the functions from step ③ of the notebook into a Python file and pass it with `--custom-edits`:

```python
# my_edits.py
def custom_edit_training_units(units):
    return units

def custom_edit_transcription_line(symbols):
    return ["a" if s.isalpha() and s != s.lower() else s for s in symbols]
```

## Using it from Python

```python
from keybridge.keys import load_key
from keybridge.model import load_model
from keybridge.pipeline import read_lines
from keybridge.transcription import read_transcription
from keybridge.data import readable

model, checkpoint = load_model("my_cipher_model.pt")
lines = read_transcription("letter.txt", checkpoint["settings"]["symbol_format"])
for r in read_lines(model, checkpoint, lines):
    print(r["line"], readable(r["units"]))
```
