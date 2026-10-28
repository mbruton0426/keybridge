# Guide: training a model for your own key

The training notebook takes a **substitution cipher key you already know** (simple or homophonic) and trains a small neural network that reads ciphertexts written with that key. It runs in your web browser on Google Colab, so you do not need to install anything or write any code. The same steps can also run on your own computer ([local version](LOCAL_INSTALL.md)).

**It can:** produce a **draft decipherment** of a transcribed ciphertext written with your key, and help you judge **whether a ciphertext was written with your key at all**.
**It cannot:** find an *unknown* key, or replace expert review.

## Contents

1. [Quick start](#1-quick-start)
2. [Your key file](#2-your-key-file)
3. [Your transcription](#3-your-transcription)
4. [Plaintext to learn from](#4-plaintext-to-learn-from)
5. [All settings explained](#5-all-settings-explained)
6. [Understanding the results](#6-understanding-the-results)
7. [Rare units and nomenclature](#7-rare-units-and-nomenclature)
8. [Sharing your model](#8-sharing-your-model)
9. [Troubleshooting](#9-troubleshooting)
10. [For Python users](#10-for-python-users)

---

## 1. Quick start

You need a Google account.

1. Open **notebook ①** (link in the [main README](../README.md)).
2. Choose **Runtime → Change runtime type**, select **T4 GPU**, and click **Save**. Without a GPU, training is very slow.
3. Fill in the form in step ①. [Section 5](#5-all-settings-explained) explains each setting. You can leave the file boxes empty.
4. Choose **Runtime → Run all**.
5. When asked, click **Choose files** and upload your key, your plaintext and your transcription.
6. Wait. With the default 5,000 training examples this usually takes a few minutes on a GPU, longer if the model-settings search needs all 20 trials.
7. A zip file with all your results downloads at the end.

**Trying it first with the examples.** The [`examples`](../examples) folder contains the Maximilian II key and a short *synthetic* ciphertext made with it. Use these settings:
- *Transcription format:* "symbols separated by spaces" (the standard)
- *Plaintext source:* "Upload my own plaintext", with some German text. German is not among the built-in languages, and the Maximilian letters are in German. For a first try of the mechanics only, "English (built-in)" also works, but the readings will be worse.
- *Letter replacements:* `J=I, V=U`

Then upload `maximilian_II_key.tsv` and `maximilian_synthetic_transcription.txt` when asked.

If something is wrong, the notebook stops and explains what to fix. Fix it in the form, then choose **Runtime → Run all** again.

---

## 2. Your key file

The key is a `.tsv` text file with one line per plaintext unit, and every entry and symbol in double quotes:

```
"A"	"J"	"/"
"SCH"	"Y"
"Austria"	"z"	yes
<SPACE>	"_"
<NULL>	"w"	"7"
```

The full rules, special entries and common mistakes are in **[KEY_FORMAT.md](KEY_FORMAT.md)**.

---

## 3. Your transcription

A plain text file (`.txt`) with **one manuscript line per line**, in the standard format: **cipher symbols separated by spaces**, written exactly as in the key, with a blank symbol (such as a word divider) written as `_`, and no quotes or special names:

```
pi oh j v hd tri arr eh three c.
0675 4770 454 0086 443
H q X 1 _ * v X D q
```

Older "one character per symbol" files (`HqX1 *vXDq`) are still accepted: choose that format in step ①, and every blank becomes `_`. They can also be converted permanently with `keybridge convert`.

Notes inside lines and uncertain readings must be removed first. Whole note lines starting with `##` are skipped. The full rules, including how word gaps, nulls and the double sign are written, are in **[TRANSCRIPTION_FORMAT.md](TRANSCRIPTION_FORMAT.md)**.

---

## 4. Plaintext to learn from

The model learns from plaintext that the notebook encrypts with your key. It should be in **the language of your cipher** and, ideally, from a similar period and kind of text.

- **Built-in plaintext** (the default): historical texts from HistCorp, in **English** (default), Czech, Dutch, French, Hungarian, Icelandic, Italian, Polish, Spanish or Swedish. Choose the language of your cipher under *Plaintext source*. These texts have no spaces or punctuation. If your key has a `<SPACE>` entry, word gaps are therefore inserted at random (see *Word spaces* below).
- **Your own text:** one or more `.txt` files. You need roughly **325,000 characters (about 130 pages) for the default 5,000 training examples**, about 130,000 for 2,000, and proportionally more for 10,000. Blank lines separate passages. Punctuation is kept only where your key has symbols for it.
- **Your own Python list of texts** (for Python users): in step ③, make `custom_plaintext_documents()` return the list, e.g. `return my_texts`. It is used instead of the source chosen in step ①. On the command line, write the texts to a `.txt` file with a blank line between texts and use `--plaintext`.

With the built-in plaintext, the checking and test examples come from the dataset's separate validation texts. With your own text, they are taken from parts of it that are not used for training.

**Letters your key does not have.** Many historical keys have no separate symbol for J, V, W or Ä. The notebook lists any plaintext characters with no key entry and leaves them out. If they are letters, map them under *Letter replacements*, e.g. `J=I, V=U`, so that the training text matches how the encipherer would have written it.

**Word gaps.** If your key has a `<SPACE>` entry, choose under *Word spaces* how gaps appear in training:
- *keep real word boundaries:* use the gaps in your plaintext. Best when your plaintext has normal spacing.
- *insert at random:* gaps at random places, at about 15% of positions. This matches the published experiments and suits plaintext without spaces.
- *remove:* no gaps.

If your plaintext has no word gaps (like the built-in texts), "keep real word boundaries" automatically becomes "insert at random", and the notebook says so. If your key has no `<SPACE>` entry, word gaps are always removed.

---

## 5. All settings explained

| Setting | What it does |
|---------|--------------|
| Project name | Used to name your result files. |
| Key file | Path to your key, or leave empty to upload. |
| Transcription format | "symbols separated by spaces" (the standard) or, for older files, "one character per symbol" ([section 3](#3-your-transcription)). |
| Plaintext source | A built-in language (English by default) or your own `.txt` files ([section 4](#4-plaintext-to-learn-from)). |
| Plaintext file | Only for your own plaintext: path to the file(s), or leave empty to upload. |
| Letter replacements | Letters to rewrite before encrypting, e.g. `J=I, V=U`. |
| Word spaces | How word gaps appear in training ([section 4](#4-plaintext-to-learn-from)). |
| Training examples | 5,000 by default. With rare units inserted, 2,000 is too few for them to be learned; 5,000 was enough in tests on authentic letters, and 10,000 gave no further gain. 2,000 is enough without rare units, or for a quick check. |
| Sequence length | Plaintext units per training example. 50 works well; roughly the length of a manuscript line is a good choice. For digit keys, an example then has more digits than units (e.g. about 80 digits for 50 units). |
| Null rate | Share of positions that get a null (only if your key has `<NULL>`). Raise it if your ciphertexts use many nulls. For a digit key whose null digit never occurs inside a code, the share of that digit in the ciphertext is a good guide. |
| Double rate | Chance that a doubled letter is written with the double-letter sign (only if your key has `<DOUBLE>`). |
| Rare unit rate | 0.02 by default. Inserts key units that occur fewer than 20 times in the plaintext (nomenclature, punctuation, letters the language lacks) at this rate per word gap ([section 7](#7-rare-units-and-nomenclature)). 0 turns it off. |
| Search model settings | On by default: tries 20 model settings (short trials of 10 rounds each, `SEARCH_TRIALS`) on the checking examples and trains the final model with the best. Adds roughly 10–60 minutes depending on training size and GPU. Untick it to use the fixed default settings instead. |
| Decipher a transcription | Untick if you only want to train a model. |
| Transcription file | Path to your transcription, or leave empty to upload. |
| Skip lines starting with | Transcription lines to ignore (default `##`; a single `#` is not skipped, since `#` can be a cipher symbol). Several markers can be given, separated by commas. |
| Save to Google Drive | Also copies all results to *My Drive/cipher_models/your project*. |
| Share my model | See [section 8](#8-sharing-your-model). |
| Digit mode | `auto` by default: keys whose codes are all digits (with optional marks such as `^.`) are trained digit by digit, so the model learns to find where codes start in running digits ([Digit ciphers](TRANSCRIPTION_FORMAT.md#digit-ciphers-running-digits)). `off` trains on whole codes as before; `on` insists on digit mode. |
| Use spaces as code boundaries | Digit keys only: treat spaces in your transcription as code boundaries. Off by default (spaces are ignored and the model finds the codes). |
| Seed | Fixes the random choices so a run can be repeated exactly. |

---

## 6. Understanding the results

You receive a zip file containing:

| File | Contents |
|------|----------|
| `..._decipherment.txt` | Each transcription line with a confidence value, every cipher symbol aligned above its reading, and the flowing text. |
| `..._decipherment_aligned.csv` | One row per cipher symbol: the symbol, what the model read, how it is shown, its confidence, and a note if the symbol is not in the key or belongs to a unit never seen in training. |
| `..._model.pt` | The trained model. Keep it to read further ciphertexts later (notebook ②), or to share it. |
| `..._training_report.txt` | All settings, warnings and accuracy figures. |
| `..._key.tsv`, `..._key_as_read.tsv` | Your key, and the key as the tool understood it. |

In the decipherment file, every cipher symbol is aligned above its reading, so the two match one to one:

```
  cipher: <SPACE>  r  Omega  Psi  f  Mu  LinkSymbol  q  Union  <SPACE>  ,
  reads:  _        L  A      N    D  T   A           G  E      _        [,]
  text:   LANDTAGE[,]
```

`_` is a word gap, `·` a null and `?` a position where the model gave no letter. The double-letter sign is shown as the repeated letter. A symbol that is not in the key, or that belongs to a key unit never seen in training (for example punctuation or nomenclature), is shown as `[symbol]` instead of the model's guess; the guess is still in the CSV. If one of these short markers is itself a plaintext unit of your key, the full tag is used instead (`<SPACE>`, `<NULL>`, `<NONE>`), and the legend at the top of the file says which markers are used. The `text:` line is the flowing reading, with nulls left out.

**Does the ciphertext fit the key?** The most reliable test is to **read the output**. Text written with your key comes out mostly readable. Text written with a different key comes out as nonsense, often with many repeated letters.

The notebook also reports the share of symbols the model is unsure about, next to two reference values: one for synthetic text in your key, and one for the same text in randomly scrambled keys. A value close to the first suggests a fit; a value close to the second suggests none. Symbols the model cannot know are left out of all three figures: symbols that are not in the key, and symbols of key units that never occur in the training text (for example punctuation, or nomenclature). **This is a rough guide that has not yet been tested on real collections.** It can mislead when the transcription writes symbols differently from the key, or when another key shares many symbols with yours.

**Accuracy on synthetic text** (in the report) shows how well the model learned your key. Accuracy on real manuscripts is usually lower, because real texts contain things the synthetic training text does not: rare nomenclature, abbreviations, transcription errors and spelling habits.

---

## 7. Rare units and nomenclature

The model can only learn a symbol if its meaning appears in the training text. Nomenclature (symbols for whole words such as names or titles) is often missing from ordinary plaintext, so the model never learns it and cannot read it. The notebook lists such units after making the training data.

What you can do:
- Use plaintext from the same context as the cipher (similar letters, the same period and topic), so that the words occur naturally.
- Mark nomenclature with `yes` in the key, so it only matches whole words and not parts of longer words.
- Keep the *Rare unit rate* on (0.02 by default). It inserts rare units into the training text at random word gaps, so the model sees their symbols. In tests on authentic letters with 5,000 examples, it removed most errors on nomenclature, punctuation and missing letters. Because the units are inserted without real context, the remaining errors are mostly rare units confused with each other.

Expect errors around rare units and check them by hand.

---

## 8. Sharing your model

If you tick **Share my model**, the notebook prepares your trained model for the public collection, so that others can test their ciphertexts against your key.

**What is shared:** the model, your key, the training report, and a short description with your name, email and the source of the key.
**What is not shared:** your transcriptions and decipherments.

A model trained on a key can decipher anything written with that key, so **sharing the model effectively shares the key**. Only share if you have the right to do so, and tick the confirmation box. Shared models are published under the CC BY 4.0 licence.

Two ways to send it:
- **Download a bundle to send by email** (default). A zip file downloads; email it to the address the notebook shows. If it is too large to attach, upload it to Google Drive or similar and send a link.
- **Hugging Face pull request.** This needs a free Hugging Face account. The notebook asks you to log in, then submits your model for review. To avoid pasting your token each time, store it as a Colab secret named `HF_TOKEN` (key icon in the left sidebar).

---

## 9. Troubleshooting

| Problem | What to do |
|---------|------------|
| "No GPU found" | Runtime → Change runtime type → T4 GPU → Save, then Runtime → Run all. |
| GPU unavailable or usage exhausted | Free Colab GPU time is limited. Wait a few hours, or use fewer training examples. Without a GPU it still works, slowly. |
| "The built-in plaintext could not be loaded" | Check your internet connection and run the step again, or upload your own plaintext. |
| "Installation failed" | Check your internet connection and run step ② again. |
| The upload button does not appear | Run the step that asks for the file again (▶ button), or give the file's path in step ①. |
| Problems listed for the key file | See [KEY_FORMAT.md](KEY_FORMAT.md#common-mistakes). |
| Many "symbols not in the key" | The transcription writes symbols differently from the key, or the transcription format is wrong. They must match exactly, including case. |
| "The plaintext is too short" | Upload more text, or choose fewer training examples. |
| Many plaintext characters "have no entry in the key" | Use *Letter replacements*, or add those letters to the key. |
| Low accuracy on synthetic text | Use more training examples, or raise `SEARCH_TRIALS`. |
| The zip did not download | Open the folder icon on the left, find the zip, right-click → Download. Or tick *Save to Google Drive*. |
| The session disconnected | Colab ends idle sessions and deletes their files. Run everything again, and tick *Save to Google Drive* to keep results. |

---

## 10. For Python users

**Custom edits.** Step ③ contains three functions that are called automatically:
- `custom_plaintext_documents()` can return your own list of plaintext strings to train on (section 4).
- `custom_edit_training_units(units)` receives the whole training text as a list of key units (e.g. `['D', 'E', 'R', ' ', 'SCH', ...]`). Return an edited list, for example to imitate abbreviation habits. Every unit must exist in the key; the special ones are `' '`, `'<NULL>'` and `'<DOUBLE>'`.
- `custom_edit_transcription_line(symbols)` receives each transcription line as a list of symbols. Use it to normalise transcription conventions.

The command line accepts the same functions in a Python file: `--custom-edits my_edits.py`.

**How training data are made.** The plaintext is split into key units by greedy longest match, the same rule as `ChronoFidelius.encrypt_homophonic(encryption_dict=...)`, with two differences:
- whole-word units only match complete words;
- characters with no key entry are reported and removed, instead of being copied into the ciphertext unchanged.

Word gaps, double-letter signs, nulls and custom edits are then applied, the text is cut into examples, and each unit is replaced by one of its symbols at random. Input and target are aligned **one cipher symbol to one plaintext unit**, so multi-letter units and nomenclature are single targets.

**Saved models** (`.pt`) are self-contained: 16-bit weights, hyperparameters, input and output vocabularies, the key, settings and metrics. Load one with `keybridge.model.load_model(path)`.
