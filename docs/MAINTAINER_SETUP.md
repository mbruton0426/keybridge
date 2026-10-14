# Maintainer setup

Everything you (the maintainer) need to do once the GitHub and Hugging Face repositories exist, and how to add models and languages later.

## 1. Fill in the names

Until this step is done, the notebooks cannot install the package from GitHub, and the shared model collection cannot be reached. Every place that needs a name is marked `PLACEHOLDER`. You can see them all with:

```
grep -rn "PLACEHOLDER\|GITHUB_REPO_NAME\|mbruton/HF_MODEL_REPO" .
```

**Do this with the script, not by hand.** A plain find-and-replace of `HF_MODEL_REPO` would also change code that checks whether the name is still a placeholder.

```
python tools/set_names.py --github-repo YOUR-REPO-NAME --hf-repo mbruton/YOUR-HF-REPO-NAME
python tools/set_names.py --contact-email you@example.org          # address for emailed model bundles
python tools/set_names.py --citation "Bruton, M., ... (2027). ..."  # once the paper is published
```

The script replaces the placeholders in these files and then rebuilds both notebooks:

| Placeholder | Where | Becomes |
|-------------|-------|---------|
| `GITHUB_REPO_NAME` | `src/keybridge/config.py`, `tools/build_notebooks.py` (install line and guide links), both notebooks, `README.md` (Colab links), `docs/LOCAL_INSTALL.md` (install commands), `pyproject.toml` (homepage), `hf_collection_template/README.md` | your GitHub repository name |
| `mbruton/HF_MODEL_REPO` | `src/keybridge/config.py` (`HF_MODEL_REPO`), `hf_collection_template/README.md` | your Hugging Face model repository |
| `SHARE_CONTACT_EMAIL` | `src/keybridge/config.py` | the address shown to users who share a model by email (empty: none shown) |
| `[citation to be added]` | `src/keybridge/config.py`, `README.md`, `hf_collection_template/README.md` | the paper's citation |

Things to do by hand:
- **`LICENSE-CC-BY-4.0.txt`**: paste the full legal code of CC BY 4.0 from https://creativecommons.org/licenses/by/4.0/legalcode.txt at the end, where marked. `LICENSE` (MIT, for the code) is complete.
- **`CITATION.cff`**: fill in the paper's title, journal, year, volume and pages (and a DOI) once published. `set_names.py --github-repo` fills in its repository address.
- **`README.md`**: delete the hidden note to yourself at the top once the paper is updated (optional; it does not show on GitHub).
- **Keys in the collection:** CC BY 4.0 covers your own work. For keys reconstructed by others (e.g. from DECODE or publications), check that redistributing them is allowed, and credit the source in each model's README.

## 2. The GitHub repository (`mbruton0426/YOUR-REPO-NAME`)

Upload **this whole folder**: `README.md`, `LICENSE`, `LICENSE-CC-BY-4.0.txt`, `CITATION.cff`, `pyproject.toml`, `src/`, `notebooks/`, `docs/`, `examples/` and `tools/`. Leave out `hf_collection_template/` if you prefer, since it is only a template for Hugging Face.

The repository must be **public**, because the notebooks install the package straight from it.

**Check it works:** open each notebook with its "Open in Colab" link from the README, and run it with the example files.

**Before the repository exists**, you can still test in Colab. Upload this folder to Colab (e.g. as `/content/keybridge_repo`), and in step ② of a notebook change `INSTALL_FROM` to that path.

**Versions.** When you publish the paper, create a release (e.g. `v1.0.0`). Then change `INSTALL_FROM` in `tools/build_notebooks.py` to `git+https://github.com/mbruton0426/YOUR-REPO-NAME.git@v1.0.0` and rebuild the notebooks, so that the published notebooks keep working the same way even after later changes. Connecting the repository to Zenodo gives each release a citable DOI.

## 3. The language statistics

Scanning uses one letter-triple profile per language. They are built from the HistCorp `<language>_year_to_doc.json` files:

```
python tools/build_language_profiles.py path/to/folder_with_year_to_doc_files --out language_profiles
```

This makes one small file per language (`sv.json.gz`, `de.json.gz`, ...; about 0.2 MB and 15 seconds each). Copy them into **`src/keybridge/language_profiles/`** and commit them; they are installed together with the package. `keybridge languages` lists what is installed.

- The input files must be named `<language>_year_to_doc.json`, e.g. `german_year_to_doc.json`. Swedish, Spanish, Slovene, Polish, Latin, Italian, Icelandic, Hungarian, German, French, English, Dutch and Czech are recognised and given their language codes. Any other name is used as it is.
- Each century contributes at most 3 million characters (`--max-chars-per-century`), so that a century with a lot of text does not dominate. Centuries with less text contribute everything they have. The centuries present can differ between languages.
- `sv.json.gz` (Swedish) is already included, built from the file you sent.
- Check that HistCorp's terms allow sharing statistics derived from it. Trigram counts are not the texts themselves, but it is worth confirming.

## 4. The Hugging Face model repository (`mbruton/YOUR-HF-REPO-NAME`)

Create a **public model repository** and add it to your collection. It must contain:

```
README.md                 ← from hf_collection_template/README.md (has the CC BY 4.0 licence tag)
index.json                ← the list of models; the tools read this first
maximilian_II/
    maximilian_II_model.pt
    maximilian_II_key.tsv
    maximilian_II_training_report.txt
    README.md
copiale/
    ...same four files...
```

`hf_collection_template/index.json` shows the format, with two example entries. Fields:

| Field | Filled in | Meaning |
|-------|-----------|---------|
| `id` | automatically | Folder name and the ID users type. Letters, digits, `_` and `-` only. |
| `name` | **by you** | Human-readable name. |
| `path`, `key_file` | automatically | Files inside the repository. |
| `sha256` | automatically | Checksum of the model file. |
| `symbol_format`, `symbol_type`, `symbols`, `symbol_lengths` | automatically | `symbols` is used to pre-filter models (symbol coverage) without downloading them, and `symbol_type` for the "digit codes / single characters / named symbols" choices. `symbol_format` only records how the training transcription was written; all models use the standard format. `symbol_type` is `digits`, `characters` (one character each) or `named`. |
| `letter_replacements`, `training_plaintext`, `training_examples`, accuracy and reference figures | automatically | From the model file. |
| `plaintext_language` | **by you** | Language code of the cipher's plaintext, if known. |
| `period`, `source`, `contributor` | **by you** | Shown to users; `source` should let people cite the key. |

### Adding a model

1. Train it with notebook ① (or `keybridge train`). Use the same `--name` as the ID you want.
2. Make its index entry. Run this in a local copy of the Hugging Face repository:
   ```
   keybridge index-entry path/to/maximilian_II_model.pt --add-to index.json
   ```
   This adds or replaces the entry. Fill in the fields marked `FILL IN`.
3. Create the folder `<id>/` with the model, key, training report and a README (see `hf_collection_template/maximilian_II/README.md`).
4. Upload the folder and the updated `index.json`.

### Reviewing shared models

Users who share by **Hugging Face pull request** add a folder under `submissions/<name>/`, which you review on the repository's *Community* tab. Users who share by **email** send a zip with the same files. To accept a model, check its README and training report, move it into a folder named after its ID, run `index-entry` for it, and fill in the fields.

### Testing without Hugging Face

The tools accept a local folder with the same layout instead of a repository:
- notebook ②: *Model collection* setting
- command line: `--models-from path/to/folder`

## 5. Changing the notebooks

The notebooks are generated by `tools/build_notebooks.py`; do not edit the `.ipynb` files by hand. After changing the script, run:

```
python tools/build_notebooks.py
```

Most behaviour lives in the package (`src/keybridge/`), so it is shared automatically by both notebooks and the command line.

## 6. Built-in plaintext for training

Notebook ① and `keybridge train` use public Hugging Face datasets of historical plaintext, one per language. They are loaded with `datasets.load_dataset(repo)`: training texts come from `ds["train"]["plaintext"]`, checking and test texts from `ds["validation"]["plaintext"]`. **English is the default** whenever no language, or an unknown one, is given.

The datasets are listed in `PLAINTEXT_DATASETS` in `src/keybridge/config.py`. They follow the naming pattern `mbruton/<language>_encrypted_HistCiph`. All must be **public**, with `train` and `validation` splits and a `plaintext` column.

To add a language:
1. add it to `PLAINTEXT_DATASETS`;
2. add `"<Language> (built-in)"` to the `PLAINTEXT_SOURCE` dropdown in `tools/build_notebooks.py`;
3. run `python tools/build_notebooks.py`.

## 7. Before going public

- Revoke the Hugging Face token and Weights & Biases key that appear in the older research files (`lstm_functions.py`, the real-world notebook). None of the files in this repository contain credentials.
- Run both notebooks once in a fresh Colab session from the public links.
