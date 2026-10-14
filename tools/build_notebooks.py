"""Generates the two Colab notebooks in notebooks/. Edit this file, not the .ipynb files,
then run:  python tools/build_notebooks.py"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"

INSTALL_FROM = "git+https://github.com/mbruton0426/keybridge.git"
GUIDE_URL = "https://github.com/mbruton0426/keybridge/blob/main/docs"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text, hidden=True):
    return {"cell_type": "code", "metadata": {"cellView": "form"} if hidden else {},
            "execution_count": None, "outputs": [], "source": text.strip("\n")}


OPTUNA_BLOCK = """if SEARCH_MODEL_SETTINGS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna"], check=False)
"""


def install_cell(extra=False):
    optuna_block = OPTUNA_BLOCK if extra else ""
    return code(f'''
#@title ② Prepare the notebook (no need to open)  {{ display-mode: "form" }}
# Installs the decipherment package from GitHub. To test before the repository exists,
# upload the repository folder to Colab and set INSTALL_FROM to its path, e.g. "/content/my_repo".
INSTALL_FROM = "{INSTALL_FROM}"

import os, sys, subprocess, importlib
try:
    import keybridge
except ImportError:
    target = INSTALL_FROM
    print("Installing (about a minute the first time)...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", target], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit("Installation failed. Check your internet connection and try again.")
    importlib.invalidate_caches()
    import keybridge
{optuna_block}
import torch
from keybridge.colab import in_colab, find_or_upload, zip_files, offer_download, copy_to_drive
from keybridge.common import StopHere, stop, safe_name, SPACE_SEPARATED, ONE_CHAR
from keybridge import config

IN_COLAB = in_colab()
WORK = "/content" if IN_COLAB else os.getcwd()
UPLOAD_DIR = os.path.join(WORK, "uploads")
print(f"Ready (package version {{keybridge.__version__}}).")
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
''')


def save(name, cells):
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": [], "toc_visible": True}, "accelerator": "GPU",
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": cells}
    for c in nb["cells"]:
        c["source"] = c["source"].splitlines(keepends=True)
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT / name)


# =============================================================================
# Notebook 1: train a model for your own key
# =============================================================================
nb1 = [md(f"""
# ① Train a decipherment model for your own cipher key

This notebook teaches a small neural network the key of a **substitution cipher** (simple or homophonic) you already know,
and can then use it to read ciphertexts that may have been written with that key.

You need:
1. **Your key** as a tab-separated `.tsv` file, with every symbol in double quotes ([key format]({GUIDE_URL}/KEY_FORMAT.md)).
2. **Plaintext in the cipher's language:** pick a built-in language (English, Czech, Dutch, French, Hungarian,
   Icelandic, Italian, Polish, Spanish, Swedish), or upload your own `.txt` files.
3. *(Optional)* **A transcription of a ciphertext** to read ([transcription format]({GUIDE_URL}/TRANSCRIPTION_FORMAT.md)).

The guide explains every step: [{GUIDE_URL}/TRAINING_GUIDE.md]({GUIDE_URL}/TRAINING_GUIDE.md)

**Before you start:** choose *Runtime → Change runtime type → T4 GPU → Save*.
**Then:** fill in the form in step ① and choose *Runtime → Run all*. The notebook asks you to upload
files when it needs them, and stops with a plain explanation if something needs fixing.

*Privacy: your files are processed on Google's servers for this session only. To keep everything on
your own computer, use the command-line version instead (see the guide). Nothing is shared unless you
tick the sharing box and confirm.*
"""),
code(r'''
#@title ① Your settings  { display-mode: "form" }

#@markdown ### Your project
PROJECT_NAME = "my_cipher"  #@param {type:"string"}

#@markdown ### Your key (.tsv)
#@markdown Leave the file box empty to get an upload button.
KEY_FILE = ""  #@param {type:"string"}
#@markdown How is your transcription written? **Standard:** *symbols separated by spaces* (`ah lam x.`, `675 4770 454`, `H q X 1 _ v`).
#@markdown *One character per symbol* (`HqX1 vX`) is also accepted and converted; a blank symbol then becomes `_`.
TRANSCRIPTION_FORMAT = "symbols separated by spaces"  #@param ["symbols separated by spaces", "one character per symbol"]

#@markdown ### Plaintext to learn from
#@markdown Built-in historical plaintext (HistCorp) in several languages, or your own `.txt` files. Choose the cipher's language; English is the default.
PLAINTEXT_SOURCE = "English (built-in)"  #@param ["English (built-in)", "Czech (built-in)", "Dutch (built-in)", "French (built-in)", "Hungarian (built-in)", "Icelandic (built-in)", "Italian (built-in)", "Polish (built-in)", "Spanish (built-in)", "Swedish (built-in)", "Upload my own plaintext (.txt)"]
#@markdown Only for "Upload my own plaintext": leave empty to get an upload button.
PLAINTEXT_FILE = ""  #@param {type:"string"}
#@markdown Letters the plaintext uses but the key does not, as `old=new` (e.g. `J=I, V=U`):
LETTER_REPLACEMENTS = ""  #@param {type:"string"}
WORD_SPACES = "keep real word boundaries"  #@param ["keep real word boundaries", "insert at random", "remove"]

#@markdown ### Training
TRAINING_EXAMPLES = 5000  #@param [2000, 5000, 10000] {type:"raw"}
SEQUENCE_LENGTH = 50  #@param {type:"integer"}
NULL_RATE = 0.05  #@param {type:"slider", min:0, max:0.3, step:0.01}
DOUBLE_RATE = 0.5  #@param {type:"slider", min:0, max:1, step:0.05}
#@markdown Insert key units that are rare or missing in the plaintext (nomenclature, punctuation, letters the language lacks) at this rate per word gap. 0 turns it off.
RARE_UNIT_RATE = 0.02  #@param {type:"slider", min:0, max:0.1, step:0.005}
SEARCH_MODEL_SETTINGS = True  #@param {type:"boolean"}
SEARCH_TRIALS = 20  #@param {type:"integer"}

#@markdown ### Ciphertext to read
DECIPHER_A_TRANSCRIPTION = True  #@param {type:"boolean"}
TRANSCRIPTION_FILE = ""  #@param {type:"string"}
SKIP_LINES_STARTING_WITH = "##"  #@param {type:"string"}

#@markdown ### Saving
SAVE_TO_GOOGLE_DRIVE = False  #@param {type:"boolean"}

#@markdown ### Sharing your trained model (optional)
#@markdown Shares the model and key (not your transcriptions) with the public collection, so others can test their ciphertexts against your key.
SHARE_MY_MODEL = False  #@param {type:"boolean"}
SHARE_METHOD = "Download a bundle to send by email"  #@param ["Download a bundle to send by email", "Hugging Face pull request"]
YOUR_NAME = ""  #@param {type:"string"}
YOUR_EMAIL = ""  #@param {type:"string"}
KEY_SOURCE = ""  #@param {type:"string"}
I_HAVE_THE_RIGHT_TO_SHARE = False  #@param {type:"boolean"}

SEED = 9  #@param {type:"integer"}
print("Settings saved.")
''', hidden=False),
install_cell(extra=True),
code(r'''
#@title ③ Optional: custom edits for Python users  { display-mode: "form" }
# Most users can skip this. Leave the functions unchanged to change nothing.

def custom_plaintext_documents():
    """To train on your own plaintexts given as a Python list of strings, return that list here,
    e.g. return my_list, or return [line for line in open("/content/my_text.txt")].
    It is used instead of the plaintext source chosen in step ①. Return None to change nothing."""
    return None


def custom_edit_training_units(units):
    """The whole training text as a list of key units, e.g. ['D', 'E', 'R', ' ', 'SCH', ...].
    Return the edited list; every unit must exist in the key (special: ' ', '<NULL>', '<DOUBLE>')."""
    return units


def custom_edit_transcription_line(symbols):
    """One transcription line as a list of symbols. Return the edited list. Example (Copiale,
    where capital letters are decorative variants of word-space symbols):
        return ["a" if s.isalpha() and s != s.lower() else s for s in symbols]"""
    return symbols

print("Custom edits loaded.")
'''),
code(r'''
#@title ④ Get your files  { display-mode: "form" }
key_path = find_or_upload(KEY_FILE, "key file (.tsv)", UPLOAD_DIR)
print("Key file:", os.path.basename(key_path))
plaintext_paths = []
if PLAINTEXT_SOURCE.startswith("Upload") and custom_plaintext_documents() is None:
    plaintext_paths = find_or_upload(PLAINTEXT_FILE, "plaintext file(s) (.txt)", UPLOAD_DIR, multiple=True)
    print("Plaintext files:", ", ".join(os.path.basename(p) for p in plaintext_paths))
transcription_path = None
if DECIPHER_A_TRANSCRIPTION:
    transcription_path = find_or_upload(TRANSCRIPTION_FILE, "transcription (.txt)", UPLOAD_DIR)
    print("Transcription:", os.path.basename(transcription_path))
'''),
code(r'''
#@title ⑤ Check the key  { display-mode: "form" }
from keybridge.keys import load_key
key = load_key(key_path, TRANSCRIPTION_FORMAT)
print(key.summary())
if key.warnings:
    print("\nThings to check in the key file:")
    for w in key.warnings[:30]:
        print("  •", w)
'''),
code(r'''
#@title ⑥ Make the training data  { display-mode: "form" }
from keybridge.data import load_plaintext_files, load_builtin_plaintext, parse_replacements, normalise_plaintext
from keybridge.pipeline import TrainSettings, prepare_data, plaintext_chars_needed

SAFE_NAME = safe_name(PROJECT_NAME)
OUT_DIR = os.path.join(WORK, "outputs", SAFE_NAME)
settings = TrainSettings(
    project=SAFE_NAME, symbol_format=TRANSCRIPTION_FORMAT,
    plaintext_source=PLAINTEXT_SOURCE, plaintext_files=plaintext_paths,
    letter_replacements=LETTER_REPLACEMENTS, word_spaces=WORD_SPACES,
    training_examples=TRAINING_EXAMPLES, sequence_length=SEQUENCE_LENGTH,
    null_rate=NULL_RATE, double_rate=DOUBLE_RATE, rare_unit_rate=RARE_UNIT_RATE,
    search_trials=SEARCH_TRIALS if SEARCH_MODEL_SETTINGS else 0, seed=SEED)
replacements = parse_replacements(LETTER_REPLACEMENTS)
heldout = None
own_list = custom_plaintext_documents()
if own_list is not None:
    print(f"Using your own list of {len(own_list)} plaintexts (step ③).")
    documents = [normalise_plaintext(t, replacements) for t in own_list if str(t).strip()]
    settings.plaintext_source = "own list"
elif plaintext_paths:
    documents = load_plaintext_files(plaintext_paths, replacements)
else:
    language = PLAINTEXT_SOURCE.replace(" (built-in)", "")
    settings.plaintext_source = language
    print(f"Loading the built-in {language} plaintext...")
    train_chars, heldout_chars = plaintext_chars_needed(TRAINING_EXAMPLES, SEQUENCE_LENGTH)
    documents = load_builtin_plaintext(language, replacements, train_chars, "train")
    heldout = load_builtin_plaintext(language, replacements, heldout_chars, "validation")
data = prepare_data(key, documents, settings, custom_edit=custom_edit_training_units, heldout_documents=heldout)
'''),
code(r'''
#@title ⑦ Train the model  { display-mode: "form" }
from keybridge.pipeline import train
if not torch.cuda.is_available():
    print("No GPU found, so training will be slow. Runtime → Change runtime type → T4 GPU.\n")
result = train(key, data, settings)
'''),
code(r'''
#@title ⑧ Read your ciphertext  { display-mode: "form" }
from keybridge.pipeline import checkpoint_like, read_lines, write_reading, preview, fit_figures, describe_fit
from keybridge.transcription import read_transcription, parse_skip_prefixes
reading_files = []
if not DECIPHER_A_TRANSCRIPTION:
    print("Skipped (no transcription requested in step ①).")
else:
    lines = read_transcription(transcription_path, TRANSCRIPTION_FORMAT, parse_skip_prefixes(SKIP_LINES_STARTING_WITH),
                               custom_edit_transcription_line, SEQUENCE_LENGTH)
    ck = checkpoint_like(key, data, result, settings)
    readings = read_lines(result["model"], ck, lines)
    reading_files = write_reading(OUT_DIR, SAFE_NAME, readings, TRANSCRIPTION_FORMAT,
                                  f"Decipherment of {os.path.basename(transcription_path)} with model '{SAFE_NAME}'",
                                  checkpoint=ck)
    print(f"Read {len(readings)} lines. First lines:\n")
    preview(readings)
    print()
    describe_fit(fit_figures(readings, ck))
'''),
code(r'''
#@title ⑨ Save and download your results  { display-mode: "form" }
from keybridge.pipeline import save_training_outputs
model_path = save_training_outputs(OUT_DIR, settings, key, key_path, data, result)
files_to_zip = [os.path.join(OUT_DIR, f) for f in sorted(os.listdir(OUT_DIR))]
print("Saved:", *[os.path.basename(f) for f in files_to_zip], sep="\n  ")
zip_path = zip_files(files_to_zip, os.path.join(WORK, f"{SAFE_NAME}_results.zip"), SAFE_NAME)
if SAVE_TO_GOOGLE_DRIVE:
    copy_to_drive(OUT_DIR, f"cipher_models/{SAFE_NAME}")
offer_download(zip_path)
'''),
code(r'''
#@title ⑩ Share your model (only if ticked in step ①)  { display-mode: "form" }
import datetime, shutil
if not SHARE_MY_MODEL:
    print("Not shared (the sharing box in step ① is not ticked).")
else:
    missing = [n for n, v in [("YOUR_NAME", YOUR_NAME), ("YOUR_EMAIL", YOUR_EMAIL), ("KEY_SOURCE", KEY_SOURCE)] if not v.strip()]
    if not I_HAVE_THE_RIGHT_TO_SHARE:
        stop("To share, tick 'I_HAVE_THE_RIGHT_TO_SHARE' in step ① and run this step again.")
    if missing:
        stop("To share, please fill in " + ", ".join(missing) + " in step ① and run this step again.")
    share_dir = os.path.join(WORK, "share", SAFE_NAME)
    os.makedirs(share_dir, exist_ok=True)
    shutil.copy(model_path, share_dir)
    key.write_tsv(os.path.join(share_dir, f"{SAFE_NAME}_key.tsv"))
    shutil.copy(os.path.join(OUT_DIR, f"{SAFE_NAME}_training_report.txt"), share_dir)
    m = result["metrics"]
    with open(os.path.join(share_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# {SAFE_NAME}

- Contributed by: {YOUR_NAME} <{YOUR_EMAIL}>
- Date: {datetime.date.today().isoformat()}
- Key source: {KEY_SOURCE}
- Transcription format: {TRANSCRIPTION_FORMAT}
- Plaintext used for training: {PLAINTEXT_SOURCE} {', '.join(os.path.basename(p) for p in plaintext_paths)}
- Training examples: {TRAINING_EXAMPLES} × {SEQUENCE_LENGTH} symbols
- Accuracy on unseen synthetic text: {m['synthetic_test_accuracy']:.3f} (F1 {m['synthetic_test_f1']:.3f})
- Package version: {keybridge.__version__}
- Licence: CC BY 4.0

The contributor confirmed that they have the right to share this key and a model trained on it.
No transcriptions or decipherments are included.
""")
    if SHARE_METHOD.startswith("Hugging Face"):
        if "HF_MODEL_REPO" in config.HF_MODEL_REPO:
            stop("Sharing through Hugging Face is not set up yet. Choose 'Download a bundle to send by email'.")
        from huggingface_hub import HfApi, notebook_login
        token = None
        if IN_COLAB:
            try:
                from google.colab import userdata
                token = userdata.get("HF_TOKEN")
            except Exception:
                token = None
        if token is None:
            notebook_login()
        pr = HfApi(token=token).upload_folder(
            repo_id=config.HF_MODEL_REPO, folder_path=share_dir, path_in_repo=f"submissions/{SAFE_NAME}",
            create_pr=True, commit_message=f"Submit model {SAFE_NAME} from {YOUR_NAME}")
        print("Thank you! Your model was submitted for review:", getattr(pr, "pr_url", pr))
    else:
        bundle = zip_files([os.path.join(share_dir, f) for f in sorted(os.listdir(share_dir))],
                           os.path.join(WORK, f"{SAFE_NAME}_share_bundle.zip"), SAFE_NAME)
        print("Your sharing bundle is ready:", os.path.basename(bundle))
        if config.SHARE_CONTACT_EMAIL:
            print(f"Please email it to {config.SHARE_CONTACT_EMAIL} with the subject 'Shared model: {SAFE_NAME}'.")
        if os.path.getsize(bundle) > 20e6:
            print("It may be too large to attach; upload it to Google Drive or similar and send a link instead.")
        offer_download(bundle)
'''),
]

# =============================================================================
# Notebook 2: read or scan a ciphertext
# =============================================================================
nb2 = [md(f"""
# ② Read a ciphertext, or test it against many cipher keys

This notebook runs a transcribed ciphertext through trained models, either from the **shared
collection** of known keys or your own model files, and ranks which keys it most likely fits.
For each model it reports:

- how many of your symbols exist in that key,
- how confident the model is (compared with its own references),
- which language the decoded text reads most like.

The best result, or the best few if their scores are close, is downloaded automatically.
**Always read the outputs yourself:** the scores are a rough guide that has not yet been validated.

Your transcription must follow the [transcription format]({GUIDE_URL}/TRANSCRIPTION_FORMAT.md): one manuscript
line per line, the cipher symbols separated by spaces and written exactly as in the keys, and a blank
symbol (word divider) written as `_`.

The guide explains the settings and how to read the results: [{GUIDE_URL}/READING_AND_SCANNING_GUIDE.md]({GUIDE_URL}/READING_AND_SCANNING_GUIDE.md)

**How to run:** fill in step ① and choose *Runtime → Run all*. A GPU is not needed.

*Privacy: your transcription is processed on Google's servers for this session only. To keep it on
your own computer, use the command-line version instead (see the guide).*
"""),
code(r'''
#@title ① Your settings  { display-mode: "form" }

#@markdown ### Your transcription
#@markdown Leave empty to get an upload button.
TRANSCRIPTION_FILE = ""  #@param {type:"string"}
#@markdown **Standard:** *symbols separated by spaces*. *One character per symbol* is also accepted and converted (a blank symbol becomes `_`).
TRANSCRIPTION_FORMAT = "symbols separated by spaces"  #@param ["symbols separated by spaces", "one character per symbol"]
SKIP_LINES_STARTING_WITH = "##"  #@param {type:"string"}

#@markdown ### Which models to test
WHICH_MODELS = "All shared models"  #@param ["All shared models", "Shared models with digit codes", "Shared models with single-character symbols", "Shared models with named symbols", "Specific shared models (IDs below)", "Only my own model file(s)"]
#@markdown Model IDs, comma-separated (only for "Specific shared models"):
MODEL_IDS = ""  #@param {type:"string"}
#@markdown Also test model files of your own (upload when asked, or give paths):
USE_MY_OWN_MODEL_FILES = False  #@param {type:"boolean"}
MY_MODEL_FILES = ""  #@param {type:"string"}

#@markdown ### Ranking
#@markdown Languages to consider, as codes (e.g. `de, la, it`). Empty means all available.
LANGUAGES = ""  #@param {type:"string"}
#@markdown Models scoring within this margin of the best are all treated as top results.
CLOSE_MARGIN = 0.05  #@param {type:"slider", min:0, max:0.3, step:0.01}
#@markdown Models whose key contains fewer than this share of your symbols are skipped. 0 tests every model.
MIN_SYMBOL_COVERAGE = 0.75  #@param {type:"slider", min:0, max:1, step:0.05}

#@markdown ### Results
AUTO_DOWNLOAD_TOP_RESULTS = True  #@param {type:"boolean"}
WRITE_DECIPHERMENTS_FOR_ALL_MODELS = False  #@param {type:"boolean"}
SAVE_TO_GOOGLE_DRIVE = False  #@param {type:"boolean"}
#@markdown *Advanced:* another model collection (Hugging Face repository or folder). Empty means the shared collection.
MODEL_COLLECTION = ""  #@param {type:"string"}
print("Settings saved.")
''', hidden=False),
install_cell(),
code(r'''
#@title ③ Optional: custom edit for Python users  { display-mode: "form" }
def custom_edit_transcription_line(symbols):
    """One transcription line as a list of symbols. Return the edited list. Leave unchanged to change nothing."""
    return symbols
print("Custom edit loaded.")
'''),
code(r'''
#@title ④ Get your files  { display-mode: "form" }
from keybridge.transcription import read_transcription, parse_skip_prefixes
transcription_path = find_or_upload(TRANSCRIPTION_FILE, "transcription (.txt)", UPLOAD_DIR)
lines = read_transcription(transcription_path, TRANSCRIPTION_FORMAT, parse_skip_prefixes(SKIP_LINES_STARTING_WITH),
                           custom_edit_transcription_line, max_length=50)
n_symbols = sum(len(s) for _, s in lines)
print(f"Transcription: {os.path.basename(transcription_path)}, {len(lines)} lines, {n_symbols} symbols, "
      f"{len({x for _, s in lines for x in s})} distinct symbols.")
own_models = []
if USE_MY_OWN_MODEL_FILES or WHICH_MODELS.startswith("Only"):
    own_models = find_or_upload(MY_MODEL_FILES, "model file(s) (.pt)", UPLOAD_DIR, multiple=True)
    print("Your model files:", ", ".join(os.path.basename(p) for p in own_models))
'''),
code(r'''
#@title ⑤ Find the models to test  { display-mode: "form" }
from keybridge.registry import ModelRegistry
registry, entries = None, []
types = {"Shared models with digit codes": "digits", "Shared models with single-character symbols": "characters",
         "Shared models with named symbols": "named"}
if not WHICH_MODELS.startswith("Only"):
    registry = ModelRegistry(MODEL_COLLECTION.strip() or None)
    if WHICH_MODELS.startswith("Specific"):
        entries = registry.entries(ids=[m.strip() for m in MODEL_IDS.split(",") if m.strip()])
    else:
        entries = registry.entries(symbol_type=types.get(WHICH_MODELS))
    total = len(registry.entries())
    print(f"The collection has {total} model(s); {len(entries)} match your choice.")
if not entries and not own_models:
    stop("There are no models to test. Check your choice in step ①, or test your own model file.")
'''),
code(r'''
#@title ⑥ Test the transcription against the models  { display-mode: "form" }
from keybridge.scan import scan, top_group, describe_ranking
langs = {c.strip() for c in LANGUAGES.split(",") if c.strip()} or None
print(f"Testing {len(entries) + len(own_models)} model(s)...")
results = scan(lines, TRANSCRIPTION_FORMAT, registry, entries, own_models, langs, MIN_SYMBOL_COVERAGE)
top = top_group(results, CLOSE_MARGIN)
print()
describe_ranking(results, top)
'''),
code(r'''
#@title ⑦ Look at the best results  { display-mode: "form" }
from keybridge.data import readable
if not top:
    print("No model produced a reading (all were skipped). See the ranking above.")
for r in top:
    langs_txt = ", ".join(f"{name} {p:.2f}" for _, name, p in r.get("languages", [])) or "no language scores"
    print(f"=== {r['id']}  (score {r['score']:.2f}; reads like: {langs_txt})")
    for rd in r["readings"][:8]:
        print(f"  [{rd['line']:>4}] {readable(rd['units'])}")
    print()
'''),
code(r'''
#@title ⑧ Save and download  { display-mode: "form" }
from keybridge.scan import write_scan_outputs
SAFE_NAME = safe_name(os.path.splitext(os.path.basename(transcription_path))[0])
OUT_DIR = os.path.join(WORK, "outputs", f"scan_{SAFE_NAME}")
all_files, top_files = write_scan_outputs(OUT_DIR, SAFE_NAME, results, top, os.path.basename(transcription_path),
                                          WRITE_DECIPHERMENTS_FOR_ALL_MODELS)
print("Saved:", *[os.path.basename(f) for f in all_files], sep="\n  ")
all_zip = zip_files(all_files, os.path.join(WORK, f"scan_{SAFE_NAME}_all_results.zip"), f"scan_{SAFE_NAME}")
if SAVE_TO_GOOGLE_DRIVE:
    copy_to_drive(OUT_DIR, f"cipher_scans/{SAFE_NAME}")
if AUTO_DOWNLOAD_TOP_RESULTS:
    top_zip = zip_files(top_files, os.path.join(WORK, f"scan_{SAFE_NAME}_top_results.zip"), f"scan_{SAFE_NAME}")
    offer_download(top_zip)
print(f"\nAll results (every model) are in {os.path.basename(all_zip)}; to get it, open the folder icon on the left.")
'''),
]

save("1_train_your_own_model.ipynb", nb1)
save("2_read_or_scan_a_ciphertext.ipynb", nb2)
