"""The shared model collection: an index.json plus one folder per key.

The collection lives in a Hugging Face model repository (config.HF_MODEL_REPO) or,
for testing and fully offline use, in a local folder with the same layout:

    index.json
    maximilian_II/maximilian_II_model.pt
    maximilian_II/maximilian_II_key.tsv
    maximilian_II/README.md
    ...
"""
import datetime
import hashlib
import json
import os

import torch

from .common import stop
from .config import CITATION, HF_MODEL_REPO, LICENSE
from .keys import CipherKey

INDEX_FILE = "index.json"


class ModelRegistry:
    def __init__(self, source=None):
        self.source = source or HF_MODEL_REPO
        self.local = os.path.isdir(self.source)
        self._index = None

    def _fetch(self, filename):
        if self.local:
            path = os.path.join(self.source, filename)
            if not os.path.exists(path):
                stop(f"{filename} was not found in the model folder {self.source}")
            return path
        if "HF_MODEL_REPO" in self.source:
            stop("The shared model collection is not set up yet (the repository name is still a placeholder).\n"
                 "Use your own model file instead, or give a model folder.")
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(repo_id=self.source, filename=filename, repo_type="model")
        except Exception as e:
            stop(f"Could not get {filename} from the shared collection {self.source} ({type(e).__name__}).\n"
                 "Check your internet connection. If you downloaded the models before, set the environment\n"
                 "variable HF_HUB_OFFLINE=1 to use the downloaded copies.")

    def index(self):
        if self._index is None:
            with open(self._fetch(INDEX_FILE), encoding="utf-8") as f:
                self._index = json.load(f)
        return self._index

    def entries(self, symbol_format=None, symbol_type=None, ids=None):
        out = []
        for e in self.index().get("models", []):
            if ids and e["id"] not in ids:
                continue
            if symbol_format and e.get("symbol_format") != symbol_format:
                continue
            if symbol_type and e.get("symbol_type") != symbol_type:
                continue
            out.append(e)
        if ids:
            missing = set(ids) - {e["id"] for e in out}
            if missing:
                stop("These model IDs are not in the collection (or do not match the transcription format): "
                     + ", ".join(sorted(missing)))
        return out

    def model_path(self, entry):
        return self._fetch(entry["path"])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_index_entry(model_path, model_id=None):
    """Builds an index.json entry from a trained model file. Fields marked FILL IN need
    to be completed by hand."""
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    key = CipherKey.from_json(ck["key"])
    s, m = ck.get("settings", {}), ck.get("metrics", {})
    model_id = model_id or s.get("project") or os.path.basename(model_path).replace("_model.pt", "")
    digit = s.get("digit_mode_info")
    lengths = {}
    for sym in key.symbols:
        lengths[str(len(sym))] = lengths.get(str(len(sym)), 0) + 1
    return {
        "id": model_id,
        "name": "FILL IN: short human-readable name",
        "path": f"{model_id}/{os.path.basename(model_path)}",
        "key_file": f"{model_id}/{model_id}_key.tsv",
        "sha256": sha256(model_path),
        "symbol_format": s.get("symbol_format"),
        "symbol_type": "digits" if digit else key.symbol_type(),
        "digit_mode": {"marks": digit["marks"], "pieces": digit["pieces"]} if digit else None,
        "symbols": key.symbols,
        "symbol_lengths": lengths,
        "plaintext_units": len(key.units),
        "letter_replacements": s.get("letter_replacements", ""),
        "plaintext_language": "FILL IN: language code of the cipher's plaintext, if known (e.g. de)",
        "training_plaintext": s.get("plaintext_source", ""),
        "period": "FILL IN: e.g. 1575",
        "source": "FILL IN: archive / shelfmark / DECODE ID / publication of the key",
        "contributor": "FILL IN",
        "training_examples": s.get("training_examples"),
        "synthetic_test_accuracy": m.get("synthetic_test_accuracy"),
        "uncertain_share_in_key_reference": m.get("uncertain_share_in_key_reference"),
        "uncertain_share_scrambled_key_reference": m.get("uncertain_share_scrambled_key_reference"),
        "package_version": s.get("package_version"),
        "added": datetime.date.today().isoformat(),
    }


def new_index():
    return {"index_version": 1, "updated": datetime.date.today().isoformat(),
            "license": LICENSE, "citation": CITATION, "models": []}


def entry_symbol_coverage(entry_symbols, numbered_lines, digit_mode=None):
    """Share of the transcription's symbols that are in the key. For digit models (digit_mode with
    'marks' and 'pieces'), the share of digits (with their marks) that the model knows."""
    if digit_mode:
        from .digits import piece_regex, tokenize_symbols
        rx = piece_regex(digit_mode.get("marks", ""))
        known = set(digit_mode.get("pieces", []))
        pieces = [p for _, line in numbered_lines for p, _ in tokenize_symbols(line, rx)]
        return sum(p in known for p in pieces) / max(1, len(pieces))
    known = set(entry_symbols)
    symbols = [s for _, line in numbered_lines for s in line]
    return sum(s in known for s in symbols) / max(1, len(symbols))


