"""Names and addresses that the maintainer sets once.

Every value marked PLACEHOLDER must be replaced when the GitHub and Hugging Face
repositories exist. See docs/MAINTAINER_SETUP.md for the full list.
"""

# GitHub repository holding this code (owner/name).
GITHUB_REPO = "mbruton0426/keybridge"

# Hugging Face model repository holding index.json and one folder per key (owner/name).
# A local folder with the same layout also works (for testing, or fully offline use).
HF_MODEL_REPO = "mbruton/keybridge-models"

# Where people send shared-model bundles by email (empty: no address is shown).
SHARE_CONTACT_EMAIL = 'micaella.bruton@ling.su.se'

# Built-in plaintext for training: one public Hugging Face dataset per language, each with
# "train" and "validation" splits and a "plaintext" column (historical text, HistCorp).
# Load them with datasets.load_dataset(repo); texts are ds["train"]["plaintext"] etc.
PLAINTEXT_DATASETS = {
    "English": "mbruton/english_encrypted_HistCiph",
    "Czech": "mbruton/czech_encrypted_HistCiph",
    "Dutch": "mbruton/dutch_encrypted_HistCiph",
    "French": "mbruton/french_encrypted_HistCiph",
    "Hungarian": "mbruton/hungarian_encrypted_HistCiph",
    "Icelandic": "mbruton/icelandic_encrypted_HistCiph",
    "Italian": "mbruton/italian_encrypted_HistCiph",
    "Polish": "mbruton/polish_encrypted_HistCiph",
    "Spanish": "mbruton/spanish_encrypted_HistCiph",
    "Swedish": "mbruton/swedish_encrypted_HistCiph",
}
DEFAULT_PLAINTEXT_LANGUAGE = "English"
PLAINTEXT_FIELD = "plaintext"

CITATION = "[citation to be added]"  # PLACEHOLDER
LICENSE = "CC-BY-4.0"  # licence of the models and keys in the collection (the code is MIT)
