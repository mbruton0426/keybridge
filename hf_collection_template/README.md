---
license: cc-by-4.0
tags:
  - cryptology
  - historical-cryptology
  - homophonic-cipher
  - substitution-cipher
  - decipherment
library_name: pytorch
---


# Shared decipherment models for historical substitution ciphers

Each model in this collection has been trained for **one known historical cipher key**. It reads transcribed ciphertexts written with that key and produces a draft decipherment. Testing a ciphertext against all models shows which known key, if any, it was written with.

Use the models with the notebooks or command-line tool at **https://github.com/mbruton0426/keybridge**. You do not need to download anything from here by hand.

## Contents

| File | Purpose |
|------|---------|
| `index.json` | Lists every model with its symbols, transcription format, language and source. The tools read this file first. |
| `<model id>/<model id>_model.pt` | The trained model. |
| `<model id>/<model id>_key.tsv` | The key it was trained on. |
| `<model id>/<model id>_training_report.txt` | Training settings and accuracy on synthetic text. |
| `<model id>/README.md` | Description, source of the key, and contributor. |
| `submissions/` | Models contributed by others, waiting for review. |

## Models

| ID | Cipher | Transcription format | Source of the key |
|----|--------|----------------------|-------------------|
| `maximilian_II` | Letters of Maximilian II (1575) | symbols separated by spaces (single characters; convert one-character-per-symbol files with `keybridge convert`) | DECODE IDs 1471, 1412, 1407 |
| `maximilian_1472` | Maximilian letter with a different key | symbols separated by spaces (named symbols) | DECODE IDs 1472, 1473 |
| `copiale` | Copiale cipher | symbols separated by spaces | DECODE ID 2 |

All three were trained on 5,000 synthetic examples from the built-in English HistCiph plaintext, with rare-unit insertion at 0.02 (see each model's training report).

## Contributing a model

Train a model for your key with notebook ① and tick *Share my model*. Your contribution will be reviewed and added to `index.json`.

## Licence and citation

All models and keys are released under **CC BY 4.0**: you may use and adapt them for any purpose, provided you give appropriate credit. Please cite:

> [citation to be added] <!-- PLACEHOLDER -->

When you use a specific model, please also credit the original source of its key, given in that model's README.
