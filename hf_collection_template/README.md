---
license: cc-by-4.0
tags:
  - cryptology
  - historical-cryptology
  - homophonic-cipher
  - substitution-cipher
  - decryption
library_name: pytorch
---


# Decryption models for historical substitution ciphers

Each model in this collection has been trained for **one known historical cipher key**. It reads transcribed ciphertexts written with that key and produces a draft decryption for expert review. Testing a ciphertext against all models (a *scan*) suggests which known key, if any, it was written with; the drafts of the best-ranked models are then read to decide. The models do not recover unknown keys.

Use the models with the notebooks or command-line tool at **https://github.com/mbruton0426/keybridge**. You do not need to download anything from here by hand.

## Contents

| File | Purpose |
|------|---------|
| `index.json` | Lists every model with its symbols, transcription format, language and source. The tools read this file first. |
| `<model id>/<model id>_model.pt` | The trained model. |
| `<model id>/<model id>_key.tsv` | The key it was trained on. |
| `<model id>/<model id>_training_report.txt` | Training settings and accuracy on synthetic text. |
| `<model id>/README.md` | Description, source of the key, test results and contributor. |
| `submissions/` | Models contributed by others, waiting for review. |

## Models

The initial release: 35 models, one for each key built from the DECODE database (records are DECODE record numbers). *Digits*: digit keys, whose ciphertexts may be written as running digits (keybridge finds the codes). All other transcriptions are written as symbols separated by spaces, exactly as in the key.

| ID | Cipher | Language | Period | Transcription | DECODE records |
|----|--------|----------|--------|---------------|----------------|
| `copiale_5000_english` | Copiale cipher, manuscript of a secret society | German | 1734-1736 | named symbols | 2 |
| `maximilian_II_5000_english` | Maximilian II to his ambassadors in Poland and Lithuania | German | 1575 | single characters | 1407, 1412, 1471 |
| `maximilian_1472_5000_english` | Johan Chodkiewicz to Maximilian II | German, Latin | 1574-1575 | named symbols | 1472, 1473 |
| `borg_A_5000_english` | Borg cipher (BAV Borg.lat. 898), Latin medical manuscript | Latin | not dated | symbols | 210 |
| `ramanacoil_A_5000_english` | Rijcklof van Goens at Ramanacoil, Ceylon, to the VOC (Heeren XVII) | Dutch | 1674 | named symbols | 1564 |
| `ucl_A_5000_english` | English letter, Brougham Archive, UCL Special Collections | English | 1724 | number codes separated by spaces | 6317 |
| `kew_E_5000_english` | Letters in TNA SP 106 (Dr Stephen Goffe and others) | English | 1645-1646 | codes separated by spaces (e.g. n3) | 928, 929 |
| `bnf_C_5000_english` | Philip II to Juan de Vargas Mexía (BnF Espagnol 132) | Spanish | 1575-1580 | symbols (key incomplete: codes with marks not included) | 1964-2025 (48 records) |
| `tna_CK_5000_english` | Italian key, TNA SP 106/10 | Italian | not dated | whole codes (e.g. z19 m23), separated by spaces | 723 |
| `na_B_5000_english` | Codebook of S. E. Croiset for Dirk van Hogendorp; letters from Saint Petersburg | Dutch | 1803 (key); 1808-1810 (letters) | digits with end marks | 1033, 1035 |
| `kha_A_5000_english` | Comte d'Avaux, The Hague, to Louis XIV | French | 1684 | digits | 1055 |
| `kew_A_5000_english` | Henrietta Maria to Charles I (TNA SP 106) | English, French | 1645 | digits | 752, 754, 785, 788, 790, 923 |
| `kew_B_5000_english` | Henrietta Maria to Charles I (TNA SP 106) | English, French | 1645 | digits | 787, 788 |
| `kew_C_5000_english` | Lord Digby to Charles I (TNA SP 106) | English | 1645 | digits | 789 |
| `kew_D_5000_english` | Henrietta Maria to Charles I (TNA SP 106) | English, French | 1646 | digits | 924, 925 |
| `kew_F_5000_english` | Unknown writer to Lord Goring (TNA SP 106) | English | 1645 | digits | 932 |
| `vatican_A_5000_english` | Papal correspondence, ASV Francia 4 | Italian | 1587 | digits | 5 |
| `vatican_B_5000_english` | Papal correspondence, ASV Francia 3 | Italian | 1552-1554 | digits | 6-10 |
| `vatican_C_5000_english` | Papal correspondence, ASV Francia 6 | Italian | 1573-1574 | digits | 11, 12 |
| `vatican_D_5000_english` | Papal correspondence, ASV Francia 17 | Italian | 1583-1585 | digits | 13, 14 |
| `vatican_E_5000_english` | Papal correspondence, ASV Francia 18 | Italian | 1585-1587 | digits | 15, 16 |
| `vatican_F_5000_english` | Papal correspondence, ASV Francia 22 | Italian | 1585-1589 | digits | 17-58 (35 records) |
| `vatican_G_5000_english` | Papal correspondence, ASV C15 | Italian | 1721 | digits | 24 |
| `vatican_H_5000_english` | Papal correspondence, ASV P117 | Italian | 1757-1761 | digits | 25, 195-201 (7 records) |
| `vatican_I_5000_english` | Papal correspondence, ASV S423 | Italian | 1736 | digits | 26, 179-188 |
| `vatican_J_5000_english` | Papal correspondence, ASV Francia 22b | Italian | 1588-1589 | digits | 53, 54 |
| `vatican_K_5000_english` | Papal correspondence, ASV Francia 64 | Italian | 1624-1625 | digits | 59-71 (13 records) |
| `vatican_L_5000_english` | Papal correspondence, ASV Francia 283 | Italian | 1580 | digits | 72, 73 |
| `vatican_M_5000_english` | Papal correspondence, ASV Francia 346 | Italian | 1632-1634 | digits | 74-84 |
| `vatican_N_5000_english` | Papal correspondence, ASV S304 | Italian | 1767 | digits | 120, 121 |
| `vatican_O_5000_english` | Papal correspondence, ASV S364 (Secretariat to the nuncio in Spain) | Italian | 1717-1720 | digits | 122-177 |
| `vatican_P_5000_english` | Papal correspondence, not given | Italian | 1535-1536 | digits | 189 |
| `vatican_Q_5000_english` | Papal correspondence, ASV P8 | Italian | 1579-1581 | digits | 190-194 |
| `vatican_R_5000_english` | Papal correspondence, ASV C13 | Italian | 1656-1657 | digits | 208, 209 |
| `vatican_S_5000_english` | Papal correspondence, BAV Barb.lat. 6956 and 6960 (Pallotto) | Italian | 1628-1629 | digits | 215-225 (10 records) |

All 35 were trained with keybridge's released defaults: 5,000 synthetic examples from the built-in English HistCiph plaintext, rare-unit insertion 0.02 (see each model's training report). Where a model has test letters, its README gives the results against the *key reference* (the transcription read through the key) and, for digit keys, against the transcribers' readings.

This table lists only the 35 models of the initial release. Models added later are listed on the collection's page on Hugging Face (https://huggingface.co/mbruton/keybridge-models), not here.

## Contributing a model

Train a model for your key with notebook ① and tick *Share my model*. Your contribution will be reviewed and added to `index.json`.

## Licence and citation

All models and keys are released under **CC BY 4.0**: you may use and adapt them for any purpose, provided you give appropriate credit. Please cite:

> [citation to be added] <!-- PLACEHOLDER -->

When you use a specific model, please also credit the original source of its key, given in that model's README.