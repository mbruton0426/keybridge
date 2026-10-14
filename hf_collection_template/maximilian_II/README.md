# maximilian_II

Example of the README that goes in each model's folder. Notebook ① writes one like this automatically when a user shares a model.

- **Cipher:** three letters of Maximilian II concerning the 1575 Polish–Lithuanian royal election
- **Key source:** DECODE IDs 1471, 1412, 1407; Kopal et al. (2022)
- **Transcription format:** symbols separated by spaces (word divider `_`)
- **Plaintext language:** German (Early New High German)
- **Training plaintext:** HistCorp German, with letter replacements J=I, V=U
- **Training examples:** 10,000 × 50 symbols
- **Accuracy on unseen synthetic text:** 0.998 *(example value)*
- **Contributed by:** Micaella Bruton
- **Licence:** CC BY 4.0

**Known limitations:** nomenclature symbols (Austria, Hungaria, Germania, enclitica) rarely or never occur in the training plaintext, so the model is unlikely to read them correctly.
