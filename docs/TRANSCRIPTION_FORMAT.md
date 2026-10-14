# Transcription format

A transcription is the ciphertext typed out symbol by symbol, so the tool can read it. It is a plain text file (`.txt`), used both for training (optional) and for reading or scanning.

## The standard format

Every key and transcription uses the same format: **cipher symbols separated by spaces.** This works for all kinds of ciphers:

```
pi oh j v hd tri arr eh three c. ah ni arr lam uh          ← named symbols (Copiale)
0675 4770 454 0086 443                                      ← digit codes
H q X 1 _ * v X D q _ + G F                                 ← single characters (Maximilian)
```

Because every model is trained in this format, any transcription can be tested against any model.

## The rules

1. **One manuscript line per line** of the file.
2. **Symbols separated by spaces.** The number of spaces does not matter. Spaces only separate symbols; they are never symbols themselves.
3. **Write the cipher symbols exactly as they appear in the key,** including upper and lower case and the spelling of symbol names. The tool matches symbols to the key letter for letter: if the key says `"lam"`, the transcription must say `lam`, not `Lam` or `lambda`.
4. **A blank used as a cipher symbol is written `_`.** Some ciphers use a gap or blank as a symbol, for example as a word divider (as in the Maximilian letters). Write it as `_`, in the key as `"_"` and in the transcription as `_`.
5. **No quotes and no special names.** Quotes are only used in the key file. Never write `<SPACE>`, `<NULL>` or `<DOUBLE>` in a transcription; write the symbol the manuscript shows (see [below](#how-word-gaps-nulls-and-the-double-sign-work)).
6. **Codes must be separated.** `067547700454` cannot be split into codes; write `0675 4770 0454`.
7. **Save the file as UTF-8** if your editor asks.

## Older "one character per symbol" files

Transcriptions in which every character is one symbol, written without spaces between symbols (`HqX1 *vXDq +GF`), are still accepted:

- **Converted automatically when read:** choose "one character per symbol" as the transcription format. Each character becomes a symbol, and every blank becomes `_`.
- **Converted permanently:** run `keybridge convert --transcription letter.txt` ([local version](LOCAL_INSTALL.md#convert-to-the-standard-format)). This writes `letter_standard.txt` in the standard format.

Likewise, if a key lists a blank `" "` as a cipher symbol, it is automatically treated as `"_"`. `keybridge convert --key my_key.tsv` writes the key in the standard form. If a key uses both `" "` and `"_"` as symbols, the tool stops and asks you to rename the existing `"_"` symbol first, so the two cannot be confused.

**In all outputs, a blank cipher symbol appears as `_`:** in the cipher lines of the decipherment file, and in the per-symbol tables. In the aligned reading, a word gap is shown as `_` (or `<SPACE>` if `_` is a plaintext unit of the key); in the flowing `text:` line it is a normal space.

## Notes, page numbers and uncertain readings

Everything on a ciphertext line is read as cipher symbols. So:

- **Whole lines of notes** (page numbers, section headings) can stay in the file if they start with a marker listed in the *Skip lines starting with* setting. The default is `##`, e.g. `## PAGE 3`. Several markers can be given, separated by commas, e.g. `##, <, Section`.
- **Only `##` is skipped by default, not a single `#`,** because `#` can itself be a cipher symbol (in the Maximilian key it is a null). If you use a different marker, add it to *Skip lines starting with* (command line: `--skip`), and make sure no ciphertext line begins with it.
- **Notes inside a line** must be removed before use: comments in brackets, `[illegible]`, `<...>` annotations, catchwords and similar.
- **Uncertain readings** such as `6/b?` must be resolved to one symbol, or the symbol removed or replaced by one that is not in the key. Symbols that are not in the key are allowed: they are counted, reported, and read as unknown.
- Empty lines are ignored.
- Very long lines (more than twice the training length, normally 100 symbols) are read in pieces.

The tool does not change the transcription in any other way. Python users can normalise transcription conventions with the custom edit described in the guides.

## How word gaps, nulls and the double sign work

A key line lists the plaintext first, then the cipher symbols that encode it: `plaintext → symbol → symbol → …`. The special entries `<SPACE>`, `<NULL>` and `<DOUBLE>` only ever appear in the plaintext position of a key. Ciphertexts never contain them; both the synthetic training ciphertexts and real transcriptions contain only symbols.

| Key line (plaintext → cipher symbols) | Training plaintext (what the model learns to output) | Training ciphertext (what the model reads) | How to write it in a real transcription | In the decipherment |
|---|---|---|---|---|
| `"E"` → `"ah"` `"eh"` | `E` | `ah` or `eh` | `ah` or `eh`, as in the manuscript | E |
| `<SPACE>` → `"zs"` `"a"` | a word gap | `zs` or `a` | `zs` or `a` | `_` in the aligned reading, a space in the flowing text |
| `<SPACE>` → `"_"` | a word gap | `_` | `_` where the manuscript has the blank or divider | `_` in the aligned reading, a space in the flowing text |
| `<NULL>` → `"w"` `"7"` | `<NULL>` (nothing) | `w` or `7`, inserted at random | `w` or `7` where they occur | `·` in the aligned reading, left out of the flowing text; `<NULL>` in the per-symbol table |
| `<DOUBLE>` → `":"` | `<DOUBLE>` ("repeat the letter before") | `:` in place of the second of two identical letters (about half the time) | `:` where it occurs | the previous letter, repeated |

For example, with the Copiale key, the plaintext `DES SOLLEN` might be enciphered for training as `pi uh bar zs bar o. c. : uu`. A real transcription is written the same way. The model reads `zs` as a word gap, `:` as "repeat L", and `uu` as the unit EN.

If the key has a unit for a doubled pair itself (Copiale has `"SS"` → `"grc"`), that unit is always used for the pair, and the double sign is not.

## Example files

- [`examples/maximilian_synthetic_transcription.txt`](../examples/maximilian_synthetic_transcription.txt): single-character symbols, with the word divider written as `_`.
- [`examples/copiale_synthetic_transcription.txt`](../examples/copiale_synthetic_transcription.txt): named symbols.

Both are synthetic (random words encrypted with the example keys), not real manuscripts, and both are in the standard format.
