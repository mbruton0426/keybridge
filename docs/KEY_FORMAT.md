# Key file format

A key file tells the tool which cipher symbols stand for which plaintext. It is a plain text file with the ending **`.tsv`** (tab-separated values). You can write it in any plain text editor, such as Notepad (Windows), TextEdit in plain-text mode (Mac), or VS Code.

## The rules

1. **One line per plaintext unit.** A unit is what the symbols mean: a letter, a group of letters, or a whole word.
2. **The plaintext unit comes first, in double quotes.** For example `"A"`, `"SCH"` or `"Austria"`. Upper and lower case are treated the same.
3. **Then every cipher symbol for that unit, each in its own double quotes,** separated by tabs: `"J"` → `"/"`.
4. **Write each symbol exactly as it appears in your transcriptions,** including upper and lower case: `"e"` and `"E"` are different symbols.
5. *Optional:* end the line with the word `yes` (no quotes) if the unit only ever stands for a **whole word**. This is typical of nomenclature.

Quotes make every character visible, so a blank space, a full stop or a comma used as a symbol can never be confused with the gaps between columns.

## Example

`→` stands for a tab:

```
plaintext→ciphertext→whole_word
"A"→"J"→"/"
"B"→"T"→"4"
"E"→"W"→"P"→"q"
"SCH"→"Y"
"UND"→"R"
"Austria"→"z"→yes
<SPACE>→"_"
<NULL>→"w"→"7"→"u"
```

Digit codes work in the same way:

```
"A"→"0675"→"112"→"4770"
"B"→"0086"
"THE KING"→"901"→yes
```

## Special entries

Write these **without quotes** in the plaintext position:

| Entry | Meaning |
|-------|---------|
| `<SPACE>` | The symbol(s) mark the gap between words. |
| `<NULL>` | The symbol(s) mean nothing (nulls, fillers). |
| `<DOUBLE>` | The symbol means "repeat the previous letter", as in the Copiale cipher. |

These names only appear in the key. Ciphertexts, both the synthetic training texts and your transcriptions, always contain the symbols themselves (e.g. `a`, `w`, `:`). See [how word gaps, nulls and the double sign work](TRANSCRIPTION_FORMAT.md#how-word-gaps-nulls-and-the-double-sign-work).

## Keys in the standard format

Keys and transcriptions always use the standard format: symbols separated by spaces, with a blank symbol written as `"_"` (see [TRANSCRIPTION_FORMAT.md](TRANSCRIPTION_FORMAT.md#the-standard-format)). `keybridge convert --key old_key.tsv` rewrites an older key in this form.

## Symbols with unusual characters

| You want the symbol | Write |
|----------------------|-------|
| a blank space, e.g. a word divider | `"_"`. A blank written as `" "` is converted to `"_"` automatically, so `"_"` cannot also be used for another symbol. |
| a double quote mark `"` | `"\""` |
| a backslash `\` | `"\\"` |
| anything else (`.` `,` `#` `€` `ø` …) | just put it in quotes: `"."` `","` `"#"` |

## Other lines

- A line starting with `#` is a note and is ignored, e.g. `# Key of the letters of Maximilian II, 1575`.
- A first line starting with the word `plaintext` (no quotes) is treated as a header and ignored.
- Empty lines are ignored.
- Each symbol can mean only one thing. If a symbol is listed under two units, the first is kept and you are warned.

## Common mistakes

The tool checks the file and lists every problem with its line number. The usual ones:

| Message | Fix |
|---------|-----|
| *the cipher symbol '5' must be in double quotes* | Write `"5"`. Each symbol needs its own quotes: `"1"→"2"`, not `"1 2"`. |
| *the plaintext 'D' must be in double quotes* | Write `"D"`, or use a special entry such as `<SPACE>`. |
| *empty symbol ""* | Remove the empty quotes, or write `"_"` if a blank space is really a symbol. |
| *symbols longer than one character* | You chose "one character per symbol", but your symbols have several characters. Use the standard format, "symbols separated by spaces". |

**If your editor turns tabs into spaces,** the file still works: spaces between quoted items are accepted too. Tabs are recommended because they keep the columns tidy.

**Save the file as UTF-8** if your editor asks. This is the default in most editors.

## Examples in this repository

- [`examples/maximilian_II_key.tsv`](../examples/maximilian_II_key.tsv): single-character symbols with the word divider written as `"_"`, nulls and nomenclature.
- [`examples/copiale_key.tsv`](../examples/copiale_key.tsv): named symbols separated by spaces, with `<DOUBLE>` and nomenclature.

## What keys can and cannot express

Supported: each cipher symbol stands for exactly one thing, whether a letter, a group of letters, a whole word, a word gap, a null or the double-letter sign.

Not supported:
- **Codes written without separators** in the transcription (e.g. `067547700454` meaning 0675 4770 0454). Transcribe with spaces between codes.
- **Symbols with several meanings** (polyphones).
- **Context-dependent rules**, where a symbol's meaning depends on its position or neighbours.
- **Cancellation signs** ("ignore the previous symbol").
- **Other cipher types**: polyalphabetic ciphers, transposition, and codebooks that need more than a symbol-for-unit table.

Python users can imitate some of these with the custom edits described in the guides.

*For programmatic use, a key can also be given as JSON: `{"units": {"A": ["J", "/"], "<SPACE>": ["_"]}, "whole_word": ["AUSTRIA"]}`.*
