---
name: word-frequency
description: Count word frequencies in a text file in the workspace. Use when asked which words are most common in a document, for a word count, or for a vocabulary or repetition check on a file.
license: MIT
metadata:
  author: agentharness
  version: "1.0"
---

# Word frequency

Counting words by eye is exactly the kind of task a language model does
confidently and wrongly. Do not do it by hand, and do not write a new script to
do it: run the bundled one.

## Usage

Call `run_skill_script` with `skill: "word-frequency"` and
`path: "scripts/wordfreq.py"`. Paths in `args` are relative to the workspace
root, which is the script's working directory.

    args: ["notes.md"]                        # top 20 words
    args: ["notes.md", "--top", "5"]          # top 5
    args: ["notes.md", "--min-length", "4"]   # skip short function words

It prints a total, a distinct count, and one line per word, most frequent
first. A missing or unreadable file exits 1 with the reason on stderr — report
that rather than guessing at the contents.

## Reporting the result

Quote the counts as given. If the user wants prose rather than a table,
summarise the top few and say how many distinct words there were.
