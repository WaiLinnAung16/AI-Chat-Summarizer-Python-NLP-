# AI Chat Summarizer

A small Python tool that turns plain text or chat logs into **bullet key points** using **extractive summarization**: it picks the most important **sentences** from your text according to **TF-IDF** scores, after light **NLP preprocessing**.

This is **not** an abstractive model (it does not rewrite or paraphrase). Output lines are **verbatim sentences** from the input, ordered by importance (highest TF-IDF mass first).

---

## Features

- **Preprocessing** (for scoring only): lowercase, strip punctuation, remove English stop words (`nltk`).
- **Sentence splitting** with NLTK (`sent_tokenize`), not naive splitting on periods.
- **TF-IDF** (`scikit-learn`) over preprocessed sentences; each sentence’s score is the sum of its term TF-IDF weights.
- **Single or multiple files**: pass one path or several; multiple files are read in order and concatenated with blank lines between parts.
- **CLI**: positional file list and optional number of bullets.

---

## Requirements

- Python 3.x
- Dependencies in `requirements.txt`: `nltk`, `scikit-learn`

---

## Installation

```bash
pip install -r requirements.txt
```

On Windows, if `python` is not on your `PATH`, use the launcher:

```bash
py -3 -m pip install -r requirements.txt
```

**NLTK data** (`punkt`, `punkt_tab`, `stopwords`) is downloaded automatically on first run via `download_nltk_if_needed()` in `preprocess.py`.

---

## Usage

```text
python main.py FILE [FILE ...] [-n N]
```

| Argument | Description |
|----------|-------------|
| `FILE` | One or more text files (required). With multiple files, content is combined in the order given. |
| `-n`, `--num-points` | Number of key-point bullets to print (default: `5`). Must be ≥ 1. |

**Examples**

```bash
python main.py conversation.txt
python main.py chat_part1.txt chat_part2.txt
python main.py notes.txt -n 8
python main.py --num-points 3 log.txt
```

### Input encoding

Files are read as **UTF-8** by default (see `file_handler.read_file`). There is no CLI flag to change encoding; if you need another encoding, convert the file to UTF-8 first or adjust `read_file` in code.

---

## Project layout

| File | Role |
|------|------|
| `main.py` | CLI entrypoint: parses arguments, loads text, prints bullets. |
| `file_handler.py` | Reads one file or many paths; joins multiple inputs with `\n\n`. |
| `preprocess.py` | `preprocess_text()`, `get_stopwords()`, `download_nltk_if_needed()`. |
| `summarizer.py` | `summarize_tfidf()`: tokenize sentences → preprocess → TF-IDF → top-N sentences. |

---

## How summarization works

1. **Segment** the full document into sentences with NLTK.
2. **Preprocess** each sentence for vectorization: lowercase, remove punctuation, drop English stop words.
3. **Fit** a `TfidfVectorizer` on the list of preprocessed sentence strings (document = one sentence).
4. **Score** each sentence by summing TF-IDF values across its row in the matrix.
5. **Select** the sentences with the highest scores (up to `-n`), keeping **original** sentence text for output.

Sentences that become empty after preprocessing still participate in a fallback path (see `summarizer.py`): if nothing survives preprocessing, the first *N* raw sentences may be returned.

---

## Limitations

- Quality depends on **punctuation and structure**: very short inputs or texts with few sentence boundaries may yield few or odd bullets.
- **English** stop words are assumed; other languages are not specially handled.
- **Importance order** is by TF-IDF score, not chronological order of the chat.

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `2` | Invalid arguments (for example `--num-points` set to less than 1) |

---

## License

Add a license file if you distribute this project; none is included by default.
