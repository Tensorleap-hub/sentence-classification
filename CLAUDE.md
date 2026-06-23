# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small clinical-assertion classifier (the "sentence_clf" task) plus the artifacts a
Tensorleap integration needs. Given a clinical sentence with one entity marked
inline (`... [ENTITY]chest pain[/ENTITY] ...`), it classifies the entity's
assertion status as **Absent**, **Hypothetical**, or **Present**.

The committed deliverables are the trained artifacts: `tokenizer/spm.model`,
`model/model.onnx`, `config.json`, and the `data/processed/` split. The
PyTorch checkpoint (`model/model.pt`) is intermediate and git-ignored.

## Commands

Setup uses Poetry with an in-project `.venv` (`poetry.toml`). Python is pinned
to `>=3.10,<3.11`.

```bash
poetry install
poetry run python scripts/<name>.py   # run any pipeline step
```

The full pipeline, in order (each step's output feeds the next):

```bash
poetry run python scripts/download_data.py     # -> data/raw/ (anonymous Kaggle download)
poetry run python scripts/prepare_data.py      # -> data/processed/{train,val}.csv, labels.json
poetry run python scripts/train_tokenizer.py   # -> tokenizer/spm.{model,vocab}, config.json
poetry run python scripts/train_model.py       # -> model/model.pt
poetry run python scripts/export_onnx.py       # -> model/model.onnx (+ ONNX/torch parity check)
```

There is no test suite or linter configured. `export_onnx.py` doubles as the
end-to-end verification: it exits non-zero if ONNX-vs-torch logit parity drifts
past `1e-3`.

### Tensorleap integration

```bash
poetry run python leap_integration.py          # progressive validator: prints the exit-status table
poetry run python <skill>/scripts/tl_check.py "$(pwd)"   # structured check_dataset() acceptance (pass ABSOLUTE path)
```

`leap_integration.py` validation fires when a decorated function is *called*, so
its `__main__` block calls `integration_test(...)` over the first few train/val
samples — run it after any edit. The structured checker (`tl_check.py`) must be
given the **absolute** repo root; a relative `.` triggers an infinite-recursion
bug in code_loader's path walker. `isValidForModel: false` is expected and fine
— it only means "no custom layers."

## Architecture

The central design principle: **`sentence_clf/text.py` is the single source of truth
for encoding**, imported by both the training scripts and `leap_integration.py`,
so a raw sentence encodes identically at train time and in Tensorleap. Never
duplicate or fork the encoding logic — change it in one place.

- **`sentence_clf/text.py`** — raw tagged sentence → fixed 40-token input.
  - `split_entity` handles two interchangeable marker conventions in the
    corpus: distinct `[ENTITY]...[/ENTITY]` tags, and a repeated `[entity]`
    marker (both case-insensitive). It raises `ValueError` on unusable rows so
    callers drop them explicitly.
  - `encode_window`: take 20 tokens immediately **before** the entity
    (left-padded) + 20 immediately **after** (right-padded) = 40. The entity
    text itself is masked out — it's the thing being classified.
  - Per-side normalization is deaccent → lowercase → whitespace-collapse.
  - `LABELS` defines the fixed, deterministic class order; the model's 3 output
    logits and Tensorleap's prediction labels both follow it.

- **`sentence_clf/model.py`** — `SentenceClassifier`, a tiny FNN: `Embedding(vocab, 32,
  padding_idx=0)` → flatten to `[B, 1280]` → `Linear(1280, 64)` + ReLU +
  dropout → `Linear(64, 3)`. Outputs **raw logits**, not probabilities —
  softmax is deliberately left to a downstream visualizer/metric so the
  Tensorleap prediction type maps cleanly onto the 3 classes.

- **`config.json`** — written by `train_tokenizer.py`, then read by
  `train_model.py` and `export_onnx.py`. It is the shared contract (vocab size,
  dims, tensor names, paths). Treat it as generated, not hand-edited.

- **ONNX I/O contract** (`export_onnx.py`): the exported graph accepts
  **float32** input of shape `(batch, 40)` and casts to int64 internally
  (`FloatInputWrapper`). This lets a Tensorleap input encoder return float32
  (the code-loader contract) and feed the model directly. Input name `tokens`,
  output name `logits`, both with a dynamic batch axis.

- **`leap_integration.py` + `leap.yaml`** — the decorator-style Tensorleap
  integration. `preprocess()` reads the split from a config-driven dir
  (`SENTENCE_CLF_DATA_DIR`, default `data/processed/`; the dataset is *mounted* on the
  platform, so it is **not** bundled in `leap.yaml`). One input encoder
  (`tokens`), one-hot GT (`assertion`), ONNX `load_model`, plus cross-entropy
  loss, accuracy metric, class-probability + decoded-token visualizers, and
  metadata. All postprocessing (softmax/argmax) lives in these decorated
  functions, never in the model. Do **not** add `from __future__ import
  annotations` — code_loader reads `__annotations__` for the real visualizer
  return *class*, and stringized annotations break registration.

## Things to know

- **Vocab size**: the spec requested a 40k-vocab tokenizer, but the ~10k-line
  corpus can't support it. `train_tokenizer.py` walks `VOCAB_CANDIDATES` down
  from 40k and records the first that trains (currently 4000) in
  `config.json`'s `vocab_size`; `requested_vocab` keeps the original ask. Read
  the actual size from config, never assume.
- **Determinism**: data split, tokenizer corpus, and training all use seed 42.
- **Class imbalance**: `prepare_data.py` does a per-label stratified split so
  the rare **Hypothetical** class lands in both train and val.
- Raw entity tags are kept in `data/processed/*.csv` (not pre-encoded) so the
  split stays reproducible from canonical text via `encode_window`.
- **numpy is pinned `<2.0`** because `code-loader` (Tensorleap's runtime)
  requires it; this was deliberately downgraded from the original numpy 2.x pin.
  Do not bump numpy back to 2.x — it makes `code-loader` uninstallable and
  breaks the integration. The ONNX model and sentencepiece encoding are
  numpy-version-independent, so 1.x is safe for training/export too.
