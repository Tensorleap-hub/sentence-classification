# sentence_clf — clinical assertion classifier (Tensorleap integration)

Given a clinical sentence with one entity marked inline
(`... [ENTITY]chest pain[/ENTITY] ...`), the model classifies the entity's
assertion status as **Absent**, **Hypothetical**, or **Present**. The trained
model, tokenizer, processed data split, and config are committed, so the
integration runs out of the box once dependencies are installed and the dataset
is placed in the Tensorleap data volume.

## 1. Install

Python is pinned to `>=3.10,<3.11`; dependencies are managed with Poetry (an
in-project `.venv`).

```bash
poetry install
```

This installs `code-loader` (the Tensorleap runtime). Note: **numpy is pinned
`<2.0`** because `code-loader` requires it — do not bump it to 2.x.

## 2. Get the data

The processed split is **already committed** at `data/processed/`
(`train.csv`, `val.csv`, `labels.json`), so you normally don't need to download
anything. To regenerate it from the public source:

```bash
poetry run python scripts/download_data.py    # -> data/raw/ (anonymous Kaggle download)
poetry run python scripts/prepare_data.py      # -> data/processed/{train,val}.csv, labels.json
```

## 3. Place the data in the Tensorleap data volume (required for the platform)

Tensorleap can only read data that lives in its **data volume** — the dataset is
*not* bundled with the integration code. Find the data volume and copy the split
into a per-project folder:

```bash
# Local server: find the data-volume host path
leap server info          # read the `datasetvolumes:` entry, e.g. /Users/you/tensorleap/data

# Create a per-project folder and copy the split into it
DATA_VOL=/Users/you/tensorleap/data            # <- from `leap server info`
mkdir -p "$DATA_VOL/sentence-classification"
cp data/processed/train.csv data/processed/val.csv data/processed/labels.json \
   "$DATA_VOL/sentence-classification/"
```

For a non-local/remote server, ask whoever administers it for the data directory.

Then point the integration's data path at that folder. The integration resolves
it in this order:

1. `SENTENCE_CLF_DATA_DIR` environment variable, else
2. `data_dir` in `config.json`, else
3. the local default `data/processed/`.

Set whichever fits your setup, e.g.:

```bash
export SENTENCE_CLF_DATA_DIR="$DATA_VOL/sentence-classification"
```

(`config.json`'s `data_dir` is currently set to a local data-volume path; change
it to match your machine, or override with the env var above.)

## 4. Model & tokenizer

`model/model.onnx` (loaded at inference) and `model/model.pt` (the PyTorch
training checkpoint) are both committed for reproducibility, along with
`tokenizer/spm.model`.

On the platform the **model is uploaded separately** as its own artifact — it is
deliberately *not* listed in `leap.yaml`. `@tensorleap_load_model` loads it only
for local runs and the integration test.

To retrain / re-export the artifacts:

```bash
poetry run python scripts/train_tokenizer.py   # -> tokenizer/spm.model, config.json
poetry run python scripts/train_model.py        # -> model/model.pt
poetry run python scripts/export_onnx.py        # -> model/model.onnx (+ parity check)
```

## 5. Validate the integration locally

```bash
poetry run python leap_integration.py
```

Expect the exit-status table with all parts ✅ and `Successful!` for each sample.
This exercises preprocess → encoders → model → loss/metric/visualizers reading the
dataset from the path resolved in step 3.

## 6. What gets uploaded where

| Artifact | Destination |
|---|---|
| `leap_integration.py`, `leap.yaml`, `config.json`, `tokenizer/`, `sentence_clf/`, `requirements.txt` | bundled via `leap.yaml` `include` |
| dataset (`train.csv`, `val.csv`) | Tensorleap **data volume** (step 3) |
| `model/model.onnx` | uploaded to Tensorleap **separately** (not in `leap.yaml`) |

See `CLAUDE.md` for architecture and implementation details.
