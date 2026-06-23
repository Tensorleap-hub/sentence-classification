"""Tensorleap integration for the sentence_clf clinical-assertion classifier.

Given a clinical sentence with one entity marked inline
(``... [ENTITY]chest pain[/ENTITY] ...``), the model predicts the entity's
assertion status as one of: Absent, Hypothetical, Present.

This file is the decorator-style integration. Encoding is delegated entirely to
``sentence_clf.text`` so the path is identical to training; the model is the committed
ONNX artifact and inference runs through ONNX Runtime (Tensorleap is
inference-only — softmax/argmax live in visualizers/metrics, never here).
"""

# NOTE: do NOT add `from __future__ import annotations` — code_loader inspects
# function.__annotations__ for the real visualizer return *class* (e.g.
# LeapHorizontalBar), and stringized annotations would fail that check.

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import pandas as pd
import sentencepiece as spm

from code_loader.contract.datasetclasses import PreprocessResponse, PredictionTypeHandler
from code_loader.contract.enums import (
    DataStateType,
    DatasetMetadataType,
    LeapDataType,
    MetricDirection,
)
from code_loader.contract.visualizer_classes import LeapText, LeapHorizontalBar
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess,
    tensorleap_input_encoder,
    tensorleap_gt_encoder,
    tensorleap_load_model,
    tensorleap_custom_loss,
    tensorleap_metadata,
    tensorleap_custom_visualizer,
    tensorleap_custom_metric,
    tensorleap_integration_test,
)

from sentence_clf.text import (
    LABELS,
    LABEL_TO_INDEX,
    PAD_ID,
    WINDOW,
    encode_window,
    split_entity,
    normalize,
)

# --- paths / config ---------------------------------------------------------
# config.json is the single source of truth shared with training. Model and
# tokenizer are bundled code/assets (listed in leap.yaml). The DATASET is NOT
# bundled — it lives in the Tensorleap data volume — so its directory is
# config-driven: SENTENCE_CLF_DATA_DIR if set, else config.json's "data_dir" (the
# data-volume project folder), else the local default. Never hardcode it in code.
ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text())
DATA_DIR = Path(
    os.environ.get("SENTENCE_CLF_DATA_DIR")
    or CONFIG.get("data_dir")
    or str(ROOT / "data" / "processed")
)

MODEL_PATH = ROOT / CONFIG["model_path"]
TOKENIZER_PATH = ROOT / CONFIG["tokenizer_path"]
INPUT_NAME = CONFIG["input_name"]      # "tokens"
INPUT_LEN = CONFIG["input_len"]        # 40

# SentencePiece tokenizer is a shared, read-only asset: load once at import.
_SP = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))


# --- preprocess -------------------------------------------------------------
def _load_split(csv_name: str) -> "tuple[List[str], Dict[str, dict]]":
    """Read one split CSV into (sample_ids, records_by_id).

    The raw entity tags are kept in the CSV, so each record carries the canonical
    raw text and its label; encoding happens later via sentence_clf.text. Sample ids are
    the row indices as strings (unique and stable within the split).
    """
    df = pd.read_csv(DATA_DIR / csv_name)
    records: Dict[str, dict] = {}
    sample_ids: List[str] = []
    for idx, row in df.iterrows():
        sid = str(idx)
        records[sid] = {"text": str(row["text"]), "label": str(row["label"])}
        sample_ids.append(sid)
    return sample_ids, records


@tensorleap_preprocess()
def preprocess() -> List[PreprocessResponse]:
    train_ids, train_records = _load_split("train.csv")
    val_ids, val_records = _load_split("val.csv")
    training = PreprocessResponse(
        sample_ids=train_ids,
        data={"records": train_records},
        state=DataStateType.training,
    )
    validation = PreprocessResponse(
        sample_ids=val_ids,
        data={"records": val_records},
        state=DataStateType.validation,
    )
    return [training, validation]


# --- input encoder ----------------------------------------------------------
@tensorleap_input_encoder(name=INPUT_NAME, channel_dim=-1)
def tokens_input(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    """Encode one sample's raw text into the fixed 40-token model input.

    Delegates to sentence_clf.text.encode_window (20 tokens before the entity +
    20 after, entity masked out) so train-time and platform encoding match
    exactly. Returns a single unbatched float32 vector of shape (input_len,);
    the ONNX graph casts to int64 internally.
    """
    record = preprocess.data["records"][sample_id]
    ids = encode_window(record["text"], _SP)
    return np.asarray(ids, dtype=np.float32)


# --- model ------------------------------------------------------------------
# The model has a single output: raw logits over the 3 assertion classes
# (axis -1). Softmax/argmax are deliberately NOT applied here.
PREDICTION_TYPES = [
    PredictionTypeHandler(name="assertion", labels=LABELS, channel_dim=-1),
]


@tensorleap_load_model(PREDICTION_TYPES)
def load_model() -> ort.InferenceSession:
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


# --- ground truth -----------------------------------------------------------
@tensorleap_gt_encoder(name="assertion")
def assertion_gt(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    """One-hot ground truth over the fixed LABELS order, shape (num_classes,)."""
    record = preprocess.data["records"][sample_id]
    one_hot = np.zeros(len(LABELS), dtype=np.float32)
    one_hot[LABEL_TO_INDEX[record["label"]]] = 1.0
    return one_hot


# --- loss -------------------------------------------------------------------
def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


@tensorleap_custom_loss("cross_entropy")
def cross_entropy_loss(predictions: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-sample categorical cross-entropy from raw logits + one-hot GT.

    Operates on already-batched arrays ((B, num_classes)) and returns a
    batch-aligned 1D array (one loss per sample). Softmax lives here, not in the
    model.
    """
    probs = _softmax(predictions.astype(np.float64))
    ce = -np.sum(gt * np.log(probs + 1e-7), axis=-1)
    return ce.astype(np.float32)


# --- visualizers ------------------------------------------------------------
@tensorleap_custom_visualizer("class_probabilities", LeapDataType.HorizontalBar)
def class_probabilities_visualizer(
    predictions: np.ndarray, gt: np.ndarray
) -> LeapHorizontalBar:
    """Softmax of the raw logits as a labelled bar chart, with the one-hot GT
    overlaid. Receives a single unbatched sample on the platform."""
    probs = _softmax(np.asarray(predictions, dtype=np.float64).reshape(-1)).astype(np.float32)
    gt_vec = np.asarray(gt, dtype=np.float32).reshape(-1)
    return LeapHorizontalBar(body=probs, labels=LABELS, gt=gt_vec)


@tensorleap_custom_visualizer("input_tokens", LeapDataType.Text)
def input_tokens_visualizer(tokens: np.ndarray) -> LeapText:
    """Render the 40-token model input as readable text: the before-window, a
    ``[ENTITY]`` marker where the (masked-out) entity sat, then the after-window.
    SentencePiece pieces are detokenized back to words and pad slots dropped."""
    ids = np.asarray(tokens).reshape(-1).astype(np.int64).tolist()
    before = _SP.decode([i for i in ids[:WINDOW] if i != PAD_ID]).split()
    after = _SP.decode([i for i in ids[WINDOW:] if i != PAD_ID]).split()
    return LeapText(data=before + ["[ENTITY]"] + after)


# --- metric -----------------------------------------------------------------
@tensorleap_custom_metric("accuracy", direction=MetricDirection.Upward)
def accuracy_metric(predictions: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-sample top-1 accuracy (1.0 if predicted class == GT class, else 0.0).

    Operates on batched arrays ((B, num_classes)) and returns a batch-aligned 1D
    array. Higher is better, hence MetricDirection.Upward.
    """
    correct = predictions.argmax(axis=-1) == gt.argmax(axis=-1)
    return correct.astype(np.float32)


# --- metadata ---------------------------------------------------------------
@tensorleap_metadata(
    "metadata",
    {
        "label": DatasetMetadataType.string,
        "entity": DatasetMetadataType.string,
        "char_len": DatasetMetadataType.int,
        "context_tokens": DatasetMetadataType.int,
    },
)
def sample_metadata(sample_id: str, preprocess: PreprocessResponse) -> dict:
    """Per-sample scalars for slicing/analysis on the platform: the GT class, the
    marked entity span, sentence length, and how many non-pad tokens of context
    the 40-token window actually carries."""
    record = preprocess.data["records"][sample_id]
    text = record["text"]
    _before, entity, _after = split_entity(text)
    ids = np.asarray(encode_window(text, _SP))
    return {
        "label": record["label"],
        "entity": normalize(entity)[:64],
        "char_len": int(len(text)),
        "context_tokens": int(np.count_nonzero(ids != PAD_ID)),
    }


# --- integration test -------------------------------------------------------
@tensorleap_integration_test()
def integration_test(sample_id: str, preprocess: PreprocessResponse):
    """Thin smoke test: encode -> GT -> metadata -> load -> infer -> loss. Only
    decorator calls and the minimal ONNX inference live here; all postprocessing
    (softmax, argmax, decoding) belongs inside the decorated functions."""
    x = tokens_input(sample_id, preprocess)
    gt = assertion_gt(sample_id, preprocess)
    sample_metadata(sample_id, preprocess)
    model = load_model()
    input_name = model.get_inputs()[0].name
    predictions = model.run(None, {input_name: x})[0]
    cross_entropy_loss(predictions, gt)
    accuracy_metric(predictions, gt)
    class_probabilities_visualizer(predictions, gt)
    input_tokens_visualizer(x)


if __name__ == "__main__":
    subsets = preprocess()
    for subset in subsets:
        if subset.state not in {DataStateType.training, DataStateType.validation}:
            continue
        for sample_id in subset.sample_ids[:3]:
            integration_test(sample_id, subset)
    print("integration_test passed for first 3 training + 3 validation samples")
