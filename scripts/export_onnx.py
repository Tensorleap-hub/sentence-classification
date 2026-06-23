"""Export the trained checkpoint to ONNX and verify it.

The exported graph accepts a float32 input of shape (batch, 40) and casts to
int64 internally before the embedding. This lets callers feed a float32 array
directly and have the model cast to int64 internally, without relying on runtime
dtype coercion. Output is raw logits (batch, 3) with a
dynamic batch axis.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentence_clf.model import SentenceClassifier  # noqa: E402
from sentence_clf.text import LABEL_TO_INDEX, encode_window  # noqa: E402


class FloatInputWrapper(torch.nn.Module):
    """Accept float32 token ids and cast to int64 for the embedding."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tokens):  # tokens: (B, L) float32
        return self.model(tokens.to(torch.int64))


def main() -> int:
    config = json.loads((ROOT / "config.json").read_text())

    model = SentenceClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        input_len=config["input_len"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        pad_id=config["pad_id"],
    )
    model.load_state_dict(torch.load(ROOT / "model" / "model.pt", map_location="cpu"))
    model.eval()

    wrapper = FloatInputWrapper(model).eval()
    onnx_path = ROOT / config["model_path"]
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, config["input_len"], dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=[config["input_name"]],
        output_names=[config["output_name"]],
        dynamic_axes={
            config["input_name"]: {0: "batch"},
            config["output_name"]: {0: "batch"},
        },
        opset_version=17,
    )

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"ONNX input : name={inp.name} shape={inp.shape} dtype={inp.type}")
    print(f"ONNX output: name={out.name} shape={out.shape} dtype={out.type}")

    # Parity check on real samples: ONNX (float32 in) vs torch (int64 in).
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / config["tokenizer_path"]))
    val = pd.read_csv(ROOT / "data" / "processed" / "val.csv").head(8)
    X = np.array([encode_window(str(t), sp) for t in val["text"]], dtype=np.float32)

    onnx_logits = sess.run([out.name], {inp.name: X})[0]
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(X.astype(np.int64))).numpy()

    max_diff = float(np.abs(onnx_logits - torch_logits).max())
    onnx_pred = onnx_logits.argmax(1)
    agree = int((onnx_pred == torch_logits.argmax(1)).sum())
    truth = np.array([LABEL_TO_INDEX[str(l)] for l in val["label"]])
    print(f"parity: max|onnx-torch|={max_diff:.2e}  argmax_agreement={agree}/{len(val)}")
    print(f"sample preds (onnx) : {[config['labels'][i] for i in onnx_pred]}")
    print(f"sample truth        : {list(val['label'])}")
    print(f"\nwrote {onnx_path}")
    return 0 if max_diff < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
