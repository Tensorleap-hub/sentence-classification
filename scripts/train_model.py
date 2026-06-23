"""Train the sentence_clf classifier and save a PyTorch checkpoint.

Encodes every row to a 40-token window via sentence_clf.text.encode_window (the exact
same encoding path used everywhere), trains a small FNN, and reports
train/val accuracy. Deterministic (seed 42).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentence_clf.model import SentenceClassifier  # noqa: E402
from sentence_clf.text import LABEL_TO_INDEX, encode_window  # noqa: E402

SEED = 42
EPOCHS = 12
BATCH = 64
LR = 1e-3


def encode_split(csv_path: Path, sp) -> TensorDataset:
    df = pd.read_csv(csv_path)
    X = np.array([encode_window(str(t), sp) for t in df["text"]], dtype=np.int64)
    y = np.array([LABEL_TO_INDEX[str(l)] for l in df["label"]], dtype=np.int64)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


@torch.no_grad()
def evaluate(model, loader, num_classes) -> tuple:
    model.eval()
    correct = total = 0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    for xb, yb in loader:
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.numel()
        for c in range(num_classes):
            mask = yb == c
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += (preds[mask] == c).sum().item()
    acc = correct / max(total, 1)
    per_class = (per_class_correct / np.maximum(per_class_total, 1)).round(3)
    return acc, per_class


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config = json.loads((ROOT / "config.json").read_text())
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / config["tokenizer_path"]))

    train_ds = encode_split(ROOT / "data" / "processed" / "train.csv", sp)
    val_ds = encode_split(ROOT / "data" / "processed" / "val.csv", sp)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = SentenceClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        input_len=config["input_len"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        pad_id=config["pad_id"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * yb.numel()
        train_loss = running / len(train_ds)
        val_acc, per_class = evaluate(model, val_loader, config["num_classes"])
        print(f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  "
              f"per_class(={config['labels']})={per_class.tolist()}")

    out = ROOT / "model"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    print(f"\nsaved checkpoint to {out / 'model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
