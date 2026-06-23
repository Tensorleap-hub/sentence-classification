"""Parse the raw Kaggle CSV, drop malformed rows, and write a stratified split.

Outputs:
- data/processed/train.csv  (text,label  -- raw text with entity tags kept)
- data/processed/val.csv
- data/processed/labels.json (the fixed class order)

The raw entity tags are kept in the split files so encoding stays reproducible
from the canonical raw text via sentence_clf.text.encode_window.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentence_clf.text import LABELS, split_entity  # noqa: E402

RAW = ROOT / "data" / "raw" / "final_combined_classificationdataset.csv"
OUT = ROOT / "data" / "processed"
VAL_FRACTION = 0.2
SEED = 42


def main() -> int:
    df = pd.read_csv(RAW)
    print(f"loaded {len(df)} rows, columns={list(df.columns)}")

    # Keep only rows with a well-formed entity span and a known label.
    def usable(row) -> bool:
        try:
            split_entity(str(row["text"]))
        except ValueError:
            return False
        return str(row["label"]) in set(LABELS)

    before = len(df)
    df = df[df.apply(usable, axis=1)].reset_index(drop=True)
    print(f"kept {len(df)} / {before} rows after dropping malformed/unknown-label")
    print("label counts:\n", df["label"].value_counts().to_string())

    # Stratified split per label so rare classes (Hypothetical) appear in both.
    train_parts, val_parts = [], []
    for label, group in df.groupby("label"):
        group = group.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n_val = max(1, int(round(len(group) * VAL_FRACTION)))
        val_parts.append(group.iloc[:n_val])
        train_parts.append(group.iloc[n_val:])

    train = pd.concat(train_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    val.to_csv(OUT / "val.csv", index=False)
    (OUT / "labels.json").write_text(json.dumps(LABELS, indent=2))

    print(f"\nwrote {len(train)} train / {len(val)} val rows to {OUT}")
    print("train label counts:\n", train["label"].value_counts().to_string())
    print("val label counts:\n", val["label"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
