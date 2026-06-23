"""Train the SentencePiece subword tokenizer on the training split.

The customer asked for a 40,000-token vocab, but a ~10k-sentence corpus cannot
support that many pieces, so we try 40k and fall back to the largest feasible
size, reporting what we used. The tokenizer is trained on normalized,
tag-stripped text (entity words kept) so it covers the full vocabulary.

Outputs:
- tokenizer/spm.model , tokenizer/spm.vocab
- config.json  (single source of truth shared with training + the integration)
"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentence_clf.text import (  # noqa: E402
    INPUT_LEN,
    LABELS,
    PAD_ID,
    WINDOW,
    normalize,
    strip_entity_tags,
)

VOCAB_CANDIDATES = [40000, 16000, 8000, 4000, 2000]
INPUT_NAME = "tokens"
OUTPUT_NAME = "logits"

OUT = ROOT / "tokenizer"


def main() -> int:
    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    OUT.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for text in train["text"]:
            line = normalize(strip_entity_tags(str(text)))
            if line:
                f.write(line + "\n")
        corpus_path = f.name

    chosen = None
    for vocab in VOCAB_CANDIDATES:
        try:
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=str(OUT / "spm"),
                vocab_size=vocab,
                model_type="unigram",
                pad_id=PAD_ID,
                unk_id=1,
                bos_id=-1,
                eos_id=-1,
                character_coverage=1.0,
            )
            chosen = vocab
            print(f"trained tokenizer with vocab_size={vocab}")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"vocab_size={vocab} failed: {str(exc).splitlines()[0][:160]}")

    if chosen is None:
        print("tokenizer training failed for all candidate sizes", file=sys.stderr)
        return 1

    sp = spm.SentencePieceProcessor(model_file=str(OUT / "spm.model"))
    actual_vocab = sp.get_piece_size()

    config = {
        "labels": LABELS,
        "window": WINDOW,
        "input_len": INPUT_LEN,
        "pad_id": PAD_ID,
        "requested_vocab": 40000,
        "vocab_size": actual_vocab,
        "tokenizer_path": "tokenizer/spm.model",
        "model_path": "model/model.onnx",
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "embedding_dim": 32,
        "hidden_dim": 64,
        "num_classes": len(LABELS),
    }
    (ROOT / "config.json").write_text(json.dumps(config, indent=2))
    print(f"actual vocab size: {actual_vocab}")
    print(f"wrote config.json and tokenizer/spm.model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
