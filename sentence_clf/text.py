"""Shared text processing for the sentence_clf clinical-assertion task.

This module is the single source of truth for how a raw labelled sentence is
turned into the 40-token model input. The training pipeline and any downstream
consumer import from here so the encoding is identical at train and inference
time.

Recipe (from the customer spec):
- Each sentence marks the target entity inline: ``... [ENTITY]balance issues[/ENTITY] ...``.
- Take a window of 20 tokens BEFORE the entity and 20 tokens AFTER it (40 total).
  The entity text itself is masked out (it is the thing being classified), so it
  is excluded from the 40 tokens.
- Per-side preprocessing: deaccent -> lowercase -> whitespace strip.
- Tokenize each side with the trained SentencePiece model, keep the 20 tokens
  nearest the entity, and pad the short side with id 0.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

ENTITY_OPEN = "[ENTITY]"
ENTITY_CLOSE = "[/ENTITY]"

WINDOW = 20          # tokens kept on each side of the entity
INPUT_LEN = 40       # WINDOW before + WINDOW after
PAD_ID = 0           # SentencePiece is trained with pad_id=0

# Fixed, deterministic class order. The model's 3 output logits follow this
# order, and downstream consumers use the same order.
LABELS: List[str] = ["Absent", "Hypothetical", "Present"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}

_WHITESPACE_RE = re.compile(r"\s+")


def deaccent(text: str) -> str:
    """café -> cafe. Strip combining marks via NFKD decomposition."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """deaccent -> lowercase -> collapse/strip whitespace."""
    return _WHITESPACE_RE.sub(" ", deaccent(text).lower()).strip()


# The corpus uses two entity-marker conventions interchangeably:
#   Format A: [ENTITY]chest pain[/ENTITY]   (distinct open/close, any case)
#   Format B: [entity] chest pain [entity]  (same marker repeated, any case)
_OPEN_B_RE = re.compile(r"\[entity\]", re.IGNORECASE)      # matches [entity], not [/entity]
_ANY_TAG_RE = re.compile(r"\[/?entity\]", re.IGNORECASE)   # any of the four marker forms


def split_entity(text: str) -> Tuple[str, str, str]:
    """Split a tagged sentence into (before, entity, after) raw substrings.

    Handles both marker formats. Raises ValueError if no usable entity span is
    found, so callers can drop unusable rows explicitly rather than silently
    mis-encoding them.
    """
    # Format A: distinct open/close tags (match case-insensitively).
    open_m = re.search(r"\[entity\]", text, re.IGNORECASE)
    close_m = re.search(r"\[/entity\]", text, re.IGNORECASE)
    if open_m and close_m and close_m.start() > open_m.start():
        return text[:open_m.start()], text[open_m.end():close_m.start()], text[close_m.end():]

    # Format B: the same [entity] marker repeated twice.
    markers = list(_OPEN_B_RE.finditer(text))
    if len(markers) >= 2:
        a, b = markers[0], markers[1]
        return text[:a.start()], text[a.end():b.start()], text[b.end():]

    raise ValueError("no usable [ENTITY] span found")


def strip_entity_tags(text: str) -> str:
    """Remove only the tag markers (any format), keep the entity words. Used to
    build the tokenizer training corpus."""
    return _ANY_TAG_RE.sub(" ", text)


def _pad_left(ids: List[int], width: int) -> List[int]:
    ids = ids[-width:]
    return [PAD_ID] * (width - len(ids)) + ids


def _pad_right(ids: List[int], width: int) -> List[int]:
    ids = ids[:width]
    return ids + [PAD_ID] * (width - len(ids))


def encode_window(text: str, sp) -> List[int]:
    """Turn one raw tagged sentence into a list of INPUT_LEN token ids.

    `sp` is a sentencepiece.SentencePieceProcessor. The 20 tokens immediately
    before the entity are kept (left-padded) and the 20 immediately after are
    kept (right-padded), then concatenated.
    """
    before, _entity, after = split_entity(text)
    before_ids = sp.encode(normalize(before), out_type=int)
    after_ids = sp.encode(normalize(after), out_type=int)
    return _pad_left(before_ids, WINDOW) + _pad_right(after_ids, WINDOW)
