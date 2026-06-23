"""The sentence_clf clinical-assertion classifier.

FNN over a fixed 40-token window (20 tokens before + 20 after the entity):

    Embedding(vocab, 32, padding_idx=0)
    Flatten            [B, 40, 32] -> [B, 1280]
    Linear(1280, 64) + ReLU
    Dropout
    Linear(64, 3)      -> raw logits (softmax is postprocessing, not in the model)

The model outputs raw logits; probability conversion (softmax) is postprocessing
that belongs downstream, not in the model.
"""

from __future__ import annotations

import torch.nn as nn


class SentenceClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        input_len: int = 40,
        hidden_dim: int = 64,
        num_classes: int = 3,
        pad_id: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.flatten = nn.Flatten()  # start_dim=1 -> keeps the batch axis
        self.fc1 = nn.Linear(input_len * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens):  # tokens: (B, input_len) int64
        embedded = self.embedding(tokens)     # (B, L, E)
        flat = self.flatten(embedded)         # (B, L*E)
        hidden = self.relu(self.fc1(flat))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)               # (B, num_classes) raw logits
