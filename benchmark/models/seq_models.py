"""
Sequence-based models for glycan immunogenicity.

Glycan IUPAC strings are tokenised into a sequence of glycoword indices
(same vocabulary used for the graph models). Each model processes the padded
token sequence and produces a binary classification logit.

  CNNClassifier     — 1-D Conv (shallow CNN, 3 layers)
  ResNetClassifier  — Residual 1-D Conv blocks
  LSTMClassifier    — Bidirectional LSTM
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Tokeniser  ───────────────────────────────────────────────────────────────
class GlycanTokenizer:
    """
    Convert IUPAC-condensed glycans to padded integer sequences.

    Uses the same node-label vocabulary produced by ImmunogenicityDataset.vocab
    (monosaccharide + linkage tokens).  Linkage tokens (e.g. 'a2-3') carry
    positional information; monosaccharide tokens carry structural identity.
    Both are included so the sequence preserves alternating structure.
    """

    PAD = 0

    def __init__(self, vocab: dict[str, int]):
        # offset by 1 so index 0 is reserved for PAD
        self.vocab = {tok: idx + 1 for tok, idx in vocab.items()}
        self.vocab_size = len(self.vocab) + 1   # +1 for PAD

    def encode(self, glycans: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            ids  : (B, L) long tensor  — padded token sequences
            mask : (B, L) bool tensor  — True = valid (non-pad) position
        """
        from glycowork.motif.graph import glycan_to_nxGraph
        seqs = []
        for g in glycans:
            try:
                nx = glycan_to_nxGraph(g, libr=None)
                tokens = [data["string_labels"] for _, data in nx.nodes(data=True)]
                seqs.append([self.vocab.get(t, self.PAD) for t in tokens])
            except Exception:
                seqs.append([self.PAD])

        max_len = max(len(s) for s in seqs)
        ids  = torch.zeros(len(seqs), max_len, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = True
        return ids, mask


# ── CNN  ─────────────────────────────────────────────────────────────────────
class CNNModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden: int = 256, kernel_size: int = 5, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.Sequential(
            nn.Conv1d(embed_dim, hidden, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids).transpose(1, 2)          # (B, E, L)
        x = self.convs(x)                             # (B, H, L)
        x = (x * mask.unsqueeze(1).float()).max(-1)[0]  # global max pool
        return self.head(x).squeeze(-1)


# ── ResNet block ─────────────────────────────────────────────────────────────
class _ResBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.LayerNorm(dim)   # applied on (B, L, D) after transpose
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L)
        residual = x
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)   # LayerNorm on D
        h = F.gelu(self.conv1(h))
        h = self.conv2(h)
        return residual + h


class ResNetModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 num_blocks: int = 4, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj  = nn.Conv1d(embed_dim, embed_dim, 1)
        self.blocks = nn.Sequential(*[_ResBlock(embed_dim) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids).transpose(1, 2)    # (B, E, L)
        x = self.proj(x)
        x = self.blocks(x)
        x = (x * mask.unsqueeze(1).float()).max(-1)[0]
        return self.head(x).squeeze(-1)


# ── LSTM ─────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden, num_layers=num_layers,
                             bidirectional=True, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.head  = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(-1).clamp(min=1).cpu()
        x = self.embed(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, (h, _) = self.lstm(packed)
        # Concatenate last forward + last backward hidden
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(h).squeeze(-1)


# ── Registry ─────────────────────────────────────────────────────────────────
SEQ_REGISTRY: dict[str, type] = {
    "CNN":    CNNModel,
    "ResNet": ResNetModel,
    "LSTM":   LSTMModel,
}
