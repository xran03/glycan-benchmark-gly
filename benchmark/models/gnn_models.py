"""
GNN-based classifiers for glycan immunogenicity.

All models use glycowork's graph loader (same format as SweetNet) but swap
the graph convolution layer:

  GCNClassifier   — GCNConv (Kipf & Welling 2017)
  GATClassifier   — GATConv (Veličković et al. 2018)
  GINClassifier   — GINConv (Xu et al. 2019)
  MPNNClassifier  — NNConv + GRU update (Gilmer et al. 2017)

Shared training loop lives in base_trainer.py.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, NNConv,
    global_mean_pool, global_max_pool,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── small helper: dual readout (mean + max) ─────────────────────────────────
def _dual_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return torch.cat([global_mean_pool(x, batch),
                      global_max_pool(x, batch)], dim=-1)


# ── GCN ─────────────────────────────────────────────────────────────────────
class GCNModel(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 128, layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, hidden)
        self.convs = nn.ModuleList(
            [GCNConv(hidden, hidden) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, batch):
        h = self.embed(x).squeeze(1)
        for conv in self.convs:
            h = F.leaky_relu(conv(h, edge_index))
        return self.head(_dual_pool(h, batch)).squeeze(-1)


# ── GAT ─────────────────────────────────────────────────────────────────────
class GATModel(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 128, layers: int = 3,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, hidden)
        self.convs = nn.ModuleList()
        for i in range(layers):
            in_ch  = hidden if i == 0 else hidden * heads
            out_ch = hidden
            concat = (i < layers - 1)
            self.convs.append(
                GATConv(in_ch, out_ch, heads=heads if concat else 1,
                        concat=concat, dropout=dropout)
            )
        # Last GAT layer always uses heads=1, concat=False → output dim = hidden
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, batch):
        h = self.embed(x).squeeze(1)
        for conv in self.convs:
            h = F.elu(conv(h, edge_index))
        return self.head(_dual_pool(h, batch)).squeeze(-1)


# ── GIN ─────────────────────────────────────────────────────────────────────
class GINModel(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 128, layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, hidden)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.ReLU(),
                nn.Linear(hidden * 2, hidden),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, batch):
        h = self.embed(x).squeeze(1)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return self.head(_dual_pool(h, batch)).squeeze(-1)


# ── MPNN ────────────────────────────────────────────────────────────────────
class MPNNModel(nn.Module):
    """
    Message-Passing NN with NNConv (edge-conditioned) + GRU node update.
    Edge features are dummy (all-ones) since glycowork edges have no features.
    """
    def __init__(self, vocab_size: int, hidden: int = 128, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, hidden)
        self.edge_nn = nn.Sequential(
            nn.Linear(1, hidden * hidden), nn.ReLU()
        )
        self.conv = NNConv(hidden, hidden, self.edge_nn, aggr="mean")
        self.gru  = nn.GRU(hidden, hidden, batch_first=False)
        self.num_layers = num_layers
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, batch):
        h = self.embed(x).squeeze(1)
        # Dummy edge features: all ones
        edge_attr = torch.ones(edge_index.size(1), 1, device=x.device)
        hx = h.unsqueeze(0)
        for _ in range(self.num_layers):
            m   = F.relu(self.conv(h, edge_index, edge_attr))
            h, hx = self.gru(m.unsqueeze(0), hx)
            h = h.squeeze(0)
        return self.head(_dual_pool(h, batch)).squeeze(-1)


# ── Registry ─────────────────────────────────────────────────────────────────
GNN_REGISTRY: dict[str, type] = {
    "GCN":  GCNModel,
    "GAT":  GATModel,
    "GIN":  GINModel,
    "MPNN": MPNNModel,
}
