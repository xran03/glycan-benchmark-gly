"""
Shared training loop for all glycan immunogenicity models.

Supports two data modes:
  "graph"  — glycowork DataLoader (used by SweetNet, GCN, GAT, GIN, MPNN)
  "seq"    — padded token-index batches  (used by CNN, ResNet, LSTM)

Usage:
    trainer = Trainer(model, mode="graph", vocab=vocab, ...)
    history = trainer.fit(train_df, valid_df)
    metrics = trainer.evaluate(test_df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import CosineAnnealingLR

from glycowork.ml.processing import dataset_to_dataloader

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """
    Generic trainer that wraps any PyTorch model for binary glycan classification.

    Parameters
    ----------
    model       : nn.Module already on DEVICE
    mode        : "graph" or "seq"
    vocab       : token→index dict (required for both modes)
    lr          : Adam learning rate
    weight_decay: L2 regularisation
    max_epochs  : hard upper bound on epochs
    batch_size  : mini-batch size
    patience    : early-stopping patience (on val loss)
    """

    def __init__(
        self,
        model: nn.Module,
        mode: Literal["graph", "seq"],
        vocab: dict[str, int],
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model      = model.to(DEVICE)
        self.mode       = mode
        self.vocab      = vocab
        self.lr         = lr
        self.wd         = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience   = patience

        if mode == "seq":
            from benchmark.models.seq_models import GlycanTokenizer
            self.tokenizer = GlycanTokenizer(vocab)

    # ── data helpers ──────────────────────────────────────────────────────

    def _graph_loader(self, df: pd.DataFrame, shuffle: bool):
        return dataset_to_dataloader(
            df["glycan"].tolist(), df["label"].tolist(),
            libr=self.vocab, batch_size=self.batch_size,
            label_type=torch.float, shuffle=shuffle,
        )

    def _seq_batch_iter(self, df: pd.DataFrame, shuffle: bool):
        """Yield (ids, mask, y) batches for sequence models."""
        glycans = df["glycan"].tolist()
        labels  = df["label"].tolist()
        idx     = np.arange(len(glycans))
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch_idx = idx[start: start + self.batch_size]
            g_batch = [glycans[i] for i in batch_idx]
            y_batch = torch.tensor([labels[i] for i in batch_idx], dtype=torch.float32)
            ids, mask = self.tokenizer.encode(g_batch)
            yield ids.to(DEVICE), mask.to(DEVICE), y_batch.to(DEVICE)

    def _forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass; return (logits, y)."""
        if self.mode == "graph":
            x, y, ei, batch = (data.labels.to(DEVICE), data.y.to(DEVICE),
                               data.edge_index.to(DEVICE), data.batch.to(DEVICE))
            return self.model(x, ei, batch), y
        else:
            ids, mask, y = data
            return self.model(ids, mask), y

    # ── training ──────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> dict[str, list[float]]:

        pos = train_df["label"].sum()
        neg = len(train_df) - pos
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(DEVICE)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        if self.mode == "graph":
            train_iter = lambda: self._graph_loader(train_df, shuffle=True)
            valid_iter = lambda: self._graph_loader(valid_df, shuffle=False)
        else:
            train_iter = lambda: self._seq_batch_iter(train_df, shuffle=True)
            valid_iter = lambda: self._seq_batch_iter(valid_df, shuffle=False)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val, wait, best_state = float("inf"), 0, None

        for epoch in range(1, self.max_epochs + 1):
            # train
            self.model.train()
            t_loss, n_t = 0.0, 0
            for data in train_iter():
                logits, y = self._forward(data)
                loss = criterion(logits, y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                t_loss += loss.item() * len(y); n_t += len(y)
            t_loss /= max(n_t, 1)
            scheduler.step()

            # validate
            self.model.eval()
            v_loss, n_v = 0.0, 0
            with torch.no_grad():
                for data in valid_iter():
                    logits, y = self._forward(data)
                    v_loss += criterion(logits, y).item() * len(y); n_v += len(y)
            v_loss /= max(n_v, 1)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)

            if epoch % 5 == 0 or epoch == 1:
                logger.info("Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e",
                            epoch, self.max_epochs, t_loss, v_loss,
                            scheduler.get_last_lr()[0])

            if v_loss < best_val - 1e-4:
                best_val, wait = v_loss, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("Early stopping at epoch %d (best val=%.4f)", epoch, best_val)
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        logger.info("Training complete. Best val_loss=%.4f", best_val)
        return history

    # ── inference ─────────────────────────────────────────────────────────

    def predict_proba(self, glycans: list[str]) -> np.ndarray:
        preds = []
        self.model.eval()
        with torch.no_grad():
            if self.mode == "graph":
                dummy_df = pd.DataFrame({"glycan": glycans, "label": [0] * len(glycans)})
                for data in self._graph_loader(dummy_df, shuffle=False):
                    x, _, ei, batch = (data.labels.to(DEVICE), data.y,
                                       data.edge_index.to(DEVICE), data.batch.to(DEVICE))
                    logits = self.model(x, ei, batch)
                    preds.extend(torch.sigmoid(logits).cpu().numpy())
            else:
                for start in range(0, len(glycans), self.batch_size):
                    batch = glycans[start: start + self.batch_size]
                    ids, mask = self.tokenizer.encode(batch)
                    logits = self.model(ids.to(DEVICE), mask.to(DEVICE))
                    preds.extend(torch.sigmoid(logits).cpu().numpy())
        return np.array(preds)

    def evaluate(
        self,
        test_df: pd.DataFrame,
    ) -> dict[str, float]:
        from benchmark.evaluate import compute_metrics
        y_prob = self.predict_proba(test_df["glycan"].tolist())
        y_true = test_df["label"].values
        return compute_metrics(y_true, y_prob), y_true, y_prob

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
