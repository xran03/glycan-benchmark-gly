"""
SweetNet-based binary classifier for glycan immunogenicity.

Trains a full SweetNet GNN from scratch on the immunogenicity dataset.
No pretrained weights required — end-to-end training from random initialization.

Architecture: glycowork SweetNet (GraphConv × 3 → global pool → MLP → 1 output)
Loss        : BCEWithLogitsLoss with pos_weight for class imbalance
Optimizer   : Adam with cosine annealing LR schedule
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from glycowork.ml.models import prep_model
from glycowork.ml.processing import dataset_to_dataloader

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SweetNetClassifier:
    """
    End-to-end SweetNet binary classifier for glycan immunogenicity.

    Trains the full GNN from random init — no HuggingFace download needed.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        seed: int = 42,
        hidden_dim: int = 128,
    ):
        self.vocab = vocab
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.hidden_dim = hidden_dim

        torch.manual_seed(seed)
        np.random.seed(seed)

        self._model: nn.Module | None = None

    # ------------------------------------------------------------------ #

    def _make_loader(self, df: pd.DataFrame, shuffle: bool) -> torch.utils.data.DataLoader:
        """Build a glycowork DataLoader from a glycan/label DataFrame."""
        return dataset_to_dataloader(
            df["glycan"].tolist(),
            df["label"].tolist(),
            libr=self.vocab,
            batch_size=self.batch_size,
            label_type=torch.float,
            shuffle=shuffle,
        )

    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> dict[str, list[float]]:
        """
        Train SweetNet end-to-end for binary immunogenicity classification.

        Returns:
            history dict with 'train_loss' and 'val_loss' per epoch.
        """
        logger.info("Building SweetNet (num_classes=1, random init) on %s", DEVICE)
        self._model = prep_model(
            "SweetNet", num_classes=1, libr=self.vocab, trained=False, hidden_dim=self.hidden_dim
        ).to(DEVICE)

        # Class-imbalance weighting
        pos_count = train_df["label"].sum()
        neg_count = len(train_df) - pos_count
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        train_loader = self._make_loader(train_df, shuffle=True)
        valid_loader = self._make_loader(valid_df, shuffle=False)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val, wait, best_state = float("inf"), 0, None

        for epoch in range(1, self.max_epochs + 1):
            # ---- train ----
            self._model.train()
            train_loss = 0.0
            n_tr = 0
            for data in train_loader:
                x, y, edge_index, batch = (
                    data.labels.to(DEVICE),
                    data.y.to(DEVICE),
                    data.edge_index.to(DEVICE),
                    data.batch.to(DEVICE),
                )
                optimizer.zero_grad()
                pred = self._model(x, edge_index, batch)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y)
                n_tr += len(y)
            train_loss /= max(n_tr, 1)
            scheduler.step()

            # ---- validate ----
            self._model.eval()
            val_loss = 0.0
            n_va = 0
            with torch.no_grad():
                for data in valid_loader:
                    x, y, edge_index, batch = (
                        data.labels.to(DEVICE),
                        data.y.to(DEVICE),
                        data.edge_index.to(DEVICE),
                        data.batch.to(DEVICE),
                    )
                    pred = self._model(x, edge_index, batch)
                    val_loss += criterion(pred, y).item() * len(y)
                    n_va += len(y)
            val_loss /= max(n_va, 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e",
                    epoch, self.max_epochs, train_loss, val_loss,
                    scheduler.get_last_lr()[0],
                )

            # Early stopping
            if val_loss < best_val - 1e-4:
                best_val, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("Early stopping at epoch %d (best val=%.4f)", epoch, best_val)
                    break

        if best_state:
            self._model.load_state_dict(best_state)
        logger.info("Training complete. Best val_loss=%.4f", best_val)
        return history

    def predict_proba(self, glycans: list[str]) -> np.ndarray:
        """Return predicted positive-class probabilities (N,)."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba()")
        loader = dataset_to_dataloader(
            glycans, [0.0] * len(glycans),
            libr=self.vocab, batch_size=self.batch_size,
            label_type=torch.float, shuffle=False,
        )
        self._model.eval()
        preds = []
        with torch.no_grad():
            for data in loader:
                x, _, edge_index, batch = (
                    data.labels.to(DEVICE),
                    data.y,
                    data.edge_index.to(DEVICE),
                    data.batch.to(DEVICE),
                )
                logits = self._model(x, edge_index, batch)
                preds.extend(torch.sigmoid(logits).cpu().numpy())
        return np.array(preds)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        logger.info("Model weights saved to %s", path)

    def load(self, path: Path | str) -> None:
        self._model = prep_model(
            "SweetNet", num_classes=1, libr=self.vocab, trained=False, hidden_dim=self.hidden_dim
        ).to(DEVICE)
        self._model.load_state_dict(torch.load(path, map_location=DEVICE))
        logger.info("Model weights loaded from %s", path)
