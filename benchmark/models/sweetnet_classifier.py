"""
SweetNet-based binary classifier for glycan immunogenicity.

Uses glycowork's SweetNet GNN (pretrained on 1075 species classes) as a
frozen feature extractor, then fine-tunes a small MLP head for binary
immunogenicity classification.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from glycowork.ml.models import prep_model
from glycowork.ml.inference import glycans_to_emb
from glycowork.glycan_data.loader import lib

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SweetNetClassifier:
    """
    Glycan immunogenicity classifier backed by a pretrained SweetNet encoder.

    Steps:
      1. Encode glycans → 128-d embeddings (SweetNet, frozen)
      2. Fine-tune a small MLP head on the immunogenicity labels
      3. Predict probabilities for evaluation
    """

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        seed: int = 42,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self._encoder: nn.Module | None = None
        self._head: _MLPHead | None = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_encoder(self):
        if self._encoder is None:
            logger.info("Loading pretrained SweetNet encoder …")
            self._encoder = prep_model("SweetNet", num_classes=1075, trained=True)
            self._encoder.eval()
            for p in self._encoder.parameters():
                p.requires_grad_(False)
            self._encoder = self._encoder.to(DEVICE)

    def _encode(self, glycans: list[str]) -> torch.Tensor:
        """Return (N, 128) float tensor of frozen SweetNet embeddings."""
        self._load_encoder()
        emb_df: pd.DataFrame = glycans_to_emb(
            glycans, self._encoder, libr=lib, batch_size=self.batch_size, rep=True
        )
        return torch.tensor(emb_df.values, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> dict[str, list[float]]:
        """
        Fine-tune the MLP head.

        Args:
            train_df: DataFrame with columns ['glycan', 'label']
            valid_df: DataFrame with columns ['glycan', 'label']

        Returns:
            history dict with 'train_loss' and 'val_loss' per epoch.
        """
        X_tr = self._encode(train_df["glycan"].tolist())
        y_tr = torch.tensor(train_df["label"].values, dtype=torch.float32)
        X_va = self._encode(valid_df["glycan"].tolist())
        y_va = torch.tensor(valid_df["label"].values, dtype=torch.float32)

        self._head = _MLPHead(in_dim=X_tr.shape[1]).to(DEVICE)
        optimizer = torch.optim.Adam(
            self._head.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Class-imbalance weighting
        pos_weight = torch.tensor(
            [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32
        ).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val, wait, best_state = float("inf"), 0, None

        for epoch in range(1, self.max_epochs + 1):
            # ----- train -----
            self._head.train()
            train_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self._head(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_tr)

            # ----- validate -----
            self._head.eval()
            with torch.no_grad():
                val_loss = criterion(
                    self._head(X_va.to(DEVICE)), y_va.to(DEVICE)
                ).item()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch, self.max_epochs, train_loss, val_loss,
                )

            # Early stopping
            if val_loss < best_val - 1e-4:
                best_val, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in self._head.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state:
            self._head.load_state_dict(best_state)
        logger.info("Training complete. Best val_loss=%.4f", best_val)
        return history

    def predict_proba(self, glycans: list[str]) -> np.ndarray:
        """Return predicted positive-class probabilities (N,)."""
        if self._head is None:
            raise RuntimeError("Call fit() before predict_proba()")
        X = self._encode(glycans).to(DEVICE)
        self._head.eval()
        with torch.no_grad():
            logits = self._head(X)
        return torch.sigmoid(logits).cpu().numpy()

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._head.state_dict(), path)
        logger.info("Head weights saved to %s", path)

    def load(self, path: Path | str, in_dim: int = 128) -> None:
        self._head = _MLPHead(in_dim=in_dim).to(DEVICE)
        self._head.load_state_dict(torch.load(path, map_location=DEVICE))
        logger.info("Head weights loaded from %s", path)
