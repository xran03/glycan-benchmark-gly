from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from benchmark.evaluate import compute_metrics
from benchmark.models.glycanaa_graph import build_glycanaa_graph, build_mono_vocab, set_mono_vocab
from benchmark.models.glycanaa_model import GlycanAAModel

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GlycanAATrainer:
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        seed: int = 42,
        concat_hidden: bool = True,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.concat_hidden = concat_hidden

        self.mono_vocab: dict[str, int] | None = None
        self.model: GlycanAAModel | None = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    @staticmethod
    def _collate(data_list):
        return Batch.from_data_list(data_list)

    def prepare_vocab(self, glycans: list[str]) -> dict[str, int]:
        self.mono_vocab = build_mono_vocab(glycans)
        set_mono_vocab(self.mono_vocab)
        return self.mono_vocab

    def _ensure_model(self):
        if self.mono_vocab is None:
            raise RuntimeError("Call prepare_vocab() before training.")
        if self.model is None:
            self.model = GlycanAAModel(
                vocab_size=len(self.mono_vocab),
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                concat_hidden=self.concat_hidden,
            ).to(DEVICE)

    def _build_dataset(self, df: pd.DataFrame, split_name: str):
        graphs = []
        skipped = 0
        iterator = tqdm(df.itertuples(index=False), total=len(df), desc=f"graphs:{split_name}", leave=False)
        for row in iterator:
            data = build_glycanaa_graph(row.glycan)
            if data is None:
                skipped += 1
                continue
            data.y = torch.tensor([float(row.label)], dtype=torch.float32)
            graphs.append(data)
        if skipped:
            logger.warning("Skipped %d/%d glycans for split=%s", skipped, len(df), split_name)
        if not graphs:
            raise RuntimeError(f"No valid GlycanAA graphs built for split={split_name}")
        return graphs

    def _loader(self, graphs, shuffle: bool):
        return DataLoader(
            graphs,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self._collate,
        )

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, list[float]]:
        self._ensure_model()
        train_graphs = self._build_dataset(train_df, "train")
        valid_graphs = self._build_dataset(valid_df, "valid")

        pos = float(train_df["label"].sum())
        neg = float(len(train_df) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        best_val = float("inf")
        best_state = None
        wait = 0
        self.history = {"train_loss": [], "val_loss": []}

        train_loader = self._loader(train_graphs, shuffle=True)
        valid_loader = self._loader(valid_graphs, shuffle=False)

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            train_loss = 0.0
            train_count = 0
            for batch in train_loader:
                batch = batch.to(DEVICE)
                logits = self.model(batch)
                y = batch.y.view(-1).to(DEVICE)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * y.numel()
                train_count += y.numel()
            train_loss /= max(train_count, 1)
            scheduler.step()

            self.model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in valid_loader:
                    batch = batch.to(DEVICE)
                    logits = self.model(batch)
                    y = batch.y.view(-1).to(DEVICE)
                    loss = criterion(logits, y)
                    val_loss += loss.item() * y.numel()
                    val_count += y.numel()
            val_loss /= max(val_count, 1)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch == 1 or epoch % 5 == 0:
                logger.info(
                    "Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e",
                    epoch,
                    self.max_epochs,
                    train_loss,
                    val_loss,
                    scheduler.get_last_lr()[0],
                )

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("Early stopping at epoch %d (best val=%.4f)", epoch, best_val)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        logger.info("Training complete. Best val_loss=%.4f", best_val)
        return self.history

    def predict_proba(self, glycans: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        graphs = []
        for glycan in glycans:
            data = build_glycanaa_graph(glycan)
            if data is None:
                continue
            data.y = torch.tensor([0.0], dtype=torch.float32)
            graphs.append(data)
        if not graphs:
            return np.array([], dtype=np.float32)
        loader = self._loader(graphs, shuffle=False)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                preds.extend(torch.sigmoid(self.model(batch)).cpu().numpy())
        return np.asarray(preds, dtype=np.float32)

    def evaluate(self, test_df: pd.DataFrame):
        test_graphs = self._build_dataset(test_df, "test")
        loader = self._loader(test_graphs, shuffle=False)
        y_true, y_prob = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                y_true.extend(batch.y.view(-1).cpu().numpy())
                y_prob.extend(torch.sigmoid(self.model(batch)).cpu().numpy())
        y_true_arr = np.asarray(y_true, dtype=np.int64)
        y_prob_arr = np.asarray(y_prob, dtype=np.float32)
        return compute_metrics(y_true_arr, y_prob_arr), y_true_arr, y_prob_arr

    def save(self, path: Path | str) -> None:
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "mono_vocab": self.mono_vocab,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "concat_hidden": self.concat_hidden,
        }, path)
