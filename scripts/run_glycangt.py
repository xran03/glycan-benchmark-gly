#!/usr/bin/env python3
"""
GlycanGT supervised end-to-end benchmark on immunogenicity.

Uses GlycanGT TokenGT architecture trained from scratch (no pretrained weights).
Paper's GlycanGT uses pretrained weights + linear probing (HuggingFace blocked by
corporate firewall). This run tests the architecture's supervised capacity.

Run with DDP on GPU 0 and 2:
  CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 --master_port=29500 \
      scripts/run_glycangt.py --epochs 100 --seed 42
"""

import os, sys, json, warnings, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, f1_score, matthews_corrcoef,
                              confusion_matrix)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
GLYCANGT_DIR = REPO_ROOT / "GlycanGT"
sys.path.insert(0, str(GLYCANGT_DIR / "model"))
sys.path.insert(0, str(GLYCANGT_DIR / "tokenizer"))
sys.path.insert(0, str(REPO_ROOT))

from tokengt_graph_encoder import TokenGTGraphEncoder
from config_tokengt import get_config
from encode_glycan import iupac_to_graph_triples
from monomer_vocab import MonomerVocab
from linkage_vocab import LinkageVocab
from benchmark.data.immunogenicity import load_splits
from benchmark.evaluate import compute_metrics, rank_against_baselines, save_results


# ── dataset ────────────────────────────────────────────────────────────────────

def glycan_to_tensor_graph(iupac, mono_vocab, link_vocab):
    """Convert IUPAC string → (node_data, edge_data, edge_index) tensors."""
    triples = iupac_to_graph_triples(iupac, mono_vocab, link_vocab)
    if not triples:
        # Single-node glycan (no edges) — create a dummy self-loop
        node_data = torch.LongTensor([1])
        edge_data = torch.LongTensor([1])
        edge_index = torch.LongTensor([[0], [0]])
        return node_data, edge_data, edge_index

    nodes = {}
    for t in triples:
        nodes[t['in_node_id']]  = t['in_node_vocab_id']
        nodes[t['out_node_id']] = t['out_node_vocab_id']

    sorted_ids = sorted(nodes.keys())
    id2idx = {nid: i for i, nid in enumerate(sorted_ids)}
    node_data = torch.LongTensor([nodes[nid] for nid in sorted_ids])

    srcs = [id2idx[t['out_node_id']] for t in triples]
    dsts = [id2idx[t['in_node_id']]  for t in triples]
    edge_data  = torch.LongTensor([t['edge_vocab_id'] for t in triples])
    edge_index = torch.LongTensor([srcs, dsts])

    return node_data, edge_data, edge_index


class ImmunogenicityGTDataset(Dataset):
    def __init__(self, df, mono_vocab, link_vocab):
        self.labels  = df['label'].values.astype(np.float32)
        self.glycans = df['glycan'].tolist()
        self.mono_vocab = mono_vocab
        self.link_vocab = link_vocab
        # Pre-convert all graphs
        self.graphs = [
            glycan_to_tensor_graph(g, mono_vocab, link_vocab)
            for g in self.glycans
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        node_data, edge_data, edge_index = self.graphs[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return node_data, edge_data, edge_index, label


def collate_fn(batch):
    node_data_list, edge_data_list, edge_index_list, labels = zip(*batch)
    node_num = [n.shape[0] for n in node_data_list]
    edge_num = [e.shape[0] for e in edge_data_list]
    total_nodes = sum(node_num)

    # edge_index must stay as LOCAL per-graph indices (0-based within each graph).
    # The tokenizer's get_batch places each graph's edges into its own slot in
    # padded_index; no global offset should be applied here.
    collated = {
        "node_data":  torch.cat(node_data_list,  dim=0),
        "edge_data":  torch.cat(edge_data_list,  dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),   # local indices only
        "node_num":   node_num,
        "edge_num":   edge_num,
        # lap placeholders (not used; config has lap_node_id=False)
        "lap_eigvec": torch.zeros(total_nodes, 16),
        "lap_eigval": torch.zeros(total_nodes, 16),
    }
    labels = torch.stack(labels)
    return collated, labels


# ── model ──────────────────────────────────────────────────────────────────────

class GlycanGTClassifier(nn.Module):
    def __init__(self, cfg, num_atoms, num_edges):
        super().__init__()
        self.encoder = TokenGTGraphEncoder(
            num_atoms=num_atoms,
            num_edges=num_edges,
            rand_node_id=False,
            rand_node_id_dim=0,
            orf_node_id=True,
            orf_node_id_dim=cfg["orf_node_id_dim"],
            lap_node_id=False,
            lap_node_id_k=0,
            lap_node_id_sign_flip=False,
            lap_node_id_eig_dropout=0.0,
            type_id=True,
            num_encoder_layers=cfg["num_encoder_layers"],
            embedding_dim=cfg["embedding_dim"],
            ffn_embedding_dim=cfg["ffn_embedding_dim"],
            num_attention_heads=cfg["num_attention_heads"],
            dropout=cfg["dropout"],
            attention_dropout=cfg["attention_dropout"],
            activation_dropout=cfg["activation_dropout"],
            apply_graphormer_init=True,
            layernorm_style=cfg["layernorm_style"],
            activation_fn=cfg["activation_fn"],
        )
        dim = cfg["embedding_dim"]
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, batch):
        _, graph_rep, _ = self.encoder(batch, last_state_only=True)
        return self.classifier(graph_rep).squeeze(-1)


# ── training helpers ────────────────────────────────────────────────────────────

def move_batch(batch, device):
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def evaluate_epoch(model, loader, device, is_ddp=False):
    raw_model = model.module if is_ddp else model
    raw_model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch  = move_batch(batch, device)
            labels = labels.to(device)
            logits = raw_model(batch)
            probs  = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)
    return y_prob, y_true


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size",  default="large", choices=["ss","small","medium","large"])
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output-dir",  default="results/GlycanGT")
    args = parser.parse_args()

    # ── DDP init ──────────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = dist.get_world_size()
    device      = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main     = (local_rank == 0)

    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    if is_main:
        print(f"[DDP] world_size={world_size}, using GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES','all')}")

    # ── vocab & data ──────────────────────────────────────────────────────────
    VOCAB_DIR = GLYCANGT_DIR / "data" / "vocab_expanded"
    mono_vocab = MonomerVocab.load(VOCAB_DIR / "monomer.json")
    link_vocab = LinkageVocab.load(VOCAB_DIR / "linkage.json")

    if is_main:
        print(f"Vocab: {len(mono_vocab)} monomers, {len(link_vocab)} linkages")

    train_df, valid_df, test_df, _ = load_splits()

    train_ds = ImmunogenicityGTDataset(train_df, mono_vocab, link_vocab)
    valid_ds = ImmunogenicityGTDataset(valid_df, mono_vocab, link_vocab)
    test_ds  = ImmunogenicityGTDataset(test_df,  mono_vocab, link_vocab)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=local_rank, shuffle=True,
                                       seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)
    # Valid/test only on rank 0 for simplicity
    valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    # ── model ─────────────────────────────────────────────────────────────────
    cfg = get_config(args.model_size)
    model = GlycanGTClassifier(cfg, num_atoms=len(mono_vocab),
                               num_edges=len(link_vocab)).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"GlycanGT-{args.model_size}: {n_params:.2f}M params")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── training loop ─────────────────────────────────────────────────────────
    best_auroc   = 0.0
    best_epoch   = 0
    patience_cnt = 0
    history      = []
    best_state   = None

    output_dir = REPO_ROOT / args.output_dir
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"\nTraining GlycanGT-{args.model_size} for {args.epochs} epochs "
              f"(patience={args.patience})")

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch, labels in train_loader:
            batch  = move_batch(batch, device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Average loss across all ranks
        loss_tensor = torch.tensor(total_loss / max(n_batches, 1), device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

        # Evaluate on rank 0 only
        if is_main:
            y_prob, y_true = evaluate_epoch(model, valid_loader, device, is_ddp=True)
            auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
            auprc = average_precision_score(y_true, y_prob)
            history.append({"epoch": epoch, "train_loss": avg_loss,
                             "val_auroc": auroc, "val_auprc": auprc})

            if auroc > best_auroc:
                best_auroc   = auroc
                best_epoch   = epoch
                patience_cnt = 0
                best_state   = {k: v.cpu().clone()
                                for k, v in model.module.state_dict().items()}
            else:
                patience_cnt += 1

            if epoch % 10 == 0 or epoch <= 5:
                print(f"  Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} "
                      f"val_AUROC={auroc:.4f} val_AUPRC={auprc:.4f} "
                      f"[best={best_auroc:.4f}@{best_epoch}]")

            # Broadcast stop signal
            stop = torch.tensor(int(patience_cnt >= args.patience), device=device)
        else:
            stop = torch.tensor(0, device=device)

        dist.broadcast(stop, src=0)
        if stop.item():
            if is_main:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── final evaluation (rank 0 only) ────────────────────────────────────────
    if is_main:
        # Load best model
        model.module.load_state_dict(best_state)
        model.module.to(device)

        y_prob_test, y_true_test = evaluate_epoch(model, test_loader, device, is_ddp=True)
        metrics = compute_metrics(y_true_test, y_prob_test)
        ranking_df = rank_against_baselines(metrics, model_label="GlycanGT (supervised)")

        print(f"\n{'='*60}")
        print(f"GlycanGT-{args.model_size} (supervised, no pretraining) — Test Results")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        our_row = ranking_df[ranking_df["Source"] == "this benchmark"]
        rank = our_row.index[0] if len(our_row) else "?"
        print(f"\nRank vs paper baselines: {rank}/{len(ranking_df)}")

        # Save results
        save_results(
            metrics=metrics,
            ranking_df=ranking_df,
            output_dir=str(output_dir),
            run_name="GlycanGT",
        )
        # Save arrays, history, model
        np.savez(output_dir / "GlycanGT_arrays.npz",
                 y_true=y_true_test, y_prob=y_prob_test)
        with open(output_dir / "GlycanGT_history.json", "w") as f:
            json.dump(history, f, indent=2)
        torch.save(best_state, output_dir / "model.pt")
        with open(output_dir / "training_config.json", "w") as f:
            json.dump({"model_size": args.model_size, "epochs": args.epochs,
                       "best_epoch": best_epoch, "batch_size": args.batch_size,
                       "lr": args.lr, "seed": args.seed,
                       "note": "supervised end-to-end, no pretrained weights (HuggingFace blocked)"}, f, indent=2)
        print(f"\nResults saved to {output_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
