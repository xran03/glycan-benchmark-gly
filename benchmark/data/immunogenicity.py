"""
Download and load the GlycanML immunogenicity dataset.

Dataset: https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_immunogenicity.csv
Stats  : Train 1,046 | Valid 131 | Test 143
Labels : binary (0 = non-immunogenic, 1 = immunogenic)
"""

import hashlib
import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATASET_URL = "https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_immunogenicity.csv"
DATASET_MD5 = "5ee0814b23304f67247e85786a8b4688"
CACHE_DIR   = Path.home() / ".cache" / "glycan_benchmark" / "immunogenicity"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(dest_dir: Path = CACHE_DIR) -> Path:
    """Download the immunogenicity CSV if not already cached."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "glycan_immunogenicity.csv"

    if dest.exists() and _md5(dest) == DATASET_MD5:
        logger.info("Dataset already cached at %s", dest)
        return dest

    logger.info("Downloading immunogenicity dataset …")
    resp = requests.get(DATASET_URL, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="immunogenicity.csv") as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    assert _md5(dest) == DATASET_MD5, "MD5 mismatch — download may be corrupt"
    logger.info("Dataset saved to %s", dest)
    return dest


def build_vocab(glycans: list[str]) -> dict[str, int]:
    """
    Build a token-to-index vocabulary from a list of IUPAC-condensed glycan strings.

    Uses libr=None so glycowork tokenises without requiring a pre-built library,
    collecting all unique node tokens (monosaccharide + linkage labels).
    """
    from glycowork.motif.graph import glycan_to_nxGraph

    vocab: set[str] = set()
    for g in glycans:
        try:
            nx = glycan_to_nxGraph(g, libr=None)
            for _, data in nx.nodes(data=True):
                vocab.add(data["string_labels"])
        except Exception:
            pass
    return {tok: idx for idx, tok in enumerate(sorted(vocab))}


class ImmunogenicityDataset:
    """
    Wrapper around the GlycanML immunogenicity CSV.

    Columns after loading:
        glycan      – IUPAC-condensed string
        label       – int (0 / 1)
        split       – 'train' | 'valid' | 'test'

    Attributes:
        vocab       – dict[str, int] built from all glycans in this dataset
    """

    def __init__(self, csv_path: Path | None = None):
        if csv_path is None:
            csv_path = download_dataset()
        self.df = self._parse(csv_path)
        logger.info("Building dataset vocabulary …")
        self.vocab = build_vocab(self.df["glycan"].tolist())
        logger.info("Vocabulary size: %d tokens", len(self.vocab))

    @staticmethod
    def _parse(path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        split_cols = [c for c in raw.columns if c in {"train", "valid", "test"}]
        if not split_cols:
            raise ValueError(f"No split columns found. Columns: {raw.columns.tolist()}")

        def resolve_split(row):
            for col in split_cols:
                try:
                    if bool(int(row[col])):
                        return col
                except (ValueError, TypeError):
                    pass
            return "train"

        raw["split"] = raw.apply(resolve_split, axis=1)
        label_col = "immunogenicity"
        raw[label_col] = pd.to_numeric(raw[label_col], errors="coerce").fillna(0).astype(int)
        return raw[["glycan", label_col, "split"]].rename(columns={label_col: "label"})

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train_df, valid_df, test_df)."""
        g = self.df.groupby("split")
        return g.get_group("train"), g.get_group("valid"), g.get_group("test")

    def stats(self) -> str:
        lines = ["ImmunogenicityDataset stats:"]
        for split, grp in self.df.groupby("split"):
            pos = grp["label"].sum()
            lines.append(f"  {split:6s}: {len(grp):4d} samples  pos={pos} ({100*pos/len(grp):.1f}%)")
        return "\n".join(lines)


def load_splits(
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Convenience function.

    Returns:
        (train_df, valid_df, test_df, vocab)
    """
    ds = ImmunogenicityDataset(csv_path)
    train_df, valid_df, test_df = ds.split()
    return train_df, valid_df, test_df, ds.vocab

