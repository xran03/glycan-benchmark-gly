from __future__ import annotations

import ast
import logging
import re
from functools import lru_cache
from pathlib import Path

import torch
from glyles import convert
from glycowork.motif.graph import glycan_to_nxGraph
from rdkit import Chem
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

NUM_MONO_RELATIONS = 16
NUM_BOND_RELATIONS = 4
CROSS_RELATION = NUM_MONO_RELATIONS + NUM_BOND_RELATIONS
TOTAL_RELATIONS = CROSS_RELATION + 1
ATOM_TYPE_DIM = 64

_MONO_RELATION_LABELS = [
    "b1-4", "a1-3", "b1-3", "b1-2", "a1-6",
    "a1-2", "a2-3", "b1-6", "a1-4", "a2-6",
    "a1-5", "a2-8", "a1-7", "a2-4", "b1-7",
]
MONO_RELATION_VOCAB = {label: idx for idx, label in enumerate(_MONO_RELATION_LABELS)}
OTHER_MONO_RELATION = NUM_MONO_RELATIONS - 1
LINKAGE_RE = re.compile(r"^[ab?][0-9?]-[0-9?]$")

MONO_VOCAB: dict[str, int] = {}


def set_mono_vocab(vocab: dict[str, int]) -> None:
    global MONO_VOCAB
    MONO_VOCAB = dict(vocab)



def build_mono_vocab(glycans: list[str]) -> dict[str, int]:
    monos: set[str] = set()
    for glycan in glycans:
        try:
            graph = glycan_to_nxGraph(glycan, libr=None)
        except Exception:
            continue
        for _, data in graph.nodes(data=True):
            label = str(data.get("string_labels", ""))
            if label and not LINKAGE_RE.match(label):
                monos.add(label)
    ordered = sorted(monos)
    if "<UNK>" not in ordered:
        ordered.append("<UNK>")
    return {mono: idx for idx, mono in enumerate(ordered)}


@lru_cache(maxsize=1)
def _load_reference_smiles_lookup() -> dict[str, str]:
    source = Path(__file__).resolve().parents[2] / "GlycanAA" / "module" / "custom_data" / "all_atom_glycan.py"
    if not source.exists():
        return {}
    try:
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "glycan2smiles":
                        value = ast.literal_eval(node.value)
                        if isinstance(value, dict):
                            return {str(k): str(v) for k, v in value.items()}
    except Exception as exc:
        logger.warning("Failed to parse reference glycan2smiles lookup: %s", exc)
    return {}


@lru_cache(maxsize=512)
def _mono_to_smiles(mono: str) -> str:
    lookup = _load_reference_smiles_lookup()
    smiles = lookup.get(mono, "")
    if smiles:
        return smiles
    try:
        converted = convert(mono)
        if isinstance(converted, list) and converted:
            _, smiles = converted[0]
            return smiles or ""
    except Exception:
        return ""
    return ""


@lru_cache(maxsize=512)
def _mono_atom_template(mono: str) -> tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]:
    smiles = _mono_to_smiles(mono)
    if not smiles:
        return (), ()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (), ()

    atom_types = []
    for atom in mol.GetAtoms():
        atomic_num = max(int(atom.GetAtomicNum()), 1)
        atom_types.append(min(atomic_num - 1, ATOM_TYPE_DIM - 1))

    bond_edges: list[tuple[int, int, int]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            rel = 0
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            rel = 1
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            rel = 2
        else:
            rel = 3
        rel += NUM_MONO_RELATIONS
        bond_edges.append((begin, end, rel))
        bond_edges.append((end, begin, rel))

    return tuple(atom_types), tuple(bond_edges)



def _mono_relation_id(label: str) -> int:
    return MONO_RELATION_VOCAB.get(label, OTHER_MONO_RELATION)



def build_glycanaa_graph(iupac: str) -> Data | None:
    """
    Build a torch_geometric graph for the GlycanAA reimplementation.

    Node ids in `x`:
      mono nodes: index into MONO_VOCAB
      atom nodes: len(MONO_VOCAB) + atom_type_index
    """
    if not MONO_VOCAB:
        raise RuntimeError("MONO_VOCAB is empty. Call set_mono_vocab() before building graphs.")

    try:
        backbone = glycan_to_nxGraph(iupac, libr=None)
    except Exception as exc:
        logger.warning("Failed to parse glycan backbone for %r: %s", iupac, exc)
        return None

    mono_nodes: list[int] = []
    mono_labels: dict[int, str] = {}
    for node_id, data in backbone.nodes(data=True):
        label = str(data.get("string_labels", ""))
        if label and not LINKAGE_RE.match(label):
            mono_nodes.append(int(node_id))
            mono_labels[int(node_id)] = label

    if not mono_nodes:
        logger.warning("No monosaccharide nodes found for %r", iupac)
        return None

    mono_to_local = {node_id: idx for idx, node_id in enumerate(mono_nodes)}
    unk_idx = MONO_VOCAB.get("<UNK>", 0)
    x = [MONO_VOCAB.get(mono_labels[node_id], unk_idx) for node_id in mono_nodes]
    is_mono = [True] * len(mono_nodes)
    edges: list[list[int]] = []
    edge_types: list[int] = []

    undirected_backbone = backbone.to_undirected()
    for node_id, data in backbone.nodes(data=True):
        label = str(data.get("string_labels", ""))
        if not LINKAGE_RE.match(label):
            continue
        neighbors = [int(n) for n in undirected_backbone.neighbors(node_id) if int(n) in mono_to_local]
        if len(neighbors) < 2:
            continue
        src = mono_to_local[neighbors[0]]
        dst = mono_to_local[neighbors[1]]
        rel = _mono_relation_id(label)
        edges.append([src, dst])
        edge_types.append(rel)
        edges.append([dst, src])
        edge_types.append(rel)

    for node_id in mono_nodes:
        mono_idx = mono_to_local[node_id]
        mono_label = mono_labels[node_id]
        atom_types, bond_edges = _mono_atom_template(mono_label)
        if not atom_types:
            continue

        atom_offset = len(x)
        for atom_type in atom_types:
            x.append(len(MONO_VOCAB) + atom_type)
            is_mono.append(False)

        for src, dst, rel in bond_edges:
            edges.append([atom_offset + src, atom_offset + dst])
            edge_types.append(rel)

        for atom_idx in range(len(atom_types)):
            atom_node = atom_offset + atom_idx
            edges.append([mono_idx, atom_node])
            edge_types.append(CROSS_RELATION)
            edges.append([atom_node, mono_idx])
            edge_types.append(CROSS_RELATION)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    return Data(
        x=torch.tensor(x, dtype=torch.long),
        edge_index=edge_index,
        edge_type=edge_type,
        is_mono=torch.tensor(is_mono, dtype=torch.bool),
        num_nodes=len(x),
        glycan=iupac,
    )
