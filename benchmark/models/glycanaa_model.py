from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool

from benchmark.models.glycanaa_graph import ATOM_TYPE_DIM, CROSS_RELATION, NUM_MONO_RELATIONS, TOTAL_RELATIONS


class RelationalGCNLayer(nn.Module):
    """Per-relation message passing with mean aggregation plus a root transform."""

    def __init__(self, in_dim, out_dim, num_relations, batch_norm=False, activation='gelu'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.root = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.batch_norm = nn.BatchNorm1d(out_dim) if batch_norm else None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.root.weight)
        nn.init.zeros_(self.bias)
        if self.batch_norm is not None:
            self.batch_norm.reset_parameters()

    def forward(self, x, edge_index, edge_type, num_nodes):
        out = self.root(x)
        if edge_index.numel() > 0:
            src_all, dst_all = edge_index
            for rel in range(self.num_relations):
                mask = edge_type == rel
                if not bool(mask.any()):
                    continue
                src = src_all[mask]
                dst = dst_all[mask]
                msg = x[src] @ self.weight[rel]
                agg = x.new_zeros((num_nodes, self.out_dim))
                agg.index_add_(0, dst, msg)
                deg = x.new_zeros((num_nodes, 1))
                deg.index_add_(0, dst, torch.ones((dst.numel(), 1), device=x.device, dtype=x.dtype))
                out = out + agg / deg.clamp_min_(1.0)
        out = out + self.bias
        if self.batch_norm is not None and out.size(0) > 1:
            out = self.batch_norm(out)
        if self.activation == 'gelu':
            out = F.gelu(out)
        elif self.activation == 'relu':
            out = F.relu(out)
        elif self.activation == 'elu':
            out = F.elu(out)
        elif self.activation in {None, 'none'}:
            pass
        else:
            raise ValueError(f'Unsupported activation: {self.activation}')
        return out


class GlycanAAModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=3,
                 concat_hidden=True, num_mono_relations=NUM_MONO_RELATIONS,
                 num_total_relations=TOTAL_RELATIONS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.concat_hidden = concat_hidden
        self.num_mono_relations = num_mono_relations
        self.num_total_relations = num_total_relations

        self.embedding = nn.Embedding(vocab_size + ATOM_TYPE_DIM, hidden_dim)
        self.atom_layers = nn.ModuleList([
            RelationalGCNLayer(hidden_dim, hidden_dim, num_total_relations, batch_norm=False, activation='gelu')
            for _ in range(num_layers)
        ])
        self.cross_layers = nn.ModuleList([
            RelationalGCNLayer(hidden_dim, hidden_dim, num_total_relations, batch_norm=False, activation='gelu')
            for _ in range(num_layers)
        ])
        self.mono_layers = nn.ModuleList([
            RelationalGCNLayer(hidden_dim, hidden_dim, num_total_relations, batch_norm=False, activation='gelu')
            for _ in range(num_layers)
        ])

        node_dim = hidden_dim * num_layers if concat_hidden else hidden_dim
        self.output_dim = node_dim
        self.classifier = nn.Linear(node_dim * 2, 1)

    def forward(self, data) -> torch.Tensor:
        x = self.embedding(data.x)
        edge_index = data.edge_index
        edge_type = data.edge_type
        is_mono = data.is_mono
        num_nodes = data.num_nodes

        src_is_mono = is_mono[edge_index[0]] if edge_index.numel() else torch.zeros(0, dtype=torch.bool, device=x.device)
        dst_is_mono = is_mono[edge_index[1]] if edge_index.numel() else torch.zeros(0, dtype=torch.bool, device=x.device)
        mono_mask = edge_type < self.num_mono_relations
        atom_mask = (edge_type >= self.num_mono_relations) & (edge_type < CROSS_RELATION)
        cross_mask = (edge_type == CROSS_RELATION) & (~src_is_mono) & dst_is_mono

        mono_edge_index = edge_index[:, mono_mask]
        mono_edge_type = edge_type[mono_mask]
        atom_edge_index = edge_index[:, atom_mask]
        atom_edge_type = edge_type[atom_mask]
        cross_edge_index = edge_index[:, cross_mask]
        cross_edge_type = edge_type[cross_mask]

        layer_input = x
        hiddens = []
        for atom_layer, cross_layer, mono_layer in zip(self.atom_layers, self.cross_layers, self.mono_layers):
            hidden_atom = atom_layer(layer_input, atom_edge_index, atom_edge_type, num_nodes)

            input_cross = layer_input.clone()
            input_cross[~is_mono] = hidden_atom[~is_mono]
            hidden_cross = cross_layer(input_cross, cross_edge_index, cross_edge_type, num_nodes)

            input_mono = torch.zeros_like(layer_input)
            input_mono[is_mono] = hidden_cross[is_mono]
            hidden_mono = mono_layer(input_mono, mono_edge_index, mono_edge_type, num_nodes)

            layer_output = torch.zeros_like(layer_input)
            layer_output[~is_mono] = hidden_atom[~is_mono]
            layer_output[is_mono] = hidden_mono[is_mono]
            hiddens.append(layer_output)
            layer_input = layer_output

        all_node_feature = torch.cat(hiddens, dim=-1) if self.concat_hidden else hiddens[-1]
        mono_feature = all_node_feature[is_mono]
        mono_batch = data.batch[is_mono]
        pooled = torch.cat([
            global_mean_pool(mono_feature, mono_batch),
            global_max_pool(mono_feature, mono_batch),
        ], dim=-1)
        return self.classifier(pooled).squeeze(-1)
