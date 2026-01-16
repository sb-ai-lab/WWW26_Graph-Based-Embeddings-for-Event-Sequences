from dgl import DGLGraph  # for typing
from dgl.nn import SAGEConv, GATConv
import torch
import torch.nn as nn

class GraphModel(nn.Module):
    """
    Given a (sub)-graph, batch of node features, and batch of edge weights, return the updated node features.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, g: DGLGraph, in_feat: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Given a (sub)-graph, batch of node features, 
        and batch of edge weights, return the updated node features.

        Arguments:
        ----------
        g: DGLGraph
            The graph to be processed.
        in_feat: torch.Tensor. Shape: (N, D) where N is the number of nodes and D is the feature dimension.
            The input node features. Correspond to g.nodes() (aka torch.arange(g.number_of_nodes())
        edge_weights: torch.Tensor. Shape: (E,) where E is the number of edges.
            The edge weights. Correspond to g.edges(). Note: g.edges() returns a tuple (src, dst).
        """
        raise NotImplementedError


class GraphSAGE(GraphModel):
    def __init__(self, in_feats: int, h_feats: int, use_edge_weights: bool, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, h_feats, 'mean'))
        self.use_edge_weights = use_edge_weights
        for _ in range(num_layers):
            self.layers.append(SAGEConv(h_feats, h_feats, 'mean'))

    def forward(self, 
                g: DGLGraph, 
                in_feat: torch.Tensor, 
                edge_weights: torch.Tensor) -> torch.Tensor:
        h = in_feat
        for layer in self.layers[:-1]:
            h = layer(g, h, edge_weight=edge_weights)
            h = torch.relu(h)
        h = self.layers[-1](g, h, edge_weights)
        return h


class GAT(GraphModel):
    def __init__(self, in_feats: int, h_feats: int, use_edge_weights: bool, 
                 num_heads: int, num_layers: int,
                 feat_drop=0.6, attn_drop=0.6):
        super().__init__()
        self.layers = nn.ModuleList()
        assert h_feats % num_heads == 0
        self.layer_hfeats = h_feats // num_heads
        self.use_edge_weights = use_edge_weights
        self.layers.append(GATConv(in_feats, self.layer_hfeats, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=nn.ELU()))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(h_feats, self.layer_hfeats, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=nn.ELU()))

    def forward(self, 
                g: DGLGraph, 
                in_feat: torch.Tensor, 
                edge_weights: torch.Tensor) -> torch.Tensor:
        h = in_feat
        for layer in self.layers[:-1]:
            h = layer(g, h, edge_weight=edge_weights)
            h = h.flatten(1)
        h = self.layers[-1](g, h, edge_weight=edge_weights)
        h = h.flatten(1)
        return h
        # return h.mean(1)
