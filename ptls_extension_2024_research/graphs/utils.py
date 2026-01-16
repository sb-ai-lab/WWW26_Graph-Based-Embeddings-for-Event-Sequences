from typing import Tuple


import dgl
import numpy as np
import torch
from torch import nn


class RandEdgeSampler:
    """
    Given a subgraph, samples random edges from it.
    Is used as a negative sampler.
    """
    def __init__(self, seed=None):
        self.seed = seed
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def _get_unique_src_dst_tuple(self, subgraph: dgl.DGLGraph) -> Tuple[np.ndarray, np.ndarray]:
        src, dst = subgraph.edges()
        src_arr = np.unique(src.cpu().numpy()).astype(int)
        dst_arr = np.unique(dst.cpu().numpy()).astype(int)
        return src_arr, dst_arr
    
    def _sample(self, src_arr: np.ndarray, dst_arr: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.seed is None:
            src_index = np.random.randint(0, len(src_arr), size)
            dst_index = np.random.randint(0, len(dst_arr), size)
        else:
            src_index = self.random_state.randint(0, len(src_arr), size)
            dst_index = self.random_state.randint(0, len(dst_arr), size)
        return src_arr[src_index], dst_arr[dst_index]


    def sample(self, subgraph: dgl.DGLGraph, size: int):
        src_arr, dst_arr = self._get_unique_src_dst_tuple(subgraph)
        return self._sample(src_arr, dst_arr, size)

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class RandEdgeSamplerFull(RandEdgeSampler):
    """
    A special case of RandEdgeSampler 
    where the training is done on one version of the graph only
    and thus the `_get_unique_src_dst_tuple` is done only once. 
    """
    def __init__(self, train_graph: dgl.DGLGraph, seed=None):
        super().__init__(seed)
        self.train_graph = train_graph
        self.src_arr, self.dst_arr = self._get_unique_src_dst_tuple(train_graph)

    def sample(self, subgraph, size):
        assert self.train_graph is subgraph
        return self._sample(self.src_arr, self.dst_arr, size)





class MLPPredictor(nn.Module):
    def __init__(self, h_feats, add_sigmoid=True):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
        self.act = nn.ReLU()
        self.add_sigmoid = add_sigmoid

    def apply_edges(self, src_feats, dst_feats):
        h = torch.cat([src_feats, dst_feats], 1)
        return {'score': self.W2(self.act(self.W1(h)))}
        # return {'score': self.sigmoid(self.W2(F.relu(self.W1(h)))).squeeze(1)}

    def forward(self, src_list, dst_list, feats):
        edge_scores = self.apply_edges(feats[src_list], feats[dst_list])
        if not self.add_sigmoid:
            return edge_scores['score']
        return edge_scores['score'].sigmoid()


class MLPPredictorGraph(nn.Module):
    def __init__(self, h_feats, add_sigmoid=True):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
        self.act = nn.ReLU()
        self.add_sigmoid = add_sigmoid

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(self.act(self.W1(h)))}
        # return {'score': self.sigmoid(self.W2(F.relu(self.W1(h)))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            if not self.add_sigmoid:
                return g.edata['score']
            return g.edata['score'].sigmoid()


class DotProductPredictor(nn.Module):
    def __init__(self, add_sigmoid=True):
        super().__init__()
        self.add_sigmoid = add_sigmoid

    def apply_edges(self, src_feats, dst_feats):
        return {'score': torch.sum(src_feats * dst_feats, dim=1)}
        # h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        # return {'score': self.W2(self.act(self.W1(h)))}
        # return {'score': self.sigmoid(self.W2(F.relu(self.W1(h)))).squeeze(1)}

    def forward(self, src_list, dst_list, feats):
        edge_scores = self.apply_edges(feats[src_list], feats[dst_list])
        if not self.add_sigmoid:
            return edge_scores['score']
        return edge_scores['score'].sigmoid()



class OneLayerPredictor(nn.Module):
    def __init__(self, h_feats, add_sigmoid=True):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, 1)
        self.add_sigmoid = add_sigmoid

    def apply_edges(self, src_feats, dst_feats):
        h = torch.cat([src_feats, dst_feats], 1)
        return {'score': self.W1(h)}
        # return {'score': self.sigmoid(self.W2(F.relu(self.W1(h)))).squeeze(1)}

    def forward(self, src_list, dst_list, feats):
        edge_scores = self.apply_edges(feats[src_list], feats[dst_list])
        if not self.add_sigmoid:
            return edge_scores['score']
        return edge_scores['score'].sigmoid()


def create_subgraph_with_all_neighbors(graph: dgl.DGLGraph, node_ids: torch.Tensor):
    # Find all neighbors by using the predecessors and successors
    in_neighbors = graph.in_edges(node_ids)[0].unique()
    out_neighbors = graph.out_edges(node_ids)[1].unique()

    # Combine the nodes of interest with their in-neighbors and out-neighbors
    all_nodes = torch.cat([node_ids, in_neighbors, out_neighbors]).unique()
    all_nodes, _ = torch.sort(all_nodes)


    # Induce a subgraph with all the relevant nodes
    subgraph = dgl.node_subgraph(graph, nodes=all_nodes)
    return subgraph


def create_subgraph_with_all_neighbors_and_isolated_items(graph: dgl.DGLGraph,
                                                          node_ids: torch.Tensor,
                                                          isolated_item_ids: torch.Tensor):
    subgraph = create_subgraph_with_all_neighbors(graph, node_ids)
    subgraph.add_nodes(len(isolated_item_ids), data={'_ID': isolated_item_ids})
    return subgraph
