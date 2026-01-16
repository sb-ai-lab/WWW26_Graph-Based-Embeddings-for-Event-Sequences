import os
from typing import Optional
import argparse

import dgl
from dgl.nn import SAGEConv, GATConv
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

from ptls_extension_2024_research.graphs.utils import MLPPredictor, MLPPredictorGraph, DotProductPredictor, OneLayerPredictor



def weighted_message(edges):
    return {'m': edges.src['h'] * edges.data['w'].unsqueeze(-1)}

def weighted_reduce(nodes):
    return {'h': torch.sum(nodes.mailbox['m'], dim=1)}


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, use_edge_weights, residual, num_layers):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, h_feats, 'mean'))
        self.use_edge_weights = use_edge_weights
        self.residual = residual
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(h_feats, h_feats, 'mean'))

    def forward(self, g, in_feat, edge_weights):
        h = in_feat

        for layer in self.layers[:-1]:
            h_old = h
            h1 = layer(g, h, edge_weight=None)
            h = h1.flatten(1)
            if self.residual:
                h = h + h_old
            h = torch.relu(h)
        h = self.layers[-1](g, h, edge_weights)
        return h


class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, use_edge_weights, residual, num_heads, num_layers,
                 feat_drop=0.6, attn_drop=0.6):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        assert h_feats % num_heads == 0
        self.layer_hfeats = h_feats // num_heads
        self.use_edge_weights = use_edge_weights
        self.residual = residual
        self.layers.append(
            GATConv(in_feats, self.layer_hfeats, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=None))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(h_feats, self.layer_hfeats, num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                                       activation=None))

    def forward(self, g, in_feat, edge_weights=None):
        h = in_feat
        for layer in self.layers[:-1]:
            h_old = h
            # Apply edge weights during message passing
            if edge_weights is not None:
                with g.local_scope():
                    g.edata['w'] = edge_weights  # Assign edge weights to the graph
                    # Update node features in the graph
                    g.ndata['h'] = h  # Input features
                    g.update_all(weighted_message, weighted_reduce)

                    # Get the updated node features after aggregation
                    aggregated_x = g.ndata['h']

                # Combine the original and aggregated features
                h = h + aggregated_x

            h1 = layer(g, h, edge_weight=None)
            h = h1.flatten(1)
            if self.residual:
                h = h + h_old
            h = torch.relu(h)

        if edge_weights is not None:
            with g.local_scope():
                g.edata['w'] = edge_weights  # Assign edge weights to the graph
                # Update node features in the graph
                g.ndata['h'] = h  # Input features
                g.update_all(weighted_message, weighted_reduce)

                # Get the updated node features after aggregation
                aggregated_x = g.ndata['h']

            # Combine the original and aggregated features
            h = h + aggregated_x

        h1 = self.layers[-1](g, h, edge_weight=None)
        h = h1.flatten(1)  # + h
        return h




class GnnLinkPredictor(nn.Module):
    """
    GNN with all components needed for link prediction
    """

    def __init__(self,
                 n_users: int,
                 n_items: int,
                 external_item_embeddings: Optional[torch.FloatTensor],
                 output_size: int = 10,
                 embedding_dim: int = 64,
                 residual: bool = False,
                 link_predictor_name: str = 'MLP',
                 link_predictor_add_sigmoid: bool = True,
                 weight_add_sigmoid: bool = True,
                 gnn_name: str = 'GraphSAGE',
                 use_edge_weights: bool = False,
                 gnn_kwargs_dict=None):
        super().__init__()

        if gnn_kwargs_dict is None:
            gnn_kwargs_dict = {}

        self._output_size = output_size
        self.n_users = n_users
        self.n_items = n_items
        self.has_external_item_embeddings = external_item_embeddings is not None

        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        if self.has_external_item_embeddings:
            self.transform_tr = nn.Linear(external_item_embeddings.shape[1], embedding_dim)
            self.item_embeddings = external_item_embeddings
        else:
            self.item_embeddings = nn.Embedding(n_items, embedding_dim)

        self.use_edge_weights = use_edge_weights
        self.gnn = self._init_gnn(gnn_name, in_feats=embedding_dim, h_feats=output_size,
                                  use_edge_weights=use_edge_weights, residual=residual, **gnn_kwargs_dict)
        self.link_predictor = self._init_link_predictor(link_predictor_name, output_size, weight_add_sigmoid)
        self.real_link_predictor = OneLayerPredictor(output_size, add_sigmoid=link_predictor_add_sigmoid)

    def _init_gnn(self, gnn_name, in_feats, h_feats, use_edge_weights, residual, **gnn_kwags):
        if gnn_name == 'GraphSAGE':
            return GraphSAGE(in_feats=in_feats, h_feats=h_feats, use_edge_weights=use_edge_weights, residual=residual,
                             **gnn_kwags)
        if gnn_name == 'GAT':
            return GAT(in_feats=in_feats, h_feats=h_feats, use_edge_weights=use_edge_weights, residual=residual,
                       **gnn_kwags)
        raise Exception(f'No such graph model {gnn_name}')

    def _init_link_predictor(self, link_predictor_name, output_size, link_predictor_add_sigmoid):
        if link_predictor_name == 'MLP':
            return MLPPredictor(output_size, link_predictor_add_sigmoid)
        if link_predictor_name == 'dot_product':
            return DotProductPredictor(link_predictor_add_sigmoid)
        if link_predictor_name == 'one_layer':
            return OneLayerPredictor(h_feats=output_size, add_sigmoid=link_predictor_add_sigmoid)

        raise Exception(f'No such link predictor {link_predictor_name}')

    def forward(self, g):
        edge_weights = None
        if self.use_edge_weights:
            edge_weights = g.edata['weight']

        user_embeddings = self.user_embeddings(torch.arange(self.n_users))

        # Combine user embeddings and the externally provided item embeddings
        if self.has_external_item_embeddings:
            item_embeddings = self.transform_tr(self.item_embeddings)
        else:
            item_embeddings = self.item_embeddings(torch.arange(self.n_items))

        # Concatenate user and item embeddings
        node_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        assert len(node_embeddings) == g.number_of_nodes()

        node_embeddings = self.gnn(g, node_embeddings, edge_weights)
        return node_embeddings


from typing import Tuple, Optional


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
        src = src[:len(src) // 2]
        dst = dst[:len(src) // 2]
        src_arr = np.unique(src.cpu().numpy()).astype(int)
        dst_arr = np.unique(dst.cpu().numpy()).astype(int)
        return src_arr, dst_arr

    def _sample(self, src_arr: np.ndarray, dst_arr: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.seed is None:
            src_index = np.random.randint(0, len(src_arr), size)
            dst_index = np.random.randint(0, len(dst_arr), size)
        else:
            raise Exception()
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
        assert len(set(self.src_arr).intersection(self.dst_arr)) == 0

    def sample(self, subgraph, size):
        #         assert self.train_graph is subgraph
        return self._sample(self.src_arr, self.dst_arr, size)


def load_graph(graph_file_path):
    g_list, _ = dgl.load_graphs(graph_file_path, [0])
    g = g_list[0]
    print('#Nodes:', g.num_nodes())
    return g



def load_external_item_embs(file_path):
    res = torch.load(file_path)
    return res


def run_pretraining_pipeline(folder_name, config):
    graph = load_graph(os.path.join(config.graph_path, 'train_graph.bin'))
    graph = graph.to(config.device)
    if config.predefined_item_embeddings_file is not None:
        external_item_embeddings = load_external_item_embs(os.path.join(config.data_path, config.predefined_item_embeddings_file))
        external_item_embeddings = external_item_embeddings.to(config.device)
        n_items = external_item_embeddings.shape[0]
    else:
        external_item_embeddings = None
        n_items = config.n_items
    model = GnnLinkPredictor(n_users=config.n_users,
                             n_items=n_items,
                             external_item_embeddings=external_item_embeddings,
                             output_size=config.output_size,
                             embedding_dim=config.embedding_dim,
                             link_predictor_name=config.link_predictor_name,
                             link_predictor_add_sigmoid=config.link_predictor_add_sigmoid,
                             weight_add_sigmoid=config.weight_add_sigmoid,
                             residual=config.residual,
                             gnn_name=config.gnn_name,
                             use_edge_weights=config.use_edge_weights,
                             gnn_kwargs_dict=config.gnn_kwargs_dict)

    model = model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lp_criterion = nn.MSELoss()
    real_lp_criterion = nn.BCELoss()
    neg_edge_sampler = RandEdgeSamplerFull(graph)

    os.makedirs(os.path.join(config.save_model_path, folder_name), exist_ok=True)
    for epoch in range(config.n_epochs):  # Number of epochs
        model.train()

        node_embeddings = model(graph)

        pos_src, pos_dst = graph.edges()
        neg_src, neg_dst = neg_edge_sampler.sample(graph, config.neg_items_per_pos * graph.number_of_edges())

        # score weigths
        pos_scores_weights = model.link_predictor(pos_src, pos_dst, node_embeddings)
        neg_scores_weights = model.link_predictor(neg_src, neg_dst, node_embeddings)

        scores_weights = torch.cat([pos_scores_weights, neg_scores_weights])
        # `like` operations ensure proper device and shape. The shape has to be EXACTLY the same as scores.
        if config.use_edge_weights:
            pos_labels_weights = graph.edata['weight'].unsqueeze(1)
        else:
            pos_labels_weights = torch.ones_like(pos_scores_weights)
        labels_weights = torch.cat([pos_labels_weights, torch.zeros_like(neg_scores_weights)])

        # score LP
        pos_scores_lp = model.real_link_predictor(pos_src, pos_dst, node_embeddings)
        neg_scores_lp = model.real_link_predictor(neg_src, neg_dst, node_embeddings)
        scores_lp = torch.cat([pos_scores_lp, neg_scores_lp])
        # `like` operations ensure proper device and shape. The shape has to be EXACTLY the same as scores.
        pos_labels_lp = torch.ones_like(pos_scores_lp)
        labels_lp = torch.cat([pos_labels_lp, torch.zeros_like(neg_scores_lp)])

        scores_lp_np = scores_lp.clone().detach().cpu().numpy()
        labels_lp_np = labels_lp.clone().detach().cpu().numpy()

        loss_for_weights = lp_criterion(scores_weights, labels_weights)
        loss_for_lp = real_lp_criterion(scores_lp, labels_lp)

        auc = roc_auc_score(labels_lp_np, scores_lp_np)

        loss = config.weight_loss_alpha * loss_for_weights + (1 - config.weight_loss_alpha) * loss_for_lp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}', "Loss: {:.2f}".format(loss.item()), "AUC: {:.2f}".format(auc))

        if epoch % config.save_every == 0:
            torch.save(node_embeddings, os.path.join(config.save_model_path, folder_name, f'{epoch + 1}.pt'))


def create_folder_name(weight_decay, residual, gnn_name, weight_loss_alpha, folder_postfix: str = ''):
    return f'wl-{weight_loss_alpha}_gnn-{gnn_name}_res-{residual}_wd-{weight_decay}{folder_postfix}'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_users', type=int, help='Number of users')
    parser.add_argument('--n_items', type=int, help='Number of items')

    parser.add_argument('--data_path', type=os.path.abspath, default='data/',
                        help='Path to the data directory (default: data/)')
    parser.add_argument('--graph_path', type=os.path.abspath, default='data/graphs/weighted/',
                        help='Path to the graph directory (default: data/graphs/weighted/)')
    parser.add_argument('--save_model_path', type=os.path.abspath, default='data/models_gnn/',
                        help='Path to save the model (default: data/models_gnn/)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (default: cpu)')
    parser.add_argument('--save_every', type=int, default=10, help='Number of epochs to save checkpoints after (default: 10)')
    parser.add_argument('--predefined_item_embeddings_file', type=str, default=None,
                        help='Path to the external item embeddings (default: null)')

    # Adding the arguments for weight_decay, residual, weight_loss_alpha, gnn_name, gnn_kwargs_dict
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (default: 0)')
    parser.add_argument('--residual', type=bool, default=True, help='Use residual connections (default: True)')
    parser.add_argument('--weight_loss_alpha', type=float, default=0.5, help='Weight loss alpha (default: 0.5)')
    parser.add_argument('--gnn_name', type=str, default='GraphSAGE', help="GNN model name (default: 'GraphSAGE')")
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers (default: 2)')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads (default: 4)')

    # Adding missing parameters
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--output_size', type=int, default=48, help='Output size (default: 48)')
    parser.add_argument('--embedding_dim', type=int, default=48, help='Embedding dimension (default: 48)')
    parser.add_argument('--link_predictor_name', type=str, default='one_layer',
                        help="Link predictor name (default: 'one_layer')")
    parser.add_argument('--link_predictor_add_sigmoid', type=bool, default=False,
                        help='Add sigmoid to the link predictor (default: False)')
    parser.add_argument('--use_edge_weights', type=bool, default=True, help='Use edge weights in GNN (default: True)')
    parser.add_argument('--weight_add_sigmoid', type=bool, default=True, help='Add sigmoid to weight (default: True)')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs (default: 200)')
    parser.add_argument('--neg_items_per_pos', type=int, default=1,
                        help='Negative items per positive sample (default: 1)')
    parser.add_argument('--default_folder_name_postfix', type=str, default='',
                        help='A postfix appended to default folder name. (default is empty string)')
    parser.add_argument(
        '--folder_name', type=str, default=None, help="Model\'s folder name (default: " \
        "f'wl-{weight_loss_alpha}_gnn-{gnn_name}_res-{residual}_wd-{weight_decay}{folder_postfix}')")
    args = parser.parse_args()

    if args.folder_name is None:
        args.folder_name = create_folder_name(
            args.weight_decay, args.residual, args.gnn_name, 
            args.weight_loss_alpha, args.default_folder_name_postfix
        )

    return args


def main():
    args = parse_args()
    if args.gnn_name == 'GraphSAGE':
        args.gnn_kwargs_dict = {"num_layers": args.num_layers}
    elif args.gnn_name == 'GAT':
        args.gnn_kwargs_dict = {"num_layers": args.num_layers, "num_heads": args.num_heads}

    run_pretraining_pipeline(args.folder_name, args)
