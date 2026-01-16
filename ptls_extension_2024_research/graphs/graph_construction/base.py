import os
import logging
import numpy as np
import torch
import dgl
from abc import ABC, abstractmethod
from typing import Optional, Iterable
import pandas as pd


class GraphBuilder(ABC):
    @abstractmethod
    def _build_weighted_edge_df(self, df, src_col, dst_col):
        pass

    @abstractmethod
    def _build_simple_edge_df(self, df, src_col, dst_col):
        pass

    def build(self, df, client_col, item_col, use_weights):
        if use_weights:
            df, client_col, item_col, weight_col = self._build_weighted_edge_df(df, client_col, item_col)
        else:
            df, client_col, item_col = self._build_simple_edge_df(df, client_col, item_col)
            weight_col = None

        g, client_id2graph_id, item_id2graph_id, items_cnt = create_graph_from_df(df, client_col, item_col, weight_col)
        return g, client_id2graph_id, item_id2graph_id, items_cnt


def create_graph_from_df(df, client_col: str, item_col: str, weight_col: Optional[str] = None):
    # Create a dictionary to map node names to integers
    unique_nodes_client = np.sort(df[client_col].unique().astype(int))
    unique_nodes_item = np.sort(df[item_col].unique().astype(int))

    # create index mapping
    client_id2graph_id = torch.zeros(unique_nodes_client.max()+1, dtype=torch.long)
    client_id2graph_id[unique_nodes_client] = torch.arange(len(unique_nodes_client))

    # items always follow user index
    item_id2graph_id = torch.zeros(unique_nodes_item.max() + 1, dtype=torch.long)
    item_id2graph_id[unique_nodes_item] = torch.arange(len(unique_nodes_item)) + len(unique_nodes_client)

    # Convert source and destination columns to integer indices
    src = client_id2graph_id[df[client_col].values]
    dst = item_id2graph_id[df[item_col].values]

    src_bi = torch.cat([src, dst])
    dst_bi = torch.cat([dst, src])

    # Create the graph
    g = dgl.graph((src_bi, dst_bi))

    # Add edge weights
    if weight_col is not None:
        weights = torch.tensor(df[weight_col].values, dtype=torch.float32)
        weights_bi = torch.cat([weights, weights])
        g.edata['weight'] = weights_bi

    return g, client_id2graph_id, item_id2graph_id, len(unique_nodes_item)



class GraphBuildPipeline(ABC):
    ENCODED_CLIENT_COL_NAME: str = ''
    graph_builder: GraphBuilder = None
    add_test_items_to_graph: bool = False

    @abstractmethod
    def preprocess_df(self, f_data, config) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_train_clients(self, df_data, config) -> Iterable:
        pass

    @abstractmethod
    def create_train_graph(self, full_g, *args, **kwargs) -> dgl.DGLGraph:
        pass

    def get_test_items(self, df_data, config) -> Iterable:
        pass

    def create_graph(self, df_data, config, logger):
        os.makedirs(config.output_graph_path, exist_ok=True)

        df_data = self.preprocess_df(df_data, config)
        full_g, client_id2full_graph_id, item_id2full_graph_id, real_items_cnt = self.graph_builder.build(df=df_data,
                                                                                                            client_col=self.ENCODED_CLIENT_COL_NAME,
                                                                                                            item_col=config.col_item_id,
                                                                                                            use_weights=config.use_weights,
                                                                                                            )
        dgl.save_graphs(os.path.join(config.output_graph_path, config.output_full_graph_file), [full_g])
        torch.save(client_id2full_graph_id,
                   os.path.join(config.output_graph_path, config.output_client_id2full_graph_id_file))
        torch.save(item_id2full_graph_id,
                   os.path.join(config.output_graph_path, config.output_item_id2full_graph_id_file))

        # print(client_id2full_graph_id)

        logger.info("Saved full graph")
        # create train graph
        train_clients = self.get_train_clients(df_data, config)
        train_clients = torch.LongTensor(sorted(train_clients))
        if self.add_test_items_to_graph:
            test_items = self.get_test_items(df_data, config)
            test_items = torch.LongTensor(sorted(test_items))
            train_g = \
                self.create_train_graph(full_g, node_ids=client_id2full_graph_id[train_clients],
                                        isolated_item_ids=item_id2full_graph_id[test_items])
        else:
            train_g = \
                self.create_train_graph(full_g, node_ids=client_id2full_graph_id[train_clients])


        # 1st part - clients, then - items
        full_graph_id2train_graph_id = torch.zeros(train_g.ndata['_ID'].max() + 1, dtype=torch.long)
        full_graph_id2train_graph_id[train_g.ndata['_ID']] = train_g.nodes()

        # set of items is the same for both graphs
        print(len(train_g.nodes()))
        print(len(train_clients))
        print(real_items_cnt)
        assert len(train_g.nodes()) - len(train_clients) == real_items_cnt

        client_id2train_graph_id = torch.zeros(len(client_id2full_graph_id), dtype=torch.long)
        client_id2train_graph_id[train_clients] = full_graph_id2train_graph_id[client_id2full_graph_id[train_clients]]

        item_id2train_graph_id = torch.zeros(len(item_id2full_graph_id), dtype=torch.long)
        all_items = torch.arange(len(item_id2full_graph_id))
        item_id2train_graph_id[all_items] = full_graph_id2train_graph_id[item_id2full_graph_id[all_items]]

        dgl.save_graphs(os.path.join(config.output_graph_path, config.output_train_graph_file), [train_g])
        torch.save(client_id2train_graph_id,
                   os.path.join(config.output_graph_path, config.output_client_id2train_graph_id_file))
        torch.save(item_id2train_graph_id,
                   os.path.join(config.output_graph_path, config.output_item_id2train_graph_id_file))