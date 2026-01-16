from abc import ABC, abstractmethod
from typing import Set

import dgl
import numpy as np
import pandas as pd

from ptls_extension_2024_research.graphs.graph_construction.base import GraphBuilder, GraphBuildPipeline
from ptls_extension_2024_research.graphs.utils import create_subgraph_with_all_neighbors_and_isolated_items


class MTSGraphBuilder(GraphBuilder):
    def _build_weighted_edge_df(self, df, client_col, item_col):
        df = df[[client_col, item_col, 'request_cnt']]
        df['weight'] = np.log1p(df['request_cnt'])
        df.drop(['request_cnt'], axis=1, inplace=True)
        return df, client_col, item_col, 'weight'

    def _build_simple_edge_df(self, df, client_col, item_col):
        return df.drop_duplicates([client_col, item_col]), client_col, item_col



class MTSGraphBuildPipeline(GraphBuildPipeline):
    ENCODED_CLIENT_COL_NAME = 'encoded_client_id'
    graph_builder: GraphBuilder = MTSGraphBuilder()
    add_test_items_to_graph: bool = True

    def _encode_item_ids(self, df_data: pd.DataFrame, config) -> pd.DataFrame:
        ITEM_MAP_ORIG_COL_NAME = f'_orig_{config.col_item_id}'
        ITEM_MAP_NULL_TOKEN = ''  # here it is empty string!!! TODO

        print(df_data[df_data[config.col_item_id].isnull()][config.col_item_id])
        print('------')

        item_map = pd.read_csv(config.item_map_file_path)
        item_map[ITEM_MAP_ORIG_COL_NAME] = item_map[ITEM_MAP_ORIG_COL_NAME].fillna(ITEM_MAP_NULL_TOKEN)

        df = df_data.rename(columns={config.col_item_id: ITEM_MAP_ORIG_COL_NAME})
        df[ITEM_MAP_ORIG_COL_NAME] = df[ITEM_MAP_ORIG_COL_NAME].fillna(ITEM_MAP_NULL_TOKEN)
        print(df[df[ITEM_MAP_ORIG_COL_NAME].isnull()][ITEM_MAP_ORIG_COL_NAME])

        df = df.merge(item_map, on=ITEM_MAP_ORIG_COL_NAME, how='left')
        df = df.drop(columns=[ITEM_MAP_ORIG_COL_NAME])
        print(df[df[config.col_item_id].isnull()][config.col_item_id])
        # df[config.col_item_id] = df[config.col_item_id].fillna(ITEM_MAP_NULL_TOKEN)
        return df

    def _encode_client_ids(self, df_data: pd.DataFrame, config) -> pd.DataFrame:
        client_map = pd.read_parquet(config.client_map_file_path)
        assert set(client_map.columns) == {config.orig_col_client_id, self.ENCODED_CLIENT_COL_NAME}
        client_map = client_map.astype({config.orig_col_client_id: df_data[config.orig_col_client_id].dtype})

        df = df_data.merge(client_map, on=config.orig_col_client_id, how='left')
        df = df.drop(columns=[config.orig_col_client_id])

        assert not df[self.ENCODED_CLIENT_COL_NAME].isnull().values.any()
        return df

    def preprocess_df(self, df_data, config):
        # df_data[config.col_item_id].fillna('', inplace=True)
        df_data = self._encode_item_ids(df_data, config)
        df_data = self._encode_client_ids(df_data, config)

        return df_data

    def get_train_clients(self, df_data: pd.DataFrame, config) -> Set[int]:
        encoded_client_ids_set__all = set(df_data[self.ENCODED_CLIENT_COL_NAME])

        original_test_client_ids = pd.read_csv(config.test_ids_path)
        encoded_client_ids_set__test = set(
            self._encode_client_ids(original_test_client_ids, config)[self.ENCODED_CLIENT_COL_NAME]
        )

        train_clients = encoded_client_ids_set__all - encoded_client_ids_set__test
        return train_clients

    def get_test_items(self, df_data: pd.DataFrame, config) -> Set[int]:
        # df_data = df_data[~df_data[config.col_item_id].isnull()]
        encoded_item_ids_set__all = set(df_data[config.col_item_id])

        original_test_client_ids = pd.read_csv(config.test_ids_path)
        encoded_client_ids_set__test = set(
            self._encode_client_ids(original_test_client_ids, config)[self.ENCODED_CLIENT_COL_NAME]
        )
        df_data = df_data[~df_data[self.ENCODED_CLIENT_COL_NAME].isin(encoded_client_ids_set__test)]

        encoded_item_ids_set__train = set(df_data[config.col_item_id])
        test_items = encoded_item_ids_set__all - encoded_item_ids_set__train
        return test_items

    def create_train_graph(self, full_g, *args, **kwargs) -> dgl.DGLGraph:
        return create_subgraph_with_all_neighbors_and_isolated_items(full_g, *args, **kwargs)