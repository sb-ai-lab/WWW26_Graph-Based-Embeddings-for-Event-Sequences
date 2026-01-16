from abc import ABC, abstractmethod
from typing import Iterable, Set

import dgl
import numpy as np
import pandas as pd

from ptls_extension_2024_research.graphs.graph_construction.base import GraphBuilder, GraphBuildPipeline
from ptls_extension_2024_research.graphs.utils import create_subgraph_with_all_neighbors


class GenderGraphBuilder(GraphBuilder):
    def preprocess(self, df, client_col, item_col):
        df = df[[client_col, item_col, 'amount']]
        df['amount'] = df['amount'].apply(abs)
        df['amount'] = np.log1p(df['amount'])
        return df

    def _build_weighted_edge_df(self, df, client_col, item_col):
        df = self.preprocess(df, client_col, item_col)

        grouped_edges = df.groupby([client_col, item_col]).agg(sum)
        edge2sum_amount = dict(zip(grouped_edges.index, grouped_edges['amount']))

        grouped_item_weights = df.groupby([item_col]).agg(sum)
        item2sum = dict(zip(grouped_item_weights.index, grouped_item_weights['amount']))

        df_total = df[[client_col, item_col]].drop_duplicates()

        df_total['weight'] = df_total.apply(lambda row:
                                            edge2sum_amount[(row[client_col], row[item_col])] / item2sum[
                                                row[item_col]], axis=1)
        assert all(df_total['weight'] > 0) and all(df_total['weight'] <= 1)
        return df_total, client_col, item_col, 'weight'

    def _build_simple_edge_df(self, df, client_col, item_col):
        return df.drop_duplicates([client_col, item_col]), client_col, item_col


class GenderGraphBuildPipeline(GraphBuildPipeline):
    ENCODED_CLIENT_COL_NAME = 'encoded_client_id'
    graph_builder: GraphBuilder = GenderGraphBuilder()

    def _encode_item_ids(self, df_data: pd.DataFrame, config) -> pd.DataFrame:
        ITEM_MAP_ORIG_COL_NAME = f'_orig_{config.col_item_id}'
        ITEM_MAP_NULL_TOKEN = '#EMPTY'

        item_map = pd.read_csv(config.item_map_file_path)

        df = df_data.rename(columns={config.col_item_id: ITEM_MAP_ORIG_COL_NAME})
        df[ITEM_MAP_ORIG_COL_NAME] = df[ITEM_MAP_ORIG_COL_NAME].fillna(ITEM_MAP_NULL_TOKEN)
        df = df.merge(item_map, on=ITEM_MAP_ORIG_COL_NAME, how='left')
        df = df.drop(columns=[ITEM_MAP_ORIG_COL_NAME])
        return df

    def _encode_client_ids(self, df_data: pd.DataFrame, config) -> pd.DataFrame:
        client_map = pd.read_parquet(config.client_map_file_path)
        assert set(client_map.columns) == {config.orig_col_client_id, self.ENCODED_CLIENT_COL_NAME}
        client_map = client_map.astype({config.orig_col_client_id: df_data[config.orig_col_client_id].dtype})

        df = df_data.merge(client_map, on=config.orig_col_client_id, how='left')
        df = df.drop(columns=[config.orig_col_client_id])

        assert not df[self.ENCODED_CLIENT_COL_NAME].isnull().values.any()
        return df

    def get_train_clients(self, df_data: pd.DataFrame, config) -> Set[int]:
        encoded_client_ids_set__all = set(df_data[self.ENCODED_CLIENT_COL_NAME])

        original_test_client_ids = pd.read_csv(config.test_ids_path)
        encoded_client_ids_set__test = set(
            self._encode_client_ids(original_test_client_ids, config)[self.ENCODED_CLIENT_COL_NAME]
        )

        train_clients = encoded_client_ids_set__all - encoded_client_ids_set__test
        return train_clients

    def preprocess_df(self, df_data, config) -> pd.DataFrame:
        df_data = self._encode_item_ids(df_data, config)
        df_data = self._encode_client_ids(df_data, config)
        return df_data


    def create_train_graph(self, full_g, *args, **kwargs) -> dgl.DGLGraph:
        return create_subgraph_with_all_neighbors(full_g, *args, **kwargs)