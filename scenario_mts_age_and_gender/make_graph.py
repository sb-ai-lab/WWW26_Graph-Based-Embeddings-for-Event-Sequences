import sys
import os

from ptls_extension_2024_research.graphs.graph_construction.utils import configure_logger

sys.path.append(os.path.abspath('..'))

from typing import Set

import argparse
import logging
from datetime import datetime

import dgl
import numpy as np
import pandas as pd
import torch

from ptls_extension_2024_research.graphs.graph_construction.mts import MTSGraphBuildPipeline

ENCODED_CLIENT_COL_NAME = 'encoded_client_id'

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=os.path.abspath)
    parser.add_argument('--trx_file', type=str)
    parser.add_argument('--orig_col_client_id', type=str)
    parser.add_argument('--col_item_id', type=str)
    parser.add_argument('--item_map_file_path', type=str)
    parser.add_argument('--client_map_file_path', type=str)
    parser.add_argument('--test_ids_path', type=os.path.abspath)

    parser.add_argument('--output_graph_path', type=os.path.abspath)
    parser.add_argument('--output_full_graph_file', type=str)
    parser.add_argument('--output_train_graph_file', type=str)
    parser.add_argument('--output_client_id2full_graph_id_file', type=str)
    parser.add_argument('--output_item_id2full_graph_id_file', type=str)
    parser.add_argument('--output_client_id2train_graph_id_file', type=str)
    parser.add_argument('--output_item_id2train_graph_id_file', type=str)
    parser.add_argument('--log_file', type=os.path.abspath)
    parser.add_argument('--use_weights', action='store_true', default=False)

    args = parser.parse_args(args)
    # args.__dict__['data_path'] = 'data'
    # args.__dict__['trx_file'] = 'competition_data_edges.parquet'
    # args.__dict__['item_map_file_path'] = 'data/url_host_preprocesed_mapping.csv'
    # args.__dict__['orig_col_client_id'] = 'user_id'
    # args.__dict__['col_item_id'] = "url_host_preprocesed"
    # args.__dict__['item_map_file_path'] = "data/url_host_preprocesed_mapping.csv"
    # args.__dict__['client_map_file_path'] = "data/client_id_map.parquet"
    # args.__dict__['test_ids_path'] = "data/test_ids_file.csv"
    # args.__dict__['output_graph_path'] = 'data/graphs/weighted'
    # args.__dict__['output_train_graph_file'] = "train_graph.bin"
    # args.__dict__['output_full_graph_file'] = "full_graph.bin"
    # args.__dict__['output_client_id2full_graph_id_file'] = "client_id2full_graph_id.pt"
    # args.__dict__['output_item_id2full_graph_id_file'] = "item_id2full_graph_id.pt"
    # args.__dict__['output_client_id2train_graph_id_file'] = "client_id2train_graph_id.pt"
    # args.__dict__['output_item_id2train_graph_id_file'] = "item_id2train_graph_id.pt"
    # args.__dict__['log_file'] = "results/dataset_mts.txt"
    # args.__dict__['use_weights'] = "true"
    # args.__dict__['client_map_file_path'] = "data/client_id_map.parquet"
    return args


if __name__ == '__main__':
    _start = datetime.now()
    config = parse_args()
    configure_logger(config)
    logger.info('Parsed args:\n' + '\n'.join([f'  {k:15}: {v}' for k, v in vars(config).items()]))

    df_data = pd.read_parquet(os.path.join(config.data_path, config.trx_file))
    MTSGraphBuildPipeline().create_graph(df_data, config, logger)

    _duration = datetime.now() - _start
    logger.info(f'Data collected in {_duration.seconds} sec ({_duration})')

