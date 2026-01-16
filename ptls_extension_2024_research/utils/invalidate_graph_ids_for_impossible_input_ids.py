import argparse
import logging
import os
from typing import Iterable, Optional

import pandas as pd
import torch
from tqdm import tqdm


ENCODED_CLIENT_ID_COLUMN = 'encoded_client_id'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--not_encoded_test_client_ids_path', type=str, required=True)
    parser.add_argument('--client_id_map_path', type=str, required=True)
    parser.add_argument('--train_dataset_path', type=str, required=True)
    parser.add_argument('--test_dataset_path', type=str, default=None)
    parser.add_argument('--include_test_items', action='store_true')
    parser.add_argument('--orig_client_id_column', type=str, required=True)
    parser.add_argument('--item_id_column', type=str, required=True)
    parser.add_argument('--orig_client_id2train_graph_id_tensor_path', type=str, required=True)
    parser.add_argument('--orig_item_id2train_graph_id_tensor_path', type=str, required=True)
    parser.add_argument('--out_client_id2train_graph_id_tensor_path', type=str, required=True)
    parser.add_argument('--out_item_id2train_graph_id_tensor_path', type=str, required=True)
    parser.add_argument('--out_client_id2train_graph_id_dict_path', type=str, default=None)
    parser.add_argument('--out_item_id2train_graph_id_dict_path', type=str, default=None)
    parser.add_argument('--invalid_train_graph_id', type=int, required=True)

    args = parser.parse_args()

    if args.include_test_items:
        assert args.test_dataset_path is not None

    return args


def get_train_client_ids(args: argparse.Namespace) -> Iterable[int]:
    orig_col = args.orig_client_id_column
    test_client_ids_not_encoded = pd.read_csv(args.not_encoded_test_client_ids_path)
    client_id_map = pd.read_parquet(args.client_id_map_path)

    logging.info("Loaded client ID map:\n%s", client_id_map)
    logging.info("Client ID map length: %s", len(client_id_map))

    client_id_map[orig_col] = client_id_map[orig_col].astype(str)
    test_client_ids_not_encoded[orig_col] = test_client_ids_not_encoded[orig_col].astype(str)

    logging.info("Test client IDs (not encoded):\n%s", test_client_ids_not_encoded)

    test_client_ids = test_client_ids_not_encoded.merge(client_id_map, on=orig_col, how='inner')[ENCODED_CLIENT_ID_COLUMN]

    logging.info("Test client IDs (encoded):\n%s", test_client_ids)
    logging.info("Test client IDs (encoded) length: %s", len(test_client_ids))

    all_client_ids = client_id_map[ENCODED_CLIENT_ID_COLUMN]

    train_client_ids = all_client_ids[~all_client_ids.isin(test_client_ids)]

    logging.info("Training client IDs:\n%s", train_client_ids)

    return train_client_ids



def get_train_item_ids(args: argparse.Namespace) -> Iterable[int]:
    partitins_paths = [
        os.path.join(args.train_dataset_path, f) 
        for f in os.listdir(args.train_dataset_path) 
        if f.endswith('.parquet')]
    
    if args.include_test_items:
        test_partitins_paths = [
            os.path.join(args.test_dataset_path, f) 
            for f in os.listdir(args.test_dataset_path) 
            if f.endswith('.parquet')]
        partitins_paths.extend(test_partitins_paths)

    train_ids = set()

    for partition_path in tqdm(partitins_paths):
        df = pd.read_parquet(partition_path)
        train_item_id_seqs = df[args.item_id_column]
        for train_item_id_seq in train_item_id_seqs:
            train_ids.update(train_item_id_seq)
    
    return list(train_ids)


def process_tensor_maps(train_ids, orig_tensor_map_path: str,
                        invalid_value: int, out_tensor_map: str, out_dict_map: Optional[str]=None):
    tensor_map = torch.load(orig_tensor_map_path)
    non_train_ids = set(range(len(tensor_map))) - set(train_ids)

    logging.info("Number of Non-training IDs: %s", len(non_train_ids))
    logging.info("%s IDs where tensor_map[id] == 0 and id not in non_train_ids: %s", 
                 orig_tensor_map_path, set(torch.where(tensor_map == 0)[0].cpu().numpy()) - non_train_ids)
    
    assert not torch.any(tensor_map == invalid_value), \
        f"Invalid value {invalid_value} is already present in map: {orig_tensor_map_path}."

    tensor_map[list(non_train_ids)] = invalid_value

    torch.save(tensor_map, out_tensor_map)

    if out_dict_map is not None:
        train_dict = {k: int(tensor_map[k]) for k in train_ids}
        torch.save(train_dict, out_dict_map)


def main():
    args = parse_args()

    process_tensor_maps(
        train_ids=get_train_client_ids(args),
        orig_tensor_map_path=args.orig_client_id2train_graph_id_tensor_path,
        invalid_value=args.invalid_train_graph_id,
        out_tensor_map=args.out_client_id2train_graph_id_tensor_path,
        out_dict_map=args.out_client_id2train_graph_id_dict_path
    )

    logging.info("Finished processing client_id2train_graph_id")

    process_tensor_maps(
        train_ids=get_train_item_ids(args),
        orig_tensor_map_path=args.orig_item_id2train_graph_id_tensor_path,
        invalid_value=args.invalid_train_graph_id,
        out_tensor_map=args.out_item_id2train_graph_id_tensor_path,
        out_dict_map=args.out_item_id2train_graph_id_dict_path
    )

    logging.info("Finished processing item_id2train_graph_id")


if __name__ == "__main__":
    main()
