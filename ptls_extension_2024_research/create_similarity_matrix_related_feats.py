from typing import Tuple
import sys
import logging
import argparse
import os

import numpy as np
import torch
import dgl
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir_path', type=str, required=True)
    parser.add_argument('--min_max_iterative_batch_size', type=int, default=None)
    parser.add_argument('--save_normalized_sparse_adj_embs', action='store_true')
    parser.add_argument('--save_similarity_matrix', action='store_true')
    parser.add_argument('--sparse_similarity_matrix', action='store_true')
    parser.add_argument('--save_min_max_array', action='store_true')
    parser.add_argument('--compress_npz', action='store_true')
    return parser.parse_args()


def load_graph(graph_file_path) -> dgl.DGLGraph:
    (g,), _ = dgl.load_graphs(graph_file_path, [0])
    return g


def l2_normalize_sparse_inplace(mat) -> scipy.sparse.csr_matrix:
    """
    Normalize rows of a sparse matrix by their L2 norm.
    """
    logging.info(f"Matrix to normalize shape: {mat.shape}")
    row_norms = scipy.sparse.linalg.norm(mat, axis=1)  # L2 norm of each ROW
    logging.info(f"row_norms shape: {row_norms.shape}")
    row_indices, col_indices = mat.nonzero()
    logging.info(f"matrix_to_normalize.data.shape {mat.data.shape}")
    mat.data /= row_norms[row_indices]
    return mat


def get_n_users_and_n_items(num_nodes: int, args) -> Tuple[int, int]:
    graph_dir_path = args.graph_dir_path

    ptls_client_id_to_train_graph_node_id_dict = torch.load(os.path.join(graph_dir_path, 'client_id2train_graph_id__dict.pt'))
    ptls_item_id_to_train_graph_node_id_dict = torch.load(os.path.join(graph_dir_path, 'item_id2train_graph_id__dict.pt'))

    n_users = len(ptls_client_id_to_train_graph_node_id_dict)
    n_items = len(ptls_item_id_to_train_graph_node_id_dict)

    if num_nodes != n_users + n_items:
        logging.warning(f"num_nodes is expected to be equal to n_users + n_items. " \
            f"Nodes number: {num_nodes}, users number: {n_users}, items number: {n_items}. " \
            "This could have happend if n_users and n_items corespond to ptls dataset with " \
            "truncated sequences and graph corresponds to original data.")
        logging.info(f"Setting num_items to num_nodes - num_users")
        n_items = num_nodes - n_users
    return n_users, n_items


def create_min_max_array_from_sparse_normalized_feats(normalized_sparse_adj_embs, args) -> np.ndarray:
    n_users = normalized_sparse_adj_embs.shape[0]
    min_max_array = np.empty((n_users, 2))
    batch_size = args.min_max_iterative_batch_size
    
    for batch_start in tqdm(range(0, n_users, batch_size), total=(n_users // batch_size) + 1):
        batch_end = min(batch_start + batch_size, n_users)
        current_batch = normalized_sparse_adj_embs[batch_start:batch_end]

        similarities = current_batch @ normalized_sparse_adj_embs.T

        for i in range(batch_start, batch_end):
            similarities[i - batch_start, i] = np.inf
        
        min_max_array[batch_start:batch_end, 0] = np.min(similarities, axis=1).toarray().squeeze(axis=1)

        for i in range(batch_start, batch_end):
            similarities[i - batch_start, i] = -np.inf

        min_max_array[batch_start:batch_end, 1] = np.max(similarities, axis=1).toarray().squeeze(axis=1)
    
    return min_max_array



def main():
    args = parse_args()
    graph_dir_path = args.graph_dir_path
    G = load_graph(os.path.join(graph_dir_path, 'train_graph.bin'))

    num_nodes = G.num_nodes()
    logging.info(f"Number of nodes: {num_nodes}")

    n_users, n_items = get_n_users_and_n_items(num_nodes, args)

    logging.info(f"Number of items: {n_items}")
    logging.info(f"Number of clients: {n_users}")

    rows = np.arange(n_users)
    cols = np.arange(n_items) + n_users

    graph_emb_matr = G.adj_external(scipy_fmt="csr")
    logging.info(f"Loaded adjacency_matrix")
    logging.info(f"adjacency_matrix's shape: {graph_emb_matr.shape}")

    graph_adj_embs = graph_emb_matr[rows, :][:, cols].astype(np.float32)
    logging.info(f"Extracted graph_adj_embs")
    logging.info(f"graph_adj_embs's shape: {graph_adj_embs.shape}")
    logging.info(f"type(graph_adj_embs): {type(graph_adj_embs)}")

    logging.info(f"{graph_adj_embs.dtype = }")
    
    l2_normalize_sparse_inplace(graph_adj_embs)
    normalized_sparse_adj_embs = graph_adj_embs  # graph_adj_embs was changed inplace

    logging.info("Normalized graph_adj_embs")

    if args.save_normalized_sparse_adj_embs:
        normalized_sparse_adj_embs_path = os.path.join(graph_dir_path, "graph_adj_embs_normalized.npz")
        scipy.sparse.save_npz(normalized_sparse_adj_embs_path, normalized_sparse_adj_embs, compressed=args.compress_npz)
        logging.info(f"Saved normalized_sparse_adj_embs to file {normalized_sparse_adj_embs_path}")

    if args.save_min_max_array and not args.save_similarity_matrix:
        # probably if we don't save_similarity_matrix it's too big and we
        # can't compute min_max_array from it.
        min_max_array = create_min_max_array_from_sparse_normalized_feats(normalized_sparse_adj_embs, args)
        min_max_array_path = os.path.join(graph_dir_path, "min_max_array.npy")
        np.save(min_max_array_path, min_max_array)
        logging.info(f"Saved min_max_arry to file: {min_max_array_path}")


    if args.save_similarity_matrix:
        if not args.sparse_similarity_matrix:
            graph_adj_embs = graph_adj_embs.toarray()
        similarity_matrix = graph_adj_embs @ graph_adj_embs.T

        # similarity_matrix = cosine_similarity(graph_adj_embs, dense_output=False)

        logging.info(f"Calculated similarity matrix")
        logging.info(f"sys.getsizeof(similarity_matrix) = {sys.getsizeof(similarity_matrix)}")
        logging.info(f"Similarity matrix shape: {similarity_matrix.shape}")
        if args.sparse_similarity_matrix:
            similarity_matrix_save_path = os.path.join(graph_dir_path, "similarity_matrix.npz")
            scipy.sparse.save_npz(similarity_matrix_save_path, similarity_matrix)
        else:
            similarity_matrix_save_path = os.path.join(graph_dir_path, "similarity_matrix.npy")
            np.save(similarity_matrix_save_path, similarity_matrix)
        logging.info(f"Similarity matrix saved to file {similarity_matrix_save_path}")            

        if args.save_min_max_array:
            min_max_array = np.empty((similarity_matrix.shape[0], 2))
            min_max_array[:, 0] = np.min(similarity_matrix, axis=1)
            min_max_array[:, 1] = np.max(similarity_matrix, axis=1)
            min_max_array_path = os.path.join(graph_dir_path, "min_max_array.npy")
            np.save(min_max_array_path, min_max_array)
            logging.info(f"Saved min_max_arry to file: {min_max_array_path}")




if __name__ == '__main__':
    main()
