import os

import numpy as np
import torch
import networkx as nx


DATA_PATH = './data/graphs/weighted'
N_USERS = 14344-184
N_ITEMS = 184
OUT_PATH = './data/graphs/weighted/adj_embs.pt'


def main():
    g = nx.read_gml(os.path.join(DATA_PATH,'train_graph.gml'))
    nx.adjacency_matrix(g)


    cols = np.arange(N_ITEMS) + N_USERS

    graph_emb_matr = nx.adjacency_matrix(g).todense()
    graph_adj_embs = graph_emb_matr[:, cols]
    graph_adj_embs_tensor = torch.tensor(graph_adj_embs)
    torch.save(graph_adj_embs_tensor, OUT_PATH)


if __name__ == '__main__':
    main()
