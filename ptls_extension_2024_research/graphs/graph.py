from typing import Optional

from dataclasses import dataclass
import dgl
import torch


@dataclass
class ClientItemGraph:
    """
    A class to represent a graph with clients and items.
    Given a list of client_ids and item_ids, it creates 
    a subgraph with the clients, items, and their neighbors.
    """
    g: dgl.DGLGraph
    device_name: Optional[str]

    def __post_init__(self):
        self.device = None
        if self.device_name is not None:
            self.device = torch.device(self.device_name)

    def create_subgraph(self,
                        client_ids: torch.Tensor,
                        item_ids: torch.Tensor, 
                        order: int) -> dgl.DGLGraph:
        """
        Creates a subgraph with neighbors of up to the specified order.
        
        Parameters:
        - client_ids: Tensor of client node IDs
        - item_ids: Tensor of item node IDs
        - order: int, the number of neighbor orders to consider

        Returns:
        - subgraph: DGLGraph, the induced subgraph containing 
            the nodes of interest and their neighbors up to the specified order.
        """
        item_ids = torch.flatten(item_ids)

        nodes_of_interest = torch.cat([client_ids, item_ids])

        for _ in range(order):
            in_neighbors = self.g.in_edges(nodes_of_interest)[0].unique()
            out_neighbors = self.g.out_edges(nodes_of_interest)[1].unique()

            nodes_of_interest = torch.cat([nodes_of_interest, in_neighbors, out_neighbors]).unique()

        nodes_of_interest = torch.sort(nodes_of_interest)

        subgraph = dgl.node_subgraph(self.g, nodes_of_interest)

        if self.device is not None:
            subgraph = subgraph.to(self.device)

        return subgraph

    @classmethod
    def from_graph_file(cls, graph_file_path: str, device_name: Optional[str] = None):
        g_list, _ = dgl.load_graphs(graph_file_path, [0])
        g = g_list[0]
        if device_name is not None:
            g = g.to(torch.device(device_name))
        return cls(g, device_name)


@dataclass
class ClientItemGraphFull(ClientItemGraph):
    """
    A special case of the ClientItemGraph where the subgraph 
    is the same as the original full graph.
    """
    def __post_init__(self):
        super().__post_init__()
        self.g.ndata['_ID'] = torch.arange(0, self.g.number_of_nodes(), device=self.device)

    def create_subgraph(self, client_ids: torch.Tensor, item_ids: torch.Tensor) -> dgl.DGLGraph:
        return self.g