import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from ptls_extension_2024_research.frames.gnn.gnn_module import GnnLinkPredictor
import dgl

class TestGnnLinkPredictor(unittest.TestCase):
    def setUp(self):
        self.n_users = 10
        self.n_items = 10
        self.embedding_dim = 64
        self.output_size = 10
        self.model = GnnLinkPredictor(
            n_users=self.n_users,
            n_items=self.n_items,
            output_size=self.output_size,
            embedding_dim=self.embedding_dim
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def test_embedding_matrixes_updated_properly(self, batch):
        # Simulate a forward pass
        user_ids = torch.LongTensor([0, 1, 2])
        item_ids = torch.LongTensor([0, 1, 2])
        batch = (item_ids, user_ids)

        # Create a dummy subgraph (assuming ClientItemGraph is properly defined)
        subgraph = self.model.get_subgraph(batch)

        # Get the initial weights
        initial_client_weights = self.model.client_feats.weight.clone().detach()
        initial_item_weights = self.model.item_feats.weight.clone().detach()
        initial_node_weights = self.model.node_feats.weight.clone().detach()

        # Forward pass
        output = self.model(subgraph)
        dummy_target = torch.randn_like(output)  # Dummy target for illustration
        loss = self.criterion(output, dummy_target)

        # Backward pass
        loss.backward()

        # Optimization step
        self.optimizer.step()

        # Check that the weights have been updated
        self.assertFalse(torch.equal(initial_client_weights, self.model.client_feats.weight))
        self.assertFalse(torch.equal(initial_item_weights, self.model.item_feats.weight))
        self.assertFalse(torch.equal(initial_node_weights, self.model.node_feats.weight))

        # Check that the client and item weights are still parts of the node weights
        updated_client_weights = self.model.node_feats.weight[:self.n_users]
        updated_item_weights = self.model.node_feats.weight[self.n_users:]
        
        self.assertTrue(torch.equal(self.model.client_feats.weight, updated_client_weights))
        self.assertTrue(torch.equal(self.model.item_feats.weight, updated_item_weights))

if __name__ == '__main__':
    unittest.main()
