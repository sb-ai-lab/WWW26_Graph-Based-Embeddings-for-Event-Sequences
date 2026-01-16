import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import torch
import torch.nn as nn
import torch.optim as optim

from ptls_extension_2024_research.frames.gnn.two_part_embedding import TwoPartEmbedding


class TestTwoPartEmbedding(unittest.TestCase):
    def setUp(self):
        num_embeddings_1 = 10
        num_embeddings_2 = 20
        embedding_dim = 5
        self.embed_1 = nn.Embedding(num_embeddings_1, embedding_dim)
        self.embed_2 = nn.Embedding(num_embeddings_2, embedding_dim)
        self.model = TwoPartEmbedding(self.embed_1, self.embed_2)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def test_gradients_updated(self):
        indices = torch.tensor([2, 5, 12, 15], dtype=torch.long)
        output = self.model(indices)
        
        loss = output.sum()

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        embed_1_grad_updated = self.embed_1.weight.grad is not None and self.embed_1.weight.grad.abs().sum() > 0
        embed_2_grad_updated = self.embed_2.weight.grad is not None and self.embed_2.weight.grad.abs().sum() > 0

        self.assertTrue(embed_1_grad_updated, "embed_1's weights were not updated.")
        self.assertTrue(embed_2_grad_updated, "embed_2's weights were not updated.")


if __name__ == '__main__':
    unittest.main()
