import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
######################################################################################################
import unittest
import logging
import json
import os

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from ptls_extension_2024_research.create_similarity_matrix_related_feats import l2_normalize_sparse_inplace


class TestCosineSimilarity(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.num_tests = 15
        self.size = (100, 100)
        self.density = 0.5
        self.input_save_dir = './ptls_extension_tests/failed_cosine_similarity_inputs'

    def generate_random_sparse_matrix(self):
        matrix = sp.random(self.size[0], self.size[1], density=self.density, format='csr', dtype=np.float32)
        return matrix

    def save_matrix_to_file(self, matrix, test_num):
        if not os.path.exists(self.input_save_dir):
            os.makedirs(self.input_save_dir)
        file_path = os.path.join(self.input_save_dir, f"failed_input_{test_num}.json")
        sp.save_npz(file_path, matrix)
        logging.error(f"Input matrix saved to {file_path} for test {test_num}")

    def test_cosine_similarity_equivalence(self):
        for i in range(self.num_tests):
            with self.subTest(i=i):
                csr_sparse_feats = self.generate_random_sparse_matrix()
                sklearn_result = cosine_similarity(csr_sparse_feats, dense_output=False)

                l2_normalize_sparse_inplace(csr_sparse_feats)
                my_result = csr_sparse_feats @ csr_sparse_feats.T

                all_close = np.allclose(sklearn_result.toarray(), my_result.toarray(), rtol=1e-5, atol=1e-8)

                if not all_close:
                    self.save_matrix_to_file(csr_sparse_feats, i)
                self.assertTrue(all_close, f"Test failed for matrix {i}")


if __name__ == '__main__':
    unittest.main()
