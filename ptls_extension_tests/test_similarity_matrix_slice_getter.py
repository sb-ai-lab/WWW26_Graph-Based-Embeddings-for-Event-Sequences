import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
######################################################################################################
import sys
import os
import unittest
import logging
import json
from typing import Union

import numpy as np
import scipy.sparse as sp

from ptls_extension_2024_research.sampling_strategies.bin_triplets_utils.similarity_matrix_slice_getter import SimilarityMatrixSliceGetter__FromFeatsBeforeDotProduct


def _convert_test_case_to_list(test_case: dict) -> dict:
    new_test_case = {}
    for k, v in test_case.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        new_test_case[k] = v
    return  new_test_case


class TestSimilarityMatrixSliceGetter__FromFeatsBeforeDotProduct(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)  # Seed for reproducibility
        self.input_save_dir = './failed_slice_getter_inputs'
        self.num_tests = 10  # Number of random index shape tests
        self.matrix_shape = (300, 100)
        self.density = 0.5

        # Generate a single matrix and similarity matrix
        self.feats = sp.random(self.matrix_shape[0], self.matrix_shape[1], density=self.density, format='csr', dtype=np.float32)
        self.similarity_matrix = (self.feats @ self.feats.T).toarray()
        self.slice_getter = SimilarityMatrixSliceGetter__FromFeatsBeforeDotProduct(self.feats)


    def save_failed_test_data(self, row_idxs, col_idxs, test_id: Union[str, int]):
        """Saves the failed test indices to a file for debugging."""
        if not os.path.exists(self.input_save_dir):
            os.makedirs(self.input_save_dir)

        if isinstance(row_idxs, np.ndarray):
            row_idxs = row_idxs.tolist()

        if col_idxs is not None:
            if isinstance(col_idxs, np.ndarray):
                col_idxs = col_idxs.tolist()

        index_data = {
            "row_idxs": row_idxs,
            "col_idxs": col_idxs
        }
        index_path = os.path.join(self.input_save_dir, f"failed_indices_{test_id}.json")
        with open(index_path, 'w') as f:
            json.dump(index_data, f)
        
        logging.error(f"Failed test indices saved for test {test_id}: indices saved to {index_path}")


    def _run_test_case(self, test_case, test_id) -> None:
        row_idxs = test_case["row_idxs"]
        col_idxs = test_case.get("col_idxs", None)

        try:
            if col_idxs is not None:
                slice_result = self.slice_getter[row_idxs, col_idxs]
                expected_slice = self.similarity_matrix[row_idxs, col_idxs]
            else:
                slice_result = self.slice_getter[row_idxs]
                expected_slice = self.similarity_matrix[row_idxs]

            np.testing.assert_array_almost_equal(slice_result, expected_slice)
        except AssertionError:
            self.save_failed_test_data(row_idxs, col_idxs, test_id)
            raise


    def _run_al_test_cases(self, test_cases) -> None:
        for i, test_case in enumerate(test_cases):
            with self.subTest(test_num=i):
                self._run_test_case(test_case, i)

    def generate_bradcastable_indices(self, row_idx_size, col_idx_size, n_matrix_rows):
        row_idxs = np.random.choice(np.arange(n_matrix_rows), size=(row_idx_size, 1), replace=False)
        col_idxs = np.random.choice(np.arange(n_matrix_rows), size=(1, col_idx_size), replace=False)
        return row_idxs, col_idxs


    def test_rows_only_1d_sequence(self):
        test_cases = [
            # Manual test cases
            {"row_idxs": np.array([0, 2])}, 
            {"row_idxs": np.array([10, 15, 50])},
        ]

        # Automatic tests generation
        row_sizes = [1, 4, 16, 32, 128]  # [4**i for i in range(int(self.matrix_shape[0]**0.25))]
        for row_size in row_sizes:
            row_idxs = np.random.choice(np.arange(0, self.matrix_shape[0]), size=row_size, replace=False)  # Random row indices
            test_cases.append({"row_idxs": row_idxs})

        # Numpy
        self._run_al_test_cases(test_cases)
        
        test_cases = [_convert_test_case_to_list(t_case) for t_case in test_cases]
        
        # List
        self._run_al_test_cases(test_cases)


    def test_broadcastable_2d_arrays(self):
        test_cases = [
            # Manual test cases
            {"row_idxs": np.array([[0], [1]]), "col_idxs": np.array([[2, 3]])},
        ]

        # Automatic tests generation
        row_and_col_idx_sizes = [(1, 1), (1, 5), (5, 1), (100, 10)]
        for row_idxs_size, col_idxs_size in row_and_col_idx_sizes:
            row_idxs, col_idxs = self.generate_bradcastable_indices(row_idxs_size, col_idxs_size, self.matrix_shape[0])
            test_cases.append({"row_idxs": row_idxs, "col_idxs": col_idxs})

        self._run_al_test_cases(test_cases)


    def test_row_int_col_1d_sequence(self):
        test_cases = [
            # Manual test cases
            {"row_idxs": 1, "col_idxs": np.array([1, 2, 3])},
        ]

        # Numpy
        self._run_al_test_cases(test_cases)
        
        test_cases = [_convert_test_case_to_list(t_case) for t_case in test_cases]
        
        # List
        self._run_al_test_cases(test_cases)


    def test_row_int_col_1d_bool_array(self):
        similarity_matrix_size = self.matrix_shape[0]
        test_cases = [
            {"row_idxs": 1, "col_idxs": np.ones((similarity_matrix_size,), dtype=bool)},
            {"row_idxs": 1, "col_idxs": np.zeros((similarity_matrix_size,), dtype=bool)}
        ]

        # Automatic tests generation
        n_random_tests = 5
        for _ in range(n_random_tests):
            row_idxs = np.random.randint(low=0, high=similarity_matrix_size-1)
            col_idxs = np.random.randint(low=0, high=1, size=(similarity_matrix_size,), dtype=bool)
            test_cases.append({"row_idxs": row_idxs, "col_idxs": col_idxs})


        self._run_al_test_cases(test_cases)


    def test_row_list_col_int(self):
        test_cases = [
            # Manual test cases
            {"row_idxs": [1, 2, 3], "col_idxs": 1},
        ]

        self._run_al_test_cases(test_cases)


    def test_two_scalars(self):
        max_possible_index = self.matrix_shape[0] - 1

        test_cases = [
            # Manual test cases
            {"row_idxs": 0, "col_idxs": max_possible_index},
            {"row_idxs": max_possible_index, "col_idxs": 0},
            {"row_idxs": 2, "col_idxs": max_possible_index},  # 2 is chosen as an arbitrary "inner" index
            {"row_idxs": max_possible_index, "col_idxs": 2},
            {"row_idxs": 2, "col_idxs": 0},
            {"row_idxs": 0, "col_idxs": 2},
        ]

        # Automatic tests generation
        n_random_tests = 5
        for _ in range(n_random_tests):
            row_idxs = np.random.randint(low=0, high=max_possible_index)
            col_idxs = np.random.randint(low=0, high=max_possible_index)
            test_cases.append({"row_idxs": row_idxs, "col_idxs": col_idxs})

        self._run_al_test_cases(test_cases)


    def test_mismatched_shapes(self):
        """Test that mismatched shapes raise a NotImplementedError."""
        row_idxs = np.array([0, 1, 2])
        col_idxs = np.array([1, 2])
        
        with self.assertRaises(NotImplementedError):
            self.slice_getter[row_idxs, col_idxs]


    def test_rows_only_int(self):
        test_cases = [
            # Manual test cases
            {"row_idxs": 3},
        ]
        self._run_al_test_cases(test_cases)

    def test_two_axis_slice(self):
        test_cases = [
            # Manual test cases
            {"row_idxs": np.array([0, 1]), "col_idxs": np.array([1, 2])},
        ]
        self._run_al_test_cases(test_cases)



if __name__ == '__main__':
    unittest.main()
