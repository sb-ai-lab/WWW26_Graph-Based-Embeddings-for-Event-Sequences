import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
######################################################################################################
import unittest

import numpy as np
import torch

from ptls_extension_2024_research.sampling_strategies.bin_triplets_utils.bin_separation_strategies import get_max_distance_idx__sequence, get_max_distance_idx__numpy, get_max_distance_idx__tensor, sparate_bin_on_max_distance, MaxDistanceBinSeparationStrategy
from ptls_extension_2024_research.sampling_strategies.bin_triplets_utils.id_converter import IdConverter_Dict


class TestMaxDistance(unittest.TestCase):


    common_options = [
        {
            'sorted_arr': [1, 3, 201, 202, 203, 204], 'margin': 2, 'expected': 2, 
            'description': 'Max difference is on the first element of search space'
        },
        {
            'sorted_arr': [1, 3, 4, 6, 203, 204], 'margin': 2, 'expected': 4, 
            'description': 'Max difference is on the last element of search space'
        },
        {
            'sorted_arr': [1, 2, 5, 6, 9, 15, 16], 'margin': 1, 'expected': 5, 
            'description': 'One of two tests with same `sorted_arr`, but different `margin` and same expected output'
        },
        {
            'sorted_arr': [1, 2, 5, 6, 9, 15, 16], 'margin': 2, 'expected': 5, 
            'description': 'One of two tests with same `sorted_arr`, but different `margin` and same expected output'
        },
        {
            'sorted_arr': [1, 2, 5, 6, 9, 15, 100], 'margin': 1, 'expected': 6, 
            'description': 'One of two tests with same `sorted_arr`, but different `margin` and different expected output'
        },
        {
            'sorted_arr': [1, 2, 5, 6, 9, 15, 100], 'margin': 2, 'expected': 5, 
            'description': 'One of two tests with same `sorted_arr`, but different `margin` and different expected output'
        },
        {
            'sorted_arr': [0, 1, 2, 10, 11, 20], 'margin': 1, 'expected': 5, 
            'description': 'One of two tests with same `sorted_arr`, different `margin` and different expected output'
        },
        {
            'sorted_arr': [0, 1, 2, 10, 11, 20], 'margin': 2, 'expected': 3, 
            'description': 'One of two tests with same `sorted_arr`, different `margin` and different expected output'
        },
        {
            'sorted_arr': [5, 10, 20, 25], 'margin': 1, 'expected': 2, 
            'description': 'Ordinary test.'},
        {
            'sorted_arr': [5, 19, 20, 25], 'margin': 2, 'expected': 2, 
            'description': 'The separation must be on the lowest difference in array due to len(arr) = 2*margin.'
        },
        {
            'sorted_arr': [5, 20, 25, 26, 40, 41], 'margin': 2, 'expected': 4, 
            'description': 'Largest difference is at index 1, but should be ignored due to margin.'
        },

         # Edge cases:
        {
            'sorted_arr': [5, 5, 5, 5, 5, 5, 5], 'margin': 1, 'expected': None, 
            'description': 'All elements are the same, so no meaningful difference exists.'
        },
        {
            'sorted_arr': [1, 2, 3, 4], 'margin': 2, 'expected': 2, 
            'description': 'Array length is exactly 2 * margin, the function should return margin.'
        },
        {
            'sorted_arr': [1, 1, 1, 1, 1, 1, 1, 6], 'margin': 2, 'expected': 7, 
            'description': 'The function should expand the search space when all elements within the original search space are constant.'
        },
        {
            'sorted_arr': [1, 2, 2, 2, 2, 2, 2, 2], 'margin': 2, 'expected': 1, 
            'description': 'The function should expand the search space when all elements within the original search space are constant.'
        },

        # Real examples
        {
            'sorted_arr': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.08276059], 'margin': 5, 'expected': 38, 
            'description': 'Real example. The function should expand the search space when all elements within the original search space are constant.'
        },
        # Real examples
        {
            'sorted_arr':[0.28937158, 0.29197866, 0.29350016, 0.29488388, 0.30645236, 0.31118804, 0.32302913, 0.32407215, 0.32850462, 0.33514008, 0.33825514, 0.34698898, 0.35037434, 0.36115748], 'margin': 5, 'expected': 6, 
            'description': 'Real example. The function should expand the search space when all elements within the original search space are constant.'
        },

    ]


    def test_get_max_distance_idx__sequence(self):
        for options in self.common_options:
            sorted_arr = options['sorted_arr']
            margin = options['margin']
            expected = options['expected']
            actual = get_max_distance_idx__sequence(sorted_arr, margin)
            self.assertEqual(actual, expected,
                             f"Decription: {options.get('description', 'no description provided')}\n"
                             f"Test failed for input: sorted_arr={sorted_arr}, margin={margin}. "
                             f"Expected: {expected}, but got: {actual}")

    def test_get_max_distance_idx__tensor(self):
        for options in self.common_options:
            sorted_arr = torch.tensor(options['sorted_arr'], dtype=torch.float)
            margin = options['margin']
            expected = options['expected']
            actual = get_max_distance_idx__tensor(sorted_arr, margin)
            self.assertEqual(actual, expected,
                             f"Decription: {options.get('description', 'no description provided')}\n"
                             f"Test failed for input: sorted_arr={sorted_arr}, margin={margin}. "
                             f"Expected: {expected}, but got: {actual}")
    
    def test_get_max_distance_idx__numpy(self):
        for options in self.common_options:
            sorted_arr = np.array(options['sorted_arr'], dtype=float)
            margin = options['margin']
            expected = options['expected']
            actual = get_max_distance_idx__numpy(sorted_arr, margin)
            self.assertEqual(actual, expected,
                             f"Decription: {options.get('description', 'no description provided')}\n"
                             f"Test failed for input: sorted_arr={sorted_arr}, margin={margin}. "
                             f"Expected: {expected}, but got: {actual}")


class TestSeparateBin(unittest.TestCase):

    def test_sparate_bin_on_max_distance(self):
        user_ids_bin = [101, 102, 103, 104, 105, 106]
        similarities = np.array([0.1, 0.2, 0.5, 0.6, 0.9, 1.5])
        min_elements_in_bin = 2
        bin_1, bin_2 = sparate_bin_on_max_distance(user_ids_bin, similarities, min_elements_in_bin)

        self.assertEqual(bin_1, [101, 102, 103, 104])
        self.assertEqual(bin_2, [105, 106])

        user_ids_bin = [201, 202, 203, 204]
        similarities = np.array([0.1, 0.5, 1.0, 1.1])
        min_elements_in_bin = 2
        bin_1, bin_2 = sparate_bin_on_max_distance(user_ids_bin, similarities, min_elements_in_bin)

        self.assertEqual(bin_1, [201, 202])
        self.assertEqual(bin_2, [203, 204])

        # Real example
        user_ids_bin = [295808, 204705, 33314, 71588, 110020, 278308, 7052, 130222, 250031, 196912, 92337, 94518, 201527, 350013]
        similarities = np.array([0.29197866, 0.34698898, 0.35037434, 0.28937158, 0.32850462,
            0.32302913, 0.33825514, 0.29350016, 0.31118804, 0.29488388,
            0.30645236, 0.33514008, 0.32407215, 0.36115748], dtype=np.float32)
        min_elements_in_bin = 5
        bin_1, bin_2 = sparate_bin_on_max_distance(user_ids_bin, similarities, min_elements_in_bin)

        self.assertEqual(bin_1, [71588, 295808, 130222, 196912, 92337, 250031])
        self.assertEqual(bin_2, [278308, 201527, 110020, 94518, 7052, 204705, 33314, 350013])



class TestMaxDistanceBinSeparationStrategy(unittest.TestCase):
    def setUp(self):
        n_users = 5
        internal_to_external = {i: i for i in range(n_users)}
        self.id_converter = IdConverter_Dict(internal_to_external)

        # symmetric matrix
        self.similarity_matrix = np.array([
            [1.0, 0.9, 0.7, 0.2, 0.3],
            [0.9, 1.0, 0.4, 0.1, 0.2],
            [0.7, 0.4, 1.0, 0.7, 0.5],
            [0.2, 0.1, 0.7, 1.0, 0.2],
            [0.3, 0.2, 0.5, 0.2, 1.0]
        ])

        self.strategy = MaxDistanceBinSeparationStrategy(
            min_elements_in_bin=2,
            similarity_matrix_slice_getter=self.similarity_matrix,
            user_id_converter=self.id_converter
        )

    def test_separate_bins(self):
        user_ids_bin = np.array([1, 2, 3, 4])
        anchor_user_id = 0

        bin_1, bin_2 = self.strategy(user_ids_bin, anchor_user_id)
        self.assertEqual(bin_1, [3, 4])
        self.assertEqual(bin_2, [2, 1])


if __name__ == '__main__':
    unittest.main()
