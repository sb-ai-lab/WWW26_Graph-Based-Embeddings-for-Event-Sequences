from typing import List, Union, Tuple
import warnings

import numpy as np
import scipy

from .cache_iterable import np_like_cache_of_size_one

SingleIterableIdxsType = Union[List[int], np.ndarray]
SingleAxisIdxsType =  Union[SingleIterableIdxsType, int, np.integer]
IdxsType = Union[Tuple[SingleAxisIdxsType, SingleAxisIdxsType], SingleAxisIdxsType]

class SimilarityMatrixSliceGetterBase:
    """
    Retrieves similarity matrix slices without necessity to store the full matrix in memory.
    """
    
    def _get_similarity_matrix_slice(self, idxs: IdxsType) -> np.ndarray:
        raise NotImplementedError
    
    def _check_single_axis_idxs_type(self, idxs: SingleAxisIdxsType, axis_name: str = 'row') -> None:
        type_is_correct = ((type(idxs) == list) 
                           or isinstance(idxs, np.ndarray) 
                           or isinstance(idxs, int) 
                           or isinstance(idxs, np.integer))
        if not type_is_correct:
            warning_message = f"Expected {axis_name} idxs to be either " \
                f"a numpy array, a list or an int. Got unexpected type `{type(idxs)}`"
            warnings.warn(warning_message)
            

    def _check_two_axis_idxs_type(self, row_idxs_and_col_idxs: Tuple[SingleAxisIdxsType, SingleAxisIdxsType]) -> None:
        if not isinstance(row_idxs_and_col_idxs, tuple):
            raise TypeError("Wrong format. Expected format: " \
            "`sm_slice_getter[row_idxs, col_idxs]`. " \
            f"Expected a tuple, got type `{type(row_idxs_and_col_idxs)}`")
        
        if len(row_idxs_and_col_idxs) != 2:
            "Expected one or two axis idxs (`sm_slice_getter[row_idxs]` or `sm_slice_getter[row_idxs, col_idxs]`). \n" \
            f"Num axis idxs recieved: {len(row_idxs_and_col_idxs)}"

        row_idxs, col_idxs = row_idxs_and_col_idxs

        self._check_single_axis_idxs_type(row_idxs, 'row')
        self._check_single_axis_idxs_type(col_idxs, 'col')


    def _check_preconditions(self, idxs: IdxsType) -> None:
        if isinstance(idxs, tuple):
            self._check_two_axis_idxs_type(idxs)
        else:
            self._check_single_axis_idxs_type(idxs)

    @property
    def shape(self) -> Tuple[int, int]:
        raise NotImplementedError
        
        
    def __getitem__(self, idxs: IdxsType) -> np.ndarray:
        """
        Retrieves a slice of the similarity matrix corresponding to the provided row and column indices. 

        Is used to access a slice of a similarity matrix as if the entire matrix were available (same interface). 
        However, it enforces restrictions compared to direct access on a full matrix.

        **Restrictions:**
        1. The matrix slice can only be retrieved in the following formats: 
            * `sm_slice_getter[row_idxs, col_idxs]`.
            * `sm_slice_getter[row_idxs]`.
        2. `row_idxs` and `col_idxs` must be of type `List[int]` or `np.ndarray` or `int`.
        3. There might be additional restrictions. See `_get_similarity_matrix_slice` docstring for details.

        **Example:**
        similarity_matrix_slice_getter = SimilarityMatrixsliceGetterBase(*args, **kwargs)

        row_idxs = np.array([[3], [5], [6]])
        col_idxs = np.array([[1, 4])
        similarity_matrix_slice = similarity_matrix_slice_getter[row_idxs, col_idxs]

        # The operation above is equivalent to the one below on the full similarity matrix loaded in memory.
        actual_similarity_matrix = np.load('similarity_matrix.npy')
        similarity_matrix_slice = actual_similarity_matrix[row_idxs, col_idxs]

        
        Arguments
        ---------
        idxs: IdxsType
            A tuple of two axis indices or a single axis index
            
        Returns
        -------
        np.ndarray
            A slice of the similarity matrix corresponding to the given indices.
        """
        self._check_preconditions(idxs)
        return self._get_similarity_matrix_slice(idxs)


class SimilarityMatrixSliceGetter__FromFeatsBeforeDotProduct(SimilarityMatrixSliceGetterBase):
    def __init__(self, feats: scipy.sparse.csr_matrix) -> None:
        self.feats = feats

    def _slice_from_two_1d_iterables(self, row_idxs: SingleIterableIdxsType, col_idxs: SingleIterableIdxsType) -> np.ndarray: 
        # Alternatively we can do dot product once self.feats[row_idxs] @ self.feats[col_idxs].T,
        # and then take proper indexes from the result.
        assert len(row_idxs) == len(col_idxs)
        prod = np.zeros(len(row_idxs))
        for i, (l_i, r_i) in enumerate(zip(row_idxs, col_idxs)):
            prod[i] = (self.feats[l_i] @ self.feats[r_i].T).toarray()
        return prod 

    # @np_like_cache_of_size_one
    def _get_similarity_matrix_slice(self, idxs: IdxsType) -> np.ndarray:
        """
        Retrieves a slice of the similarity matrix corresponding to the provided row and column indices.

        **Restrictions:**
        1. The matrix slice can only be retrieved in the following formats: 
            * `sm_slice_getter[row_idxs, col_idxs]`.
            * `sm_slice_getter[row_idxs]`.
        2. `row_idxs` and `col_idxs` must be of type `List[int]` or `np.ndarray` or `int`.
        
        In case `sm_slice_getter[row_idxs]` all argument types valid for an actual numpy array are allowed.
        Allowed arguments types for `sm_slice_getter[row_idxs, col_idxs]` are:
            * `row_idxs`: int, `col_idxs`: anything valid for a numpy array.
            * `row_idxs`: anything valid for a numpy array, `col_idxs`: int
            * `row_idxs`: np.ndarray of shape (n, 1), `col_idxs`: np.ndarray of shape (1, m), 
                where `n` and `m` are any integers less than the number of rows in the similarity matrix. 
            * `row_idxs`: 1d array or List[int], `col_idxs`: 1d array or List[int]; `len(row_idxs) == len(col_idxs)`
        """
        int_classes = (int, np.integer)

        # If format is `sm_slice_getter[row_idxs]`
        if type(idxs) != tuple:
            row_idxs = idxs
            prod = self.feats[row_idxs] @ self.feats.T
            prod = prod.toarray()
            if isinstance(idxs, int_classes):
                prod = prod.squeeze()
            return prod 
        
        # If format is `sm_slice_getter[row_idxs, col_idxs]`
        row_idxs, col_idxs = idxs

        # Condition definitons
        has_scalar_idx = isinstance(row_idxs, int_classes) or isinstance(col_idxs, int_classes)
        are_broadcastable_2d_arrays = (
            (type(row_idxs) == np.ndarray) 
            and (type(col_idxs) == np.ndarray)
            and len(row_idxs.shape) == 2 
            and len(col_idxs.shape) == 2 
            and row_idxs.shape[1] == 1 
            and col_idxs.shape[0] == 1
        )
        check_is_1d_iterable = lambda axis_idxs: (type(axis_idxs) == list) or ((type(axis_idxs) == np.ndarray) and (len(axis_idxs.shape) == 1))
        are_1d_iterables_of_same_shape = check_is_1d_iterable(row_idxs) and check_is_1d_iterable(col_idxs) and len(row_idxs) == len(col_idxs) 
        check_is_int_and_1d_array = (lambda x, y:
                                    isinstance(x, int_classes) and check_is_1d_iterable(y)
                                    or isinstance(y, int_classes) and check_is_1d_iterable(x))

        if has_scalar_idx or are_broadcastable_2d_arrays:
            if are_broadcastable_2d_arrays:
                row_idxs = row_idxs.squeeze()
                col_idxs = col_idxs.squeeze() 
            left = self.feats[row_idxs]
            right = self.feats[col_idxs]
            prod = left @ right.T
            prod = prod.toarray()
            if check_is_int_and_1d_array(row_idxs, col_idxs):
                # When indexing via an int and a 1d sequence, 
                # a numpy array returns a slice of shape `(len(sequence_1d),)`.
                # Note:
                # If x is a 2d sparse matrix `x[some_int]@x[sequence_1d]`
                # has shape `(1, len(sequence_1d))`. If x is a np.ndarray,
                # the output has shape `(len(sequence_1d),)`.
                prod = prod.squeeze()
            return prod
        elif are_1d_iterables_of_same_shape:
            return self._slice_from_two_1d_iterables(row_idxs, col_idxs)
        raise NotImplementedError("Unexpected types of indexes are met. See docstring for allowed types.")
    
    @property
    def shape(self) -> Tuple[int, int]:
        n_users = self.feats.shape[0]
        return n_users, n_users
