# Author: Hongwei Zhang
# Email: hw_zhang@outlook.com

import mmh3
import numpy as np
from scipy.sparse import coo_matrix
from abc import ABC, abstractmethod


class Encoder(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass


class FFMHashEncoder(Encoder):
    def __init__(self, hash_base=1000000, hash_offset=100, seed=0):
        self._counter = dict()
        self._hash_base = hash_base
        self._hash_offset = hash_offset
        self._seed = seed

    def fit(self, X):
        """
        Parameters
        ----------
        X: [[field1_index1, filed1_index2, field2_index1],
            [field1_index3, field3_index1]]

        Returns
        -------
        self
        """
        for x in X:
            for word in x:
                try:
                    self._counter[word] += 1
                except KeyError:
                    self._counter[word] = 1

        return self

    def transform(self, X, threshold=20, dummy_field=False):
        """
        unpresented word will be dropped

        Parameters
        ----------
        X: [[field1_index1, filed1_index2, field2_index1],
            [field1_index3, field3_index1]]

        threshold: words that occur less than threshold will be dropped

        dummy_field: in order to follow input format of ffm,
                     use the same dummy field for all features

        Returns
        -------
        FFM format data: [[filed1:hash(filed1_index1):1,
                           filed1:hash(filed1_index2):1,
                           filed2:hash(filed2_index1):1],
                          [filed1:hash(field1_index3):1,
                           filed3:hash(field3_index1):1]
        """
        result = list()
        for x in X:
            row_result = list()
            for word in x:
                if word in self._counter and self._counter[word] >= threshold:
                    field = "0"
                    if dummy_field is False:
                        field = word.split("_")[0]

                    index = str(
                            mmh3.hash(
                                word,
                                self._seed, signed=False) % self._hash_base +
                            self._hash_offset)

                    row_result.append(":".join([field, index, "1"]))

            result.append(row_result)

        return result


class OneHotEncoder(Encoder):
    def __init__(self):
        self._encoder = dict()
        self._counter = dict()

    def fit(self, X, threshold=3):
        """
        Parameters
        ----------
        X: [[field1_index1, filed1_index2, field2_index1],
            [field1_index3, field3_index1]]

        threshold: words that occur less than threshold will be dropped

        Returns
        -------
        self
        """
        for x in X:
            for word in x:
                try:
                    self._counter[word] += 1
                except KeyError:
                    self._counter[word] = 1

        first_index = 0
        for key, value in self._counter.items():
            if value >= threshold:
                self._encoder[key] = first_index
                first_index += 1

        return self

    def transform(self, X):
        """
        unpresented word will be dropped

        Parameters
        ----------
        X: [[field1_index1, filed1_index2, field2_index1],
            [field1_index3, field3_index1]]

        Returns
        -------
        csr matrix
        """
        nrows = len(X)
        ncols = len(self._encoder)

        row_indices = list()
        col_indices = list()
        data = list()
        for row in range(nrows):
            x = X[row]
            for word in x:
                if word in self._encoder:
                    col = self._encoder[word]
                    row_indices.append(row)
                    col_indices.append(col)
                    data.append(1)

        return coo_matrix((
            np.array(data),
            (np.array(row_indices), np.array(col_indices))),
            shape=(nrows, ncols)).tocsr()

    @property
    def codebook(self):
        return self._encoder


class SVMHashEncoder(Encoder):
    def __init__(self, hash_base=100000, hash_offset=100, seed=0):
        self._counter = dict()
        self._hash_base = hash_base
        self._hash_offset = hash_offset
        self._seed = seed

    def fit(self, X):
        """
        Parameters
        ----------
        X: [[field1_index1, filed1_index2, field2_index1],
            [field1_index3, field3_index1]]

        Returns
        -------
        self
        """
        for x in X:
            for word in x:
                try:
                    self._counter[word] += 1
                except KeyError:
                    self._counter[word] = 1

        return self

    def transform(self, X, threshold=3):
        """
        unpresented word will be dropped
        index is not in ascending order!!!

        Parameters
        ----------
        X: [[field1_index1, filed1_index2, field2_index1],
            [field1_index3, field3_index1]]

        threshold: words that occur less than threshold will be dropped

        Returns
        -------
        LibSVM format data: [[hash(filed1_index1):1,
                              hash(filed1_index2):1,
                              hash(filed2_index1):1],
                             [hash(filed1_index3):1,
                              hash(filed3_index1):1]]
        """
        result = list()
        for x in X:
            row_result = list()
            for word in x:
                if word in self._counter and self._counter[word] >= threshold:
                    index = str(mmh3.hash(
                        word, self._seed, signed=False) % self._hash_base +
                        self._hash_offset)
                    row_result.append(":".join([index, "1"]))

            result.append(row_result)

        return result


class BinaryStatsCalculator(Encoder):
    def __init__(self):
        self._encoder = dict()
        self._counter = dict()

    def fit(self, X, y, threshold=3):
        """
        Parameters
        ----------
        X: [[field1_index1, filed1_index2, field2_index1],
            [field1_index3, field3_index1]]

        y: [0, 1]

        threshold: words that occur less than threshold will be dropped

        Returns
        -------
        self
        """
        for x in X:
            for word in x:
                try:
                    self._counter[word] += 1
                except KeyError:
                    self._counter[word] = 1

        first_index = 0
        for key, value in self._counter.items():
            if value >= threshold:
                self._encoder[key] = [first_index, 0, 0]
                first_index += 1

        for row in range(len(X)):
            words = X[row]
            label = y[row]

            for word in words:
                if word in self._encoder:
                    self._encoder[word][1] += 1
                    if label == 1:
                        self._encoder[word][2] += 1

        return self

    def transform(self, X):
        """
        unpresented word will be dropped
        index is not in ascending order!!!

        Parameters
        ----------
        X: [[field1_index1, filed1_index2, field2_index1],
            [field1_index3, field3_index1]]

        Returns
        -------
        csr matrix
        """
        nrows = len(X)
        ncols = len(self._encoder)

        row_indices = list()
        col_indices = list()
        data = list()
        for row in range(nrows):
            x = X[row]
            for word in x:
                if word in self._encoder:
                    col = self._encoder[word][0]
                    row_indices.append(row)
                    col_indices.append(col)
                    data.append(
                            1.0 * self._encoder[word][2] /
                            self._encoder[word][1])

        return coo_matrix((
            np.array(data),
            (np.array(row_indices), np.array(col_indices))),
            shape=(nrows, ncols)).tocsr()

    @property
    def codebook(self):
        return self._encoder
