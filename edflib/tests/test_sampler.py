import unittest
from typing import get_args

import numpy as np
from sklearn.datasets import make_classification
from edflib.resampler import Resampler, RESAMPLING_STRATEGIES
from edflib.datastore import DataStore


class TestResampler(unittest.TestCase):
    def setUp(self):
        self.data = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=3,
            n_redundant=3,
            n_repeated=4,
            n_clusters_per_class=2,
            weights=[0.8, 0.2],
        )
        self.data_size = self.data[0].shape[0]
        self.sample_shape = self.data[0].shape[1:]

    def test_invalid_strategy(self):
        # Test case: Check if an error is raised for an invalid resampling strategy
        with self.assertRaises(ValueError):
            Resampler(strategy="invalid_strategy")

    def test_invalid_classes(self):
        # Test case: Check if an error is raised for an invalid class
        resampler = Resampler(strategy="random")
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_clusters_per_class=1,
            n_classes=3,
        )
        with self.assertRaises(ValueError):
            resampler.fit_resample(X, y)

    def test_invalid_datatype(self):
        # Test case: Check if an error is raised for an invalid input
        resampler = Resampler(strategy="random")

        with self.assertRaises(ValueError):
            resampler.fit_resample([1, 5, 3, 2], np.array([0, 1, 0, 1]))
        with self.assertRaises(ValueError):
            resampler.fit_resample(np.array([1, 5, 3, 2]), [0, 1, 0, 1])

    def test_resampled_data_shape(self):
        # Test case: Check if the resampler preserves the shape of the input data
        resampler = Resampler(strategy="tomeklinks")
        X, y = self.data
        X_resampled, y_resampled = resampler.fit_resample(X, y)

        self.assertLessEqual(len(y_resampled), self.data_size)
        self.assertEqual(X_resampled.shape[1:], self.sample_shape)

    def test_all_resampling_strategies(self):
        # Test case: Check if all resampling strategies are available
        X, y = self.data
        strategies = get_args(RESAMPLING_STRATEGIES)
        for strategy in strategies:
            resampler = Resampler(strategy=strategy)
            X_resampled, y_resampled = resampler.fit_resample(X, y)

            counts = np.bincount(y_resampled)
            self.assertEqual(counts[0], counts[1])

    # def test_with_real_data(self):
    #     # Test case: Check if the resampler works with real data
    #     ds = DataStore(
    #         subjectdir=SUBJECT_DIR,
    #         seg=120,
    #         sampling_strategy=None,
    #     )
    #     seizure = ds.get_trainable_seizures()[0]
    #     _, y_original = ds.get_test_data(seizure)

    #     ds.sampling_strategy = "allknn"
    #     X, y = ds.get_test_data(seizure)

    #     self.assertEqual(X.shape[0], y.shape[0])
    #     self.assertEqual(X.shape[1:], ds.frame_shape)
    #     self.assertEqual(np.count_nonzero(y == 1), np.count_nonzero(y_original == 1))


if __name__ == "__main__":
    unittest.main()
