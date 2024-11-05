import unittest

import numpy as np

from edflib.datastore import DataStore
from edflib._constants import EEG_CHANNELS_SEL

SUBJECT_DIR = "data/chb08/"


class TestDataStore(unittest.TestCase):

    def setUp(self):
        self.datastore = DataStore(
            subjectdir=SUBJECT_DIR,
            seg=8,
            sph=5,
            pil=60,
            psl=120,
            iid=120,
            sampling_strategy=None,
        )

    def test_frame_shape(self):
        self.assertEqual(
            self.datastore.frame_shape,
            (len(EEG_CHANNELS_SEL), (8 * self.datastore.sampling_rate)),
        )

    def test_read_file(self):
        # Write test case to test the read_file method
        X = self.datastore.read_file("chb08_02.edf")

        self.assertAlmostEqual(X.shape[0], len(EEG_CHANNELS_SEL))
        self.assertEqual(X.shape[1], (3600 * self.datastore.sampling_rate))

    def test_invalid_siezure(self):
        # Write test case to test seizure validation
        with self.assertRaises(ValueError):
            self.datastore.get_data(seizures=["invalid_seizure"])
        with self.assertRaises(ValueError):
            self.datastore.get_train_data(test_seizure="chb08_13.edf.seizure")

    def test_get_data(self):
        # Write test case to test the get_data method
        X, y = self.datastore.get_data(
            seizures=[
                "chb08_02.edf.seizure",
                "chb08_05.edf.seizure",
                "chb08_11.edf.seizure",
                "chb08_21.edf.seizure",
            ]
        )

        self.assertAlmostEqual(X.shape[0] * 8, (35500), delta=300)
        self.assertEqual(X.shape[1], len(EEG_CHANNELS_SEL))
        self.assertAlmostEqual(X.shape[2], (8 * self.datastore.sampling_rate))

        class_counts = np.bincount(y)
        self.assertEqual(X.shape[0], len(y))
        self.assertEqual(len(class_counts), 2)
        self.assertAlmostEqual(class_counts[0], 2800, delta=10)
        self.assertAlmostEqual(class_counts[1], 1650, delta=10)

    def test_get_train_data(self):
        # Write test case to test the get_train_data method
        X, y = self.datastore.get_train_data(test_seizure="chb08_02.edf.seizure")

        self.assertEqual(X.shape[1], len(EEG_CHANNELS_SEL))
        self.assertAlmostEqual(X.shape[2], (8 * self.datastore.sampling_rate))

        class_counts = np.bincount(y)
        self.assertEqual(X.shape[0], len(y))
        self.assertEqual(len(class_counts), 2)

    def test_get_test_data(self):
        # Write test case to test the get_test_data method
        X, y = self.datastore.get_test_data(test_seizure="chb08_02.edf.seizure")

        self.assertEqual(X.shape[1], len(EEG_CHANNELS_SEL))
        self.assertAlmostEqual(X.shape[2], (8 * self.datastore.sampling_rate))

        class_counts = np.bincount(y)
        self.assertEqual(X.shape[0], len(y))
        self.assertEqual(len(class_counts), 2)

    def test_loop_over_datastore(self):
        # Write test case to test the __iter__ and __next__ methods
        seizures = [seizure for seizure in self.datastore]
        self.assertEqual(seizures, self.datastore.get_trainable_seizures())

    def test_resample(self):
        # Write test case to test the _resample method
        self.datastore.set_sampling_strategy("allknn")

        X_resampled, y_resampled = self.datastore.get_train_data(
            test_seizure="chb08_02.edf.seizure"
        )

        class_counts = np.bincount(y_resampled)
        self.assertEqual(X_resampled.shape[0], len(y_resampled))
        self.assertEqual(class_counts[0], class_counts[1])


if __name__ == "__main__":
    unittest.main()
