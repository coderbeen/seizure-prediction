"""The edfds module contains the DataStore class, which is used to manage the EEG data for the model."""

import logging

import numpy as np
from numpy.typing import NDArray

from .reader import EdfReader
from .epochs import Epochs, ContEpochs
from .resampler import Resampler, RESAMPLING_STRATEGIES
from ._constants import (
    ICTAL_TYPES,
    _INTERICTAL,
    _PREICTAL,
    EEG_CHANNELS_SEL,
)

logger = logging.getLogger(__name__)


class DataStore(Epochs):
    """
    A class for managing the EEG data.

    Args:
        subjectdir (str): The subject directory containing the edf files.
        seg (int): Length of the segmenting process in seconds.
        sph (int): Stands for 'Seizure Prediction Horizon', the discarded period between pre-ictal
            and inter-ictal class in minutes.
        pil (int): Stands for 'Pre-Ictal Length', the pre-ictal class period in minutes.
        psl (int): Stands for 'Post Seizure Length', allowed distance between two seizures, to
            avoid post-ictal period.
        iid (int): Stands for 'Inter-Ictal Distance', the discarded period between seizures and
            inter-ictal class in minutes.
        resampling_strategy (str | None): The sampling strategy to use for the data.

    Attributes:
        seg (int): Length of the segmenting process in seconds.
        frame_shape (tuple[int, int]): The shape of the EEG signal frames.
        currnet_test_seizure (str): The current test seizure for indexing.
    """

    def __init__(
        self,
        subjectdir: str = None,
        seg: int = 8,
        sph: int = 5,
        pil: int = 60,
        psl: int = 120,
        iid: int = 120,
        resampling_strategy: None | RESAMPLING_STRATEGIES = "random_under_sampler",
        resampling_kwargs: dict = None,
    ):
        super().__init__(subjectdir, sph, pil, psl, iid)
        self._seg = seg
        self._resampler = None
        self.set_resampling_strategy(resampling_strategy, **(resampling_kwargs or {}))
        self._readfn = EdfReader(subjectdir)
        self._index = 0
        logger.info("DataStore initialized at %s", subjectdir)

    @property
    def seg(self):
        return self._seg

    @seg.setter
    def seg(self, value: int):
        self._seg = value

    @property
    def resampler(self):
        return self._resampler

    def set_resampling_strategy(self, strategy: RESAMPLING_STRATEGIES, **kwargs):
        """Set the sampling strategy for the data."""
        if strategy is None:
            if kwargs != {}:
                msg = "strategy is None, but additional keyword arguments are provided."
                raise ValueError(msg)
            self._resampler = None
        else:
            self._resampler = Resampler(strategy, **kwargs)

    @property
    def frame_shape(self):
        return (len(EEG_CHANNELS_SEL), self.seg * self.sampling_rate)

    @property
    def currnet_test_seizure(self):
        return self.get_trainable_seizures()[self._index]

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.get_trainable_seizures()):
            seizure = self.currnet_test_seizure
            self._index += 1
            return seizure
        else:
            self._index = 0
            raise StopIteration

    def read_file(self, file: str, period: tuple[int, int] = None):
        """
        Reads the specified EDF file and returns the signal data within the specified period.

        Args:
            file (str): The path to the EDF file.
            period (tuple[int, int], optional): The start and end time of the period to read, in seconds. If not provided, the entire signal will be read. Defaults to None.

        Returns:
            numpy.ndarray: The signal data within the specified period.
        """
        if period is None:
            return self._readfn(file)
        return self._readfn(file, period)

    def get_data(
        self,
        seizures: list[str] = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Retrieves EEG frames and their classes for the specified seizures.

        Args:
            seizures (list[str], optional): A list of seizure names to retrieve data for. If not provided, data for all trainable seizures will be retrieved.

        Returns:
            tuple[NDArray, NDArray]: A tuple containing two arrays: x (input data) and y (target labels).

        Raises:
            None

        """
        if seizures is None:
            seizures = self.get_trainable_seizures()
        if not isinstance(seizures, list):
            seizures = [seizures]
        for seizure in seizures:
            self._validate_seizure(seizure)

        pfiles = self.get_epochs_table(_PREICTAL, seizures)
        ifiles = self.get_epochs_table(_INTERICTAL, seizures)

        # Calculate the number of frames required to initialize the array
        num_pframes = self._num_frames_in_files(pfiles)
        num_iframes = self._num_frames_in_files(ifiles)
        num_frames = num_pframes + num_iframes
        num_channels = len(EEG_CHANNELS_SEL)
        frame_len = self.seg * self.sampling_rate
        x = np.empty((num_frames, num_channels, frame_len), dtype=np.float32)
        y = np.empty(num_frames, dtype=bool)

        logger.info("Retrieving %s frames for seizures: %s", num_frames, seizures)

        plen = self._get_files_frames(x, pfiles, from_index=0)
        ilen = self._get_files_frames(x, ifiles, from_index=plen)

        if plen + ilen < num_frames:
            x = x[: plen + ilen]
            y = y[: plen + ilen]
        y[:plen] = np.ones(plen, dtype=bool)
        y[plen:] = np.zeros(ilen, dtype=bool)

        logger.info("Data retrieved, class counts =%s", np.bincount(y))
        return x, y

    def get_train_data(
        self,
        test_seizure: str = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Retrieves the training EEG frames, which represents data for all seizures excluding `test_seizure` data.

        Args:
            test_seizure (str): The seizure data to be excluded from the training data.
            resampling_kwargs (dict, optional): The keyword arguments to be passed to the resampling method. Defaults to None.

        Returns:
            tuple[NDArray, NDArray]: A tuple containing the input data (x) and the corresponding labels (y).

        Raises:
            ValueError: If the test_seizure is not found in trainable seizures.

        Notes:
            - The available sampling methods are a subset of the sampling methods provided by the imbalanced-learn (imblearn) package.
        """
        if test_seizure is None:
            test_seizure = self.currnet_test_seizure
        else:
            self._validate_seizure(test_seizure)
        seizures = self.get_trainable_seizures()
        seizures.pop(seizures.index(test_seizure))

        x, y = self.get_data(seizures)

        if self.resampler is not None:
            x, y = self.resampler.fit_resample(x, y)

        return x, y

    def get_test_data(
        self,
        test_seizure: str = None,
        resampled: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """
        Get test data for a given seizure `test_seizure`.

        Args:
            test_seizure (str): The seizure data to be used as test data.
            resampled (bool, optional): Whether to resample the data randomly. Defaults to False.

        Returns:
            tuple[NDArray, NDArray]: The test data and targets.

        Notes:
            - The sampling method is "random_under_sampler" for accurate metrics measuring, with a defined state to ensure reproducibility.

        Raises:
            ValueError: If the test_seizure is not found in the trainable seizures.
        """
        if test_seizure is None:
            test_seizure = self.currnet_test_seizure
        else:
            self._validate_seizure(test_seizure)

        x, y = self.get_data(test_seizure)

        if resampled:
            resampler = Resampler("random_under_sampler", random_state=999)
            x, y = resampler.fit_resample(x, y)

        return x, y

    def _read_frames(self, file: str, period: tuple[int, int]):
        """
        Reads and returns the segmented signal from the specified EDF file within the given period.

        Args:
            file (str): The name of the file to read.
            period (tuple[int, int], optional): The period of the signal frames to read, specified as a tuple of start and end indices. If not provided, the entire signal will be read. Defaults to None.

        Returns:
            numpy.ndarray: An array containing the signal frames.

        Raises:
            ValueError: If the period start is greater than the period end.
        """
        start, end = period
        # end time might be more than file length due to rounding of chb-mit times
        end -= 1

        num_frames = (end - start) // self.seg
        frame_len = self.seg
        # modify 'start' to make period multiple of 'self.seg'
        start += (end - start) % frame_len
        if num_frames == 0:
            return None

        # load the signal, truncate, and segment by splitting
        signal = self._readfn(file, period=(start, end))
        # truncate the signal to make it multiple of 'frame_len' * 'self.sampling_rate'
        to_drop = signal.shape[1] % (frame_len * self.sampling_rate)
        if to_drop > 0:
            signal = signal[:, :-(to_drop)]
        signal = np.split(signal, num_frames, axis=1)
        return np.array(signal)

    def _get_files_frames(self, frames, files, from_index):
        """
        Fill the frames array with the signal frames from the specified files starting from the specified index. Returns the number of frames added to the array.
        """
        start = from_index
        for _, (file, *period, _) in files.iterrows():
            frame = self._read_frames(file, period)
            if frame is None:
                continue
            num_frames = frame.shape[0]
            frames[start : start + num_frames] = frame
            start = start + num_frames
        return start - from_index

    def _num_frames_in_files(self, files):
        """Calculate the number of frames in a given period."""
        seg = self.seg
        num_frames = 0
        for _, (_, *period, _) in files.iterrows():
            period_len = period[1] - period[0]
            num_frames += period_len // seg
        return num_frames

    def _validate_seizure(self, seizure):
        """Check if the seizure is in the epochs table."""
        if seizure not in self.get_trainable_seizures():
            msg = f"Seizure '{seizure}' not a trainable seizure."
            raise ValueError(msg)

    def __repr__(self) -> str:
        return (
            f"DataStore(subject_id={self.sub_id} seg={self.seg}) "
            f"sph={self.sph} pil={self.pil} psl={self.psl} iid={self.iid}"
        )


class ContDataStore(ContEpochs):
    """
    A class for managing the continuous EEG data.

    Args:
        subjectdir (str): The subject directory containing the edf files.
        seg (int): Length of the segmenting process in seconds.
        sph (int): Stands for 'Seizure Prediction Horizon', the discarded period between pre-ictal
            and inter-ictal class in minutes.
        pil (int): Stands for 'Pre-Ictal Length', the pre-ictal class period in minutes.
        psl (int): Stands for 'Post Seizure Length', allowed distance between two seizures, to
            avoid post-ictal period.
        iid (int): Stands for 'Inter-Ictal Distance', the discarded period between seizures and
            inter-ictal class in minutes.
        resampling_strategy (str | None): The sampling strategy to use for the data.

    Attributes:
        seg (int): Length of the segmenting process in seconds.
        frame_shape (tuple[int, int]): The shape of the EEG signal frames.
        currnet_test_seizure (str): The current test seizure for indexing.
    """

    def __init__(
        self,
        subjectdir: str = None,
        seg: int = 8,
        sph: int = 0,
        psl: int = 120,
    ):
        super().__init__(subjectdir, sph, psl)
        self._seg = seg
        self._readfn = EdfReader(subjectdir)
        logger.info("DataStore initialized at %s", subjectdir)

    @property
    def seg(self):
        return self._seg

    @seg.setter
    def seg(self, value: int):
        self._seg = value

    @property
    def frame_shape(self):
        return (len(EEG_CHANNELS_SEL), self.seg * self.sampling_rate)

    def read_file(self, file: str, period: tuple[int, int] = None):
        """
        Reads the specified EDF file and returns the signal data within the specified period.

        Args:
            file (str): The path to the EDF file.
            period (tuple[int, int], optional): The start and end time of the period to read, in seconds. If not provided, the entire signal will be read. Defaults to None.

        Returns:
            numpy.ndarray: The signal data within the specified period.
        """
        if period is None:
            return self._readfn(file)
        return self._readfn(file, period)

    def get_data(
        self,
        seizures: list[str] = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Retrieves EEG frames and their classes for the specified seizures.

        Args:
            seizures (list[str], optional): A list of seizure names to retrieve data for. If not provided, data for all trainable seizures will be retrieved.

        Returns:
            tuple[NDArray, NDArray]: A tuple containing two arrays: x (input data) and y (target labels).

        Raises:
            None

        """
        if seizures is None:
            seizures = self.get_trainable_seizures()
        # todo: _validate_seizures must move to the parent class
        if not isinstance(seizures, list):
            seizures = [seizures]
        for seizure in seizures:
            self._validate_seizure(seizure)

        files = self.get_epochs_table(seizures=seizures)

        # Calculate the number of frames required to initialize the array
        num_frames = self._num_frames_in_files(files)
        num_channels = len(EEG_CHANNELS_SEL)
        frame_len = self.seg * self.sampling_rate
        x = np.empty((num_frames, num_channels, frame_len), dtype=np.float32)
        y = np.empty(num_frames, dtype=np.float32)

        logger.info("Retrieving %s frames for seizures: %s", num_frames, seizures)

        got_frames = self._get_files_frames(x, y, files)

        if got_frames < num_frames:
            x = x[:got_frames]
            y = y[:got_frames]

        logger.info("Data retrieved, frame counts =%s", len(y))
        return x, y

    def _read_frames(self, file: str, period: tuple[int, int]):
        """
        Reads and returns the segmented signal from the specified EDF file within the given period.

        Args:
            file (str): The name of the file to read.
            period (tuple[int, int], optional): The period of the signal frames to read, specified as a tuple of start and end indices. If not provided, the entire signal will be read. Defaults to None.

        Returns:
            numpy.ndarray: An array containing the signal frames.

        Raises:
            ValueError: If the period start is greater than the period end.
        """
        seg = self.seg
        start, end = period
        # end time might be more than file length due to rounding of chb-mit times
        end -= 1

        num_frames = (end - start) // seg
        frame_len = seg
        # modify 'start' to make period multiple of 'self.seg'
        shift = (end - start) % frame_len
        start += shift
        if num_frames == 0:
            return None

        # load the signal, truncate, and segment by splitting
        signal = self._readfn(file, period=(start, end))
        # truncate the signal to make it multiple of 'frame_len' * 'self.sampling_rate'
        to_drop = signal.shape[1] % (frame_len * self.sampling_rate)
        if to_drop > 0:
            signal = signal[:, :-(to_drop)]
        signal = np.split(signal, num_frames, axis=1)
        return np.array(signal), shift

    def _get_files_frames(self, frames, labels, files):
        """
        Fill the frames array with the signal frames from the specified files.
        Returns the number of frames added to the array.
        """
        seg = self.seg
        start = 0
        for _, (file, *period, leng, t) in files.iterrows():
            frame, shift = self._read_frames(file, period)
            if frame is None:
                continue
            num_frames = frame.shape[0]
            frames[start : start + num_frames] = frame

            t1 = t - shift
            t2 = t1 - num_frames * seg
            frame_labels = np.linspace(t1, t2, num_frames, endpoint=False)
            labels[start : start + num_frames] = frame_labels

            start = start + num_frames
        return start

    def _num_frames_in_files(self, files):
        """Calculate the number of frames in a given period."""
        seg = self.seg
        num_frames = 0
        for _, (_, *period, _) in files.iterrows():
            period_len = period[1] - period[0]
            num_frames += period_len // seg
        return num_frames

    def _validate_seizure(self, seizure):
        """Check if the seizure is in the epochs table."""
        if seizure not in self.get_trainable_seizures():
            msg = f"Seizure '{seizure}' not a trainable seizure."
            raise ValueError(msg)
