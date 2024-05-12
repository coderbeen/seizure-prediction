"""The edfds module contains the DataStore class, which is used to manage the EEG data for the model."""

import math
import logging
from typing import Literal
from functools import cached_property

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
        phop (float | "auto"): The length of the hop between preictal frames in seconds.
        ihop (float | "auto"): The length of the hop between interictal frames in seconds.
        max_overlap (float): The maximum overlap between preictal frames.
        resampling_strategy (str | None): The sampling strategy to use for the data.

    Attributes:
        seg (int): Length of the segmenting process in seconds.
        frame_shape (tuple[int, int]): The shape of the EEG signal frames.
        currnet_test_seizure (str): The current test seizure for indexing.
        data (dict): A dictionary containing the EEG data for each seizure.
        num_frames (dict): A dictionary containing the number of frames for each seizure.
    """

    def __init__(
        self,
        subjectdir: str = None,
        seg: int = 8,
        sph: int = 5,
        pil: int = 60,
        psl: int = 120,
        iid: int = 120,
        phop: float | Literal["auto"] = "auto",
        ihop: float | Literal["auto"] = "auto",
        max_overlap: float = 0.5,
        resampling_strategy: None | RESAMPLING_STRATEGIES = "random_under_sampler",
        resampling_kwargs: dict = {},
        reader_kwargs: dict = {},
    ):
        super().__init__(subjectdir, sph, pil, psl, iid)
        logger.info("DataStore initialized at %s", subjectdir)

        self._seg = seg
        self._phop = self._parse_phop(phop, max_overlap)
        self._ihop = self._parse_ihop(ihop)

        self._resampler = None
        self._set_resampling_strategy(resampling_strategy, **resampling_kwargs)

        self._readfn = EdfReader(subjectdir, **reader_kwargs)
        self._index = 0
        self._data = dict()
        self._data_len = dict()

    @property
    def seg(self):
        return self._seg

    @seg.setter
    def seg(self, value: int):
        self._seg = value
        self._data = None

    @property
    def resampler(self):
        return self._resampler

    @property
    def frame_shape(self):
        return (len(EEG_CHANNELS_SEL), self.seg * self.sampling_rate)

    @property
    def currnet_test_seizure(self):
        return self.get_trainable_seizures()[self._index]

    @property
    def data(self):
        return self._data

    @property
    def num_frames(self):
        return self._data_len

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

    def load_data(self):
        data = dict()
        data_len = dict()
        for seizure in self.get_trainable_seizures():
            x, y = self.read_data(seizure)
            if self.resampler is not None:
                x, y = self.resampler.fit_resample(x, y)
            data[seizure] = (x, y)
            data_len[seizure] = len(y)
        self._data = data
        self._data_len = data_len

    def read_data(self, seizures: list[str] = None) -> tuple[NDArray, NDArray]:
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
        # Only use overlapping if preictal data is less than percentage of interictal data
        num_pframes = self._num_frames_in_files(pfiles, hop=self._phop)
        num_iframes = self._num_frames_in_files(ifiles, hop=self._ihop)

        num_frames = num_pframes + num_iframes
        num_channels = len(EEG_CHANNELS_SEL)
        frame_len = self.seg * self.sampling_rate
        x = np.empty((num_frames, num_channels, frame_len), dtype=np.float32)
        y = np.empty(num_frames, dtype=bool)

        logger.info("Retrieving %s frames for seizures: %s", num_frames, seizures)

        plen = self._get_files_frames(x, pfiles, from_index=0, hop=self._phop)
        ilen = self._get_files_frames(x, ifiles, from_index=plen, hop=self._ihop)

        if plen + ilen < num_frames:
            x = x[: plen + ilen]
            y = y[: plen + ilen]
        y[:plen] = np.ones(plen, dtype=bool)
        y[plen:] = np.zeros(ilen, dtype=bool)

        logger.info("Data retrieved, class counts =%s", np.bincount(y))
        return x, y

    def get_data(self, seizures: list[str]):
        """
        Retrieves EEG frames and their classes for the specified seizures.

        Args:
            seizures (list[str]): A list of seizure names to retrieve data for.

        Returns:
            tuple[NDArray, NDArray]: A tuple containing the input data (x) and the corresponding labels (y).
        """
        if not isinstance(seizures, list):
            seizures = [seizures]

        if not self._data:
            self.load_data()

        num_frames = sum(self._data_len[seizure] for seizure in seizures)
        shape = (num_frames, len(EEG_CHANNELS_SEL), self.seg * self.sampling_rate)
        x = np.empty(shape, dtype=np.float32)
        y = np.empty(num_frames, dtype=bool)

        start = 0
        for seizure in seizures:
            xs, ys = self._data[seizure]
            stop = start + len(ys)
            x[start:stop] = xs
            y[start:stop] = ys
            start = stop

        return x, y

    def get_train_data(self, test_seizure: str = None) -> tuple[NDArray, NDArray]:
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

        return self.get_data(seizures)

    def get_test_data(
        self, test_seizure: str = None, resampled: bool = False
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

        if self._resampler is not None:
            x, y = self.read_data(test_seizure)
        else:
            x, y = self.get_data(test_seizure)

        if resampled:
            resampler = Resampler("random_under_sampler", random_state=999)
            x, y = resampler.fit_resample(x, y)

        return x, y

    def _read_frames(self, file: str, period: tuple[int, int], hop: int):
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

        frame_len = self.seg * self.sampling_rate
        num_frames = (end - start) * self.sampling_rate // hop
        # modify 'start' to make period multiple of 'stride'
        if num_frames == 0:
            return None

        # truncate the signal from both ends to make it multiple of 'stride'
        signal = self._readfn(file, period=(start, end))
        num_samples = signal.shape[1]
        # last period must have a length of 'frame_len'
        to_drop = (num_samples - frame_len) % hop
        if to_drop > 0:
            signal = signal[:, to_drop:]
        num_samples = signal.shape[1]
        if num_samples < frame_len:
            return None

        stop = num_samples - frame_len
        splits = [signal[:, i : i + frame_len] for i in range(0, stop, hop)]
        return np.array(splits)

    def _get_files_frames(self, frames, files, from_index, hop):
        """
        Fill the frames array with the signal frames from the specified files starting from the specified index. Returns the number of frames added to the array.
        """
        start = from_index
        for _, (file, *period, _) in files.iterrows():
            frame = self._read_frames(file, period, hop)
            if frame is None:
                continue
            num_frames = frame.shape[0]
            frames[start : start + num_frames] = frame
            start = start + num_frames
        return start - from_index

    def _num_frames_in_files(self, files, hop=None):
        """Calculate the number of frames in a given period."""
        frame_len = self.seg * self.sampling_rate
        if hop is None:
            hop = frame_len
        num_frames = 0
        for _, (_, *period, _) in files.iterrows():
            period_len = (period[1] - period[0]) * self.sampling_rate
            num_frames += math.floor(period_len // hop)
        return int(num_frames)

    def _parse_phop(self, phop, max_overlap):
        if phop == "auto":
            frame_len = self.seg * self.sampling_rate
            pfiles = self.get_epochs_table(_PREICTAL)
            ifiles = self.get_epochs_table(_INTERICTAL)
            num_pframes = self._num_frames_in_files(pfiles)
            num_iframes = self._num_frames_in_files(ifiles)
            if num_pframes < num_iframes:
                hop = frame_len * (num_pframes / num_iframes)
                lowest_hop = frame_len * (1 - max_overlap)
                hop = max(hop, lowest_hop)
            else:
                hop = frame_len
        elif isinstance(phop, (int, float)):
            if phop > self.seg:
                msg = f"phop must be less than or equal to {self.seg}."
                raise ValueError(msg)
            elif phop < 0:
                msg = "phop must be greater than or equal to 0."
                raise ValueError(msg)
            hop = phop * self.sampling_rate
        else:
            msg = "phop must be an integer, float, or 'auto'."
            raise ValueError(msg)

        hop = int(hop)
        logger.info("Preictal hop length set to %s", hop)
        return hop

    def _parse_ihop(self, ihop):
        if ihop == "auto":
            frame_len = self.seg * self.sampling_rate
            seizures = self.get_trainable_seizures()
            lowest_factor = 1e10
            for seizure in seizures:
                pfiles = self.get_epochs_table(_PREICTAL, [seizure])
                ifiles = self.get_epochs_table(_INTERICTAL, [seizure])
                num_pframes = self._num_frames_in_files(pfiles, hop=self._phop)
                num_iframes = self._num_frames_in_files(ifiles)
                lowest_factor = min(num_iframes / num_pframes, lowest_factor)
            hop = frame_len * lowest_factor
        elif isinstance(ihop, (int, float)):
            if ihop < 0:
                msg = "ihop must be greater than or equal to 0."
                raise ValueError(msg)
            hop = ihop * self.sampling_rate
        else:
            msg = "ihop must be an integer, float, or 'auto'."
            raise ValueError(msg)

        hop = int(hop)
        logger.info("Interictal hop length set to %s", hop)
        return hop

    def _validate_seizure(self, seizure):
        """Check if the seizure is in the epochs table."""
        if seizure not in self.get_trainable_seizures():
            msg = f"Seizure '{seizure}' not a trainable seizure."
            raise ValueError(msg)

    def _set_resampling_strategy(self, strategy: RESAMPLING_STRATEGIES, **kwargs):
        """Set the sampling strategy for the data."""
        if strategy is None:
            if kwargs != {}:
                msg = "strategy is None, but additional keyword arguments are provided."
                raise ValueError(msg)
            self._resampler = None
        else:
            self._resampler = Resampler(strategy, **kwargs)

    def __repr__(self) -> str:
        return (
            f"DataStore(subject_id={self.sub_id} seg={self.seg}) "
            f"sph={self.sph} pil={self.pil} psl={self.psl} iid={self.iid}"
        )

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

    def load_data(self):
        data = dict()
        for seizure in self.get_trainable_seizures():
            data[seizure] = self.get_data(seizure)

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
