import os
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import mne
from mne.io import read_raw_edf
from mne.decoding import Scaler

from ._constants import EEG_CHANNELS_SEL


class EdfReader:
    """
    A class for reading signals from EDF files.

    Args:
        sdir (str): The subject directory path.
    """

    def __init__(self, sdir: str):
        self._validate_sdir(sdir)
        self.sdir = sdir

    def __call__(
        self,
        edf_fname: str,
        period: tuple[int, int] = None,
        ch_names: list[str] = EEG_CHANNELS_SEL,
        scale: Literal["mean", "median"] = None,
    ) -> NDArray:
        """
        Method for reading the signals in the EDF file.

        Args:
            edf_fname (str): The EDF file name.
            ch_names (list[str], optional): List of channel names. Defaults to EEG_CHANNELS_SEL.
            scale (None | Literal["mean", "median"], optional): Scaling method. Defaults to "mean".
            period (tuple[int, int], optional): Start and stop indices for the period of interest. Defaults to None.
            filters (dict, optional): Dictionary of filter parameters. Defaults to None.

        Returns:
            NDArray: A numpy array containing the signal data of the selected channels.

        Raises:
            ValueError: If the EDF file is invalid.
            ValueError: If the specified period is invalid.
        """
        self._validate_edf_file(edf_fname)

        verbose = "ERROR"
        edf_fpath = os.path.join(self.sdir, edf_fname)
        with read_raw_edf(edf_fpath, verbose=verbose) as raw:

            if ch_names is not None:
                raw = raw.pick(picks=EEG_CHANNELS_SEL)
                raw = raw.rename_channels({"T8-P8-0": "T8-P8"})

            if period is not None:
                self._validate_period(period)
                raw = raw.crop(tmin=period[0], tmax=period[1], verbose=verbose)

            # # Filter the power line noise and the low frequency drifts
            # raw.load_data(verbose=verbose)
            # raw.filter(l_freq=1, h_freq=100, verbose=verbose)
            # raw.notch_filter([60], notch_widths=6, verbose=verbose)

            data = raw.get_data(verbose=verbose)

            if scale is not None:
                scaler = Scaler(scalings=scale)
                data = scaler.fit_transform(data)
                data = data.squeeze()
        data = data.astype(np.float32).copy()
        return data

    def _validate_sdir(self, sdir):
        """
        check if the subject directory `sdir` exist.
        """
        if not os.path.isdir(sdir):
            msg = f"The subject directory {sdir} does not exist"
            raise OSError(msg)

    def _validate_edf_file(self, edf_fname):
        """
        check if the edf file `edf_fname` exist.
        """
        edf_fpath = os.path.join(self.sdir, edf_fname)
        if not os.path.isfile(edf_fpath) and edf_fpath.endswith(".edf"):
            msg = f"The edf file {edf_fname} does not exist"
            raise OSError(msg)

    @staticmethod
    def _validate_period(period):
        """Check if the period is valid."""
        if not isinstance(period, (tuple, list)):
            msg = "Period must be a tuple or list of start and end times."
            raise ValueError(msg)
        if len(period) != 2:
            msg = "Period must consist of start and end times."
            raise ValueError(msg)
        if not all(isinstance(i, int) for i in period):
            msg = "Period indices must be integers."
            raise ValueError(msg)
        if period[0] < 0 or period[1] < 0:
            msg = "Period indices must be positive."
            raise ValueError(msg)
        if period[0] >= period[1]:
            msg = "Period start must be less than end."
            raise ValueError(msg)
