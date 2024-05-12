from functools import cached_property
from typing import get_args
from math import ceil

import pandas as pd

from .summary import Summary
from ._constants import ColNames, ICTAL_TYPES, _PREICTAL, _INTERICTAL


def _evenly_divide_interictal_epochs(epochs: pd.DataFrame):
    """Return segments of `epochs` with one 'preictal' epoch for each segment, and
    evenly divided 'interictal' epochs to balance the division of the interictal epochs
    for each seizure.
    """
    pi_epochs = epochs.xs(_PREICTAL, level=1, drop_level=True)
    num_pi_seizures = pi_epochs.index.nunique()
    if num_pi_seizures <= 1:
        raise Warning("Subject has {num_pi_seizures}: no segmenting is possible.")

    segments = list()
    ii_epochs = epochs.xs(_INTERICTAL, level=1, drop_level=True)
    num_ii_files = len(ii_epochs)

    if num_ii_files < num_pi_seizures:
        new_ii_epochs = list()
        multiplier = ceil(num_pi_seizures / num_ii_files)
        for _, (file, start, end, length) in ii_epochs.iterrows():
            new_len = length // multiplier
            for _ in range(multiplier):
                new_ii_epochs.append(
                    {
                        ColNames.FILE: file,
                        ColNames.START: start,
                        ColNames.END: start + new_len,
                        ColNames.LEN: new_len,
                    }
                )
                start += new_len
        ii_epochs = pd.DataFrame(new_ii_epochs).sample(frac=1)
        num_ii_files = len(ii_epochs)

    pi_idx = 0

    for seizure, group in pi_epochs.groupby(level=0):
        for _, (file, start, end, length) in group.iterrows():
            segments.append(
                {
                    ColNames.SEIZ: seizure,
                    ColNames.CLASS: _PREICTAL,
                    ColNames.FILE: file,
                    ColNames.START: start,
                    ColNames.END: end,
                    ColNames.LEN: length,
                }
            )
        for i in range(pi_idx, num_ii_files, num_pi_seizures):
            file, start, end, length = ii_epochs.iloc[i]
            segments.append(
                {
                    ColNames.SEIZ: seizure,
                    ColNames.CLASS: _INTERICTAL,
                    ColNames.FILE: file,
                    ColNames.START: start,
                    ColNames.END: end,
                    ColNames.LEN: length,
                }
            )
        pi_idx += 1
    return pd.DataFrame(segments).set_index([ColNames.SEIZ, ColNames.CLASS])


class Epochs(Summary):
    """Epochs class for extracting information from a `Summary` data instance. Acts as
    the base class for the `EdfDS` class. This class inherets and instantiates a
    `Summary` class to extract the epoching data from the `records` attribute.

    Parameters:
    sdir : str
        The subject directory containing the edf files

    sph : int
        Stands for 'Seizure Prediction Horizon', the discarded period between pre-ictal
        and inter-ictal class in (minutes).

    pil : int
        Stands for 'Pre-Ictal Length', the pre-ictal class period in (minutes)

    psl : int
        Stands for 'Post Seizure Length', allowed distance between two seizures, to
        avoid post-ictal period.

    iid : int
        Stands for 'Inter-Ictal Distance', the the discarded period between seizures
        and inter-ictal class in (minutes).


    Examples
    --------
    Example of retrieving the epoching table with interictal epochs divided based on
    upcoming seizure event.

    >>> epochs = Epochs(
    ...     "data/chb08/",
    ...     sph=5,
    ...     pil=60,
    ...     psl=120,
    ...     iid=120,
    ... )
    >>> epochs.table
                                            File  Start   End Length
    Seizure              Class
    chb08_02.edf.seizure Preictal    chb08_02.edf      0  2370   2370
    chb08_05.edf.seizure Preictal    chb08_04.edf   2563  3600   1037
                         Preictal    chb08_05.edf      0  2556   2556
    chb08_11.edf.seizure Preictal    chb08_10.edf   2695  3600    905
                         Preictal    chb08_11.edf      0  2688   2688
    chb08_21.edf.seizure Interictal  chb08_15.edf   2562  3600   1038
                         Interictal  chb08_16.edf      0  3600   3600
                         Interictal  chb08_17.edf      0  3600   3600
                         Interictal  chb08_18.edf      0  3600   3600
                         Interictal  chb08_19.edf      0  2098   2098
                         Preictal    chb08_20.edf   1791  3600   1809
                         Preictal    chb08_21.edf      0  1783   1783
    chb08_29.edf.none    Interictal  chb08_23.edf   2332  3600   1268
                         Interictal  chb08_24.edf      0  3600   3600
                         Interictal  chb08_29.edf      0  3623   3623


    The even epoching can be retrieved using the `even_table` attribute.

    >>> epochs.even_table
                                             File  Start   End  Length
    Seizure              Class
    chb08_02.edf.seizure Preictal    chb08_02.edf      0  2370    2370
                         Interictal  chb08_15.edf   2562  3600    1038
                         Interictal  chb08_19.edf      0  2098    2098
    chb08_05.edf.seizure Preictal    chb08_04.edf   2563  3600    1037
                         Preictal    chb08_05.edf      0  2556    2556
                         Interictal  chb08_16.edf      0  3600    3600
                         Interictal  chb08_23.edf   2332  3600    1268
    chb08_11.edf.seizure Preictal    chb08_10.edf   2695  3600     905
                         Preictal    chb08_11.edf      0  2688    2688
                         Interictal  chb08_17.edf      0  3600    3600
                         Interictal  chb08_24.edf      0  3600    3600
    chb08_21.edf.seizure Preictal    chb08_20.edf   1791  3600    1809
                         Preictal    chb08_21.edf      0  1783    1783
                         Interictal  chb08_18.edf      0  3600    3600
                         Interictal  chb08_29.edf      0  3623    3623
    """

    def __init__(
        self,
        sdir: str,
        sph: int,
        pil: int,
        psl: int,
        iid: int,
    ):
        self._sph = sph
        self._pil = pil
        self._psl = psl
        self._iid = iid
        super().__init__(sdir)

    @property
    def sph(self):
        return self._sph

    @property
    def pil(self):
        return self._pil

    @property
    def psl(self):
        return self._psl

    @property
    def iid(self):
        return self._iid

    @sph.setter
    def sph(self, value: int):
        self._sph = value
        self.__dict__.pop("table", None)

    @pil.setter
    def pil(self, value: int):
        self._pil = value
        self.__dict__.pop("table", None)

    @psl.setter
    def psl(self, value: int):
        self._psl = value
        self.__dict__.pop("table", None)

    @iid.setter
    def iid(self, value: int):
        self._iid = value
        self.__dict__.pop("table", None)

    @cached_property
    def table(self):
        """Table of all extracted epochs with each epoch containing only the files
        relative to the epoch's upcoming seizure. The interictal data are divided based
        on `evenly_divided` attribute.

        Return
        ------
        DataFrame:
            The indices:
            - "Seizure": str, upcoming seizure event name of the epoch file.
            - "Class": str, class name of the epoch file.
            The columns:
            - "File": str, name of the epoch file.
            - "Start": int, start time of epoch file.
            - "End": int, end time of epoch file.
            - "Length": int, length of epoch file.
        """
        epochs = self._generate_table()
        return epochs

    @property
    def even_table(self):
        """Similar to `table` with the interictal data evenly divided among seizures
        with preictal periods, discarding any seizure event without any preictal epoch.

        See Also
        --------
        - refer to the `table` attribute for data structure.

        Notes
        -----
        - The interictal data usully aren't evenly divided among seizure events epochs,
        this is the reason for the `even_table` attribute.
        - If `evenly_epoched` is True, seizure events with interictal data only are
        dropped due to fact that they will be distributed among other seizure event
        with preictal events.
        """
        df = self.table
        return _evenly_divide_interictal_epochs(df)

    @property
    def len_interictal(self):
        """Length of the extracted interictal data"""
        df = self.get_epochs_table(_INTERICTAL)
        return df.loc[:, ColNames.LEN].sum()

    @property
    def len_preictal(self):
        """Length of the extracted preictal data"""
        df = self.get_epochs_table(_PREICTAL)
        return df.loc[:, ColNames.LEN].sum()

    @property
    def len_both_ictals(self):
        """Total length of the extracted interictal and preictal data."""
        return self.len_preictal + self.len_interictal

    def get_epochs_table(
        self,
        klass: ICTAL_TYPES = None,
        seizures: str | list[str] = None,
        evenly_divided: bool = True,
    ):
        """Return table of epochs data.

        Parameters
        ----------
        klass: str | None
            Class of the epochs data, can be None, 'Preictal' or 'Interictal'. If None
            return both classes epochs data.

        seizures: list[str]
            seizures of the epochs data, can be None or a list of seizure indices from
            `table` 'Seizure' index level.

        evenly_divided: bool
            If False, epoching will divide interictal data according to the upcoming
            seizure. If True, epoching will divide interictal files among segments of
            the preictal epochs for a balanced interictal data division. Defaults to
            True.

        Return
        ------
        DataFrame:
            epochs information with structure similar to the `table` structure.

        Notes
        -----
        If one of the two classes is returned, the 'Class' index is dropped for convinience.

        See Also
        --------
        Refer to `table` attrubute for the structure of returned DataFrame.


        Examples
        --------
        >>> epochs = Epochs(
        ...     "data/chb08/",
        ...     sph=5,
        ...     pil=60,
        ...     psl=120,
        ...     iid=120,
        ...     evenly_epoched=False,
        ... )
        >>> epochs.get_epochs_table('Preictal')
                                      File  Start   End Length
        Seizure
        chb08_02.edf.seizure  chb08_02.edf      0  2370   2370
        chb08_05.edf.seizure  chb08_04.edf   2563  3600   1037
        chb08_05.edf.seizure  chb08_05.edf      0  2556   2556
        chb08_11.edf.seizure  chb08_10.edf   2695  3600    905
        chb08_11.edf.seizure  chb08_11.edf      0  2688   2688
        chb08_21.edf.seizure  chb08_20.edf   1791  3600   1809
        chb08_21.edf.seizure  chb08_21.edf      0  1783   1783
        """
        if evenly_divided:
            table = self.even_table
        else:
            table = self.table

        if klass is None:
            df = table
        elif klass in get_args(ICTAL_TYPES):
            df = table.xs(klass, level=1, drop_level=True)
        else:
            raise ValueError(
                f"'klass' parameter must be 'None' or one of {get_args(ICTAL_TYPES)}"
            )

        if isinstance(seizures, str):
            seizures = [seizures]

        if isinstance(seizures, list):
            try:
                df = df.loc[seizures]
            except KeyError as e:
                raise ValueError(
                    f"One of the seizures 'seizures' is not in the `table` indices with class of {klass if klass is not None else 'both'}",
                    e,
                )
        elif seizures is not None:
            raise ValueError(
                f"'seizures' parameter must be 'None' or a list of strings."
            )
        return df

    def get_seizure_names(self, klass: ICTAL_TYPES = None):
        """Return the names of the seizure events of `klass`, if None, return seizure events of both classes."""
        if klass is None:
            df = self.table
        elif klass in get_args(ICTAL_TYPES):
            df = self.table.xs(klass, level=1, drop_level=True)
        else:
            raise ValueError(
                f"'klass' parameter must be 'None' or one of {get_args(ICTAL_TYPES)}"
            )
        return df.index.unique().tolist()

    def get_trainable_seizures(self):
        """Return names of trainable seizure events, similar to the preictal seizures"""
        return self.get_seizure_names(_PREICTAL)

    def num_trainable_seizures(self):
        """Return the number of trainable seizures"""
        return len(self.get_trainable_seizures())

    def _generate_table(self) -> pd.DataFrame:
        """get the preictal and interictal data from `summary`."""
        # Load the epochs info and insert the epoch class
        ii_df = self.get_interictal_epochs(self.iid)
        ii_df[ColNames.CLASS] = _INTERICTAL

        pi_df = self.get_preictal_epochs(self.sph, self.pil, self.psl)
        pi_df[ColNames.CLASS] = _PREICTAL

        df = (
            pd.concat([ii_df, pi_df])
            .astype({ColNames.START: int, ColNames.END: int})
            .set_index([ColNames.SEIZ, ColNames.CLASS])
            .sort_index()
        )
        return df

    def __repr__(self) -> str:
        return (
            f"Epochs(subject_id={self.sub_id},"
            f" sph={self.sph}, pil={self.pil}, psl={self.psl}, iid={self.iid})"
        )


class ContEpochs(Summary):
    """Epochs class for extracting information from a `Summary` data instance. Acts as
    the base class for the `EdfDS` class. This class inherets and instantiates a
    `Summary` class to extract the epoching data from the `records` attribute.

    Args:
    sdir (str): The subject directory containing the edf files
    sph (int): Stands for 'Seizure Prediction Horizon', the discarded period between pre-ictal and ictal class in (minutes).
    psl (int): Stands for 'Post Seizure Length', discarded data after seizures, to avoid post-ictal class.
    """

    def __init__(self, sdir: str, sph: int, psl: int):
        self._sph = sph
        self._psl = psl
        super().__init__(sdir)

    @property
    def sph(self):
        return self._sph

    @property
    def psl(self):
        return self._psl

    @sph.setter
    def sph(self, value: int):
        self._sph = value
        self.__dict__.pop("table", None)

    @psl.setter
    def psl(self, value: int):
        self._psl = value
        self.__dict__.pop("table", None)

    @cached_property
    def table(self):
        """Table of all extracted epochs with each epoch containing countdown time
        relative to the epoch's upcoming seizure.

        Return
        ------
        DataFrame:
            The indices:
            - "Seizure": str, upcoming seizure event name of the epoch file.
            The columns:
            - "File": str, name of the epoch file.
            - "Start": int, start time of epoch file.
            - "End": int, end time of epoch file.
            - "Length": int, length of epoch file.
            - "Time": int, countdown time to the upcoming seizure event.
        """
        epochs = self._generate_table()
        return epochs

    def get_epochs_table(
        self,
        seizures: str | list[str] = None,
    ):
        """Return table of epochs data.

        Args:
            seizures (list[str]): seizures of the epochs data, can be None or a list of seizure indices from `table` 'Seizure' index level.

        Return:
            DataFrame: epochs information with structure similar to the `table` structure.

        See Also:
            Refer to `table` attrubute for the structure of returned DataFrame.


        Examples
        --------
        >>> epochs = Epochs(
        ...     "data/chb08/",
        ...     sph=5,
        ...     pil=60,
        ...     psl=120,
        ...     iid=120,
        ...     evenly_epoched=False,
        ... )
        >>> epochs.get_epochs_table('Preictal')
                                      File  Start   End Length
        Seizure
        chb08_02.edf.seizure  chb08_02.edf      0  2370   2370
        chb08_05.edf.seizure  chb08_04.edf   2563  3600   1037
        chb08_05.edf.seizure  chb08_05.edf      0  2556   2556
        chb08_11.edf.seizure  chb08_10.edf   2695  3600    905
        chb08_11.edf.seizure  chb08_11.edf      0  2688   2688
        chb08_21.edf.seizure  chb08_20.edf   1791  3600   1809
        chb08_21.edf.seizure  chb08_21.edf      0  1783   1783
        """
        df = self.table

        if isinstance(seizures, str):
            seizures = [seizures]

        if isinstance(seizures, list):
            try:
                df = df.loc[seizures]
            except KeyError as e:
                msg = f"One of the seizures {seizures} is not in the `table` indices."
                raise ValueError(msg, e)
        elif seizures is not None:
            msg = f"'seizures' parameter must be 'None' or a list of strings"
            raise ValueError(msg)
        return df

    def get_trainable_seizures(self):
        """Return names of trainable seizure events, similar to the preictal seizures"""
        df = self.table
        return df.index.unique().tolist()

    def num_trainable_seizures(self):
        """Return the number of trainable seizures"""
        return len(self.get_trainable_seizures())

    def _generate_table(self) -> pd.DataFrame:
        """get the preictal and interictal data from `summary`."""
        # Load the epochs info and insert the epoch class
        df = self.get_proictal_epochs(self.sph, self.psl)
        df = df.set_index([ColNames.SEIZ])
        return df

    def __repr__(self) -> str:
        return f"ContEpochs(subject_id={self.sub_id}, sph={self.sph}, psl={self.psl})"
