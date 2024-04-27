from functools import cached_property

import pandas as pd

from .chbmit import read_summary
from ._constants import ColNames, _SECONDS_IN_DAY, _SECONDS_IN_HOUR, _SECONDS_IN_MINUTE


def _len_total(records: pd.DataFrame):
    """Return the total length of the subject session including the gaps between the recordings."""
    return int(records.iloc[-1][ColNames.END] - records.iloc[0][ColNames.START])


def _len_records(records: pd.DataFrame):
    """Return the total length of the subject recordings without the gaps between the recordings."""
    len_ = 0
    for _, record in records.iterrows():
        len_ += record[ColNames.LEN]
    return len_


def _seizures_times(records: pd.DataFrame):
    """Return the index, name, start and end timings of all the seizures within the `records`"""
    seizures = list()
    for idx, record in records.iterrows():
        if record[ColNames.NSEIZ] > 0:
            seiz_times = record[ColNames.TSEIZ]
            seiz_name = record[ColNames.FILE] + ".seizure"
            if len(seiz_times) == 1:
                seizures.append([idx, seiz_name, *seiz_times[0].values()])
                continue
            seiz_id = 1
            for seiz_time in seiz_times:
                seiz_name = record[ColNames.FILE] + f".seizure {seiz_id}"
                seizures.append([idx, seiz_name, *seiz_time.values()])
                seiz_id += 1
    return seizures


def _extract_period(
    records: pd.DataFrame,
    margin: tuple[int, int],
    length: int = 0,
    include_last: bool = False,
    include_time: bool = False,
):
    """
    Filter and return all recordings that satify the `margin` condition after and before two
    seizures. Return only recordings within the period `length` from the preceding seizure if
    `length` is greater than zero.

    Paramters
    ---------
    records: DataFrame
        The recordings to be filtered.
    margin: Tuple[int]
        Left and Right margin, excluded period in seconds for the recent and preceding seizures,
        respectively. Margin defines start and end of epochd for each seizure.
    length: int
        The epoch length starting from the preceding seizure backwards. Extract whole period
        satisfying margin if this argument is zero.
    include_last: bool
        If True, include the period after the last seizure until the end of the subject session.
    include_time: bool
        If True, include the time distance between file and upcomming seizure.
    """
    left, right = margin
    seizures = _seizures_times(records)
    epochs = pd.DataFrame()
    # Last period ends at the last record
    if include_last:
        last_idx = len(records) - 1
        last_name = records.at[last_idx, ColNames.FILE] + ".none"
        last_start = records.at[last_idx, ColNames.END] + right
        seizures.append([last_idx, last_name, last_start, 0])
    # First period has no preceding seizure, adding margin to cancel out.
    start = records.at[0, ColNames.START] - left
    for seiz_idx, seiz_name, seiz_start, seiz_end in seizures:
        # Period end before seizure start shifted right.
        end = records.at[seiz_idx, ColNames.START] + (seiz_start - right)
        if start >= end:
            # Next Period start after seizure start shifted left.
            start = records.at[seiz_idx, ColNames.START] + (seiz_end + left)
            continue
        # Filter recordings based on allowed margin (current onset and last offset).
        group = _between_times(records, start, end, include_time)
        start = records.at[seiz_idx, ColNames.START] + (seiz_end + left)

        if group is None:
            continue
        # Reduce recordings times based on period length.
        if length > 0:
            group = _between_times(records, end - length, end, include_time)
        if group is None:
            continue

        if include_time:
            seiz_time = records.at[seiz_idx, ColNames.START] + seiz_start
            if seiz_name.endswith(".none"):
                group[ColNames.TIME] = pd.NA
            else:
                group[ColNames.TIME] = seiz_time - group[ColNames.TIME]

        group[ColNames.SEIZ] = seiz_name
        epochs = pd.concat([epochs, group])

    if len(epochs) > 0:
        epochs[ColNames.LEN] = epochs[ColNames.END] - epochs[ColNames.START]
        cols = [
            ColNames.SEIZ,
            ColNames.FILE,
            ColNames.START,
            ColNames.END,
            ColNames.LEN,
        ]
        if include_time:
            cols.append(ColNames.TIME)
        epochs = epochs[cols]
        return epochs
    raise ValueError("_extract_period(): No epoch was extracted for the given 'margin'")


def _between_times(records: pd.DataFrame, start, end, include_time=False):
    """
    Filter and return all recordings in-between `start` and `end` times,
    both times are absolute and of type `RecordTime`
    """
    indices = records[ColNames.END].ge(start) & records[ColNames.START].le(end)
    if not indices.any():
        return None
    group = records.loc[indices, [ColNames.FILE, ColNames.START, ColNames.END]].copy()

    # Return the times relative to the recording itself.
    def relative_times(s):
        st = 0 if s[ColNames.START] >= start else (start - s[ColNames.START])
        et = (
            (s[ColNames.END] - s[ColNames.START])
            if s[ColNames.END] <= end
            else (end - s[ColNames.START])
        )
        return int(st), int(et)

    times = group.apply(relative_times, axis=1).tolist()

    if include_time:
        group[ColNames.TIME] = group[ColNames.START] + [st for st, _ in times]
    group.loc[:, [ColNames.START, ColNames.END]] = times
    return group


def _pretty(df: pd.DataFrame):
    """Format the `records` file and seizure times to be more human readable."""

    def format_time(c):
        day, c = divmod(c, _SECONDS_IN_DAY)
        hour, c = divmod(c, _SECONDS_IN_HOUR)
        minute, c = divmod(c, _SECONDS_IN_MINUTE)
        second = c
        return "day: {:d}, {:02d}:{:02d}:{:02d}".format(day, hour, minute, second)

    def format_seiz_times(c):
        return "\\n".join(f"{x[ColNames.START]} -> {x[ColNames.END]}" for x in c)

    df[ColNames.START] = df[ColNames.START].apply(format_time)
    df[ColNames.END] = df[ColNames.END].apply(format_time)
    df[ColNames.TSEIZ] = df[ColNames.TSEIZ].apply(format_seiz_times)

    return df


class Summary:
    """
    Subject summary information class. This class is the base class for the Epocher class.

    Parameters
    ----------
    sdir: str
        The subject directory containing the summary data.
    """

    def __init__(self, sdir: str):
        records, sid, rate = read_summary(sdir)
        self._records = records
        self._sub_id = sid
        self._sampling_rate = rate

    @property
    def records(self):
        """
        The recording files information as a DataFrame

        Return
        ------
        records : DataFrame
            Rows represent the the extracted information about each file in the `sdir`
            subject directory. Columns are:
            - "File": str, the name of file.
            - "Start": int, the start timestamp of file in seconds.
            - "End": int, the end timestamp of file in seconds.
            - "Length": int, the length of file in seconds.
            - "N Seizures: int, the number of seizure events in file.
            - "Seizure Times": list, dictionaries of times of each seizure event with keys:
                - "Start": int, start time of the seizure event.
                - "End": int, end time of the seizure event.

        Examples
        --------
        >>> summary = Summary('data/chb08/')
        >>> summary.records[0:3]
                   File  Start    End  Length  N Seizures                   Seizure Times
        0  chb08_02.edf  44937  48537    3600           1  [{'Start': 2670, 'End': 2841}]
        1  chb08_03.edf  48539  52139    3600           0                              []
        2  chb08_04.edf  52147  55747    3600           0                              []
        """
        return self._records

    @property
    def sub_id(self):
        """The subject id."""
        return self._sub_id

    @property
    def sampling_rate(self):
        """The sampling rate of the recordings."""
        return self._sampling_rate

    @cached_property
    def num_records(self):
        """Number of recordings in `records` data."""
        return len(self.records)

    @cached_property
    def len_records(self):
        """The total length of recordings in `records` data."""
        return _len_records(self.records)

    @cached_property
    def len_avg_record(self):
        """The average length of recordings in `records` data."""
        return self.len_records // self.num_records

    @cached_property
    def len_total(self):
        """The total length of subject session, different than `len_records` in accounting the time gap between each recording."""
        return _len_total(self.records)

    @cached_property
    def seizures(self):
        """
        The seizure events timing information

        Return
        ------
        list[list]: Each list contains `records` index of record containing the seizure, id of the seizure event, start and end times of the seizure event.

        Examples
        --------
        >>> summary = Summary('data/chb08/')
        >>> summary.seizures[0]
        [0, 'chb08_02.edf.seizure', 2670, 2841
        """
        return _seizures_times(self.records)

    @cached_property
    def num_seizures(self):
        """Number of seizure events in the `records` data."""
        return len(self.seizures)

    def pretty_records(self):
        """
        Format the `records` file and seizure times to be more human readable.

        See Also
        --------
        Summary.records: The original recordings DataFrame

        Examples
        --------
        >>> summary = Summary('data/chb08/')
        >>> summary.pretty_records()[0:3]
                   File             Start               End  Length  N Seizures Seizure Times
        0  chb08_02.edf  day: 0, 12:28:57  day: 0, 13:28:57    3600           1  2670 -> 2841
        1  chb08_03.edf  day: 0, 13:28:59  day: 0, 14:28:59    3600           0
        2  chb08_04.edf  day: 0, 14:29:07  day: 0, 15:29:07    3600           0
        """
        return _pretty(self.records.copy())

    def get_interictal_epochs(self, iid: int):
        """
        extract interictal epochs within `iid` after seizure and `iid` before seizure.

        Parameters
        ----------
        iid: int
            stands for 'Inter-Ictal Distance', the the discarded period between seizures and inter-ictal class in (minutes).

        Returns
        -------
        DataFrame:
            A pandas DataFrame, containing the preictal epochs within edf files. The DataFrame has the following columns:
            - Multi-Index: str, contain preceding seizure file name and epoch edf file name.
            - Start: int, The start time of the epoch within edf file.
            - End: int, The end time of the epoch within edf file.

        Examples
        --------
        >>> summary = Summary("data/chb08/")
        >>> print(summary.get_interictal_epochs(iid=120))
                         Seizure          File Start   End Length
        0   chb08_02.edf.seizure  chb08_02.edf     0  2370   2370
        2   chb08_05.edf.seizure  chb08_04.edf  2563  3600   1037
        3   chb08_05.edf.seizure  chb08_05.edf     0  2556   2556
        4   chb08_11.edf.seizure  chb08_10.edf  2695  3600    905
        5   chb08_11.edf.seizure  chb08_11.edf     0  2688   2688
        14  chb08_21.edf.seizure  chb08_20.edf  1791  3600   1809
        15  chb08_21.edf.seizure  chb08_21.edf     0  1783   1783
        """
        iid *= _SECONDS_IN_MINUTE
        return _extract_period(self.records, (iid, iid), include_last=True)

    def get_preictal_epochs(self, sph: int, pil: int, psl: int):
        """
        extract preictal epochs within `psl` after seizure and `sph` before seizure, starting at `sph` before seizure with length of `pil`.

        Parameters
        ----------
        sph: int
            stands for 'Seizure Prediction Horizon', the discarded period between pre-ictal and seizure start in (minutes).
        pil: int
            stands for 'Pre-Ictal Length', the pre-ictal class period in (minutes).
        psl: int
            stands for 'Post Seizure Length', allowed distance between two seizures in (minutes).

        Returns
        -------
        DataFrame:
            A pandas DataFrame, containing the preictal epochs within edf files. The DataFrame has the following columns:
            - Multi-Index: str, contain preceding seizure file name and epoch edf file name.
            - Start: int, The start time of the epoch within edf file.
            - End: int, The end time of the epoch within edf file.

        Example
        -------
        >>> summary = Summary(sdir="data/chb01/")
        >>> print(summary.get_preictal_epochs(sph=5, pil=60, psl=120))
                         Seizure          File Start   End Length
        9   chb08_21.edf.seizure  chb08_15.edf  2562  3600   1038
        10  chb08_21.edf.seizure  chb08_16.edf     0  3600   3600
        11  chb08_21.edf.seizure  chb08_17.edf     0  3600   3600
        12  chb08_21.edf.seizure  chb08_18.edf     0  3600   3600
        13  chb08_21.edf.seizure  chb08_19.edf     0  2098   2098
        17     chb08_29.edf.none  chb08_23.edf  2332  3600   1268
        18     chb08_29.edf.none  chb08_24.edf     0  3600   3600
        19     chb08_29.edf.none  chb08_29.edf     0  3623   3623
        """
        sph *= _SECONDS_IN_MINUTE
        pil *= _SECONDS_IN_MINUTE
        psl *= _SECONDS_IN_MINUTE
        return _extract_period(self.records, (psl, sph), pil)

    def get_proictal_epochs(self, sph: int, psl: int):
        """
        extract proictal epochs within `psl` after seizure and `sph` before seizure, starting at `sph` before seizure with length of `pil`.

        Parameters
        ----------
        sph: int
            stands for 'Seizure Prediction Horizon', the discarded period between pre-ictal and seizure start in (minutes).
        psl: int
            stands for 'Post Seizure Length', discarded post-ictal data after seizures in (minutes).

        Returns
        -------
        DataFrame:
            A pandas DataFrame, containing the proictal epochs within edf files. The DataFrame has the following columns:
            - Multi-Index: str, contain preceding seizure file name and epoch edf file name.
            - Start: int, The start time of the epoch within edf file.
            - End: int, The end time of the epoch within edf file.
            - Length: int, The length of the epoch in seconds.
            - Time: int, The countdown time in seconds from the upcoming seizure start.

        Example
        -------
        """
        sph *= _SECONDS_IN_MINUTE
        psl *= _SECONDS_IN_MINUTE
        return _extract_period(self.records, (psl, sph), include_time=True)

    def __repr__(self) -> str:
        return f"Summary(subject_id={self.sub_id})"
