import os
import re

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

from ._constants import ColNames, _SECONDS_IN_DAY, _SECONDS_IN_HOUR, _SECONDS_IN_MINUTE

_CHBMIT_TIME_PATTERN = re.compile(
    r"^(?P<hour>[0-2]?[0-9]):(?P<minute>[0-5][0-9]):(?P<second>[0-5][0-9])$"
)
_CHBMIT_NAME_PATTERN = re.compile(
    r"^chb(?P<pid>\d{2})([a-c])?_(?P<fid>\d{2})(\+)?\.edf$"
)


def _read_txt(sdir: str):
    """Read the lines of the summary text file in the subject `sdir`."""
    # CHB-MIT summary file 'chbxx-summary.txt' is located in 'chbxx/'.
    subject = os.path.basename(os.path.dirname(sdir))
    txt_fname = subject + "-summary.txt"
    txt_fpath = os.path.join(sdir, txt_fname)
    try:
        with open(txt_fpath) as txt_file:
            lines = txt_file.readlines()
    except OSError:
        raise OSError(f"The {txt_fpath} file is not accessable.")
    return lines


def _parse_lines(lines: list[str]):
    """Return the Pandas `DataFrame` table of the subject files info"""
    records = list()
    # split the file content into groups based on empty lines
    groups = "".join(lines).strip().split("\n\n")

    for group in groups:
        lines = group.strip().split("\n")
        if "File Name" not in lines[0]:
            continue
        fname = lines[0].split(": ")[1]
        start = lines[1].split(": ")[1]
        end = lines[2].split(": ")[1]
        nseiz = lines[3].split(": ")[1]
        siez_times = lines[4 : len(lines)]
        records.append(
            {
                ColNames.FILE: fname,
                ColNames.START: start,
                ColNames.END: end,
                ColNames.LEN: -1,
                ColNames.NSEIZ: nseiz,
                ColNames.TSEIZ: siez_times,
            }
        )
    records = DataFrame[SummarySchema](records)
    return records


def _parse_subject_id(sdir: str):
    """Return the subject id from the subject directory."""
    return os.path.basename(sdir)


def _parse_sampling_rate(lines: str):
    """Return the sampling rate from the summary text file."""
    for line in lines:
        if "Sampling Rate" in line:
            return int(line.rstrip(" Hz\n").split(": ")[1])
    return None


def _parse_time(tstr: str):
    """Return the time in seconds from the time string `tstr` in the format 'HH:MM:SS'."""
    matches = re.match(_CHBMIT_TIME_PATTERN, tstr)
    hour = int(matches["hour"])
    minute = int(matches["minute"])
    second = int(matches["second"])
    timestamp = hour * _SECONDS_IN_HOUR + minute * _SECONDS_IN_MINUTE + second
    return timestamp


def _parse_seizures(lines: list[str]):
    """Return a dictionary of all seizure timings in the line group `lines`."""
    seiz_times = list()
    for i in range(0, len(lines), 2):
        start = int(lines[i].strip(" seconds").split(": ")[1])
        end = int(lines[i + 1].strip(" seconds").split(": ")[1])
        seiz_times.append({ColNames.START: start, ColNames.END: end})
    return seiz_times


def _missing_records(prev_record, record):
    """Return the number of missing records based on the files identifier of two records."""
    match = re.match(_CHBMIT_NAME_PATTERN, prev_record[ColNames.FILE])
    prev_id = int(match["fid"])
    match = re.match(_CHBMIT_NAME_PATTERN, record[ColNames.FILE])
    id = int(match["fid"])
    return id - prev_id - 1


def _time_reset(prev_record, record):
    """Return True (1) if a 24h time reset occured from the previous to current record."""
    if record[ColNames.START] < prev_record[ColNames.END]:
        if prev_record[ColNames.END] <= _SECONDS_IN_DAY:
            return True
    return False


def _validate_continuity(records: pd.DataFrame):
    """Validate the start and end times and add the trial day to the recordings start and end times.

    Args:
        records (pd.DataFrame): The DataFrame containing the recordings information.

    Returns:
        None
    """
    avg_len = records[ColNames.LEN].sum() // len(records)
    for idx, record in records.iterrows():
        # No need to validate first record
        if idx == 0:
            prev_record = record
            continue
        correction = 0
        # Check if the missing records length is more than a day
        if avg_len * _missing_records(prev_record, record) > 86400:
            correction += 1
        # Check if a 24h time reset occured in the transition
        # This happens when recording started at a new day
        # For reference, check file chb11_58.edf
        if _time_reset(prev_record, record):
            correction += 1
        # Update the time and add the correction days
        day = records.at[idx - 1, ColNames.END] // _SECONDS_IN_DAY
        records.at[idx, ColNames.START] += (day + correction) * _SECONDS_IN_DAY
        records.at[idx, ColNames.END] += (day + correction) * _SECONDS_IN_DAY
        prev_record = record


class SummarySchema(pa.DataFrameModel):
    """Pandera DataFrame schema for the CHB-MIT summary file."""

    File: Series[str] = pa.Field(alias=ColNames.FILE, str_matches=_CHBMIT_NAME_PATTERN)
    Start: Series = pa.Field(alias=ColNames.START, str_matches=_CHBMIT_TIME_PATTERN)
    End: Series = pa.Field(alias=ColNames.END, str_matches=_CHBMIT_TIME_PATTERN)
    Length: Series[int] = pa.Field(alias=ColNames.LEN)
    Num_Seizures: Series[int] = pa.Field(alias=ColNames.NSEIZ, ge=0, coerce=True)
    Seizure_Times: Series = pa.Field(alias=ColNames.TSEIZ)

    @pa.dataframe_check
    def init_times(cls, records: DataFrame):
        records[ColNames.START] = records[ColNames.START].apply(_parse_time)
        records[ColNames.END] = records[ColNames.END].apply(_parse_time)
        return True

    @pa.dataframe_check
    def check_length(cls, records: DataFrame):
        records[ColNames.LEN] = records[ColNames.END] - records[ColNames.START]
        cls._avg_len = records[ColNames.LEN].sum() // len(records)
        return True

    @pa.dataframe_check
    def check_times(cls, records: DataFrame):
        _validate_continuity(records)
        return records[ColNames.START].lt(records[ColNames.END])

    @pa.dataframe_check
    def check_seizures(cls, records: DataFrame):
        records[ColNames.TSEIZ] = records[ColNames.TSEIZ].apply(_parse_seizures)
        return records.apply(
            lambda x: x[ColNames.NSEIZ] == len(x[ColNames.TSEIZ]), axis=1
        )


def read_summary(sdir: str) -> tuple[DataFrame[SummarySchema], str, int]:
    """Read and extract subject information from the summary text file located in the `sdir` directory.

    Args:
    sdir (str): Path string to the subject directory.

    Returns:
    Tuple[DataFrame, str, int]: A tuple containing the following:
        - DataFrame: A pandas DataFrame object that contains the following columns:
            - 'File': str, the name of the recording edf files.
            - 'Start': RecordTime, the start time of the recordings.
            - 'End': RecordTime, the end time of the recordings.
            - 'Length': int, the length of the recordings, in seconds.
            - 'Number of Seizures': int, the number of seizures in the recordings.
            - 'Seizure Times': list[dict], a list of dictionaries containing seizure times with keys:
                - 'Start': int, the relative time of seizure onset flag in seconds.
                - 'End': int, the relative time of seizure offset flag in seconds.
        - str: The subject id.
        - int: The sampling rate.
    """
    lines = _read_txt(sdir)
    subject_id = _parse_subject_id(sdir)
    sampling_rate = _parse_sampling_rate(lines)
    return _parse_lines(lines), subject_id, sampling_rate
