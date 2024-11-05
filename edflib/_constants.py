from typing import Literal, get_args


# plot constants
class PlotColors:
    backg: str = "#FFFFFF"
    record: str = "#7EAA92"
    seiz: str = "#d90429"
    inter: str = "#40A2D8"
    pre: str = "#FE7A36"
    gap: str = "#B4B4B8"
    annot: str = "#2D3250"
    axis: str = "#000000"


_RECT_BOTTOM: int = 0.3
_RECT_TOP: int = 0.6
_RECT_HEIGHT: int = _RECT_TOP - _RECT_BOTTOM
_ANNOT_OFFSET: float = 0.2 * _RECT_HEIGHT
_FLAG_OFFSET: float = 0.2 * _RECT_HEIGHT
_LINE_WIDTH: float = 1
_LARGE_FONT: int = 16
_SMALL_FONT: int = 14
_AXIS_FONT: int = 18
_LABEL_FONT: int = 24

# dataset constants
_SAMPLING_RATE: int = 256


# dataframe constants
class ColNames:
    FILE: str = "File"
    START: str = "Start"
    END: str = "End"
    LEN: str = "Length"
    TIME: str = "Time"
    SEIZ: str = "Seizure"
    NSEIZ: str = "N Seizures"
    TSEIZ: str = "Seizure Times"
    CLASS: str = "Class"


# class constants
ICTAL_TYPES = Literal["Interictal", "Preictal"]
_INTERICTAL, _PREICTAL = get_args(ICTAL_TYPES)

# time constants
_SECONDS_IN_DAY: int = 86400
_SECONDS_IN_HOUR: int = 3600
_SECONDS_IN_MINUTE: int = 60

# eeg constants
EEG_CHANNELS_SEL: list[str] = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]
