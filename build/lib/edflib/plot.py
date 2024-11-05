from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .epochs import Epochs
from .summary import Summary
from ._constants import (
    ColNames,
    _INTERICTAL,
    _PREICTAL,
    _SECONDS_IN_HOUR,
    _SECONDS_IN_MINUTE,
    _SAMPLING_RATE,
    EEG_CHANNELS_SEL,
)
from ._constants import (
    PlotColors,
    _RECT_BOTTOM,
    _RECT_TOP,
    _RECT_HEIGHT,
    _ANNOT_OFFSET,
    _FLAG_OFFSET,
    _LINE_WIDTH,
    _LARGE_FONT,
    _SMALL_FONT,
    _AXIS_FONT,
    _LABEL_FONT,
)


def _init_timeline_axes():
    """Initilize the figure and axes for the timeline plot."""

    # initialize the figure
    fig = plt.figure("edflib Plotter", figsize=(24, 6), layout="constrained")

    # initialize the axes
    ax = plt.subplot()
    # ax.set_title("Recordigs time information", loc="left", fontweight="bold")
    ax.set(facecolor=PlotColors.backg, yticks=[], ylim=[0, 1.8])

    # initialize the legends
    ax.legend(
        handles=[
            Patch(facecolor=PlotColors.record),
            Patch(facecolor=PlotColors.inter),
            Patch(facecolor=PlotColors.pre),
            Line2D(
                [],
                [],
                linestyle="None",
                marker="|",
                markersize=10,
                markeredgewidth=_LINE_WIDTH * 3,
                color=PlotColors.seiz,
            ),
        ],
        labels=["Excluded", _INTERICTAL, _PREICTAL, ColNames.SEIZ],
        fontsize=_LABEL_FONT,
        ncol=4,
        loc="upper right",
    )

    return fig, ax


def _draw_rect(ax: Axes, xcoords: tuple[int, int], color: str):
    """Draw a rectangle area on the `ax` with the `color` and `xcoords`."""
    start, end = xcoords
    length = end - start
    Rec = Rectangle(
        xy=(start, _RECT_BOTTOM),
        width=length,
        height=_RECT_HEIGHT,
        color=color,
    )
    ax.add_patch(Rec)


def _draw_flag(ax: Axes, x: int):
    """Draw a vertical line representing a seizure on the `ax`."""
    arr = FancyArrowPatch(
        posA=(x, _RECT_BOTTOM - _FLAG_OFFSET),
        posB=(x, _RECT_TOP + _FLAG_OFFSET),
        linewidth=_LINE_WIDTH * 2,
        color=PlotColors.seiz,
    )
    ax.add_patch(arr)


def _annotate_label(ax: Axes, xcoords: tuple[int, int], label: str):
    """Annotate the area `label` on the `ax` at the center of the `xcoords`."""
    start, end = xcoords
    width = end - start
    wlabel = f"({width//60:d}m)"
    rec = Rectangle(
        xy=(start, _RECT_BOTTOM),
        width=width,
        height=_RECT_HEIGHT,
        linewidth=_LINE_WIDTH,
        fill=None,
        edgecolor=PlotColors.axis,
    )
    annot = ax.annotate(
        label,
        xy=(0.5, 1 + _ANNOT_OFFSET),
        xycoords=rec,
        xytext=(0, 1),
        textcoords="offset fontsize",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=_LARGE_FONT,
        fontweight="bold",
        color=PlotColors.annot,
    )
    ax.annotate(
        wlabel,
        xy=(0.5, 1),
        xycoords=annot,
        xytext=(0, 0.5),
        textcoords="offset fontsize",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=_SMALL_FONT,
        fontweight="bold",
        color=PlotColors.record,
    )
    ax.add_patch(rec)


def _annotate_width(
    ax: Axes,
    xcoords: tuple[int, int],
    width: int = -1,
    where: str = "top",
):
    """Annotation of the width of the area `xcoords` on the `ax` at the `where` position."""
    start, end = xcoords

    if width < 0:
        width = int(end - start)
    label = f"{width//60:d}m"

    if where == "top":
        y = _RECT_TOP + _ANNOT_OFFSET
        l_offet = 5
        l_va = "bottom"
    elif where == "bottom":
        y = _RECT_BOTTOM - _ANNOT_OFFSET
        l_offet = -5
        l_va = "top"

    arr = FancyArrowPatch(
        posA=(start, y),
        posB=(end, y),
        arrowstyle="|-|, widthA=2, widthB=2",
        linewidth=_LINE_WIDTH * 2,
        color=PlotColors.annot,
    )
    ax.annotate(
        label,
        xy=(0.5, 0.5),
        xycoords=arr,
        xytext=(l_offet, l_offet),
        textcoords="offset pixels",
        ha="center",
        va=l_va,
        rotation=45,
        fontsize=_SMALL_FONT,
        color=PlotColors.annot,
    )
    ax.add_patch(arr)


def _get_epochs(epochs: Epochs):
    """Return the interictal and preictal epochs tables with absolute timing information."""
    records = (
        epochs.records.astype({ColNames.START: int, ColNames.END: int}, copy=True)
        .filter([ColNames.FILE, ColNames.START, ColNames.END])
        .set_index(ColNames.FILE)
    )
    records[ColNames.END] = records[ColNames.START]

    ii_df = (
        epochs.get_epochs_table(_INTERICTAL, evenly_divided=False)
        .reset_index()
        .filter([ColNames.SEIZ, ColNames.FILE, ColNames.START, ColNames.END])
        .set_index(ColNames.FILE)
    )
    ii_df[[ColNames.START, ColNames.END]] += records.loc[ii_df.index]
    ii_df = ii_df.reset_index().set_index([ColNames.SEIZ, ColNames.FILE])

    pi_df = (
        epochs.get_epochs_table(_PREICTAL, evenly_divided=False)
        .reset_index()
        .filter([ColNames.SEIZ, ColNames.FILE, ColNames.START, ColNames.END])
        .set_index(ColNames.FILE)
    )
    pi_df[[ColNames.START, ColNames.END]] += records.loc[pi_df.index]
    pi_df = pi_df.reset_index().set_index([ColNames.SEIZ, ColNames.FILE])

    return ii_df, pi_df


def _get_records_times(epochs: Epochs):
    """Return the records table indexed by file names."""
    return (
        epochs.records.copy()
        .astype({ColNames.START: int, ColNames.END: int})
        .filter([ColNames.FILE, ColNames.START, ColNames.END])
        .set_index(ColNames.FILE)
    )


def _get_seizures(epochs: Epochs):
    """Return a list of absolute timing of all seizures in the epochs."""
    return [
        seiz_start + int(epochs.records.at[seiz_idx, ColNames.START])
        for seiz_idx, _, seiz_start, _ in epochs.seizures
    ]


def _set_timeaxis(ax: Axes, epochs: Epochs):
    """Set the time x-axis of the `ax` based on the `epochs` information."""
    start = int(epochs.records[ColNames.START].iloc[0])
    end = int(epochs.records[ColNames.END].iloc[-1])
    len_total = epochs.len_total
    margin = len_total * 0.01

    major_ticks = np.arange(start, end, 4 * _SECONDS_IN_HOUR)
    minor_ticks = np.arange(start, end, _SECONDS_IN_HOUR)
    major_labels = np.arange(0, len_total / _SECONDS_IN_HOUR, 4, dtype=int)

    ax.set_xlabel("Time (hours)", fontsize=_LABEL_FONT)
    ax.set_xlim(start - margin, end + margin)
    ax.set_xticks(major_ticks, major_labels, fontsize=_AXIS_FONT)
    ax.set_xticks(minor_ticks, minor=True)
    ax.axhspan(_RECT_BOTTOM, _RECT_TOP, color=PlotColors.gap)


def timeline(epochs: Epochs):
    """Plot the timeline of the recordings' interictal, and preictal epochs."""
    records = _get_records_times(epochs)
    ii_epochs, pi_epochs = _get_epochs(epochs)
    seizures = _get_seizures(epochs)

    fig, ax = _init_timeline_axes()
    _set_timeaxis(ax, epochs)

    # Plot recordings excluded time
    for _, record in records.iterrows():
        _draw_rect(ax, xcoords=record.values, color=PlotColors.record)
    # Plot recordings inter time
    for _, epoch in ii_epochs.iterrows():
        _draw_rect(ax, xcoords=epoch.values, color=PlotColors.inter)
    for _, group in ii_epochs.groupby(level=0):
        first_iter = True
        annotated = False
        for idx, epoch in group.iterrows():
            start = epoch[ColNames.START]
            if first_iter:
                gstart = epoch[ColNames.START]
                gend = epoch[ColNames.END]
                first_iter = False
                continue
            if start - gend > 600:
                _annotate_width(ax, xcoords=(gstart, gend), where="bottom")
                gstart = start
                annotated = True
                # Annotate last epoch if a gap exist before
                if idx[1] in idx[0]:
                    annotated = False
            else:
                annotated = False
            gend = epoch[ColNames.END]
        if not annotated:
            _annotate_width(ax, xcoords=(gstart, gend), where="bottom")

    # Plot recordings pre time
    for _, epoch in pi_epochs.iterrows():
        _draw_rect(ax, xcoords=epoch.values, color=PlotColors.pre)
    for _, group in pi_epochs.groupby(level=0):
        glen = int(group[ColNames.END].sum() - group[ColNames.START].sum())
        gstart = group[ColNames.START].iloc[0]
        gend = group[ColNames.END].iloc[-1]
        _annotate_width(ax, xcoords=(gstart, gend), width=glen, where="bottom")

    # Annotate recordings name and time
    for _, record in records.iterrows():
        # _annotate_width(ax, xcoords=record.values, where="top")
        _annotate_label(ax, xcoords=record.values, label=record.name)

    # Plot recordings seiz flag
    for seiz_flag in seizures:
        _draw_flag(ax, seiz_flag)


def summarize_cont(summary: Summary, pil: int = None, psl: int = None, iid: int = None):
    """Print the summary of the records classified into the four epileptic states."""
    records = summary.records.copy()
    total_len = summary.len_total
    row_len = 2 * _SECONDS_IN_HOUR
    nrows = total_len // row_len + 1
    beginning = records[ColNames.START].iloc[0]
    records[[ColNames.START, ColNames.END]] -= beginning

    # initialize the figure
    fig = plt.figure(
        "edflib Plotter",
        figsize=(16, nrows * 0.3),
        layout="constrained",
    )
    plt.rcParams["hatch.linewidth"] = 0.5

    # Plot the background
    for row in range(nrows):
        ax = plt.subplot(nrows, 1, row + 1)
        ax.set_facecolor(PlotColors.backg)
        ax.set_yticks([0], [f"{row*row_len//_SECONDS_IN_HOUR:02d}:00"])
        ax.set_ylim(-0.1, 0.1)
        ax.set_xticks(np.arange(0, row_len, _SECONDS_IN_MINUTE * 6), [])
        ax.set_xlim(0, row_len)

    # Plot the records
    for _, (start, end) in records[[ColNames.START, ColNames.END]].iterrows():
        start_row, start_time = divmod(start, row_len)
        end_row, end_time = divmod(end, row_len)
        if start_row == end_row:
            ax = fig.axes[start_row]
            ax.axvspan(start_time, end_time, color=PlotColors.record)
        else:
            ax = fig.axes[start_row]
            ax.axvspan(start_time, row_len, color=PlotColors.record)
            ax = fig.axes[end_row]
            ax.axvspan(0, end_time, color=PlotColors.record)

    # Plot the seizures
    seizures = [
        (
            seiz_start + int(records.at[seiz_idx, ColNames.START]),
            seiz_end + int(records.at[seiz_idx, ColNames.START]),
        )
        for seiz_idx, _, seiz_start, seiz_end in summary.seizures
    ]
    for start, end in seizures:
        row = start // row_len
        ax = fig.axes[row]
        start = start % row_len
        end = end % row_len
        ax.axvspan(start, end, color=PlotColors.seiz)

    # Plot the interictal periods
    if iid is not None:
        periods = summary.get_interictal_epochs(iid)
        for _, (_, file, start, end, _) in periods.iterrows():
            idx = records[ColNames.FILE] == file
            start += records[ColNames.START].loc[idx].values[0]
            end += records[ColNames.START].loc[idx].values[0]
            start_row, start_time = divmod(start, row_len)
            end_row, end_time = divmod(end, row_len)
            if start_row == end_row:
                ax = fig.axes[start_row]
                ax.axvspan(start_time, end_time, color=PlotColors.inter)
            else:
                ax = fig.axes[start_row]
                ax.axvspan(start_time, row_len, color=PlotColors.inter)
                ax = fig.axes[end_row]
                ax.axvspan(0, end_time, color=PlotColors.inter)

    # Plot the preictal periods
    if pil is not None and psl is not None:
        periods = summary.get_preictal_epochs(0, pil, psl)
        for _, (_, file, start, end, _) in periods.iterrows():
            idx = records[ColNames.FILE] == file
            start += records[ColNames.START].loc[idx].values[0]
            end += records[ColNames.START].loc[idx].values[0]
            start_row, start_time = divmod(start, row_len)
            end_row, end_time = divmod(end, row_len)
            if start_row == end_row:
                ax = fig.axes[start_row]
                ax.axvspan(start_time, end_time, color=PlotColors.pre)
            else:
                ax = fig.axes[start_row]
                ax.axvspan(start_time, row_len, color=PlotColors.pre)
                ax = fig.axes[end_row]
                ax.axvspan(0, end_time, color=PlotColors.pre)


def summarize(
    summary: Summary, pil: int = None, psl: int = None, iid: int = None, figlabel=None
):
    """Print the summary of the records classified into the four epileptic states."""
    records = summary.records.copy()
    row_len = records[ColNames.LEN].max()
    nrows = summary.num_records

    # initialize the figure
    label = f" - {figlabel}" if figlabel is not None else ""
    fig = plt.figure(
        f"edflib Plotter{label}",
        figsize=(16, nrows * 0.3),
        layout="constrained",
    )
    plt.rcParams["hatch.linewidth"] = 0.5

    # Plot the background
    for row in range(nrows):
        ax = plt.subplot(nrows, 1, row + 1)
        ax.set_facecolor(PlotColors.backg)
        ax.set_yticks([0], [records.at[row, ColNames.FILE]])
        ax.set_ylim(-0.1, 0.1)
        ax.set_xticks(np.arange(0, row_len, _SECONDS_IN_MINUTE * 6), [])
        ax.set_xlim(0, row_len)

    # Plot the records
    for row, length in enumerate(records[ColNames.LEN].values):
        ax = fig.axes[row]
        ax.axvspan(0, length, color=PlotColors.record)

    # Plot the seizures
    for row, _, start, end in summary.seizures:
        ax = fig.axes[row]
        ax.axvspan(start, end, color=PlotColors.seiz)

    # Plot the interictal periods
    if iid is not None:
        periods = summary.get_interictal_epochs(iid)
        for _, (_, file, start, end, _) in periods.iterrows():
            row = np.argmax(records[ColNames.FILE] == file)
            ax = fig.axes[row]
            ax.axvspan(start, end, color=PlotColors.inter)

    # Plot the preictal periods
    if pil is not None and psl is not None:
        periods = summary.get_preictal_epochs(0, pil, psl)
        for _, (_, file, start, end, _) in periods.iterrows():
            row = np.argmax(records[ColNames.FILE] == file)
            ax = fig.axes[row]
            ax.axvspan(start, end, color=PlotColors.pre)


def plot(data: NDArray):
    """Plot the EEG data `data` with the `EEG_CHANNELS_SEL` as y-axis labels."""
    nsamples = data.shape[-1]
    nchannels = data.shape[-2]
    t = np.arange(0, nsamples)

    ax = plt.subplot()
    ax.set(
        facecolor=PlotColors.backg,
        xlabel="time(sec)",
        xmargin=0.01,
        yticks=[],
    )

    # dx = int(5 * np.ceil(nsamples / (_SECONDS_IN_HOUR * 4)))
    # xticks = np.arange(0, nsamples, _SAMPLING_RATE * dx)
    # xlabels = np.arange(0, nsamples / _SAMPLING_RATE, dx)
    # ax.set_xticks(ticks=xticks, labels=xlabels)

    dy = 0.0001 * 5
    yticks = np.flip([dy * i for i in range(nchannels)])
    ylabels = [ch[:6].rstrip("-") for ch in EEG_CHANNELS_SEL]
    ax.set_yticks(ticks=yticks, labels=ylabels)
    for channel, center in zip(data, yticks):
        ax.plot(t, channel + center, color="black", linewidth=0.5)
    return ax
