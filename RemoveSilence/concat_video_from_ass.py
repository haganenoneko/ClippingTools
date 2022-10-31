# ---------------------------------------------------------------------------- #
#         Use dialog timestamps in an .ASS file to concatenate a video         #
# ---------------------------------------------------------------------------- #

import enum
import tkinter as tk
from tkinter.filedialog import askopenfilenames

import re
import numpy as np

from pathlib import Path
from typing import Generator
from subprocess import Popen

# ---------------------------------------------------------------------------- #
#                            Functions and constants                           #
# ---------------------------------------------------------------------------- #

INITDIR = Path("C:/Users/delbe/Videos/subtitles/subs/")
DIALOG_PAT = re.compile(r"^(?:Dialogue\:\s\d\,)([\d\:\.]+),([\d\:\.]+).*$")

# list[str] -> tuple[list[str], list[str]]
parse_dialog = np.vectorize(lambda x: DIALOG_PAT.match(x).groups())


def ts2secs(ts: str) -> float:
    hms, msec = ts.split('.')
    h, m, s = map(int, hms.split(':'))
    return 3600*h + 60*m + s + int(msec)/100


# list[str] -> np.ndarray[float]
vts2secs = np.vectorize(lambda x: ts2secs(x))


def open_fd(initdir: Path = INITDIR) -> list[str]:
    """
    Initialize file dialog at `initdir` and return list of selected .ASS file paths
    """
    root = tk.Tk()
    filenames = askopenfilenames(
        multiple=True,
        initialdir=initdir,
        filetypes=[("ASS files", "*.ass")],
        title="Select .ASS files to parse"
    )
    root.destroy()
    return filenames


def read_ass_file(fp: Path) -> tuple[Path, list[str]]:
    """Extract dialog lines and `Path` to corresponding video for an input .ASS file

    Args:
        fp (Path): path to .ASS file

    Raises:
        ValueError: Dialog found, but no video.
        ValueError: No dialog (Video may have been found)

    Returns:
        tuple[Path, list[str]]: video `Path` and dialog lines
    """
    with open(fp, 'r', encoding='utf-8') as io:
        lines = io.readlines()

    video_path = None
    for i, line in enumerate(lines):

        if "Video File:" == line[:11]:
            video_path = Path(line.split(': ')[1].strip())
            continue

        if "Dialogue:" == line[:9]:
            if video_path is None:
                raise ValueError(f"No video found, but dialog was found")

            return video_path, lines[i:]

    raise ValueError(f"No dialogue found in {fp}")

def _merge(starts: np.ndarray, ends: np.ndarray) -> tuple[float, float]:
    return starts.min(), ends.max()

def extract_timestamps(dialog: list[str]) -> list[tuple[float, float]]:
    raise NotImplementedError()

    # extract timestamps for each dialog line 
    starts, ends = list(map(vts2secs, parse_dialog(dialog)))

    # sort lines by ascending start time 
    starts, ends = sorted(zip(starts, ends), key=lambda tup: tup[0])

    if isinstance(starts, list):
        starts = np.array(starts)
        ends = np.array(ends)

    isDup = (starts[1:] - starts[:-1]) < 0.05
    # while isDup.any():


def create_splice_pair(ind: int, start_end: tuple[float]) -> str:
    """Add single command for `ffmpeg` to concatenate a video from `start_end[0]` to `start_end[1]`
    """
    t0, t1 = start_end
    return f"[0:v]trim=start={t0}:end={t1},setpts=PTS-STARTPTS[{ind}v]; [0:a]atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS[{ind}a];"


def get_filter(
        intervals: Generator[tuple, None, None]) -> tuple[int, str]:
    """Get the (long part) of the `ffmpeg` `concat` filter

    Args:
        intervals (Generator[tuple, None, None]): an iterator for tuples of `(start_time, end_time)` for each interval to concatenate

    Returns:
        tuple[int, str]: number of intervals to concatenate, and (part of) the `concat` filter for `ffmpeg`
    """
    num = 0
    pairs = ''
    suffix = ''

    for ind, interval in enumerate(intervals):
        pairs += f"{create_splice_pair(ind, interval)}\n"
        suffix += f"[{ind}v][{ind}a]"
        num += 1

    return num, f"{pairs} {suffix}"


def parse_ass_file(fp: Path, run_concat=True) -> str:
    """Parse .ASS file and construct command to concatenate intervals with subtitles, assuming all intervals are non-overlapping.

    Args:
        fp (Path): path to .ASS file. To concatenate the corresponding video file, a video file must be specified in the .ASS file.
        run_concat (bool, optional): whether to run the final `ffmpeg` function. Defaults to True.

    Returns:
        str: `ffmpeg` command for concatenating dialog intervals
    """
    vp, dialog = read_ass_file(fp)

    intervals = zip(*map(vts2secs, parse_dialog(dialog)))
    num, filters = get_filter(intervals)
    suffix = f"concat=n={num}:v=1:a=1[outv][outa]"

    concat_ = f"-filter_complex \"{filters}{suffix}\""
    map_ = "-map [outv] -map [outa]"
    outname = vp.parent / f"{vp.stem}_concat.mp4"

    cmd = f"ffmpeg -hide_banner -i " +\
        f"\"{vp}\" {concat_} {map_} {outname}"

    print(f"Concatenating {num} streams to:\n{outname}")
    if run_concat:
        try:
            Popen(['powershell.exe', cmd])
        except:
            raise RuntimeError(f"Failed to concatenate {fd} due to\n{e}")

    return cmd

# ---------------------------------------------------------------------------- #
#                Function to execute on, e.g. file double-click                #
# ---------------------------------------------------------------------------- #

def main(run_concat=True):

    files = open_fd()
    for f in files:
        parse_ass_file(f, run_concat=run_concat)
        # try:
        # except Exception as e:
        #     if isinstance(e, RuntimeError):
        #         continue
        #     elif isinstance(e, KeyboardInterrupt):
        #         print("Interrupted. Files found:\n{files}")
        #         break


if __name__ == '__main__':
    main()
