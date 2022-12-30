# ---------------------------------------------------------------------------- #
#         Use dialog timestamps in an .ASS file to concatenate a video         #
# ---------------------------------------------------------------------------- #

import tkinter as tk
from tkinter import filedialog as fd

import re
import numpy as np

from pathlib import Path
from typing import Generator
from subprocess import Popen

from ass_tools import clean_intervals

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
    filenames = fd.askopenfilenames(
        multiple=True,
        initialdir=initdir,
        filetypes=[("ASS files", "*.ass")],
        title="Select .ASS files to parse"
    )
    root.destroy()
    return filenames


def read_ass_file(fp: Path, no_comment: bool) -> tuple[Path, list[str]]:
    """Extract dialog lines and `Path` to corresponding video for an input .ASS file

    Args:
        fp (Path): path to .ASS file
        no_comment (bool): whether to remove Comment lines

    Raises:
        ValueError: Dialog found, but no video.
        ValueError: No dialog (Video may have been found)

    Returns:
        tuple[Path, list[str]]: video `Path` and dialog lines
    """
    with open(fp, 'r', encoding='utf-8') as io:
        lines = io.readlines()

    video_path = None
    subtitles = None 
    for i, line in enumerate(lines):

        if "Video File:" == line[:11]:
            video_path = Path(line.split(': ')[1].strip())
            continue

        if "Dialogue:" == line[:9]:
            subtitles = [
                l for l in lines[i:] 
                if "Dialogue:" == l[:9]
            ]
            break 
    
    if subtitles is None:
        raise ValueError(f"No dialogue found in {fp}")
    else:
        return video_path, subtitles

def _merge(starts: np.ndarray, ends: np.ndarray) -> tuple[float, float]:
    return starts.min(), ends.max()

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

def select_video(fp: Path) -> Path:
    root = tk.Tk()
    vp = fd.askopenfilename(
        initialdir=fp.parent, 
        title="Select video file to concatenate.",
        filetypes=(("MP4 files", "*.mp4"),)
    )
    root.destroy()
    vp = Path(vp)
    return vp 


def parse_ass_file(fp: Path, select_vp=False, run_concat=True, confirm=True) -> str:
    """Parse .ASS file and construct command to concatenate intervals with subtitles, assuming all intervals are non-overlapping.

    Args:
        fp (Path): path to .ASS file. To concatenate the corresponding video file, a video file must be specified in the .ASS file.
        run_concat (bool, optional): whether to run the final `ffmpeg` function. Defaults to True.

    Returns:
        str: `ffmpeg` command for concatenating dialog intervals
    """
    if select_vp:
        vp = select_video(fp)
        _, dialog = read_ass_file(fp, no_comment=True)
    else:
        vp, dialog = read_ass_file(fp, no_comment=True)
    
    if vp is None:
        vp = select_video(fp)

    try:
        raw_intervals = zip(*map(vts2secs, parse_dialog(dialog)))
        intervals = clean_intervals(list(raw_intervals))
    except Exception as e:
        print(dialog)
        raise e

    if confirm:
        print(f"The following intervals will be used.", intervals, sep='\n\n')
        if input("Continue program? [y/n]").lower() == 'n':
            return 

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

def main(select_vp=False, run_concat=True):

    files = open_fd()
    for f in files:
        parse_ass_file(
            Path(f), 
            run_concat=run_concat,
            select_vp=select_vp
        )

if __name__ == '__main__':
    main(select_vp=True)
