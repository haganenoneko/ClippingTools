import re
from tabnanny import check
from tkinter import dialog
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from datetime import timedelta

from common import check_overwrite
# from RemoveSilence.common import check_overwrite

INIT_DIR = Path("C:/Users/delbe/Videos/subtitles/subs/")

HOME_DIR = Path.cwd() / "RemoveSilence"
ASS_HEADER_PATH = HOME_DIR / "ASS_header.txt"

# ---------------------------------------------------------------------------- #
#                                Regex patterns                                #
# ---------------------------------------------------------------------------- #

DIALOG_PAT = re.compile(r"^(?:Dialogue\:\s\d\,)([\d\:\.]+),([\d\:\.]+).*$")

DIALOG_TEMPLATE =\
    "Dialogue: {layer},{start_time},{end_time},{style},{name},{marginL},{marginR},{marginV},{effect},{text}"

# ---------------------------------------------------------------------------- #
#                               Writing ASS files                              #
# ---------------------------------------------------------------------------- #


class ASSWriter:
    def __init__(
        self,
        dialog_template: str = DIALOG_TEMPLATE,
        dialog_kwargs: dict[str, str] = dict(
            layer=0, style='Default', name='',
            marginL=0, marginR=0, marginV=0, effect=''),
        header: Union[str, Path] = ASS_HEADER_PATH,
    ) -> None:

        self._dialogTemplate = dialog_template
        self._dialog_kw = dialog_kwargs
        self._headerPath = header

        self._vec_ts = np.vectorize(self._ts)

    @staticmethod
    def _ts(seconds: float) -> str:
        """Convert seconds into a HH:MM:SS timestamp"""
        return str(timedelta(seconds=seconds))

    def add_dialog(self, text: str, start_time: float, end_time: float, **kwargs) -> str:
        """Create single line of ASS dialog"""

        return self._dialogTemplate.format(
            start_time=start_time, end_time=end_time, text=text,
        )

    def create_ass_dialog(
            self,
            times: list[tuple[float, float]],
            text: list[str] = None,) -> str:

        dialog: str = ''
        tstamps: list[str] = self._vec_ts(times)

        for i in range(0, len(times), 2):
            start, end = tstamps[i:i+2]
            line = self.add_dialog(
                text[int(i/2)] if text else 'n/a',
                start, end,
                **self._dialog_kw)

            dialog += f"{line}\n"

        return dialog.rstrip()

    @staticmethod
    def add_media(
            header: str,
            vpath: Union[str, Path],
            apath: Union[str, Path]) -> str:

        if vpath:
            if not isinstance(vpath, Path):
                vpath = Path(vpath)

        if not isinstance(apath, Path):
            apath = Path(apath)

        lines = header.split('\n')
        if vpath:
            lines[13] += str(vpath.absolute())
        if apath:
            lines[14] += str(apath.absolute())

        return '\n'.join(lines)

    def get_header(
            self,
            header: Union[str, Path],
            vpath: Union[str, Path],
            apath: Union[str, Path]) -> str:
        """
        Get header for the ASS file
        TO DO: add path to video/audio files
        """

        if isinstance(header, Path):
            with open(header, 'r', encoding='utf-8') as io:
                header = io.read()

        if (vpath is None) and\
                (apath is None):
            return header

        if vpath:
            if apath:
                header = self.add_media(header, vpath, apath)
            else:
                header = self.add_media(header, vpath, vpath)
        else:
            header = self.add_media(header, None, apath)

        return header

    def write(
            self,
            outname: str,
            seconds: list[float],
            text: list[str] = None,
            outdir: Path = HOME_DIR,
            vpath: Union[str, Path] = None,
            apath: Union[str, Path] = None,) -> str:
        """Create a new ASS file based on list of times and (optional) dialog text."""

        ass_header = self.get_header(self._headerPath, vpath, apath)
        dialog = self.create_ass_dialog(seconds, text)
        full_content = ass_header + '\n' + dialog

        if not isinstance(outdir, Path):
            raise TypeError(f"{outdir} must be a Path")
        if not outdir.is_dir():
            raise FileNotFoundError(f"{outdir} is not a valid directory.")

        outpath = outdir / f"{outname}.ass"

        if outpath.is_file():
            overwrite = input(
                f"{outpath} already exists. Overwrite? [y/n]").lower()
            if overwrite != 'y':
                raise FileExistsError(outpath)

        with open(outpath, 'w', encoding='utf-8') as io:
            io.write(full_content)

        print(f"New ass file created at {outpath}")
        return full_content


def parse_interval_file(fpath: Path) -> list[float]:
    ext = fpath.suffix
    if ext in ['.csv', '.tsv']:
        df = pd.read_csv(
            fpath, header=None, index_col=None,
            sep=',' if ext == '.csv' else '\t'
        )

        df = pd.read_csv(fpath, header=None)
        secs = df.values.flatten()
        secs.sort()

        return secs

    if ext == '.txt':
        with open(ext, 'r', encoding='utf-8') as file:
            return [float(x.strip()) for x in file.read().split(',')]

    raise Exception(f"{fpath} must have .csv, .tsv, or .txt extension")


# ---------------------------------------------------------------------------- #
#                               Reading ASS files                              #
# ---------------------------------------------------------------------------- #

class ASSReader:
    def __init__(self) -> None:

        # list[str] -> np.np.ndarray[float]
        self._v_ts2secs = np.vectorize(
            lambda x: self._ts2secs(x)
        )

        # list[str] -> tuple[list[str], list[str]]
        self._v_extractTimes = np.vectorize(
            lambda x: DIALOG_PAT.match(x).groups()
        )

    @staticmethod
    def _ts2secs(s: str) -> float:
        hms, msec = s.split('.')
        h, m, s = map(int, hms.split(':'))
        return 3600*h + 60*m + s + int(msec)/100

    def get_intervals(self, lines: list[str]) -> list[tuple[float, float]]:
        return list(zip(*map(
            self._v_ts2secs,
            self._v_extractTimes(lines))))

    @staticmethod
    def clean_intervals(
        intervals: list[tuple],
        min_dur: float,
        min_gap: float,
    ) -> list[tuple[float, float]]:
        """Clean up intervals by

            1. ensuring intervals are greater than a minimum duration
            2. ensuring intervals are separated by a minimum duration
            3. concatenating overlapping intervals (https://stackoverflow.com/a/58976449/10120386)

        Args:
            intervals (list[tuple]): list of `(start, end)` times
            min_dur (float, optional): minimum interval duration. Defaults to 0.1.
            min_gap (float, optional): minimum separation between adjacent intervals. Defaults to 0.2.

        Returns:
            list[tuple[float, float]]: list of processed `(start, end)` times
        """

        arr = np.sort(np.array(intervals), axis=0)

        # remove intervals with less than the minimum duration
        durs = arr[:, 1] - arr[:, 0]
        arr = arr[durs > min_dur]

        # remove overlapping intervals and ensure gaps > min_gap
        valid = np.zeros(arr.shape[0]+1, dtype=bool)
        valid[[0, -1]] = 1
        valid[1:-1] = arr[1:, 0] -\
            np.maximum.accumulate(arr[:-1, 1]) >= min_gap

        merged = np.vstack(
            (
                arr[valid[:-1], 0],
                arr[valid[1:], 1]
            )
        ).T

        return tuple(map(tuple, merged))

    def read_dialog(self, fp: Path) -> tuple[Path, list[str]]:
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

    def read_intervals(
            self,
            lines: list[str],
            clean=True,
            min_dur=0.1, min_gap=0.2) -> list[float]:

        intervals = self.get_intervals(lines)
        if clean:
            return self.clean_intervals(intervals, min_dur, min_gap)
        else:
            return intervals

    @staticmethod
    def intervals2csv(outpath: Path, intervals: list[tuple[float, float]]) -> pd.DataFrame:

        df = pd.DataFrame.from_records(intervals)
        df.columns = ['Start', 'End']

        if check_overwrite(outpath):
            df.to_csv(outpath)

        return df 