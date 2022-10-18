import logging
import regex as re
import numpy as np
import pandas as pd
from typing import Any, Union
from pathlib import Path

from common import check_overwrite, create_logger, AudioClip

DIAGPAT = re.compile(
    "^Dialogue: 0,([\d:\.]{10}),([\d:\.]{10}),([^,]+)(?=.*)"
)


class ASSReader:
    """
    Class to load an audio file and parse its associated subtitles
    """

    def __init__(
        self, filename: str,
        dialog_pattern: Union[str, re.Pattern] = DIAGPAT,
    ) -> None:

        self.dialog_pattern = dialog_pattern
        self.read_ass(filename)

    def read_ass(self, filename: str) -> None:
        with open(filename, 'r', encoding='utf-8') as io:
            subs = io.readlines()

        self.subs = subs

    def get_raw_dialog(self, skiprows: int) -> pd.DataFrame:
        """Create dataframe from subtitles
        Includes all subtitle lines, and matches the line numbering in Aegisub.
        ```python
        >>> df.head()
            0
        0  Dialogue: 0,0:00:00.00,0:00:05.00,Ichinose,,0,...
        1  Dialogue: 0,0:00:00.00,0:00:05.00,Tamaki,,0,0,...
        2  Dialogue: 0,0:00:03.48,0:00:05.82,Tamaki,,0,0,...
        3  Dialogue: 0,0:00:05.82,0:00:07.71,Tamaki,,0,0,...
        4  Dialogue: 0,0:00:07.71,0:00:08.85,Tamaki,,0,0,...
        ```
        """

        for ind, line in enumerate(self.subs):
            if "[Events]" in line:
                break

        df_raw = pd.DataFrame(self.subs[ind+skiprows+2:]).\
            reset_index(drop=True)

        df_raw.index += skiprows + 1
        return df_raw

    @staticmethod
    def parse_line(line: str, pat: re.Pattern) -> list[str]:
        """Extract start time, stop time, and speaker from a subtitle line
        ```python
        >>> test = r"Dialogue: 0,0:00:05.82,0:00:07.71,Tamaki,,0,0,0,,same discord server"

        >>> Reader().parse_line(test, DIAGPAT)
        ('0:00:05.82', '0:00:07.71', 'Tamaki')
        ```
        """
        return pat.search(line).groups()

    def parse_dialog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract start time, stop time, and speaker from dataframe of subtitles
        ```python
        >>> df.head()
                Start         End   Speaker
        0  0:00:03.48  0:00:05.82    Tamaki
        1  0:00:05.82  0:00:07.71    Tamaki
        2  0:00:07.71  0:00:08.85    Tamaki
        3  0:00:08.85  0:00:10.00  Ichinose
        4  0:00:10.00  0:00:14.05    Tamaki
        ```
        """
        rename_cols = {i: col for i, col in enumerate(
            ["Start", "End", "Speaker"])}

        df = df.iloc[:, 0].apply(lambda s: pd.Series(
            self.parse_line(s, self.dialog_pattern)
        )).\
            rename(rename_cols, axis=1).\
            sort_values(by="Start", ascending=True)

        return df

    @staticmethod
    def ts2secs(ts: str) -> float:
        """Convert timestamp from string to seconds"""
        HH, MM, SS_ms = [float(x) for x in ts.split(":")]
        return 3600*HH + 60*MM + SS_ms

    @staticmethod
    def get_sub_times(
        df: pd.DataFrame, rate: float, func: staticmethod
    ) -> pd.DataFrame:
        """Convert timestamps from string to `datetime`s and `int`s (no. of seconds, samples)
        ```python
        >>> df.head()
                Start         End  ... End_seconds  End_samples
        0  0:00:03.48  0:00:05.82  ...        5.82       256662       
        1  0:00:05.82  0:00:07.71  ...        7.71       340011       
        2  0:00:07.71  0:00:08.85  ...        8.85       390285       
        3  0:00:08.85  0:00:10.00  ...       10.00       441000       
        4  0:00:10.00  0:00:14.05  ...       14.05       619605
        ```
        """

        for col in ['Start', 'End']:
            df[f"{col}_seconds"] = df[col].apply(
                lambda ts: func(ts)
            )

            df[f"{col}_samples"] = (
                df[f"{col}_seconds"] * rate
            ).astype(int)

        return df

    @staticmethod
    def rename_ts_index_cols(df: pd.DataFrame, dropcols: list[str]) -> pd.DataFrame:
        """Rename index and columns of timestamp dataframe
        ```python
        >>> df_ts.head()
                             Start         End  ...  End_seconds  End_samples
        Speaker                                 ...
        Ichinose    3   0:00:08.85  0:00:10.00  ...        10.00       441000
                    9   0:00:22.45  0:00:29.11  ...        29.11      1283751
                    10  0:00:29.11  0:00:33.82  ...        33.82      1491462
                    11  0:00:33.82  0:00:36.39  ...        36.39      1604799
                    12  0:00:36.39  0:00:38.91  ...        38.91      1715930
        """
        df_ts = df.copy()
        if dropcols:
            df_ts.drop(
                df_ts.loc[df_ts.Speaker.isin(dropcols), :].index,
                axis=0, inplace=True
            )

        return df_ts.set_index(['Speaker', df_ts.index]).sort_index()

    def parse_subs(self, Hz: int, skiprows: int = 0) -> pd.DataFrame:
        """Extract timestamps from subtitles into dataframe
        ```python
                   Start         End  ... End_seconds  End_samples
        0     0:00:03.48  0:00:05.82  ...        5.82       256662
        1     0:00:05.82  0:00:07.71  ...        7.71       340011
        2     0:00:07.71  0:00:08.85  ...        8.85       390285
        3     0:00:08.85  0:00:10.00  ...       10.00       441000
        4     0:00:10.00  0:00:14.05  ...       14.05       619605
        ```
        """

        df = self.get_raw_dialog(skiprows)
        df = self.parse_dialog(df)

        df_ts = self.get_sub_times(df, Hz, self.ts2secs)
        df_ts = self.rename_ts_index_cols(df_ts, ['Default', 'Translator'])

        self.df_subs = df
        self.df_ts = df_ts

        return df

    def __str__(self) -> str:

        if self.df_subs is None or self.df_ts is None:
            raise AttributeError()

        n_speakers = self.df_subs['Speaker'].unique().shape[0]
        n_subs = self.df_subs.shape[0]

        return f"""
        Subtitles have {n_subs} lines and {n_speakers} speakers:
        {self.df_subs['Speaker'].value_counts().to_dict()}        

        Start time (s): {self.df_ts['Start_seconds'].min()}
        End time (s): {self.df_ts['End_seconds'].max()}
        """


class ConsecutiveGrouper:
    def __init__(self) -> None:
        self.idx = pd.IndexSlice

    def select_speaker(self, df: pd.DataFrame, speaker: str) -> pd.DataFrame:
        """Select line for given `speaker`
        >>> TG.select_speaker(rdr.df_ts, 'Ichinose').head()

                             Start         End  ...  End_seconds  End_samples
        Speaker                                 ...
        Ichinose    3   0:00:08.85  0:00:10.00  ...        10.00       441000
                    9   0:00:22.45  0:00:29.11  ...        29.11      1283751
                    10  0:00:29.11  0:00:33.82  ...        33.82      1491462
                    11  0:00:33.82  0:00:36.39  ...        36.39      1604799
                    12  0:00:36.39  0:00:38.91  ...        38.91      1715930
        """
        return df.loc[self.idx[speaker, :], :]

    @staticmethod
    def get_consecutive_groups(df: pd.DataFrame, ass_inds: pd.Int64Index) -> tuple[list[list], list[int]]:
        """
        ```python
        >>> groups[:5]
        [[12, 13, 14, 15], [28, 29], [41, 42], [52, 53, 54], [56, 57, 58]]

        >>> ungrouped[:5]
        [6, 17, 23, 47, 66]
        ```
        """
        start_end_delta: np.ndarray = \
            df['End_seconds'].iloc[:-1].values -\
            df['Start_seconds'].iloc[1:].values

        isConsec = (start_end_delta == 0)

        groups: list[list[int]] = []

        ungrouped: list[int] = []
        current_group: list[int] = []

        for i, ind in enumerate(ass_inds[:-1]):
            if isConsec[i]:
                current_group.append(ind)
                continue
            if isConsec[i-1]:
                current_group.append(ind)
                groups.append(current_group)
                current_group = []
                continue

            ungrouped.append(ind)
            continue

        if isConsec[-1]:
            groups[-1].append(ass_inds[-1])
        else:
            ungrouped.append(ass_inds[-1])

        return groups, ungrouped

    def aggregate_consecutive(
        self,
        df_: pd.DataFrame,
        groups: list[list[int]],
        ungrouped: list[int]
    ) -> pd.DataFrame:
        """Aggregate consecutive subtitles, then concatenate with non-consecutive subtitles.
        ```python
        >>> df_merge.head()

        Start	End	Start_seconds	Start_samples	End_seconds	End_samples
        6	0:04:43.95	0:04:45.10	283.95	12522195	285.10	12572910
        12	0:04:57.55	0:05:14.01	297.55	13121955	314.01	13847841
        17	0:05:16.30	0:05:20.70	316.30	13948830	320.70	14142870
        23	0:05:34.18	0:05:38.47	334.18	14737338	338.47	14926527
        28	0:05:47.32	0:05:54.58	347.32	15316812	354.58	15636978        
        ```
        """

        df_dict: dict[int, dict[str, Any]] = {}
        end_cols = [c for c in df_.columns if 'End' in c]

        df_2 = df_.copy().droplevel(0)

        for g in groups:
            start = df_2.loc[g[0], :].to_dict()
            end = df_2.loc[g[-1], :]

            for col in end_cols:
                start[col] = end.at[col]

            df_dict[g[0]] = start

        df_merge = pd.DataFrame.from_dict(df_dict, orient='index').\
            append(df_2.loc[ungrouped, :]).\
            sort_index()

        return df_merge

    def aggregate(self, df: pd.DataFrame, speaker: str) -> pd.DataFrame:
        """
        """
        df_ = self.select_speaker(df, speaker)
        ass_inds = df_.index.get_level_values(level=1)
        groups, ungrouped = self.get_consecutive_groups(df_, ass_inds)

        df_agg = self.aggregate_consecutive(df_, groups, ungrouped)
        df_agg.index.name = speaker
        return df_agg


def main(
    filename_prefix: str,
    speakers: list[str],
    read_clip=True,
    ass_kwargs: dict = {}
) -> None:

    datadir = Path.cwd() / 'data'
    fname = str(datadir / filename_prefix)

    create_logger(filename_prefix, level=20)

    rdr = ASSReader(f"{fname}.ass")

    if read_clip:
        clip = AudioClip(f"{fname}.m4a")
        print(clip)
        rdr.parse_subs(clip.rate, **ass_kwargs)
    else:
        rdr.parse_subs(44100, **ass_kwargs)

    logging.info(rdr)

    df_lst: list[pd.DataFrame] = []

    for speaker in speakers:
        df_speaker = ConsecutiveGrouper().\
            aggregate(rdr.df_ts, speaker)

        logging.info(df_speaker.head())

        df_lst.append(df_speaker)

        outp = datadir / f'{speaker}_agg-subs.csv'
        check_overwrite(outp, df_speaker.to_csv)

    df_merge = pd.concat(df_lst, axis=0, keys=speakers).sort_index()
    logging.info(df_merge)

    outp = datadir / f"{filename_prefix}_merge-subs.csv"
    check_overwrite(outp, df_merge.to_csv)


if __name__ == '__main__':
    main(
        filename_prefix="ichinose_tamaki_taidan",
        speakers=['Ichinose', 'Tamaki'],
        read_clip=False,
        ass_kwargs=dict(skiprows=2)
    )
