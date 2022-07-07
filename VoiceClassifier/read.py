import audiofile 
import regex as re 
import numpy as np 
import pandas as pd 
from typing import Union
from pathlib import Path 

DIAGPAT = re.compile(
    "^Dialogue: 0,([\d:\.]{10}),([\d:\.]{10}),([^,]+)(?=.*)"
)

class AudioClip:
    def __init__(
        self, filename: str, offset=None, duration=None
    ) -> None:
        self.load_data(filename, offset=offset, duration=duration)

    def load_data(self, filename: str, offset: int, duration: int):
        """Read .m4a file"""
        data, rate = audiofile.read(
            filename,
            offset=offset,
            duration=duration
        )

        self.data = data 
        self.rate = rate
        self.shape = data.shape 

    def __str__(self) -> str:
        return f"""
        Data has shape {self.data.shape} sampled at {self.rate} Hz
        """

class ASSReader:
    """
    Class to load an audio file and parse its associated subtitles
    """
    def __init__(
        self, filename: str, 
        dialog_pattern: Union[str, re.Pattern]=DIAGPAT,
    ) -> None:

        self.dialog_pattern = dialog_pattern
        self.read_ass(filename)
    
    def read_ass(self, filename: str) -> None:
        with open(filename, 'r', encoding='utf-8') as io:
            subs = io.readlines()

        self.subs = subs 
    
    def get_raw_dialog(self) -> pd.DataFrame:
        """Create dataframe from subtitles
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
            if "[Events]" in line: break 
            
        return pd.DataFrame(self.subs[ind+2:])

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
        df = df.iloc[:,0].apply(
            lambda s: pd.Series(
                self.parse_line(s, self.dialog_pattern)
            )
        ).rename(
            {i : col for i, col in enumerate(["Start", "End", "Speaker"])},
            axis=1
        )
        
        df = df.drop([0, 1], axis=0).\
            reset_index(drop=True).\
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
    def rename_ts_index_cols(df_ts: pd.DataFrame, dropcols: list[str]) -> pd.DataFrame:
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
        if dropcols:
            df_ts.drop(
                df_ts.loc[df_ts.Speaker.isin(dropcols), :].index,
                axis=0, inplace=True
            )

        df_ts.reset_index(drop=True, inplace=True)
        return df_ts.set_index(['Speaker', df_ts.index]).sort_index()

    def parse_subs(self, Hz: int) -> pd.DataFrame:
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
        
        df = self.get_raw_dialog()
        df = self.parse_dialog(df)

        df_ts = self.get_sub_times(
            df, Hz, self.ts2secs
        )
        
        self.df_subs = df 
        self.df_ts = self.rename_ts_index_cols(
            df_ts, ['Default', 'Translator']
        )

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
    def get_consecutive_mask(inds: pd.Int64Index) -> tuple[np.ndarray]:
        """Get masks whose elements are `True` when consecutive dialog indices differ by 1, i.e. are consecutive in the original file as well.
        ```python
        >>> mask, mask_1L = TG.get_true_masks(inds)
        
        >>> mask[:5]
        array([False, False,  True,  True,  True])

        >>> mask_1L[:5]
        array([False,  True,  True,  True,  True])
        ```
        """
        mask = ((inds[1:] - inds[:-1]) == 1)
        mask = np.insert(mask, 0, mask[0] == True)

        mask_1L = mask.copy()
        for i, m in enumerate(mask_1L[1:-1]):
            if m == False and mask_1L[i+2] == True:
                mask_1L[i+1] = True 
        
        return mask, mask_1L 

    @staticmethod
    def get_true_groups(mask: np.ndarray, mask_1L: np.ndarray) -> list[pd.Index]:
        trues = pd.Series(~mask).\
            cumsum().\
            mask(mask).\
            ffill().\
            mask(~mask_1L)
        
        for u in trues.unique():
            if np.isnan(u): continue 
            yield trues.loc[trues == u].index 

    def aggregate_consecutive(
        self,
        df_: pd.DataFrame, 
        inds: pd.Int64Index, 
        mask: np.ndarray, 
        mask_1L: np.ndarray, 
        speaker: str
    ) -> pd.DataFrame:
        """Aggregate timestamps for subtitles that are consecutive.
        """
        df_dict: dict[str, dict] = {col : {} for col in df_.columns}
        for grp in  self.get_true_groups(mask, mask_1L):
            if grp.shape[0] < 2: continue 
            key = inds[grp[0]]
            for col in df_dict.keys():
                if 'Start' in col:
                    df_dict[col][key] = df_.at[(speaker, key), col]
                else:
                    df_dict[col][key] = df_.at[(speaker, inds[grp[-1]]), col]
        
        return pd.DataFrame.from_dict(df_dict, orient='columns')

    def concat_non_consecutive(self, df_: pd.DataFrame, trues: pd.DataFrame, mask_1L: np.ndarray) -> pd.DataFrame:
        """Concatenate dataframes containing timestamps of non-consecutive subtitles and aggregated consecutive subtitles.
        """
        return pd.concat(
            [df_.loc[self.idx[:, ~mask_1L], :].droplevel(0), trues], 
            axis=0
        ).sort_index()

    def aggregate(self, df: pd.DataFrame, speaker: str) -> pd.DataFrame:
        """
        """
        df_ = self.select_speaker(df, speaker)
        inds = df_.index.get_level_values(level=1)
        mask, mask_1L = self.get_consecutive_mask(inds)
        grouped = self.aggregate_consecutive(df_, inds, mask, mask_1L, speaker)
        return self.concat_non_consecutive(df_, grouped, mask_1L)


def main(filename_prefix: str, read_clip=True):
    datadir = Path.cwd() / 'data'
    fname = str(datadir / filename_prefix)
    
    rdr = ASSReader(f"{fname}.ass")

    if read_clip:
        clip = AudioClip(
            f"{fname}.m4a", 
            offset=98,
            duration=3737
        )
        print(clip)
        rdr.parse_subs(clip.rate)
    else:
        rdr.parse_subs(44100)

    print(rdr)

    for speaker in ['Ichinose', 'Tamaki']:        
        outp = datadir / f'{speaker}_agg-subs.csv'

        if outp.is_file():
            overwrite = input(f"{outp} exists. Overwrite? [y/n]")
            if overwrite == 'n': continue 

        df_speaker = ConsecutiveGrouper().\
            aggregate(rdr.df_ts, speaker)

        df_speaker.to_csv(outp)
        print(f"File saved at: {outp}")

if __name__ == '__main__':
    main(
        filename_prefix="ichinose_tamaki_taidan",
        read_clip=False 
    )