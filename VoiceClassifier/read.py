from typing import Union
import regex as re 
import pandas as pd 
import numpy as np 
import audiofile 
from pathlib import Path 

DIAGPAT = re.compile(
    "^Dialogue: 0,([\d:\.]{10}),([\d:\.]{10}),([^,]+)(?=.*)"
)

class Reader:
    def __init__(
        self, 
        datadir: Path=None, 
        suffix: str="ichinose_tamaki_taidan",
        offset=None, duration=None,
        dialog_pattern: Union[str, re.Pattern]=DIAGPAT,
    ) -> None:

        self.dialog_pattern = dialog_pattern
        self.get_data_files(
            datadir, suffix, 
            offset=offset, duration=duration
        )
    
    def get_data_files(
        self, 
        datadir: Path, suffix: str, 
        offset=None, duration=None
    ) -> tuple[list[float], int]:
        
        if datadir is None:
            datadir = Path.cwd() / 'data'
        if suffix is None:
            raise ValueError("No suffix given")
        
        # read .m4a 
        data, rate = audiofile.read(
            datadir / f"{suffix}.m4a",
            offset=98,
            duration=3737
        )

        # read subtitles 
        with open(
            datadir / f"{suffix}.ass", 'r', encoding='utf-8'
        ) as io:
            subs = io.readlines()

        self.data = data 
        self.rate = rate 
        self.subs = subs 
    
    @staticmethod
    def parse_line(line: str, pat: re.Pattern) -> list[str]:
        return pat.search(line).groups()

    @staticmethod
    def ts2secs(ts: str) -> float:
        HH, MM, SS_ms = [float(x) for x in ts.split(":")]
        return 3600*HH + 60*MM + SS_ms 

    @staticmethod
    def get_sub_times(
        df: pd.DataFrame, rate: float, func: staticmethod
    ) -> pd.DataFrame:
            
        for col in ['Start', 'End']:
            df[f"{col}_seconds"] = df[col].apply(
                lambda ts: func(ts)
            )
            
            df[f"{col}_samples"] = (
                df[f"{col}_seconds"] * rate
            ).astype(int)

        return df 

    @staticmethod
    def process_ts(df_ts: pd.DataFrame, dropcols: list[str]) -> pd.DataFrame:
        
        if dropcols:
            df_ts.drop(
                df_ts.loc[df_ts.Speaker.isin(dropcols), :].index,
                axis=0, inplace=True
            )

        df_ts.reset_index(drop=True, inplace=True)
        return df_ts.set_index(['Speaker', df_ts.index]).sort_index()

    def parse_subs(self) -> pd.DataFrame:

        subs = self.subs 
        for ind, line in enumerate(subs):
            if "[Events]" in line: break 
            
        df = pd.DataFrame(subs[ind+2:])
        df = df.iloc[:,0].apply(
            lambda s: pd.Series(self.parse_line(s, self.dialog_pattern))
        )

        df.columns = ["Start", "End", "Speaker"]

        df = df.drop([0, 1], axis=0).\
            reset_index(drop=True).\
            sort_values(by="Start", ascending=True)

        df_ts = self.get_sub_times(
            df, self.rate, self.ts2secs
        )
        
        self.df_subs = df 
        self.df_ts = self.process_ts(
            df_ts, ['Default', 'Translator']
        )

        return df 
        
    def __str__(self) -> str:
        return f"""
        Data has shape {self.data.shape} sampled at {self.rate} Hz
        {self.df_ts.head()}
        """
    
class TruesGrouper:
    def __init__(self) -> None: 
        self.idx = pd.IndexSlice

    def select_speaker(self, df: pd.DataFrame, speaker: str) -> pd.DataFrame:
        return df.loc[self.idx[speaker, :], :]

    @staticmethod
    def get_true_masks(inds: pd.Int64Index) -> tuple[np.ndarray]:
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

    def group_trues(
        self,
        df_: pd.DataFrame, 
        inds: pd.Int64Index, 
        mask: np.ndarray, 
        mask_1L: np.ndarray, 
        speaker: str
    ) -> pd.DataFrame:

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

    def concat_trues_falses(self, df_: pd.DataFrame, trues: pd.DataFrame, mask_1L: np.ndarray) -> pd.DataFrame:
        return pd.concat(
            [df_.loc[self.idx[:, ~mask_1L], :].droplevel(0), trues], 
            axis=0
        ).sort_index()

    def group_consecutive_trues(self, df: pd.DataFrame, speaker: str) -> pd.DataFrame:
        df_ = self.select_speaker(df, speaker)
        inds = df_.index.get_level_values(level=1)
        mask, mask_1L = self.get_true_masks(inds)
        grouped = self.group_trues(df_, inds, mask, mask_1L, speaker)
        return self.concat_trues_falses(df_, grouped, mask_1L)

rdr = Reader()
rdr.parse_subs()
TruesGrouper().group_consecutive_trues(rdr.df_ts, "Ichinose")
print(rdr)

