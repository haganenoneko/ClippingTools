import regex as re 
from sys import getsizeof
import numpy as np 
import pandas as pd 
import audiofile 
from pathlib import Path 

DATADIR = Path.cwd() / 'data'
suffix = "ichinose_tamaki_taidan"

data, sampling_rate = audiofile.read(
    DATADIR / f"{suffix}.m4a",
    offset=98,
    duration=3737
)

with open(DATADIR / f"{suffix}.ass", 'r', encoding='utf-8') as io:
    subs = io.readlines()

DIAGPAT = re.compile("^Dialogue: 0,([\d:\.]{10}),([\d:\.]{10}),([^,]+)(?=.*)")
def parse_line(line: str, pat=DIAGPAT) -> list[str]:
    return pat.search(line).groups()

def parse_subs(subs: list[str]) -> pd.DataFrame:
    for ind, line in enumerate(subs):
        if "[Events]" in line: break 
        
    df = pd.DataFrame(subs[ind+2:])
    df = df.iloc[:,0].apply(lambda s: pd.Series(parse_line(s)))
    df.columns = ["Start", "End", "Speaker"]
    df = df.drop([0, 1], axis=0).\
        reset_index(drop=True).\
        sort_values(by="Start", ascending=True)

    return df 

def ts2secs(ts: str) -> float:
    HH, MM, SS_ms = [float(x) for x in ts.split(":")]
    return 3600*HH + 60*MM + SS_ms 

def get_sub_times(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    for col in ['Start', 'End']:
        df[f"{col}_seconds"] = df[col].apply(lambda ts: ts2secs(ts))
        df[f"{col}_samples"] = (df[f"{col}_seconds"] * rate).astype(int)
    return df 

df = parse_subs(subs)
df_ts = get_sub_times(df, sampling_rate)
df_ts.max()

df_ts.drop(
    df_ts.loc[df_ts.Speaker.isin(['Default', 'Translator']), :].index,
    axis=0, inplace=True
)

df_ts.reset_index(drop=True, inplace=True)
df_ts.set_index(['Speaker', df_ts.index]).sort_index()

import matplotlib.pyplot as plt 
from scipy.signal import spectrogram

row = df_ts.iloc[0,:]
a, b = row[['Start_samples', 'End_samples']]
f, t, Sxx = spectrogram(data[0, a:b], fs=sampling_rate)

plt.pcolormesh(
    t, f, Sxx, shading='gouraud', cmap='copper', vmax=np.quantile(Sxx, 0.95)
); plt.show()
