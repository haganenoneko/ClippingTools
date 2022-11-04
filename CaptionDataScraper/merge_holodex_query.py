import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

import seaborn as sns 

plt.style.use('default')
plt.rcParams['figure.dpi'] = 135
sns.set_style('whitegrid')

DATADIR = Path.cwd() / 'data/vspo_clippers/'

def rename_part_files(dir=DATADIR) -> None:
    for file in Path(dir).glob("*.part"):
        file.rename(file.parent / f"{file.name}.csv")

def merge_csv_files(dir=DATADIR) -> pd.DataFrame:
    read = lambda f: pd.read_csv(
        f, index_col=0, header=None, 
        encoding='utf-8', skiprows=1
    )

    files = list(dir.glob("*.csv"))
    files.sort()

    df = pd.concat([read(f) for f in files], axis=0).\
        reset_index(drop=True)
            
    # df.columns = ['Date', 'Duration', 'ChannelName']
    # df = df.set_index('Date').sort_index()
    
    return df 

def main():

    df_merge = merge_csv_files()
    df_merge.columns = ['Date', 'Duration', 'Channel']
    df_merge['Date'] = pd.to_datetime(df_merge['Date'])
    df_merge.sort_values('Date', inplace=True)
    df_merge.to_csv(DATADIR / "merged.csv")

if __name__ == '__main__':
    main()
