"""
Remove members-only videos from a playlist and construct a new playlist from the remaining videos.
"""

import yt_dlp
import re
import pandas as pd
from typing import Any
from datetime import datetime
from tkinter.filedialog import asksaveasfilename


VIDEOID = re.compile("(?:watch\?v\=)(.*)")
PLAYLISTID = re.compile(r"(?:playlist\?list\=)(.*)")


def get_playlist_info(url: str, **kwargs) -> dict[str, Any]:
    ydl_opts = dict(
        quiet=True,
        ignoreerrors=True,
        **kwargs
    )

    with yt_dlp.YoutubeDL(params=ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return info


def extract(info: dict[str, Any]) -> tuple[datetime, str]:
    return [
        (datetime.strptime(entry['upload_date'], r"%Y%m%d"),
         VIDEOID.search(entry['original_url']).group(1))
        for entry in info['entries']
        if entry
    ]


def info2df(info: dict[str, Any]) -> pd.DataFrame:
    e = extract(info)
    df = pd.DataFrame.from_records(e)
    df.sort_values(0, inplace=True)
    df.columns = ['DATE', 'VIDEO_ID']
    return df


def create_playlist(info_df: pd.DataFrame) -> str:
    base_url = r"http://www.youtube.com/watch_videos?video_ids="
    return base_url + ','.join(info_df['VIDEO_ID'].tolist())


def main(url: str):
    info = get_playlist_info(url)
    df = info2df(info)
    return create_playlist(df)


if __name__ == '__main__':
    url = r"https://www.youtube.com/playlist?list=PLA7GVF9npsZw_qZlJho5u1yJB5C0Pnn2J"

    id = PLAYLISTID.search(url).group(1)
    outname = asksaveasfilename(
        title="Save the filtered playlist",
        defaultextension=".txt",
        initialdir='./'
    )
    with open(outname, 'w') as io:
        io.write(main(url))
