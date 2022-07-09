import logging
import audiofile
from pathlib import Path

from numpy import ndarray
from pandas import DataFrame

class AudioClip:
    def __init__(
        self, filename: str, offset=0, duration=None
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

    def clip(self, start: int, end: int, unit='samples') -> ndarray:
        if unit in ['sample', 'samples']:
            return self.data[:, start:end]
        if unit == 'seconds':
            start *= self.rate
            end *= self.rate
            return self.data[:, start:end]
        else:
            raise ValueError(
                f"unit must be `samples` or `seconds`, not {unit}.")

    def __str__(self) -> str:
        return f"""
        Data has shape {self.data.shape} sampled at {self.rate} Hz
        """


def check_overwrite(filename: Path, func: callable, **kwargs) -> None:
    ow = None
    try:
        while ow not in ['y', 'n']:
            ow = input(f"{filename} exists. Overwrite? [y/n]").lower()[0]
    except KeyboardInterrupt:
        print(f"Save interrupted. File not saved: {filename}.")

    if ow == 'y':
        func(filename, **kwargs)
        print(f"File saved at:\n{filename}")

    return


LOGDIR = Path.cwd() / 'logs'
def create_logger(
    filename: str,
    logdir: Path = LOGDIR,
    filemode='a',
    force=True,
    level=logging.INFO,
    **kwargs
) -> None:
    """Create a basic log file for the session"""

    if not logdir.is_dir():
        logdir.mkdir()

    if not '.log' == filename[-4:]:
        filename += ".log"

    filename = logdir / filename
    if filename.is_file() and\
            filemode != 'a' and\
            (not force):

        msg = f"Log already exists at {filename} and will be overwritten. Should I create a new log file instead? [y/n]"

        how = input(msg).lower()[0]
        if how == 'y':
            ind = 1
            while filename.is_file():
                filename = logdir / f"{filename.stem}_{ind}.log"

    print(f"Logs will be available at:\n{filename}")

    logging.basicConfig(
        filename=logdir / filename,
        filemode=filemode,
        force=force,
        level=level,
        **kwargs
    )


def save_audio_clips(clip: AudioClip, df_ts: DataFrame, outdir: Path, prefix: str):
    """
    ```python
            Start	End	Start_seconds	Start_samples	End_seconds	End_samples
    645	0:36:28.85	0:36:33.25	2188.85	96528285	2193.25	96722325
    938	0:53:11.80	0:53:16.30	3191.80	140758380	3196.30	140956830
    763	0:43:09.60	0:43:11.35	2589.60	114201360	2591.35	114278535
    744	0:42:15.45	0:42:21.55	2535.45	111813344	2541.55	112082355
    713	0:40:19.60	0:40:27.35	2419.60	106704360	2427.35	107046135
    ```
    """
    for s in df_ts.index:
        start, end = df_ts.loc[s, ['Start_samples', 'End_samples']]
        audio_sample = clip.clip(start, end, unit='samples')

        outp = outdir / f"{prefix}_{s}.wav"
        if not outp.is_file():
            audiofile.write(outp, audio_sample, clip.rate)
            print(f"Saved file to {outp}")
