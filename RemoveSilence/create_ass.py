from pathlib import Path
import pandas as pd 
from datetime import timedelta
from typing import Union
import numpy as np 

from common import get_filename

BASEDIR = Path("./RemoveSilence/")
ASS_HEADER_PATH = BASEDIR / "ASS_header.txt"
DIALOG_BASE = "Dialogue: {layer},{start_time},{end_time},{style},{name},{marginL},{marginR},{marginV},{effect},{text}"


def add_dialog(
        text: str, start_time: float, end_time: float,
        layer=0, style='Default', name='',
        marginL=0, marginR=0, marginV=0, effect='') -> str:
    """Create single line of ASS dialog"""
    return DIALOG_BASE.format(
        layer=layer, start_time=start_time, end_time=end_time,
        style=style, name=name, marginL=marginL, marginR=marginR,
        marginV=marginV, effect=effect, text=text
    )


def time2stamp(seconds: float) -> str:
    """Convert seconds into timestamp"""
    stamp = timedelta(seconds=seconds)
    # milliseconds = int(round(stamp.microseconds/1e6, ndigits=2)*100)
    # return str(stamp)[:8] + str(milliseconds)
    return str(stamp)


vtime2stamp = np.vectorize(time2stamp)

def create_ass_dialog(times: list[float], text: list[str] = None, **diag_kw) -> str:
    lines: list[str] = []
    stamps = vtime2stamp(times)

    for i in range(0, len(times), 2):
        start, end = stamps[i:i+2]
        lines.append(
            add_dialog(
                text[int(i/2)] if text else 'n/a',
                start, end, **diag_kw
            )
        )

    return "\n".join(lines)

def add_audio_video_paths(
    header: str, 
    video_file: Union[str, Path], 
    audio_file: Union[str, Path]) -> str:

    if video_file:
        if not isinstance(video_file, Path):
            video_file = Path(video_file)
        
    if not isinstance(audio_file, Path):
        audio_file = Path(audio_file)

    lines = header.split('\n')
    if video_file:
        lines[13] += str(video_file.absolute())
    if audio_file:
        lines[14] += str(audio_file.absolute())
    
    return '\n'.join(lines)

def get_header(
    header: Union[str, Path], 
    video_file: Union[str, Path], 
    audio_file: Union[str, Path]) -> str:
    """
    Get header for the ASS file
    TO DO: add path to video/audio files
    """
        
    if isinstance(header, Path):
        with open(header, 'r', encoding='utf-8') as io:
            header = io.read()

    if (video_file is None) and\
        (audio_file is None):
        return header 

    if video_file:
        if audio_file:
            header = add_audio_video_paths(header, video_file, audio_file)
        else:
            header = add_audio_video_paths(header, video_file, video_file)
    else:
        header = add_audio_video_paths(header, None, audio_file)
    
    return header


def create_ass_file(
        outname: str, 
        times: list[float],
        text: list[str] = None, 
        outdir: Path = BASEDIR, 
        header: Union[str, Path] = ASS_HEADER_PATH,
        video_file: Union[str, Path]=None,
        audio_file: Union[str, Path]=None, 
        **diag_kw) -> str:
    """Create a new ASS file based on list of times and (optional) dialog text."""
    
    ass_header = get_header(header, video_file, audio_file)
    dialog = create_ass_dialog(times, text, **diag_kw)
    full_content = ass_header + '\n' + dialog

    if not isinstance(outdir, Path):
        raise TypeError(f"{outdir} must be a Path")
    if not outdir.is_dir():
        raise FileNotFoundError(f"{outdir} is not a valid directory.")

    outpath = outdir / f"{outname}.ass"

    if outpath.is_file():
        overwrite = input(f"{outpath} already exists. Overwrite? [y/n]").lower()
        if overwrite != 'y':
            raise FileExistsError(outpath)

    with open(outpath, 'w', encoding='utf-8') as io:
        io.write(full_content)

    print(f"New ass file created at {outpath}")
    return full_content


def test():
    times = [2.08537, 3.52213, 4.41288, 6.19252, 13.4706, 15.2075, 15.6473, 18.2017, 20.9115, 28.1991, 28.2443, 29.6525, 30.6995, 33.0659, 34.9095, 36.6367, 42.2235, 43.4144, 44.8013, 48.952, 49.976, 51.1188, 53.0409, 55.1211, 56.3427, 60.219, 66.3007, 68.408, 70.0401, 71.3288, 81.1773, 85.8789, 89.3248,
             96.5284, 96.7173, 97.9433, 133.987, 134.987, 141.353, 143.93, 153.379, 156.288, 157.083, 158.595, 160.322, 165.112, 173.652, 176.251, 176.74, 178.074, 189.812, 194.121, 200.143, 201.729, 210.695, 216.173, 220.295, 222.54, 224.514, 226.925, 236.61, 237.676, 244.907, 246.721, 247.67, 249.449]

    return create_ass_file("ASS_test", times)

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


def main():
    fname = get_filename(
        title="Select file of silence intervals.",
        filetypes=(
            ('CSV files', '*.csv'),
            # ('Text files', '*.txt'),
            # ('All files', '*.*')
        )
    )

    fp = Path(fname)
    secs = parse_interval_file(fp)
    create_ass_file(fp.stem, secs, outdir=fp.parent)

if __name__ == '__main__':
    # test()
    main()