from common import get_filenames, run_powershell
from pathlib import Path 
from numpy import vectorize 

VIDEO_PATH = "C:/Users/delbe/Videos/subtitles/full_raw/"
SECS_PATH = VIDEO_PATH + "removesilence_timecodes/"

def create_splice_pair(i: int, j: int, times: tuple[float, float]) -> str:
    """Command for splicing one interval of a video (from `times[0]` to `times[1]`)

    Args:
        i (int): index of the video input (e.g. `i=0` if there is only one input)
        j (int): index of the spliced clip. Used to name intermediate streams. 
            Does not need to be 0-indexed, but if it is, then, e.g. the second clip would have `j=1`. 
        times (tuple[float, float]): `(start time, end time)` in seconds.

    Returns:
        str: splice command for both video and audio streams.
    """    
    tA, tB = times 
    return f"""
    [{i}:v]trim=start={tA}:end={tB},setpts=PTS-STARTPTS[{j}v];
    [{i}:a]atrim=start={tA}:end={tB},asetpts=PTS-STARTPTS[{j}a];
    """

def split_splice_cmd(cmd: str, audio_only: bool) -> str:
    v, a = map(lambda x: x.strip(), cmd.split(";")[:2])
    if audio_only:
        return a 
    else:
        return v 

def build_concat_filter(intervals: list[tuple[float]], has_audio=True) -> str:
    pairs = ''.join(vectorize(create_splice_pair)(intervals))
    
    num = len(intervals)
    streams = ''.join((f"[{i}v][{i}a]" for i in range(num)))

    filter = f"\"{pairs}{streams}\"concat=n={num}:v=1:a=1[outv][outa]"
    return filter 

def splice_and_concat(
    fpath: Path, intervals: list[tuple[float]], has_audio=True) -> None:
    
    filter = build_concat_filter(intervals, has_audio=has_audio)
    map_ = "-map [outv] -map [outa]"
    outname = fpath.parent / fpath.stem + f"_concat{fpath.suffix}"
    cmd = f"ffmpeg -i {fpath} -filter_complex {filter} {map_} {outname}"

    run_powershell(cmd)

def main():
    fpath = get_filenames(
        init_dir=SECS_PATH,
        title="Select CSV file containing dialog intervals",
        filetypes=(
            ("CSV files", "*.csv"),
            ("TSV files", "*.tsv"),
        )
    )