from pathlib import Path 
# from RemoveSilence.common import get_filename
from common import get_filename
from pathlib import Path 
from subprocess import Popen, PIPE 

VIDEODIR = Path(r"C:/Users/delbe/Videos/subtitles/full_raw/")
CMD = "ffmpeg --hide_banner -i {input} -an -filter_complex \"{filter}\" {map} \"{output}\""

def tup2secs(tup: tuple[int]) -> float:
    h, m, s, ms = tup 
    return 3600*h + 60*m + s + ms/1e3 

def get_preview_secs(tup: tuple[int]) -> tuple[float]:
    return tuple(map(tup2secs, preview))

def extract_ps_output(cmd: str) -> str:
    return Popen(['powershell.exe', cmd], stdout=PIPE).\
        communicate()[0].\
        decode('utf-8').\
        strip().\
        split('\n')[-1]

def get_video_format(fp: Path) -> str:
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 \"{fp}\""
    fmt = extract_ps_output(cmd)
    return fmt 

def get_video_duration(fp: Path) -> float:
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {fp}"
    dur = extract_ps_output(cmd)
    return float(dur)

in_file = get_filename(
    init_dir=VIDEODIR, 
    title="Select input file",
    filetypes=(("MP4 Files", "*.mp4"), )
)

in_dur = get_video_duration(in_file)

preview = ((0, 5, 12, 83), (0, 5, 14, 93))

s = get_preview_secs(preview)


def build_filter(
    preview_durs: tuple[tuple[int]], 
    xdur: float = 1.0,
    fmt: str) -> str:

    tA, tB = get_preview_secs(preview_durs)
    
    if round(tB - tA, ndigits=1) < xdur:
        tB += xdur/2 
    
    while (tB - tA) < xdur:
        tB += xdur/4

    tA, tB = map(
        lambda x: round(x, ndigits=3), 
        (tA, tB)
    )

    f"""
    [0:v]trim=start={tA}:end={tB-xdur},setpts=PTS-STARTPTS[1v];
    [0:v]trim=start={xdur},setpts=PTS-STARTPTS[2v];
    [0:v]trim=start={tB-xdur}:end={tB},setpts=PTS-STARTPTS[fadeoutsrc];
    [0:v]trim=start=0:end={xdur},setpts=PTS-STARTPTS[fadeinsrc];
    [fadeinsrc]format={fmt},
            fade=t=in:st=0:d={xdur}:alpha=1[fadein];
    [fadeoutsrc]format={fmt},
            fade=t=out:st=0:d={xdur}:alpha=1[fadeout];
    [fadein]fifo[fadeinfifo];
    [fadeout]fifo[fadeoutfifo];
    [fadeoutfifo][fadeinfifo]overlay[crossfade];
    [1v][crossfade][2v]concat=n=3[output];
    [0:a] acrossfade=d=1 [audio]
    """

    # TODO: ADD `ATRIM` for audio trimmming; then, similar to `concat` (e.g. `remove_silence.jl`), we also need to map the audio streams

    tmp.format(start=tA, end=tB-xdur, i=1)
    tmp.format(start=xdur)

[0:v]trim=start=0:end=9,setpts=PTS-STARTPTS[firstclip];

[0:v]trim=start=0:end=9,setpts=PTS-STARTPTS[firstclip];
[1:v]trim=start=1,setpts=PTS-STARTPTS[secondclip];
[0:v]trim=start=9:end=10,setpts=PTS-STARTPTS[fadeoutsrc];
[1:v]trim=start=0:end=1,setpts=PTS-STARTPTS[fadeinsrc];
