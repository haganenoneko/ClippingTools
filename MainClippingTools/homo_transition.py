from pathlib import Path
from common import get_filename, run_powershell, get_video_duration, get_video_format

# path to video files
VIDEODIR = Path('..\\..\\..\\..\\Videos\\subtitles\\full_raw')


def tup2secs(tup: tuple[int]) -> float:
    """Convert a tuple of (hours, minutes, seconds, milliseconds) to total seconds"""
    h, m, s, ms = tup
    return 3600*h + 60*m + s + ms/1e2


def get_preview_secs(tup: tuple[int]) -> tuple[float]:
    return tuple(map(tup2secs, tup))


def format_clip_times(
        times: tuple[float, float],
        xdur: float,
        padfrac: float,
        padsecs: float,
        totaldur: float) -> tuple[float, float]:

    tA, tB = times
    pad = max(padfrac*xdur, padsecs)

    if round(tB - tA, ndigits=1) < xdur:
        tB += xdur/2

    while (tB - tA) < xdur:
        tB += xdur/4

    tA, tB = map(
        lambda x: round(x, ndigits=3),
        (tA, tB)
    )

    if tA > pad:
        tA -= pad
    if tB + pad <= totaldur:
        tB += pad

    return tA, tB


def build_filter(
        preview: list[tuple[tuple[int]]],
        xdur: float = 1.0,
        xpad: float = 0.1,
        pfrac: float = 0.1,
        fadetype: str = "fade",
        afade_curvetypes: tuple[str, str] = ('exp', 'exp'),) -> str:

    # curve types for audio cross fade
    # examples: http://underpop.online.fr/f/ffmpeg/help/afade.htm.gz
    # common types: tri (linear, default), exp, [q/h/e]sin, cub, squ, cbr, log
    ac1, ac2 = afade_curvetypes

    tA, tB = preview
    if round(tB - tA, ndigits=1) < xdur:
        tB += xdur/2

    while (tB - tA) < xdur:
        tB += xdur/4

    tA, tB = map(
        lambda x: round(x, ndigits=3),
        (tA, tB)
    )

    # create video and audio segments that will be crossfaded
    src = f"""
	[0:v]trim=start={tA}:end={tB},setpts=PTS-STARTPTS[1v];
	[0:v]trim=start=0,setpts=PTS-STARTPTS[2v];
	[0:a]atrim=start={tA}:end={tB},asetpts=PTS-STARTPTS[1a];
	[0:a]atrim=start=0,asetpts=PTS-STARTPTS[2a];
	"""
    # calculating offset: https://stackoverflow.com/a/63570355
    offset = tB-tA-xdur
    # fadetypes: https://trac.ffmpeg.org/wiki/Xfade
    # common examples: fade[black/white], smooth[up/left/right/down]
    xfade = f"""
	[1v][2v]xfade=transition={fadetype}:duration={xdur}:offset={offset};
	[1a][2a]acrossfade=d={xdur}:c1={ac1}:c2={ac2}
	"""

    return f"\"{src}\n{xfade}\""


def cross_fade(
        fpath: Path,
        preview: tuple[tuple[int]],
        xdur: float = 1.0,
        **xfade_kwargs) -> None:

    fmt = get_video_format(fpath)
    tot = get_video_duration(fpath)

    filter = build_filter(
        preview, fmt,
        xdur=xdur,
        **xfade_kwargs
    )

    outpath = fpath.parent / f"{fpath.stem}_xfade{fpath.suffix}"
    cmd = f"ffmpeg -i \"{fpath}\" -filter_complex {filter} {outpath}"
    run_powershell(cmd)


def test():
    fpath = Path(
        r'C:/Users/delbe/Videos/subtitles/full_raw/uruha_wanchan_concat.mp4')
    preview = ((0, 5, 12, 83), (0, 5, 14, 93))
    cmd = cross_fade(fpath, preview)
    return cmd


def main(preview: tuple[tuple[int]], **xfade_kwargs):
    in_file = get_filename(
        init_dir=VIDEODIR, title="Select video file",
        filetypes=(("MP4 files", "*.mp4"))
    )

    cross_fade(in_file, preview, **xfade_kwargs)


if __name__ == '__main__':
    main(
        preview=((0, 0, 0, 0), (0, 0, 10, 0)),
        xdur=0.5,
        fadetype='fade',
    )
