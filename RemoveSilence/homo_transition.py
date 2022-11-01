from pathlib import Path
# from RemoveSilence.common import get_filename
from common import get_filename
from pathlib import Path
from subprocess import Popen, PIPE

VIDEODIR = Path(r"C:/Users/delbe/Videos/subtitles/full_raw/")


def run_powershell(cmd: str, stdout=PIPE, stderr=PIPE) -> Popen:
	return Popen(['powershell.exe', cmd], stdout=PIPE, stderr=PIPE)


def tup2secs(tup: tuple[int]) -> float:
	"""Convert a tuple of (hours, minutes, seconds, milliseconds) to total seconds"""
	h, m, s, ms = tup
	return 3600*h + 60*m + s + ms/1e2


def get_preview_secs(tup: tuple[int]) -> tuple[float]:
	return tuple(map(tup2secs, tup))


def extract_ps_output(cmd: str) -> str:
	"""Extract the last line in `stdout` for a powershell command"""
	return run_powershell(cmd).\
		communicate()[0].\
		decode('utf-8').\
		strip().\
		split('\n')[-1]


def get_video_format(fp: Path) -> str:
	"""Get format of a video using powershell"""
	cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 \"{fp}\""
	fmt = extract_ps_output(cmd)
	return fmt


def get_video_duration(fp: Path) -> float:
	"""Get duration of a video in seconds using powershell"""
	cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {fp}"
	dur = extract_ps_output(cmd)
	return float(dur)

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
		fadetype: str="fade",
		afade_curvetypes: tuple[str, str] = ('exp', 'exp'),) -> str:

	# curve types for audio cross fade
	# examples: http://underpop.online.fr/f/ffmpeg/help/afade.htm.gz
	# common types: tri (linear, default), exp, [q/h/e]sin, cub, squ, cbr, log
	ac1, ac2 = afade_curvetypes

	# create video and audio segments that will be crossfaded
	src = f"""
	[0:v]trim=start={tA}:end={tB},setpts=PTS-STARTPTS[1v];
	[0:v]trim=start=0,setpts=PTS-STARTPTS[2v];
	[0:a]atrim=start={tA}:end={tB},asetpts=PTS-STARTPTS[1a];
	[0:a]atrim=start=0,asetpts=PTS-STARTPTS[2a];
	"""
	# calculating offset: https://stackoverflow.com/a/63570355
	offset = tB-tA-xdur
	xfade = f"""
	[1v][2v]xfade=transition={fadetype}:duration={xdur}:offset={offset};
	[1a][2a]acrossfade=d={xdur}:c1={ac1}:c2={ac2}
	"""
	
	return f"\"{src}\n{xfade}\""


def cross_fade(
		fpath: Path,
		preview: tuple[tuple[int]],
		crossfade_dur: float = 1.0,
		**xfade_kwargs) -> None:

	fmt = get_video_format(fpath)
	tot = get_video_duration(fpath)

	filter = build_filter(
		preview, fmt,
		xdur=crossfade_dur,
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

