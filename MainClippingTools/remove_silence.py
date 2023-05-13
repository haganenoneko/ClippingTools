import audiofile
import numpy as np
from pathlib import Path
from typing import Union 
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
from scipy.signal import savgol_filter
from common import get_video_duration


def db_to_AR(dB: float) -> float:
	"""Convert decibels to amplitude ratio"""
	return np.sqrt(10 ** (dB / 10))


def AR_to_dB(AR: float) -> float:
	"""Convert amplitude ratio to decibels"""
	return np.log10(AR ** 2) * 10


def stereo_to_mono(signal: np.ndarray) -> np.ndarray:
	if len(signal.shape) < 2:
		return signal

	if signal.shape[0] == 1:
		return signal[0, :]
	elif signal.shape[0] == 2:
		return np.mean(signal / 2, axis=0)
	else:
		raise ValueError(
			f"Signal has shape: {signal.shape}. Expected (0-2, N)")


def open_audio(fp: Path, mono=True, **kwargs) -> tuple[np.ndarray, int]:
	signal, sr = audiofile.read(fp, **kwargs)
	if mono:
		return stereo_to_mono(signal), sr
	else:
		return signal, sr


def filter(signal: np.ndarray, window_length=121, polyorder=2) -> np.ndarray:
	return savgol_filter(np.abs(signal), window_length, polyorder=polyorder)


def rle(inarray: np.ndarray) -> tuple[np.ndarray]:
	""" run length encoding. Partial credit to R rle function. 
Multi datatype arrays catered for including non Numpy
returns: tuple (runlengths, startpositions, values) """
	ia = np.asarray(inarray)                # force numpy
	n = len(ia)
	if n == 0:
		return (None, None, None)
	else:
		y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
		i = np.append(np.where(y), n - 1)   # must include last element posi
		z = np.diff(np.append(-1, i))       # run lengths
		p = np.cumsum(np.append(0, z))[:-1]  # positions
		return(z, p, ia[i])


def partition(arr: np.ndarray, size=2, step=1, min_delta=0):
	"""Partition 1D array `arr` into tuples of size `size`, with difference of size `step` between the last element of the `i`th and first element of the `i+1`th tuples"""
	n = arr.shape[0]
	if step > 0:
		num_parts = ((n - size) // step) + 1
	else:
		num_parts = n // size

	ret: list[np.ndarray] = []
	for i in range(num_parts):
		start = i*step if (step > 0) else i*size
		stop = start + size

		if stop > arr.shape[0]:
			break
		if min_delta > 0 and (arr[stop-1] - arr[start]) < min_delta:
			continue

		ret.append(arr[start:stop])

	if ret:
		return np.vstack(ret)
	else:
		raise ValueError(f"No partitions found.")


def pad_intervals(intervals: np.ndarray, pad: int, total: int) -> np.ndarray:

	padded = intervals.copy()
	padded[1:, 0] -= pad
	padded[:-1, 1] += pad

	if padded[0, 0] > pad:
		padded[0, 0] -= pad

	if padded[-1, 1] + pad < total:
		padded[-1, 1] += pad

	return padded


def clean_intervals(
		intervals: np.ndarray,
		min_dur: float = 0.5,
		min_gap: float = 0.2,
) -> list[tuple[float, float]]:
	"""Clean up intervals by

1. ensuring intervals are greater than a minimum duration
2. ensuring intervals are separated by a minimum duration
3. concatenating overlapping intervals (https://stackoverflow.com/a/58976449/10120386)

	Args:
intervals (np.ndarray): start and end times in first and second columns
min_dur (float, optional): minimum interval duration. Defaults to 0.1.
min_gap (float, optional): minimum separation between adjacent intervals. Defaults to 0.2.

	Returns:
list[tuple[float, float]]: list of processed `(start, end)` times
	"""
	# remove intervals with less than the minimum duration
	durs = np.diff(intervals).flatten()
	arr = intervals[durs > min_dur]

	# remove overlapping intervals and ensure gaps > min_gap
	valid = np.zeros(arr.shape[0]+1, dtype=bool)
	valid[[0, -1]] = 1
	valid[1:-1] = arr[1:, 0] - np.maximum.accumulate(arr[:-1, 1]) >= min_gap

	merged = np.vstack(
		(
			arr[valid[:-1], 0],
			arr[valid[1:], 1]
		)
	).T

	print(
		"Results of `clean_intervals`\n"
		f"\tInitial no.: {len(intervals)}\n"
		f"\tCleaned no. (%): {merged.shape[0]} "
		f"({100*merged.shape[0] / len(intervals):.1f}%)\n"
	)

	return merged


def invert_intervals(intervals: np.ndarray, total: float) -> np.ndarray:
	"""Invert intervals of non-silence, returning intervals of silence

	Args:
intervals (np.ndarray): intervals of non-silence
total (float): total duration in units of `intervals`

	Raises:
AssertionError: odd number of intervals

	Returns:
np.ndarray: intervals of silence
	"""
	flat = intervals.flatten()

	if flat[0] == 0:
		flat = flat[1:]
	else:
		flat = np.insert(flat, 0, 0)

	if flat[-1] == total:
		flat = flat[:-1]
	else:
		flat = np.append(flat, total)

	try:
		assert flat.shape[0] % 2 == 0
	except AssertionError:
		raise AssertionError(f"odd number of intervals: {flat.shape[0]}")

	return partition(flat, size=2, step=0)

# ---------------------------------------------------------------------------- #
#                               Silence detection                              #
# ---------------------------------------------------------------------------- #


def plot_intervals(
		signal: np.ndarray,
		sampling_rate: float,
		intervals: np.ndarray,
		downsample=10
) -> None:
	plt.rcParams['font.size'] = 11
	fig, ax = plt.subplots(figsize=(9, 5), dpi=120, constrained_layout=True)

	y = signal[::downsample]
	t = np.arange(start=0, step=downsample,
				  stop=signal.shape[0]) / sampling_rate

	ax.set_ylabel("Amplitude")
	ax.set_xlabel("Time (s)")
	ax.plot(t, y, lw=1, alpha=0.5, color='b', label='Signal')

	ylim = ax.get_ylim()
	kw = dict(alpha=0.35, color='r', linewidth=0, label='Silence')
	for i in range(intervals.shape[0]):
		x1, x2 = intervals[i, :]
		ax.fill_betweenx(ylim, x1, x2, **kw)

	handles, labels = ax.get_legend_handles_labels()
	ax.legend(
		handles=handles[:2],
		labels=labels[:2],
		loc='lower right',
		bbox_to_anchor=[0.98, 0.02],
		fancybox=False,
	)

	ax.locator_params(axis="both", nbins=5)
	plt.show()


def get_intervals(
		signal: np.ndarray,
		sampling_rate: float,
		window_length=21,
		polyorder=3,
		max_sil_vol=-33,
		min_seg_dur=0.1,
		min_sil_dur=0.5,
		min_gap_dur=0.1,
		adj_seg_pad=0.1,
		plot=True
) -> np.ndarray:

	min_seg_dur *= sampling_rate
	min_sil_dur *= sampling_rate

	filt = filter(signal, window_length=window_length, polyorder=polyorder)
	lens, inds, vals = rle(filt <= db_to_AR(max_sil_vol))

	sil_dur_mask = np.logical_and(vals == 1, lens >= min_sil_dur)
	seg_dur_mask = np.logical_and(vals == 0, lens <= min_seg_dur)
	mask = np.logical_or(sil_dur_mask, seg_dur_mask)
	ret: list[tuple] = [(inds[i], inds[i] + lens[i])
						for i in np.where(mask)[0]]

	out = np.array(ret)
	out[1:, 0] += 1

	# add padding between adjacent intervals
	if adj_seg_pad > 0:
		out = pad_intervals(out, round(
			adj_seg_pad*sampling_rate), signal.shape[0])

	# clean up intervals
	out = clean_intervals(
		out / sampling_rate,
		min_seg_dur / sampling_rate,
		min_gap_dur
	)

	tot_sil_dur = (out[:, 1] - out[:, 0]).sum()
	tot_sil_dur = f"{tot_sil_dur:.2f} s ({tot_sil_dur*100/(signal.shape[0] / sampling_rate):.2f}%)"
	avg_sil_amp = np.mean([np.mean(np.abs(signal[r[0]:r[1]]))
						   for r in ret[:100]])

	avg_sil_amp = f"{AR_to_dB(avg_sil_amp):.1f} dB ({avg_sil_amp*100/np.mean(np.abs(signal)):.2f}%)"

	print(
		f"""Result of `get_intervals`
		Total detected silence duration: {tot_sil_dur:>8}
		Average silence amplitude (n=100): {avg_sil_amp:>8}
		"""
	)

	if plot:
		plot_intervals(signal, sampling_rate, out)

	# invert silent intervals to get intervals of non-silence for ffmpeg to concatenate
	return invert_intervals(out, signal.shape[0]/sampling_rate)

# ---------------------------------------------------------------------------- #
#                                   Splicing                                   #
# ---------------------------------------------------------------------------- #


VIDEO_SPLICE = "[0:v]trim=start={t0}:end={t1},setpts=PTS-STARTPTS[{ind}v];"
AUDIO_SPLICE = "[0:a]atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS[{ind}a];"

def create_splice_pair_str(row: np.ndarray, ind: int) -> str:
	"""Return the command for including one interval (from `row[1]` to `row[2]` of the video) in the `concat` filter of `ffmpeg`"""
	t0, t1 = row
	return f"[0:v]trim=start={t0}:end={t1},setpts=PTS-STARTPTS[{ind}v];" +\
		f"[0:a]atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS[{ind}a];"


def splice_video(filepath: Path, intervals: np.ndarray) -> None:
	filepath = filepath.parent / filepath.stem
	output_name = f"{filepath}_silenceremove.mp4"

	num = intervals.shape[0]
	inds = np.arange(0, num, dtype=int)

	# join all intervals into a single filter
	# pairs = [create_splice_pair_str(intervals[i], i)
	# 		 for i in range(intervals.shape[0])]
	
	pairs = np.apply_along_axis(
		create_splice_pair_str,
		axis=1,
		arr=np.c_[intervals, inds].astype('<U300'),
	)

	concat = ''.join(
		[f"[{i}v][{i}a]" for i in inds]
	) + f"concat=n={num}:v=1:a=1[outv][outa]"

	# write filter to file
	filterpath = f"{filepath}__filter.txt"
	with open(filterpath, "w", newline="", encoding="utf-8", errors="ignore") as io:
		io.writelines(pairs)
		io.write(concat)
		io.flush()

	# assemble final command
	cmd = f"ffmpeg -hide_banner -i \"{filepath}.mp4\" -filter_complex_script {filterpath}" +\
		" -map [outv] -map [outa] " +\
		output_name

	Popen(['powershell', cmd])
	return

# ---------------------------------------------------------------------------- #
#                                   Crossfade                                  #
# ---------------------------------------------------------------------------- #

def get_fadetypes(fadetypes: Union[str, list[str]], num: int) -> list[str]:
    
    if isinstance(fadetypes, str):
        return [fadetypes] * num 
    
    if len(fadetypes) < num:
        fadetypes.extend(
            [fadetypes[-1]] * (num - len(fadetypes))
        )

    return fadetypes

def random_vfade(n: int, ignore=None) -> list[str]:
    """Choose `n` random video transitions.

    Args:
        n (int): number of transitions to choose
        ignore (list[str], optional): names of transitions to not use. Defaults to None.

    Returns:
        list[str]: names of transitions
    """
    FADETYPES = [
        'fade', 'fadeblack', 'fadewhite', 'distance', 'wipeleft', 'wiperight', 'wipeup', 'wipedown', 'slideleft', 'slideright', 'slideup', 'slidedown', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown', 'circleclose', 'circleopen', 'horzclose', 'horzopen', 'vertclose', 'vertopen', 'hlslice', 'hrslice', 'vuslice', 'vdslice', 'pixelize', 'hblur', 'zoomin'
    ]

    if ignore: 
        FADETYPES = [f for f in FADETYPES if f not in ignore]
        
    vfadetypes = [choice(FADETYPES) for _ in range(n)]
    print("Using the following vfadetypes:", vfadetypes, sep='\n')
    
    return vfadetypes

def build_crossfade_filter(
        files: list[str],
        xdur: float = 1.0,
        vfadetypes: Union[str, list[str]] = "fade",
        afadetypes: Union[str, list[str]] = "exp") -> str:
    """
    curve types for audio cross fade
    examples: http://underpop.online.fr/f/ffmpeg/help/afade.htm.gz
    common types: tri (linear, default), exp, [q/h/e]sin, cub, squ, cbr, log
    """
    
    vfade = "[{i}:v]xfade=transition={fade}:duration={d}:offset={off}[xv{i}]"
    afade = "[{i}:a]acrossfade=d={d}:c1={fade}:c2={fade}[xa{i}]"

    prev_offset = 0. 
    vfades = [] 
    afades = [] 

    # number of fades 
    num = (len(files) - 1)
    vfadetypes = get_fadetypes(vfadetypes, num)
    afadetypes = get_fadetypes(afadetypes, num)

    for i, f in enumerate(files[:-1]):
        dur = get_video_duration(f)
        off = dur + prev_offset - xdur 
        prev_offset = off 
        
        if i == 0:
            v_in, a_in = "[0:v]", "[0:a]"
        else:
            v_in, a_in = f"[xv{i}]", f"[xa{i}]"

        vfades.append(
            v_in +\
            vfade.format(
                d=xdur, off=off, i=i+1, fade=vfadetypes[i]
            )
        )

        afades.append(
            a_in +\
            afade.format(d=xdur, i=i+1, fade=afadetypes[i])
        )

    params = f"{'; '.join(vfades)}; {'; '.join(afades)}"
    lastMap = f"-map \"[xv{i+1}]\" -map \"[xa{i+1}]\""

    return f"-filter_complex \"{params}\" {lastMap}"

"""
Example of xfading select trimmed streams, and concatting the xfade output to a different trim
```
ffmpeg -i ".\saikou_menhera.mp4" -filter_complex "
    [0:v]trim=start=0:end=10,setpts=PTS-STARTPTS[1v];
    [0:a]atrim=start=0:end=10,asetpts=PTS-STARTPTS[1a];
    [0:v]trim=start=960:end=970,setpts=PTS-STARTPTS[2v];
    [0:a]atrim=start=960:end=970,asetpts=PTS-STARTPTS[2a];
    [0:v]trim=start=2820:end=2830,setpts=PTS-STARTPTS[3v];
    [0:a]atrim=start=2820:end=2830,asetpts=PTS-STARTPTS[3a];
    [1v][2v]xfade=transition=fade:duration=1:offset=9[xv1];
    [1a][2a]acrossfade=d=1:c1=exp:c2=exp[xa1];
    [xv1][xa1][3v][3a]concat=n=2:v=1:a=1[outv][outa]" `
    -map "[outv]" -map "[outa]" -movflags +faststart -c:v libx264 -c:a aac -shortest "./test_trim_concat_xfade.mp4"
```
"""

def crossfade(
    files: list[str],
    xdur: float = 1.0,
    vfadetypes: Union[str, list[str]] = "fade",
    afadetypes: Union[str, list[str]] = 'exp',) -> str:
    """Apply `xafde` to a list of video files

    Args:
        files (list[str]): list of video files
        xdur (float, optional): crossfade duration. Defaults to 1.0.
        fadetype (str, optional): video fade type. Defaults to "fade".
        afade_types (tuple[str, str], optional): audio fade type. Defaults to ('exp', 'exp').

    Returns:
        str: ffmpeg command
    
    For more `fadetype` arguments, see:
    https://trac.ffmpeg.org/wiki/Xfade

    For more `afade_types`, see:
    https://trac.ffmpeg.org/wiki/AfadeCurves
    """

    ins = build_input_string(files)
    filter_ = build_filter(
        files, 
        xdur=xdur, 
        vfadetypes=vfadetypes, 
        afadetypes=afadetypes
    )

    outpath = files[0].stem
    if "_" in outpath: 
        outpath = '_'.join(outpath.split("_")[:-1])
    outpath = files[0].parent / f"{outpath}_crossfade{files[0].suffix}"    

    if check_overwrite(outpath):
        cmd = f"{ins} {filter_} \"{outpath}\""
    
    Popen(['powershell.exe', cmd])
    return cmd 

# ---------------------------------------------------------------------------- #
#                                     Usage                                    #
# ---------------------------------------------------------------------------- #


VIDEO_PATH = Path("../../../../Videos/subtitles/full_raw/")


def main(filename: str, dir: Path = VIDEO_PATH, plot=True, splice=True) -> None:
	fp = VIDEO_PATH / filename
	signal, sr = open_audio(fp)
	intervals = get_intervals(
		signal, sr,
		window_length=51,
		polyorder=3,
		min_seg_dur=1.,
		min_sil_dur=0.25,
		max_sil_vol=-45,
		plot=plot
	)

	print("First 10 intervals:\n", intervals[:10, :])

	if splice:
		splice_video(fp.parent / fp.stem, np.round(intervals, decimals=3))


if __name__ == '__main__':
	main("ichinoseuruha_king.m4a", plot=False, splice=True)
