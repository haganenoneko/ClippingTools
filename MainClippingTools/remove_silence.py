import audiofile 
import numpy as np
from pathlib import Path 
from scipy.signal import savgol_filter

def db_to_AR(dB: float) -> float:
	"""Convert decibels to amplitude ratio"""
	return np.sqrt(10 ** (dB / 10))

def AR_to_dB(AR: float) -> float:
	"""Convert amplitude ratio to decibels"""
	return  np.log10(AR ** 2) * 10

def stereo_to_mono(signal: np.ndarray) -> np.ndarray:
	if len(signal.shape) < 2:
		return signal 

	if signal.shape[0] == 1:
		return signal[0,:]
	elif signal.shape[0] == 2:
		return np.mean(signal / 2, axis=0)
	else:
		raise ValueError(f"Signal has shape: {signal.shape}. Expected (0-2, N)")

def open_audio(fp: Path, mono=True, **kwargs) -> tuple[np.ndarray, int]:
	signal, sr = audiofile.read(fp, **kwargs)
	if mono:
		return stereo_to_mono(signal), sr 
	else:
		return signal, sr 
	
def filter(signal: np.ndarray, window_length=121, polyorder=2) -> np.ndarray:
	return savgol_filter(np.abs(signal), window_length, polyorder=polyorder)

def array_to_tuples(arr: np.ndarray) -> list[tuple]:
	return list(map(tuple, arr))

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
		p = np.cumsum(np.append(0, z))[:-1] # positions
		return(z, p, ia[i])

def partition(arr: np.ndarray, size=2, step=1, min_delta=0):
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
) -> np.ndarray:
	
	min_seg_dur *= sampling_rate
	min_sil_dur *= sampling_rate

	filt = filter(signal, window_length=window_length, polyorder=polyorder)
	lens, inds, vals = rle(filt <= db_to_AR(max_sil_vol))
	
	n = lens.shape[0]	
	ret: list[tuple] = [] 
	
	for i in range(lens.shape[0]):
		if vals[i] == 1 and lens[i] < min_sil_dur: 
			continue
		if vals[i] == 0 and lens[i] > min_seg_dur:
			continue
		ret.append((inds[i], inds[i] + lens[i]))
	
	out = np.array(ret)
	out[1:, 0] += 1

	# add padding between adjacent intervals
	if adj_seg_pad > 0:
		out = pad_intervals(out, round(adj_seg_pad*sampling_rate), signal.shape[0])

	# clean up intervals 
	out = clean_intervals(
		out / sampling_rate, 
		min_seg_dur / sampling_rate,
		min_gap_dur
	)

	return invert_intervals(out, signal.shape[0]/sampling_rate) 

VIDEO_PATH = Path("../../../../Videos/subtitles/full_raw/")
fp = VIDEO_PATH / "2022_top22/kamito_voice_aug_26.m4a"
signal, sr = open_audio(fp)
intervals = get_intervals(
	signal, sr,
	window_length=21,
	polyorder=3,
	min_seg_dur=1.,
	min_sil_dur=0.25,
	max_sil_vol=-45
)
print(intervals)