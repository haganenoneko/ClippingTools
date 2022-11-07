import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from types import NoneType
from datetime import timedelta

from common import check_overwrite

# directory containing subtitle files
INIT_DIR = Path("C:/Users/delbe/Videos/subtitles/subs")

# directory for video files
VIDEO_DIR = Path(r"C:/Users/delbe/Videos/subtitles/full_raw")

HOME_DIR = Path.cwd() / "MainClippingTools"

ASS_HEADER_PATH = HOME_DIR / "assets/ASS_header.txt"

# ---------------------------------------------------------------------------- #
#                                Regex patterns                                #
# ---------------------------------------------------------------------------- #

DIALOG_PAT = re.compile(r"^(?:Dialogue\:\s\d\,)([\d\:\.]+),([\d\:\.]+).*$")

DIALOG_TEMPLATE =\
	"Dialogue: {layer},{start_time},{end_time},{style},{name},{marginL},{marginR},{marginV},{effect},{text}"

# ------------------------------ Other functions ----------------------------- #


def clean_intervals(
		intervals: list[tuple],
		min_dur: float = 0.5,
		min_gap: float = 0.2,
) -> list[tuple[float, float]]:
	"""Clean up intervals by

			1. ensuring intervals are greater than a minimum duration
			2. ensuring intervals are separated by a minimum duration
			3. concatenating overlapping intervals (https://stackoverflow.com/a/58976449/10120386)

	Args:
			intervals (list[tuple]): list of `(start, end)` times
			min_dur (float, optional): minimum interval duration. Defaults to 0.1.
			min_gap (float, optional): minimum separation between adjacent intervals. Defaults to 0.2.

	Returns:
			list[tuple[float, float]]: list of processed `(start, end)` times
	"""

	arr = np.sort(np.array(intervals), axis=0)

	# remove intervals with less than the minimum duration
	durs = arr[:, 1] - arr[:, 0]
	arr = arr[durs > min_dur]

	# remove overlapping intervals and ensure gaps > min_gap
	valid = np.zeros(arr.shape[0]+1, dtype=bool)
	valid[[0, -1]] = 1
	valid[1:-1] = arr[1:, 0] -\
		np.maximum.accumulate(arr[:-1, 1]) >= min_gap

	merged = np.vstack(
		(
			arr[valid[:-1], 0],
			arr[valid[1:], 1]
		)
	).T

	print(
		f"""Results of `clean_intervals`
		Initial no.:\t{len(intervals)}
		Cleaned no.:\t{merged.shape[0]}
		"""
	)

	return tuple(map(tuple, merged))


# ---------------------------------------------------------------------------- #
#                               Writing ASS files                              #
# ---------------------------------------------------------------------------- #


class ASSWriter:
	def __init__(
			self,
			dialog_template: str = DIALOG_TEMPLATE,
			dialog_kwargs: dict[str, str] = dict(
				layer=0, style='Default', name='',
				marginL=0, marginR=0, marginV=0, effect=''),
			header: Union[str, Path] = ASS_HEADER_PATH,
	) -> None:

		self._dialogTemplate = dialog_template
		self._dialog_kw = dialog_kwargs
		self._headerPath = header

		self._vec_ts = np.vectorize(self._ts)

	@staticmethod
	def _ts(seconds: float) -> str:
		"""Convert seconds into a HH:MM:SS timestamp"""
		return str(timedelta(seconds=seconds))

	def add_dialog(self, text: str, start_time: float, end_time: float, **kwargs) -> str:
		"""Create single line of ASS dialog"""

		return self._dialogTemplate.format(
			start_time=start_time, end_time=end_time, text=text,
			**kwargs
		)

	def create_ass_dialog(
			self,
			intervals: list[tuple[float, float]],
			text: list[str] = None,) -> str:

		dialog: str = ''
		starts, ends = map(self._vec_ts, zip(*intervals))

		if text is None:
			for start, end in zip(starts, ends):
				line = self.add_dialog(
					'n/a', start, end,
					**self._dialog_kw)

				dialog += f"{line}\n"
		else:
			for start, end, t in zip(starts, ends, text):
				line = self.add_dialog(
					t, start, end,
					**self._dialog_kw)

				dialog += f"{line}\n"

		return dialog.rstrip()

	@staticmethod
	def add_media(
			header: str,
			vpath: Union[str, Path],
			apath: Union[str, Path]) -> str:

		if vpath:
			if not isinstance(vpath, Path):
				vpath = Path(vpath)

		if not isinstance(apath, Path):
			apath = Path(apath)

		lines = header.split('\n')
		if vpath:
			lines[13] += str(vpath.absolute())
		if apath:
			lines[14] += str(apath.absolute())

		return '\n'.join(lines)

	def get_header(
			self,
			header: Union[str, Path],
			vpath: Union[str, Path],
			apath: Union[str, Path],
			styles: str=None) -> str:
		"""
		Get header for the ASS file
		TO DO: add path to video/audio files
		"""

		if isinstance(header, Path):
			with open(header, 'r', encoding='utf-8') as io:
				if styles is None:
					header = io.read()
				else:
					header = io.readlines()
					header = ''.join(header[:24]) +\
						f"{styles}\n" +\
						''.join(header[25:])

		if (vpath is None) and\
				(apath is None):
			return header

		if vpath:
			if apath:
				header = self.add_media(header, vpath, apath)
			else:
				header = self.add_media(header, vpath, vpath)
		else:
			header = self.add_media(header, None, apath)

		return header

	@staticmethod
	def format_dataframe_section(df: pd.DataFrame, prefix: str) -> str:
		return '\n'.join(
			df.astype(str).T.apply(
				lambda row: f"{prefix}: {','.join(row)}"
			).values.tolist()
		)

	@staticmethod
	def save(
			outname: str,
			header: str,
			dialog: str,
			outdir: Path,) -> str:

		full_content = header + '\n' + dialog

		if not isinstance(outdir, Path):
			raise TypeError(f"{outdir} must be a Path")
		if not outdir.is_dir():
			raise FileNotFoundError(
				f"{outdir} is not a valid directory."
			)

		outpath = outdir / f"{outname}.ass"
		if check_overwrite(outpath):
			with open(outpath, 'w', encoding='utf-8') as io:
				io.write(full_content)

		print(f"New ass file created at {outpath}")
		return full_content

	def write_from_intervals(
			self,
			outname: str,
			seconds: list[float],
			text: list[str] = None,
			outdir: Path = HOME_DIR,
			vpath: Union[str, Path] = None,
			apath: Union[str, Path] = None,) -> str:
		"""Create a new ASS file based on list of times and (optional) dialog text."""

		ass_header = self.get_header(self._headerPath, vpath, apath)
		dialog = self.create_ass_dialog(seconds, text)
		return self.save(outname, ass_header, dialog, outdir)

	def write_from_dataframes(
			self,
			outname: str,
			df_styles: pd.DataFrame,
			df_dialog: pd.DataFrame,
			outdir: Path = HOME_DIR,
			vpath: Union[str, Path] = None,
			apath: Union[str, Path] = None

	) -> str:
		styles = self.format_dataframe_section(df_styles, prefix='Style')

		ass_header = self.get_header(
			self._headerPath, vpath, apath,
			styles=styles
		)
		
		dialog = self.format_dataframe_section(df_dialog, prefix='Dialogue')
		return self.save(outname, ass_header, dialog, outdir)

	def write(
			self,
			outname: str,
			seconds: list[float] = None,
			text: list[str] = None,
			df_styles: pd.DataFrame = None,
			df_dialog: pd.DataFrame = None,
			outdir: Path = HOME_DIR,
			vpath: Union[str, Path] = None,
			apath: Union[str, Path] = None
	) -> str:

		if seconds is not None:
			return self.write_from_intervals(
				outname,
				seconds,
				text=text,
				outdir=outdir,
				vpath=vpath,
				apath=apath
			)

		if df_styles is not None:
			return self.write_from_dataframes(
				outname,
				df_styles,
				df_dialog,
				outdir=outdir,
				vpath=vpath,
				apath=apath
			)


def parse_interval_file(fpath: Path, as_tuples=False) -> list[float]:
	ext = fpath.suffix
	if ext in ['.csv', '.tsv']:
		df = pd.read_csv(
			fpath, header=None, index_col=None,
			sep=',' if ext == '.csv' else '\t'
		)

		df = pd.read_csv(fpath, header=None)

		if as_tuples:
			return list(df.itertuples(name=None, index=False))
		else:
			secs = df.values.flatten()
			secs.sort()
			return secs

	if ext == '.txt':
		with open(ext, 'r', encoding='utf-8') as file:
			secs = [float(x.strip()) for x in file.read().split(',')]

		if as_tuples:
			secs = [
				(secs[i], secs[i+1])
				for i in range(0, len(secs), 2)
			]

		return secs

	raise Exception(f"{fpath} must have .csv, .tsv, or .txt extension")


# ---------------------------------------------------------------------------- #
#                               Reading ASS files                              #
# ---------------------------------------------------------------------------- #

class VideoNotFoundError(FileNotFoundError):
	def __init__(self, ass_path: Path) -> None:
		super().__init__(f"No video was found for ASS file at\n{ass_path}")


class ASSReader:
	def __init__(self) -> None:

		self.lines: list[str] = None 
		self.ass_path: Path = None 

		# list[str] -> np.np.ndarray[float]
		self._v_ts2secs = np.vectorize(
			lambda x: self._ts2secs(x)
		)

		# list[str] -> tuple[list[str], list[str]]
		self._v_extractTimes = np.vectorize(
			lambda x: DIALOG_PAT.match(x).groups()
		)

	@staticmethod
	def _ts2secs(s: str) -> float:
		hms, msec = s.split('.')
		h, m, s = map(int, hms.split(':'))
		return 3600*h + 60*m + s + int(msec)/100

	def get_intervals(self, lines: list[str]) -> list[tuple[float, float]]:
		return list(zip(*map(
			self._v_ts2secs,
			self._v_extractTimes(lines))))

	def get_ass_elements(self) -> tuple[Path, list[str], list[str]]:
		"""Find styles, dialog, and video path from an ASS file

		Raises:
				ValueError: No styles found
				ValueError: No dialog found
				ValueError: Neither styles nor dialog were found
				VideoNotFoundError: No valid video path was found

		Returns:
				tuple[Path, list[str], list[str]]: video path, styles, and dialog 
		"""

		video_path, i_style, i_dialog = [None]*3

		for i, line in enumerate(self.lines):
			if ("Video File:" == line[:11]) and\
				(video_path is None):
				video_path = Path(line.split(': ')[1].strip())

			if ("Style: " == line[:7]) and\
				(i_style is None):
				i_style = i

			if ("Dialogue:" == line[:9]) and\
				(i_dialog is None):
				i_dialog = i
				break

		err = "No <{0}> found in " + str(self.ass_path)
		if (i_style is None) and (i_dialog is None):
			raise ValueError(err.format("styles or dialog"))
		elif i_style is None:
			raise ValueError(err.format("styles"))
		elif i_dialog is None:
			raise ValueError(err.format("dialog"))

		styles = self.lines[i_style:i_dialog-3]
		dialog = self.lines[i_dialog:]

		return video_path, styles, dialog 

	@staticmethod
	def _splitCommas(lines: list[str], n: int, ltrunc: int, strip=True) -> list[str]:

		if strip:
			def func(x): return x[ltrunc:].strip().split(',', n)
		else:
			def func(x): return x[ltrunc:].split(',', n)

		return [func(line) for line in lines]

	def _parseSection(self, lines: list[str], colnames: list[str], ltrunc: int) -> pd.DataFrame:
		data = self._splitCommas(lines, len(colnames)-1, ltrunc)
		df = pd.DataFrame.from_records(data, columns=colnames)
		return df 

	def parse_dialog(self, dialog: list[str], timecodes_as_seconds=False) -> pd.DataFrame:

		# Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
		colnames = [
			'Layer', 'Start', 'End', 'Style', 'Name',
			'MarginL', 'MarginR', 'MarginV', 'Effect', 'Text']

		df = self._parseSection(dialog, colnames, ltrunc=10)

		if timecodes_as_seconds:
			for col in ['Start', 'End']:
				df[col] = pd.to_timedelta(df[col]).\
					apply(lambda x: x.total_seconds())
		return df

	def parse_styles(self, styles: list[str]) -> pd.DataFrame:
		# Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
		colnames = [
			'Name', 'Fontname', 'Fontsize', 'PrimaryColour', 'SecondaryColour', 'OutlineColour', 'BackColour', 'Bold', 'Italic', 'Underline', 'StrikeOut', 'ScaleX', 'ScaleY', 'Spacing', 'Angle', 'BorderStyle', 'Outline', 'Shadow', 'Alignment', 'MarginL', 'MarginR', 'MarginV', 'Encoding']

		df = self._parseSection(styles, colnames, ltrunc=7)
		return df 

	def read_intervals(
			self,
			dialog: list[str],
			clean=True,
			min_dur=0.1, min_gap=0.2) -> list[float]:

		intervals = self.get_intervals(dialog)
		if clean:
			return clean_intervals(intervals, min_dur, min_gap)
		else:
			return intervals

	@staticmethod
	def intervals2csv(outpath: Path, intervals: list[tuple[float, float]]) -> pd.DataFrame:

		df = pd.DataFrame.from_records(intervals)
		df.columns = ['Start', 'End']

		if check_overwrite(outpath):
			df.to_csv(outpath)

		return df

	def read_file(self, fp: Path, return_dataframes=False) -> Union[NoneType, tuple[pd.DataFrame]]:

		if fp.is_file():
			self.ass_path = fp
		else:
			raise ValueError(f"{fp} is not a valid file.")

		with open(fp, 'r', encoding='utf-8') as io:
			self.lines = io.readlines()

		if return_dataframes:
			vp, styles, dialog = self.get_ass_elements()
			return vp, self.parse_styles(styles), self.parse_dialog(dialog)
		else:
			return self.get_ass_elements()

# ---------------------------------------------------------------------------- #
#                               Process ASS file                               #
# ---------------------------------------------------------------------------- #

class ASSProcessor:
	def __init__(self, reader: ASSReader=None, ass_path=None) -> None:
		self.reader: ASSReader = None 
		self.df_dialog: pd.DataFrame = None 
		self.df_styles: pd.DataFrame = None 

	def load_data(self, reader: ASSReader, ass_path: Path) -> None:
		if reader.lines is None:
			if ass_path is None:
				raise ValueError(f"Must provided ASSReader with data loaded already or valid ASS file path")
			else:
				self.df_styles, self.df_dialog = reader.read_file(ass_path, return_dataframes=True)
		else:
			_, styles, dialog = reader.get_ass_elements()
			self.df_styles = reader.parse_styles(styles)
			self.df_dialog = reader.parse_dialog(dialog, timecodes_as_seconds=True)
	
	@staticmethod
	def get_posIndices(df: pd.DataFrame) -> tuple[np.ndarray]:
		times = df.loc[:, ['Start', 'End']].\
			sort_values('Start').values  
		
		deltas = np.triu(
			times[:,0] - times[:,1][:, np.newaxis], 
			k=1
		)
		
		numOverlaps = (deltas < 0).sum(axis=0)
		
		pos = np.zeros(times.shape[0], dtype=int)
		inds = np.arange(times.shape[0], dtype=int)
		
		for i in range(1, times.shape[0]):
			if numOverlaps[i] > 0:
				pos[i] = np.setdiff1d(
					inds[:i+1], 
					pos[:i+1][deltas[:i+1, i] < 0]
				).min()

		return numOverlaps, pos 
	
	def process_dialog(self) -> pd.DataFrame:
		return self.get_posIndices(self.df_dialog)