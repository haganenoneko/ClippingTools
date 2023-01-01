"""
Resize pictures of VTubers and overlay them on a video at specific times.
"""
import numpy as np
from subprocess import Popen
import tkinter as tk
from pathlib import Path
import pandas as pd
from typing import Union, Any

from common import get_filenames, check_overwrite, get_video_resolution, get_save_filename
from ass_tools import ASSReader, ASSWriter, INIT_DIR, VIDEO_DIR, ASSProcessor

ICONS_DIR = Path(
	r"C:/Users/delbe/Videos/subtitles/thumbs/vtuber_pngs/circle_icons/vspo")

VSPO_NAMES = [
	'Asumi', 'Kaminari', 'Kisaragi', 'Mimi',
	'Nekota', 'Shinomiya', 'Toto', 'Yakumo',
	'Hanabusa', 'Nazuna', 'Ramune', 'Sumire',
	'Ichinose', 'Ema', 'Beni', 'Hinano', 'Kurumi'
]


def _totsecs(x): return x.total_seconds()


class DropdownFileSelector:
	def __init__(self, icon_paths: dict[str, list[str]],) -> None:
		self.icon_paths = icon_paths
		self.boxes: list[tk.Frame] = []
		self.box_kw = dict(fill='x', expand=True)

		self.setup()
		self.createVariables()
		self.createButton()
		self.root.mainloop()

	def setup(self, geom='500x500', title='Find VTuber PNGs'):
		self.root = tk.Tk()
		self.root.geometry(geom)
		self.root.resizable(True, True)
		self.root.title(title)

	def createFrame(self) -> tk.Frame:
		frame = tk.Frame(self.root)
		frame.pack(padx=10, pady=10, **self.box_kw)
		self.boxes.append(frame)
		return frame

	def createVariables(self):

		self.variables: dict[str, tk.StringVar] = {}
		for name, paths in self.icon_paths.items():
			n = len(paths)
			if n == 1:
				self.icon_paths[name] = paths[0]
				continue

			if n > 1:
				var = tk.StringVar()
				var.set(paths[0])

				frame = self.createFrame()
				label = tk.Label(
					frame, text=f"Select a PNG for <{name}>")
				label.pack(**self.box_kw)

				menu = tk.OptionMenu(
					frame, var, *paths
				)
				menu.pack(**self.box_kw)

				self.variables[name] = var

			else:
				self.icon_paths[name] = get_filenames(
					title=f"Find icon PNG for {n}",
					filetypes=(("PNG files", "*.png"),),
					init_dir=ICONS_DIR,
					multiple=False
				)

	def createButton(self):
		button = tk.Button(
			self.boxes[-1],
			text='OK!',
			command=self.clicked)

		button.pack(pady=10, **self.box_kw)

	def clicked(self):
		for name, var in self.variables.items():
			self.icon_paths[name] = var.get()

		self.root.quit()
		self.root.destroy()

	def get(self):
		return self.icon_paths


class ImageOverlay:
	def __init__(
			self,
			icon_paths: dict[str, Path],
			noIcons: list[str] = ['Translator'],
			marginTop: float = 50.,
			marginBottom: float = 300.,
			borderPadding: float = 10.,
			iconPadding: float = 50.,
			minIconWidth: float = 140.,
			maxIconWidth: float = 280.) -> None:

		self.noIcons = noIcons
		self.icon_paths = icon_paths
		self.marginTop = marginTop
		self.marginBottom = marginBottom

		self.iconPadding = iconPadding
		self.borderPadding = borderPadding
		self.minIconWidth = minIconWidth
		self.maxIconWidth = maxIconWidth

		self.stylesWithIcons: list[str] = None

	def get_stylesWithIcons(
			self, df: pd.DataFrame, noIcons: list[str]) -> list[str]:
		"""Find styles in the ASS file that have icons

		Args:
				df (pd.DataFrame): dataframe of ASS dialog
				noIcons (list[str]): style names that do not have icons

		Returns:
			list[str]: style names that have icons. Paths to all icons, both in and not in the ASS file, are values of corresponding keys in `self.icon_paths`
		"""

		icon_names: list[str] = [s.title() for s in self.icon_paths.keys()]
		
		style_names = np.array([s.title() for s in df.Style.unique()])
		style_names.sort()

		hasIcon = np.isin(style_names, icon_names)

		stylesWithIcons: list[str] = [
			s for s in style_names[hasIcon]
			if s not in self.noIcons
		]

		for n in style_names[~hasIcon]:
			if n in self.noIcons:
				continue

			print(f"{n} has no icon. Select an icon, or close to ignore.")

			try:
				p = get_filenames(
					title=f"Select icon for {n}",
					filetypes=(("PNG files", "*.png"),),
					multiple=False,
					init_dir=ICONS_DIR
				)

				self.icon_paths[n] = p
				stylesWithIcons.append(n)

			except (IndexError, KeyboardInterrupt) as e:
				print(f"Error selecting icon for <{n}>. No icon will be used.")
				self.noIcons.append(n)

		return stylesWithIcons

	def check_icon_resolutions(self, reference_name=None) -> None:

		if reference_name in self.icon_paths:
			xy0 = get_video_resolution(
				self.icon_paths[reference_name], as_ints=True)
		else:
			xy0 = (None, None)

		for name, p in self.icon_paths.items():
			xy = get_video_resolution(p, as_ints=True)

			if xy[0] < self.minIconWidth:
				raise ValueError(
					f"The icon for {name} at {p} has width {xy[0]}, which is below the minimum {self.minIconWidth}"
				)

			if xy0[0] is None:
				xy0 = xy
				continue
			if xy == xy0:
				continue
			else:
				raise ValueError(
					f"The icon for {name} at {p} has dimensions {xy}, which differ from the reference icon {xy0}.")

		return

	def get_overlaps(self, df: pd.DataFrame, max_gap=0.1) -> list[int]:
		# if there are `n` overlapping subtitles,
		# use the `n+1`-th icon position

		numOverlaps = np.zeros(df.shape[0], dtype=int)
		hasIcon = df['hasIcon'] > 0
		df_icon = df.loc[hasIcon, :]

		numOverlaps[hasIcon] = [
			(
				(row.Start - df_icon['End'].iloc[:j]) < max_gap
			).sum()
			for j, row in
			enumerate(df_icon.itertuples())
		]

		return numOverlaps

	def get_icon_coords(
			self,
			num_icons: int,
			h_eff: float,
			w: float,
			mode: str) -> tuple[float, dict[str, tuple]]:

		# number of icons to place on the first side
		if 'bilateral' in mode:
			n1 = round(num_icons/2, ndigits=0)
		else:
			n1 = num_icons

		# height/width of each icon (1:1)
		y_icon = round(
			h_eff / max(n1, num_icons-n1),
			ndigits=1
		) - self.iconPadding

		if y_icon < self.minIconWidth:
			print(
				f"Too many icons ({num_icons}) to fit in effective height {h_eff} using mode <{mode}>"
			)
		elif y_icon > self.maxIconWidth:
			y_icon = self.maxIconWidth - self.iconPadding

		xy_dict: dict[str, tuple[float, float]] = {}
		for i, name in enumerate(self.stylesWithIcons):

			y = self.marginTop +\
				self.iconPadding/2

			if i > n1-1:
				y += y_icon*(i-n1)
			else:
				y += y_icon*i

			if 'right' in mode:
				x = self.borderPadding\
					if i > n1-1\
					else w - self.borderPadding - y_icon
			else:
				x = w - self.borderPadding - y_icon\
					if i > n1-1\
					else self.borderPadding\

			xy_dict[name] = (x, y)

		# even if the mode is 'speaker', we can simply take
		# xy_dict.values() and iterate through these coordinates
		# as needed
		return y_icon, xy_dict

	def compute_positions(
			self,
			video_path: Path,
			df: pd.DataFrame,
			mode: str) -> dict[str, Union[str, float, tuple[float]]]:

		width, height = get_video_resolution(
			video_path, as_ints=True)

		# icon resolution, assumed to be the same for each icon
		# x_im, y_im = get_video_resolution(
		# 	self.icon_paths[stylesWithIcons[0]],
		# 	as_ints=True
		# )

		# effective height = total height -\
		# 	space for main (center bottom) subtitles -\
		#	top and bottom padding
		# effective width = total width - left and right padding
		height_eff = height -\
			self.marginBottom -\
			self.marginTop -\
			2*self.borderPadding

		if 'speaker' in mode:
			num_icons = 1 + df.sort_values('Start').\
				loc[:, 'PositionIndex'].\
				max()
		else:
			num_icons = len(self.stylesWithIcons)

		icon_xy: dict[str, tuple[float, float]] = {}
		y_icon, icon_xy = self.get_icon_coords(
			num_icons, height_eff, width, mode
		)

		return dict(y_icon=y_icon, icon_xy=icon_xy, mode=mode)

	@staticmethod
	def scale_split(ind: int, y: float, num: int, suffix='png') -> str:
		in_ = f"[{ind}:v]"
		src = f"{ind}{suffix}"

		if num > 0:
			cmd = f"scale={y}:-1,split={num}"
			out_ = ''.join([f"[{src}_{i}]" for i in range(1, num+1)])
		else:
			cmd = f"scale={y}:-1"
			out_ = f"[{src}]"

		return in_ + cmd + out_

	@staticmethod
	def joinFilt(scale: list[str], overlay: list[str]) -> str:

		scale, overlay = map(
			lambda L: '; '.join(L),
			[scale, overlay]
		)

		return f"{scale}; {overlay}"

	def speaker_filter_elements(
			self,
			df: pd.DataFrame,
			y_icon: float,
			icon_xy: dict[str, tuple[float]]) -> tuple[str, str]:

		# scale and then split each icon PNG into
		# `num` streams, i.e. however many times
		# the icon appears
		scales_splits = []

		# number of times the icon appears throughout the video
		num_styles: dict[str, int] = {}

		for i, style in enumerate(self.stylesWithIcons):
			num = (df['Style'] == style).sum()
			num_styles[style] = num
			scales_splits.append(
				self.scale_split(i+1, y_icon, num,)
			)

		# i = counter for filter elements
		i = 0

		df_hasIcon = df.loc[df['hasIcon'] > 0, :]
		coords = list(icon_xy.values())

		OVRL = "overlay={x}:{y}:enable='between(t,{tA},{tB})'"
		overlays: list[str] = []

		# j = counter for subtitle lines
		for j, row in enumerate(df_hasIcon.itertuples()):

			# count number of times this style has been
			# overlaid, i.e. index of cloned `scale`
			if j == 0:
				n_clone = 1
			else:
				n_clone = (
					df_hasIcon.iloc[:j, :]['Style']
					== row.Style
				).sum() + 1

			# icon coordinates
			x, y = coords[row.PositionIndex]

			src1 = "[0:v]" if j == 0 else f"[{i}ov]"

			if num_styles[row.Style] > 0:
				src2 = f"[{row.StyleInd+1}png_{n_clone}]"
			else:
				src2 = f"[{row.StyleInd+1}png]"

			ovrl = OVRL.format(
				x=x, y=y, tA=row.Start, tB=row.End
			) + f"[{i+1}ov]"

			overlays.append(src1 + src2 + ovrl)
			i += 1

		# connect last output
		lastMap = f"-map \"[{i}ov]\""

		return self.joinFilt(scales_splits, overlays), lastMap

	def nonSpeaker_filter_elements(
			self,
			y_icon: float,
			icon_xy: dict[str, tuple[float, float]]) -> tuple[str, str]:

		SCALE = "[{i}:v]scale={y}:-1[{i}png]"
		OVERLAY = "[{src}][{ind}png]overlay={x}:{y}[{ind}ov]"

		scales, overlays = [], []

		for i, name in enumerate(self.stylesWithIcons):
			x, y = icon_xy[name]

			scales.append(
				SCALE.format(i=i+1, y=y_icon)
			)

			overlays.append(
				OVERLAY.format(
					src="0:v" if i == 0
					else f"{i}ov",
					ind=i+1,
					x=x, y=y
				)
			)

		lastMap = f"-map \"[{i+1}ov]\""
		return self.joinFilt(scales, overlays), lastMap

	def build_inputs(self, vp: Path) -> str:
		cmd = f" -i \"{vp}\""
		for style in self.stylesWithIcons:
			cmd += f" -i \"{self.icon_paths[style]}\" "

		return cmd

	def run_overlay(
			self,
			vp: Path,
			df: pd.DataFrame,
			y_icon: float,
			icon_xy: dict[str, tuple[float, float]],
			mode: str,
			overwrite=False) -> None:

		inputs = self.build_inputs(vp)

		if 'speaker' in mode:
			filt, lastMap = self.speaker_filter_elements(
				df, y_icon, icon_xy)
		else:
			filt, lastMap = self.nonSpeaker_filter_elements(
				y_icon, icon_xy)

		outpath = vp.parent /\
			f"{vp.stem}_overlay{vp.suffix}"

		tmpfile = Path("./logs/tmp_overlay_params.txt")
		with open(tmpfile, 'w') as tmp:
			tmp.write(filt)
		print(f"Filter written to: {tmpfile}")

		if check_overwrite(outpath):
			out = f"{lastMap} -map 0:a -c:a copy \"{outpath}\""

			paramsFile = str(tmpfile.absolute()).replace('/', r'//')
			cmd = f"ffmpeg {inputs} -filter_complex_script {paramsFile} {out}"

			print(cmd)
			Popen(['powershell.exe', cmd])
		else:
			raise FileExistsError(outpath)

	def get_subtitle_coordinates(
			self,
			df_xy: pd.DataFrame,
			y_icon: float,
			hasIcon: np.ndarray,
			onRight: list[bool] = None) -> np.ndarray:
		"""Get coordinates for subtitles

		Args:
			df_xy (pd.DataFrame): `DataFrame` with columns for `x` and `y` coordinates
			y_icon (float): height (and width) of icons
			hasIcon (np.ndarray): Boolean mask of rows in `df_xy` with icons, i.e. `df_xy['x'] == 0`
			onRight (list[bool], optional): Boolean mask of rows in `df_xy` that are located on the right. Defaults to None.

		Returns:
			np.ndarray: `N x 2` array of subtitle positions 
		"""

		coords = np.zeros((df_xy.shape[0], 2))

		coords[:, 1] = (df_xy['y'] + y_icon/2) * hasIcon
		coords[hasIcon, 0] += self.iconPadding + y_icon

		if onRight is None:
			return coords
		else:
			coords[onRight, 0] = df_xy.\
				loc[onRight, 'x'].iat[0] -\
				self.iconPadding
			return coords

	def add_RHS_styles(
			self,
			df1: pd.DataFrame,
			df_styles: pd.DataFrame,
			onRight: list[bool],
			margin_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
		"""Add right-center aligned text styles for subtitles on the right"""

		if onRight is None:
			return df1, df_styles

		# names of styles that need a RHS duplicate
		names = df1.loc[onRight, 'Style'].unique()

		# rename styles on the right
		df1.loc[onRight, 'Style'] = df1.loc[onRight, 'Style'] + "_RHS"

		# select styles which need a RHS duplicate
		needsRHS = df_styles['Name'].isin(names)

		# RHS styles
		df_rhs = df_styles.loc[needsRHS, :].copy()
		df_rhs['MarginL'] = margin_size

		df_rhs['Name'] = df_rhs['Name'].\
			apply(lambda n: f"{n}_RHS")

		# change alignment
		df_rhs['Alignment'] = 6

		df_styles = pd.concat(
			[df_styles, df_rhs],
			axis=0
		)

		return df1, df_styles

	@staticmethod
	def _get_xy_coordinates(
		df2: pd.DataFrame,
		mode: str,
		posInfo: dict[str, Any],
		hasIcon: pd.Series) -> pd.DataFrame:
		
		if 'speaker' in mode:
			# x, y coordinates for each subtitle
			xy: list[tuple[float, float]] = list(
				posInfo['icon_xy'].values()
			)

			df_xy = df2['PositionIndex'].\
				apply(lambda n: xy[n])

		else:
			xy: dict[str, tuple] = posInfo['icon_xy']
			df_xy = df2['Style'].apply(
				lambda k: xy[k] 
				if k in xy 
				else (0, 0)
			)

		# epxand dataframe with 2-tuples as values into 
		# N x 2 dataframe with separate x, y columns 
		df_xy = df_xy.apply(lambda x: pd.Series(x))
		df_xy.columns = ['x', 'y']

		# zero all positions for subtitles without icons
		df_xy.loc[~hasIcon, :] = (0, 0)

		return df_xy 

	def adjust_ASS_subtitle_positions(
			self,
			df1: pd.DataFrame,
			df2: pd.DataFrame,
			df_styles: pd.DataFrame,
			mode: str,
			posInfo: dict[str, Any]) -> pd.DataFrame:

		hasIcon = df2['hasIcon'] > 0

		df_xy = self._get_xy_coordinates(
			df2, mode, posInfo, hasIcon
		)
		
		# find subtitles that are located on the right 
		if (df_xy['x'] > self.borderPadding).any():
			onRight = df_xy['x'] > self.borderPadding
		else:
			onRight = None

		coords = self.get_subtitle_coordinates(
			df_xy, 
			posInfo['y_icon'], 
			hasIcon.values, 
			onRight=onRight
		)

		# every style with an icon is
		# aligned 'center left', ie. 4 in Aegisub
		_pos = "{{\pos({x},{y})}} "

		df1.loc[hasIcon, 'Text'] = np.apply_along_axis(
			lambda tup: _pos.format(x=tup[0], y=tup[1]),
			arr=coords[hasIcon, :],
			axis=1,
		) + df1.loc[hasIcon, 'Text'].str.strip()

		# set alignment of styles with icons to center left
		hasIcon = df_styles.Name.isin(self.stylesWithIcons)
		df_styles.loc[hasIcon, 'Alignment'] = 4

		margin_size = int(
			self.borderPadding +
			posInfo['y_icon'] + 
			self.iconPadding
		)

		df_styles.loc[hasIcon, 'MarginR'] = margin_size

		# any subtitles on the left are given a duplicate
		# style that is aligned to the center right
		if onRight is None:
			return df1, df_styles
		else:
			return self.add_RHS_styles(
				df1,
				df_styles,
				onRight,
				margin_size
			)

	def process_dialog(self, df: pd.DataFrame) -> pd.DataFrame:

		df_out = df.loc[:, ['Start', 'End', 'Style']].copy()

		style_inds: dict[str, int] = {
			style: i for i, style in
			enumerate(self.stylesWithIcons)
		}

		for col in ['Start', 'End']:
			df_out[col] = pd.to_timedelta(df[col]).\
				apply(lambda x: x.total_seconds())

		df_out['StyleInd'] = df.Style.str.title().\
			apply(
			lambda s: -1 if s in self.noIcons
			else style_inds[s]
		)

		df_out['hasIcon'] = df['Style'].str.title().\
			isin(self.stylesWithIcons)

		newcols = ['NumOverlaps', 'PositionIndex']
		df_out.loc[df_out.hasIcon, newcols] = np.vstack(
			ASSProcessor().
			get_posIndices(
				df_out.loc[df_out.hasIcon, :]
			)
		).T

		df_out.loc[:, newcols] = df_out.loc[:, newcols].\
			fillna(0).astype('Int64')

		return df_out

	def overlay(
			self,
			df_dialog: pd.DataFrame,
			df_styles: pd.DataFrame,
			video_path: Path,
			mode: str = 'left',
			noIcons: list[str] = None,
			run_overlay=True,
			write_ass=True) -> pd.DataFrame:

		self.stylesWithIcons = self.get_stylesWithIcons(
			df_dialog,
			noIcons=noIcons if noIcons
			else self.noIcons
		)

		# processed dialog dataframe
		df_pro = self.process_dialog(df_dialog)

		# icon position information
		posInfo = self.compute_positions(
			video_path,
			df_pro,
			mode
		)

		print(posInfo)

		if run_overlay:
			self.run_overlay(video_path, df_pro, **posInfo)

		if write_ass:
			df_dialog, df_styles = self.adjust_ASS_subtitle_positions(
				df_dialog, df_pro, df_styles, mode, posInfo
			)

			outname = get_save_filename(
				title="Save output ASS file after overlay.",
				init_dir=INIT_DIR,
				initialfile=f"{video_path.stem}_overlay",
				filetypes=(("ASS files", "*.ass"),)
			)

			Writer = ASSWriter()
			Writer.write(
				outname.stem,
				df_dialog=df_dialog,
				df_styles=df_styles,
				outdir=outname.parent,
				vpath=video_path,
				apath=video_path
			)


def get_icon_paths(names=VSPO_NAMES, icondir=ICONS_DIR) -> dict[str, Path]:

	icons: dict[str, list[str]] = {
		name.title(): list(
			icondir.glob(f"*{name.lower()}*.png")
		) for name in names
	}

	icons = DropdownFileSelector(icons).get()
	return icons


def main(
	names=VSPO_NAMES, icondir=ICONS_DIR,
	noIcons: list[str] = ['Translator'],
	dim_kw: dict[str, float] = dict(
		marginTop=110.,
		marginBottom=145.,
		iconPadding=10.,
		borderPadding=10.,
		minIconWidth=100.,
		maxIconWidth=230.
	),
	overlay_mode='left',
	**kwargs
) -> None:

	icon_paths = get_icon_paths(names, icondir)
	Overlay = ImageOverlay(icon_paths, noIcons=noIcons, **dim_kw)

	ass_files = get_filenames(
		title='Open ASS files',
		filetypes=(('ASS files', '*.ass'),),
		init_dir=INIT_DIR,
		multiple=True
	)

	Reader = ASSReader()

	for fp in ass_files:

		vp, df_styles, df_dialog = Reader.read_file(
			fp, return_dataframes=True)

		if not vp.is_file():
			vp = get_filenames(
				f"Find video for {fp.stem}...",
				filetypes=(('Video file', '*.mp4'),),
				init_dir=VIDEO_DIR,
				multiple=False
			)

		Overlay.overlay(
			df_dialog=df_dialog,
			df_styles=df_styles,
			video_path=vp,
			mode=overlay_mode,
			**kwargs
		)


if __name__ == '__main__':
	main(
		overlay_mode='left-bilateral',
		noIcons=['Default', 'Translator', 'Chat'],
		dim_kw=dict(
			marginTop=50.,
			marginBottom=50.,
			iconPadding=10.,
			borderPadding=5.,
			minIconWidth=150.,
			maxIconWidth=300.
		),
		run_overlay=True,
		write_ass=True,
	)

"""
TODO
1. Grid positioning of icons, i.e. in different columns/rows, e.g. gridspec
2. Specify icon sizes by ratios, instead of all being the same.
"""
