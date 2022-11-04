"""
Resize pictures of VTubers and overlay them on a video at specific times.
"""
import numpy as np
from subprocess import Popen
import tkinter as tk
from pathlib import Path
import pandas as pd
from typing import Union

from common import get_filenames, check_overwrite, run_powershell, get_video_resolution
from ass_tools import ASSReader, INIT_DIR, VideoNotFoundError

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
			marginBottom: float = 200.,
			borderPadding: float = 10.,
			iconPadding: float=10.,
			minIconWidth: float = 150.,
			maxIconWidth: float = 300.) -> None:

		self.noIcons = noIcons
		self.icon_paths = icon_paths
		self.marginBottom = marginBottom

		self.iconPadding = iconPadding
		self.borderPadding = borderPadding
		self.minIconWidth = minIconWidth
		self.maxIconWidth = maxIconWidth

		self.fmts = self.get_fmts()
		self.stylesWithIcons: list[str] = None

	def get_fmts(self):
		fmts = {
			# input
			'i': "-i \"{p}\"",
			# scaling
			's': "[{i}:v]scale={y}:-1[{i}png]",
			# overlay
			'o': "[{src}][{i}png]overlay={x}:{y}[{i}ov]",
			'ot': "[{src}][{pi}png]overlay={x}:{y}:enable='between(t,{tA},{tB})'[{oi}ov]"
		}

		return fmts 

	def get_stylesWithIcons(
			self, df: pd.DataFrame, noIcons: list[str]) -> list[str]:
		"""Find styles in the ASS file that have icons

		Args:
				df (pd.DataFrame): dataframe of ASS dialog
				noIcons (list[str]): style names that do not have icons

		Returns:
				list[str]: style names that have icons. Paths to all icons, both in and not in the ASS file, are values of corresponding keys in `self.icon_paths`
		"""

		icon_names: list[str] = list(self.icon_paths.keys())
		style_names = np.array([s.title() for s in df.Style.unique()])
		hasIcon = np.isin(style_names, icon_names)
		
		stylesWithIcons = style_names[hasIcon].tolist()

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

	def get_icon_xy(
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

		xy_dict: dict[str, tuple[float, float]] = {}
		for i, name in enumerate(self.stylesWithIcons):
			y = (i+1)*y_icon - self.iconPadding/2

			if 'right' in mode:
				x = self.borderPadding\
					if i > n1\
					else w - self.borderPadding
			else:
				x = self.borderPadding\
					if i < n1\
					else w - self.borderPadding -\
						y_icon

			xy_dict[name] = (x, y)

		# even if the mode is 'speaker', we can simply take
		# xy_dict.values() and iterate through these coordinates
		# as needed
		return y_icon, xy_dict

	def compute_positions(
			self,
			video_path: Path,
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
		height_eff = height - self.marginBottom - 2*self.borderPadding

		num_icons = len(self.stylesWithIcons)

		y_icon = height_eff / num_icons
		icon_xy: dict[str, tuple[float, float]] = {}

		if y_icon > self.maxIconWidth:
			y_icon = self.maxIconWidth

		elif y_icon < self.minIconWidth:
			if y_icon*2 < self.minIconWidth:
				raise ValueError(
					f"Too many icons to fit with minimum height {self.minIconWidth}.")
			else:
				y_icon *= 2
				mode = 'bilateral'
				print("Can't fit all icons on one side. Setting `mode=bilateral`")

		y_icon, icon_xy = self.get_icon_xy(
			num_icons, height_eff, width, mode
		)

		return dict(y_icon=y_icon, icon_xy=icon_xy, mode=mode)

	def get_speaker_overlays(
			self,
			df: pd.DataFrame,
			coords: list[tuple[float, float]]) -> list[str]:

		overlays: list[str] = []

		# i = coutner for filter elements
		i = 0 

		# j = counter for subtitle lines 
		for j, row in enumerate(df.itertuples()):
			if row.Style not in self.stylesWithIcons: continue 

			n_overlap = (row.Start < df.iloc[1:j, 1]).sum()
			x, y = coords[n_overlap]

			overlays.append(
				self.fmts['ot'].format(
					src="0:v" if j == 0\
						else f"{i}ov",
					pi=row.StyleInds+1,
					oi=i+1,
					x=x, y=y,
					tA=row.Start, tB=row.End
				)
			)

			i += 1 

		return f"{'; '.join(overlays)}\" -map \"[{i}ov]\""

	def build_filter(
			self,
			df: pd.DataFrame,
			y_icon: float,
			icon_xy: dict[str, tuple[float, float]],
			mode: str) -> str:

		inputs, scales, overlays = [], [], []
		for i, name in enumerate(self.stylesWithIcons):
			inputs.append(
				self.fmts['i'].format(p=self.icon_paths[name])
			)
			scales.append(
				self.fmts['s'].format(
					i=i+1, y=y_icon)
			)

			if not 'speaker' in mode:
				x, y = icon_xy[name]
				overlays.append(
					self.fmts['o'].format(
						src="0:v" if i == 0\
							else f"{i+1}ov",
						i=i+1, 
						x=x, y=y
					)
				)

		if 'speaker' in mode:
			overlays = self.get_speaker_overlays(
				df, list(icon_xy.values()))
		else:
			overlays = f"{'; '.join(overlays)}\" -map \"[{i+1}ov]\""

		filter_ = f"-filter_complex \"{'; '.join(scales)}; {overlays}"
		return f"{' '.join(inputs)} {filter_}"

	def run_overlay(
			self,
			video_path: Path,
			df: pd.DataFrame,
			y_icon: float,
			icon_xy: dict[str, tuple[float, float]],
			mode: str) -> None:

		filter_ = self.build_filter(df, y_icon, icon_xy, mode)

		outpath = video_path.parent /\
			f"{video_path.stem}_overlay{video_path.suffix}"

		if check_overwrite(outpath):
			cmd = f"ffmpeg -i {video_path} {filter_} \"{outpath}\""
			Popen(['powershell.exe', cmd])
			print(cmd)
			run_powershell(cmd)
		else:
			raise FileExistsError(outpath)

	def process_dialog(self, df: pd.DataFrame) -> pd.DataFrame:

		df_out = df.loc[:, ['Start', 'End', 'Style']].copy()

		style_inds: dict[str, ind] = {
			style: i
			for i, style in
			enumerate(self.stylesWithIcons)
		}

		for col in ['Start', 'End']:
			df_out[col] = pd.to_timedelta(df[col]).\
				apply(lambda x: x.total_seconds())

		df_out['StyleInds'] = df.Style.\
			apply(
				lambda s: style_inds[s]
				if s not in self.noIcons else -1 
			)

		return df_out

	def parse_ass_dialog(self, fp: Path) -> pd.DataFrame:

		with open(fp, 'r', encoding='utf-8') as io:
			lines = io.readlines()

		ind = next(i for i, line in enumerate(lines) if "Dialogue:" in line)

		# Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
		lines = list(map(lambda x: x.strip().split(',', 9)[1:], lines[ind:]))

		df = pd.DataFrame.from_records(
			lines,
			columns=['Start', 'End', 'Style', 'Name', 'MarginL',
					 'MarginR', 'MarginV', 'Effect', 'Text'],
		)

		return df

	def overlay(
			self,
			ass_path: Path,
			video_path: Path,
			mode: str = 'left',
			noIcons: list[str] = None) -> None:

		df = self.parse_ass_dialog(ass_path)

		self.stylesWithIcons = self.get_stylesWithIcons(
			df,
			noIcons=noIcons if noIcons else self.noIcons
		)

		# processed dialog dataframe
		df_pro = self.process_dialog(df)

		# icon position information
		posInfo = self.compute_positions(video_path, mode)

		self.run_overlay(
			video_path,
			df_pro,
			**posInfo
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
			marginBottom=200.,
			borderPadding=10.,
			minIconWidth=150.,
			maxIconWidth=300.
		),
		overlay_mode='left',
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
		try:
			vp, _ = Reader.read_dialog(fp)
		except VideoNotFoundError:
			vp = get_filenames(
				f"Find video for {fp.stem}...",
				filetypes=(('Video file', '*.mp4'),),
				init_dir=INIT_DIR,
				multiple=False
			)

		Overlay.overlay(fp, vp, mode=overlay_mode)

if __name__ == '__main__':
	main(overlay_mode='speaker')

"""
TODO
1. add `split` to the output stream for `scale` when using `speaker` mode for overlaying
2. fix the positioning of icons, e.g. proper padding
3. clean up code
"""	