import tkinter as tk
from pathlib import Path
from subprocess import Popen, PIPE
from tkinter import filedialog as fd

from typing import Union

HOMEDIR = Path.cwd()

# ---------------------------------------------------------------------------- #
#                          Filesystem and OS commands                          #
# ---------------------------------------------------------------------------- #


def check_overwrite(fp: Path) -> bool:
    if not fp.is_file():
        return True
    ow = input(f"\"{fp}\" already exists. Overwrite? [y/n]").lower()
    if ow != 'y':
        return False
    else:
        return True


def get_filenames(
        title: str,
        filetypes: list[tuple[str, str]],
        init_dir: Path = None,
        multiple=False,
        **kwargs) -> Union[list[Path], Path]:

    root = tk.Tk()
    filenames = fd.askopenfilenames(
        title=title,
        multiple=multiple,
        initialdir=init_dir,
        filetypes=filetypes,
        **kwargs
    )
    root.destroy()

    if multiple:
        return list(map(lambda x: Path(x), filenames))
    else:
        return Path(filenames[0])

def get_save_filename(
    title: str,
    init_dir: Path=None,
    defaultextension: str=".ass",
    **kwargs
) -> Path:

    root = tk.Tk()

    filename = fd.asksaveasfilename(
        title=title, initialdir=init_dir, defaultextension=defaultextension,
        **kwargs)
    
    root.destroy()

    if filename is None: return 
    return Path(filename)


def run_powershell(cmd: str, stdout=PIPE, stderr=PIPE) -> Popen:
    return Popen(['powershell.exe', cmd], stdout=PIPE, stderr=PIPE)


def extract_ps_output(cmd: str) -> str:
    """Extract the last line in `stdout` for a powershell command"""
    return run_powershell(cmd).\
        communicate()[0].\
        decode('utf-8').\
        strip()

# ---------------------------------------------------------------------------- #
#                               ffprobe commands                               #
# ---------------------------------------------------------------------------- #


def get_video_format(fp: Path) -> str:
    """Get format of a video using powershell"""
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 \"{fp}\""
    fmt = extract_ps_output(cmd)
    return fmt


def get_video_codecs(fp: Path) -> str:
    """Get a list of codecs for a video"""
    cmd = f"ffprobe -v error -hide_banner -of default=noprint_wrappers=0 -print_format flat  -select_streams v:0 -show_entries stream=codec_name,codec_long_name,profile,codec_tag_string \"{fp}\""
    codecs = extract_ps_output(cmd)
    print(codecs)
    return codecs


def get_video_duration(fp: Path) -> float:
    """Get duration of a video in seconds using powershell"""
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{fp}\""
    dur = extract_ps_output(cmd)
    return float(dur)

def get_video_resolution(fp: Path, as_ints=False) -> str:
    """Get resolution of a video `(width x height)`"""
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 \"{fp}\""
    res = extract_ps_output(cmd)
    if as_ints:
        return [int(x) for x in res.split('x')]
    else:
        return res 

# ---------------------------------------------------------------------------- #
#                            Simple ffmpeg commands                            #
# ---------------------------------------------------------------------------- #


def cut_video(
    starts: list, ends: list, fp: Path,
    vcodec='copy', acodec='copy',
    ignore_errors=False
) -> None:
    """Create a list of clips from `starts` and `ends`"""
    template = "ffmpeg -ss {T1} -to {T2} -i \"{I}\" -c:v {V} -c:a {A} \"{N}\""
    fname = fp.parent / '_'.join([fp.stem, "{i}", fp.suffix])

    for i, (t1, t2) in enumerate(zip(starts, ends)):
        outname = fname.format(i=i)
        cmd = template.format(
            T1=t1, T2=t2, I=fp,
            V=vcodec, A=acodec,
            N=outname
        )

        try:
            run_powershell(cmd)
        except Exception as e:
            if ignore_errors:
                print(e)
                continue
            else:
                raise e
