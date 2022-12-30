from pathlib import Path 
from typing import Union 
from subprocess import Popen
from secrets import choice 
from common import get_filenames, HOMEDIR, get_video_duration, check_overwrite

def get_files() -> list[Path]:
    files = get_filenames(
        "Select two files to crossfade and concatenate",
        filetypes=(("MP4 files", "*.mp4"),),
        init_dir=HOMEDIR,
        multiple=True
    )
    return files 

def build_input_string(files: list[str]) -> str:
    ins = ' '.join(f"-i \"{f}\"" for f in files)
    return f"ffmpeg {ins}"

def get_fadetypes(fadetypes: Union[str, list[str]], num: int) -> list[str]:
    
    if isinstance(fadetypes, str):
        return [fadetypes] * num 
    
    if len(fadetypes) < num:
        fadetypes.extend(
            [fadetypes[-1]] * (num - len(fadetypes))
        )

    return fadetypes

def build_filter(
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

def sort_files(files: list[Path]) -> list[Path]:
    fileOrder = list(range(len(files)))

    for i, f in enumerate(files):
        msg = f"""./{f.stem}{f.suffix}
        Current order: {i}
        Current full order: {fileOrder}
        Input new order: (0-{len(fileOrder)-1}), or 'n' to skip.
        """
        
        resp = input(msg)
        if resp == 'n':
            continue
        
        try:
            fileOrder[i] = int(resp.strip())
        except TypeError:
            raise TypeError()
    
    order = sorted(zip(fileOrder, files), key=lambda p: p[0])
    files = [f for _, f in order]
    return files 

def random_vfade(n: int, ignore=None) -> list[str]:
    """Choose `n` random video transitions.

    Args:
        n (int): number of transitions to choose
        ignore (_type_, optional): names of transitions to not use. Defaults to None.

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

def main(manual_order=True, **kwargs):
    files = get_files()
    
    if manual_order: 
        files = sort_files(files)
    
    print(
        'Files will be crossfaded in the following order:', 
        [f.stem for f in files], 
        sep='\n'
    )

    crossfade(files, **kwargs)

if __name__ == '__main__':
    main(
        manual_order=True, 
        xdur=0.8, 
        vfadetypes=random_vfade(17),
        afadetypes="dese"
    )