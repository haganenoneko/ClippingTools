from pathlib import Path 
from subprocess import Popen, PIPE
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

def build_filter(
        files: list[str],
        xdur: float = 1.0,
        fadetype: str = "fade",
        afade_types: tuple[str, str] = ('exp', 'exp'),) -> str:

    # curve types for audio cross fade
    # examples: http://underpop.online.fr/f/ffmpeg/help/afade.htm.gz
    # common types: tri (linear, default), exp, [q/h/e]sin, cub, squ, cbr, log
    ac1, ac2 = afade_types
    
    vfade = "[{i}:v]" +\
        f"xfade=transition={fadetype}" +\
        ":duration={d}:offset={off}" +\
        "[xv{i}]"

    afade = "[{i}:a]acrossfade=d={d}" +\
        f":c1={ac1}:c2={ac2}" +\
        "[xa{i}]"

    prev_offset = 0. 
    vfades = [] 
    afades = [] 
    for i, f in enumerate(files[:-1]):
        dur = get_video_duration(f)
        off = dur + prev_offset - xdur 
        prev_offset = off 
        
        if i == 0:
            v_in, a_in = "[0:v]", "[0:a]"
        else:
            v_in, a_in = f"[xv{i}]", f"[xa{i}]"

        vfades.append(v_in + vfade.format(d=xdur, off=off, i=i+1))
        afades.append(a_in + afade.format(d=xdur, i=i+1))

    params = f"{'; '.join(vfades)}; {'; '.join(afades)}"
    lastMap = f"-map \"[xv{i+1}]\" -map \"[xa{i+1}]\""
    return f"-filter_complex \"{params}\" {lastMap}"

def crossfade(
    files: list[str],
    xdur: float = 1.0,
    fadetype: str = "fade",
    afade_types: tuple[str, str] = ('exp', 'exp'),) -> str:

    ins = build_input_string(files)
    filter_ = build_filter(
        files, 
        xdur=xdur, 
        fadetype=fadetype, 
        afade_types=afade_types
    )

    outpath = files[0].stem
    if "_" in outpath: 
        outpath = '_'.join(outpath.split("_")[:-1])
    outpath = files[0].parent / f"{outpath}_crossfade{files[0].suffix}"    

    if check_overwrite(outpath):
        cmd = f"{ins} {filter_} {outpath}"
    
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
    main(manual_order=True)