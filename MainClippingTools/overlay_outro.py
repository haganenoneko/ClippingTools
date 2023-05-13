from pathlib import Path
from subprocess import Popen
from common import get_filenames, HOMEDIR, get_video_duration, check_overwrite


def get_files() -> list[Path]:
    """Open file dialogs to select paths of video and outro to concatenate

    Returns:
        list[Path]: video and outro paths
    """
    kwargs = dict(
        filetypes=(("MP4 files", "*.mp4"),),
        init_dir=HOMEDIR,
        multiple=False
    )

    video = get_filenames("Select video", **kwargs)
    outro = get_filenames("Select outro", **kwargs)
    return video, outro


def get_overlay_cmd(out_path: Path, video_path: Path, outro_path: Path, dur=1., vfade="fade", afade="exp") -> str:
    """Generate command to overlay full length of an outro at the end of a video

    Args:
        out_path (Path): output path
        video_path (Path): input video path
        outro_path (Path): outro video path
        dur (_type_, optional): duration of transition between input video and outro, in seconds. Defaults to 1..
        vfade (str, optional): video transition type. Defaults to "fade".
        afade (str, optional): audio transition type. Defaults to "exp".

    Raises:
        FileExistsError: whether output file already exists

    Returns:
        str: output command
    """    
    if not check_overwrite(out_path):
        raise FileExistsError(f"File already exists at {out_path}")

    total_dur = get_video_duration(video_path)

    cmd = f"""ffmpeg `
    -i \"{video_path}\" -i \"{outro_path}\" `
    -filter_complex \" `
    [1:v] scale=1280:720 [scaled]; `
    [0:v][scaled] xfade=transition={vfade}:duration={dur}:offset={total_dur-dur} [outv]; `
    [0:a][1:a] acrossfade=d={dur}:c1={afade}:c2={afade} [outa]\" `
    -map:v \"[outv]\" -map:a \"[outa]\" `
    -c:v libx264 -c:a aac -shortest `
    {out_path}
    """

    return cmd


def main(overlay_duration=1., vfade="fade", afade="exp") -> None:
    video_path, outro_path = get_files()
    out_path = video_path.parent / \
        f"{video_path.stem}__outro.{video_path.suffix}"
    cmd = get_overlay_cmd(
        out_path, video_path, outro_path,
        dur=overlay_duration, vfade=vfade, afade=afade
    )

    Popen(['powershell.exe', cmd])


if __name__ == '__main__':
    main()
