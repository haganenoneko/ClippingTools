from pathlib import Path
from subprocess import Popen
import tkinter as tk
from tkinter import filedialog as fd
import re
from time import sleep

from common import get_filename, HOMEDIR
from create_ass import create_ass_file


SILENCE_PAT = re.compile(
    r"^(?:\[silencedetect\s@\s[\w]*\]\ssilence_(?:start|end)\:\s)([\d\.]+)"
)


def run_silence_remover(filename: Path, noise_level=0.015, noise_duration=1) -> Path:
    outname = f"{filename.stem}_silencedetect.txt"
    cmd = f"""ffmpeg -i "{str(filename)}" -hide_banner -af silencedetect=n={noise_level}:d={noise_duration} -f null - 2> {outname}"""
    Popen(['powershell.exe', cmd])
    return Path(f"./{outname}")


def process_silence_txt(outname: str) -> str:
    with open(f"{outname}", "r", encoding='utf-16') as file:
        lines = [
            line for line in file.readlines()
            if line[:14] == r"[silencedetect"
        ]
    return lines


def extract_silence_times(lines: list[str]) -> list[float]:
    times: list[float] = [
        float(SILENCE_PAT.match(line).group(1).strip())
        for line in lines
    ]
    times.sort()
    return times

### THIS WILL RAISE AN ERROR 
def detect_silence(
        init_dir: Path,
        noise_level='-30dB', noise_duration=1,
        ass_name: str = None, create_ass=False) -> tuple[Path, list[float]]:

    raise NotImplementedError(f"Silence detection with ffmpeg is unreliable. Please instead provide silence intervals from a file.")

    filename = get_filename(
        init_dir=init_dir,
        title="Open input video file.",
        filetypes=(
            ('Video files', '*.mp4'),
            ('All files', '*.*')
        )
    )

    print("Detecting silence with ffmpeg...")
    outname = run_silence_remover(
        filename, noise_level=noise_level, noise_duration=noise_duration)
    
    if not outname.is_file():
        sleep(10)
    if not outname.is_file():
        raise FileNotFoundError(outname)

    print("Reading ffmpeg output...")
    lines = process_silence_txt(outname)

    print("Extracting silence intervals...")
    times = extract_silence_times(lines)

    if create_ass:
        print("Creating .ass file..")
        create_ass_file(
            filename.stem if not ass_name else ass_name,
            times, outdir=HOMEDIR, video_file=filename
        )

    return filename, times


def create_splice_pair(ind: int, start: float, end: float) -> str:
    return f"""[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[{ind}v];
    [0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[{ind}a];"""


def run_splice_and_join(filename: Path, times: list[float]):
    n = int(len(times)/2)
    head = f"""ffmpeg -i "{filename}" -filter_complex \""""
    tail_a = ''.join([f"[{i}v][{i}a]" for i in range(n)])
    tail_b = f"""concat=n={n}:v=1:a=1[outv][outa]" -map "[outv]" -map "[outa]" """

    splice_pairs: list[str] = []
    for i in range(0, len(times), 2):
        a, b = times[i:i+2]
        splice_pairs.append(
            create_splice_pair(int(i/2), a, b)
        )

    cmd = head + ''.join(splice_pairs) + tail_a + tail_b +\
        f"{filename.stem}_OUT.mp4"

    print(cmd)
    Popen(cmd)

def main(
    video_dir=r"C:/Users/delbe/Videos/subtitles/full_raw/",
    noise_level="-45dB", noise_duration=1, remove=False):
    filename, times = detect_silence(
        init_dir=video_dir,
        noise_level=noise_level, 
        noise_duration=noise_duration, 
        ass_name=None, create_ass=True
    )

    if remove:
        run_splice_and_join(filename, times)

if __name__ == '__main__':
    main(noise_level="-40dB", noise_duration=1.5)