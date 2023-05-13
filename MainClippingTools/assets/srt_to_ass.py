import pysrt
from pathlib import Path 

ASS_HEADER_PATH = "MainClippingTools/assets/ASS_header.txt"

def read_srt(fp: Path, encoding='utf-8') -> list[str]:
    return pysrt.open(fp, encoding=encoding)
    
def get_ass_header(header_path=ASS_HEADER_PATH) -> Path:
    with open(header_path, 'r', encoding='utf-8') as file:
        header = file.read()
    return header 

LINE_FMT = "Dialogue: {Layer},{Start},{End},{Style},{Name},{MarginL},{MarginR},{MarginV},{Effect},{Text}"

def add_dialogue(
        start: str, 
        end: str,
        text: str,
        layer=0,
        style='Default',
        name='',
        marginL=0,
        marginR=0,
        marginV=0,
        effect='',
) -> str:
    return LINE_FMT.format(
        Layer=layer,
        Start=start,
        End=end,
        Style=style,
        Name=name,
        MarginL=marginL,
        MarginR=marginR,
        MarginV=marginV,
        Effect=effect,
        Text=text
    )

def fmt_times(sub: pysrt.SubRipItem) -> list[str]:
    return [str(t).replace(',', '.') for t in [sub.start, sub.end]]

def convert(srt_path: Path, empty_text=False, video_path=None) -> None:
    subs = read_srt(srt_path)
    header = get_ass_header()

    if video_path:
        header.replace("Video File: ", f"Video File: {video_path}")
        header.replace("Audio File: ", f"Audio File: {video_path}")

    lines = [header] 

    for sub in subs:
        start, end = fmt_times(sub)

        if empty_text:
            lines.append(add_dialogue(start=start, end=end, text=''))
        else:
            lines.append(add_dialogue(start=start, end=end, text=sub.text))

    outp = srt_path.parent / f"{srt_path.stem}.ass"
    with open(outp, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))
        file.flush()
    
    print(f"ASS file: {outp}")
    return

if __name__ == '__main__':
    fp = Path("C:/Users/delbe/Videos/subtitles/full_raw/ichinoseuruha_king_silenceremove.srt")
    convert(fp, empty_text=True)