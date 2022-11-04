from ass_tools import ASSWriter, parse_interval_file, clean_intervals
from common import get_filenames
from pathlib import Path 

SUBS_DIR = Path(r"C:/Users/delbe/Videos/subtitles/full_raw/removesilence_timecodes")

get_video = lambda: get_filenames("Select video file.", (("Video files", "*.mp4"),("All files", "*.*"),),)
get_audio = lambda: get_filenames("Select audio file.", (("Audio files", "*.m4a"),("All files", "*.*"),))

def main(outdir=None, init_dir=SUBS_DIR):
    files = get_filenames(
        title="Select file of silence intervals.",
        init_dir=init_dir,
        filetypes=(
            ('CSV files', '*.csv'),
            ('TSV files', '*.tsv'),
            ('Text files', '*.txt'),
            # ('All files', '*.*')
        ),
        multiple=True,
    )
    if len(files) < 1:
        raise ValueError(f"No files were selected.")

    for file in files:
        secs = parse_interval_file(file, as_tuples=True)
        # secs = clean_intervals(secs, 0.5, 0.2)
        vpath = get_video()

        ASSWriter().write(
            file.stem,
            secs,
            vpath=vpath,
            apath=vpath,
            outdir=file.parent if outdir is None else outdir,
        )


if __name__ == '__main__':
    main()
