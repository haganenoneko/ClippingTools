from RemoveSilence.ass_tools import ASSWriter, parse_interval_file, INIT_DIR
from RemoveSilence.common import get_filenames

get_video = lambda: get_filenames("Select video file.", (("Video files", "*.mp4"),("All files", "*.*"),),)
get_audio = lambda: get_filenames("Select audio file.", (("Audio files", "*.m4a"),("All files", "*.*"),))

def main():
    files = get_filenames(
        title="Select file of silence intervals.",
        init_dir=INIT_DIR,
        filetypes=(
            ('CSV files', '*.csv'),
            ('TSV files', '*.tsv'),
            ('Text files', '*.txt'),
            # ('All files', '*.*')
        ),
        multiple=True,
    )

    for file in files:
        secs = parse_interval_file(file)
        vpath = get_video()

        ASSWriter().write(
            file.stem,
            secs,
            outdir=file.parent,
            vpath=vpath,
            apath=vpath
        )


if __name__ == '__main__':
    main()
