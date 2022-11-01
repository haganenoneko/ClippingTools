# from RemoveSilence.ass_tools import ASSReader, INIT_DIR
# from RemoveSilence.common import get_filenames, check_overwrite
from ass_tools import ASSReader, INIT_DIR
from common import get_filenames, check_overwrite
import pandas as pd 

def main(to_csv=True, **interval_kw):
    files = get_filenames(
        title="Select ASS file.",
        init_dir=INIT_DIR,
        filetypes=(
            ('ASS files', '*.ass'),
        ),
        multiple=True,
    )

    reader = ASSReader()
    df_dict: dict[str, pd.DataFrame] = {} 
    for file in files:
        vp, dialog = reader.read_dialog(file)
        intervals = reader.read_intervals(dialog, **interval_kw)

        outname = file.parent / f"{file.stem}_intervals.csv"
        df_dict[file.stem] = reader.intervals2csv(outname, intervals)

    return df_dict 

if __name__ == '__main__':
    main()

main()