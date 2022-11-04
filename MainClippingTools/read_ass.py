# from RemoveSilence.ass_tools import ASSReader, INIT_DIR
# from RemoveSilence.common import get_filenames, check_overwrite
from ass_tools import ASSReader, INIT_DIR
from common import get_filenames, check_overwrite
import pandas as pd 

def main(outdir: str=None, to_csv=True, **interval_kw):
    """Extract and process intervals from an ASS file

    Args:
        outdir (str, optional): directory to save CSV file containing intervals. Defaults to None.
        to_csv (bool, optional): whether to save intervals to a CSV file. Defaults to True.

    Returns:
        dict[str, pd.DataFrame]: dictionary of `ASS filename` and `pd.DataFrame`s of corresponding processed intervals as keys and values, respectively
    """
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

        if to_csv:
            outdir = file.parent if outdir is None else outdir 
            outname = outdir / f"{file.stem}_intervals.csv"
            df_dict[file.stem] =\
                reader.intervals2csv(outname, intervals)

    return df_dict 

if __name__ == '__main__':
    main()
