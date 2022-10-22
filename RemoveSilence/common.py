from pathlib import Path 
import tkinter as tk 
from tkinter import filedialog as fd 

HOMEDIR = Path.cwd()

def get_filename(init_dir: Path=None, **kwargs) -> Path:
    root = tk.Tk()

    filename = fd.askopenfilename(
        initialdir=init_dir if init_dir else HOMEDIR,
        **kwargs
    )

    root.destroy()
    return Path(filename)