import subprocess
import sys
from pathlib import Path

def open_in_napari(store_path: Path):
    subprocess.Popen([
        sys.executable,
        "-m",
        "napari",
        str(store_path)
    ])