import h5py
import zarr
import numpy as np
import time
from datetime import timedelta
from numcodecs import Blosc
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import dask.array as da
from pathlib import Path
from conversionHandling.helpers.sysinfo import SystemInfo
from conversionHandling.helpers.pyramid_write import pyramid_write
from conversionHandling.helpers.pyramid_levels import n_pyramid_levels
from conversionHandling.helpers.write_metadata import write_metadata
from conversionHandling.helpers.workers import choose_n_workers
from conversionHandling.helpers.storage import StorageType

def hybrid_conversion(
        h5_path, 
        store_path, 
        target_chunks,
        safety_factor,
        system,
        compression_level,
        storage
        ):
    print("DEBUG: entering hybrid_conversion")
    print("hello")