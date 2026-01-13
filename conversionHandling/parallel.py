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

def parallel_conversion(
    h5_path: Path,
    output_path: Path,
    target_chunks: tuple[int, int, int],
    safety_factor: float,
    system: SystemInfo,
    compression_level: int,
    storage: StorageType,
    dataset_path = 'exchange/data',
):

    # =========================
    # OPEN HDF5
    # =========================
    h5 = h5py.File(h5_path, "r")
    dset = h5[dataset_path]
    read_chunks = dset.chunks
    shape = dset.shape
    data_size_mb = dset.nbytes / (1024**2)

    print(f"Dataset shape: {shape}, dtype={dset.dtype}")

    print(read_chunks)
    read_chunks_bytes = np.prod() * dset.dtype.itemsize

    n_workers = choose_n_workers(      
        system,
        storage,
        read_chunks_bytes,
        safety_factor 
        )
  
    threads_per_worker=1
    memory_limit = (system.available_ram_bytes * safety_factor)/n_workers
    # =========================
    # CLUSTER
    # =========================
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit=memory_limit,
        dashboard_address=":8787",
    )
    client = Client(cluster)
    print(f"✓ Dask dashboard: {client.dashboard_link}")

    # =========================
    # DASK ARRAY
    # =========================
    darr = da.from_array(
        dset,
        chunks=read_chunks,
        lock=True  # h5py is not thread-safe
    )

    # =========================
    # RECHUNK (single stage)
    # =========================
    print("Rechunking...")
    darr = darr.rechunk(target_chunks)

    # =========================
    # WRITE ZARR
    # =========================
    compressor = Blosc(
        cname="zstd",
        clevel=compression_level,
        shuffle=Blosc.BITSHUFFLE
    )

    start = time.time()
    with ProgressBar():
        darr.to_zarr(
            output_path,
            component="0",
            compressor=compressor,
            overwrite=True,
            dimension_separator="/"
        )

    elapsed = time.time() - start
    total_gb = np.prod(dset.shape) * dset.dtype.itemsize / 1e9

    print(f"\n✓ Done in {elapsed:.1f}s")
    print(f"✓ Throughput: {total_gb / elapsed:.2f} GB/s")

    # =========================
    # CLEANUP
    # =========================
    h5.close()
    client.close()
    cluster.close()

    pyramid_levels = n_pyramid_levels(
        data_size_mb,
        target_top_level_mb=10,
        downsample_factor=2
        )

    print("Stage 2: Write Multi-Resolution Pyramid from level 0")

    pyramid_write(
        compression_level,
        output_path,
        target_chunks,
        pyramid_levels,
        downsample_factor=2,
    )

    print("Stage 3: Write OME-Zarr Metadata")

    write_metadata(
            output_path,
            pyramid_levels,
            downsample_factor=2
    )