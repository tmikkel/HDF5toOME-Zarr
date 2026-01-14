import h5py
import zarr
import numpy as np
import time
from datetime import timedelta
from numcodecs import Blosc
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import dask
import dask.array as da
from pathlib import Path
from conversionHandling.helpers.sysinfo import SystemInfo
from conversionHandling.helpers.pyramid_write import pyramid_write
from conversionHandling.helpers.pyramid_levels import n_pyramid_levels
from conversionHandling.helpers.write_metadata import write_metadata
from conversionHandling.helpers.workers import choose_n_workers
from conversionHandling.helpers.block_size import block_size
from conversionHandling.helpers.storage import StorageType

def hybrid_conversion(
        h5_path: Path,
        output_path: Path,
        target_chunks: tuple[int, int, int],
        safety_factor: float,
        system: SystemInfo,
        compression_level: int,
        storage: StorageType,
        dataset_path = 'exchange/data',
        ):
    print("DEBUG: entering hybrid_conversion")


    with h5py.File(h5_path, "r") as f:
            dataset = f[dataset_path]
            shape = dataset.shape
            dtype = dataset.dtype
            dtype_size = dtype.itemsize
            data_size_mb = dataset.nbytes / (1024**2)
        
            block_shape = block_size(
                shape,
                target_chunks,
                safety_factor,
                dtype_size,
                system,
                mem_divider=4
            )

            block_z, block_y, block_x = block_shape
            z_total, y_total, x_total = shape

            # For hybrid conversion block_z shouldn't be greater than target_z
            block_z = target_chunks[0]

    read_chunks_bytes = np.prod(block_shape) * dtype_size

    n_workers = choose_n_workers(      
        system,
        storage,
        read_chunks_bytes,
        safety_factor
        )
    
    print(f"Number of Workers: {n_workers}")

    memory_limit = (system.available_ram_bytes * safety_factor)/n_workers

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        memory_limit=memory_limit,
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    store = zarr.NestedDirectoryStore(output_path)
    root = zarr.open_group(store, mode="w")
    compressor = Blosc(cname="zstd", clevel=compression_level, shuffle=Blosc.BITSHUFFLE)

    root.create_dataset(
        "0",
        shape=shape,
        chunks=target_chunks,
        dtype=dtype,
        compressor=compressor
    )

    @dask.delayed
    def copy_block(z_start, z_end, y_start, y_end, x_start, x_end):
        with h5py.File(h5_path, "r") as f:
            block = f[dataset_path][z_start:z_end, y_start:y_end, x_start:x_end]
        
        store = zarr.NestedDirectoryStore(output_path)
        root = zarr.open_group(store, mode="a")
        root["0"][z_start:z_end, y_start:y_end, x_start:x_end] = block
        return (z_end - z_start, y_end - y_start, x_end - x_start)

    tasks = []
    for z_start in range(0, z_total, block_z):
        z_end = min(z_start + block_z, shape[0])

        for y_start in range(0, y_total, block_y):
            y_end = min(y_start + block_y, y_total)

            for x_start in range(0, x_total, block_x):
                x_end = min(x_start + block_x, x_total)

                tasks.append(copy_block(z_start, z_end, y_start, y_end, x_start, x_end))

    print(f"\n✓ Submitting {len(tasks)} tasks for parallel execution...")

    start = time.time()
    with ProgressBar():
        dask.compute(*tasks)
    elapsed = time.time() - start

    total_gb = np.prod(shape) * dtype_size / 1e9

    print(f"\n✓ Complete: {elapsed:.1f}s | {total_gb/elapsed:.2f} GB/s")

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