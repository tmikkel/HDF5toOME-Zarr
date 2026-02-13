import h5py
import zarr
import numpy as np
import time
from numcodecs import Blosc
from dask.distributed import as_completed
import dask
from pathlib import Path
from conversionHandling.helpers.sysinfo import SystemInfo
from conversionHandling.helpers.block_size import block_size

def parallel_conversion(
        h5_path: Path,
        output_path: Path,
        target_chunks: tuple[int, int, int],
        safety_factor: float,
        system: SystemInfo,
        memory_limit_bytes: int,
        compression_level: int,
        worker_limit: int,
        progress_levels: int,
        progress_callback=None,
        client=None,
        dataset_path = 'exchange/data',
        ):
    print("DEBUG: entering parallel_conversion")


    with h5py.File(h5_path, "r") as f:
            dataset = f[dataset_path]
            shape = dataset.shape
            dtype = dataset.dtype
            dtype_size = dtype.itemsize
        
            block_shape = block_size(
                shape,
                target_chunks,
                safety_factor,
                dtype_size,
                system,
                memory_limit_bytes,
                worker_limit,
                parallel=True
            )

            block_z, block_y, block_x = block_shape
            z_total, y_total, x_total = shape

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

    total_tasks = len(tasks)
    print(f"\n✓ Submitting {total_tasks} tasks for parallel execution...")

    start = time.time()

    # Submit all tasks and get futures
    futures = client.compute(tasks)

    # Track progress
    completed = 0
    
    for future in as_completed(futures):
        completed += 1
        elapsed = time.time() - start
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total_tasks - completed) / rate if rate > 0 else 0
        
        progress_callback(
            level=0,
            progress_levels=progress_levels,
            block_count=completed,
            total_blocks=total_tasks,
            rate=rate,
            eta=eta
        )

    elapsed = time.time() - start

    total_gb = np.prod(shape) * dtype_size / 1e9

    print(f"\n✓ Complete: {elapsed:.1f}s | {total_gb/elapsed:.2f} GB/s")