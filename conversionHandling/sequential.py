import h5py
import zarr
import time
from datetime import timedelta
import numpy as np
from numcodecs import Blosc
from pathlib import Path
from conversionHandling.helpers.sysinfo import SystemInfo
from conversionHandling.helpers.block_size import block_size

def sequential_conversion(
    h5_path: Path,
    output_path: Path,
    target_chunks: tuple[int, int, int],
    safety_factor: float,
    system: SystemInfo,
    compression_level: int,
    memory_limit: int,
    progress_levels: int,
    progress_callback=None,
    dataset_path = 'exchange/data'
):
    print("DEBUG: entering sequential_conversion")
    # Inspect HDF5 file
    with h5py.File(h5_path, 'r') as f:
        dataset = f[dataset_path]
        shape = dataset.shape
        dtype = dataset.dtype
        data_size_mb = dataset.nbytes / (1024**2)
        dtype_size = dtype.itemsize

        block_shape = block_size(
            shape,
            target_chunks,
            safety_factor,
            dtype_size,
            system,
            memory_limit,
            parallel=False
        )

        block_z, block_y, block_x = block_shape
        z_total, y_total, x_total = shape
            
        # Setup output zarr store    
        store = zarr.NestedDirectoryStore(output_path)
        root = zarr.open_group(store=store, mode='w')

        # Compressor for all levels
        compressor = Blosc(cname='zstd', clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
            
        level_0 = root.create_dataset(
            '0',
            shape=shape,
            chunks=target_chunks,
            dtype=dtype,
            compressor=compressor
        )
            
        # Copy in large slabs (efficient for contiguous HDF5)

        level0_start = time.time()
        block_count = 0
            
        # Calculate total blocks
        total_blocks = (
            ((z_total + block_z - 1) // block_z) *
            ((y_total + block_y - 1) // block_y) *
            ((x_total + block_x - 1) // block_x)
        )
            
        print(f"Processing {total_blocks} blocks...")

        # Iterate over blocks
        for z_start in range(0, z_total, block_z):
            z_end = min(z_start + block_z, z_total)
                
            for y_start in range(0, y_total, block_y):
                y_end = min(y_start + block_y, y_total)

                for x_start in range(0, x_total, block_x):
                    x_end = min(x_start + block_x, x_total)
                        
                    block_count += 1
                            
                    # Read block from HDF5
                    block = dataset[z_start:z_end, y_start:y_end, x_start:x_end]
                            
                    # Write to zarr (zarr will internally chunk to target_chunks)
                    level_0[z_start:z_end, y_start:y_end, x_start:x_end] = block
                            
                    del block

                    if block_count % 1 == 0 or block_count == total_blocks:
                        elapsed = time.time() - level0_start
                        rate = block_count / elapsed if elapsed > 0 else 0
                        eta = (total_blocks - block_count) / rate if rate > 0 else 0

                        progress_callback(
                            level=0,
                            progress_levels=progress_levels,
                            block_count=block_count,
                            total_blocks=total_blocks,
                            rate=rate,
                            eta=eta
                        )
                
        elapsed_level0 = time.time() - level0_start
        throughput = (np.prod(shape) * dtype_size / 1e9) / elapsed
            
        print(f"\n✓ Level 0 complete in {elapsed_level0:.1f}s")
        print(f"  {timedelta(seconds=int(elapsed_level0))}")
        print(f"  Throughput: {throughput:.2f} GB/s")