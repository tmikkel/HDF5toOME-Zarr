import dask
from numcodecs import Blosc
import dask.array as da
from dask.diagnostics import ProgressBar
from datetime import timedelta
from pathlib import Path
import time
import numpy as np

def pyramid_write(
        compression_level: int,
        output_path: Path,
        target_chunks: tuple[int, int, int],
        pyramid_levels: int,
        downsample_factor: int
):

    # Configure dask for memory constraints
    dask.config.set({
        "distributed.worker.memory.target": 0.7,
        "distributed.worker.memory.spill": 0.8,
        "distributed.worker.memory.pause": 0.9,
        "distributed.scheduler.worker-ttl": None,
    })

    print()
    print("=" * 60)
    print("Building OME-Zarr Multi-Resolution Pyramid")

        
    # Compressor for all levels
    compressor = Blosc(cname='zstd', clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
        
    # Load input zarr
    source = da.from_zarr(output_path, component='0')
    source_shape = source.shape
    source_chunks = source.chunksize
        
    # ===== PYRAMID LEVELS: Build each level from previous =====
    current_shape = source_shape

    pyramid_start = time.time()
        
    for level in range(1, pyramid_levels):
        print(f"\n{'='*60}")
        print(f"LEVEL {level}: Downsampling")
        print(f"{'='*60}")
            
        # Calculate new shape after downsampling
        new_shape = tuple(max(1, s // downsample_factor) for s in current_shape)
            
        print(f"Previous shape: {current_shape}")
        print(f"New shape: {new_shape}")
            
        # Load previous level
        prev_array = da.from_zarr(output_path, component=str(level - 1))
        
        # Downsample using coarsen (block mean)
        # This is memory efficient and HDD-friendly
        downsampled = da.coarsen(
            np.mean,
            prev_array,
            {0: downsample_factor, 1: downsample_factor, 2: downsample_factor},
            trim_excess=True
        ).astype(prev_array.dtype)
            
        print(f"After coarsen: shape={downsampled.shape}, chunks={downsampled.chunksize}")
            
        # Adjust target chunks if array is smaller
        level_chunks = tuple(min(tc, ns) for tc, ns in zip(target_chunks, new_shape))
            
        # Rechunk to target (only once, at the end)
        downsampled = downsampled.rechunk(level_chunks)
            
        print(f"Rechunked to target chunks: {level_chunks}")
            
        # Estimate memory
        level_size_gb = np.prod(new_shape) * prev_array.dtype.itemsize / 1e9
        print(f"Level size: {level_size_gb:.2f} GB")
            
        # Write to zarr with progress bar
        print(f"Writing level {level}...")
            
        with ProgressBar():
            downsampled.to_zarr(
                output_path,
                component=str(level),
                compressor=compressor,
                dimension_separator='/',  # Nested storage
                overwrite=True
            )
            
        # Update current shape
        current_shape = new_shape

    elapsed_pyramid = time.time() - pyramid_start

    print("\n" + "=" * 60)
    print("Pyramid Complete!")
    print("=" * 60)

    print(f"\nTiming breakdown:")
    print(f"  Pyramid complete in: {elapsed_pyramid:.1f}s")
    print(f"  {timedelta(seconds=int(elapsed_pyramid))}")
