import os
import zarr
from dask.distributed import wait
from numcodecs import Blosc
from datetime import timedelta
from pathlib import Path
import time
import numpy as np
from conversionHandling.helpers.mean_downsampling import mean_downsample_block


def pyramid_write(
        compression_level: int,
        output_path: Path,
        target_chunks: tuple[int, int, int],
        pyramid_levels: int,
        progress_levels: int,
        max_in_flight: int,
        downsample_factor: int,
        progress_callback=None,
        client=None
):

    def build_level(
        client,
        output_path,
        level,
        downsample_factor,
        target_chunks,
        compressor,
        max_in_flight,
        progress_levels,
        progress_callback=None
    ):

        print(f"\n{'='*60}")
        print(f"LEVEL {level}: Block-Mean Downsampling")
        print(f"{'='*60}")

        source_path = os.path.join(output_path, str(level - 1))
        destination_path = os.path.join(output_path, str(level))

        #load previous level as source
        source = zarr.open(source_path, mode="r")
        current_shape = source.shape
        new_shape = tuple(max(1, s // downsample_factor) for s in current_shape)

        print(f"Previous shape: {current_shape}")
        print(f"New shape: {new_shape}")

        # Create destination array
        zarr.open(
            destination_path,
            mode="w",
            shape=new_shape,
            chunks=target_chunks,
            dtype=source.dtype,
            compressor=compressor,
            dimension_separator="/"
        )

        futures = []
        
        chunk_z, chunk_y, chunk_x = target_chunks

        current_total_tasks = (
            (int(np.ceil(new_shape[0] / chunk_z)))*
            (int(np.ceil(new_shape[1] / chunk_y)))*
            (int(np.ceil(new_shape[2] / chunk_x)))
        )
        print(f"Total tasks for current level: {current_total_tasks}")

        in_flight = current_total_tasks // max_in_flight

        print(f"Total recurring in flights: {in_flight}")

        level_start = time.time()
        
        completed = 0

        # Iterate over output blocks
        for z_start in range(0, new_shape[0], chunk_z):
            for y_start in range(0, new_shape[1], chunk_y):
                for x_start in range(0, new_shape[2], chunk_x):

                    #Tuple holding python slice objects (block write coordinates)
                    destination_coords = (
                        slice(z_start, min(z_start + chunk_z, new_shape[0])),
                        slice(y_start, min(y_start + chunk_y, new_shape[1])),
                        slice(x_start, min(x_start + chunk_x, new_shape[2])),
                    )

                    # Mapping block
                    source_start = (
                        z_start * downsample_factor,
                        y_start * downsample_factor,
                        x_start * downsample_factor,
                    )
                    source_end = (
                        min((z_start + chunk_z) * downsample_factor, current_shape[0]),
                        min((y_start + chunk_y) * downsample_factor, current_shape[1]),
                        min((x_start + chunk_x) * downsample_factor, current_shape[2]),
                    )

                    #Tuple holding python slice objects (Block to be read from current array)
                    block_region = (
                        slice(source_start[0], source_end[0]),
                        slice(source_start[1], source_end[1]),
                        slice(source_start[2], source_end[2])
                    )

                    # Submit the block task
                    future = client.submit(
                        mean_downsample_block,
                        source_path,
                        destination_path,
                        block_region,
                        destination_coords,
                        downsample_factor
                    )

                    futures.append(future)

                    if len(futures) >= max_in_flight:
                        completed += 1
                        print(f"completed: {completed}")
                        wait(futures)
                        futures = []

                        elapsed = time.time() - level_start
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (in_flight - completed) / rate if rate > 0 else 0

                        progress_callback(
                            level=level,
                            progress_levels=progress_levels,
                            block_count=completed,
                            total_blocks=in_flight,
                            rate=rate,
                            eta=eta
                        )

        if futures:
            wait(futures)

        print(f"Finished level {level} in {(time.time() - level_start):.1f}s")


    # ------------------------------------------------------------
    # Main Pyramid Builder
    # ------------------------------------------------------------

    print("="*60)
    print("Building OME-Zarr Multi-Resolution Pyramid (Block-Mean)")
    print("="*60)

    compressor = Blosc(
        cname="zstd",
        clevel=compression_level,
        shuffle=Blosc.BITSHUFFLE
    )

    pyramid_start = time.time()
    for level in range(1, pyramid_levels):
        build_level(
            client,
            output_path,
            level,
            downsample_factor,
            target_chunks,
            compressor,
            max_in_flight,
            progress_levels,
            progress_callback
        )

    print("\nTotal pyramid time: "
          f"  {timedelta(seconds=int(time.time() - pyramid_start))}")
