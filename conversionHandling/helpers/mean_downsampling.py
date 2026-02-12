import zarr

def mean_downsample_block(
        source_path,
        destination_path,
        block_region,
        destination_coords,
        downsample_factor
    ):
        """
        Each worker task:
        1. Opens source and destination Zarr arrays.
        2. Reads source block.
        3. Trims it to dimensions divisible by downsample_factor.
        4. Computes block mean over each non-overlapping cube of size downsample_factor.
        5. Writes the downsampled block to the destination array.
        """
        src = zarr.open(source_path, mode="r")
        store = zarr.open(destination_path, mode="r+")

        block = src[block_region]

        d = downsample_factor

        # Trim dimensions to be divisible by downsample factor
        block_z = (block.shape[0] // d) * d
        block_y = (block.shape[1] // d) * d
        block_x = (block.shape[2] // d) * d
        block = block[:block_z, :block_y, :block_x]

        # Reshape to compute block mean
        # Each axis is split into (num_blocks, block_size)
        reshaped = block.reshape(
            block_z // d, d,
            block_y // d, d,
            block_x // d, d
        )

        # Compute mean over the block axes (1,3,5) (downsample the block)
        downsampled = reshaped.mean(axis=(1, 3, 5)).astype(block.dtype)

        # Write to pyramid array
        store[destination_coords] = downsampled