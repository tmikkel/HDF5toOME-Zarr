    
def block_size(
        shape: tuple[int, int, int],
        target_chunks: tuple[int, int, int],
        safety_factor: float,
        dtype_size: int,
        max_mem_gb: float

):
    """ Calculate optimal block size that:
        1. Maximizes memory usage (fewer HDF5 reads)
        2. Aligns with target chunks (efficient zarr writes)
        3. Stays within memory budget
    """
        
    z, y, x = shape
    block_z, block_y, block_x = target_chunks
        
    # Available memory given safety factor
    available_bytes = max_mem_gb * 1e9 * safety_factor
        
    # Calculate maximum amount of Z-planes that fit in memory
    bytes_per_z_plane = y * x * dtype_size
    max_z_planes = int(available_bytes / bytes_per_z_plane)


    if max_z_planes >= block_z:
        # Align to target_z for efficient zarr chunking
        # Use largest multiple of target_z that fits
        optimal_z = (max_z_planes // block_z) * block_z
        optimal_z = max(block_z, optimal_z)  # At least one chunk depth
        optimal_z = min(optimal_z, z)   # Don't exceed dataset
        block_z = optimal_z
    else:
        print(f"\nFull Target Z plane ({target_chunks[0]}) too large for memory")
        print("Reducing Y axis to fit block in memory")
        
        # Calculate max Y that fits with target Z and full X
        bytes_per_y_row = block_z * x * dtype_size
        max_y_rows = int(available_bytes / bytes_per_y_row)
        optimal_y = (max_y_rows // block_y) * block_y 
        optimal_y = max(block_y, optimal_y)  # At least one chunk depth
        if max_y_rows >= y/2+block_y:
            optimal_y = int(min(optimal_y, ((y/2)//block_y)*block_y+block_y))   # Don't exceed half of y + target_y
        y = optimal_y

    block_shape = block_z, y, x
        
    # Calculate actual memory usage
    actual_gb = block_z * y * x * dtype_size / 1e9
        
    print(f"\n{'='*60}")
    print("Optimal Block Size Calculation")
    print(f"{'='*60}")
    print(f"Memory budget: {max_mem_gb:.2f} GB (using {int(safety_factor*100)}%)")
    print(f"Available for block: {available_bytes/1e9:.2f} GB")
    print(f"Actual block size: {actual_gb:.2f} GB")
    print(f"{'='*60}")
    print(block_shape)
    return block_shape