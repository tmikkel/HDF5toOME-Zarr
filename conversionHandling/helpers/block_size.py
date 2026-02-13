from conversionHandling.helpers.sysinfo import SystemInfo

def block_size(
        shape: tuple[int, int, int],
        target_chunks: tuple[int, int, int],
        safety_factor: float,
        dtype_size: int,
        system: SystemInfo,
        memory_limit_bytes: int,
        worker_limit: int,
        parallel: bool

):
    """ 
        Calculates largest block size that:
            1. Maximizes memory usage (fewer HDF5 reads)
            2. Aligns with target chunks
            3. Stays within memory budget
    """
        
    z, y, x = shape
    chunk_z, chunk_y, chunk_x = target_chunks
    
    if parallel == True:
        print("parallel block limit")
        available_bytes = worker_limit * (2/3) - 400_000_000 # Extra safety buffer
    elif parallel == False:
        print("sequential block limit")
        # Available memory given safety factor
        available_bytes = memory_limit_bytes * safety_factor
        
    # Calculate maximum amount of Z-planes that fit in memory
    bytes_per_z_plane = y * x * dtype_size
    max_z_planes = int(available_bytes / bytes_per_z_plane)


    if max_z_planes < chunk_z:
        print(f"\nFull Target Z plane ({target_chunks[0]}) too large for memory")
        print("Reducing Y axis to fit block in memory")
        
        # Calculate max Y that fits with target Z and full X
        bytes_per_y_row = chunk_z * x * dtype_size
        max_y_rows = int(available_bytes / bytes_per_y_row)

        if max_y_rows < chunk_y:
            print(f"\nFull Target Y rows ({target_chunks[1]}) too large for memory")
            print("Reducing X axis to fit block in memory")
            bytes_per_x_column = chunk_z * chunk_y * dtype_size
            max_x_columns = int(available_bytes / bytes_per_x_column)
            optimal_x = (max_x_columns // chunk_x) * chunk_x
            optimal_x = max(chunk_x, optimal_x)  # At least one chunk depth
            if max_x_columns >= x / 2 + chunk_x:
                optimal_x = int(min(optimal_x, ((x / 2) // chunk_x) * chunk_x + chunk_x))   # Cap to a multiple of chunk_y just above half of y
            x = optimal_x

        optimal_y = (max_y_rows // chunk_y) * chunk_y 
        optimal_y = max(chunk_y, optimal_y)  # At least one chunk depth
        if max_y_rows >= y / 2 + chunk_y:
            optimal_y = int(min(optimal_y, ((y / 2) // chunk_y) * chunk_y + chunk_y))   # Cap to a multiple of chunk_y just above half of y
        y = optimal_y

    block_shape = chunk_z, y, x
        
    # Calculate actual memory usage
    actual_gb = chunk_z * y * x * dtype_size / 1e9

    max_mem_gb = memory_limit_bytes / 1e9
        
    print(f"\n{'='*60}")
    print("Optimal Block Size Calculation")
    print(f"{'='*60}")
    print(f"Memory budget: {max_mem_gb:.2f} GB (using {int(safety_factor*100)}%)")
    print(f"Available for block: {available_bytes/1e9:.2f} GB")
    print(f"Actual block size: {actual_gb:.2f} GB")
    print(f"{'='*60}")
    print(f"Read chunks: {block_shape}")
    return block_shape