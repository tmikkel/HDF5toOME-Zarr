def n_pyramid_levels(
        data_size_mb: float,
        target_top_level_mb: int,
        downsample_factor: int,
):    
    # Calculate optimal pyramid levels
    levels = 1
    current_size_mb = data_size_mb
    
    while current_size_mb > target_top_level_mb:
        current_size_mb = current_size_mb / (downsample_factor ** 3)
        levels += 1
    
    print(f"Target top level: {target_top_level_mb} MB")
    print(f"Recommended levels: {levels}")
    print(f"Actual top level: {current_size_mb:.1f} MB")

    pyramid_levels = levels

    return pyramid_levels