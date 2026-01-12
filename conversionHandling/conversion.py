from pathlib import Path
from conversionHandling.parallel import parallel_conversion
from conversionHandling.sequential import sequential_conversion
from conversionHandling.helpers.sysinfo import detect_system

system = detect_system()

def convert_hdf5_to_omezarr(
    h5_path: Path,
    output_dir: Path,
    target_chunks: tuple[int, int, int],
    mode: str,  # "sequential" or "parallel"
    safety_factor: float,
    compression_level: int,
    
):
    
    base_name = h5_path.stem 
    zarr_path = output_dir / f"{base_name}.ome.zarr"
    i = 1
    while zarr_path.exists():
        zarr_path = output_dir / f"{base_name}_{i}.ome.zarr"
        i += 1
    
    print(f"DEBUG: mode='{mode}'")
    if mode.lower() == "sequential":
        sequential_conversion(h5_path, 
                              zarr_path, 
                              target_chunks,
                              safety_factor,
                              system,
                              compression_level
                              )
    else:
        parallel_conversion(h5_path, 
                            zarr_path, 
                            target_chunks,
                            safety_factor,
                            system,
                            compression_level
                            )