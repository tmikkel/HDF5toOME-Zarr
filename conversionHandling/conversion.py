from pathlib import Path
from parallel import parallel_conversion
from sequential import sequential_conversion
from parallel import parallel_conversion
from helpers.system import detect_system

system = detect_system()

def convert_hdf5_to_omezarr(
    h5_path: Path,
    output_dir: Path,
    target_chunks: tuple[int, int, int],
    safety_factor: float,
    compression_level: int,
    mode: str,  # "sequential" or "parallel"
):
    base_name = h5_path.stem 
    zarr_path = output_dir / f"{base_name}.ome.zarr"
    i = 1
    while zarr_path.exists():
        zarr_path = output_dir / f"{base_name}_{i}.ome.zarr"
        i += 1

    if mode == "sequential":
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