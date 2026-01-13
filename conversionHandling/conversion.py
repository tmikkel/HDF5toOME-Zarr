import h5py
from pathlib import Path
from conversionHandling.hybrid import hybrid_conversion
from conversionHandling.parallel import parallel_conversion
from conversionHandling.sequential import sequential_conversion
from conversionHandling.helpers.sysinfo import detect_system
from conversionHandling.helpers.storage import StorageType

def convert_hdf5_to_omezarr(
    h5_path: Path,
    output_dir: Path,
    target_chunks: tuple[int, int, int],
    mode: str,  # "sequential" "hybrid" "parallel"
    safety_factor: float,
    compression_level: int,
    storage: StorageType
    
):
    system = detect_system()

    base_name = h5_path.stem 
    store_path = output_dir / f"{base_name}.ome.zarr"
    i = 1
    while store_path.exists():
        store_path = output_dir / f"{base_name}_{i}.ome.zarr"
        i += 1
    
    with h5py.File(h5_path, "r") as f:
        dset = f["exchange/data"]
        source_chunks = dset.chunks

    print(f"Mode sanity check: {mode}")


    mode_map = {
        "Sequential": sequential_conversion,
        "Hybrid": hybrid_conversion,
        "Parallel": parallel_conversion
        }

    # fallback for chunked HDF5
    if source_chunks is not None:
        func = parallel_conversion
    else:
        func = mode_map[mode]

    func(
        h5_path, 
        store_path, 
        target_chunks, 
        safety_factor, 
        system, 
        compression_level, 
        storage
        )