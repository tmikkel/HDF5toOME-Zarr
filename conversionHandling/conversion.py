import h5py
from pathlib import Path
from dask.distributed import Client, LocalCluster
from conversionHandling.parallel import parallel_conversion
from conversionHandling.sequential import sequential_conversion
from conversionHandling.helpers.sysinfo import detect_system
from conversionHandling.helpers.storage import StorageType, STORAGE_WORKER_CAP
from conversionHandling.pyramid_write import pyramid_write
from conversionHandling.helpers.pyramid_levels import n_pyramid_levels
from conversionHandling.write_metadata import write_metadata

def convert_hdf5_to_omezarr(
    h5_path: Path,
    output_dir: Path,
    target_chunks: tuple[int, int, int],
    mode: str,  # "sequential" "parallel"
    safety_factor: float,
    compression_level: int,
    storage: StorageType,
    progress_callback,
    downsample_factor
):
    system = detect_system()

    base_name = h5_path.stem 
    store_path = output_dir / f"{base_name}.ome.zarr"
    i = 1
    while store_path.exists():
        store_path = output_dir / f"{base_name}_{i}.ome.zarr"
        i += 1
    
    dataset_path = "exchange/data"

    with h5py.File(h5_path, "r") as f:
        if dataset_path not in f:
            print(f"  ERROR: Dataset '{dataset_path}' not found")
            print(f"  Available paths: {list(f.keys())}")

        data_size_mb = f[dataset_path].nbytes / (1024**2)
    
    # Dynamically calculate number of pyramid levels
    pyramid_levels = n_pyramid_levels(
        data_size_mb,
        downsample_factor,
        target_top_level_mb=100,
        
    )
    progress_levels = pyramid_levels - 1

    # Worker cap, cpu or user defined
    storage_cap = STORAGE_WORKER_CAP[storage]
    cpu_cap = system.physical_cores
    min_mem_per_worker = 1_000_000_000  # 1GB
    available_bytes = system.available_ram_bytes * safety_factor

    # Maximum workers allowed by RAM constraint
    max_workers_by_ram = int(available_bytes // min_mem_per_worker)

    # Final worker count
    n_workers = max(1, min(storage_cap, cpu_cap, max_workers_by_ram)
    )

    memory_limit = available_bytes / n_workers
    print(f"System available mem: {system.available_ram_gb * safety_factor}")
    print(f"Mem per worker: {(system.available_ram_gb * safety_factor)/n_workers}")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        memory_limit=memory_limit,
    )
    client = Client(cluster)

    print(f"Dask dashboard: {client.dashboard_link}")

    print(f"Mode sanity check: {mode}")

    mode_map = {
        "Sequential": sequential_conversion,
        "Parallel": parallel_conversion
        }

    func = mode_map[mode]

    print("Stage 1: HDF5 to level 0 OME-Zarr")

    func(
        h5_path, 
        store_path, 
        target_chunks, 
        safety_factor, 
        system, 
        compression_level,
        memory_limit,
        progress_levels,
        progress_callback,
        client=client
        )

    print("Stage 2: Write Multi-Resolution Pyramid from level 0")

    # Cap number of task at a time per worker
    max_in_flight = n_workers * 16

    pyramid_write(
        compression_level,
        store_path,
        target_chunks,
        pyramid_levels,
        progress_levels,
        max_in_flight,
        downsample_factor,
        progress_callback,
        client=client
    )

    print("Stage 3: Write OME-Zarr Metadata")

    write_metadata(
        store_path,
        pyramid_levels,
        downsample_factor=2
    )

    client.close()
    cluster.close()
    
    return store_path