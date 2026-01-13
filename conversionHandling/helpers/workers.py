from .storage import StorageType, STORAGE_WORKER_CAP
from .sysinfo import SystemInfo

def choose_n_workers(
    system: SystemInfo,
    storage: StorageType,
    read_chunks_bytes: int,
    safety_factor: float
) -> int:
    # Decide worker count based on storage type, CPU, and memory constraints.


    # Storage cap
    storage_cap = STORAGE_WORKER_CAP[storage]

    # CPU cap (prefer physical cores)
    cpu_cap = system.physical_cores

    # Memory cap
    usable_ram = system.available_ram_bytes * safety_factor

    # Conservative per-worker memory footprint
    per_worker_mem = read_chunks_bytes * 2

    mem_cap = max(1, int(usable_ram // per_worker_mem))

    return max(1, min(storage_cap, cpu_cap, mem_cap))