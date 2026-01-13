import numpy as np
from conversionHandling.helpers.sysinfo import SystemInfo

def compute_read_chunks(
        system: SystemInfo,
        target_chunks: tuple[int, int, int], 
        dtype_size: int,
        safety_factor: float,
        multiple: int = 8, 
        max_bytes: int = 1_000_000_000,
        min_chunks_in_memory: int = 8
        ) -> tuple[int, int, int]:

    read_chunks = tuple(tc * multiple for tc in target_chunks)

    # Check total bytes
    chunk_bytes = np.prod(read_chunks) * dtype_size

    max_mem_bytes = system.available_ram_bytes * safety_factor / min_chunks_in_memory

    if chunk_bytes > max_bytes or chunk_bytes > max_mem_bytes:
        scale_factor = min((max_bytes / chunk_bytes) ** (1/3),
                           (max_mem_bytes / chunk_bytes) ** (1/3))
        # Scale each axis
        read_chunks_scaled = tuple(max(1, int(rc * scale_factor)) for rc in read_chunks)

        # Round down to nearest multiple of target_chunks
        read_chunks = tuple((rc // tc) * tc for rc, tc in zip(read_chunks_scaled, target_chunks))

        # Ensure at least target_chunks in each dimension
        read_chunks = tuple(max(tc, rc) for rc, tc in zip(read_chunks, target_chunks))

    return read_chunks