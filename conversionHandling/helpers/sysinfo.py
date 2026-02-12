from dataclasses import dataclass
import os
import psutil

@dataclass(frozen=True)
class SystemInfo:
    logical_cores: int
    physical_cores: int
    total_ram_bytes: int
    available_ram_bytes: int

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / 1e9

    @property
    def available_ram_gb(self) -> float:
        return self.available_ram_bytes / 1e9

def detect_system() -> SystemInfo:
    vm = psutil.virtual_memory()

    return SystemInfo(
        logical_cores=os.cpu_count() or 1,
        physical_cores=psutil.cpu_count(logical=False) or 1,
        total_ram_bytes=vm.total,
        available_ram_bytes=vm.available
    )