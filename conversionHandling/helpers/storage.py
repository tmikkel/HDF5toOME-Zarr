from enum import Enum

class StorageType(Enum):
    HDD = "hdd"
    SATA_SSD = "sata"
    NVME = "nvme"

STORAGE_WORKER_CAP = {
    StorageType.HDD: 2,
    StorageType.SATA_SSD: 4,
    StorageType.NVME: 8,
}