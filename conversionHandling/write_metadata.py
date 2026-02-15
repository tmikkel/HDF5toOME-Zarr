from pathlib import Path
import numpy as np
import zarr

def write_metadata(
        output_path: Path,
        pyramid_levels: int,
        downsample_factor:int
):
    print(f"{'='*60}")
    print("Adding OME-Zarr Metadata")
    print(f"{'='*60}")

    store = zarr.NestedDirectoryStore(output_path)
    root = zarr.open_group(store, mode="a")  # append mode

    # Build datasets list
    datasets = []
    for level in range(pyramid_levels):
        scale_factor = downsample_factor ** level
        datasets.append({
            'path': str(level),
            'coordinateTransformations': [{
                'type': 'scale',
                'scale': [
                    float(scale_factor),  # z
                    float(scale_factor),  # y
                    float(scale_factor)   # x
                ]
            }]
        })
        
    # Add multiscales metadata
    root.attrs['multiscales'] = [{
        'version': '0.4',
        'name': 'pyramid',
        'axes': [
            {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
            {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
            {'name': 'x', 'type': 'space', 'unit': 'micrometer'}
        ],
        'datasets': datasets,
        'type': 'mean',  # Downsampling method
        'metadata': {
            'description': 'Multi-resolution pyramid',
            'method': 'block mean downsampling'
        }
    }]
    print("\nOME-Zarr Summary:")
    print("-" * 60)
        
    for level in range(pyramid_levels):
        arr = zarr.open(store, mode='r')[str(level)]
        size_gb = np.prod(arr.shape) * arr.dtype.itemsize / 1e9
        print(f"  Level {level}: shape={arr.shape}, chunks={arr.chunks}, size={size_gb:.2f} GB")