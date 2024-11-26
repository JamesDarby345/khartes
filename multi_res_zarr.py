import zarr
import dask.array as da
import numpy as np
import argparse
from tqdm import tqdm
import time
import shutil
import os

"""
Transform a single resolution Zarr array into a multi-resolution pyramid.

This script converts a single-resolution Zarr array into a multi-resolution pyramid format
following the OME-NGFF specification. It uses 2x downsampling at each level, implemented
with Dask arrays for parallel processing and memory-efficient computation. The pyramid
is created using mean pooling, suitable for most microscopy data.

Key features:
- Parallel processing via Dask
- Memory-efficient streaming computation
- OME-NGFF compliant metadata
- Configurable number of pyramid levels
- Optional in-place transformation
"""

def add_multiscale_resolutions(zarr_path, num_levels=5, in_place=False):
    """
    Create a new multi-resolution Zarr array from a single-resolution input.
    
    Args:
        zarr_path: Path to the input .zarr file/directory
        num_levels: Number of additional resolution levels to add (default 5)
        in_place: If True, overwrites the original file instead of creating a new one
    """
    start_time = time.time()
    if in_place:
        output_path = zarr_path
    else:
        output_path = zarr_path.rstrip('/').replace('.zarr', '_multires.zarr') if zarr_path.endswith('.zarr') else zarr_path.rstrip('/') + '_multires.zarr'
    
    # Remove existing directory if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    # Open the original array read-only
    z = zarr.open(zarr_path, mode='r')
    
    # Handle both single array and multiscale inputs
    if isinstance(z, zarr.hierarchy.Group):
        # Check if it's a multiscale zarr
        if 'multiscales' in z.attrs:
            # Use the highest resolution level (usually '0')
            source_array = z['0']
        else:
            raise ValueError("Input is a group but not in multiscales format", z.info)
    else:
        source_array = z
        
    # Get original array properties
    shape = source_array.shape
    dtype = source_array.dtype
    chunks = source_array.chunks
    
    # Create new group for output
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store)
    
    # Copy original data to resolution 0
    copy_start = time.time()
    print(f"Copying original data to resolution 0")
    data = da.from_zarr(source_array)
    data.map_blocks(lambda x: x).to_zarr(output_path + '/0', compute=True)
    print(f"Copy completed in {time.time() - copy_start:.2f} seconds")
    
    # Create pyramid levels with 2x downsampling each time
    for level in tqdm(range(1, num_levels+1), desc="Creating pyramid levels"):
        level_start = time.time()
        
        # Calculate new shape
        new_shape = tuple(max(s // (2**level), 1) for s in shape)
        
        # Calculate new chunks (optional - could keep same chunk size)
        new_chunks = tuple(min(c, s) for c, s in zip(chunks, new_shape))
        
        # Create downsampled array
        root.create_dataset(str(level), shape=new_shape, chunks=new_chunks, dtype=dtype)
        downsampled = da.coarsen(
            np.mean, 
            da.from_zarr(root[str(level-1)]), 
            {d: 2 for d in range(len(shape))},
            trim_excess=True
        )
        downsampled.to_zarr(root[str(level)], compute=True)
        print(f"Level {level} completed in {time.time() - level_start:.2f} seconds")
    
    # Update multiscales metadata to match OME-NGFF spec
    multiscales = [{
        "version": "0.4",
        "name": "pyramid",
        "datasets": [
            {
                "path": str(i),
                "coordinateTransformations": [{
                    "type": "scale",
                    "scale": [2**i, 2**i, 2**i]
                }],
                "size": [s // (2**i) for s in shape]
            } for i in range(num_levels+1)
        ],
        "type": "gaussian",
        "metadata": {
            "method": "mean",
            "version": "0.4"
        }
    }]
    
    # Add required OME-NGFF metadata
    root.attrs["multiscales"] = multiscales
    root.attrs["omero"] = {
        "channels": [{
            "label": f"channel_{i}",
            "color": "FFFFFF",
            "window": {"start": 0, "end": 255}
        } for i in range(shape[-1] if len(shape) > 2 else 1)]
    }

    # Add dimension names
    axes = list("xyzct"[-len(shape):])  # Assume standard bio-image dimensions
    root.attrs["_ARRAY_DIMENSIONS"] = axes
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add pyramid levels to a single-resolution Zarr array')
    parser.add_argument('--zarr-path', help='Path to the input .zarr file/directory')
    parser.add_argument('--levels', type=int, default=5, help='Number of resolution levels to add (default: 5)')
    parser.add_argument('--in-place', action='store_true', help='Overwrite the original zarr instead of creating a new file')
    
    args = parser.parse_args()
    add_multiscale_resolutions(args.zarr_path, args.levels, args.in_place)
