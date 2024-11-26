import numpy as np
import zarr

# Create random data
shape = (1012, 5012, 5012)
print("adding data")
# data = np.random.randint(0, 255, size=shape, dtype=np.uint8)
data = np.zeros(shape, dtype=np.uint8)

# Save to zarr
print("saving to zarr")
store = zarr.DirectoryStore('/Users/jamesdarby/Desktop/nnUNet_preds/test.zarr')
z = zarr.create(shape=shape, dtype=np.uint8, store=store, overwrite=True)
z[:] = data
