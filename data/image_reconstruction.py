import os
import numpy as np
import nibabel as nib
from PIL import Image

# Set the directory containing the 2D slices
slice_dir = 'path/to/slice/directory'

# Get a list of all slice files in the directory
slice_files = sorted([f for f in os.listdir(slice_dir) if f.endswith('.png')])

# Determine the dimensions of the 3D volume
num_slices = len(slice_files)
first_slice = Image.open(os.path.join(slice_dir, slice_files[0]))
width, height = first_slice.size

# Create a 3D numpy array to store the volume
volume = np.zeros((num_slices, height, width), dtype=np.uint8)

# Load each slice into the 3D volume
for i, slice_file in enumerate(slice_files):
    slice_path = os.path.join(slice_dir, slice_file)
    slice_img = Image.open(slice_path)
    volume[i, :, :] = np.array(slice_img)

# Create a NIfTI image from the 3D volume
img = nib.Nifti1Image(volume, np.eye(4))

# Save the NIfTI image to a file
nib.save(img, 'path/to/output/file.nii.gz')