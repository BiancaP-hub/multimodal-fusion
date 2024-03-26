import os
import numpy as np
import nibabel as nib
from PIL import Image

import argparse

# Add arguments for fused_slices_dir and reconstructed_images_dir with default values
parser = argparse.ArgumentParser(description='Convert 2D slice images into 3D NIfTI volumes.')
parser.add_argument('--fused_slices_dir', type=str, default='results/fused_slices', help='Path to the directory containing the 2D slice images.')
parser.add_argument('--reconstructed_images_dir', type=str, default='results/reconstructed_images', help='Path to the directory to save the reconstructed 3D NIfTI volumes.')
args = parser.parse_args()

def load_and_process_slice(slice_path):
    """
    Loads a 2D slice image from the given path, rotates it 90 degrees to the left,
    and flips it vertically.
    """
    slice_img = Image.open(slice_path)
    processed_img = slice_img.rotate(-90).transpose(Image.FLIP_TOP_BOTTOM)
    return np.array(processed_img)

def create_volume_from_slices(slice_files, patient_path):
    """
    Creates a 3D volume from the list of 2D slice files.
    """
    num_slices = len(slice_files)
    first_slice = load_and_process_slice(os.path.join(patient_path, slice_files[0]))
    height, width = first_slice.shape

    volume = np.zeros((num_slices, height, width), dtype=np.uint8)
    for i, slice_file in enumerate(slice_files):
        slice_path = os.path.join(patient_path, slice_file)
        volume[i, :, :] = load_and_process_slice(slice_path)
    
    return volume

def save_nifti(volume, output_path):
    """
    Saves the given 3D volume as a NIfTI image.
    """
    img = nib.Nifti1Image(volume, np.eye(4))  # Use an identity matrix for the affine
    nib.save(img, output_path)

def create_nifti_from_slices(fused_slices_dir, reconstructed_images_dir):
    """
    Convert 2D slice images into a 3D NIfTI volume for each patient and model type.
    """
    for model_type in os.listdir(fused_slices_dir):
        if not os.path.isdir(os.path.join(fused_slices_dir, model_type)):
            continue
        for patient in os.listdir(os.path.join(fused_slices_dir, model_type)):
            print(f'Processing patient {patient} in model type {model_type}')
            patient_path = os.path.join(fused_slices_dir, model_type, patient)
            slice_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.png')])
            if not slice_files:
                print(f"No slice files found for {patient} in {model_type}. Skipping.")
                continue

            volume = create_volume_from_slices(slice_files, patient_path)
            output_path = os.path.join(reconstructed_images_dir, model_type, f'{patient}_T2w_T1w.nii.gz')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_nifti(volume, output_path)
            print(f'NIfTI image saved to {output_path}')

def get_nifti_orientation(nifti_file_path):
    """
    Returns the orientation of a NIfTI image as a string (e.g., 'RAS', 'LPI', etc.).

    Parameters:
    - nifti_file_path: Path to the NIfTI file.
    """
    img = nib.load(nifti_file_path)
    header = img.header

    # Get the affine matrix that maps voxel indices to RAS space
    affine = img.affine

    # NiBabel's `aff2axcodes` function can decipher the orientation from the affine
    orientation = nib.aff2axcodes(affine)

    return ''.join(orientation)

if __name__ == "__main__":
    # Create NIfTI volumes from 2D slices
    create_nifti_from_slices(args.fused_slices_dir, args.reconstructed_images_dir)
