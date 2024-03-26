import subprocess
import os

import argparse

# Add argument for dataset directory with default value
parser = argparse.ArgumentParser(description='Generate segmentations for spinal cord images.')
parser.add_argument('--dataset_dir', type=str, default='results/reconstructed_images/ssim_pixel/T1w_aligned', help='Path to the dataset directory.')
args = parser.parse_args()

def segment_spinal_cord(image_path):
    """
    Generate a segmentation of a spinal cord image using SCT's deep segmentation tool.

    Args:
        image_path (str): Path to the input MRI image.
        segmentation_folder (str, optional): Folder to save the generated segmentation.
    """
    # segmentation_cmd = ['sct_deepseg_sc', '-i', image_path, '-c', 't2', '-ofolder', seg_dir]
    contrast_agnostic_segmentation_cmd = ['sct_deepseg', '-i', image_path, '-task', 'seg_sc_contrast_agnostic']
    subprocess.run(contrast_agnostic_segmentation_cmd)

def generate_segmentations(dataset_dir):
    """
    Generate segmentations for all patients in the dataset.
    """
    patients = [p for p in os.listdir(dataset_dir) if p.startswith('sub-')]

    for patient in patients:
        image_path = os.path.join(dataset_dir, patient)
        segment_spinal_cord(image_path)

if __name__ == '__main__':
    generate_segmentations(args.dataset_dir)