import os
import sys
import numpy as np
import nibabel as nib
import argparse
from evaluation.Dice import dice_coefficient

def load_nifti(file_path):
    """
    Load a NIFTI file as a NumPy array.

    Args:
        file_path (str): Path to the NIFTI file.

    Returns:
        np.ndarray: The image data.
    """
    return nib.load(file_path).get_fdata()

def evaluate_dice(segmentation_dir, mask_dir):
    """
    Evaluates the overall Dice coefficient for pairs of segmentation images and masks.

    Args:
        segmentation_dir (str): Directory containing segmentation images.
        mask_dir (str): Directory containing mask images.
    """
    image_files = sorted([f for f in os.listdir(segmentation_dir) if f.endswith('_seg.nii.gz')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('_seg-manual.nii.gz')])

    print(f'Found {len(image_files)} segmentation images and {len(mask_files)} mask images.')

    if len(image_files) != len(mask_files):
        print("Error: The number of segmentation images and mask images does not match.")
        sys.exit(1)

    dice_scores = []

    for image_file, mask_file in zip(image_files, mask_files):
        segmentation_path = os.path.join(segmentation_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)

        segmentation_img = load_nifti(segmentation_path)
        mask_img = load_nifti(mask_path)

        if mask_img.shape != segmentation_img.shape:
            mask_img = np.transpose(mask_img, (2, 1, 0))

        dice_score = dice_coefficient(segmentation_img, mask_img)
        dice_scores.append(dice_score)

    # Print all Dice scores
    for i, dice_score in enumerate(dice_scores):
        print(f'Dice coefficient for image {image_files[i]}: {dice_score:.4f}')
        
    overall_dice = np.mean(dice_scores)
    print(f'Overall Dice coefficient: {overall_dice:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate segmentation quality using Dice coefficient.')
    parser.add_argument('--seg_dir', type=str, required=True, help='Directory containing segmentation images.')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing mask images.')

    args = parser.parse_args()

    evaluate_dice(args.seg_dir, args.mask_dir)
