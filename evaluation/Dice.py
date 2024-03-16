import os
import numpy as np
import nibabel as nib

def dice_coefficient(segmentation_dir, mask_dir):
    """
    Calculates the overall Dice coefficient between segmentation images and corresponding masks across all test samples.

    Args:
        segmentation_dir (str): Path to the directory containing the segmentation images (.nii.gz).
        mask_dir (str): Path to the directory containing the segmentation masks (.nii.gz).

    Returns:
        float: The overall Dice coefficient across all samples.
    """
    # Get a list of all segmentation image files and corresponding mask files
    image_files = sorted([f for f in os.listdir(segmentation_dir) if f.endswith('.nii.gz')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])

    # Ensure that the number of segmentation images and masks match
    assert len(image_files) == len(mask_files)

    # Initialize variables to store the total intersection and union
    total_intersection = 0
    total_union = 0

    # Loop through each segmentation image and mask pair
    for image_file, mask_file in zip(image_files, mask_files):
        # Load the segmentation image and mask
        segmentation_path = os.path.join(segmentation_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)
        segmentation_img = nib.load(segmentation_path).get_fdata()
        mask_img = nib.load(mask_path).get_fdata()

        # Binarize the segmentation image and mask
        segmentation_img = np.where(segmentation_img > 0, 1, 0)
        mask_img = np.where(mask_img > 0, 1, 0)

        # Calculate the intersection and union for this sample
        intersection = np.sum(segmentation_img * mask_img)
        union = np.sum(segmentation_img) + np.sum(mask_img)

        # Accumulate the intersection and union values
        total_intersection += intersection
        total_union += union

    # Calculate the overall Dice coefficient across all samples
    overall_dice = 2.0 * total_intersection / total_union

    return overall_dice