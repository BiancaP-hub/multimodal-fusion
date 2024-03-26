import numpy as np

def dice_coefficient(segmentation_img, mask_img):
    """
    Calculates the Dice coefficient between a single segmentation image and a corresponding mask.

    Args:
        segmentation_img (np.ndarray): The segmentation image as a NumPy array.
        mask_img (np.ndarray): The mask image as a NumPy array.

    Returns:
        float: The Dice coefficient.
    """
    segmentation_img = np.where(segmentation_img > 0, 1, 0)
    mask_img = np.where(mask_img > 0, 1, 0)
    intersection = np.sum(segmentation_img * mask_img)
    union = np.sum(segmentation_img) + np.sum(mask_img)

    return 2.0 * intersection / union