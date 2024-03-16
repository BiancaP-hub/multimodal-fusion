import subprocess
import os

def sct_deepseg_sc(image_path, seg_dir):
    """
    Generate a segmentation of a spinal cord image using SCT's deep segmentation tool.

    Args:
        image_path (str): Path to the input MRI image.
        segmentation_folder (str, optional): Folder to save the generated segmentation.
    """
    segmentation_cmd = ['sct_deepseg_sc', '-i', image_path, '-c', 't2', '-ofolder', seg_dir]
    subprocess.run(segmentation_cmd)

def generate_segmentations(dataset_dir='../../data-multi-subject', seg_dir='../seg_images', modality='T2w'):
    """
    Generate segmentations for all patients in the dataset.
    """
    # Create segmentation directory if it does not exist
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    patients = [p for p in os.listdir(dataset_dir) if p.startswith('sub-')]

    # # Skip patients with known issues
    # patients = [p for p in patients if p not in ['sub-mgh01', 'sub-vallHebron04', 'sub-cmrra05']]

    for patient in patients:
        image_path = os.path.join(dataset_dir, patient, 'anat', patient + f'_{modality}.nii.gz')
        sct_deepseg_sc(image_path, seg_dir)

if __name__ == '__main__':
    generate_segmentations()