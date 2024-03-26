import os
import ants

def register_modalities(modality1, modality2, base_path='../../data-multi-subject', registered_images_path='./registered_images'):
    if not os.path.exists(registered_images_path):
        os.makedirs(registered_images_path)

    modalities = [modality1, modality2]

    for subject_folder in os.listdir(base_path):
        if subject_folder.startswith('sub-'):
            subject_path = os.path.join(base_path, subject_folder, 'anat')
            if os.path.isdir(subject_path):
                images = {}
                for modality in modalities:
                    image_path = os.path.join(subject_path, f'{subject_folder}_{modality}.nii.gz')
                    if os.path.exists(image_path):
                        images[modality] = ants.image_read(image_path, reorient='LSA')
                    else:
                        print(f'Missing {modality} image for {subject_folder}')
                        break 

                if len(images) == len(modalities):
                    transform = ants.registration(fixed=images[modality1], moving=images[modality2], type_of_transform='SyN', verbose=True)
                    registered_img = transform['warpedmovout']

                    registered_image_path = os.path.join(registered_images_path, f'{subject_folder}_{modality2}_to_{modality1}.nii.gz')
                    ants.image_write(registered_img, registered_image_path)
                    print(f'Registered {modality2} to {modality1} image saved for {subject_folder}')

def align_images(fixed_image_path, moving_image_path):
    """
    Aligns the moving image to the fixed image using ANTsPy.

    Args:
        fixed_image_path (str): Path to the fixed image (the reference image).
        moving_image_path (str): Path to the moving image (the image to be aligned).

    Returns:
        np.ndarray: The aligned moving image data.
    """
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    
    # Perform registration (alignment)
    mytx = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN')
    
    # Apply the transformation to align the moving image to the fixed image
    aligned_moving_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=mytx['fwdtransforms'])
    
    return aligned_moving_image

if __name__ == '__main__':
    moving_image_dir = 'results/reconstructed_images/ssim_pixel_multi_scale'
    fixed_image_dir = 'results/input_images/T1w'
    aligned_images_dir = 'results/reconstructed_images/ssim_pixel_multi_scale/T1w_aligned'
    for image in os.listdir(moving_image_dir):
        # if the image does not end with _seg.nii.gz and is not a directory
        if not image.endswith('_seg.nii.gz') and not os.path.isdir(os.path.join(moving_image_dir, image)):
            moving_image_path = os.path.join(moving_image_dir, image)
            # Remove '_T1w' from the image name to get the fixed image name
            fixed_image_name = image.replace('_T2w', '')      
            fixed_image_path = os.path.join(fixed_image_dir, fixed_image_name)
            aligned_image = align_images(fixed_image_path, moving_image_path)
            aligned_image_path = os.path.join(aligned_images_dir, image)
            ants.image_write(aligned_image, aligned_image_path)
            print(f'Aligned image saved to {aligned_image_path}')