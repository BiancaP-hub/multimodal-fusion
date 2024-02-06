import os
import ants

# Define the base path to the dataset
base_path = '../../data-multi-subject'

# Define the path to save registered images
registered_images_path = './registered_images'
if not os.path.exists(registered_images_path):
    os.makedirs(registered_images_path)

# Iterate over each subject's folder
for subject_folder in os.listdir(base_path):
    if subject_folder.startswith('sub-'):
        # Rest of the code
        subject_path = os.path.join(base_path, subject_folder, 'anat')
        if os.path.isdir(subject_path):
            # Construct the file paths for T1w and T2w images
            t2w_image_path = os.path.join(subject_path, f'{subject_folder}_T2w.nii.gz')
            t1w_image_path = os.path.join(subject_path, f'{subject_folder}_T1w.nii.gz')
            
            # Check if both images exist before proceeding
            if os.path.exists(t2w_image_path) and os.path.exists(t1w_image_path):
                # Read and reorient images
                t2w = ants.image_read(t2w_image_path, reorient='LSA')
                t1w = ants.image_read(t1w_image_path, reorient='LSA')
                
                # Register T1w to T2w
                transform = ants.registration(fixed=t2w, moving=t1w, type_of_transform='SyN', verbose=True)
                registered_img = transform['warpedmovout']
                
                # Save the registered image
                registered_image_path = os.path.join(registered_images_path, f'{subject_folder}_T1w_to_T2w.nii.gz')
                ants.image_write(registered_img, registered_image_path)
                print(f'Registered image saved for {subject_folder}')
            else:
                print(f'Missing T1w or T2w image for {subject_folder}')
