import os
import ants

# Define the base path to the dataset
base_path = '../../data-multi-subject'

# Define the path to save registered images
registered_images_path = './registered_images'
if not os.path.exists(registered_images_path):
    os.makedirs(registered_images_path)

# Specify the modalities to be processed
modalities = ['T1w', 'T2w']

# Iterate over each subject's folder
for subject_folder in os.listdir(base_path):
    if subject_folder.startswith('sub-'):
        subject_path = os.path.join(base_path, subject_folder, 'anat')
        if os.path.isdir(subject_path):
            images = {}
            for modality in modalities:
                # Construct the file path for each modality
                image_path = os.path.join(subject_path, f'{subject_folder}_{modality}.nii.gz')
                if os.path.exists(image_path):
                    # Read and reorient the image
                    images[modality] = ants.image_read(image_path, reorient='LSA')
                else:
                    print(f'Missing {modality} image for {subject_folder}')
                    break 
            
            # Check if all modalities are loaded
            if len(images) == len(modalities):
                # Register the first modality to the second modality
                fixed_modality = modalities[0]
                moving_modality = modalities[1]
                transform = ants.registration(fixed=images[fixed_modality], moving=images[moving_modality], type_of_transform='SyN', verbose=True)
                registered_img = transform['warpedmovout']
                
                # Save the registered image
                registered_image_path = os.path.join(registered_images_path, f'{subject_folder}_{moving_modality}_to_{fixed_modality}.nii.gz')
                ants.image_write(registered_img, registered_image_path)
                print(f'Registered {moving_modality} to {fixed_modality} image saved for {subject_folder}')
