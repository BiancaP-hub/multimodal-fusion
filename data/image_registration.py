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

# Usage
register_modalities('T2w', 'T1w')