import tensorflow as tf
import numpy as np
import os
import cv2
from data.image_processing import normalize_image
from util.dataset_utils import load_dataset_from_tfrecord, save_dataset_to_tfrecord

import argparse

# Create a parser object and add arguments for modalities (list of strings, by default ['t2w', 't1w'])
parser = argparse.ArgumentParser(description='Create a dataset for image fusion')
parser.add_argument('--modalities', nargs='+', default=['T2w', 'T1w'], help='List of modalities to use for training')
args = parser.parse_args()


def extract_slice_number(filename):
    """Extract and return the slice number from a filename."""
    parts = filename.split('_')
    slice_number = parts[2].split('.')[0]
    return int(slice_number)

def load_image_pairs(patient_path, modalities):
    """
    Load and normalize image pairs from a given patient path with matching slice numbers for specified modalities.
    Returns a dictionary with keys for each modality and an additional key for slice numbers.
    """
    image_map = {}  # Maps slice numbers to modality paths

    for image_name in sorted(os.listdir(patient_path)):
        slice_number = extract_slice_number(image_name)
        for modality in modalities:
            if modality in image_name:
                if slice_number not in image_map:
                    image_map[slice_number] = {}
                image_map[slice_number][modality] = os.path.join(patient_path, image_name)

    modality_images = {modality: [] for modality in modalities}
    slice_numbers = []  # List to store slice numbers
    for slice_number, paths in sorted(image_map.items()):
        if all(modality in paths for modality in modalities):
            slice_numbers.append(slice_number)  # Store the slice number
            for modality in modalities:
                img = cv2.imread(paths[modality], cv2.IMREAD_GRAYSCALE)
                img = img[..., np.newaxis]  # Ensure the image has a channel dimension
                img_normalized = normalize_image(img)
                modality_images[modality].append(img_normalized)

    # Convert lists of images to numpy arrays
    modality_images_arrays = {modality: np.array(images) for modality, images in modality_images.items()}
    modality_images_arrays['slice_numbers'] = np.array(slice_numbers)  # Add slice numbers to the dict
    
    return modality_images_arrays

def create_fusion_dataset(base_dir, modalities):
    combined_data = {modality: [] for modality in modalities}
    all_patient_ids = []
    all_slice_numbers = []  # Initialize a list to store all slice numbers

    for patient in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient)
        modality_data = load_image_pairs(patient_path, modalities)
        # Check if the modality data contains images and slice numbers
        if all(len(data) > 0 for data in modality_data.values()) and 'slice_numbers' in modality_data:
            slice_numbers = modality_data.pop('slice_numbers')  # Extract and remove slice numbers from the data
            num_slices = len(slice_numbers)
            for modality, images in modality_data.items():
                combined_data[modality].append(images)
            all_patient_ids.extend([patient] * num_slices)
            all_slice_numbers.extend(slice_numbers)  # Append the slice numbers for this patient
        else:
            print(f"Skipping patient {patient} due to missing modality data")

    # Concatenate all arrays
    for modality in modalities:
        combined_data[modality] = np.concatenate(combined_data[modality], axis=0)
    all_patient_ids = np.array(all_patient_ids)
    all_slice_numbers = np.array(all_slice_numbers)  # Convert slice numbers list to a NumPy array

    print(f"Combined images shape for modality {modality}: {combined_data[modality].shape}")
    print(f"Total number of patient IDs: {len(all_patient_ids)}")
    print(f"Total number of unique patient IDs: {len(np.unique(all_patient_ids))}")
    print(f"Total number of slices: {len(all_slice_numbers)}")

    # Include all slice numbers in the dataset elements
    dataset_elements = tuple(combined_data[modality] for modality in modalities) + (all_patient_ids, all_slice_numbers)
    dataset = tf.data.Dataset.from_tensor_slices(dataset_elements)

    # Debug print for the final dataset
    print("Final dataset created with element specifications:")
    print(dataset.element_spec)

    return dataset

def get_datasets(modalities, batch_size, train_ratio=0.8, val_ratio=0.1):
    # Set a fixed seed for NumPy and TensorFlow random number generators
    np.random.seed(42)
    tf.random.set_seed(42)
    print("Batch size:", batch_size)

    loaded_dataset = load_dataset_from_tfrecord(os.path.join('datasets', '_'.join(modalities) + '_dataset.tfrecord'), modalities)

    # Get unique patient IDs
    patient_ids = np.unique([data['patient_id'].numpy().decode('utf-8') for data in loaded_dataset])

    # Split the patient IDs into training, validation, and test sets
    num_patients = len(patient_ids)
    print(f"Total number of patients: {num_patients}")
    num_train = int(num_patients * train_ratio)
    num_val = int(num_patients * val_ratio)

    train_patients = patient_ids[:num_train]
    val_patients = patient_ids[num_train:num_train + num_val]
    test_patients = patient_ids[num_train + num_val:]

    # Filter the dataset by patient ID
    train_patients_tf = tf.constant(train_patients)
    train_dataset = loaded_dataset.filter(lambda x: tf.reduce_any(tf.equal(x['patient_id'], train_patients_tf)))

    val_patients_tf = tf.constant(val_patients)
    val_dataset = loaded_dataset.filter(lambda x: tf.reduce_any(tf.equal(x['patient_id'], val_patients_tf)))

    test_patients_tf = tf.constant(test_patients)
    test_dataset = loaded_dataset.filter(lambda x: tf.reduce_any(tf.equal(x['patient_id'], test_patients_tf)))

    # Print lengths of the datasets
    print(f"Number of training samples: {len(list(train_dataset))}")
    print(f"Number of validation samples: {len(list(val_dataset))}")
    print(f"Number of test samples: {len(list(test_dataset))}")

    # Shuffle and batch the datasets
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)


    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the path to the 'slices' folder
    slices_folder = os.path.join(current_dir, 'slices')

    # Define the path to the datasets folder
    datasets_folder = os.path.join(current_dir, '..', 'datasets')

    # Create the datasets folder if it doesn't exist
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)

    dataset = create_fusion_dataset(slices_folder, args.modalities)
    
    # Save the dataset to a TFRecord file
    tfrecord_filename = '_'.join(args.modalities) + '_dataset.tfrecord'
    tfrecord_path = os.path.join(datasets_folder, tfrecord_filename)
    save_dataset_to_tfrecord(dataset, tfrecord_path, args.modalities)
