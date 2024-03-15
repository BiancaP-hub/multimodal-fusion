import tensorflow as tf
import numpy as np
import os
import cv2
from data.image_processing import normalize_image
from util.image_utils import load_dataset_from_tfrecord, save_dataset_to_tfrecord

import argparse

# Create a parser object and add arguments for modalities (list of strings, by default ['t2w', 't1w'])
parser = argparse.ArgumentParser(description='Create a dataset for image fusion')
parser.add_argument('--modalities', nargs='+', default=['t2w', 't1w'], help='List of modalities to use for training')
args = parser.parse_args()


def extract_slice_number(filename):
    """Extract and return the slice number from a filename."""
    parts = filename.split('_')
    slice_number = parts[2].split('.')[0]
    return int(slice_number)

def load_image_pairs(patient_path, modalities):
    """Load and normalize image pairs from a given patient path with matching slice numbers for specified modalities."""
    image_map = {}  # Maps slice numbers to modality paths

    for image_name in sorted(os.listdir(patient_path)):
        slice_number = extract_slice_number(image_name)
        for modality in modalities:
            if modality in image_name:
                if slice_number not in image_map:
                    image_map[slice_number] = {}
                image_map[slice_number][modality] = os.path.join(patient_path, image_name)

    modality_images = {modality: [] for modality in modalities}
    for slice_number, paths in image_map.items():
        if all(modality in paths for modality in modalities):
            for modality in modalities:
                img = cv2.imread(paths[modality], cv2.IMREAD_GRAYSCALE)
                img = img[..., np.newaxis]
                modality_images[modality].append(normalize_image(img))

    return {modality: np.array(images) for modality, images in modality_images.items()}

def create_fusion_dataset(base_dir, modalities):
    combined_data = {modality: [] for modality in modalities}
    all_patient_ids = []
    for patient in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient)
        modality_data = load_image_pairs(patient_path, modalities)
        if all(len(data) > 0 for data in modality_data.values()):
            for modality, images in modality_data.items():
                combined_data[modality].append(images)
            all_patient_ids.extend([patient] * len(next(iter(modality_data.values()))))
        else:
            print(f"Skipping patient {patient} due to missing modality data")

    # Concatenate all arrays
    for modality in modalities:
        combined_data[modality] = np.concatenate(combined_data[modality], axis=0)
        print(f"Combined images shape for modality {modality}: {combined_data[modality].shape}")
    all_patient_ids = np.array(all_patient_ids)
    print(f"Total number of patient IDs: {len(all_patient_ids)}")

    dataset_elements = tuple(combined_data[modality] for modality in modalities) + (all_patient_ids,)
    dataset = tf.data.Dataset.from_tensor_slices(dataset_elements)

    # Debug print for the final dataset
    print("Final dataset created with element specifications:")
    print(dataset.element_spec)

    return dataset

def get_datasets(modalities, batch_size, train_ratio=0.8, val_ratio=0.1):
    # Set a fixed seed for NumPy and TensorFlow random number generators
    np.random.seed(42)
    tf.random.set_seed(42)

    loaded_dataset = load_dataset_from_tfrecord(os.path.join('datasets', '_'.join(modalities) + '_dataset.tfrecord'), modalities)

    dataset_size = sum(1 for _ in loaded_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)

    # Shuffle the dataset with a fixed buffer size and seed
    full_dataset = loaded_dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=False)

    # Split the dataset
    train_dataset = full_dataset.take(train_size)
    test_val_dataset = full_dataset.skip(train_size)
    val_dataset = test_val_dataset.take(val_size)
    test_dataset = test_val_dataset.skip(val_size)

    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    dataset = create_fusion_dataset('paired_images', args.modalities)
    # Save the dataset to a TFRecord file
    save_dataset_to_tfrecord(dataset, os.path.join('datasets', '_'.join(args.modalities) + '_dataset.tfrecord'), args.modalities)
