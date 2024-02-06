import tensorflow as tf
import numpy as np
import os
import cv2

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is a Tensor
        value = value.numpy()  # get its value as a numpy array
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(modality_data, patient_id):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {modality: _bytes_feature(image_data) for modality, image_data in modality_data.items()}
    feature['patient_id'] = _bytes_feature(patient_id.encode())  # assume patient_id is a string
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def save_dataset_to_tfrecord(dataset, filename, modalities):
    """
    Saves the dataset to a TFRecord file, handling any number of modalities.
    """
    with tf.io.TFRecordWriter(filename) as writer:
        for data in dataset:
            modality_data = {modality: tf.io.serialize_tensor(data[i]).numpy() for i, modality in enumerate(modalities)}
            patient_id = data[-1].numpy().decode('utf-8')  # Assuming the last element is the patient ID
            example = serialize_example(modality_data, patient_id)
            writer.write(example)

def parse_tfrecord_fn(example, modalities):
    """
    Parses a single tf.train.Example into tensors for each modality and patient ID.
    """
    feature_description = {modality: tf.io.FixedLenFeature([], tf.string) for modality in modalities}
    feature_description['patient_id'] = tf.io.FixedLenFeature([], tf.string)
    example = tf.io.parse_single_example(example, feature_description)
    
    parsed_data = {modality: tf.io.parse_tensor(example[modality], out_type=tf.float32) for modality in modalities}
    parsed_data['patient_id'] = example['patient_id']
    
    return parsed_data

def load_dataset_from_tfrecord(filename, modalities):
    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = raw_dataset.map(lambda x: parse_tfrecord_fn(x, modalities))
    return parsed_dataset

def normalize_image(image):
    """Normalize a single image array to the 0-1 range."""
    return image.astype('float32') / 255.0

def extract_slice_number(filename):
    """Extract and return the slice number from a filename."""
    parts = filename.split('_')
    slice_number = parts[2].split('.')[0]  # Assuming filename format is 'modality_slice_X.png'
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

if __name__ == '__main__':
    modalities = ['t2w', 't1w'] 
    dataset = create_fusion_dataset('paired_images', modalities)
    # Save the dataset to a TFRecord file
    save_dataset_to_tfrecord(dataset, os.path.join('datasets', '_'.join(modalities) + '_dataset.tfrecord'), modalities)

    # Load the dataset
    # loaded_dataset = load_dataset_from_tfrecord('dataset.tfrecord', modalities)
