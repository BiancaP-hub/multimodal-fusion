import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is a Tensor
        value = value.numpy()  # get its value as a numpy array
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(modality_data, patient_id, slice_number):
    """
    Creates a tf.train.Example message ready to be written to a file, including slice number.
    """
    # Convert modality images and patient ID to features
    feature = {modality: _bytes_feature(image_data) for modality, image_data in modality_data.items()}
    feature['patient_id'] = _bytes_feature(patient_id.encode())  # Assume patient_id is a string
    # Add slice number as an int64 feature
    feature['slice_number'] = _int64_feature(slice_number)
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def save_dataset_to_tfrecord(dataset, filename, modalities):
    """
    Saves the dataset to a TFRecord file, including slice numbers and handling any number of modalities.
    """
    with tf.io.TFRecordWriter(filename) as writer:
        for *modality_tensors, patient_id_tensor, slice_number_tensor in dataset:
            # Serialize modality data
            modality_data = {modality: tf.io.serialize_tensor(modality_tensors[i]).numpy() for i, modality in enumerate(modalities)}
            # Decode patient ID and slice number
            patient_id = patient_id_tensor.numpy().decode('utf-8')
            slice_number = int(slice_number_tensor.numpy())  # Ensure slice number is an integer
            # Create and write the example
            example = serialize_example(modality_data, patient_id, slice_number)
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

# def parse_tfrecord_fn(example, modalities):
#     """
#     Parses a single tf.train.Example into tensors for each modality, patient ID, and slice_number.
#     """
#     feature_description = {modality: tf.io.FixedLenFeature([], tf.string) for modality in modalities}
#     feature_description['patient_id'] = tf.io.FixedLenFeature([], tf.string)
#     feature_description['slice_number'] = tf.io.FixedLenFeature([], tf.int64)  # Add slice_number to the features
    
#     example = tf.io.parse_single_example(example, feature_description)
    
#     parsed_data = {modality: tf.io.parse_tensor(example[modality], out_type=tf.float32) for modality in modalities}
#     parsed_data['patient_id'] = example['patient_id']
#     parsed_data['slice_number'] = example['slice_number']  # Extract the slice_number
    
#     return parsed_data

# def load_dataset_from_tfrecord(filename, modalities):
#     raw_dataset = tf.data.TFRecordDataset(filename)
#     parsed_dataset = raw_dataset.map(lambda x: parse_tfrecord_fn(x, modalities))
#     return parsed_dataset
