import tensorflow as tf

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
            patient_id = data[-1].numpy().decode('utf-8')  # Last element is the patient ID
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