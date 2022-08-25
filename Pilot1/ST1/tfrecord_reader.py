import tensorflow as tf
fname='test.tfrecord'
raw_dataset = tf.data.TFRecordDataset([fname])


# Create a description of the features.
# In our case, we have only one feature, the smile string.

feature_description = {
    'smiles': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)


parsed_dataset = raw_dataset.map(_parse_function)
print(parsed_dataset)

# for parsed_record in parsed_dataset.take(10):
#   print(repr(parsed_record))
