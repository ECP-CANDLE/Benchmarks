import tensorflow as tf
import numpy as np
import pandas as pd
#import pprofile
import sys

#profiler = pprofile.Profile()
#fname='ml.3CLpro.100'
fname = sys.argv[1]

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  # type(tf.constant(0) is <class 'tensorflow.python.framework.ops.EagerTensor'>"
  #if isinstance(value, type(tf.constant(0))):
  #  value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_smile(smile):
    feature = {
        'smiles': smile
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

df_train = pd.read_csv(fname)
smiles = np.array(df_train['smiles'].to_numpy().astype(str))
smiles = [n.encode() for n in smiles]
smiles = [_bytes_feature(n) for n in smiles]

serialized_smiles = [serialize_smile(smile) for smile in smiles]
dataset = tf.data.Dataset.from_tensor_slices(serialized_smiles)

# filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(fname+'.tfrecord')
writer.write(dataset)
