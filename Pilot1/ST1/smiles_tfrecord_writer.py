import tensorflow as tf
import numpy as np
import pandas as pd
import pprofile
import sys
import smiles_tokenizer as st

DEBUG=1
from datetime import datetime

def _dt():
    '''helper'''
    if DEBUG==1:
        return datetime.today().strftime('%Y-%m-%d %H:%M:%S ')
    else:
        return ''

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

def _int_feature(value):
    """takes an np.array of ints"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value = value))

def serialize_example(smile,label):
    '''returns a feature with keys smile and label'''
    
    smile = _int_feature(smile)
    label = _float_feature(label)
    
    feature = {
        'smile': smile,
        'label': label
    }

    # returns an Example (sample which has one feature, the smile and one label, the score)
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # serializes the sample to a string
    return example_proto.SerializeToString()

if __name__ == '__main__':
    fname = sys.argv[1]

    print(_dt(), 'reading and tokenizing smile file')
    smiles, labels = st.read_csv_and_tokenize(fname)
    print(_dt(), 'done reading and tokenizing smile file')

    print(_dt(), 'serializing {} samples'.format(len(smiles)))
    serialized_samples = [serialize_example(smile,label) for smile,label in zip(smiles,labels)]
    print(_dt(), 'done serializing samples')

    print(_dt(), 'creating tf dataset')
    dataset = tf.data.Dataset.from_tensor_slices(serialized_samples)
    print(_dt(), 'done creating tf dataset')

    print(_dt(), 'writing tf dataset to disk')
    ofname = fname + '.tok.tfrecord'
    writer = tf.data.experimental.TFRecordWriter(ofname)
    writer.write(dataset)
    print(_dt(), 'done writing tf dataset to disk')
