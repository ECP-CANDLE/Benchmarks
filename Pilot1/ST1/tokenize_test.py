import tensorflow as tf
import sys
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
import pandas as pd
import numpy as np

## Input and prepare dataset
# fname is a csv file with type,smile headers

fname=sys.argv[1]
vocab_size = 40000  #
maxlen = 250  #

data_path_train = sys.argv[1]
data_train = pd.read_csv(data_path_train, sep=',', index_col=None)

print('done loading csv file\n', data_train.head())

y_train = data_train["type"].values.astype(float).reshape(-1, 1) * 1.0

print('y_train shape {}\ny_train {}'.format(y_train.shape, y_train))
# y_train shape (1000, 1)
# y_train [[14.209795],[...], ...]

tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(data_train["smiles"])

## Tokenize and pad
def prep_text(texts, tokenizer, max_sequence_length):
    # Turns text into into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(texts)
    return sequence.pad_sequences(text_sequences, maxlen=maxlen)

x_train = prep_text(data_train["smiles"], tokenizer, maxlen)

print('x_train shape {}\nx_train {}'.format(x_train.shape, x_train))

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

def serialize_smile(smile,label):
    feature = {
        'smile': smile,
        'label': label
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

df_train = pd.read_csv(fname)

smiles = np.array(df_train['smiles'].to_numpy().astype(str))
smiles = [n.encode() for n in smiles]
smiles = [_bytes_feature(n) for n in smiles]

labels = np.array(df_train['type'].to_numpy().astype(float))
labels = [_float_feature(n) for n in labels]

serialized_samples = [serialize_smile(smile,label) for smile,label in zip(smiles,labels)]
dataset = tf.data.Dataset.from_tensor_slices(serialized_samples)

ofname = fname + '.tok.tfrecord'
writer = tf.data.experimental.TFRecordWriter(ofname)
writer.write(dataset)
