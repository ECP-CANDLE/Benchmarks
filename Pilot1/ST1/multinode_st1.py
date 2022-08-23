
import tensorflow as tf
import json
import os
import numpy as np
from pbsutils import tf_config


# Set up input files:
path = '/lambda_stor/data/enums-100-test/'
filenames = [
    'ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet.xform-smiles.csv.reg.00',
    #'ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet.xform-smiles.csv.reg.01',
    #'ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet.xform-smiles.csv.reg.02',
    #'ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet.xform-smiles.csv.reg.03'
]
for n in range(len(filenames)):
    filenames[n] = path + filenames[n]



# Set GPUs to use only the memory that they need:
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



# Set up TF_CONFIG environment variable:
os.environ['TF_CONFIG'] = json.dumps(tf_config)
num_workers = len(tf_config['cluster']['worker'])
print(os.environ['TF_CONFIG'])     


# Create a distributed computing strategy
mirrored_strategy = tf.distribute.MirroredStrategy()


# Create a tf.data.Dataset object.
dataset = tf.data.experimental.CsvDataset(
    filenames,
    [tf.float32, tf.string]
)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
dataset = dataset.with_options(options)


# Convert it to numpy array x_train and y_train
np_dataset = np.array(list(dataset.as_numpy_iterator()))
x_train = np_dataset[:,1]
y_train = np_dataset[:,0].reshape(-1,1)


# Tokenize the smiles
vocab_size = 40000  # 
maxlen = 250  # 

vectorize_layer = tf.keras.layers.TextVectorization(
    output_mode='int',
    standardize=None,
    max_tokens=vocab_size,
    split='character',
    output_sequence_length=maxlen,
    pad_to_max_tokens=True
)

v_samples=x_train.shape[0]
v_batch_size=1000
v_steps=samples/v_batch_size
vectorize_layer.adapt(x_train, batch_size=v_batch_size, steps=v_steps)

fn = lambda x: vectorize_layer(x)
x_train = fn(x_train)

train=tf.Dataset.zip((x_train,y_train))

