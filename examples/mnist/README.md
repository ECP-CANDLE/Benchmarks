# MNIST Example

This example demonstrate how to convert keras code into CANDLE compliant. 
Please refer [tutorial](https://ecp-candle.github.io/Candle/html/tutorials/writing_candle_code.html) for more detail.

Here is the list of files,

- mnist.py: CANDLE class
- mnist_cnn.py and mnist_mlp.py: original mnist implementation from keras project
- mnist_cnn_candle.py: mnist_cnn.py converted in CANDLE compliant mode
- mnist_mlp_candle.py: mnist_mlp.py converted in CANDLE compliant mode
- mnist_params.txt: model parameters are stored in a file for reproduciblity


```
$ python mnist_cnn_candle.py -e 3
Using TensorFlow backend.

Importing candle utils for keras
Params:
{'activation': 'relu',
'batch_size': 128,
'data_type': <class 'numpy.float32'>,
'epochs': 3,
'experiment_id': 'EXP000',
'gpus': [],
'logfile': None,
'optimizer': 'rmsprop',
'output_dir': '/Users/hsyoo/projects/CANDLE/Benchmarks/examples/mnist/Output/EXP000/RUN000',
'profiling': False,
'rng_seed': 7102,
'run_id': 'RUN000',
'shuffle': False,
'timeout': -1,
'train_bool': True,
'verbose': None}
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
11493376/11490434 [==============================] - 2s 0us/step
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.

Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 60000 samples, validate on 10000 samples
Epoch 1/3
2020-05-13 11:53:17.373979: I tensorflow/core/platform/cpu_feature_guard.cc:145] This TensorFlow binary is optimized with Intel(R) MKL-DNN to use the following CPU instructions in performance critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
2020-05-13 11:53:17.374474: I tensorflow/core/common_runtime/process_util.cc:115] Creating new thread pool with default inter op setting: 16. Tune using inter_op_parallelism_threads for best performance.
60000/60000 [==============================] - 56s 932us/step - loss: 0.2719 - acc: 0.9157 - val_loss: 0.0683 - val_acc: 0.9774
Epoch 2/3
60000/60000 [==============================] - 55s 909us/step - loss: 0.0904 - acc: 0.9733 - val_loss: 0.0411 - val_acc: 0.9872
Epoch 3/3
60000/60000 [==============================] - 55s 909us/step - loss: 0.0666 - acc: 0.9808 - val_loss: 0.0339 - val_acc: 0.9893
Test loss: 0.03386178284487105
Test accuracy: 0.9893
```
