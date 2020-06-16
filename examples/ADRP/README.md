# Pilot1 ADRP Benchmark

## loads a csv file

Benchmark auto downloads the file below:
http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/ (~500MB)

## Sample run:

```
$ export CUDA_VISIBLE_DEVICES=1
$ python adrp_baseline_keras2.py
Using TensorFlow backend.
Importing candle utils for keras
Configuration file:  /home/jain/CANDLE/fork/Benchmarks/examples/ADRP/adrp_default_model.txt
{'activation': 'relu',
 'batch_normalization': False,
 'batch_size': 32,
 'data_url': 'ftp://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/',
 'dense': [250, 125, 60, 30],
 'drop': 0.1,
 'early_stop': True,
 'epochs': 1,
 'epsilon_std': 1.0,
 'feature_subsample': 0,
 'in': 'adrp-p1.csv',
 'initialization': 'glorot_uniform',
 'latent_dim': 2,
 'learning_rate': 0.0001,
 'loss': 'mean_squared_error',
 'model_name': 'adrp',
 'momentum': 0.9,
 'nb_classes': 2,
 'optimizer': 'sgd',
 'reduce_lr': True,
 'rng_seed': 2017,
 'save_path': './001/',
 'scaling': 'minmax',
 'timeout': 3600,
 'use_cp': False,
 'validation_split': 0.1}
Params:
{'activation': 'relu',
 'batch_normalization': False,
 'batch_size': 32,
 'data_url': 'ftp://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/',
 'datatype': <class 'numpy.float32'>,
 'dense': [250, 125, 60, 30],
 'drop': 0.1,
 'early_stop': True,
 'epochs': 400,
 'epsilon_std': 1.0,
 'experiment_id': 'EXP000',
 'feature_subsample': 0,
 'gpus': [],
 'in': 'adrp-p1.csv',
 'initialization': 'glorot_uniform',
 'latent_dim': 2,
 'learning_rate': 0.0001,
 'logfile': None,
 'loss': 'mean_squared_error',
 'model_name': 'adrp',
 'momentum': 0.9,
 'nb_classes': 2,
 'optimizer': 'sgd',
 'output_dir': '/home/jain/CANDLE/fork/Benchmarks/examples/ADRP/Output/EXP000/RUN000',
 'profiling': False,
 'reduce_lr': True,
 'residual': False,
 'rng_seed': 2017,
 'run_id': 'RUN000',
 'save_path': './001/',
 'scaling': 'minmax',
 'shuffle': False,
 'timeout': 0,
 'train_bool': True,
 'tsne': False,
 'use_cp': False,
 'use_tb': False,
 'validation_split': 0.1,
 'verbose': None,
 'warmup_lr': False}
WARNING:tensorflow:From /home/jain/CANDLE/fork/Benchmarks/common/keras_utils.py:51: The name tf.set_random_seed is deprecated. Please use tf.compat.v1.set_random_seed instead.

Params: {'data_url': 'ftp://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/', 'in': 'adrp-p1.csv', 'model_name': 'adrp', 'dense': [250, 125, 60, 30], 'batch_size': 32, 'epochs': 1, 'activation': 'relu', 'loss': 'mean_squared_error', 'optimizer': 'sgd', 'drop': 0.1, 'learning_rate': 0.0001, 'momentum': 0.9, 'scaling': 'minmax', 'validation_split': 0.1, 'epsilon_std': 1.0, 'rng_seed': 2017, 'initialization': 'glorot_uniform', 'latent_dim': 2, 'batch_normalization': False, 'save_path': './001/', 'use_cp': False, 'early_stop': True, 'reduce_lr': True, 'feature_subsample': 0, 'nb_classes': 2, 'timeout': 3600, 'verbose': None, 'logfile': None, 'train_bool': True, 'experiment_id': 'EXP000', 'run_id': 'RUN000', 'shuffle': False, 'gpus': [], 'profiling': False, 'residual': False, 'warmup_lr': False, 'use_tb': False, 'tsne': False, 'datatype': <class 'numpy.float32'>, 'output_dir': '/home/jain/CANDLE/fork/Benchmarks/examples/ADRP/Output/EXP000/RUN000'}
processing csv in file adrp-p1.csv
PL= 1614
X_train shape: (27447, 1613)
X_test shape: (6862, 1613)
Y_train shape: (27447,)
Y_test shape: (6862,)
WARNING:tensorflow:From /home/jain/.local/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 1613)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 250)               403500    
_________________________________________________________________
dropout_1 (Dropout)          (None, 250)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 125)               31375     
_________________________________________________________________
dropout_2 (Dropout)          (None, 125)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 60)                7560      
_________________________________________________________________
dropout_3 (Dropout)          (None, 60)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 30)                1830      
_________________________________________________________________
dropout_4 (Dropout)          (None, 30)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 31        
=================================================================
Total params: 444,296
Trainable params: 444,296
Non-trainable params: 0
_________________________________________________________________
/home/jain/.local/lib/python3.7/site-packages/keras/callbacks/callbacks.py:998: UserWarning: `epsilon` argument is deprecated and will be removed, use `min_delta` instead.
  warnings.warn('`epsilon` argument is deprecated and '
2020-03-23 14:36:20.461062: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-03-23 14:36:20.463626: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: CUDA_ERROR_UNKNOWN: unknown error
2020-03-23 14:36:20.463720: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (jain): /proc/driver/nvidia/version does not exist
2020-03-23 14:36:20.464039: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-03-23 14:36:20.475490: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2808000000 Hz
2020-03-23 14:36:20.475685: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2dab430 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-23 14:36:20.475708: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /home/jain/.local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 27447 samples, validate on 6862 samples
Epoch 1/1
27447/27447 [==============================] - 5s 173us/step - loss: 3.4695 - mae: 1.3269 - r2: -2.1720 - val_loss: 1.2343 - val_mae: 0.9235 - val_r2: -0.1880

Epoch 00001: val_loss improved from inf to 1.23431, saving model to ./001/agg_adrp.autosave.model.h5
[1.2343122459159792, 0.9235042333602905, -0.18803702294826508]
dict_keys(['val_loss', 'val_mae', 'val_r2', 'loss', 'mae', 'r2', 'lr'])
saving to path:  ./001/
Test val_loss: 1.2343122459159792
Test val_mae: 0.9235042333602905
```
