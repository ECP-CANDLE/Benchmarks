The NT3 benchmark is a binary classification task on 1400 RNA-seq based gene expression profiles from the NCI Genomic Data Commons (GDC). 700 of these samples are from tumor tissues and the other 700 are their matched normals. There are 60483 features for each sample that are fed into a neural network with two dense layers on top of two convolution layers by default. 

#### Sample output

The following example out put is from a truncated run, using only 20 epochs, which is accomplished by modifying the nt3_default_model.txt file and then running:

```
python nt3_baseline_keras2.py
```
```

Using TensorFlow backend.
Params: {'logfile': None, 'metrics': 'accuracy', 'shuffle': True, 'out_activation': 'softmax', 'activation': 'relu', 'run_id': 'RUN000', 'train_bool': True, 'scaling': 'maxabs', 'output_dir': '/CANDLE/benchmarks/Pilot1/NT3', 'gpus': [], 'optimizer': 'sgd', 'learning_rate': 0.001, 'initialization': 'glorot_uniform', 'train_data': 'nt_train2.csv', 'classes': 2, 'verbose': False, 'conv': [128, 20, 1, 128, 10, 1], 'dense': [200, 20], 'timeout': 3600, 'rng_seed': 7102, 'experiment_id': 'EXP000', 'pool': [1, 10], 'test_data': 'nt_test2.csv', 'loss': 'categorical_crossentropy', 'config_file': '/CANDLE/benchmarks/Pilot1/NT3/nt3_default_model.txt', 'datatype': <class 'numpy.float32'>, 'batch_size': 20, 'model_name': 'nt3', 'drop': 0.1, 'data_url': 'ftp://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/normal-tumor/', 'epochs': 20}
X_train shape: (1120, 60483)
X_test shape: (280, 60483)
Y_train shape: (1120, 2)
Y_test shape: (280, 2)
X_train shape: (1120, 60483, 1)
X_test shape: (280, 60483, 1)
0 128 20 1
1 128 10 1
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d_1 (Conv1D)            (None, 60464, 128)        2688
_________________________________________________________________
activation_1 (Activation)    (None, 60464, 128)        0
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 60464, 128)        0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 60455, 128)        163968
_________________________________________________________________
activation_2 (Activation)    (None, 60455, 128)        0
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 6045, 128)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 773760)            0
_________________________________________________________________
dense_1 (Dense)              (None, 200)               154752200
_________________________________________________________________
activation_3 (Activation)    (None, 200)               0
_________________________________________________________________
dropout_1 (Dropout)          (None, 200)               0
_________________________________________________________________
dense_2 (Dense)              (None, 20)                4020
_________________________________________________________________
activation_4 (Activation)    (None, 20)                0
_________________________________________________________________
dropout_2 (Dropout)          (None, 20)                0
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 42
_________________________________________________________________
activation_5 (Activation)    (None, 2)                 0
=================================================================
Total params: 154,922,918
Trainable params: 154,922,918
Non-trainable params: 0
_________________________________________________________________
Train on 1120 samples, validate on 280 samples
Epoch 1/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6930 - acc: 0.5091Current time ....43.788
1120/1120 [==============================] - 43s - loss: 0.6931 - acc: 0.5054 - val_loss: 0.6929 - val_acc: 0.5143
Epoch 2/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6929 - acc: 0.5618Current time ....84.896
1120/1120 [==============================] - 41s - loss: 0.6929 - acc: 0.5598 - val_loss: 0.6928 - val_acc: 0.5143
Epoch 3/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6928 - acc: 0.5627Current time ....126.122
1120/1120 [==============================] - 41s - loss: 0.6928 - acc: 0.5598 - val_loss: 0.6927 - val_acc: 0.5143
Epoch 4/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6927 - acc: 0.5600Current time ....167.333
1120/1120 [==============================] - 41s - loss: 0.6927 - acc: 0.5563 - val_loss: 0.6926 - val_acc: 0.6893
Epoch 5/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6926 - acc: 0.5945Current time ....208.666
1120/1120 [==============================] - 41s - loss: 0.6926 - acc: 0.5929 - val_loss: 0.6925 - val_acc: 0.7500
Epoch 6/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6925 - acc: 0.6873Current time ....250.219
1120/1120 [==============================] - 41s - loss: 0.6925 - acc: 0.6866 - val_loss: 0.6923 - val_acc: 0.5143
Epoch 7/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6924 - acc: 0.5136Current time ....291.738
1120/1120 [==============================] - 41s - loss: 0.6924 - acc: 0.5161 - val_loss: 0.6923 - val_acc: 0.7607
Epoch 8/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6922 - acc: 0.6818Current time ....333.193
1120/1120 [==============================] - 41s - loss: 0.6922 - acc: 0.6839 - val_loss: 0.6921 - val_acc: 0.6857
Epoch 9/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6921 - acc: 0.6855Current time ....374.670
1120/1120 [==============================] - 41s - loss: 0.6921 - acc: 0.6848 - val_loss: 0.6920 - val_acc: 0.7571
Epoch 10/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6918 - acc: 0.7027Current time ....416.112
1120/1120 [==============================] - 41s - loss: 0.6918 - acc: 0.7009 - val_loss: 0.6918 - val_acc: 0.6786
Epoch 11/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6917 - acc: 0.6882Current time ....457.586
1120/1120 [==============================] - 41s - loss: 0.6917 - acc: 0.6893 - val_loss: 0.6916 - val_acc: 0.6214
Epoch 12/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6916 - acc: 0.6964Current time ....499.014
1120/1120 [==============================] - 41s - loss: 0.6916 - acc: 0.6955 - val_loss: 0.6915 - val_acc: 0.6393
Epoch 13/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6914 - acc: 0.5982Current time ....540.471
1120/1120 [==============================] - 41s - loss: 0.6914 - acc: 0.6027 - val_loss: 0.6914 - val_acc: 0.8071
Epoch 14/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6913 - acc: 0.7391Current time ....581.941
1120/1120 [==============================] - 41s - loss: 0.6913 - acc: 0.7420 - val_loss: 0.6911 - val_acc: 0.7250
Epoch 15/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6910 - acc: 0.6100Current time ....623.388
1120/1120 [==============================] - 41s - loss: 0.6909 - acc: 0.6143 - val_loss: 0.6910 - val_acc: 0.8179
Epoch 16/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6908 - acc: 0.7400Current time ....664.814
1120/1120 [==============================] - 41s - loss: 0.6908 - acc: 0.7402 - val_loss: 0.6908 - val_acc: 0.8143
Epoch 17/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6905 - acc: 0.7373Current time ....706.261
1120/1120 [==============================] - 41s - loss: 0.6905 - acc: 0.7384 - val_loss: 0.6905 - val_acc: 0.7071
Epoch 18/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6904 - acc: 0.6809Current time ....747.715
1120/1120 [==============================] - 41s - loss: 0.6904 - acc: 0.6795 - val_loss: 0.6904 - val_acc: 0.8107
Epoch 19/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6903 - acc: 0.7436Current time ....789.152
1120/1120 [==============================] - 41s - loss: 0.6903 - acc: 0.7455 - val_loss: 0.6901 - val_acc: 0.7000
Epoch 20/20
1100/1120 [============================>.] - ETA: 0s - loss: 0.6898 - acc: 0.7027Current time ....830.611
1120/1120 [==============================] - 41s - loss: 0.6899 - acc: 0.6946 - val_loss: 0.6899 - val_acc: 0.8143
```
