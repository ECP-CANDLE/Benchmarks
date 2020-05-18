The Pilot1 Attn Benchmark requires an hdf5 file specified by the hyperparameter "in", name of this file for default case is: top_21_1fold_001.h5

Benchmark auto downloads the file below:
http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_1fold_001.h5 (~4GB)

Any file of the form top*21_1fold*"ijk".h5 can be used as input

## Sample run:

```
python attn_baseline_keras2.py
Params: {'model_name': 'attn', 'dense': [2000, 600], 'batch_size': 32, 'epochs': 1, 'activation': 'relu', 'loss': 'categorical_crossentropy', 'optimizer': 'sgd', 'drop': 0.2, 'learning_rate': 1e-05, 'momentum': 0.7, 'scaling': 'minmax', 'validation_split': 0.1, 'epsilon_std': 1.0, 'rng_seed': 2017, 'initialization': 'glorot_uniform', 'latent_dim': 2, 'batch_normalization': False, 'in': 'top_21_1fold_001.h5', 'save_path': 'candle_save', 'save_dir': './save/001/', 'use_cp': False, 'early_stop': True, 'reduce_lr': True, 'feature_subsample': 0, 'nb_classes': 2, 'timeout': 3600, 'verbose': None, 'logfile': None, 'train_bool': True, 'experiment_id': 'EXP000', 'run_id': 'RUN000', 'shuffle': False, 'gpus': [], 'profiling': False, 'residual': False, 'warmup_lr': False, 'use_tb': False, 'tsne': False, 'datatype': <class 'numpy.float32'>, 'output_dir': '/nfs2/jain/Benchmarks/Pilot1/Attn/Output/EXP000/RUN000'}
...
...
processing h5 in file top_21_1fold_001.h5

x_train shape: (271915, 6212)
x_test shape: (33989, 6212)
Examples:
Total: 339893
Positive: 12269 (3.61% of total)

X_train shape: (271915, 6212)
X_test shape: (33989, 6212)
Y_train shape: (271915, 2)
Y_test shape: (33989, 2)
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 6212)         0
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1000)         6213000     input_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 1000)         4000        dense_1[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1000)         1001000     batch_normalization_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 1000)         4000        dense_2[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1000)         1001000     batch_normalization_1[0][0]
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 1000)         0           batch_normalization_2[0][0]
                                                                 dense_3[0][0]
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 500)          500500      multiply_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 500)          2000        dense_4[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 500)          0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 250)          125250      dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 250)          1000        dense_5[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 250)          0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 125)          31375       dropout_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 125)          500         dense_6[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 125)          0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 60)           7560        dropout_3[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 60)           240         dense_7[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 60)           0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 30)           1830        dropout_4[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 30)           120         dense_8[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 30)           0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 2)            62          dropout_5[0][0]
==================================================================================================

Total params: 8,893,437
Trainable params: 8,887,507
Non-trainable params: 5,930
..
..
271915/271915 [==============================] - 631s 2ms/step - loss: 0.8681 - acc: 0.5548 - tf_auc: 0.5371 - val_loss: 0.6010 - val_acc: 0.8365 - val_tf_auc: 0.5743
Current time ....631.567

Epoch 00001: val_loss improved from inf to 0.60103, saving model to ./save/001/Agg_attn_bin.autosave.model.h5
creating table of predictions
creating figure 1 at ./save/001/Agg_attn_bin.auroc.pdf
creating figure 2 at ./save/001/Agg_attn_bin.auroc2.pdf
f1=0.234 auroc=0.841 aucpr=0.990
creating figure 3 at ./save/001/Agg_attn_bin.aurpr.pdf
creating figure 4 at ./save/001/Agg_attn_bin.confusion_without_norm.pdf
Confusion matrix, without normalization
[[27591 5190][ 360 848]]
Confusion matrix, without normalization
[[27591 5190][ 360 848]]
Normalized confusion matrix
[[0.84 0.16][0.3 0.7 ]]
Examples:
Total: 339893
Positive: 12269 (3.61% of total)

0.7718316679565835
0.7718316679565836
precision recall f1-score support

           0       0.99      0.84      0.91     32781
           1       0.14      0.70      0.23      1208

micro avg 0.84 0.84 0.84 33989
macro avg 0.56 0.77 0.57 33989
weighted avg 0.96 0.84 0.88 33989

[[27591 5190][ 360 848]]
score
[0.5760348070144456, 0.8367118835449219, 0.5936741828918457]
Test val_loss: 0.5760348070144456
Test accuracy: 0.8367118835449219
Saved model to disk
Loaded json model from disk
json Validation loss: 0.560062773128295
json Validation accuracy: 0.8367118835449219
json accuracy: 83.67%
Loaded yaml model from disk
yaml Validation loss: 0.560062773128295
yaml Validation accuracy: 0.8367118835449219
yaml accuracy: 83.67%
Yaml_train_shape: (271915, 2)
Yaml_test_shape: (33989, 2)
```
