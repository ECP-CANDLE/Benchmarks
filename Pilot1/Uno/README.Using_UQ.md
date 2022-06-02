## *UQ*: Predicting Tumor Dose Response across Multiple Data Sources with added UQ functionality.



## Functionality

*UQ* adds uncertainty quantification (UQ) functionality to the Uno model. For information about the underlaying model, please refer to the Uno benchmark documentation.



This page overviews the added UQ functionality provided, which includes:

- Generation of holdout set.

- Training excluding the holdout set.

- Inference for the specified data.

- Training for homoscedastic and heteroscedastic models.

- Empirical calibration of UQ for the trained models.



## Holdout

The holdout script generates a set of identifiers to holdout during training, depending on the --partition_by argument.

If --partition_by is 'drug_pair' it generates a set of drug IDs.

If --partition_by is 'cell' it generates a set of cell IDs.

In any other case it generates a set of indices.



The fraction to reserve in the holdout set is given by the --val_split argument.



#### Example output

```
python uno_holdoutUQ_data.py
Using TensorFlow backend.
Importing candle utils for keras
Params:
{'activation': 'relu',
 'agg_dose': 'AUC',
 'base_lr': None,
 'batch_normalization': False,
 'batch_size': 32,
 'by_cell': None,
 'by_drug': None,
 'cache': None,
 'cell_feature_subset_path': '',
 'cell_features': ['rnaseq'],
 'cell_subset_path': '',
 'cell_types': None,
 'cp': False,
 'cv': 1,
 'data_type': <class 'numpy.float32'>,
 'dense': [1000, 1000, 1000],
 'dense_feature_layers': [1000, 1000, 1000],
 'dropout': 0.1,
 'drug_feature_subset_path': '',
 'drug_features': ['descriptors', 'fingerprints'],
 'drug_median_response_max': 1,
 'drug_median_response_min': -1,
 'drug_subset_path': '',
 'epochs': 10,
 'exclude_cells': [],
 'exclude_drugs': [],
 'experiment_id': 'EXP000',
 'export_csv': None,
 'export_data': None,
 'feature_subsample': 0,
 'feature_subset_path': '',
 'gpus': [],
 'growth_bins': 0,
 'initial_weights': None,
 'learning_rate': 0.01,
 'logfile': None,
 'loss': 'mse',
 'max_val_loss': 1.0,
 'no_feature_source': True,
 'no_gen': False,
 'no_response_source': True,
 'optimizer': 'sgd',
 'output_dir': './Output/EXP000/RUN000',
 'partition_by': 'cell',
 'preprocess_rnaseq': 'none',
 'profiling': False,
 'reduce_lr': False,
 'residual': False,
 'rng_seed': 2018,
 'run_id': 'RUN000',
 'sample_repetition': False,
 'save_path': 'save_default/',
 'save_weights': 'default.weights.h5',
 'scaling': 'std',
 'shuffle': False,
 'single': True,
 'tb': False,
 'tb_prefix': 'tb',
 'test_sources': ['train'],
 'timeout': 3600,
 'train_bool': True,
 'train_sources': ['gCSI'],
 'use_exported_data': None,
 'use_filtered_genes': False,
 'use_landmark_genes': True,
 'val_split': 0.2,
 'verbose': None,
 'warmup_lr': False}
partition_by:  cell
Cell IDs in holdout set written in file: save_default/infer_cell_ids

```



## Train

The train script trains the model, as in the underlying Uno benchmark, but excluding the IDs in the holdout  file. The file with the holdout set should be provided via one of the following arguments

- --uq_exclude_drugs_file='file'     if the file contains a set of drug IDs.

- --uq_exclude_cells_file='file'       if the file contains a set of cell IDs.

- --uq_exclude_indices_file='file'   if the file contains a set of indices.



Additional --loss 'het' and --loss 'qtl' options are available. This will learn the input-dependent noise level as well as the main regression variable specified (i.e. growth or AUC).



#### Example output

```

python uno_trainUQ_keras2.py --cp True --uq_exclude_cells_file 'save_default/infer_cell_ids'

Using TensorFlow backend.
Importing candle utils for keras
Params:
{'activation': 'relu',
 'agg_dose': 'AUC',
 'base_lr': None,
 'batch_normalization': False,
 'batch_size': 32,
 'by_cell': None,
 'by_drug': None,
 'cache': None,
 'cell_feature_subset_path': '',
 'cell_features': ['rnaseq'],
 'cell_subset_path': '',
 'cell_types': None,
 'cp': True,
 'cv': 1,
 'data_type': <class 'numpy.float32'>,
 'dense': [1000, 1000, 1000],
 'dense_feature_layers': [1000, 1000, 1000],
 'dropout': 0.1,
 'drug_feature_subset_path': '',
 'drug_features': ['descriptors', 'fingerprints'],
 'drug_median_response_max': 1,
 'drug_median_response_min': -1,
 'drug_subset_path': '',
 'epochs': 10,
 'exclude_cells': [],
 'exclude_drugs': [],
 'exclude_indices': [],
 'experiment_id': 'EXP000',
 'export_csv': None,
 'export_data': None,
 'feature_subsample': 0,
 'feature_subset_path': '',
 'gpus': [],
 'growth_bins': 0,
 'initial_weights': None,
 'learning_rate': 0.01,
 'logfile': None,
 'loss': 'mse',
 'max_val_loss': 1.0,
 'no_feature_source': True,
 'no_gen': False,
 'no_response_source': True,
 'optimizer': 'sgd',
 'output_dir': './Output/EXP000/RUN000',
 'partition_by': 'cell',
 'preprocess_rnaseq': 'none',
 'reduce_lr': False,
 'reg_l2': 0.0,
 'residual': False,
 'rng_seed': 2018,
 'run_id': 'RUN000',
 'sample_repetition': False,
 'save_path': 'save_default/',
 'save_weights': 'saved.weights.h5',
 'scaling': 'std',
 'shuffle': False,
 'single': True,
 'tb': False,
 'tb_prefix': 'tb',
 'test_sources': ['train'],
 'timeout': 3600,
 'train_bool': True,
 'train_sources': ['gCSI'],
 'uq_exclude_cells_file': 'save_default/infer_cell_ids',
 'use_exported_data': None,
 'use_filtered_genes': False,
 'use_landmark_genes': True,
 'val_split': 0.2,
 'verbose': None,
 'warmup_lr': False}
Read file: save_default/infer_cell_ids
Number of elements read: 72
Cells to exclude: ['gCSI.NCI-H889', 'gCSI.MEWO', 'gCSI.PA-TU-8902', 'gCSI.BCPAP', 'gCSI.CAL-12T', 'gCSI.NCI-H727', 'gCSI.HUH-1', 'gCSI.NUGC-4', 'gCSI.MKN74', 'gCSI.PK-1', 'gCSI.A2058', 'gCSI.RAJI', 'gCSI.JHH-7', 'gCSI.SUIT-2', 'gCSI.OE21', 'gCSI.HCC1806', 'gCSI.PANC-10-05', 'gCSI.RMG-I', 'gCSI.NCI-H1703', 'gCSI.KMS-34', 'gCSI.G-361', 'gCSI.EPLC-272H', 'gCSI.HEP-G2', 'gCSI.RERF-LC-MS', 'gCSI.COLO-800', 'gCSI.KM12', 'gCSI.DOHH-2', 'gCSI.EFM-19', 'gCSI.MDA-MB-468', 'gCSI.MHH-ES-1', 'gCSI.IPC-298', 'gCSI.GRANTA-519', 'gCSI.8305C', 'gCSI.KYSE-140', 'gCSI.MALME-3M', 'gCSI.MIA-PACA-2', 'gCSI.NCI-H1666', 'gCSI.PC-3', 'gCSI.RT4', 'gCSI.HUP-T4', 'gCSI.NCI-H1869', 'gCSI.WM-266-4', 'gCSI.KMM-1', 'gCSI.OE33', 'gCSI.SU-DHL-6', 'gCSI.QGP-1', 'gCSI.IGR-37', 'gCSI.VMRC-RCW', 'gCSI.NCI-H1838', 'gCSI.SW948', 'gCSI.COLO-679', 'gCSI.CAL-51', 'gCSI.HUCCT1', 'gCSI.LP-1', 'gCSI.RPMI-7951', 'gCSI.HPAF-II', 'gCSI.OCUM-1', 'gCSI.HOP-92', 'gCSI.NCI-H661', 'gCSI.TOV-112D', 'gCSI.PANC-03-27', 'gCSI.AGS', 'gCSI.HEC-59', 'gCSI.LN-18', 'gCSI.U-87-MG', 'gCSI.U-2-OS', 'gCSI.ABC-1', 'gCSI.IGR-1', 'gCSI.SK-MEL-3', 'gCSI.A549', 'gCSI.HCC4006', 'gCSI.NCI-H1355']
Combined model:
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input.cell.rnaseq (InputLayer)  (None, 942)          0
__________________________________________________________________________________________________
input.drug1.descriptors (InputL (None, 5270)         0
__________________________________________________________________________________________________
input.drug1.fingerprints (Input (None, 2048)         0
__________________________________________________________________________________________________
cell.rnaseq (Model)             (None, 1000)         2945000     input.cell.rnaseq[0][0]
__________________________________________________________________________________________________
drug.descriptors (Model)        (None, 1000)         7273000     input.drug1.descriptors[0][0]
__________________________________________________________________________________________________
drug.fingerprints (Model)       (None, 1000)         4051000     input.drug1.fingerprints[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 3000)         0           cell.rnaseq[1][0]
                                                                 drug.descriptors[1][0]
                                                                 drug.fingerprints[1][0]
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1000)         3001000     concatenate_1[0][0]
__________________________________________________________________________________________________
permanent_dropout_10 (Permanent (None, 1000)         0           dense_10[0][0]
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 1000)         1001000     permanent_dropout_10[0][0]
__________________________________________________________________________________________________
permanent_dropout_11 (Permanent (None, 1000)         0           dense_11[0][0]
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 1000)         1001000     permanent_dropout_11[0][0]
__________________________________________________________________________________________________
permanent_dropout_12 (Permanent (None, 1000)         0           dense_12[0][0]
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 1)            1001        permanent_dropout_12[0][0]
==================================================================================================
Total params: 19,273,001
Trainable params: 19,273,001
Non-trainable params: 0
__________________________________________________________________________________________________
Training homoscedastic model:
partition:train, rank:0, sharded index size:2784, batch_size:32, steps:87
partition:val, rank:0, sharded index size:704, batch_size:32, steps:22
Between random pairs in y_val:
  mse: 0.0604
  mae: 0.1978
  r2: -0.9105
  corr: 0.0447
Data points per epoch: train = 2784, val = 704
Steps per epoch: train = 87, val = 22
Epoch 1/10
87/87 [==============================] - 15s 174ms/step - loss: 0.2165 - mae: 0.2144 - r2: -6.4761 - val_loss: 0.0247 - val_mae: 0.1244 - val_r2: 0.1916
Current time ....15.176
Epoch 2/10
87/87 [==============================] - 12s 142ms/step - loss: 0.0247 - mae: 0.1240 - r2: 0.1302 - val_loss: 0.0208 - val_mae: 0.1147 - val_r2: 0.3058
Current time ....28.323
Epoch 3/10
87/87 [==============================] - 12s 143ms/step - loss: 0.0219 - mae: 0.1157 - r2: 0.2278 - val_loss: 0.0197 - val_mae: 0.1112 - val_r2: 0.3565
Current time ....41.321
Epoch 4/10
87/87 [==============================] - 12s 143ms/step - loss: 0.0203 - mae: 0.1111 - r2: 0.2897 - val_loss: 0.0182 - val_mae: 0.1072 - val_r2: 0.3980
Current time ....54.330
Epoch 5/10
87/87 [==============================] - 13s 153ms/step - loss: 0.0187 - mae: 0.1066 - r2: 0.3388 - val_loss: 0.0189 - val_mae: 0.1090 - val_r2: 0.3804
Current time ....68.120
Epoch 6/10
87/87 [==============================] - 13s 148ms/step - loss: 0.0185 - mae: 0.1075 - r2: 0.3412 - val_loss: 0.0186 - val_mae: 0.1088 - val_r2: 0.3921
Current time ....80.967
Epoch 7/10
87/87 [==============================] - 13s 147ms/step - loss: 0.0185 - mae: 0.1069 - r2: 0.3468 - val_loss: 0.0177 - val_mae: 0.1043 - val_r2: 0.4259
Current time ....93.769
Epoch 8/10
87/87 [==============================] - 13s 150ms/step - loss: 0.0176 - mae: 0.1031 - r2: 0.3791 - val_loss: 0.0159 - val_mae: 0.0994 - val_r2: 0.4793
Current time ....107.421
Epoch 9/10
87/87 [==============================] - 13s 150ms/step - loss: 0.0177 - mae: 0.1034 - r2: 0.3745 - val_loss: 0.0161 - val_mae: 0.1000 - val_r2: 0.4696
Current time ....120.945
Epoch 10/10
87/87 [==============================] - 14s 159ms/step - loss: 0.0169 - mae: 0.1022 - r2: 0.4086 - val_loss: 0.0173 - val_mae: 0.1029 - val_r2: 0.4337
Current time ....134.744
Comparing y_true and y_pred:
  mse: 0.0165
  mae: 0.1016
  r2: 0.4782
  corr: 0.7072
Testing predictions stored in file: save_default/uno.A=relu.B=32.E=10.O=sgd.LS=mse.LR=0.01.CF=r.DF=df.DR=0.1.L1000.D1=1000.D2=1000.D3=1000.predicted.tsv
Model stored in file: save_default/uno.A=relu.B=32.E=10.O=sgd.LS=mse.LR=0.01.CF=r.DF=df.DR=0.1.L1000.D1=1000.D2=1000.D3=1000.model.json
Model stored in file: save_default/uno.A=relu.B=32.E=10.O=sgd.LS=mse.LR=0.01.CF=r.DF=df.DR=0.1.L1000.D1=1000.D2=1000.D3=1000.model.h5
Model weights stored in file: save_default//default.weights.h5
partition:test, rank:0, sharded index size:0, batch_size:32, steps:0

```



## Infer

The infer script does inference on a trained model, as in the underlying Uno benchmark. This script is able to use a pre-generated file or it can construct the data to do inference if a set of identifiers are provided.



The argument --uq_infer_file must be used to specify the name of the file with the data (or the identifiers) to do inference.



Additionally, if the data needs to be constructed, then one of the following arguments should be used to specify what type of identifiers are provided

- --uq_infer_given_drugs=True     if the file contains a set of drug IDs.

- --uq_infer_given_cells=True       if the file contains a set of cell IDs.

- --uq_infer_given_indices=True   if the file contains a set of indices.



Note that the latter works if all the arguments for the data construction are set as well (usually those are taken from the model configuration file). Of course this specification and the trained model should be consistent for the script to work.



Likewise, in the case that a pre-generated file is provided, the features included and the trained model should be consistent for the script to work.



Note also that the --loss 'het' or --loss 'qtl' option should be specified if the model was trained to predict as well the heterogeneous noise, or the quantile distributions.



#### Example output

This assumes that a trained model (files saved.model.h5 and saved.weights.h5) is available at save_default folder. These files are usually generated when running the training and using the --cp True option. Both, in combination, can be used for testing the inference demo script and would produce a similar output to the one shown next.

```

python uno_inferUQ_keras2.py  --uq_infer_file save_default/infer_cell_ids --uq_infer_given_cells True --model_file save_default/saved.model.h5 --weights_file save_default/saved.weights.h5 --n_pred 10
Using TensorFlow backend.
Importing candle utils for keras
Params:
{'activation': 'relu',
 'agg_dose': 'AUC',
 'base_lr': None,
 'batch_normalization': False,
 'batch_size': 32,
 'by_cell': None,
 'by_drug': None,
 'cache': None,
 'cell_feature_subset_path': '',
 'cell_features': ['rnaseq'],
 'cell_subset_path': '',
 'cell_types': None,
 'cp': False,
 'cv': 1,
 'data_type': <class 'numpy.float32'>,
 'dense': [1000, 1000, 1000],
 'dense_feature_layers': [1000, 1000, 1000],
 'dropout': 0.1,
 'drug_feature_subset_path': '',
 'drug_features': ['descriptors', 'fingerprints'],
 'drug_median_response_max': 1,
 'drug_median_response_min': -1,
 'drug_subset_path': '',
 'epochs': 10,
 'exclude_cells': [],
 'exclude_drugs': [],
 'experiment_id': 'EXP000',
 'export_csv': None,
 'export_data': None,
 'feature_subsample': 0,
 'feature_subset_path': '',
 'gpus': [],
 'growth_bins': 0,
 'initial_weights': None,
 'learning_rate': 0.01,
 'logfile': None,
 'loss': 'mse',
 'max_val_loss': 1.0,
 'model_file': 'save_default/saved.model.h5',
 'n_pred': 10,
 'no_feature_source': True,
 'no_gen': False,
 'no_response_source': True,
 'optimizer': 'sgd',
 'output_dir': './Output/EXP000/RUN000',
 'partition_by': 'cell',
 'preprocess_rnaseq': 'none',
 'profiling': False
 'reduce_lr': False,
 'residual': False,
 'rng_seed': 2018,
 'run_id': 'RUN000',
 'sample_repetition': False,
 'save_path': 'save_default/',
 'save_weights': None,
 'scaling': 'std',
 'shuffle': False,
 'single': True,
 'tb': False,
 'tb_prefix': 'tb',
 'test_sources': ['train'],
 'timeout': 3600,
 'train_bool': True,
 'train_sources': ['gCSI'],
 'uq_infer_file': 'save_default/infer_cell_ids',
 'uq_infer_given_cells': True,
 'uq_infer_given_drugs': False,
 'uq_infer_given_indices': False,
 'use_exported_data': None,
 'use_filtered_genes': False,
 'use_landmark_genes': True,
 'val_split': 0.2,
 'verbose': None,
 'warmup_lr': False,
 'weights_file': 'save_default/saved.weights.h5'}
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input.cell.rnaseq (InputLayer)  (None, 942)          0
__________________________________________________________________________________________________
input.drug1.descriptors (InputL (None, 5270)         0
__________________________________________________________________________________________________
input.drug1.fingerprints (Input (None, 2048)         0
__________________________________________________________________________________________________
cell.rnaseq (Model)             (None, 1000)         2945000     input.cell.rnaseq[0][0]
__________________________________________________________________________________________________
drug.descriptors (Model)        (None, 1000)         7273000     input.drug1.descriptors[0][0]
__________________________________________________________________________________________________
drug.fingerprints (Model)       (None, 1000)         4051000     input.drug1.fingerprints[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 3000)         0           cell.rnaseq[1][0]
                                                                 drug.descriptors[1][0]
                                                                 drug.fingerprints[1][0]
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1000)         3001000     concatenate_1[0][0]
__________________________________________________________________________________________________
permanent_dropout_10 (Permanent (None, 1000)         0           dense_10[0][0]
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 1000)         1001000     permanent_dropout_10[0][0]
__________________________________________________________________________________________________
permanent_dropout_11 (Permanent (None, 1000)         0           dense_11[0][0]
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 1000)         1001000     permanent_dropout_11[0][0]
__________________________________________________________________________________________________
permanent_dropout_12 (Permanent (None, 1000)         0           dense_12[0][0]
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 1)            1001        permanent_dropout_12[0][0]
==================================================================================================
Total params: 19,273,001
Trainable params: 19,273,001
Non-trainable params: 0
__________________________________________________________________________________________________
partition:test, rank:0, sharded index size:0, batch_size:32, steps:0
Read file: save_default/infer_cell_ids
Number of elements read: 72
Comparing y_true and y_pred:
  mse: 0.0173
  mae: 0.1012
  r2: 0.4687
  corr: 0.7001
Comparing y_true and y_pred:
  mse: 0.0172
  mae: 0.1005
  r2: 0.4720
  corr: 0.7010
Comparing y_true and y_pred:
  mse: 0.0171
  mae: 0.1033
  r2: 0.4751
  corr: 0.7064
Comparing y_true and y_pred:
  mse: 0.0175
  mae: 0.1045
  r2: 0.4627
  corr: 0.6945
Comparing y_true and y_pred:
  mse: 0.0162
  mae: 0.1007
  r2: 0.5017
  corr: 0.7277
Comparing y_true and y_pred:
  mse: 0.0166
  mae: 0.1008
  r2: 0.4921
  corr: 0.7141
Comparing y_true and y_pred:
  mse: 0.0181
  mae: 0.1059
  r2: 0.4443
  corr: 0.6878
Comparing y_true and y_pred:
  mse: 0.0167
  mae: 0.1015
  r2: 0.4875
  corr: 0.7087
Comparing y_true and y_pred:
  mse: 0.0169
  mae: 0.1032
  r2: 0.4805
  corr: 0.7106
Comparing y_true and y_pred:
  mse: 0.0169
  mae: 0.0999
  r2: 0.4817
  corr: 0.7075
Predictions stored in file: save_default/uno.A=relu.B=32.E=10.O=sgd.LS=mse.LR=None.CF=r.DF=df.DR=0.1.L1000.D1=1000.D2=1000.D3=1000.predicted_INFER.tsv
```



## Empirical Calibration

Scripts included in the calibration subfolder compute empirical calibration for the inference results. The scripts with suffix HOM compute empirical calibration for inference with homoscedastic model, while the script with suffix HET computes empirical calibration for inference with a heteroscedastic model.



To run the scripts it is necessary to provide the path to the file and the file with the inference results. Note that it is assumed that the file with the inference results includes each realization of the inference (implicit in the 'all' suffix), but for the homoscedastic case a script is provided to process an inference file with only the consolidated statistics (generally the average over all the realizations). Also, note that a specific format of the file with the inference results is assumed. Thus, a set of default values, reflecting the format of current CANDLE infer scripts, is used. More arbitrary formats may be usable, if they incurr in similar column offsets, but it would require passing the right parameters to the function reading the inference file.



The script generates a series of plots and pickle (dill) files, displaying and encoding the empirical calibration computed.

