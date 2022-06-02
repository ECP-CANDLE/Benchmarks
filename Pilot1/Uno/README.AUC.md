# Predicting AUC values for Top21 cancer types

## Data prep
A static dataset is prebuilt and available at `http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5`.

```
$ wget http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5
```


## Training
```
python uno_baseline_keras2.py --config_file uno_auc_model.txt \
  --use_exported_data top_21_auc_1fold.uno.h5 --es True

...
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
 'config_file': 'uno_auc_model.txt',
 'cp': True,
 'cv': 1,
 'datatype': <class 'numpy.float32'>,
 'dense': [1000, 1000, 1000, 1000, 1000],
 'dense_cell_feature_layers': None,
 'dense_drug_feature_layers': None,
 'dense_feature_layers': [1000, 1000, 1000],
 'drop': 0.1,
 'drug_feature_subset_path': '',
 'drug_features': ['descriptors'],
 'drug_median_response_max': 1,
 'drug_median_response_min': -1,
 'drug_subset_path': '',
 'epochs': 50,
 'es': True,
 'experiment_id': 'EXP000',
 'export_csv': None,
 'export_data': None,
 'feature_subsample': 0,
 'feature_subset_path': '',
 'gpus': [],
 'growth_bins': 0,
 'initial_weights': None,
 'learning_rate': 0.0001,
 'logfile': None,
 'loss': 'mse',
 'max_val_loss': 1.0,
 'no_feature_source': True,
 'no_gen': False,
 'no_response_source': True,
 'optimizer': 'adamax',
 'output_dir': '/ssd1/homes/hsyoo/projects/CANDLE/Benchmarks/Pilot1/Uno/Output/EXP000/RUN000',
 'partition_by': None,
 'preprocess_rnaseq': 'source_scale',
 'profiling': False,
 'reduce_lr': True,
 'residual': False,
 'rng_seed': 2018,
 'run_id': 'RUN000',
 'save_path': 'save/uno',
 'save_weights': None,
 'scaling': 'std',
 'shuffle': False,
 'single': True,
 'tb': False,
 'tb_prefix': 'tb',
 'test_sources': ['train'],
 'timeout': -1,
 'train_bool': True,
 'train_sources': ['CCLE'],
 'use_exported_data': 'top_21_auc_1fold.uno.h5',
 'use_filtered_genes': False,
 'use_landmark_genes': True,
 'validation_split': 0.2,
 'verbose': None,
 'warmup_lr': True}

 ...
Total params: 16,224,001
Trainable params: 16,224,001
Non-trainable params: 0
...
Between random pairs in y_val:
  mse: 0.0474
  mae: 0.1619
  r2: -1.0103
  corr: -0.0051
Data points per epoch: train = 423952, val = 52994, test = 52994
Steps per epoch: train = 13248, val = 1656, test = 1656
Epoch 1/50
13248/13248 [==============================] - 102s 8ms/step - loss: 0.0268 - mae: 0.0794 - r2: -0.2754 - val_loss: 0.0092 - val_mae: 0.0725 - val_r2: 0.5657
Current time ....101.892
...
13248/13248 [==============================] - 102s 8ms/step - loss: 0.004572, lr: 0.000010, mae: 0.046159, r2: 0.782253, val_loss: 0.005335, val_mae: 0.049082, val_r2: 0.748585
Comparing y_true and y_pred:
   mse: 0.0053
   mae: 0.0490
   r2: 0.7742
   corr: 0.8800
```


## Inference
The script `uno_infer.py` takes a couple of parameters for inferences. You are required to specify a datafile (the same dataset for training, `top_21_auc_1fold.uno.h5` in this case), model file, and trained weights. You can choose a partition as a inference input (training, validation, or all) and number of predictions for each data points (-n).
```
$ python uno_infer.py --data top_21_auc_1fold.uno.h5 \
  --model_file top21_ref/model.json \
  --weights_file top21_ref/weights.h5 \
  --partition val \
  -n 30 \
  --single True \
  --agg_dose AUC
...
  mse: 0.0058
  mae: 0.0505
  r2: 0.7543
  corr: 0.8688
     mean    std    min    max
mse: 0.0058, 0.0000, 0.0058, 0.0058
mae: 0.0505, 0.0001, 0.0504, 0.0506
r2: 0.7543, 0.0007, 0.7527, 0.7557
corr: 0.8688, 0.0004, 0.8679, 0.8696
```

After the inference script completes, you should be able to find `uno_pred.all.tsv` and `uno_pred.tsv` files, which contains all predicted value and error, and aggregated statistics for each data point respectively. See below for example,
```
$ head -n 4 uno_pred.all.tsv
AUC	Sample	Drug1	PredictedAUC	AUCError
0.7153	CCLE.22RV1	CCLE.1	0.726853	0.011553
0.7153	CCLE.22RV1	CCLE.1	0.745033	0.0297334
0.7153	CCLE.22RV1	CCLE.1	0.752899	0.0375985

$ head -n 4 uno_pred.tsv
AUC	Sample	Drug1	PredAUCMean	PredAUCStd	PredAUCMin	PredAUCMax
0.918	CTRP.HCC-1438	CTRP.302	0.954987	0.0109111	0.938283	0.983576
0.6474	NCI60.IGR-OV1	NSC.757440	0.680934	0.0279046	0.644829	0.755912
0.5675	NCI60.CCRF-CEM	NSC.381866	0.591151	0.0228838	0.553855	0.645553
```
