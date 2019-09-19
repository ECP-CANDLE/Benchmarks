# Training with static datafile
Use static datafile prebuilt and shared at `/vol/ml/hsyoo/shared/top_21_auc_1fold.uno.h5`

```
python uno_baseline_keras2.py --config_file uno_auc_model.txt --cache cache/top6_auc --use_exported_data /vol/ml/hsyoo/shared/top_21_auc_1fold.uno.h5
```

The log will look like below,

```
Using TensorFlow backend.
Importing candle utils for keras
Configuration file:  /ssd1/homes/hsyoo/projects/CANDLE/Benchmarks/Pilot1/Uno/uno_auc_model.txt
{'activation': 'relu',
 'agg_dose': 'AUC',
 'base_lr': None,
 'batch_normalization': False,
 'batch_size': 32,
 'cell_features': ['rnaseq'],
 'cell_types': None,
 'cp': True,
 'cv': 1,
 'dense': [1000, 1000, 1000, 1000, 1000],
 'dense_feature_layers': [1000, 1000, 1000],
 'drop': 0.1,
 'drug_features': ['descriptors'],
 'epochs': 50,
 'feature_subsample': 0,
 'gpus': 1,
 'learning_rate': 0.0001,
 'loss': 'mse',
 'max_val_loss': 1.0,
 'no_feature_source': True,
 'no_gen': False,
 'no_response_source': True,
 'optimizer': 'adamax',
 'preprocess_rnaseq': 'source_scale',
 'reduce_lr': True,
 'residual': False,
 'rng_seed': 2018,
 'save_path': 'save/uno',
 'scaling': 'std',
 'single': True,
 'solr_root': '',
 'test_sources': ['train'],
 'timeout': -1,
 'train_sources': ['CCLE'],
 'use_landmark_genes': True,
 'validation_split': 0.2,
 'verbose': False,
 'warmup_lr': True}
Params:
{'activation': 'relu',
 'agg_dose': 'AUC',
 'base_lr': None,
 'batch_normalization': False,
 'batch_size': 32,
 'by_cell': None,
 'by_drug': None,
 'cache': 'cache/top6_auc',
 'cell_feature_subset_path': '',
 'cell_features': ['rnaseq'],
 'cell_subset_path': '',
 'cell_types': None,
 'config_file': 'uno_auc_model.txt',
 'cp': True,
 'cv': 1,
 'datatype': <class 'numpy.float32'>,
 'dense': [1000, 1000, 1000, 1000, 1000],
 'dense_feature_layers': [1000, 1000, 1000],
 'drop': 0.1,
 'drug_feature_subset_path': '',
 'drug_features': ['descriptors'],
 'drug_median_response_max': 1,
 'drug_median_response_min': -1,
 'drug_subset_path': '',
 'epochs': 50,
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
 'reduce_lr': True,
 'residual': False,
 'rng_seed': 2018,
 'run_id': 'RUN000',
 'save_path': 'save/uno',
 'save_weights': None,
 'scaling': 'std',
 'shuffle': False,
 'single': True,
 'solr_root': '',
 'tb': False,
 'tb_prefix': 'tb',
 'test_sources': ['train'],
 'timeout': -1,
 'train_bool': True,
 'train_sources': ['CCLE'],
 'use_exported_data': '/vol/ml/hsyoo/shared/top_21_auc_1fold.uno.h5',
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
Data points per epoch: train = 423952, val = 52994
Steps per epoch: train = 13248, val = 1656
Epoch 1/50
13248/13248 [==============================] - 198s 15ms/step - loss: 0.0235 - mae: 0.1048 - r2: -0.1311 - val_loss: 0.0145 - val_mae: 0.0903 - val_r2: 0.3393
Current time ....198.278
Epoch 2/50
...
```
