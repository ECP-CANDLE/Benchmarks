## Uno: Predicting Tumor Dose Response across Multiple Data Sources

#### Example output
Uno can be trained with a subset of dose response data sources. Here is an command line example of training with all 6 sources: CCLE, CTRP, gCSI, GDSC, NCI60 single drug response, ALMANAC drug pair response.

```
uno_baseline_keras2.py --train_sources all --cache cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True
Using TensorFlow backend.
Params: {'activation': 'relu', 'batch_size': 32, 'dense': [1000, 1000, 1000], 'dense_feature_layers': [1000, 1000, 1000], 'drop': 0, 'epochs': 10, 'learning_rate': None, 'loss':
'mse', 'optimizer': 'adam', 'residual': False, 'rng_seed': 2018, 'save': 'save/uno', 'scaling': 'std', 'feature_subsample': 0, 'validation_split': 0.2, 'solr_root': '', 'timeout'
: -1, 'train_sources': ['all'], 'test_sources': ['train'], 'cell_types': None, 'cell_features': ['rnaseq'], 'drug_features': ['descriptors', 'fingerprints'], 'cv': 1, 'max_val_lo
ss': 1.0, 'base_lr': None, 'reduce_lr': False, 'warmup_lr': False, 'batch_normalization': False, 'no_gen': False, 'config_file': '/raid/fangfang/Benchmarks/Pilot1/Uno/uno_default
_model.txt', 'verbose': False, 'logfile': None, 'train_bool': True, 'shuffle': True, 'alpha_dropout': False, 'gpus': [], 'experiment_id': 'EXP.000', 'run_id': 'RUN.000', 'by_cell
': None, 'by_drug': None, 'drug_median_response_min': -1, 'drug_median_response_max': 1, 'no_feature_source': True, 'no_response_source': True, 'use_landmark_genes': True, 'use_f
iltered_genes': False, 'preprocess_rnaseq': 'source_scale', 'cp': False, 'tb': False, 'partition_by': None, 'cache': 'cache/ALL', 'single': False, 'export_data': None, 'growth_bi
ns': 0, 'datatype': <class 'numpy.float32'>}
Cache parameter file does not exist: cache/ALL.params.json
Loading data from scratch ...
Loaded 27769716 single drug dose response measurements
Loaded 3686475 drug pair dose response measurements
Combined dose response data contains sources: ['CCLE' 'CTRP' 'gCSI' 'GDSC' 'NCI60' 'SCL' 'SCLC' 'ALMANAC.FG'
 'ALMANAC.FF' 'ALMANAC.1A']
Summary of combined dose response by source:
              Growth  Sample  Drug1  Drug2  MedianDose
Source
ALMANAC.1A    208605      60    102    102    7.000000
ALMANAC.FF   2062098      60     92     71    6.698970
ALMANAC.FG   1415772      60    100     29    6.522879
CCLE           93251     504     24      0    6.602060
CTRP         6171005     887    544      0    6.585027
GDSC         1894212    1075    249      0    6.505150
NCI60       18862308      59  52671      0    6.000000
SCL           301336      65    445      0    6.908485
SCLC          389510      70    526      0    6.908485
gCSI           58094     409     16      0    7.430334
Combined raw dose response data has 3070 unique samples and 53520 unique drugs
Limiting drugs to those with response min <= 1, max >= -1, span >= 0, median_min <= -1, median_max >= 1 ...
Selected 47005 drugs from 53520
Loaded combined RNAseq data: (15198, 943)
Loaded combined dragon7 drug descriptors: (53507, 5271)
Loaded combined dragon7 drug fingerprints: (53507, 2049)
Filtering drug response data...
  2375 molecular samples with feature and response data
  46837 selected drugs with feature and response data
Summary of filtered dose response by source:
              Growth  Sample  Drug1  Drug2  MedianDose
Source
ALMANAC.1A    206580      60    101    101    7.000000
ALMANAC.FF   2062098      60     92     71    6.698970
ALMANAC.FG   1293465      60     98     27    6.522879
CCLE           80213     474     22      0    6.602060
CTRP         3397103     812    311      0    6.585027
GDSC         1022204     672    213      0    6.505150
NCI60       17190561      59  46272      0    6.000000
gCSI           50822     357     16      0    7.430334
Grouped response data by drug_pair: 51763 groups
Input features shapes:
  dose1: (1,)
  dose2: (1,)
  cell.rnaseq: (942,)
  drug1.descriptors: (5270,)
  drug1.fingerprints: (2048,)
  drug2.descriptors: (5270,)
  drug2.fingerprints: (2048,)
Total input dimensions: 15580
Saved data to cache: cache/all.pkl
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
input.drug2.descriptors (InputL (None, 5270)         0
__________________________________________________________________________________________________
input.drug2.fingerprints (Input (None, 2048)         0
__________________________________________________________________________________________________
input.dose1 (InputLayer)        (None, 1)            0
__________________________________________________________________________________________________
input.dose2 (InputLayer)        (None, 1)            0
__________________________________________________________________________________________________
cell.rnaseq (Model)             (None, 1000)         2945000     input.cell.rnaseq[0][0]
__________________________________________________________________________________________________
drug.descriptors (Model)        (None, 1000)         7273000     input.drug1.descriptors[0][0]
                                                                 input.drug2.descriptors[0][0]
__________________________________________________________________________________________________
drug.fingerprints (Model)       (None, 1000)         4051000     input.drug1.fingerprints[0][0]
                                                                 input.drug2.fingerprints[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 5002)         0           input.dose1[0][0]
                                                                 input.dose2[0][0]
                                                                 cell.rnaseq[1][0]
                                                                 drug.descriptors[1][0]
                                                                 drug.fingerprints[1][0]
                                                                 drug.descriptors[2][0]
                                                                 drug.fingerprints[2][0]
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1000)         5003000     concatenate_1[0][0]
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 1000)         1001000     dense_10[0][0]
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 1000)         1001000     dense_11[0][0]
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 1)            1001        dense_12[0][0]
==================================================================================================
Total params: 21,275,001
Trainable params: 21,275,001
Non-trainable params: 0
__________________________________________________________________________________________________
Between random pairs in y_val:
  mse: 0.6069
  mae: 0.5458
  r2: -0.9998
  corr: 0.0001
Data points per epoch: train = 20158325, val = 5144721
Steps per epoch: train = 629948, val = 160773
Epoch 1/10
629948/629948 [==============================] - 196053s 311ms/step - loss: 0.0993 - mae: 0.2029 - r2: 0.6316 - val_loss: 0.1473 - val_mae: 0.2404 - val_r2: 0.4770
Current time ....196052.671
Epoch 2/10
629948/629948 [==============================] - 194858s 309ms/step - loss: 0.0872 - mae: 0.1890 - r2: 0.6755 - val_loss: 0.1469 - val_mae: 0.2393 - val_r2: 0.4771
Current time ....390911.212
Epoch 3/10
629948/629948 [==============================] - 192603s 306ms/step - loss: 0.0848 - mae: 0.1861 - r2: 0.6840 - val_loss: 0.1486 - val_mae: 0.2409 - val_r2: 0.4720
Current time ....583514.913
Epoch 4/10
629948/629948 [==============================] - 192734s 306ms/step - loss: 0.0836 - mae: 0.1846 - r2: 0.6885 - val_loss: 0.1500 - val_mae: 0.2417 - val_r2: 0.4657
Current time ....776248.738
Epoch 5/10
629948/629948 [==============================] - 190948s 303ms/step - loss: 0.0829 - mae: 0.1836 - r2: 0.6912 - val_loss: 0.1498 - val_mae: 0.2412 - val_r2: 0.4678
Current time ....967196.253
Epoch 6/10
629948/629948 [==============================] - 191344s 304ms/step - loss: 0.0824 - mae: 0.1829 - r2: 0.6931 - val_loss: 0.1506 - val_mae: 0.2417 - val_r2: 0.4631
Current time ....1158540.613
Epoch 7/10
629948/629948 [==============================] - 195056s 310ms/step - loss: 0.0820 - mae: 0.1824 - r2: 0.6945 - val_loss: 0.1518 - val_mae: 0.2431 - val_r2: 0.4596
Current time ....1353596.930
Epoch 8/10
629948/629948 [==============================] - 193873s 308ms/step - loss: 0.0817 - mae: 0.1820 - r2: 0.6956 - val_loss: 0.1525 - val_mae: 0.2428 - val_r2: 0.4570
Current time ....1547470.041
Epoch 9/10
629948/629948 [==============================] - 191701s 304ms/step - loss: 0.0815 - mae: 0.1818 - r2: 0.6963 - val_loss: 0.1525 - val_mae: 0.2434 - val_r2: 0.4593
Current time ....1739170.656
Epoch 10/10
629948/629948 [==============================] - 194420s 309ms/step - loss: 0.0813 - mae: 0.1815 - r2: 0.6971 - val_loss: 0.1528 - val_mae: 0.2432 - val_r2: 0.4600
Current time ....1933590.940
Comparing y_true and y_pred:
  mse: 0.1528
  mae: 0.2432
  r2: 0.4966
  corr: 0.7077
```

Training Uno on all data sources is slow. The `--train_sources` parameter can be used to test the code with a smaller set of training data. An example command line is the following.
```
uno_baseline_keras2.py --train_sources CCLE --cache cache/CCLE --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True
```

A faster example is given in the `uno_by_drug_example.txt` configuration file. This example focuses on a single drug (paclitaxel) and trains at 15s/epoch on a single P100.
```
uno_baseline_keras2.py --config_file uno_by_drug_example.txt
```

## Profile runs
We have run the same configuration across multiple machines and compared the resource utilization. 
```
python uno_baseline_keras2.py --conf uno_perf_benchmark.txt
```

| Machine | Time to complete (HH:mm:ss) | Time per epoch (s) | Perf factor <sup>*</sup> | CPU % | Mem % | Mem GB | GPU % | GPU Mem % | Note |
| ------- | --------------------------: | -----------------: | -----------------------: | ----: | ----: | -----: | ----: | --------: | ---- |
| Theta   | 2:26:10 | 3268 | 0.20 | 1.1 | 5.9 | 9.6| | | |
| Nucleus | 0:32:11 | 518 | 1.23 | 39.1 | 12.7 | 30.6 | 2.1 | 4.8 | |
| Tesla (K20) | 0:43:21 | 638 | 1.00 | 35.5 | 31.8 | 9.6 | 8.9 | 6.5 | |
| Titan | | | | | | | | |keras version 2.0.3 does not supprot model.clone_model() which is introduced in 2.0.7
* Time per epoch on the machine divided by time per epoch of Titan (or Tesla)

## Training and Inferencing Uno with Pre-staged Datasets
We can expedite the training and inferencing using pre-staged dataset file. You may need to regenerate the files for a different combination of parameters, which are relevant to the data processing such as preprocess_rnaseq, train_sources, cell_feature_subset, etc. but you don't need to for training related params (i.e., batch_size, number of epochs, etc.)

1. Generate pre-staged dataset file. Use `--export_data` to specify the file name and use a large batch size to speed up.
```
python uno_baseline_keras2.py --train_sources all --cache cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z 4096 --export_data All.h5 --shuffle True

Using TensorFlow backend.
...
partition:train, rank:0, sharded index size:20156416, batch_size:4096, steps:4921, total_samples:20158325
partition:val, rank:0, sharded index size:5144576, batch_size:4096, steps:1256, total_samples:5144721
Generating train dataset. 0 / 4921
Generating train dataset. 1 / 4921
..
Generating train dataset. 4919 / 4921
Generating train dataset. 4920 / 4921
Generating val dataset. 0 / 1256
Generating val dataset. 1 / 1256
..
Generating val dataset. 1254 / 1256
Generating val dataset. 1255 / 1256
Completed generating All.h5
```
This took ~3 hours.

2. Training with pre-staged dataset. Use `--use_exported_data` to point dataset file.
```
python uno_baseline_keras2.py --train_sources all --cache cache/all \
--use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True \
-z 512 --use_exported_data All.h5 --cp True --shuffle True --tb True
Using TensorFlow backend.
...
Total params: 21,275,001
Trainable params: 21,275,001
Non-trainable params: 0
__________________________________________________________________________________________________
Between random pairs in y_val:
  mse: 0.6070
  mae: 0.5458
  r2: -1.0002
  corr: -0.0001
Data points per epoch: train = 20156416, val = 5144576
Steps per epoch: train = 39368, val = 10048
Epoch 1/10
39368/39368 [==============================] - 2530s 64ms/step - loss: 0.0928 - mae: 0.1893 - r2: 0.6876 - val_loss: 0.1316 - val_mae: 0.2259 - val_r2: 0.5644
Current time ....2529.994
Epoch 2/10
39368/39368 [==============================] - 2329s 59ms/step - loss: 0.0698 - mae: 0.1689 - r2: 0.7645 - val_loss: 0.1283 - val_mae: 0.2212 - val_r2: 0.5753
Current time ....4858.745
Epoch 3/10
39368/39368 [==============================] - 2314s 59ms/step - loss: 0.0669 - mae: 0.1652 - r2: 0.7740 - val_loss: 0.1339 - val_mae: 0.2214 - val_r2: 0.5569
Current time ....7172.407
Epoch 4/10
39368/39368 [==============================] - 2308s 59ms/step - loss: 0.0657 - mae: 0.1634 - r2: 0.7784 - val_loss: 0.1333 - val_mae: 0.2233 - val_r2: 0.5584
Current time ....9480.256
Epoch 5/10
39368/39368 [==============================] - 2328s 59ms/step - loss: 0.0651 - mae: 0.1627 - r2: 0.7802 - val_loss: 0.1348 - val_mae: 0.2216 - val_r2: 0.5540
Current time ....11808.681
Epoch 6/10
39368/39368 [==============================] - 2323s 59ms/step - loss: 0.0640 - mae: 0.1612 - r2: 0.7839 - val_loss: 0.1335 - val_mae: 0.2217 - val_r2: 0.5580
Current time ....14131.437
Epoch 7/10
39368/39368 [==============================] - 2335s 59ms/step - loss: 0.0634 - mae: 0.1603 - r2: 0.7860 - val_loss: 0.1331 - val_mae: 0.2215 - val_r2: 0.5596
Current time ....16466.285
Epoch 8/10
39368/39368 [==============================] - 2327s 59ms/step - loss: 0.0630 - mae: 0.1597 - r2: 0.7873 - val_loss: 0.1313 - val_mae: 0.2193 - val_r2: 0.5654
Current time ....18793.203
Epoch 9/10
39368/39368 [==============================] - 2347s 60ms/step - loss: 0.0627 - mae: 0.1593 - r2: 0.7884 - val_loss: 0.1360 - val_mae: 0.2212 - val_r2: 0.5501
Current time ....21140.630
Epoch 10/10
39368/39368 [==============================] - 2372s 60ms/step - loss: 0.0624 - mae: 0.1588 - r2: 0.7895 - val_loss: 0.1352 - val_mae: 0.2216 - val_r2: 0.5524
Current time ....23512.750

```

3. Inferencing with pre-staged dataset.
```
python uno_infer.py --data All.h5 --model_file model.h5 --n_pred 30
```
