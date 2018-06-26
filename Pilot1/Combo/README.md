## Combo: Predicting Tumor Cell Line Response to Drug Pairs

**Overview**: Given combination drug screening results on NCI60 cell lines available at the NCI-ALMANAC database, build a deep learning network that can predict the growth percentage from the cell line molecular features and the descriptors of both drugs.

**Relationship to core problem**: This benchmark is an example one of the core capabilities needed for the Pilot 1 Drug Response problem: combining multiple molecular assays and drug descriptors in a single deep learning framework for response prediction.

**Expected outcome**: Build a DNN that can predict growth percentage of a cell line treated with a pair of drugs.

### Benchmark Specs Requirements

#### Description of the Data
* Data source: Combo drug response screening results from NCI-ALMANAC; 5-platform normalized expression, microRNA expression, and proteome abundance data from the NCI; Dragon7 generated drug descriptors based on 2D chemical structures from NCI
* Input dimensions: ~30K with default options: 26K normalized expression levels by gene + 4K drug descriptors; 59 cell lines; a subset of 54 FDA-approved drugs
Output dimensions: 1 (growth percentage)
* Sample size: 85,303 (cell line, drug 1, drug 2) tuples from the original 304,549 in the NCI-ALMANAC database
* Notes on data balance: there are more ineffective drug pairs than effective pairs; data imbalance is somewhat reduced by using only the best dose combination for each (cell line, drug 1, drug 2) tuple as training and validation data

#### Expected Outcomes
* Regression. Predict percent growth for any NCI-60 cell line and drugs combination
* Dimension: 1 scalar value corresponding to the percent growth for a given drug concentration. Output range: [-100, 100]

#### Evaluation Metrics
* Accuracy or loss function: mean squared error, mean absolute error, and R^2
* Expected performance of a naïve method: mean response, linear regression or random forest regression.

#### Description of the Network
* Proposed network architecture: two-stage neural network that is jointly trained for feature encoding and response prediction; shared submodel for each drug in the pair
* Number of layers: 3-4 layers for feature encoding submodels and response prediction submodels, respectively

![Combo model architecture](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/frameworks/Pilot1/Combo/figs/network-small.png "Combo model network architecture")

### Running the baseline implementation

```
$ cd Pilot1/Combo
$ python combo_baseline_keras2.py
```

#### Example output
```
$ python combo_baseline_keras2.py --use_landmark_genes --warmup_lr --reduce_lr -z 128 --epochs 10                              [ /raid/fangfang/Benchmarks/Pilot1/Combo ]

Using TensorFlow backend.
Params: {'activation': 'relu', 'batch_size': 128, 'dense': [1000, 1000, 1000], 'dense_feature_layers': [1000, 1000, 1000], 'drop': 0, 'epochs': 10, 'learning_rate': None, 'loss':
 'mse', 'optimizer': 'adam', 'residual': False, 'rng_seed': 2017, 'save': 'save/combo', 'scaling': 'std', 'feature_subsample': 0, 'validation_split': 0.2, 'solr_root': '', 'timeo
ut': -1, 'cell_features': ['expression'], 'drug_features': ['descriptors'], 'cv': 1, 'cv_partition': 'overlapping', 'max_val_loss': 1.0, 'base_lr': None, 'reduce_lr': True, 'warm
up_lr': True, 'batch_normalization': False, 'gen': False, 'use_combo_score': False, 'config_file': '/raid/fangfang/Benchmarks/Pilot1/Combo/combo_default_model.txt', 'verbose': False, 'logfile': None, 'train_bool': True, 'shuffle': True, 'alpha_dropout': False, 'gpus': [], 'experiment_id': 'EXP.000', 'run_id': 'RUN.000', 'use_landmark_genes': True, 'preprocess_rnaseq': None, 'response_url': None, 'cp': False, 'tb': False, 'exclude_cells': [], 'exclude_drugs': [], 'datatype': <class 'numpy.float32'>}
Loaded 317899 unique (CL, D1, D2) response sets.
Filtered down to 310820 rows with matching information.
Unique cell lines: 60
Unique drugs: 104
Distribution of dose response:
              GROWTH
count  310820.000000
mean        0.315812
std         0.530731
min        -1.000000
25%         0.042420
50%         0.398400
75%         0.765842
max         1.693300
Rows in train: 248656, val: 62164
Input features shapes:
  cell.expression: (942,)
  drug1.descriptors: (3820,)
  drug2.descriptors: (3820,)
Total input dimensions: 8582
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 942)               0
_________________________________________________________________
dense_1 (Dense)              (None, 1000)              943000
_________________________________________________________________
dense_2 (Dense)              (None, 1000)              1001000
_________________________________________________________________
dense_3 (Dense)              (None, 1000)              1001000
=================================================================
Total params: 2,945,000
Trainable params: 2,945,000
Non-trainable params: 0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_2 (InputLayer)         (None, 3820)              0
_________________________________________________________________
dense_4 (Dense)              (None, 1000)              3821000
_________________________________________________________________
dense_5 (Dense)              (None, 1000)              1001000
_________________________________________________________________
dense_6 (Dense)              (None, 1000)              1001000
=================================================================
Total params: 5,823,000
Trainable params: 5,823,000
Non-trainable params: 0
_________________________________________________________________
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input.cell.expression (InputLay (None, 942)          0
__________________________________________________________________________________________________
input.drug1.descriptors (InputL (None, 3820)         0
__________________________________________________________________________________________________
input.drug2.descriptors (InputL (None, 3820)         0
__________________________________________________________________________________________________
cell.expression (Model)         (None, 1000)         2945000     input.cell.expression[0][0]
__________________________________________________________________________________________________
drug.descriptors (Model)        (None, 1000)         5823000     input.drug1.descriptors[0][0]
                                                                 input.drug2.descriptors[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 3000)         0           cell.expression[1][0]
                                                                 drug.descriptors[1][0]
                                                                 drug.descriptors[2][0]
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 1000)         3001000     concatenate_1[0][0]
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 1000)         1001000     dense_7[0][0]
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 1000)         1001000     dense_8[0][0]
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1)            1001        dense_9[0][0]
==================================================================================================
Total params: 13,772,001
Trainable params: 13,772,001
Non-trainable params: 0
__________________________________________________________________________________________________
Between random pairs in y_val:
  mse: 0.5654
  mae: 0.5928
  r2: -1.0105
  corr: -0.0052
Train on 248656 samples, validate on 62164 samples
Epoch 1/10
248656/248656 [==============================] - 25s 101us/step - loss: 0.1535 - mae: 0.2450 - r2: 0.4573 - val_loss: 0.0730 - val_mae: 0.1911 - val_r2: 0.7356
Epoch 2/10
248656/248656 [==============================] - 25s 99us/step - loss: 0.0622 - mae: 0.1791 - r2: 0.7754 - val_loss: 0.0535 - val_mae: 0.1631 - val_r2: 0.8064
Epoch 3/10
248656/248656 [==============================] - 25s 100us/step - loss: 0.0479 - mae: 0.1565 - r2: 0.8269 - val_loss: 0.0437 - val_mae: 0.1486 - val_r2: 0.8419
Epoch 4/10
248656/248656 [==============================] - 25s 101us/step - loss: 0.0406 - mae: 0.1433 - r2: 0.8535 - val_loss: 0.0375 - val_mae: 0.1396 - val_r2: 0.8643
Epoch 5/10
248656/248656 [==============================] - 25s 101us/step - loss: 0.0355 - mae: 0.1342 - r2: 0.8720 - val_loss: 0.0397 - val_mae: 0.1426 - val_r2: 0.8565
Epoch 6/10
248656/248656 [==============================] - 25s 100us/step - loss: 0.0311 - mae: 0.1256 - r2: 0.8875 - val_loss: 0.0308 - val_mae: 0.1244 - val_r2: 0.8879
Epoch 7/10
248656/248656 [==============================] - 26s 105us/step - loss: 0.0268 - mae: 0.1168 - r2: 0.9030 - val_loss: 0.0303 - val_mae: 0.1233 - val_r2: 0.8903
Epoch 8/10
248656/248656 [==============================] - 25s 99us/step - loss: 0.0229 - mae: 0.1083 - r2: 0.9172 - val_loss: 0.0247 - val_mae: 0.1103 - val_r2: 0.9104
Epoch 9/10
248656/248656 [==============================] - 26s 106us/step - loss: 0.0207 - mae: 0.1029 - r2: 0.9253 - val_loss: 0.0243 - val_mae: 0.1081 - val_r2: 0.9119
Epoch 10/10
248656/248656 [==============================] - 27s 108us/step - loss: 0.0192 - mae: 0.0991 - r2: 0.9306 - val_loss: 0.0249 - val_mae: 0.1089 - val_r2: 0.9099
Comparing y_true and y_pred:
  mse: 0.0249
  mae: 0.1089
  r2: 0.9115
  corr: 0.9554
```

#### Train a model that can be used for inference with UQ
```
$ python combo_baseline_keras2.py --use_landmark_genes --drug_features descriptors --cell_features rnaseq --preprocess_rna source_scale --residual --warmup_lr --reduce_lr --residual --lr 0.0003 -z 128 --drop 0.2 --cp --epochs 100
```

#### Inference

There is a separate inference script that can be used to predict drug pair response on combinations of sample sets and drug sets with a trained model.
```
$ python infer.py --sample_set NCIPDM --drug_set ALMANAC

Using TensorFlow backend.
Predicting drug response for 6381440 combinations: 590 samples x 104 drugs x 104 drugs
100%|██████████████████████████████████████████████████████████████████████| 639/639 [14:56<00:00,  1.40s/it]
```
Example trained model files can be downloaded here: [saved.model.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.model.h5) and [saved.weights.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.weights.h5).

The inference script also accepts models trained with [dropout as a Bayesian Approximation](https://arxiv.org/pdf/1506.02142.pdf) for uncertainty quantification. Here is an example command line to make 100 point predictions for each sample-drugs combination in a subsample of the GDSC data.

```
$ python infer.py -s GDSC -d NCI_IOA_AOA --ns 10 --nd 5 -m saved.uq.model.h5 -w saved.uq.weights.h5 -n 100

$ cat comb_pred_GDSC_NCI_IOA_AOA.tsv
Sample  Drug1   Drug2   N       PredGrowthMean  PredGrowthStd   PredGrowthMin   PredGrowthMax
GDSC.22RV1      NSC.102816      NSC.102816      100     0.1688  0.0899  -0.0762 0.3912
GDSC.22RV1      NSC.102816      NSC.105014      100     0.3189  0.0920  0.0914  0.5550
GDSC.22RV1      NSC.102816      NSC.109724      100     0.6514  0.0894  0.4739  0.9055
GDSC.22RV1      NSC.102816      NSC.118218      100     0.5682  0.1164  0.2273  0.8891
GDSC.22RV1      NSC.102816      NSC.122758      100     0.3787  0.0833  0.1779  0.5768
GDSC.22RV1      NSC.105014      NSC.102816      100     0.1627  0.1060  -0.0531 0.5077
...
```

A version of trained model files with dropout are available here: [saved.uq.model.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.uq.model.h5) and [saved.uq.weights.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.uq.weights.h5).
