## P1B3: MLP Regression Drug Response Prediction

**Overview**: Given drug screening results on NCI60 cell lines, build a deep learning network that can predict the growth percentage from cell line expression and drug descriptors.

**Relationship to core problem**: This benchmark is a simplified form of the core drug response prediction problem in which we need to combine multiple molecular assays and a diverse array of drug descriptors to make a prediction.

**Expected outcome**: Build a DNN that can predict growth percentage of a cell line treated with a new drug.

### Benchmark Specs Requirements

#### Description of the Data
* Data source: Dose response screening results from NCI; 5-platform normalized expression data from NCI; Dragon7 generated drug descriptors based on 2D chemical structures from NCI
* Input dimensions: ~30K; 26K normalized expression levels by gene + 4K drug descriptors [+ drug concentration]
Output dimensions: 1 (growth percentage)
* Sample size: ~2.5 M screening results (combinations of cell line and drug)
* Notes on data balance: original data imbalanced with many drugs that have little inhibition effect.

#### Expected Outcomes
* Regression. Predict percent growth per NCI-60 cell lines and per drug
* Dimension: 1 scalar value corresponding to the percent growth for a given drug concentration. Output range: [-100, 100]

#### Evaluation Metrics
* Accuracy or loss function: mean squared error or rank order.
* Expected performance of a naïve method: mean response, linear regression or random forest regression.

#### Description of the Network
* Proposed network architecture: MLP
* Number of layers: ~5 layers

### Running the baseline implementation

```
$ cd P1B3
$ python p1b3_baseline.py
```

#### Example output
```
Using TensorFlow backend.
Command line args = Namespace(activation='relu', batch_size=100, category_cutoffs=[0.0, 0.5], dense=[1000, 500, 100, 50], drop=0.1, drug_features='descriptors', epochs=20, feature_subsample=500, loss='mse', max_logconc=-4.0, min_logconc=-5.0, optimizer='adam', save='save', scaling='std', scramble=False, subsample='naive_balancing', train_samples=0, val_samples=0, verbose=False, workers=1)
Loaded 2328562 unique (D, CL) response sets.
Distribution of dose response:
             GROWTH
count  1.004870e+06
mean  -1.357397e+00
std    6.217888e+01
min   -1.000000e+02
25%   -5.600000e+01
50%    0.000000e+00
75%    4.600000e+01
max    2.580000e+02
Category cutoffs: [0.0]
Dose response bin counts:
  Class 0:  497382 (0.4950) - between -1.00 and +0.00
  Class 1:  507488 (0.5050) - between +0.00 and +2.58
  Total:   1004870
Rows in train: 800068, val: 200017, test: 4785
Input dim = 1001
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
dense_1 (Dense)                  (None, 1000)          1002000     dense_input_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1000)          0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 500)           500500      dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 500)           0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 100)           50100       dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 100)           0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 50)            5050        dropout_3[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 50)            0           dense_4[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             51          dropout_4[0][0]
====================================================================================================
Total params: 1,557,701
Trainable params: 1,557,701
Non-trainable params: 0

Epoch 1/20
800000/800000 [==============================] - 420s - loss: 0.2554 - val_loss: 0.2037 - val_acc: 0.7519 - test_loss: 1.0826 - test_acc: 0.5651
Epoch 2/20
800000/800000 [==============================] - 426s - loss: 0.1885 - val_loss: 0.1620 - val_acc: 0.7720 - test_loss: 1.1407 - test_acc: 0.5689
Epoch 3/20
800000/800000 [==============================] - 427s - loss: 0.1600 - val_loss: 0.1403 - val_acc: 0.7853 - test_loss: 1.1443 - test_acc: 0.5689
... ...
Epoch 18/20
800000/800000 [==============================] - 349s - loss: 0.0912 - val_loss: 0.0881 - val_acc: 0.8339 - test_loss: 1.0033 - test_acc: 0.5653
Epoch 19/20
800000/800000 [==============================] - 418s - loss: 0.0898 - val_loss: 0.0844 - val_acc: 0.8354 - test_loss: 1.0039 - test_acc: 0.5652
Epoch 20/20
800000/800000 [==============================] - 343s - loss: 0.0894 - val_loss: 0.0849 - val_acc: 0.8354 - test_loss: 1.0039 - test_acc: 0.5652

```

### Preliminary performance
Cristina's results: Using the 5 layer MLP with standard normalization and sizes : L1 = 1000, L2 = 500, L3 = 100, L4 = 50. MSE obtained is 0.0482 in training and  0.0421 in validation (~80% -20% split. Data: 2642218 unique (D, CL) response sets). This is at iteration 140, in a GPU Geforce GTX Titan X, taking around 15 hours.

![Histogram of errors: Random vs Epoch 1](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/master/P1B3/images/histo_It0.png)

![Histogram of errors after 141 epochs](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/master/P1B3/images/histo_It140.png)

![Measure vs Predicted percent growth after 141 epochs](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/master/P1B3/images/meas_vs_pred_It140.png)
