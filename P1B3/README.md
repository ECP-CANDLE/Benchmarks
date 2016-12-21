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
Using Theano backend.
Using gpu device 0: Tesla K80 (CNMeM is enabled with initial size: 95.0% of memory, cuDNN 5004)
Loaded 2642218 unique (D, CL) response sets.
count    2.642218e+06
mean     6.906977e+01
std      4.860752e+01
min     -1.000000e+02
25%      5.400000e+01
50%      8.900000e+01
75%      9.900000e+01
max      2.990000e+02
Name: GROWTH, dtype: float64
Input dim = 998
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
dense_1 (Dense)                  (None, 1000)          999000      dense_input_1[0][0]
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
dense_5 (Dense)                  (None, 1)             51          dense_4[0][0]
====================================================================================================
Total params: 1554701
____________________________________________________________________________________________________
Epoch 1/20
2113731/2113700 [==============================] - 1794s - loss: 0.2039 - val_loss: 0.1932
Epoch 2/20
2113751/2113700 [==============================] - 1791s - loss: 0.1915 - val_loss: 0.1869
Epoch 3/20
2113744/2113700 [==============================] - 1786s - loss: 0.1886 - val_loss: 0.1887
Epoch 4/20
2113717/2113700 [==============================] - 1773s - loss: 0.1873 - val_loss: 0.1889
Epoch 5/20
2113732/2113700 [==============================] - 1776s - loss: 0.1857 - val_loss: 0.2158
Epoch 6/20
2113719/2113700 [==============================] - 1791s - loss: 0.1856 - val_loss: 0.1926
Epoch 7/20
2113742/2113700 [==============================] - 1793s - loss: 0.1849 - val_loss: 0.1779
Epoch 8/20
2113720/2113700 [==============================] - 1784s - loss: 0.1843 - val_loss: 0.1863
Epoch 9/20
2113733/2113700 [==============================] - 1783s - loss: 0.1841 - val_loss: 0.1945
Epoch 10/20
2113764/2113700 [==============================] - 1792s - loss: 0.1843 - val_loss: 0.1889
...

```

### Preliminary performance
Cristina's results: Using the 5 layer MLP with standard normalization and sizes : L1 = 1000, L2 = 500, L3 = 100, L4 = 50. MSE obtained is 0.0482 in training and  0.0421 in validation (~80% -20% split. Data: 2642218 unique (D, CL) response sets). This is at iteration 140, in a GPU Geforce GTX Titan X, taking around 15 hours.

![Histogram of errors: Random vs Epoch 1](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/master/P1B3/images/histo_It0.png)

![Histogram of errors after 141 epochs](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/master/P1B3/images/histo_It140.png)

![Measure vs Predicted percent growth after 141 epochs](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/master/P1B3/images/meas_vs_pred_It140.png)

