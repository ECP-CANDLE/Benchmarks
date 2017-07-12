## P3B1: Multi-task Deep Neural Net (DNN) for data extraction from clinical reports

**Overview**: Given a corpus of patient-level clinical reports, build a deep learning network that can simultaneously identify: (i) b tumor sites, (ii) t tumor laterality, and (iii) g clinical grade of tumors.

**Relationship to core problem**: Instead of training individual deep learning networks for individual machine learning tasks, build a multi-task DNN that can exploit task-relatedness to simultaneously learn multiple concepts.

**Expected outcome**: Multi-task DNN that trains on same corpus and can automatically classify across three related tasks.

### Benchmark Specs

#### Description of data
* Data source: Annotated pathology reports
* Input dimensions: 250,000-500,000 [characters], or 5,000-20,000 [bag of words], or 200-500 [bag of concepts]
* Output dimensions: (i) b tumor sites, (ii) t tumor laterality, and (iii) g clinical grade of tumors

* Sample size: O(1,000)
* Notes on data balance and other issues: standard NLP pre-processing is required, including (but not limited to) stemming words, keywords, cleaning text, stop words, etc. Data balance is an issue since the number of positive examples vs. control is skewed

#### Expected Outcomes
* Classification
* Output range or number of classes: Initially, 4 classes; can grow up to 32 classes, depending on number of tasks simultaneously trained.

#### Evaluation Metrics
* Accuracy or loss function: Standard approaches such as F1-score, accuracy, ROC-AUC, etc. will be used.
* Expected performance of a naïve method: Compare performance against (i) deep neural nets against single tasks, (ii) multi-task SVM based predictions, and (iii) random forest based methods.

#### Description of the Network
* Proposed network architecture: Deep neural net across individual tasks
* Number of layers: 5-6 layers

A graphical representation of the MTL-DNN is shown below:
![MTL-DNN Architecture](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/master/Pilot3/P3B1/images/MTL1.png)

### Running the baseline implementation
The baseline implementation includes a standard 10-fold cross-validation loop and for building and training the MTL-based deep neural net.

As with all the benchmarks, this is accomplished by;
```
cd P3B1
python p3b1_baseline_keras2.py
```

Note that the training and testing data files are provided as standard CSV files in a folder called data. The code is documented to provide enough information to reproduce the code on other platforms.

The original data from the pathology reports cannot be made available online. Hence, we have pre-processed the reports so that example training/testing sets can be generated. Contact yoonh@ornl.gov for more information for generating additional training and testing data. A generic data loader that generates training and testing sets will be provided in the near future.

#### Example output

```
Using TensorFlow backend.
Params: {'learning_rate': 0.01, 'activation': 'relu', 'valid_truths': 'data/task0_0_test_label.csv;data/task1_0_test_label.csv;data/task2_0_test_label.csv', 'output_files': 'result0_0.csv;result1_0.csv;result2_0.csv', 'datatype': <class 'numpy.float32'>, 'logfile': None, 'scaling': 'none', 'run_id': 'RUN000', 'train_features': 'data/task0_0_train_feature.csv;data/task1_0_train_feature.csv;data/task2_0_train_feature.csv', 'case': 'CenterZ', 'verbose': False, 'valid_features': 'data/task0_0_test_feature.csv;data/task1_0_test_feature.csv;data/task2_0_test_feature.csv', 'ind_nnet_spec': '1200, 1200:1200, 1200:1200, 1200', 'shared_nnet_spec': '1200', 'data_url': 'ftp://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P3B1/', 'config_file': '/CANDLE/benchmarks/Pilot3/P3B1/p3b1_default_model.txt', 'train_bool': True, 'drop': 0.0, 'loss': 'categorical_crossentropy', 'out_activation': 'softmax', 'feature_names': 'Primary site:Tumor laterality:Histological grade', 'output_dir': '/CANDLE/benchmarks/Pilot3/P3B1/Output/EXP000/RUN000', 'n_fold': 1, 'rng_seed': 7102, 'model_name': 'p3b1', 'experiment_id': 'EXP000', 'train_truths': 'data/task0_0_train_label.csv;data/task1_0_train_label.csv;data/task2_0_train_label.csv', 'batch_size': 10, 'optimizer': 'sgd', 'timeout': 1800, 'metrics': 'accuracy', 'fig_bool': False, 'train_data': 'P3B1_data.tar.gz', 'initialization': 'glorot_uniform', 'gpus': [], 'epochs': 10, 'shuffle': True}
Feature names:
Primary site
Tumor laterality
Histological grade
Model:  0
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 400)               0
_________________________________________________________________
shared_layer_0 (Dense)       (None, 1200)              481200
_________________________________________________________________
indiv_layer_0_0 (Dense)      (None, 1200)              1441200
_________________________________________________________________
indiv_layer_0_1 (Dense)      (None, 1200)              1441200
_________________________________________________________________
out_0 (Dense)                (None, 13)                15613
=================================================================
Total params: 3,379,213
Trainable params: 3,379,213
Non-trainable params: 0
_________________________________________________________________
Model:  1
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 400)               0
_________________________________________________________________
shared_layer_0 (Dense)       (None, 1200)              481200
_________________________________________________________________
indiv_layer_1_0 (Dense)      (None, 1200)              1441200
_________________________________________________________________
indiv_layer_1_1 (Dense)      (None, 1200)              1441200
_________________________________________________________________
out_1 (Dense)                (None, 2)                 2402
=================================================================
Total params: 3,366,002
Trainable params: 3,366,002
Non-trainable params: 0
_________________________________________________________________
Model:  2
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 400)               0
_________________________________________________________________
shared_layer_0 (Dense)       (None, 1200)              481200
_________________________________________________________________
indiv_layer_2_0 (Dense)      (None, 1200)              1441200
_________________________________________________________________
indiv_layer_2_1 (Dense)      (None, 1200)              1441200
_________________________________________________________________
out_2 (Dense)                (None, 4)                 4804
=================================================================
Total params: 3,368,404
Trainable params: 3,368,404
Non-trainable params: 0
_________________________________________________________________
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3730/3900 [===========================>..] - ETA: 0s - loss: 2.5461 - acc: 0.1912Current time ....2.599
3900/3900 [==============================] - 2s - loss: 2.5451 - acc: 0.2008 - val_loss: 2.5387 - val_acc: 0.2525
Train on 600 samples, validate on 86 samples
Epoch 1/1
490/600 [=======================>......] - ETA: 0s - loss: 0.6490 - acc: 0.6939Current time ....0.243
600/600 [==============================] - 0s - loss: 0.6380 - acc: 0.7300 - val_loss: 0.5922 - val_acc: 0.8605
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1190/1200 [============================>.] - ETA: 0s - loss: 1.2678 - acc: 0.4521Current time ....0.426
1200/1200 [==============================] - 0s - loss: 1.2665 - acc: 0.4533 - val_loss: 1.1895 - val_acc: 0.3971
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3890/3900 [============================>.] - ETA: 0s - loss: 2.4869 - acc: 0.4550Current time ....1.199
3900/3900 [==============================] - 1s - loss: 2.4867 - acc: 0.4554 - val_loss: 2.4689 - val_acc: 0.3535
Train on 600 samples, validate on 86 samples
Epoch 1/1
510/600 [========================>.....] - ETA: 0s - loss: 0.5012 - acc: 0.9431Current time ....0.196
600/600 [==============================] - 0s - loss: 0.4938 - acc: 0.9467 - val_loss: 0.5007 - val_acc: 0.9070
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1170/1200 [============================>.] - ETA: 0s - loss: 1.0152 - acc: 0.6667Current time ....0.384
1200/1200 [==============================] - 0s - loss: 1.0141 - acc: 0.6658 - val_loss: 1.1178 - val_acc: 0.3971
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3890/3900 [============================>.] - ETA: 0s - loss: 2.3278 - acc: 0.4416Current time ....1.213
3900/3900 [==============================] - 1s - loss: 2.3274 - acc: 0.4415 - val_loss: 2.2659 - val_acc: 0.2727
Train on 600 samples, validate on 86 samples
Epoch 1/1
490/600 [=======================>......] - ETA: 0s - loss: 0.3542 - acc: 0.9694Current time ....0.200
600/600 [==============================] - 0s - loss: 0.3461 - acc: 0.9700 - val_loss: 0.4129 - val_acc: 0.9302
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1160/1200 [============================>.] - ETA: 0s - loss: 0.7984 - acc: 0.7448Current time ....0.382
1200/1200 [==============================] - 0s - loss: 0.7981 - acc: 0.7475 - val_loss: 1.0306 - val_acc: 0.5000
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3840/3900 [============================>.] - ETA: 0s - loss: 2.0177 - acc: 0.4354Current time ....1.227
3900/3900 [==============================] - 1s - loss: 2.0157 - acc: 0.4372 - val_loss: 1.9542 - val_acc: 0.4646
Train on 600 samples, validate on 86 samples
Epoch 1/1
490/600 [=======================>......] - ETA: 0s - loss: 0.2423 - acc: 0.9816Current time ....0.203
600/600 [==============================] - 0s - loss: 0.2324 - acc: 0.9817 - val_loss: 0.3808 - val_acc: 0.9302
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1170/1200 [============================>.] - ETA: 0s - loss: 0.4739 - acc: 0.8838Current time ....0.379
1200/1200 [==============================] - 0s - loss: 0.4719 - acc: 0.8833 - val_loss: 0.7585 - val_acc: 0.6765
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3850/3900 [============================>.] - ETA: 0s - loss: 1.6362 - acc: 0.5301Current time ....1.217
3900/3900 [==============================] - 1s - loss: 1.6348 - acc: 0.5310 - val_loss: 1.8301 - val_acc: 0.2828
Train on 600 samples, validate on 86 samples
Epoch 1/1
490/600 [=======================>......] - ETA: 0s - loss: 0.1181 - acc: 0.9878Current time ....0.201
600/600 [==============================] - 0s - loss: 0.1178 - acc: 0.9867 - val_loss: 0.3921 - val_acc: 0.9302
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1030/1200 [========================>.....] - ETA: 0s - loss: 0.3711 - acc: 0.9097Current time ....0.375
1200/1200 [==============================] - 0s - loss: 0.3572 - acc: 0.9125 - val_loss: 0.6792 - val_acc: 0.6912
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3770/3900 [============================>.] - ETA: 0s - loss: 1.4697 - acc: 0.5963Current time ....1.237
3900/3900 [==============================] - 1s - loss: 1.4665 - acc: 0.5969 - val_loss: 1.7400 - val_acc: 0.4040
Train on 600 samples, validate on 86 samples
Epoch 1/1
490/600 [=======================>......] - ETA: 0s - loss: 0.0922 - acc: 0.9898Current time ....0.198
600/600 [==============================] - 0s - loss: 0.0887 - acc: 0.9917 - val_loss: 0.4270 - val_acc: 0.9419
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1180/1200 [============================>.] - ETA: 0s - loss: 0.2829 - acc: 0.9331Current time ....0.380
1200/1200 [==============================] - 0s - loss: 0.2831 - acc: 0.9325 - val_loss: 0.6785 - val_acc: 0.6912
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3870/3900 [============================>.] - ETA: 0s - loss: 1.2898 - acc: 0.6362Current time ....1.216
3900/3900 [==============================] - 1s - loss: 1.2901 - acc: 0.6362 - val_loss: 1.5841 - val_acc: 0.4848
Train on 600 samples, validate on 86 samples
Epoch 1/1
490/600 [=======================>......] - ETA: 0s - loss: 0.0694 - acc: 0.9939Current time ....0.204
600/600 [==============================] - 0s - loss: 0.0678 - acc: 0.9933 - val_loss: 0.4263 - val_acc: 0.9302
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1150/1200 [===========================>..] - ETA: 0s - loss: 0.2152 - acc: 0.9557Current time ....0.390
1200/1200 [==============================] - 0s - loss: 0.2187 - acc: 0.9542 - val_loss: 0.6817 - val_acc: 0.6765
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3860/3900 [============================>.] - ETA: 0s - loss: 1.1219 - acc: 0.6655Current time ....1.216
3900/3900 [==============================] - 1s - loss: 1.1202 - acc: 0.6656 - val_loss: 1.5461 - val_acc: 0.4747
Train on 600 samples, validate on 86 samples
Epoch 1/1
500/600 [========================>.....] - ETA: 0s - loss: 0.0563 - acc: 0.9960Current time ....0.198
600/600 [==============================] - 0s - loss: 0.0545 - acc: 0.9950 - val_loss: 0.4550 - val_acc: 0.9302
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1180/1200 [============================>.] - ETA: 0s - loss: 0.1672 - acc: 0.9737Current time ....0.381
1200/1200 [==============================] - 0s - loss: 0.1678 - acc: 0.9733 - val_loss: 0.8137 - val_acc: 0.6618
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3820/3900 [============================>.] - ETA: 0s - loss: 0.9720 - acc: 0.7084Current time ....1.228
3900/3900 [==============================] - 1s - loss: 0.9697 - acc: 0.7092 - val_loss: 1.5724 - val_acc: 0.4646
Train on 600 samples, validate on 86 samples
Epoch 1/1
520/600 [=========================>....] - ETA: 0s - loss: 0.0420 - acc: 0.9962Current time ....0.198
600/600 [==============================] - 0s - loss: 0.0444 - acc: 0.9950 - val_loss: 0.4880 - val_acc: 0.9186
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1140/1200 [===========================>..] - ETA: 0s - loss: 0.1301 - acc: 0.9816Current time ....0.390
1200/1200 [==============================] - 0s - loss: 0.1289 - acc: 0.9808 - val_loss: 0.6855 - val_acc: 0.7059

```

### Preliminary Performance
The performance numbers are output after running the 10-fold cross-validation loop. The output produced includes the following:
```
Task 1 : Primary site - Macro F1 score 0.267651585929
Task 1 : Primary site - Micro F1 score 0.464646464646
Task 2 : Tumor laterality - Macro F1 score 0.918505482605
Task 2 : Tumor laterality - Micro F1 score 0.918604651163
Task 3 : Histological grade - Macro F1 score 0.540748060313
Task 3 : Histological grade - Micro F1 score 0.705882352941
Average loss:  0.915194450984
```
