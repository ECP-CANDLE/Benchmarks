## P3B1: Multi-task Deep Neural Net (DNN) for data extraction from clinical reports

**Overview**: Given a corpus of patient-level clinical reports, build a deep learning network that can simultaneously identify: (i) b tumor sites, (ii) t tumor laterality, and (iii) g clinical grade of tumors.

**Relationship to core problem**: Instead of training individual deep learning networks for individual machine learning tasks, build a multi-task DNN that can exploit task-relatedness to simultaneously learn multiple concepts.

**Expected outcome**: Multi-task DNN that trains on same corpus and can automatically classify across three related tasks.

### Benchmark Specs

#### Description of data

- Data source: Annotated pathology reports
- Input dimensions: 250,000-500,000 [characters], or 5,000-20,000 [bag of words], or 200-500 [bag of concepts]
- Output dimensions: (i) b tumor sites, (ii) t tumor laterality, and (iii) g clinical grade of tumors

- Sample size: O(1,000)
- Notes on data balance and other issues: standard NLP pre-processing is required, including (but not limited to) stemming words, keywords, cleaning text, stop words, etc. Data balance is an issue since the number of positive examples vs. control is skewed

#### Expected Outcomes

- Classification
- Output range or number of classes: Initially, 4 classes; can grow up to 32 classes, depending on number of tasks simultaneously trained.

#### Evaluation Metrics

- Accuracy or loss function: Standard approaches such as F1-score, accuracy, ROC-AUC, etc. will be used.
- Expected performance of a naïve method: Compare performance against (i) deep neural nets against single tasks, (ii) multi-task SVM based predictions, and (iii) random forest based methods.

#### Description of the Network

- Proposed network architecture: Deep neural net across individual tasks
- Number of layers: 5-6 layers

A graphical representation of the MTL-DNN is shown below:
![MTL-DNN Architecture](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/master/Pilot3/P3B1/images/MTL1.png)

### Running the baseline implementation

There are two broad options for running our MTL implementation. The first baseline option includes the basic training of an MTL-based deep neural net. The second implementation includes a standard 10-fold cross-validation loop and depends on the first baseline for building and training the MTL-based deep neural net.

For the first baseline run, an example run is shown below:

```
cd P3B1
python MTL_run.py
```

For the second baseline run, including the 10-fold cross-validation loop, the set up is shown below.

```
cd P3B1
python keras_p3b1_baseline.py
```

Note that the training and testing data files are provided as standard CSV files in a folder called data. The code is documented to provide enough information to reproduce the code on other platforms.

The original data from the pathology reports cannot be made available online. Hence, we have pre-processed the reports so that example training/testing sets can be generated. Contact yoonh@ornl.gov for more information for generating additional training and testing data. A generic data loader that generates training and testing sets will be provided in the near future.

#### Example output

For the first baseline run involving just the basic training process of a MTL deep neural network, the output is shown below:

```
Using TensorFlow backend.
('Args:', Namespace(activation='relu', batch_size=10, dropout=0.1, individual_nnet_spec='1200,1200;1200,1200;1200,1200', learning_rate=0.01, n_epochs=10, output_files='result0_0.csv;result1_0.csv;result2_0.csv', shared_nnet_spec='1200', train_features='data/task0_0_train_feature.csv;data/task1_0_train_feature.csv;data/task2_0_train_feature.csv', train_truths='data/task0_0_train_label.csv;data/task1_0_train_label.csv;data/task2_0_train_label.csv', valid_features='data/task0_0_test_feature.csv;data/task1_0_test_feature.csv;data/task2_0_test_feature.csv', valid_truths='data/task0_0_test_label.csv;data/task1_0_test_label.csv;data/task2_0_test_label.csv', verbose=True))
/Users/v33/anaconda2/envs/tensorflow/lib/python2.7/site-packages/Keras-2.0.1-py2.7.egg/keras/legacy/interfaces.py:86: UserWarning: Update your `Model` call to the Keras 2 API: `Model(outputs=[<tf.Tenso..., inputs=[<tf.Tenso...)`
  '` call to the Keras 2 API: ' + signature)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 400)               0
_________________________________________________________________
shared_layer_0 (Dense)       (None, 1200)              481200
_________________________________________________________________
dropout_1 (Dropout)          (None, 1200)              0
_________________________________________________________________
indiv_layer_0_0 (Dense)      (None, 1200)              1441200
_________________________________________________________________
dropout_2 (Dropout)          (None, 1200)              0
_________________________________________________________________
indiv_layer_0_1 (Dense)      (None, 1200)              1441200
_________________________________________________________________
dropout_3 (Dropout)          (None, 1200)              0
_________________________________________________________________
out_0 (Dense)                (None, 13)                15613
=================================================================
Total params: 3,379,213.0
Trainable params: 3,379,213.0
Non-trainable params: 0.0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 400)               0
_________________________________________________________________
shared_layer_0 (Dense)       (None, 1200)              481200
_________________________________________________________________
dropout_1 (Dropout)          (None, 1200)              0
_________________________________________________________________
indiv_layer_1_0 (Dense)      (None, 1200)              1441200
_________________________________________________________________
dropout_4 (Dropout)          (None, 1200)              0
_________________________________________________________________
indiv_layer_1_1 (Dense)      (None, 1200)              1441200
_________________________________________________________________
dropout_5 (Dropout)          (None, 1200)              0
_________________________________________________________________
out_1 (Dense)                (None, 2)                 2402
=================================================================
Total params: 3,366,002.0
Trainable params: 3,366,002.0
Non-trainable params: 0.0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 400)               0
_________________________________________________________________
shared_layer_0 (Dense)       (None, 1200)              481200
_________________________________________________________________
dropout_1 (Dropout)          (None, 1200)              0
_________________________________________________________________
indiv_layer_2_0 (Dense)      (None, 1200)              1441200
_________________________________________________________________
dropout_6 (Dropout)          (None, 1200)              0
_________________________________________________________________
indiv_layer_2_1 (Dense)      (None, 1200)              1441200
_________________________________________________________________
dropout_7 (Dropout)          (None, 1200)              0
_________________________________________________________________
out_2 (Dense)                (None, 4)                 4804
=================================================================
Total params: 3,368,404.0
Trainable params: 3,368,404.0
Non-trainable params: 0.0
_________________________________________________________________
/Users/v33/anaconda2/envs/tensorflow/lib/python2.7/site-packages/Keras-2.0.1-py2.7.egg/keras/engine/training.py:1393: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
  warnings.warn('The `nb_epoch` argument in `fit` '
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 2.5503 - acc: 0.2038 - val_loss: 2.5393 - val_acc: 0.3333
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.6646 - acc: 0.6317 - val_loss: 0.6134 - val_acc: 0.8023
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 1.2899 - acc: 0.4325 - val_loss: 1.2316 - val_acc: 0.4706
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 9s - loss: 2.5048 - acc: 0.4090 - val_loss: 2.4860 - val_acc: 0.4141
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.5467 - acc: 0.8717 - val_loss: 0.5172 - val_acc: 0.8837
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 1.0717 - acc: 0.6067 - val_loss: 1.2034 - val_acc: 0.3676
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 2.3923 - acc: 0.4421 - val_loss: 2.3336 - val_acc: 0.3434
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.4160 - acc: 0.9533 - val_loss: 0.4214 - val_acc: 0.8953
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.8751 - acc: 0.6517 - val_loss: 0.9763 - val_acc: 0.5588
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 2.1131 - acc: 0.4077 - val_loss: 2.0320 - val_acc: 0.3737
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.2832 - acc: 0.9717 - val_loss: 0.3664 - val_acc: 0.9186
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.7104 - acc: 0.7492 - val_loss: 0.9136 - val_acc: 0.5882
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 1.8495 - acc: 0.4549 - val_loss: 1.8962 - val_acc: 0.3737
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.2041 - acc: 0.9733 - val_loss: 0.3443 - val_acc: 0.9186
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.5606 - acc: 0.8100 - val_loss: 0.7196 - val_acc: 0.6912
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 9s - loss: 1.6929 - acc: 0.4915 - val_loss: 1.8462 - val_acc: 0.3232
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.1462 - acc: 0.9817 - val_loss: 0.3675 - val_acc: 0.9186
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.4458 - acc: 0.8683 - val_loss: 0.6699 - val_acc: 0.7353
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 1.5355 - acc: 0.5451 - val_loss: 1.7320 - val_acc: 0.3939
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.1104 - acc: 0.9833 - val_loss: 0.3814 - val_acc: 0.9186
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.3538 - acc: 0.9075 - val_loss: 0.6676 - val_acc: 0.7500
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 9s - loss: 1.3579 - acc: 0.6049 - val_loss: 1.6628 - val_acc: 0.4444
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.0832 - acc: 0.9817 - val_loss: 0.3789 - val_acc: 0.9302
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.2938 - acc: 0.9175 - val_loss: 0.7699 - val_acc: 0.7353
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 1.1794 - acc: 0.6531 - val_loss: 1.6515 - val_acc: 0.4141
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.0703 - acc: 0.9900 - val_loss: 0.4057 - val_acc: 0.9186
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 2s - loss: 0.2452 - acc: 0.9200 - val_loss: 0.8022 - val_acc: 0.6912
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 1.0181 - acc: 0.6921 - val_loss: 1.5341 - val_acc: 0.4646
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.0597 - acc: 0.9883 - val_loss: 0.4196 - val_acc: 0.9186
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.2160 - acc: 0.9417 - val_loss: 0.9318 - val_acc: 0.6765

```

For the second baseline run involving the 10-fold cross-validation loop, the outputs are shown below:

```
Using TensorFlow backend.

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
Total params: 3,379,213.0
Trainable params: 3,379,213.0
Non-trainable params: 0.0
_________________________________________________________________
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
Total params: 3,366,002.0
Trainable params: 3,366,002.0
Non-trainable params: 0.0
_________________________________________________________________
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
Total params: 3,368,404.0
Trainable params: 3,368,404.0
Non-trainable params: 0.0
_________________________________________________________________

Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 11s - loss: 2.5459 - acc: 0.2200 - val_loss: 2.5323 - val_acc: 0.3636
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.6348 - acc: 0.7817 - val_loss: 0.5944 - val_acc: 0.8488
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 1.2616 - acc: 0.4750 - val_loss: 1.2342 - val_acc: 0.3382
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 2.4852 - acc: 0.4185 - val_loss: 2.4643 - val_acc: 0.4242
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.4908 - acc: 0.9333 - val_loss: 0.4956 - val_acc: 0.8953
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 1.0065 - acc: 0.6625 - val_loss: 1.1576 - val_acc: 0.4118
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 2.3170 - acc: 0.4603 - val_loss: 2.2293 - val_acc: 0.4444
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.3421 - acc: 0.9733 - val_loss: 0.4095 - val_acc: 0.9070
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.7931 - acc: 0.7333 - val_loss: 0.9583 - val_acc: 0.5588
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 1.9996 - acc: 0.4697 - val_loss: 2.0365 - val_acc: 0.2828
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.2271 - acc: 0.9817 - val_loss: 0.3943 - val_acc: 0.9070
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 2s - loss: 0.6163 - acc: 0.8158 - val_loss: 0.9127 - val_acc: 0.5735
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 9s - loss: 1.7831 - acc: 0.5087 - val_loss: 1.8651 - val_acc: 0.4242
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.1557 - acc: 0.9817 - val_loss: 0.3668 - val_acc: 0.9070
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.4764 - acc: 0.8800 - val_loss: 0.9771 - val_acc: 0.6471
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 1.6234 - acc: 0.5492 - val_loss: 1.9028 - val_acc: 0.2828
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.1121 - acc: 0.9850 - val_loss: 0.3876 - val_acc: 0.9070
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.3680 - acc: 0.9042 - val_loss: 0.7091 - val_acc: 0.6765
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 12s - loss: 1.4569 - acc: 0.6095 - val_loss: 1.7021 - val_acc: 0.4343
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.0887 - acc: 0.9850 - val_loss: 0.4135 - val_acc: 0.9070
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.2802 - acc: 0.9400 - val_loss: 0.7738 - val_acc: 0.6618
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 1.2737 - acc: 0.6492 - val_loss: 1.5725 - val_acc: 0.4848
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.0683 - acc: 0.9933 - val_loss: 0.4555 - val_acc: 0.9070
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.2122 - acc: 0.9550 - val_loss: 0.7898 - val_acc: 0.7206
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 10s - loss: 1.1024 - acc: 0.6849 - val_loss: 1.5658 - val_acc: 0.4848
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.0547 - acc: 0.9933 - val_loss: 0.4346 - val_acc: 0.9186
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.1733 - acc: 0.9683 - val_loss: 0.7971 - val_acc: 0.7059
Train on 3900 samples, validate on 99 samples
Epoch 1/1
3900/3900 [==============================] - 11s - loss: 0.9462 - acc: 0.7259 - val_loss: 1.6119 - val_acc: 0.4141
Train on 600 samples, validate on 86 samples
Epoch 1/1
600/600 [==============================] - 1s - loss: 0.0466 - acc: 0.9950 - val_loss: 0.4706 - val_acc: 0.9186
Train on 1200 samples, validate on 68 samples
Epoch 1/1
1200/1200 [==============================] - 3s - loss: 0.1355 - acc: 0.9800 - val_loss: 0.7776 - val_acc: 0.7206
```

### Preliminary Performance

The performance numbers are output after running the 10-fold cross-validation loop. The output produced includes the following:

```
Task 1: Primary site - Macro F1 score 0.230512521335
Task 1: Primary site - Micro F1 score 0.414141414141
Task 2: Tumor laterality - Macro F1 score 0.918593644354
Task 3: Tumor laterality - Micro F1 score 0.918604651163
Task 3: Histological grade - Macro F1 score 0.537549407115
Task 3: Histological grade - Micro F1 score 0.720588235294
```
