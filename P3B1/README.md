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

### Running the baseline implementation
```
cd P3B1
python keras_p3b1_baseline.py
```

Note that the training and testing data files are provided as standard Python pickle files in two separate directories: train/ and test/. The code is documented to provide enough information to reproduce the code on other platforms. 

The original data from the pathology reports cannot be made available online. Hence, we have pre-processed the reports so that example training/testing sets can be generated. Contact yoonh@ornl.gov for more information for generating additional training and testing data. A generic data loader that generates training and testing sets will be provided in the near future. 

#### Example output
```
Using Theano backend.
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
main_input (InputLayer)          (None, 400)           0                                            
____________________________________________________________________________________________________
layer1 (Dense)                   (None, 400)           160400      main_input[0][0]                 
____________________________________________________________________________________________________
layer2 (Dense)                   (None, 400)           160400      layer1[0][0]                     
____________________________________________________________________________________________________
layer3a (Dense)                  (None, 400)           160400      layer2[0][0]                     
____________________________________________________________________________________________________
layer4a (Dense)                  (None, 256)           102656      layer3a[0][0]                    
____________________________________________________________________________________________________
layer5a (Dense)                  (None, 2)             514         layer4a[0][0]                    
====================================================================================================
Total params: 584370
____________________________________________________________________________________________________
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
main_input (InputLayer)          (None, 400)           0                                            
____________________________________________________________________________________________________
layer1 (Dense)                   (None, 400)           160400      main_input[0][0]                 
____________________________________________________________________________________________________
layer2 (Dense)                   (None, 400)           160400      layer1[0][0]                     
____________________________________________________________________________________________________
layer3b (Dense)                  (None, 400)           160400      layer2[0][0]                     
____________________________________________________________________________________________________
layer4b (Dense)                  (None, 256)           102656      layer3b[0][0]                    
____________________________________________________________________________________________________
layer5b (Dense)                  (None, 2)             514         layer4b[0][0]                    
====================================================================================================
Total params: 584370
____________________________________________________________________________________________________
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
main_input (InputLayer)          (None, 400)           0                                            
____________________________________________________________________________________________________
layer1 (Dense)                   (None, 400)           160400      main_input[0][0]                 
____________________________________________________________________________________________________
layer2 (Dense)                   (None, 400)           160400      layer1[0][0]                     
____________________________________________________________________________________________________
layer3c (Dense)                  (None, 400)           160400      layer2[0][0]                     
____________________________________________________________________________________________________
layer4c (Dense)                  (None, 256)           102656      layer3c[0][0]                    
____________________________________________________________________________________________________
layer5c (Dense)                  (None, 4)             1028        layer4c[0][0]                    
====================================================================================================
Total params: 584884
____________________________________________________________________________________________________
```

### Preliminary Performance
```
Task 1: Primary site - Macro F1 score 0.981970027703
Task 1: Primary site - Micro F1 score 0.98202247191
Task 2: Tumor laterality - Macro F1 score 0.903931684754
Task 3: Tumor laterality - Micro F1 score 0.904
Task 3: Histological grade - Macro F1 score 0.48080174122
Task 3: Histological grade - Micro F1 score 0.497777777778
```

