## P2B2: Predictive, recurrent, Autoencoder Compressed Representation for Molecular Dynamics Simulation Data

**Overview**: Cut down on manual inspection time for molecular simulation data

**Relationship to core problem**: Identify possible forming of cancer-causing proteins by molecular simulation

**Expected outcome**: Improvement in the understanding of protein formation and easing of the handling large-scale molecular dynamics output

### Benchmark Specs Requirements

#### Description of the Data
* See Pilot2 Readme for description

#### Expected Outcomes
* 'Telescope' into data: Find regions of interest based on higher level of structure than rest of regions
* Output range: Dimension: 1 scalar value corresponding to each frame of simulation representing the structured-ness of the data compared to the mean. Output range: [0, 100] 0=mean noise level, 100=very structured.

#### Evaluation Metrics
* Accuracy or loss function: Domain experts agreeing on utility
* Expected performance of a naive method: Comparison of different technical approaches against each other and against labels (see above)

#### Description of the Network
* Proposed network architecture: stacked fully-connected autoencoder feeding RNN
* Number of layers: 4

### Running the baseline implementation

Using virtualenv

```
cd P2B2
workon keras
python p2b2_baseline_keras1.py
```
### Scaling Options
* ```--case=FULL``` Design autoencoder for data frame with coordinates for all beads
* ```--case=CENTER``` Design autoencoder for data frame with coordinates of the center-of-mass
* ```--case=CENTERZ``` Design autoencoder for data frame with z-coordiate of the center-of-mass

### Expected Results

(keras) vanessen1@vandamme: ~/Research/DeepLearning/ECP CANDLE/Benchmarks/Benchmarks.git/Pilot2/P2B2$ python p2b2_baseline_keras1.py
Using Theano backend.
{'num_hidden': [], 'num_recurrent': [16, 16, 16], 'noise_factor': 0, 'learning_rate': 0.01, 'batch_size': 32, 'look_forward': 1, 'epochs': 1, 'weight_decay': 0.0005, 'look_back': 10, 'cool': 'True'}
Reading Data...
Reading Data Files... 3k_Disordered->3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir
('X_train type and shape:', dtype('float64'), (89, 10, 3040))
('X_train.min():', 38.831248919169106)
('X_train.max():', 100.46649742126465)
Define the model and compile
using mlp network
Autoencoder Regression problem
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 10, 3040)      0                                            
____________________________________________________________________________________________________
timedistributed_1 (TimeDistribut (None, 10, 3040)      9244640     input_1[0][0]                    
====================================================================================================
Total params: 9,244,640
Trainable params: 9,244,640
Non-trainable params: 0
____________________________________________________________________________________________________
  0%|                                                                                                          | 0/1 [00:00<?, ?it/s]
Loss on epoch 0: 47.9408
100%|------------------------------------|                                                                     | 1/1 [00:35<00:00, 35.68s/it]
Cooling Learning Rate by factor of 10...
  0%|                                                                                                          | 0/1 [00:00<?, ?it/s]
Loss on epoch 0: 30.2241
100%|---------------------------------------------------------------|                                          | 1/1 [00:35<00:00, 35.29s/it]
Cooling Learning Rate by factor of 10...
  0%|                                                                                                          | 0/1 [00:00<?, ?it/s]
Loss on epoch 0: 23.4605
100%|----------------------------------------------------------------------------------------------------------| 1/1 [00:35<00:00, 35.72s/it]
(keras)
