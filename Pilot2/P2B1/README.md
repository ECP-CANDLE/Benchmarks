## P2B1: Autoencoder Compressed Representation for Molecular Dynamics Simulation Data

**Overview**: Generate automatically extracted features representing molecular simulation data

**Relationship to core problem**: Establish framework for building future tools using learned features

**Expected outcome**: Improvement in the understanding of protein formation and easing of the handling large-scale molecular dynamics output

### Benchmark Specs Requirements

#### Description of the Data
* See Pilot2 Readme for description

#### Expected Outcomes
* Reconstructed MD simulation state
* Output range: automatically learned features that discriminate the data set

#### Evaluation Metrics
* Accuracy or loss function: L2 reconstruction error
* Expected performance of a naive method: N/A

#### Description of the Network
* Proposed network architecture: stacked fully-connected autoencoder
* Number of layers: 5-8

### Running the baseline implementation
```
cd Pilot2/2B1
python p2b1_baseline_keras1.py
```

The training and test data files will be downloaded the first time this is run and will be cached for future runs.

### Scaling Options
* ```--case=FULL``` Design autoencoder for data frame with coordinates for all beads
* ```--case=CENTER``` Design autoencoder for data frame with coordinates of the center-of-mass
* ```--case=CENTERZ``` Design autoencoder for data frame with z-coordiate of the center-of-mass

### Expected Output

```
Using TensorFlow backend.
Params: {'shuffle': True, 'metrics': 'mean_squared_error', 'train_data': '3k_Disordered', 'run_id': 'RUN000', 'rng_seed': 7102, 'case': 'CenterZ', 'experiment_id': 'EXP000', 'epochs': 10, 'conv_bool': True, 'type_bool': False, 'molecular_activation': 'elu', 'data_url': 'ftp://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot2/', 'learning_rate': 0.01, 'verbose': False, 'datatype': <class 'numpy.float32'>, 'loss': 'mse', 'cool': False, 'scaling': 'none', 'train_bool': True, 'output_dir': '/home/jamal/Code/ECP/CANDLE/benchmarks/Pilot2/P2B1/Output/EXP000/RUN000', 'optimizer': 'adam', 'timeout': 3600, 'activation': 'elu', 'fig_bool': False, 'weight_decay': 0.0005, 'initialization': 'glorot_uniform', 'weight_path': None, 'gpus': [], 'dense': [512, 32], 'molecular_num_hidden': [54, 12], 'model_name': 'p2b1', 'molecular_epochs': 1, 'config_file': '/home/jamal/Code/ECP/CANDLE/benchmarks/Pilot2/P2B1/p2b1_default_model.txt', 'batch_size': 32, 'logfile': None, 'noise_factor': 0.0}
Reading Data...
Reading Data Files... 3k_Disordered->3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20-f20.dir
X shape:  (100, 3040, 12, 20)
The input dimension is  36480
Data Format:
  [Frames (100), Molecules (3040), Beads (12), odict_keys(['x', 'y', 'z', 'CHOL', 'DPPC', 'DIPC', 'Head', 'Tail', 'BL1', 'BL2', 'BL3', 'BL4', 'BL5', 'BL6', 'BL7', 'BL8', 'BL9', 'BL10', 'BL11', 'BL12']) (20)]
Define the model and compile
using mlp network
Autoencoder Regression problem
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 36480)             0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               18678272
_________________________________________________________________
dense_2 (Dense)              (None, 32)                16416
_________________________________________________________________
dense_3 (Dense)              (None, 512)               16896
_________________________________________________________________
dense_4 (Dense)              (None, 36480)             18714240
=================================================================
Total params: 37,425,824
Trainable params: 37,425,824
Non-trainable params: 0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_2 (InputLayer)         (None, 1, 180)            0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 1, 54)             68094
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 1, 12)             4548
_________________________________________________________________
flatten_1 (Flatten)          (None, 12)                0
_________________________________________________________________
dense_5 (Dense)              (None, 54)                702
_________________________________________________________________
dense_6 (Dense)              (None, 180)               9900
=================================================================
Total params: 83,244
Trainable params: 83,244
Non-trainable params: 0
_____________________________________________________________
  0%|                                                                                | 0/10 [00:00<?, ?it/s]
/home/jamal/Code/ECP/CANDLE/benchmarks/common/../Data/Pilot2/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20-f20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_02_outof_29.npy
[Frame 0] Inner AE loss.. 0.134360544776
(3040, 1, 180)
Epoch 1/1
2688/3040 [=========================>....] - ETA: 0s - loss: 0.0245 - mean_squared_error: 0.0139Current time ....0.512
3040/3040 [==============================] - 0s - loss: 0.0231 - mean_squared_error: 0.0132
Loss on epoch 0: 1.7191
 10%|#######2                                                                | 1/10 [00:12<01:55, 12.88s/it]Loss on epoch 1: 14.4972
 20%|##############4                                                         | 2/10 [00:17<01:22, 10.35s/it]Loss on epoch 2: 7434.85
 30%|#####################5                                                  | 3/10 [00:21<00:59,  8.55s/it]Loss on epoch 3: 287.711
 40%|############################8                                           | 4/10 [00:26<00:43,  7.32s/it]Loss on epoch 4: 7.41576
 50%|####################################                                    | 5/10 [00:30<00:32,  6.46s/it]Loss on epoch 5: 2772.88
 60%|###########################################1                            | 6/10 [00:35<00:23,  5.90s/it]Loss on epoch 6: 25.0351
 70%|##################################################4                     | 7/10 [00:39<00:16,  5.46s/it]Loss on epoch 7: 178.582
 80%|#########################################################6              | 8/10 [00:43<00:10,  5.13s/it]Loss on epoch 8: 144.15
 90%|################################################################8       | 9/10 [00:48<00:04,  4.90s/it]Loss on epoch 9: 40759.1
100%|#######################################################################| 10/10 [00:52<00:00,  4.78s/it]
```
