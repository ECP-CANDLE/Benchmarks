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
$ python p2b1_baseline_keras1.py
Using Theano backend.
Reading Data Files... 3k_Disordered->3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir
Data Format: [Num Sample (2861), Num Molecules (3040), Num Atoms (12), Position + Molecule Tag (One-hot encoded) (6)]
Define the model and compile
using mlp network
Autoencoder Regression problem
--------------------------------------------------------------------------------------------------------------------
|           type | feature size | #Filters  | FilterSize |                        #params |                  #MACs |
--------------------------------------------------------------------------------------------------------------------
|          dense |  0x0         |     512   | 01x01     |  1.56 Mill,  5.94 MB (  45.0%) |    1.56 Mill (  45.0%) |
|          dense |  0x0         |     256   | 01x01     |  0.13 Mill,  0.50 MB (   3.8%) |    0.13 Mill (   3.8%) |
|          dense |  0x0         |     128   | 01x01     |  0.03 Mill,  0.12 MB (   0.9%) |    0.03 Mill (   0.9%) |
|          dense |  0x0         |      64   | 01x01     |  0.01 Mill,  0.03 MB (   0.2%) |    0.01 Mill (   0.2%) |
|          dense |  0x0         |      32   | 01x01     |  0.00 Mill,  0.01 MB (   0.1%) |    0.00 Mill (   0.1%) |
|          dense |  0x0         |      16   | 01x01     |  0.00 Mill,  0.00 MB (   0.0%) |    0.00 Mill (   0.0%) |
|          dense |  0x0         |      32   | 01x01     |  0.00 Mill,  0.00 MB (   0.0%) |    0.00 Mill (   0.0%) |
|          dense |  0x0         |      64   | 01x01     |  0.00 Mill,  0.01 MB (   0.1%) |    0.00 Mill (   0.1%) |
|          dense |  0x0         |     128   | 01x01     |  0.01 Mill,  0.03 MB (   0.2%) |    0.01 Mill (   0.2%) |
|          dense |  0x0         |     256   | 01x01     |  0.03 Mill,  0.12 MB (   0.9%) |    0.03 Mill (   0.9%) |
|          dense |  0x0         |     512   | 01x01     |  0.13 Mill,  0.50 MB (   3.8%) |    0.13 Mill (   3.8%) |
|          dense |  0x0         |    3040   | 01x01     |  1.56 Mill,  5.94 MB (  45.0%) |    1.56 Mill (  45.0%) |
--------------------------------------------------------------------------------------------------------------------
|++++++++++++++++++++++++++++++++++++++++++++++++|   3.46 Mill, 13.21 MB ( 100.0%)|    3.46 Mill ( 100.0%) |           |
--------------------------------------------------------------------------------------------------------------------
  0%|                                                                                                          | 0/3 [00:00<?, ?it/s]
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_01_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_02_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_03_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_04_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_05_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_06_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_07_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_08_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_09_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_10_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_11_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_12_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_13_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_14_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_15_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_16_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_17_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_18_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_19_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_20_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_21_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_22_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_23_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_24_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_25_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_26_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_27_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_28_outof_29.npy
../Data/common/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_29_outof_29.npy

Loss on epoch 0: 18.1642
 33%|------------------------------------|                                                                     | 1/3 [00:22<00:44, 22.09s/it]
Loss on epoch 1: 17.4444
 67%|---------------------------------------------------------------|                                          | 2/3 [00:37<00:19, 19.96s/it]
Loss on epoch 2: 17.5339
100%|----------------------------------------------------------------------------------------------------------| 3/3 [00:51<00:00, 18.40s/it]
Cooling Learning Rate by factor of 10...
  0%|                                                                                                          | 0/3 [00:00<?, ?it/s]
Loss on epoch 0: 15.2757
 33%|------------------------------------|                                                                     | 1/3 [00:15<00:31, 15.71s/it]
Loss on epoch 1: 15.3096
 67%|---------------------------------------------------------------|                                          | 2/3 [00:30<00:15, 15.44s/it]
Loss on epoch 2: 15.3504
100%|----------------------------------------------------------------------------------------------------------| 3/3 [00:45<00:00, 15.44s/it]
Cooling Learning Rate by factor of 10...
  0%|                                                                                                          | 0/3 [00:00<?, ?it/s]
Loss on epoch 0: 14.8456
 33%|------------------------------------|                                                                     | 1/3 [00:15<00:30, 15.09s/it]
Loss on epoch 1: 14.8246
 67%|---------------------------------------------------------------|                                          | 2/3 [00:30<00:15, 15.32s/it]
Loss on epoch 2: 14.8405
100%|----------------------------------------------------------------------------------------------------------| 3/3 [00:47<00:00, 15.66s/it]
```
