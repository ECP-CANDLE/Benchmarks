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

#### Using virtualenv
```
cd P2B1
workon keras
python p2b1.py
```

#### Using Spack
```
# Activate all of these tools in the spack python environment
spack activate ipython
spack activate py-ipython
spack activate py-keras
spack activate py-matplotlib

# Load the ipython environment into your path
module avail
module load py-ipython-5.1.0-gcc-4.9.3-3y6j6uo

# Lauch ipython and then run the example
ipython
[1]: run p2b1.py
```

### Scaling Options
* ```--case=FULL``` Design autoencoder for data frame with coordinates for all beads
* ```--case=CENTER``` Design autoencoder for data frame with coordinates of the center-of-mass
* ```--case=CENTERZ``` Design autoencoder for data frame with z-coordiate of the center-of-mass

### Expected Results

```
$ python p2b1.py
Using Theano backend.
Reading Data Files... 3k_Disordered->3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir
Data Format: [Num Sample (2861), Num Molecules (4864), Num Atoms (12), Position + Molecule Tag (One-hot encoded) (6)]
Define the model and compile
using mlp network
Autoencoder Regression problem
--------------------------------------------------------------------------------------------------------------------
|           type | feature size | #Filters  | FilterSize |                        #params |                  #MACs |
--------------------------------------------------------------------------------------------------------------------
|          dense |  0x0         |     512   | 01x01     |  2.49 Mill,  9.50 MB (  46.7%) |    2.49 Mill (  46.7%) |
|          dense |  0x0         |     256   | 01x01     |  0.13 Mill,  0.50 MB (   2.5%) |    0.13 Mill (   2.5%) |
|          dense |  0x0         |     128   | 01x01     |  0.03 Mill,  0.12 MB (   0.6%) |    0.03 Mill (   0.6%) |
|          dense |  0x0         |      64   | 01x01     |  0.01 Mill,  0.03 MB (   0.2%) |    0.01 Mill (   0.2%) |
|          dense |  0x0         |      32   | 01x01     |  0.00 Mill,  0.01 MB (   0.0%) |    0.00 Mill (   0.0%) |
|          dense |  0x0         |      16   | 01x01     |  0.00 Mill,  0.00 MB (   0.0%) |    0.00 Mill (   0.0%) |
|          dense |  0x0         |      32   | 01x01     |  0.00 Mill,  0.00 MB (   0.0%) |    0.00 Mill (   0.0%) |
|          dense |  0x0         |      64   | 01x01     |  0.00 Mill,  0.01 MB (   0.0%) |    0.00 Mill (   0.0%) |
|          dense |  0x0         |     128   | 01x01     |  0.01 Mill,  0.03 MB (   0.2%) |    0.01 Mill (   0.2%) |
|          dense |  0x0         |     256   | 01x01     |  0.03 Mill,  0.12 MB (   0.6%) |    0.03 Mill (   0.6%) |
|          dense |  0x0         |     512   | 01x01     |  0.13 Mill,  0.50 MB (   2.5%) |    0.13 Mill (   2.5%) |
|          dense |  0x0         |    4864   | 01x01     |  2.49 Mill,  9.50 MB (  46.7%) |    2.49 Mill (  46.7%) |
--------------------------------------------------------------------------------------------------------------------
|++++++++++++++++++++++++++++++++++++++++++++++++|   5.33 Mill, 20.33 MB ( 100.0%)|    5.33 Mill ( 100.0%) |           |
--------------------------------------------------------------------------------------------------------------------
...
Loss on epoch 0: 12.2727
 33%|------------------------------------|                                                                     | 1/3 [00:28<00:57, 28.79s/it]
Loss on epoch 1: 11.5519
 67%|---------------------------------------------------------------|                                          | 2/3 [00:53<00:27, 27.44s/it]
Loss on epoch 2: 11.3828
100%|----------------------------------------------------------------------------------------------------------| 3/3 [01:17<00:00, 26.53s/it]
Cooling Learning Rate by factor of 10...
  0%|                                                                                                          | 0/3 [00:00<?, ?it/s]
Loss on epoch 0: 10.2965
 33%|------------------------------------|                                                                     | 1/3 [00:24<00:48, 24.49s/it]
Loss on epoch 1: 10.3103
 67%|---------------------------------------------------------------|                                          | 2/3 [00:49<00:24, 24.71s/it]
Loss on epoch 2: 10.3181
100%|----------------------------------------------------------------------------------------------------------| 3/3 [01:13<00:00, 24.58s/it]
Cooling Learning Rate by factor of 10...
  0%|                                                                                                          | 0/3 [00:00<?, ?it/s]
Loss on epoch 0: 9.9807
 33%|------------------------------------|                                                                     | 1/3 [00:24<00:48, 24.45s/it]
Loss on epoch 1: 10.0237
 67%|---------------------------------------------------------------|                                          | 2/3 [00:48<00:24, 24.47s/it]
Loss on epoch 2: 10.0357
100%|----------------------------------------------------------------------------------------------------------| 3/3 [01:14<00:00, 24.73s/it]
```
