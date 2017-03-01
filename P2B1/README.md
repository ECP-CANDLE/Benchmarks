## P2B1: Autoencoder Compressed Representation for Molecular Dynamics Simulation Data

**Overview**: Generate automatically extracted features representing molecular simulation data

**Relationship to core problem**: Establish framework for building future tools using learned features

**Expected outcome**: Improvement in the understanding of protein formation and easing of the handling large-scale molecular dynamics output

### Benchmark Specs Requirements

#### Description of the Data
* Data source: MD Simulation output as PDB files (coarse-grained bead simulation)
* Input dimensions: ~1.26e6 per time step (6000 lipids x 30 beads per lipid x (position + velocity + type))
* Output dimensions: 500
* Latent representation dimension:
* Sample size: O(10^6) for simulation requiring O(10^8) time steps
* Notes on data balance and other issues: unlabeled data with rare events

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

Using virtualenv

```
cd P2B1
workon keras
python __main__.py --train --home-dir=${HOME}/.virtualenvs/keras/lib/python2.7/site-packages
```
