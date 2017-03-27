## P2B2: Autoencoder Compressed Representation for Molecular Dynamics Simulation Data

**Overview**: Cut down on manual inspection time for molecular simulation data

**Relationship to core problem**: Identify possible forming of cancer-causing proteins by molecular simulation

**Expected outcome**: Improvement in the understanding of protein formation and easing of the handling large-scale molecular dynamics output

### Benchmark Specs Requirements

#### Description of the Data
* Data source: MD Simulation output as PDB files (coarse-grained bead simulation)
* Input dimensions: ~1.26e6 per time step (6000 lipids x 30 beads per lipid x (position + velocity + type))
* Output dimensions: 1xN_Frame (N=100 hidden units)
* Latent representation dimension:
* Sample size: O(10^6) for simulation requiring O(10^8) time steps
* Notes on data balance and other issues: unlabeled data with rare events

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
python __main__.py  --home-dir=${HOME}/.virtualenvs/keras/lib/python2.7/site-packages --look-back 15 --train --epochs 20 --learning-rate 0.01 --cool --seed --batch-size 10 --seed
```
