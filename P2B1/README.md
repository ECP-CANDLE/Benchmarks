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

Using spack

```
# Install the keras python tools
spack install py-keras ^py-theano +gpu

# Also include opencv with python support
spack install opencv@3.2.0 +python

# Install iPython so that you can play interactively
spack install py-ipython

# Add matplotlib with image support
spack install py-matplotlib +image

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
[1]: run __main__.py --train --home-dir=/g/g19/vanessen/spack.git/opt/spack/linux-rhel7-ppc64le/gcc-4.9.3/py-ipython-5.1.0-3y6j6uookmr2spiokorkiskor5uhvig3/bin
```

### Scaling Options
* ```--case=FULL``` Design autoencoder for data frame with coordinates for all beads
* ```--case=CENTER``` Design autoencoder for data frame with coordinates of the center-of-mass
* ```--case=CENTERZ``` Design autoencoder for data frame with z-coordiate of the center-of-mass
