## Installing Keras framework

Here is an alternate set of directions for using the Spack tool
(spack.io) to install Keras and associated tools.

### Using spack

```
# Use the install directions at: http://spack.readthedocs.io/en/latest/
git clone https://github.com/spack/spack.git <path>/spack.git

# Put spack into your path
export SPACK_ROOT=/usr/workspace/wsa/vanessen/spack_test.git;
. $SPACK_ROOT/share/spack/setup-env.sh

# Install the set of packages required to run the CANDLE benchmarks
spack install candle-benchmarks %gcc@7.1.0
```

#### Activating Spack

```
# Activate all of these tools in the spack python environment
spack activate py-ipython
spack activate py-keras
spack activate py-matplotlib
spack activate py-tqdm
spack activate py-scikit-learn
spack activate py-mdanalysis
spack activate py-mpi4py
spack activate py-h5py

# Show what packages are activated in your python environment
spack extensions python

# Load the ipython environment into your path
module avail
module load py-ipython-5.1.0-gcc-7.1.0
module load python/2.7.13-gcc-7.1.0 jpeg/9b-gcc-7.1.0

# Lauch ipython and then run the example
ipython
[1]: run p2b1_baseline_keras1.py
```
