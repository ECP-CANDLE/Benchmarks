## Installing Keras framework

Here is an alternate set of directions for using the Spack tool
(spack.io) to install Keras and associated tools.

### Using spack

```
# Install the keras python tools
spack install py-keras ^py-theano +gpu

# Also include opencv with python support
spack install opencv@3.2.0 +python

# Install iPython so that you can play interactively
spack install py-ipython

# Add matplotlib with image support
spack install py-matplotlib +image
```

#### Activating Spack

```
# Activate all of these tools in the spack python environment
spack activate py-ipython
spack activate py-keras
spack activate py-matplotlib

# Load the ipython environment into your path
module avail
module load py-ipython-5.1.0-gcc-4.9.3

# Lauch ipython and then run the example
ipython
[1]: run p2b2_baseline_keras1.py
```
