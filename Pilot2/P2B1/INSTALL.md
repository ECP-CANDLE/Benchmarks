## Installing Keras framework

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
