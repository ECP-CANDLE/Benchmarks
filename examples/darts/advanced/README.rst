==============
DARTS Advanced
==============

In this example we will take a look at how to define our own primitives to be handled by DARTS. If 
you have not read the `Uno example`_, I would recommend taking a look at that first. There we showed 
how we can use the built in primitives to DARTS. As reference, you can also look to see how those 
built it primitives are defined in `darts.modules.operations.linear.py`_ and 
`darts.modules.operations.conv.py`_.

In order to define custom networks to be handled by DARTS, you need to define a few things:

1. **Network Stem**: This is an *nn.Module* that takes in your input data, processes it in some way,
   and feeds its features of size *cell_dim* to your remaining network primitives. The parameter 
   *cell_dim* must be the input size for all of your primitives. Since DARTS can compose your primitives
   in *any* order, the input and output dimension of all of your primitives must be of size *cell_dim*.

2. **Primitives**: These *nn.Modules* are the basic building blocks for your network. They can be anything
   that you dream of, so long as their input and output dimensions are of size *cell_dim*.

3. **A constructor for your primitives**: This is a dictionary of lambda functions used to construct your
   network primitives. By convention, this is a dictionary called *OPS*. We will look at this a bit closer
   below.

Defining our Components
-----------------------

Let's take a look at the various pieces that we need to define. All of these components can be found in 
`operations.py`_.

Network Stem
------------

As we mentioned above, this is the module that is defined at the beginning of your network, mapping your
input data to *cell_dim*.

.. code-block:: python

    class Stem(nn.Module):
    """ Network stem

    This will always be the beginning of the network.
    DARTS will only recompose modules after the stem.
    For this reason, we define this separate from the
    other modules in the network.

    Args:
        input_dim: the input dimension for your data

        cell_dim: the intermediate dimension size for
                  the remaining modules of the network.
    """
    def __init__(self, in_channels: int=1, cell_dim: int=100, kernel_size=3):
        super(Stem, self).__init__()
        self.stem = nn.Conv2d(in_channels, cell_dim, kernel_size)

    def forward(self, x):
        return self.stem(x)

Primitives
----------

DARTS primitives are Pytorch *nn.Modules*. For this example, we have defined three primitives: *ConvBlock*,
*DilConv*, and the *Identity* (a skip layer). It is important to remember DARTS will try many different 
orderings of these primitives between *nodes*. Therefore, the imput and output dimensions of each of these 
primitives must be of size *cell_dim*. 

It is also important to know that DARTS expects the *Identity* function to be included in the primitives. 
This is so that DARTS can account for varying depths of neural networks. Since at each node, DARTS must choose
one primitive (choosing meaning taking the softmax over the primitives), having the no-op *Identity* means 
that we can optimize over the depth of the network. It would be possible to define a 100 layer network and
have the output *Genotype* be only a few layers deep. If we were to not include the *Identity*, every layer
would be some transformation of the previous layer's features, and we could run the risk of overparameterizing
our network.

A Constructor for our Primitives
--------------------------------

Since DARTS does not control what primitives you define, we need to provide it with a constructor for those
primitives. By convention, this is handled by a dictionary of lambda functions called *OPS*. The keys of this 
dictionary are the names of our primitives, and the values of the dictionary are lambda functions that 
construct those primitives. Let's take a look at the example's *OPS*:

.. code-block:: python

    """ DARTS operations contstructor """
    OPS = {
        'none'    : lambda c, stride, affine: Identity(),
        'conv_3'  : lambda c, stride, affine: ConvBlock(c, c, 3, stride),
        'dil_conv': lambda c, stride, affine: DilConv(c, c, 3, stride, 2, 2, affine=affine)
    }

As mentioned, the keys of *OPS* are the names we give to each of our primitives. These keys will be 
what DARTS uses when defining *Genotypes*. Note that the the lambda functions take three parameters: 
1. *c*, the number of channels (or features) of the layer; 2. *stride*, the stride for convolutions; and
3. *affine* whether to use affine transforms in batch normalization. These parameters are the default 
implementation of DARTS, and must be present. Any other hyperparameters of our custom primitives must be
given default values. One last thing to note: in order to keep things consistent, DARTS reserves the keyword
*none* for the *Identity* primitive. Again, this primitive must be included in any custom primitive set, and
it's key must be *none*. This method of constructing our primitives could be changed in future versions of 
DARTS to better acccommodate fancier primitives. As always, pull requests are welcome!

Putting it all Together
-----------------------

Once we have defined our stem, primitives, and our *OPS* constructor, we can that hand them over to DARTS:

.. code-block:: python

    model = darts.Network(
        stem, cell_dim=100, classifier_dim=676,
        ops=OPS, tasks=tasks, criterion=criterion, device=device
    ).to(device)

    architecture = darts.Architecture(model, args, device=device)

Note that we must specify the *classifier_dim* the number of input features from our primitives. Since each 
of the primitives must have the same number of input and output features, this will be the flattned number 
of features from any of your primitives. Since DARTS cannot know ahead of time what your primitives will be,
we must specify how many features will go into our final fully connected layer of the network.

Run the Example
---------------

First, make sure that you can get the example data by installing `torchvision`:

.. code-block::

    pip install torchvision

Then run the example with

.. code-block::

    python example.py

.. References
.. ----------
.. _paper: https://openreview.net/forum?id=S1eYHoC5FX
.. _darts.modules.operations.conv.py: ../../../common/darts/modules/operations/conv.py
.. _darts.modules.operations.linear.py: ../../../common/darts/modules.operations.linear.py
.. _operations.py: ./operations.py
.. _Uno example: ../uno
