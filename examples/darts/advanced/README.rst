==============
DARTS Advanced
==============


Differentiable architecture search

This is an adaptation of Hanxiao Liu et al's DARTS algorithm, extending 
the work to handle convolutional neural networks for NLP problems and more. 
Details of the original authors' approach can be found in their 2019 ICLR paper_.

DARTS works by composing various neural net primitives, defined as Pytorch *nn.Modules*,
to create a larger directed acyclic graph (DAG) that is to be your model. This 
composition is differentiable as we take the softmax of the choice of primitive types 
at each layer of the network. To make this more clear, let's first define a few abstractions
in the algorithm:

1. **Primitve**: this is the fundamental block of computation, defined as an *nn.Module*. 
   At each layer of your network, one of these primitves will be chosen by taking the 
   softmax of all possible primitives at that layer. Examples could be a convolution block, 
   a linear layer, a skip connect, or anything that you can come up with (subject to a few 
   constraints).

2. **Cell**: this is an abstraction that holds each of the primitive types for level of your 
   network. This is where we perform the softmax over the possible primitive types.

3. **Nodes**: this is the level of abstraction that would normally be considered a layer in
   your network. It can contain one or more *Cells*.

4. **Architecture**: The abstraction that contains all nodes in the graph. This computes a 
   Hessian product with respect to the *alpha* parameters as defined in the paper. 

5. **Genotype**: genotypes are instances of a particular configuration of the graph. As the 
   optimization runs, and each cell computes the softmax over their primitive types, the final
   configuration of all nodes with their resulting primitive is a genotype.

In the DARTS algorithm, we define a number of primitives that we would like to compose together 
to form our neural network. The original paper started with 8 primitive types. These types 
were originally designed for a vision task, and largely consist of convolution type operations. 
We have since adapted these types for the *P3B5* benchmark, creating 1D convolution types for
our NLP tasks. If you would like to see how these primitives are defined, along with their 
necessary constructors used by DARTS, you can find them in 
`darts.modules.operations.conv.py`_.

These primitives are then contained within a cell, and one or more cells are contained within a 
node in the graph. DARTS then works by composing these nodes together and taking the softmax over
their primitives in each cell. Finally, the *Architecture* abstraction contains all nodes, and is
responsible for differentiating the composition of the nodes with respect to two *alpha* parameters
as defined in the paper. The end result is that we have a differentiable model that composes its 
components as the model is training.

As the optimization runs, the model will print the resulting loss with respect to a given *Genotype*.
The final model will be the *Genotype* with corresponding to the lowest loss.

Adnvanced Example
-----------------

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

Finally, to run this example:

.. code-block::

    python example.py

.. References
.. ----------
.. _paper: https://openreview.net/forum?id=S1eYHoC5FX
.. _darts.modules.operations.conv.py: ../../../common/darts/modules/operations/conv.py
.. _darts.modules.operations.linear.py: ../../../common/darts/modules.operations.linear.py
.. _operations.py: ./operations.py
