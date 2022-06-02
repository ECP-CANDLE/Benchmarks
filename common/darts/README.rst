=====
DARTS
=====


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


.. References
.. ----------
.. _paper: https://openreview.net/forum?id=S1eYHoC5FX
.. _darts.modules.operations.conv.py: ../../../common/darts/modules/operations/conv.py
.. _darts.modules.operations.linear.py: ../../../common/darts/modules.operations.linear.py
.. _operations.py: ./operations.py
.. _Uno example: ../uno
