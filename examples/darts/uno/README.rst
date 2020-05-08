=========
DARTS UNO
=========

Let's take a look at a look at using DARTS for the Pilot 1 Uno example. In the Uno
problem the task is to classify tumor dose response with respect to a few different 
data sources. For simplicity, we will use one source, Uno's gene data, to be used 
for this classification. 

The Uno models are typically fully connected deep networks. DARTS provides some basic linear network
primitives which can be found in `darts.modules.operations.linear.py`_. For simplicity, we will make 
use of those primitives for this example. To see how we can define new primitives, see the `advanced`_
example.

There are two main abstractions that we need to instantiate in order to get up and running:

* **LinearNetwork**:

.. code-block:: python

    LinearNetwork(input_dim, tasks, criterion, device)

The *LinearNetwork* takes a few parameters:

1. *input_dim* (int): the data input dimension
2. *tasks* (Dict[str, int]): a dictionary of classification tasks where the keys are the task names
   and the values are the number of classes for that task.
3. *criterion*: a Pytorch loss function
4. *device* (str): either "cpu" or "gpu"

* **Architecture**:

.. code-block:: python

    Architecture(model, args, device)

The *Architecture* expects the following arguments:

1. *model*: and instance of the *LinearNetwork*
2. *args*: an instance of argparse args containing the weight decay and momentum parameters for the 
   *Architecture*'s optimizer controlling the Hessian optimization.
3. *device* (str): "cpu" or "gpu"

Model training should familiar to those that are accustomed to using Pytorch with one small difference:

.. code-block:: python

    # ...
    for step, (data, target) in enumerate(trainloader):
        #...
        architecture.step(
            data, target, x_search, target_search, lr, optimizer, unrolled
        )
        # ...
    # ...

To understand what is going on here, recall that DARTS is a bi-level optimization procedure, 
where there are two Pytorch optimizers, one for the normal gradient step for our model weights, 
and another to for our *Architecture* to step in the composition of our neural net's nodes. The 
*architecture.step* function is then taking that composition step. It expects that we pass it our 
data and labels of the training set, but also the data and labels of our validation set. For 
simplicity of this tutorial, *x_search* and *target_search* are from our training set, but these 
would normally use a separate validation set.

Run the Example
---------------

.. code-block::

    python uno_example.py

.. References
.. ----------
.. _paper: https://openreview.net/forum?id=S1eYHoC5FX
.. _darts.modules.operations.conv.py: ../../../common/darts/modules/operations/conv.py
.. _darts.modules.operations.linear.py: ../../../common/darts/modules.operations.linear.py
.. _advanced: ../advanced
