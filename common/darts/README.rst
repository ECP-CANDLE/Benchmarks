=====
DARTS
=====

Differentiable architecture search

This is an adaptation of Hanxiao Liu et al's DARTS algorithm, extending 
the work to handle convolutional neural networks for NLP problems and more. 
Details of the original authors' approach can be found in their 2019 ICLR paper_.

Notes
-----

The following steps should be finished before merging the PR:

- [  ] Expert level `Network` with user defined primitives and stem
- [  ] Examples
- [  ] README overview of the library

Expert Level Network
--------------------

The user must define:

1. Fundamental operations
2. Ops constructor for fundamental operations
3. Primitives list

Draft
-----

.. code-block:: python

    class Network:
        """ Expert mode network """

        def __init__(self, stem, primitives, ops):
            self.stem = stem
            self.primitives = primitives
            self ops = ops

        def _helper_init(self, ...):
            """ Helper to construct the private member variables """
            raise NotImplementedError


.. References
.. ----------
.. _paper: https://openreview.net/forum?id=S1eYHoC5FX

