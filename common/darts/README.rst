=====
DARTS
=====

Differentiable architecture search


Notes
-----

The following steps should be finished before merging the PR:

[] Expert level `Network` with user defined primitives and stem
[] Examples
[] README overview of the library

Expert Level Network
--------------------

The user must define:

1. Fundamental operations
2. Ops constructor for fundamental operations
3. Primitives list

Draft
-----

.. code-block:: python

    class Network(stem, primitives, ops):
        self.stem = stem
        self.primitives = primitives
        self ops = ops

    def _helper_init(self, ...):
        """ Helper to construct the private member variables """
        raise NotImplementedError

