=========
P3B6 BERT 
=========

``Note:`` this is currently under active development.

Finetuning BERT on synthetic data that imitates the Mimic dataset.


For a single node run without Horovod:

.. code-block:: console

    bsub launch_single.lsf

For a multi-node data parallel example using Horovod:

.. code-block:: console

    bsub horovod.lsf
