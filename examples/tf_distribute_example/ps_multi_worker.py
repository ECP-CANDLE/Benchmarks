import multiprocessing
import os
import random
import portpicker
import tensorflow as tf
import sys


_index=sys.argv[1]
_type=sys.argv[2]



# config for 7 node cluster
# with 3 worker nodes
# with 2 parameter serving nodes
# with 1 evaluator node

# worker config
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"],
        "chief": ["host6:port"]
    },
    "task": {"type": "worker", "index": index}
})


# evaluator config
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "evaluator": ["host7:port"]
    },
    "task": {"type": "evaluator", "index": 0}
})








cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

if cluster_resolver.task_type in ("worker", "ps"):
  # Start a TensorFlow server and wait.
elif cluster_resolver.task_type == "evaluator":
  # Run sidecar evaluation
else:
  # Run the coordinator.

