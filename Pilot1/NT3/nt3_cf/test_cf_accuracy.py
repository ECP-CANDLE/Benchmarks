import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from time import time
from alibi.explainers import CounterFactual, CounterFactualProto
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False
print(tf.test.is_gpu_available())
import pickle

model_nt3 = tf.keras.models.load_model('/vol/ml/shahashka/xai-geom/nt3/nt3.autosave.model')
# results = []
# for i in np.arange(0.1,1.0, 0.1):
#     cf_dataset = pickle.load(open("nt3.data.scale_1.0.cluster_0_1.noise_{}.pkl".format(round(i,2)), "rb"))
#     X_cf_dataset = cf_dataset[0]
#     y_cf_dataset = cf_dataset[1]
#     cluster_inds = cf_dataset[-1]
#     print(model_nt3.metrics_names)
#     acc = model_nt3.evaluate(X_cf_dataset, y_cf_dataset)
#     cluster_acc = model_nt3.evaluate(X_cf_dataset[cluster_inds], y_cf_dataset[cluster_inds])
#     print(i, acc, cluster_acc)
#     results.append([acc[1], cluster_acc[1]])
# plt.plot(np.arange(0.1,1.0,0.1), results[:,0], label="accuracy", marker='o')
# plt.plot(np.arange(0.1,1.0, 0.1), results[:,1], label="cluster accuracy", marker='o')

results = []
for i in np.arange(0.1,1.0, 0.1):
    cf_dataset = pickle.load(open("nt3.data.threshold_scale_1.0_cluster_0_1.noise_{}.pkl".format(round(i,2)), "rb"))
    X_cf_dataset = cf_dataset[0]
    y_cf_dataset = cf_dataset[1]
    cluster_inds = cf_dataset[-1]
    print(model_nt3.metrics_names)
    acc = model_nt3.evaluate(X_cf_dataset, y_cf_dataset)
    cluster_acc = model_nt3.evaluate(X_cf_dataset[cluster_inds], y_cf_dataset[cluster_inds])
    print(i, acc, cluster_acc)
    results.append([acc[1], cluster_acc[1]])
results = np.array(results)
plt.plot(np.arange(0.1,1.0,0.1), results[:,0], label="accuracy", marker='o')
plt.plot(np.arange(0.1,1.0, 0.1), results[:,1], label="cluster accuracy", marker='o')

results = []
for i in np.arange(0.1,1.0, 0.1):
    cf_dataset = pickle.load(open("nt3.data.random.scale_1.0_cluster_0_1.noise_{}.pkl".format(round(i,2)), "rb"))
    X_cf_dataset = cf_dataset[0]
    y_cf_dataset = cf_dataset[1]
    cluster_inds = cf_dataset[-1]
    print(model_nt3.metrics_names)
    acc = model_nt3.evaluate(X_cf_dataset, y_cf_dataset)
    cluster_acc = model_nt3.evaluate(X_cf_dataset[cluster_inds], y_cf_dataset[cluster_inds])
    print(i, acc, cluster_acc)
    results.append([acc[1], cluster_acc[1]])
results = np.array(results)
plt.plot(np.arange(0.1,1.0,0.1), results[:,0], label="accuracy with Gaussian noise", marker='o')
plt.plot(np.arange(0.1,1.0, 0.1), results[:,1], label="cluster accuracy with Gaussian noise", marker='o')


plt.xlabel("Noise fraction in cluster")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model accuracy with counterfactual noise injection for class 0, cluster 1")
plt.savefig("abstract_plot.png")
