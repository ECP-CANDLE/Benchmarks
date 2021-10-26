import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, help="model file")
    parser.add_argument("-prefix", type=str, help="noise file prefix")
    parser.add_argument("-prefix_rand", type=str, help="random noise file prefix")
    parser.add_argument("-folder", type=str, help="folder path to noise files")
    parser.add_argument("-o", type=str, help="name of saved png")
    parser.add_argument("-n", type=str, help="name of cluster")
    args = parser.parse_args()
    return args
def main():
    args = get_args()
    model_nt3 = tf.keras.models.load_model(args.m)

    results = []
    for i in np.arange(0.1,1.0, 0.1):
        cf_dataset = pickle.load(open("{}_{}.pkl".format(args.prefix, round(i,2)), "rb"))
        X_cf_dataset = np.concatenate([cf_dataset[0], cf_dataset[1]])
        y_cf_dataset = np.concatenate([cf_dataset[2], cf_dataset[3]])
        #X_cf_dataset = cf_dataset[0]
        #y_cf_dataset = cf_dataset[1]
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
        cf_dataset = pickle.load(open("{}_{}.pkl".format(args.prefix_rand, round(i,2)), "rb"))
        X_cf_dataset = np.concatenate([cf_dataset[0], cf_dataset[1]])
        y_cf_dataset = np.concatenate([cf_dataset[2], cf_dataset[3]])
        #X_cf_dataset = cf_dataset[0]
        #y_cf_dataset = cf_dataset[1]
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
    plt.title("Model accuracy with counterfactual noise injection for {}".format(args.n))
    plt.savefig(args.o)

if __name__ == "__main__":
    main()
