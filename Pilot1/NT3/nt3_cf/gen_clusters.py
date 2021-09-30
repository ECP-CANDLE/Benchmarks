
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="threshod input file")
    parser.add_argument("-t_value", type=float, help="threshold value")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()

    thresholds_9 = pickle.load(open(args.t, 'rb'))

    perturb_vector=thresholds_9['perturbation vector']
    cf_class = thresholds_9['counterfactual class']
    indices = thresholds_9['sample index']

    # split by class
    perturb_vector_0=[]
    perturb_vector_1=[]
    indices_0 = []
    indices_1 = []
    for i,j,k in zip(perturb_vector, cf_class, indices):
        if j==0:
            perturb_vector_0.append(i)
            indices_0.append(k)
        else:
            perturb_vector_1.append(i)
            indices_1.append(k)

    indices_0 = np.array(indices_0)
    indices_1 = np.array(indices_1)
    sil = []
    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(perturb_vector_0)
        labels = kmeans.labels_
        sil.append(silhouette_score(perturb_vector_0, labels, metric='euclidean'))
    plt.plot(np.arange(2, kmax+1), sil)
    plt.title("Silhouette scores to determine optimal k")
    plt.xlabel("k")
    plt.show()
    k = np.argmax(sil) + 2
    print(k)
    data_2D = PCA(2).fit_transform(perturb_vector_0)
    kmeans_0 = KMeans(n_clusters=k).fit(perturb_vector_0)
    labels_0 = kmeans_0.labels_
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(k):
        plt.scatter(data_2D[:,0][labels_0==i], data_2D[:,1][labels_0==i], c=colors[i%len(colors)])
    plt.title("CF 0 KMeans clusters with 2D PCA")
    plt.savefig("CF_0.png")

    sil=[]
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(perturb_vector_1)
        labels = kmeans.labels_
        sil.append(silhouette_score(perturb_vector_1, labels, metric='euclidean'))
    plt.plot(np.arange(2, kmax+1), sil)
    plt.title("Silhouette scores to determine optimal k")
    plt.xlabel("k")
    plt.show()
    k = np.argmax(sil) + 2
    print(k)
    data_2D = PCA(2).fit_transform(perturb_vector_1)
    kmeans_1 = KMeans(n_clusters=k).fit(perturb_vector_1)
    labels_1 = kmeans_1.labels_
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(k):
        plt.scatter(data_2D[:,0][labels_1==i], data_2D[:,1][labels_1==i], c=colors[i%len(colors)])
    plt.title("CF 1 KMeans clusters with 2D PCA")
    plt.savefig("CF_1.png")

for i in range(len(kmeans_0.cluster_centers_)):
    diff_0=kmeans_0.cluster_centers_[i]
    max_value = np.max(np.abs(diff_0))
    ind_pos = np.where(diff_0 > args.t_value*max_value)
    ind_neg = np.where(diff_0 < -1*args.t_value*max_value)
    output = {'centroid perturb vector': diff_0,
                 'positive threshold indices':ind_pos,
                 'negative threshold indices':ind_neg,
                 'sample indices in this cluster':indices_0[labels_0==i]}
    print(output)
    pickle.dump(output,
                open("cf_class_0_cluster{}.pkl".format(i), "wb"))

for i in range(len(kmeans_1.cluster_centers_)):
    diff_1=kmeans_1.cluster_centers_[i]
    max_value = np.max(np.abs(diff_1))
    ind_pos = np.where(diff_1 > args.t_value*max_value)
    ind_neg = np.where(diff_1 < -1*args.t_value*max_value)
    output = {'centroid perturb vector': diff_1,
                 'positive threshold indices':ind_pos,
                 'negative threshold indices':ind_neg,
                 'sample indices in this cluster':indices_1[labels_1==i]}
    print(output)
    pickle.dump(output,
                open("cf_class_1_cluster{}.pkl".format(i), "wb"))
