# Script to analyze perturbation by cluster
# Plot the perturbations by cluster
# Plot the pertubation centroids

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
directory = 'clusters_0911_0.5/'
orig_dataset = pickle.load(open("nt3.autosave.data.pkl", 'rb'))[0]
cf_dataset = pickle.load(open("threshold_0905.pkl", 'rb'))['perturbation vector']
for filename in os.listdir(directory):
    if filename.startswith("cf_class_0") or filename.startswith("cf_class_1") :
        data = pickle.load(open(os.path.join(directory, filename), 'rb'))
        x_range = np.arange(len(data['centroid perturb vector']))
        ind_in_cluster = data['sample indices in this cluster'][0:5]
        fig,ax = plt.subplots(3, figsize=(20,15))
        fig.suptitle("Perturbation Vectors for counterfactual class 1, cluster 1", fontsize=25)
        for i,ax_i in zip(ind_in_cluster,ax):
            d = cf_dataset[i]
            ax_i.plot(x_range, d, label='perturbation vector')
            ax_i.plot(x_range ,data['centroid perturb vector'], label='centroid')
            #ax_i.axhline(y=0.5*np.max(np.abs(d)), color='r', linestyle='-')
            #ax_i.axhline(y=-0.5*np.max(np.abs(d)), color='r', linestyle='-')
            ax_i.axvline(x=9603, color='r', linestyle='-', linewidth=5, alpha=0.3)

            ax_i.set_title("sample {}".format(i))
            ax_i.legend()
        fig.supxlabel("Feature index", fontsize=18)
        plt.savefig("centroids_{}.png".format(filename))

    else:
        continue
