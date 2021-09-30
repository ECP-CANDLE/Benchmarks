import pickle
import numpy as np
import copy
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="threshold pickle file")
    parser.add_argument("-c1", type=list, help="cluster 1")
    parser.add_argument("-c2", type=list, help="cluster 2")
    parser.add_argument("-scale", type=float, help="scale factor for noise injection")
    parser.add_argument("-r", type=bool, help="flag to add random noise")
    args = parser.parse_args()
    return args
def random_noise(c1,c2,scale,size, cluster_inds):
    X_train, y_train, X_test, y_test = pickle.load(open("nt3.autosave.data.pkl", 'rb'))
    X_data = np.concatenate([X_train, X_test])
    y_data = np.concatenate([y_train, y_test])
    genes = np.random.choice(np.arange(X_data.shape[0]), replace=False, size=size)
    noise = np.random.normal(0,1,size)
    X_data_noise = copy.deepcopy(X_data)
    print(c1,c2)
    for p in np.arange(0.1,1.0, 0.1):
        for i in cluster_inds:
            for j in range(size):
                X_data_noise[i][genes[j]]+=noise[j]
        pickle.dump([X_data_noise, y_data, [], cluster_inds], open("nt3.data.random.scale_{}_cluster_{}_{}.noise_{}.pkl".format(scale,c1,c2,round(p,1)), "wb"))
    
def main():
    args = get_args()
    # For 2 clusters (with sparse injection feature vector) add CF noise to x% of samples
    X_train, y_train, X_test, y_test = pickle.load(open("nt3.autosave.data.pkl", 'rb'))
    threshold_dataset = pickle.load(open(args.t, 'rb'))
    perturb_dataset = threshold_dataset['perturbation vector']
    #failed index 
    perturb_dataset.insert(919, np.zeros(X_train.shape[1]))
    perturb_dataset = np.array(perturb_dataset)
    X_data = np.concatenate([X_train, X_test])
    y_data = np.concatenate([y_train, y_test])
    clusters = [(0,1),(1,1)]
    cluster_files = []
    for c in clusters:
        cluster_files.append(pickle.load(open("clusters_0911_0.5/cf_class_{}_cluster{}.pkl".format(c[0], c[1]), 'rb')))
    for i in range(len(cluster_files)):
        d=cluster_files[i]
        cluster_inds = d['sample indices in this cluster']
        random_noise(clusters[i][0],clusters[i][1],args.scale,20, cluster_inds)
        #return
        for p in np.arange(0.1,1.0, 0.1):
            print("p={}".format(p))
            X_data_noise = copy.deepcopy(X_data)
            # Full cf injection
            # Choose x% of the indices to be perturbed
            selector = np.random.choice(a=cluster_inds, replace=False, size = (int)(p*len(cluster_inds)))
            #print(perturb_dataset[selector])
            X_data_noise[selector]-= args.scale*perturb_dataset[selector][:,:,None]
            #print(np.sum(X_data_noise - X_data))
            pickle.dump([X_data_noise, y_data, selector, cluster_inds], open("nt3.data.scale_{}_cluster_{}_{}.noise_{}.pkl".format(args.scale,clusters[i][0], clusters[i][1], round(p,1)), "wb"))

            # Threshold cf injection
            inds = []
            print(d)
            for j in d['positive threshold indices'][0]:
                inds.append(j)
            for j in d['negative threshold indices'][0]:
                inds.append(j)
            print(len(inds))
            X_data_noise_2 = copy.deepcopy(X_data)
            for j in inds:
                perturb_dataset[:,j]=0
            X_data_noise_2[selector]-= args.scale*perturb_dataset[selector][:,:,None]
            pickle.dump([X_data_noise_2, y_data, selector, cluster_inds], open("nt3.data.threshold_scale_{}_cluster_{}_{}.noise_{}.pkl".format(args.scale, clusters[i][0], clusters[i][1], round(p,1)), "wb"))
            
if __name__ == "__main__":
    main()





# Save dataset file
