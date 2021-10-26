import pickle
import numpy as np
import copy
import argparse
import os
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="threshold pickle file")
    parser.add_argument("-c1", type=str, help="cluster 1 file")
    parser.add_argument("-c2", type=str, help="cluster 2 file")
    parser.add_argument("-scale", type=float, help="scale factor for noise injection")
    parser.add_argument("-r", type=bool, help="flag to add random noise")
    parser.add_argument("-o", type=str, help="folder for output files")
    parser.add_argument("-d", type=str, help="nt3 data file")
    parser.add_argument("-f", type=str, help="pickle file containing failed cf indices")
    args = parser.parse_args()
    return args

# Choose a random set of indices to inject cf noise into
def random_noise(s,scale,size, cluster_inds, args):
    X_train, X_test, y_train, y_test = pickle.load(open(args.d, 'rb'))
    #X_data, y_data = pickle.load(open(args.d, 'rb'))
    X_data = np.concatenate([X_train, X_test])
    genes = np.random.choice(np.arange(X_data.shape[0]), replace=False, size=size)
    noise = np.random.normal(0,1,size)
    X_data_noise = copy.deepcopy(X_data)
    s, _ = s.split(".")
    cluster_name = s[3:]
    for p in np.arange(0.1,1.0, 0.1):
        for i in cluster_inds:
            for j in range(size):
                X_data_noise[i][genes[j]]+=noise[j]
        # Now split back into train test for output                                                             
        X_train = X_data_noise[0:(int)(0.8*X_data.shape[0])]
        X_test = X_data_noise[(int)(0.8*X_data.shape[0]):]
        pickle.dump([X_train, X_test, y_train, y_test, [], cluster_inds], open("{}/nt3.data.random.scale_{}_{}.noise_{}.pkl".format(args.o,scale,cluster_name,round(p,1)), "wb"))
    
def main():
    args = get_args()
    isExist = os.path.exists(args.o)
    if not isExist:
        os.makedirs(args.o)
    # For 2 clusters (with sparse injection feature vector) add CF noise to x% of samples
    X_train, X_test, y_train, y_test = pickle.load(open(args.d, 'rb'))
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #X_data, y_data = pickle.load(open(args.d, 'rb')) 
    threshold_dataset = pickle.load(open(args.t, 'rb'))
    perturb_dataset = threshold_dataset['perturbation vector']
    
    
    #combine for easier indexing later
    X_data = np.concatenate([X_train, X_test])

    #account for failed indices
    failed_indices = pickle.load(open(args.f, 'rb'))[0]
    print(failed_indices)
    for i in failed_indices:
        perturb_dataset.insert(i, np.zeros(X_data.shape[1]))
    perturb_dataset = np.array(perturb_dataset)
    
    _, cf1 = os.path.split(args.c1)
    _, cf2 = os.path.split(args.c2)
    cluster_files = [cf1, cf2]
    for i in range(len(cluster_files)):
        print(cluster_files[i])
        d = pickle.load(open(cluster_files[i], "rb"))
        cluster_inds = d['sample indices in this cluster']
        if args.r:
            random_noise(cluster_files[i],args.scale,20, cluster_inds, args)
        
        # Sweep through percentages
        for p in np.arange(0.1,1.0, 0.1):
            print("p={}".format(p))
            X_data_noise = copy.deepcopy(X_data)
            
            #Full cf injection
            # Choose x% of the indices to be perturbed
            selector = np.random.choice(a=cluster_inds, replace=False, size = (int)(p*len(cluster_inds)))
            X_data_noise[selector]-= args.scale*perturb_dataset[selector][:,:,None]
            
            # Now split back into train test for output
            X_train = X_data_noise[0:(int)(0.8*X_data.shape[0])]
            X_test = X_data_noise[(int)(0.8*X_data.shape[0]):]

            s,_ = cluster_files[i].split(".")
            cluster_name = s[3:]
            pickle.dump([X_train, X_test, y_train, y_test, selector, cluster_inds], open("{}/nt3.data.scale_{}_{}.noise_{}.pkl".format(args.o, args.scale,cluster_name, round(p,1)), "wb"))

            # Add cf noise only to those indices that passed the threshold value (instead of the full cf profile)
            inds = []
            for j in d['positive threshold indices'][0]:
                inds.append(j)
            for j in d['negative threshold indices'][0]:
                inds.append(j)
            X_data_noise_2 = copy.deepcopy(X_data)
            
            for j in inds:
                perturb_dataset[:,j]=0
            X_data_noise_2[selector]-= args.scale*perturb_dataset[selector][:,:,None]
            
            # Now split back into train test
            X_train = X_data_noise_2[0:(int)(0.8*X_data.shape[0])]
            X_test = X_data_noise_2[(int)(0.8*X_data.shape[0]):]
            
            pickle.dump([X_train, X_test, y_train, y_test, selector, cluster_inds], open("{}/nt3.data.threshold.scale_{}_{}.noise_{}.pkl".format(args.o, args.scale, cluster_name, round(p,1)), "wb"))
            
if __name__ == "__main__":
    main()





# Save dataset file
