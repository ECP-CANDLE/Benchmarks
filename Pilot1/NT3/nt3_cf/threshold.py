# Example run python threshold.py -d nt3.autosave.data.pkl -c small_cf.pkl -t 0.2 -o small_threshold.pkl
import pickle
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, 
                        help='data input file', required=True)
    parser.add_argument('-c', type=str, 
                        help='counterfactual input file', required=True)
    parser.add_argument('-o', type=str, 
                        help='output file', required=True)
    parser.add_argument('-t', type=float,
                        help='threshold value', required=True)

    args = parser.parse_args()
    return args

def threshold(t_value, X, y, cf):
    pos = []
    neg = []
    cf_classes = []
    inds = []
    diffs = []
    for i in range(len(cf)):
        test_y = X[i].flatten()
        test_cf = cf[i][1].flatten()
        
        diff = test_y-test_cf
        max_value = np.max(np.abs(diff))

        ind_pos = np.where(diff > t_value*max_value)
        ind_neg = np.where(diff < -t_value*max_value)

        cf_class = np.abs(1-np.argmax(y[i]))

        pos.append(ind_pos)
        neg.append(ind_neg)
        cf_classes.append(cf_class)
        inds.append(cf[i][0])
        diffs.append(diff)
            
    return pos,neg,cf_classes,inds, diffs

def main():
    args = get_args()
    with open(args.d, 'rb') as pickle_file:
        X_train,X_test, Y_train,Y_test = pickle.load(pickle_file)
        
    with open(args.c, 'rb') as pickle_file:
        cf = pickle.load(pickle_file)
    
    X = np.concatenate([X_train,X_test])
    Y = np.concatenate([Y_train, Y_test])
#     X=X_test
#     Y=Y_test
    pos,neg,cf_classes,inds, diff = threshold(args.t, X, Y, cf)
    
    # Note that sample index is here to keep track of counterfactuals that succeeded, counterfactuals that failed are not included here
    results = {'sample index': inds, 
               'positive threshold indices': pos, 
               'negative threshold indices':neg, 
               'counterfactual class':cf_classes,
               'perturbation vector': diff}
    pickle.dump(results, open(args.o, "wb"))
        
    
if __name__ == "__main__":
    main()
