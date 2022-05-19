import pandas as pd
import pickle
import argparse
import glob, os
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",type=str, help="Run folder")
    parser.add_argument("-c1", type=str, help="cluster 1 name")
    parser.add_argument("-c2", type=str, help="cluster 2 name")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    l1 = []
    l2 = []
    runs = glob.glob(args.f+"/EXP000/*/")
    print(runs)
    for r in runs:
        global_data = pd.read_csv(r+"training.log")
        val_abs = global_data['val_abstention'].iloc[-1]
        val_abs_acc = global_data['val_abstention_acc'].iloc[-1]
        cluster_data = pickle.load(open(r+"cluster_trace.pkl", "rb"))
        polluted_abs = cluster_data['Abs polluted']
        val_abs_cluster = cluster_data['Abs val cluster']
        val_abs_acc_cluster = cluster_data['Abs val acc']
        ratio = float(r[-4:-1])
        if args.c1 in r:
            l1.append([ratio, val_abs, val_abs_acc, val_abs_cluster, val_abs_acc_cluster, polluted_abs])
        elif args.c2 in r:
            l2.append([ratio, val_abs, val_abs_acc, val_abs_cluster, val_abs_acc_cluster, polluted_abs])

    df1 = pd.DataFrame(l1, columns=['Noise Fraction', 'Val Abs', 'Val Abs Acc', 'Val Abs Cluster', 'Val Abs Acc Cluster', 'Polluted Abs'])
    df2 = pd.DataFrame(l2, columns=['Noise Fraction', 'Val Abs', 'Val Abs Acc', 'Val Abs Cluster', 'Val Abs Acc Cluster', 'Polluted Abs'])
    print(df1)    
    df1.to_csv("cluster_1.csv")
    df2.to_csv("cluster_2.csv")
if __name__ == "__main__":
    main()
