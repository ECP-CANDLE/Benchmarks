import argparse
import json
import pandas as pd
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_from', type=str, default='GDSC.h5',
                        help='Dataframe file name contains all data points')
    parser.add_argument('--plan', type=str, default='plan.json',
                        help='Plan data file')
    parser.add_argument('--node', type=str, default=None,
                        help='node number to execute')

    args, unparsed = parser.parse_known_args()
    return args, unparsed


def read_plan(filename, node):
    print("reading {} file for node {}".format(filename, node))
    with open(filename, 'r') as plan_file:
        plan = json.load(plan_file)
        if node in plan:
            return plan[node]
        else:
            raise Exception('Node index "{}" was not found in plan file'.format(node))


def build_masks(args, df):
    if args.node is None:
        raise Exception('Node id is not given')

    plan = read_plan(args.plan, args.node)
    mask = {}
    for partition in ['train', 'val']:
        _mask = df['Sample'] is None
        for i, element in enumerate(plan[partition]):
            cl_filter = element['cell']
            dr_filter = element['drug']
            __mask = df['Sample'].isin(cl_filter) & df['Drug1'].isin(dr_filter)
            _mask = _mask | __mask
        mask[partition] = _mask

    return mask['train'], mask['val']


def training_mask(df):
    return np.random.rand(len(df)) < 0.8


def build_dataframe(args):
    store = pd.HDFStore(args.dataframe_from, 'r')
    df_y = store.get('y_train')
    df_ds = store.get('x_train_0')
    df_cl = store.get('x_train_1')
    df_dd = store.get('x_train_2')
    df_fp = store.get('x_train_3')

    train_mask, val_mask = build_masks(args, df_y)

    y_train = pd.DataFrame(data=df_y[train_mask].reset_index(drop=True))
    y_val = pd.DataFrame(data=df_y[val_mask].reset_index(drop=True))

    x_train_0 = df_ds[train_mask].reset_index(drop=True)
    x_train_1 = df_cl[train_mask].reset_index(drop=True)
    x_train_2 = df_dd[train_mask].reset_index(drop=True)
    x_train_3 = df_fp[train_mask].reset_index(drop=True)

    x_val_0 = df_ds[val_mask].reset_index(drop=True)
    x_val_1 = df_cl[val_mask].reset_index(drop=True)
    x_val_2 = df_dd[val_mask].reset_index(drop=True)
    x_val_3 = df_fp[val_mask].reset_index(drop=True)

    # store
    store = pd.HDFStore('topN.uno.h5', 'w')
    store.put('y_train', y_train)
    store.put('y_val', y_val)
    store.put('x_train_0', x_train_0)
    store.put('x_train_1', x_train_1)
    store.put('x_train_2', x_train_2)
    store.put('x_train_3', x_train_3)
    store.put('x_val_0', x_val_0)
    store.put('x_val_1', x_val_1)
    store.put('x_val_2', x_val_2)
    store.put('x_val_3', x_val_3)


if __name__ == '__main__':
    parsed, unparsed = parse_arguments()
    build_dataframe(parsed)
