import argparse
import os
import json
import pandas as pd
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_from', type=str, default='top21_dataframe_8x8.csv',
                        help='Dataframe file name contains all data points')
    parser.add_argument('--plan', type=str, default='plan.json',
                        help='Plan data file')
    parser.add_argument('--node', type=str, default=None,
                        help='node number to execute')
    parser.add_argument('--incremental', action='store_true',
                        help='True for building dataset incrementally')
    parser.add_argument('--fold', type=str, default=None,
                        help='pre-calculated indexes for cross fold validation')

    args, unparsed = parser.parse_known_args()
    return args, unparsed


def read_plan(filename, node):
    print("reading {} file for node {}".format(filename, node))
    with open(filename, 'r') as plan_file:
        plan = json.load(plan_file)
        if node is None:
            return plan

        if node in plan:
            return plan[node]
        else:
            raise Exception('Node index "{}" was not found in plan file'.format(node))


# def build_masks(args, df):
#     if args.node is None:
#        print('node is None. Generate Random split')
#        mask = training_mask(df)
#        return mask, ~mask
#
#    plan = read_plan(args.plan, args.node)
#    mask = {}
#    for partition in ['train', 'val']:
#        _mask = df['Sample'] is None
#        for i, element in enumerate(plan[partition]):
#            cl_filter = element['cell']
#            dr_filter = element['drug']
#            __mask = df['Sample'].isin(cl_filter) & df['Drug1'].isin(dr_filter)
#            _mask = _mask | __mask
#        mask[partition] = _mask
#
#    return mask['train'], mask['val']


def build_masks(args, df):
    if args.node is None:
        print('node is None. Generate Random split')
        mask = training_mask(df)
        return mask, ~mask

    print('from new build_mask: {} {} {}'.format(args.plan, args.node, args.incremental))
    import plangen
    plan = read_plan(args.plan, None)
    ids = {}
    mask = {}
    _, _, ids['train'], ids['val'] = plangen.get_subplan_features(plan, args.node, args.incremental)

    for partition in ['train', 'val']:
        _mask = df['Sample'] is None
        for i in range(len(ids[partition]['cell'])):
            if 'cell' in ids[partition] and 'drug' in ids[partition]:
                cl_filter = ids[partition]['cell'][i]
                dr_filter = ids[partition]['drug'][i]
                __mask = df['Sample'].isin(cl_filter) & df['Drug1'].isin(dr_filter)
            elif 'cell' in ids[partition]:
                cl_filter = ids[partition]['cell'][i]
                __mask = df['Sample'].isin(cl_filter)
            elif 'drug' in ids[partition]:
                dr_filter = ids[partition]['drug'][i]
                __mask = df['Drug1'].isin(dr_filter)

            _mask = _mask | __mask
        mask[partition] = _mask
    return mask['train'], mask['val']


def training_mask(df):
    return np.random.rand(len(df)) < 0.8


def read_dataframe_from_csv(args):
    df = pd.read_csv(args.dataframe_from, low_memory=False, na_values='na').fillna(0)
    df.rename(columns={'CELL': 'Sample', 'DRUG': 'Drug1'}, inplace=True)
    df_y = df[['AUC', 'Sample', 'Drug1']]

    cols = df.columns.to_list()
    cl_columns = list(filter(lambda x: x.startswith('GE_'), cols))
    dd_columns = list(filter(lambda x: x.startswith('DD_'), cols))

    df_cl = df.loc[:, cl_columns]
    df_dd = df.loc[:, dd_columns]

    return df_y, df_cl, df_dd


def read_dataframe_from_feather(args):
    df = pd.read_feather(args.dataframe_from).fillna(0)
    df.rename(columns={'CELL': 'Sample', 'DRUG': 'Drug1'}, inplace=True)
    df_y = df[['AUC', 'Sample', 'Drug1']]

    cols = df.columns.to_list()
    cl_columns = list(filter(lambda x: x.startswith('GE_'), cols))
    dd_columns = list(filter(lambda x: x.startswith('DD_'), cols))

    df_cl = df.loc[:, cl_columns]
    df_dd = df.loc[:, dd_columns]

    return df_y, df_cl, df_dd


def read_dataframe_from_hdf(args):
    store = pd.HDFStore(args.dataframe_from, 'r')
    df = store.get('df')
    df.rename(columns={'CELL': 'Sample', 'DRUG': 'Drug1'}, inplace=True)
    df_y = df[['AUC', 'Sample', 'Drug1']]

    cols = df.columns.to_list()
    cl_columns = list(filter(lambda x: x.startswith('GE_'), cols))
    dd_columns = list(filter(lambda x: x.startswith('DD_'), cols))

    df_cl = df.loc[:, cl_columns]
    df_dd = df.loc[:, dd_columns]

    return df_y, df_cl, df_dd


def build_dataframe(args):
    _, ext = os.path.splitext(args.dataframe_from)
    if ext == '.h5' or ext == '.hdf5':
        df_y, df_cl, df_dd = read_dataframe_from_hdf(args)
    elif ext == '.feather':
        df_y, df_cl, df_dd = read_dataframe_from_feather(args)
    else:
        df_y, df_cl, df_dd = read_dataframe_from_csv(args)

    if args.fold is not None:
        tr_id = pd.read_csv('{}_tr_id.csv'.format(args.fold))
        vl_id = pd.read_csv('{}_vl_id.csv'.format(args.fold))
        tr_idx = tr_id.iloc[:, 0].dropna().values.astype(int).tolist()
        vl_idx = vl_id.iloc[:, 0].dropna().values.astype(int).tolist()

        y_train = df_y.iloc[tr_idx, :]
        y_val = df_y.iloc[vl_idx, :]

        x_train_0 = df_cl.iloc[tr_idx, :]
        x_train_1 = df_dd.iloc[tr_idx, :]
        x_train_1.columns = [''] * len(x_train_1.columns)

        x_val_0 = df_cl.iloc[vl_idx, :]
        x_val_1 = df_dd.iloc[vl_idx, :]
        x_val_1.columns = [''] * len(x_val_1.columns)
    else:
        train_mask, val_mask = build_masks(args, df_y)

        y_train = pd.DataFrame(data=df_y[train_mask].reset_index(drop=True))
        y_val = pd.DataFrame(data=df_y[val_mask].reset_index(drop=True))

        x_train_0 = df_cl[train_mask].reset_index(drop=True)
        x_train_1 = df_dd[train_mask].reset_index(drop=True)
        x_train_1.columns = [''] * len(x_train_1.columns)

        x_val_0 = df_cl[val_mask].reset_index(drop=True)
        x_val_1 = df_dd[val_mask].reset_index(drop=True)
        x_val_1.columns = [''] * len(x_val_1.columns)

    # store
    store = pd.HDFStore('topN.uno.h5', 'w', complevel=9, complib='blosc:snappy')
    store.put('y_train', y_train, format='table')
    store.put('y_val', y_val, format='table')
    store.put('x_train_0', x_train_0, format='table')
    store.put('x_train_1', x_train_1, format='table')
    store.put('x_val_0', x_val_0, format='table')
    store.put('x_val_1', x_val_1, format='table')


if __name__ == '__main__':
    parsed, unparsed = parse_arguments()
    build_dataframe(parsed)
