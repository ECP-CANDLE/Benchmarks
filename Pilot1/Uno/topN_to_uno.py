import argparse
import os
import json
from collections import OrderedDict
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
    parser.add_argument('--cell_feature_selection', default=None,
                        help='Plain text list for cell feature filtering. one item per line')
    parser.add_argument('--drug_feature_selection', default=None,
                        help='Plain text list for drug feature filtering. one item per line')
    parser.add_argument('--output', type=str, default='topN.uno.h5',
                        help='output filename')

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


def build_masks(args, df):
    if args.node is None:
        print('node is None. Generate Random split')
        mask = get_random_mask(df)
        return mask, ~mask

    print('from new build_mask: {} {} {}'.format(args.plan, args.node, args.incremental))
    import plangen
    plan = read_plan(args.plan, None)
    ids = {}
    mask = {}
    _, _, ids['train'], ids['val'] = plangen.get_subplan_features(plan, args.node, args.incremental)

    for partition in ['train', 'val']:
        _mask = df['Sample'] == None  # noqa Should keep == operator here. This is a pandas operation.
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


def build_masks_w_holdout(args, df):
    if args.node is None:
        print('node is None. Generate Random split')
        mask = get_random_mask(df)
        return mask, ~mask

    print('from new build_mask: {} {} {}'.format(args.plan, args.node, args.incremental))
    import plangen
    plan = read_plan(args.plan, None)
    ids = {}
    mask = {}
    _, _, ids['train'], ids['val'] = plangen.get_subplan_features(plan, args.node, args.incremental)

    # holdout
    from sklearn.model_selection import ShuffleSplit
    idx_vec = df.index.to_numpy(copy=True)
    splitter = ShuffleSplit(n_splits=1, test_size=0.1, random_state=123)
    tr_vl_id, test_id = next(splitter.split(X=idx_vec))
    mask['test'] = df.index.isin(test_id)

    # new df
    df_new = df.iloc[tr_vl_id, :]

    for partition in ['train', 'val']:
        _mask = df['Sample'] == None  # noqa Should keep == operator here. This is a pandas operation.
        for i in range(len(ids[partition]['cell'])):
            if 'cell' in ids[partition] and 'drug' in ids[partition]:
                cl_filter = ids[partition]['cell'][i]
                dr_filter = ids[partition]['drug'][i]
                __mask = df_new['Sample'].isin(cl_filter) & df_new['Drug1'].isin(dr_filter)
            elif 'cell' in ids[partition]:
                cl_filter = ids[partition]['cell'][i]
                __mask = df_new['Sample'].isin(cl_filter)
            elif 'drug' in ids[partition]:
                dr_filter = ids[partition]['drug'][i]
                __mask = df_new['Drug1'].isin(dr_filter)
            _mask = _mask | __mask
        mask[partition] = _mask
    return mask['train'], mask['val'], mask['test']


def get_random_mask(df):
    return np.random.rand(len(df)) < 0.8


def read_dataframe(args):
    _, ext = os.path.splitext(args.dataframe_from)
    if ext == '.h5' or ext == '.hdf5':
        store = pd.HDFStore(args.dataframe_from, 'r')
        df = store.get('df')
        store.close()
    elif ext == '.feather':
        df = pd.read_feather(args.dataframe_from).fillna(0)
    elif ext == '.parquet':
        df = pd.read_parquet(args.dataframe_from).fillna(0)
    else:
        df = pd.read_csv(args.dataframe_from, low_memory=False, na_values='na').fillna(0)

    df.rename(columns={'CELL': 'Sample', 'DRUG': 'Drug1'}, inplace=True)
    df_y = df[['AUC', 'Sample', 'Drug1']]

    cols = df.columns.to_list()
    cl_columns = list(filter(lambda x: x.startswith('GE_'), cols))
    dd_columns = list(filter(lambda x: x.startswith('DD_'), cols))

    if args.cell_feature_selection is not None:
        features = set(pd.read_csv(args.cell_feature_selection, skip_blank_lines=True, header=None)[0].to_list())
        cl_columns = list(filter(lambda x: x in features, cl_columns))

    if args.drug_feature_selection is not None:
        features = set(pd.read_csv(args.drug_feature_selection, skip_blank_lines=True, header=None)[0].to_list())
        dd_columns = list(filter(lambda x: x in features, dd_columns))

    df_cl = df.loc[:, cl_columns]
    df_dd = df.loc[:, dd_columns]

    return df_y, df_cl, df_dd


def build_dataframe(args):
    df_y, df_cl, df_dd = read_dataframe(args)

    if args.fold is not None:
        tr_id = pd.read_csv('{}_tr_id.csv'.format(args.fold))
        vl_id = pd.read_csv('{}_vl_id.csv'.format(args.fold))
        tr_idx = tr_id.iloc[:, 0].dropna().values.astype(int).tolist()
        vl_idx = vl_id.iloc[:, 0].dropna().values.astype(int).tolist()
        tr_vl_idx = tr_idx + vl_idx

        y_train = df_y.iloc[tr_idx, :].reset_index(drop=True)
        y_val = df_y.iloc[vl_idx, :].reset_index(drop=True)
        y_test = df_y.loc[~df_y.index.isin(tr_vl_idx), :].reset_index(drop=True)

        x_train_0 = df_cl.iloc[tr_idx, :].reset_index(drop=True)
        x_train_1 = df_dd.iloc[tr_idx, :].reset_index(drop=True)
        x_train_1.columns = [''] * len(x_train_1.columns)

        x_val_0 = df_cl.iloc[vl_idx, :].reset_index(drop=True)
        x_val_1 = df_dd.iloc[vl_idx, :].reset_index(drop=True)
        x_val_1.columns = [''] * len(x_val_1.columns)

        x_test_0 = df_cl.iloc[~df_cl.index.isin(tr_vl_idx), :].reset_index(drop=True)
        x_test_1 = df_dd.iloc[~df_dd.index.isin(tr_vl_idx), :].reset_index(drop=True)
        x_test_1.columns = [''] * len(x_val_1.columns)
    else:
        # train_mask, val_mask = build_masks(args, df_y)
        train_mask, val_mask, test_mask = build_masks_w_holdout(args, df_y)

        y_train = pd.DataFrame(data=df_y[train_mask].reset_index(drop=True))
        y_val = pd.DataFrame(data=df_y[val_mask].reset_index(drop=True))
        y_test = pd.DataFrame(data=df_y[test_mask].reset_index(drop=True))

        x_train_0 = df_cl[train_mask].reset_index(drop=True)
        x_train_1 = df_dd[train_mask].reset_index(drop=True)
        x_train_1.columns = [''] * len(x_train_1.columns)

        x_val_0 = df_cl[val_mask].reset_index(drop=True)
        x_val_1 = df_dd[val_mask].reset_index(drop=True)
        x_val_1.columns = [''] * len(x_val_1.columns)

        x_test_0 = df_cl[test_mask].reset_index(drop=True)
        x_test_1 = df_dd[test_mask].reset_index(drop=True)
        x_test_1.columns = [''] * len(x_test_1.columns)

    # store
    store = pd.HDFStore(args.output, 'w', complevel=9, complib='blosc:snappy')
    store.put('y_train', y_train, format='table')
    store.put('y_val', y_val, format='table')
    store.put('x_train_0', x_train_0, format='table')
    store.put('x_train_1', x_train_1, format='table')
    store.put('x_val_0', x_val_0, format='table')
    store.put('x_val_1', x_val_1, format='table')

    # keep input feature list and shape
    cl_width = len(df_cl.columns)
    dd_width = len(df_dd.columns)
    store.put('model', pd.DataFrame())
    store.get_storer('model').attrs.input_features = OrderedDict([('cell.rnaseq', 'cell.rnaseq'), ('drug1.descriptors', 'drug.descriptors')])
    store.get_storer('model').attrs.feature_shapes = OrderedDict([('cell.rnaseq', (cl_width,)), ('drug.descriptors', (dd_width,))])

    if y_test is not None:
        store.put('y_test', y_test, format='table')
        store.put('x_test_0', x_test_0, format='table')
        store.put('x_test_1', x_test_1, format='table')
    store.close()


if __name__ == '__main__':
    parsed, unparsed = parse_arguments()
    build_dataframe(parsed)
