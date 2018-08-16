#! /usr/bin/env python

import argparse
import os
import pickle
import pandas as pd


OUT_DIR = 'p1save'


def get_parser(description='Run a trained machine learningn model in inference mode on new data'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-d", "--data",
                        help="data file to train on")
    parser.add_argument("-m", "--model_file",
                        help="saved trained model file")
    parser.add_argument("-k", "--keepcols", nargs='+', default=[],
                        help="columns from input data file to keep in prediction file; use 'all' to keep all original columns")
    parser.add_argument("-o", "--out_dir", default=OUT_DIR,
                        help="output directory")
    parser.add_argument("-p", "--prefix",
                        help="output prefix")
    parser.add_argument("-y", "--ycol", default=None,
                        help="0-based index or name of the column to be predicted")
    parser.add_argument("-C", "--ignore_categoricals", action='store_true',
                        help="ignore categorical feature columns")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    prefix = args.prefix or os.path.basename(args.data)
    prefix = os.path.join(args.out_dir, prefix)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    df = pd.read_table(args.data, engine='c')
    df_x = df.copy()
    cat_cols = df.select_dtypes(['object']).columns
    if args.ignore_categoricals:
        df_x[cat_cols] = 0
    else:
        df_x[cat_cols] = df_x[cat_cols].apply(lambda x: x.astype('category').cat.codes)

    keepcols = args.keepcols
    ycol = args.ycol
    if ycol:
        if ycol.isdigit():
            ycol = df_x.columns[int(ycol)]
        df_x = df_x.drop(ycol, axis=1)
        keepcols = [ycol] + keepcols
    else:
        df_x = df_x
    if 'all' in keepcols:
        keepcols = list(df.columns)

    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)

    x = df_x.as_matrix()
    y = model.predict(x)

    df_pred = df[keepcols]
    df_pred.insert(0, 'Pred', y)

    fname = '{}.predicted.tsv'.format(prefix)
    df_pred.to_csv(fname, sep='\t', index=False, float_format='%.3g')
    print('Predictions saved in {}\n'.format(fname))


if __name__ == '__main__':
    main()
