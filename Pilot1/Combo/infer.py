#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import pandas as pd
import keras
from keras import backend as K
from keras.models import Model
from tqdm import tqdm

import NCI60


def get_parser(description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--sample_set',
                        default='NCIPDM',
                        help='cell sample set: NCI60, NCIPDM, GDSC, ...')
    parser.add_argument('-d', '--drug_set',
                        default='ALMANAC',
                        help='drug set: ALMANAC, GDSC, NCI_IOA_AOA, ...')
    parser.add_argument('-z', '--batch_size', type=int,
                        default=10000,
                        help='batch size')
    parser.add_argument('--step', type=int,
                        default=10000,
                        help='number of rows to inter in each step')
    parser.add_argument('-m', '--model_file',
                        default='saved.model.h5',
                        help='trained model file')
    parser.add_argument('-w', '--weights_file',
                        default='saved.weights.h5',
                        help='trained weights file (loading model file alone sometimes does not work in keras)')

    return parser


def cross_join(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    # res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res


def cross_join3(df1, df2, df3, **kwargs):
    return cross_join(cross_join(df1, df2), df3, **kwargs)


def prepare_data(sample_set='NCI60', drug_set='ALMANAC'):
    df_expr = NCI60.load_sample_rnaseq(use_landmark_genes=True, sample_set=sample_set)
    df_old = NCI60.load_cell_expression_rnaseq(use_landmark_genes=True)
    df_desc = NCI60.load_drug_descriptors_new()
    return df_expr, df_desc


def main():
    description = 'Infer drug pair response from trained combo model.'
    parser = get_parser(description)
    args = parser.parse_args()

    model = keras.models.load_model(args.model_file)
    model.load_weights(args.weights_file)
    # model.summary()

    df_expr, df_desc = prepare_data(sample_set=args.sample_set, drug_set=args.drug_set)
    # df_expr, df_desc = prepare_data(sample_set='NCI60')
    # df_expr, df_desc = prepare_data(sample_set='NCIPDM')
    df_sample_ids = df_expr[['Sample']].head(10)
    df_drug_ids = df_desc[['Drug']].head()
    # df_sample_ids = df_expr[['Sample']].copy()
    # df_drug_ids = df_desc[['Drug']].copy()
    df_all = cross_join3(df_sample_ids, df_drug_ids, df_drug_ids, suffixes=('1', '2'))

    step = 100
    total = df_all.shape[0]
    for i in tqdm(range(0, total, step)):
        j = min(i+step, total)

        x_all_list = []
        df_x_all = pd.merge(df_all[['Sample']].iloc[i:j], df_expr, on='Sample', how='left')
        x_all_list.append(df_x_all.drop(['Sample'], axis=1).values)

        drugs = ['Drug1', 'Drug2']
        for drug in drugs:
            df_x_all = pd.merge(df_all[[drug]].iloc[i:j], df_desc, left_on=drug, right_on='Drug', how='left')
            x_all_list.append(df_x_all.drop([drug, 'Drug'], axis=1).values)

        y_pred = model.predict(x_all_list, batch_size=100, verbose=0).flatten()
        print(len(y_pred))
        df_all.loc[i:j-1, 'Pred_Growth'] = y_pred

    csv = 'comb_pred_{}_{}.csv'.format(args.sample_set, args.drug_set)
    df_all.to_csv(csv, index=False, float_format='%.4f')


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
