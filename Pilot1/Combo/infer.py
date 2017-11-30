#! /usr/bin/env python

from __future__ import division, print_function

import pandas as pd
import keras
from keras.models import Model
from tqdm import tqdm

import NCI60


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
    model = keras.models.load_model('saved.model.h5')
    model.load_weights('saved.weights.h5')
    model.summary()
    # df_expr, df_desc = prepare_data(sample_set='NCI60')
    df_expr, df_desc = prepare_data(sample_set='NCIPDM')
    # df_sample_ids = df_expr[['Sample']].head(10)
    # df_drug_ids = df_desc[['Drug']].head()
    df_sample_ids = df_expr[['Sample']].copy()
    df_drug_ids = df_desc[['Drug']].copy()
    df_all = cross_join3(df_sample_ids, df_drug_ids, df_drug_ids, suffixes=('1', '2'))

    step = 1024
    total = df_all.shape[0]
    for i in tqdm(range(0, total, step)):
        j = min(i+step, total)

        x_all_list = []
        df_x_all = pd.merge(df_all[['Sample']].iloc[i::j], df_expr, on='Sample', how='left')
        x_all_list.append(df_x_all.drop(['Sample'], axis=1).values)

        drugs = ['Drug1', 'Drug2']
        for drug in drugs:
            df_x_all = pd.merge(df_all[[drug]].iloc[i::j], df_desc, left_on=drug, right_on='Drug', how='left')
            x_all_list.append(df_x_all.drop([drug, 'Drug'], axis=1).values)

        y_pred = model.predict(x_all_list, batch_size=512, verbose=0).flatten()
        df_all.loc[i::j, 'Pred_Growth'] = y_pred

    # df_all['Pred_Growth'] = y_pred
    df_all.to_csv('tmp.csv', index=False, float_format='%.4f')


if __name__ == '__main__':
    main()
