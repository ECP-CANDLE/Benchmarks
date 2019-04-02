#! /usr/bin/env python

import argparse

import pandas as pd
import numpy as np
import keras

from uno_data import DataFeeder
from uno_baseline_keras2 import evaluate_prediction
import candle_keras


def log_evaluation(metric_outputs, description='Comparing y_true and y_pred:'):
    print(description)
    for metric, value in metric_outputs.items():
        print('  {}: {:.4f}'.format(metric, value))


def get_parser():
    parser = argparse.ArgumentParser(description='Uno infer script')
    parser.add_argument("--data",
                        help="data file to infer on. expect exported file from uno_baseline_keras2.py")
    parser.add_argument("--model_file", help="json model description file")
    parser.add_argument("--weights_file", help="model weights file")
    parser.add_argument("--partition", default='all',
                        choices=['train', 'val', 'all'], help="partition of test dataset")
    parser.add_argument("-n", "--n_pred", type=int, default=1, help="the number of predictions to make")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    candle_keras.register_permanent_dropout()
    if args.model_file.split('.')[-1] == 'json':
        with open(args.model_file, 'r') as model_file:
            model_json = model_file.read()
            model = keras.models.model_from_json(model_json)
            model.load_weights(args.weights_file)
    else:
        model = keras.models.load_model(args.model_file)

    model.summary()

    cv_pred_list = []
    cv_y_list = []
    df_pred_list = []
    for cv in range(args.n_pred):
        cv_pred = []
        dataset = ['train', 'val'] if args.partition == 'all' else [args.partition]
        for partition in dataset:
            test_gen = DataFeeder(filename=args.data, partition=partition, batch_size=1024)
            y_test_pred = model.predict_generator(test_gen, test_gen.steps)
            y_test_pred = y_test_pred.flatten()

            df_y = test_gen.get_response(copy=True)
            y_test = df_y['Growth'].values

            df_pred = df_y.assign(PredictedGrowth=y_test_pred, GrowthError=y_test_pred-y_test)
            df_pred_list.append(df_pred)
            test_gen.close()

            if cv == 0:
                cv_y_list.append(df_pred['Growth'])
            cv_pred.append(y_test_pred)
        cv_pred_list.append(np.concatenate(cv_pred))

    df_pred = pd.concat(df_pred_list)
    cv_y = pd.concat(cv_y_list)

    # save to tsv
    df_pred.sort_values(['Sample', 'Drug1', 'Drug2', 'Dose1', 'Dose2', 'Growth'], inplace=True)
    df_pred.to_csv('uno_pred.all.tsv', sep='\t', index=False, float_format='%.4g')

    df_sum = pd.DataFrame()
    df_sum['Growth'] = cv_y
    df_sum['PredGrowthMean'] = np.mean(cv_pred_list, axis=0)
    df_sum['PredGrowthStd'] = np.std(cv_pred_list, axis=0)
    df_sum['PredGrowthMin'] = np.min(cv_pred_list, axis=0)
    df_sum['PredGrowthMax'] = np.max(cv_pred_list, axis=0)

    df_sum.to_csv('uno_pred.tsv', index=False, sep='\t', float_format='%.4f')

    scores = evaluate_prediction(df_sum['Growth'], df_sum['PredGrowthMean'])
    log_evaluation(scores, description='Testing on data from {} on partition {}, ({})'.format(args.data, args.partition, len(cv_y)))


if __name__ == '__main__':
    main()
