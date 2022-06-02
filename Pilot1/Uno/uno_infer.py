#! /usr/bin/env python

import argparse

import pandas as pd
import numpy as np
from tensorflow import keras

from uno_data import DataFeeder
from uno_baseline_keras2 import evaluate_prediction
import candle


def log_evaluation(metric_outputs, description='Comparing y_true and y_pred:'):
    print(description)
    for metric, value in metric_outputs.items():
        print('  {}: {:.4f}'.format(metric, value))


def get_parser():
    parser = argparse.ArgumentParser(description='Uno infer script')
    parser.add_argument("--data",
                        required=True,
                        help="data file to infer on. expect exported file from uno_baseline_keras2.py")
    parser.add_argument("--model_file", required=True, help="json model description file")
    parser.add_argument("--weights_file", help="model weights file")
    parser.add_argument("--partition", default='all',
                        choices=['train', 'val', 'all'], help="partition of test dataset")
    parser.add_argument("-n", "--n_pred", type=int, default=1, help="the number of predictions to make")

    parser.add_argument("--single", default=False, help="do not use drug pair representation")
    parser.add_argument("--agg_dose", default=None,
                        choices=['AUC', 'IC50', 'HS', 'AAC1', 'AUC1', 'DSS1'],
                        help="use dose-independent response data with the specified aggregation metric")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    candle.register_permanent_dropout()
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
    cv_stats = {'mae': [], 'mse': [], 'r2': [], 'corr': []}
    target = args.agg_dose or 'Growth'

    for cv in range(args.n_pred):
        cv_pred = []
        dataset = ['train', 'val'] if args.partition == 'all' else [args.partition]
        for partition in dataset:
            test_gen = DataFeeder(filename=args.data, partition=partition, batch_size=1024, single=args.single, agg_dose=args.agg_dose)
            y_test_pred = model.predict(test_gen, steps=test_gen.steps + 1)
            y_test_pred = y_test_pred[:test_gen.size]
            y_test_pred = y_test_pred.flatten()

            df_y = test_gen.get_response(copy=True)
            y_test = df_y[target].values

            df_pred = df_y.assign(**{f'Predicted{target}': y_test_pred, f'{target}Error': y_test_pred - y_test})
            df_pred_list.append(df_pred)
            test_gen.close()

            if cv == 0:
                cv_y_list.append(df_y)
            cv_pred.append(y_test_pred)
        cv_pred_list.append(np.concatenate(cv_pred))

        # calcuate stats for mse, mae, r2, corr
        scores = evaluate_prediction(df_pred[target], df_pred[f'Predicted{target}'])
        # log_evaluation(scores, description=cv)
        [cv_stats[key].append(scores[key]) for key in scores.keys()]

    df_pred = pd.concat(df_pred_list)
    cv_y = pd.concat(cv_y_list)

    # save to tsv
    headers = ['Sample', 'Drug1']
    if not args.single:
        headers.append('Drug2')
    if not args.agg_dose:
        headers.append('Dose1')
    if not args.single and not args.agg_dose:
        headers.append('Dose2')
    headers.append(target)

    df_pred.sort_values(headers, inplace=True)
    df_pred.to_csv('uno_pred.all.tsv', sep='\t', index=False, float_format='%.6g')

    df_sum = cv_y.assign()
    df_sum[f'Pred{target}Mean'] = np.mean(cv_pred_list, axis=0)
    df_sum[f'Pred{target}Std'] = np.std(cv_pred_list, axis=0)
    df_sum[f'Pred{target}Min'] = np.min(cv_pred_list, axis=0)
    df_sum[f'Pred{target}Max'] = np.max(cv_pred_list, axis=0)

    df_sum.to_csv('uno_pred.tsv', index=False, sep='\t', float_format='%.6g')

    scores = evaluate_prediction(df_pred[f'{target}'], df_pred[f'Predicted{target}'])
    log_evaluation(scores, description='Testing on data from {} on partition {} ({} rows)'.format(args.data, args.partition, len(cv_y)))

    print('     mean    std    min    max')
    for key in ['mse', 'mae', 'r2', 'corr']:
        print('{}: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(key,
                                                          np.around(np.mean(cv_stats[key], axis=0), decimals=4),
                                                          np.around(np.std(cv_stats[key], axis=0), decimals=4),
                                                          np.around(np.min(cv_stats[key], axis=0), decimals=4),
                                                          np.around(np.max(cv_stats[key], axis=0), decimals=4)
                                                          ))


if __name__ == '__main__':
    main()
