#! /usr/bin/env python

import argparse
import logging

import keras
import pandas as pd

from uno_data import DataFeeder
from uno_baseline_keras2 import evaluate_prediction, log_evaluation

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

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    test_gen = DataFeeder(filename=args.data, batch_size=1024)

    if 'json' == args.model_file.split('.')[-1]:
        with open(args.model_file, 'r') as f:
            model_json = f.read()
            model = keras.models.model_from_json(model_json)
            model.load_weights(args.weights_file, by_name=True)
    else:
        model = keras.models.load_model(args.model_file, compile=False)
        model.load_weights(args.weights_file)

    model.summary()

    y_test_pred = model.predict_generator(test_gen, test_gen.steps)
    y_test_pred = y_test_pred.flatten()

    df_y = test_gen.get_response(copy=True)
    y_test = df_y['Growth'].values
    scores = evaluate_prediction(y_test, y_test_pred)
    log_evaluation(scores, description='Testing on data from {} ({})'.format(args.data, len(y_test_pred)))

    # save to tsv
    df_pred = df_y.assign(PredictedGrowth=y_test_pred, GrowthError=y_test_pred-y_test)
    df_pred.sort_values(['Sample', 'Drug1', 'Drug2', 'Dose1', 'Dose2', 'Growth'], inplace=True)
    df_pred.to_csv('prediction.tsv', sep='\t', index=False, float_format='%.4g')


if __name__ == '__main__':
    main()

