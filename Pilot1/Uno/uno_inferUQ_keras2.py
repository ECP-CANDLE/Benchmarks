#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import logging
import os

import numpy as np
import pandas as pd

from itertools import cycle

from keras import backend as K

import keras
from keras.utils import get_custom_objects

import uno as benchmark
import candle

from uno_data import CombinedDataLoader, CombinedDataGenerator, DataFeeder, read_IDs_file

from unoUQ_data import FromFileDataGenerator, find_columns_with_str

from uno_baseline_keras2 import build_feature_model, build_model, evaluate_prediction

from uno_trainUQ_keras2 import extension_from_parameters, log_evaluation

logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


additional_definitions_local = [
{'name':'uq_infer_file',
    'default':argparse.SUPPRESS,
    'action':'store',
    'help':'File to do inference'},
{'name':'uq_infer_given_drugs',
    'type': candle.str2bool,
    'default': False,
    'help':'Use given inference file to obtain drug ids to do inference'},
{'name':'uq_infer_given_cells',
    'type': candle.str2bool,
    'default': False,
    'help':'Use given inference file to obtain cell ids to do inference'},
{'name':'uq_infer_given_indices',
    'type': candle.str2bool,
    'default': False,
    'help':'Use given inference file to obtain indices to do inference'},
{'name':'model_file',
    'type':str,
    'default':'saved.model.h5',
    'help':'trained model file'},
{'name':'weights_file',
    'type':str,
    'default':'saved.weights.h5',
    'help':'trained weights file (loading model file alone sometimes does not work in keras)'},
{'name':'n_pred',
    'type':int,
    'default':1,
    'help':'the number of predictions to make for each sample-drug combination for uncertainty quantification'}
]

required_local = ( 'model_file', 'weights_file', 'uq_infer_file',
             'agg_dose', 'batch_size')


def initialize_parameters(default_model='uno_default_inferUQ_model.txt'):

    # Build benchmark object
    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, default_model, 'keras',
    prog='uno_inferUQ', desc='Read models to predict tumor response to single and paired drugs.')

    unoBmk.additional_definitions += additional_definitions_local
    unoBmk.required = unoBmk.required.union(required_local)

    # Finalize parameters
    gParameters = candle.finalize_parameters(unoBmk)
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters



def from_file(args, model):

    df_data = pd.read_csv(args.uq_infer_file, sep='\t')
    logger.info('data shape: {}'.format(df_data.shape))
    logger.info('Size of data to infer: {}'.format(df_data.shape))

    test_indices = range(df_data.shape[0])
    target_str = args.agg_dose or 'Growth'
    
    # Extract size of input layers to get number of features
    num_features_list = []
    feature_names_list = []
    for layer in model.layers: # All layers in model
        dict = layer.get_config() # getting layer config info
        name = dict['name'] # getting layer name
        if name.find('input') > -1: # if layer is an input layer
            feature_names_list.append(name.split('.')[-1])
            size_ = dict['batch_input_shape'] # get layer size
            num_features_list.append(size_[1])

    feature_names_list.append('dragon7')
        
    test_gen = FromFileDataGenerator(df_data, test_indices,
                target_str, feature_names_list, num_features_list,
                batch_size=args.batch_size, shuffle=False)

    return test_gen



def given_drugs(args, loader):

    test_gen = CombinedDataGenerator(loader, partition='test', batch_size=args.batch_size)
    
    # Include specified drugs
    include_drugs = read_IDs_file(args.uq_infer_file)
    df_response = test_gen.data.df_response
    if np.isin('Drug', df_response.columns.values):
        df = df_response[['Drug']]
        index = df.index[df['Drug'].isin(include_drugs)]
    else:
        df = df_response[['Drug1', 'Drug2']]
        index = df.index[df['Drug1'].isin(include_drugs) |
                        df['Drug2'].isin(include_drugs)]
    
    # Update object
    test_gen.index = index
    test_gen.index_cycle = cycle(index)
    test_gen.size = len(index)
    test_gen.steps = np.ceil(test_gen.size / args.batch_size)

    return test_gen



def given_cells(args, loader):

    test_gen = CombinedDataGenerator(loader, partition='test', batch_size=args.batch_size)
    
    # Include specified cells
    include_cells = read_IDs_file(args.uq_infer_file)
    df = test_gen.data.df_response[['Sample']]
    index = df.index[df['Sample'].isin(include_cells)]
    
    # Update object
    test_gen.index = index
    test_gen.index_cycle = cycle(index)
    test_gen.size = len(index)
    test_gen.steps = np.ceil(test_gen.size / args.batch_size)

    return test_gen



def given_indices(args, loader):

    test_gen = CombinedDataGenerator(loader, partition='test', batch_size=args.batch_size)
    
    # Include specified indices
    index = read_IDs_file(args.uq_infer_file)
    
    # Update object
    test_gen.index = index
    test_gen.index_cycle = cycle(index)
    test_gen.size = len(index)
    test_gen.steps = np.ceil(test_gen.size / args.batch_size)

    return test_gen



def run(params):
    args = candle.ArgumentStruct(**params)
    candle.set_seed(args.rng_seed)
    logfile_def = 'uno_infer_from_' + args.uq_infer_file + '.log'
    logfile = args.logfile if args.logfile else logfile_def
    candle.set_up_logger(logfile, logger, args.verbose)
    logger.info('Params: {}'.format(params))
    
    ext = extension_from_parameters(args)
    candle.verify_path(args.save_path)
    prefix = args.save_path + 'uno' + ext

    # Load trained model
    candle.register_permanent_dropout()
    model = keras.models.load_model(args.model_file, compile=False)
    model.load_weights(args.weights_file)
    logger.info('Loaded model:')
    model.summary(print_fn=logger.info)
    
    # Determine output to infer
    target = args.agg_dose or 'Growth'
    
    if (args.uq_infer_given_drugs or args.uq_infer_given_cells or args.uq_infer_given_indices):
        loader = CombinedDataLoader(args.rng_seed)
        loader.load(cache=args.cache,
                ncols=args.feature_subsample,
                agg_dose=args.agg_dose,
                cell_features=args.cell_features,
                drug_features=args.drug_features,
                drug_median_response_min=args.drug_median_response_min,
                drug_median_response_max=args.drug_median_response_max,
                use_landmark_genes=args.use_landmark_genes,
                use_filtered_genes=args.use_filtered_genes,
                cell_feature_subset_path=args.cell_feature_subset_path or args.feature_subset_path,
                drug_feature_subset_path=args.drug_feature_subset_path or args.feature_subset_path,
                preprocess_rnaseq=args.preprocess_rnaseq,
                single=args.single,
                train_sources=args.train_sources,
                test_sources=args.test_sources,
                embed_feature_source=not args.no_feature_source,
                encode_response_source=not args.no_response_source,
                )

        if args.uq_infer_given_drugs:
            test_gen = given_drugs(args, loader)
        elif args.uq_infer_given_cells:
            test_gen = given_cells(args, loader)
        else:
            test_gen = given_indices(args, loader)

    else:
        test_gen = from_file(args, model)


    df_test = test_gen.get_response(copy=True)
    y_test = df_test[target].values

    for i in range(args.n_pred):
    
        if args.no_gen:
            x_test_list, y_test = test_gen.get_slice(size=test_gen.size, single=args.single)
            y_test_pred = model.predict(x_test_list, batch_size=args.batch_size)
        else:
            test_gen.reset()
            y_test_pred = model.predict_generator(test_gen.flow(single=args.single), test_gen.steps)
            y_test_pred = y_test_pred[:test_gen.size]

        if args.loss == 'het':
            y_test_pred_ = y_test_pred[:,0]
            s_test_pred = y_test_pred[:,1]

            y_test_pred = y_test_pred_.flatten()

            df_test['Predicted_'+target+'_'+str(i+1)] = y_test_pred
            df_test['Pred_S_'+target+'_'+str(i+1)] = s_test_pred

            pred_fname = prefix + '.predicted_INFER_HET.tsv'

        elif args.loss == 'qtl':
        
            y_test_pred_50q = y_test_pred[:,0]
            y_test_pred_10q = y_test_pred[:,1]
            y_test_pred_90q = y_test_pred[:,2]

            y_test_pred = y_test_pred_50q.flatten() # 50th quantile prediction

            df_test['Predicted_50q_'+target+'_'+str(i+1)] = y_test_pred
            df_test['Predicted_10q_'+target+'_'+str(i+1)] = y_test_pred_10q.flatten()
            df_test['Predicted_90q_'+target+'_'+str(i+1)] = y_test_pred_90q.flatten()

            pred_fname = prefix + '.predicted_INFER_QTL.tsv'

        else:
            y_test_pred = y_test_pred.flatten()
            df_test['Predicted_'+target+'_'+str(i+1)] = y_test_pred
            pred_fname = prefix + '.predicted_INFER.tsv'

        if args.n_pred < 21:
            scores = evaluate_prediction(y_test, y_test_pred)
            log_evaluation(scores, logger)

    df_pred = df_test
    if args.agg_dose:
        if args.single:
            df_pred.sort_values(['Sample', 'Drug1', target], inplace=True)
        else:
            df_pred.sort_values(['Sample', 'Drug1', 'Drug2', target], inplace=True)
    else:
        if args.single:
            df_pred.sort_values(['Sample', 'Drug1', 'Dose1', 'Growth'], inplace=True)
        else:
            df_pred.sort_values(['Sample', 'Drug1', 'Drug2', 'Dose1', 'Dose2', 'Growth'], inplace=True)

    df_pred.to_csv(pred_fname, sep='\t', index=False, float_format='%.4g')
    logger.info('Predictions stored in file: {}'.format(pred_fname))


    if K.backend() == 'tensorflow':
        K.clear_session()

    logger.handlers = []


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()

