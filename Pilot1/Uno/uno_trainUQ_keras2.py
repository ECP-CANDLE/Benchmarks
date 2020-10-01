#! /usr/bin/env python

from __future__ import division, print_function

import logging
import os
import random
import argparse

import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from scipy.stats.stats import pearsonr

import uno as benchmark
import candle

from uno_data import CombinedDataLoader, CombinedDataGenerator, DataFeeder, read_IDs_file
from uno_data import logger as unologger

from uno_baseline_keras2 import build_feature_model, build_model, evaluate_prediction

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


additional_definitions = [
    {'name': 'uq_exclude_drugs_file',
     'default': argparse.SUPPRESS,
     'action': 'store',
     'help': 'File with drug ids to exclude from training'},
    {'name': 'uq_exclude_cells_file',
     'default': argparse.SUPPRESS,
     'action': 'store',
     'help': 'File with cell ids to exclude from training'},
    {'name': 'uq_exclude_indices_file',
     'default': argparse.SUPPRESS,
     'action': 'store',
     'help': 'File with indices to exclude from training'},
    {'name': 'exclude_drugs', 'nargs': '+',
     'default': [],
     'help':'drug ids to exclude'},
    {'name': 'exclude_cells', 'nargs': '+',
     'default': [],
     'help':'cell ids to exclude'},
    {'name': 'exclude_indices', 'nargs': '+',
     'default': [],
     'help':'indices to exclude'},
    {'name': 'reg_l2',
     'type': float,
     'default': 0.,
     'help': 'weight of regularization for l2 norm of nn weights'}
]

required = ['exclude_drugs', 'exclude_cells', 'exclude_indices']


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    ext += '.E={}'.format(args.epochs)
    ext += '.O={}'.format(args.optimizer)
    ext += '.LOSS={}'.format(args.loss)
    ext += '.LR={}'.format(args.learning_rate)
    ext += '.CF={}'.format(''.join([x[0] for x in sorted(args.cell_features)]))
    ext += '.DF={}'.format(''.join([x[0] for x in sorted(args.drug_features)]))
    if args.feature_subsample > 0:
        ext += '.FS={}'.format(args.feature_subsample)
    if args.dropout > 0:
        ext += '.DR={}'.format(args.dropout)
    if args.warmup_lr:
        ext += '.wu_lr'
    if args.reduce_lr:
        ext += '.re_lr'
    if args.residual:
        ext += '.res'
    if args.use_landmark_genes:
        ext += '.L1000'
    if args.no_gen:
        ext += '.ng'
    for i, n in enumerate(args.dense):
        if n > 0:
            ext += '.D{}={}'.format(i + 1, n)
    if args.dense_feature_layers != args.dense:
        for i, n in enumerate(args.dense):
            if n > 0:
                ext += '.FD{}={}'.format(i + 1, n)

    return ext


def log_evaluation(metric_outputs, logger, description='Comparing y_true and y_pred:'):
    logger.info(description)
    for metric, value in metric_outputs.items():
        logger.info('  {}: {:.4f}'.format(metric, value))


def initialize_parameters(default_model='uno_defaultUQ_model.txt'):

    # Build benchmark object
    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, default_model, 'keras',
                                    prog='uno_trainUQ', desc='Build and train neural network based models to predict tumor response to single and paired drugs with UQ.')

    # update locals
    unoBmk.required.update(required)
    unoBmk.additional_definitions += additional_definitions
    # Finalize parameters
    gParameters = candle.finalize_parameters(unoBmk)
    # benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters


def run(params):
    args = candle.ArgumentStruct(**params)
    candle.set_seed(args.rng_seed)
    ext = extension_from_parameters(args)
    candle.verify_path(args.save_path)
    prefix = args.save_path + 'uno' + ext
    logfile = args.logfile if args.logfile else prefix + '.log'
    candle.set_up_logger(logfile, logger, args.verbose)
    logger.info('Params: {}'.format(params))

    # Exclude drugs / cells for UQ
    if 'uq_exclude_drugs_file' in params.keys():
        args.exclude_drugs = read_IDs_file(args.uq_exclude_drugs_file)
        logger.info('Drugs to exclude: {}'.format(args.exclude_drugs))
    else:
        args.exclude_drugs = []
    if 'uq_exclude_cells_file' in params.keys():
        args.exclude_cells = read_IDs_file(args.uq_exclude_cells_file)
        logger.info('Cells to exclude: {}'.format(args.exclude_cells))
    else:
        args.exclude_cells = []

    if 'uq_exclude_indices_file' in params.keys():
        exclude_indices_ = read_IDs_file(args.uq_exclude_indices_file)
        args.exclude_indices = [int(x) for x in exclude_indices_]
        logger.info('Indices to exclude: {}'.format(args.exclude_indices))
    else:
        args.exclude_indices = []

    if (len(args.gpus) > 0):
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = ",".join(map(str, args.gpus))
        K.set_session(tf.Session(config=config))

    loader = CombinedDataLoader(seed=args.rng_seed)
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
                use_exported_data=args.use_exported_data,
                )

    target = args.agg_dose or 'Growth'
    nout = 1
    val_split = args.val_split
    train_split = 1 - val_split

    if args.export_csv:
        fname = args.export_csv
        loader.partition_data(cv_folds=args.cv,
                              train_split=train_split,
                              val_split=val_split,
                              cell_types=args.cell_types,
                              by_cell=args.by_cell,
                              by_drug=args.by_drug,
                              cell_subset_path=args.cell_subset_path,
                              drug_subset_path=args.drug_subset_path,
                              exclude_cells=args.exclude_cells,
                              exclude_drugs=args.exclude_drugs,
                              exclude_indices=args.exclude_indices)
        train_gen = CombinedDataGenerator(loader, batch_size=args.batch_size, shuffle=args.shuffle)
        val_gen = CombinedDataGenerator(loader, partition='val', batch_size=args.batch_size, shuffle=args.shuffle)

        x_train_list, y_train = train_gen.get_slice(size=train_gen.size, dataframe=True, single=args.single)
        x_val_list, y_val = val_gen.get_slice(size=val_gen.size, dataframe=True, single=args.single)
        df_train = pd.concat([y_train] + x_train_list, axis=1)
        df_val = pd.concat([y_val] + x_val_list, axis=1)
        df = pd.concat([df_train, df_val]).reset_index(drop=True)
        if args.growth_bins > 1:
            df = uno_data.discretize(df, 'Growth', bins=args.growth_bins)
        df.to_csv(fname, sep='\t', index=False, float_format="%.3g")
        return

    if args.export_data:
        fname = args.export_data
        loader.partition_data(cv_folds=args.cv,
                              train_split=train_split,
                              val_split=val_split,
                              cell_types=args.cell_types,
                              by_cell=args.by_cell,
                              by_drug=args.by_drug,
                              cell_subset_path=args.cell_subset_path,
                              drug_subset_path=args.drug_subset_path,
                              exclude_cells=args.exclude_cells,
                              exclude_drugs=args.exclude_drugs,
                              exclude_indices=args.exclude_indices)
        train_gen = CombinedDataGenerator(loader, batch_size=args.batch_size, shuffle=args.shuffle)
        val_gen = CombinedDataGenerator(loader, partition='val', batch_size=args.batch_size, shuffle=args.shuffle)
        store = pd.HDFStore(fname, complevel=9, complib='blosc:snappy')

        config_min_itemsize = {'Sample': 30, 'Drug1': 10}
        if not args.single:
            config_min_itemsize['Drug2'] = 10

        for partition in ['train', 'val']:
            gen = train_gen if partition == 'train' else val_gen
            for i in range(gen.steps):
                x_list, y = gen.get_slice(size=args.batch_size, dataframe=True, single=args.single)

                for j, input_feature in enumerate(x_list):
                    input_feature.columns = [''] * len(input_feature.columns)
                    store.append('x_{}_{}'.format(partition, j), input_feature.astype('float32'), format='table', data_column=True)
                store.append('y_{}'.format(partition), y.astype({target: 'float32'}), format='table', data_column=True,
                             min_itemsize=config_min_itemsize)
                logger.info('Generating {} dataset. {} / {}'.format(partition, i, gen.steps))

        # save input_features and feature_shapes from loader
        store.put('model', pd.DataFrame())
        store.get_storer('model').attrs.input_features = loader.input_features
        store.get_storer('model').attrs.feature_shapes = loader.feature_shapes

        store.close()
        logger.info('Completed generating {}'.format(fname))
        return

    if args.use_exported_data is None:
        loader.partition_data(partition_by=args.partition_by, cv_folds=args.cv,
                              train_split=train_split,
                              val_split=val_split,
                              cell_types=args.cell_types,
                              by_cell=args.by_cell,
                              by_drug=args.by_drug,
                              cell_subset_path=args.cell_subset_path,
                              drug_subset_path=args.drug_subset_path,
                              exclude_cells=args.exclude_cells,
                              exclude_drugs=args.exclude_drugs,
                              exclude_indices=args.exclude_indices)

    model = build_model(loader, args)
    logger.info('Combined model:')
    model.summary(print_fn=logger.info)
    # plot_model(model, to_file=prefix+'.model.png', show_shapes=True)
    if args.loss == 'het' or args.loss == 'qtl':
        model = candle.add_model_output(model, mode=args.loss)
        logger.info('After adjusting for UQ loss function')
        model.summary(print_fn=logger.info)

    if args.cp:
        model_json = model.to_json()
        with open(prefix + '.model.json', 'w') as f:
            print(model_json, file=f)

    def warmup_scheduler(epoch):
        lr = args.learning_rate or base_lr * args.batch_size / 100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5 - epoch) + lr * epoch) / 5)
        logger.debug('Epoch {}: lr={:.5g}'.format(epoch, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    df_pred_list = []

    cv_ext = ''
    cv = args.cv if args.cv > 1 else 1

    for fold in range(cv):
        if args.cv > 1:
            logger.info('Cross validation fold {}/{}:'.format(fold + 1, cv))
            cv_ext = '.cv{}'.format(fold + 1)

        template_model = build_model(loader, args, silent=True)
        if args.loss == 'het' or args.loss == 'qtl':
            template_model = candle.add_model_output(template_model, mode=args.loss)

        if args.initial_weights:
            logger.info("Loading initial weights from {}".format(args.initial_weights))
            template_model.load_weights(args.initial_weights)

        if len(args.gpus) > 1:
            from keras.utils import multi_gpu_model
            gpu_count = len(args.gpus)
            logger.info("Multi GPU with {} gpus".format(gpu_count))
            model = multi_gpu_model(template_model, cpu_merge=False, gpus=gpu_count)
        else:
            model = template_model

        optimizer = optimizers.deserialize({'class_name': args.optimizer, 'config': {}})
        base_lr = args.base_lr or K.get_value(optimizer.lr)
        if args.learning_rate:
            K.set_value(optimizer.lr, args.learning_rate)

        if args.loss == 'het':
            logger.info('Training heteroscedastic model:')
            mae_heteroscedastic = candle.mae_heteroscedastic_metric(nout)
            r2_heteroscedastic = candle.r2_heteroscedastic_metric(nout)
            meanS_heteroscedastic = candle.meanS_heteroscedastic_metric(nout)
            model.compile(loss=candle.heteroscedastic_loss(nout), optimizer=optimizer, metrics=[mae_heteroscedastic, r2_heteroscedastic, meanS_heteroscedastic])
        elif args.loss == 'qtl':
            logger.info('Training quantile model:')
            quantile50 = candle.quantile_metric(nout, 0, 0.5)
            quantile10 = candle.quantile_metric(nout, 1, 0.1)
            quantile90 = candle.quantile_metric(nout, 2, 0.9)
            model.compile(loss=candle.triple_quantile_loss(nout, 0.1, 0.9), optimizer=optimizer, metrics=[quantile50, quantile10, quantile90])
        else:
            logger.info('Training homoscedastic model:')
            model.compile(loss=args.loss, optimizer=optimizer, metrics=[candle.mae, candle.r2])

        # calculate trainable and non-trainable params
        params.update(candle.compute_trainable_params(model))

        candle_monitor = candle.CandleRemoteMonitor(params=params)
        timeout_monitor = candle.TerminateOnTimeOut(params['timeout'])
        es_monitor = keras.callbacks.EarlyStopping(patience=10, verbose=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        warmup_lr = LearningRateScheduler(warmup_scheduler)
        checkpointer = candle.MultiGPUCheckpoint(prefix + cv_ext + '.model.h5', save_best_only=True)
        tensorboard = TensorBoard(log_dir="tb/{}{}{}".format(args.tb_prefix, ext, cv_ext))
        history_logger = candle.LoggingCallback(logger.debug)

        callbacks = [candle_monitor, timeout_monitor, history_logger]
        if args.es:
            callbacks.append(es_monitor)
        if args.reduce_lr:
            callbacks.append(reduce_lr)
        if args.warmup_lr:
            callbacks.append(warmup_lr)
        if args.cp:
            callbacks.append(checkpointer)
        if args.tb:
            callbacks.append(tensorboard)
        if args.save_weights:
            logger.info("Will save weights to: " + args.save_weights)
            callbacks.append(candle.MultiGPUCheckpoint(args.save_weights))

        if args.use_exported_data is not None:
            train_gen = DataFeeder(filename=args.use_exported_data, batch_size=args.batch_size, shuffle=args.shuffle, single=args.single, agg_dose=args.agg_dose)
            val_gen = DataFeeder(partition='val', filename=args.use_exported_data, batch_size=args.batch_size, shuffle=args.shuffle, single=args.single, agg_dose=args.agg_dose)
            test_gen = DataFeeder(partition='test', filename=args.use_exported_data, batch_size=args.batch_size, shuffle=args.shuffle, single=args.single, agg_dose=args.agg_dose)
        else:
            train_gen = CombinedDataGenerator(loader, fold=fold, batch_size=args.batch_size, shuffle=args.shuffle, single=args.single)
            val_gen = CombinedDataGenerator(loader, partition='val', fold=fold, batch_size=args.batch_size, shuffle=args.shuffle, single=args.single)
            test_gen = CombinedDataGenerator(loader, partition='test', fold=fold, batch_size=args.batch_size, shuffle=args.shuffle, single=args.single)

        df_val = val_gen.get_response(copy=True)
        y_val = df_val[target].values
        y_shuf = np.random.permutation(y_val)
        log_evaluation(evaluate_prediction(y_val, y_shuf), logger, description='Between random pairs in y_val:')

        if args.no_gen:
            x_train_list, y_train = train_gen.get_slice(size=train_gen.size, single=args.single)
            x_val_list, y_val = val_gen.get_slice(size=val_gen.size, single=args.single)
            history = model.fit(x_train_list, y_train,
                                batch_size=args.batch_size,
                                epochs=args.epochs,
                                callbacks=callbacks,
                                validation_data=(x_val_list, y_val))
        else:
            logger.info('Data points per epoch: train = %d, val = %d, test = %d', train_gen.size, val_gen.size, test_gen.size)
            logger.info('Steps per epoch: train = %d, val = %d, test = %d', train_gen.steps, val_gen.steps, test_gen.steps)
            history = model.fit_generator(train_gen, train_gen.steps,
                                          epochs=args.epochs,
                                          callbacks=callbacks,
                                          validation_data=val_gen,
                                          validation_steps=val_gen.steps)

        # prediction on holdout(test) when exists or use validation set
        if test_gen.size > 0:
            df_val = test_gen.get_response(copy=True)
            y_val = df_val[target].values
            y_val_pred = model.predict_generator(test_gen, test_gen.steps + 1)
            y_val_pred = y_val_pred[:test_gen.size]
            if args.loss == 'het':
                y_val_pred_ = y_val_pred[:, 0]
                y_val_pred = y_val_pred_.flatten()
            elif args.loss == 'qtl':
                y_val_pred_50q = y_val_pred[:, 0]
                y_val_pred = y_val_pred_50q.flatten()   # 50th quantile prediction
        else:
            if args.no_gen:
                y_val_pred = model.predict(x_val_list, batch_size=args.batch_size)
            else:
                val_gen.reset()
                y_val_pred = model.predict_generator(val_gen, val_gen.steps + 1)
                y_val_pred = y_val_pred[:val_gen.size]

            if args.loss == 'het':
                y_val_pred_ = y_val_pred[:, 0]
                s_val_pred = y_val_pred[:, 1]

                y_val_pred = y_val_pred_.flatten()

                df_val['Predicted_' + target] = y_val_pred
                df_val[target + '_Error'] = y_val_pred - y_val
                df_val['Pred_S_' + target] = s_val_pred

            elif args.loss == 'qtl':
                y_val_pred_50q = y_val_pred[:, 0]
                y_val_pred_10q = y_val_pred[:, 1]
                y_val_pred_90q = y_val_pred[:, 2]

                y_val_pred = y_val_pred_50q.flatten()   # 50th quantile prediction

                df_val['Predicted_50q_' + target] = y_val_pred
                df_val[target + '_Error_50q'] = y_val_pred - y_val
                df_val['Predicted_10q_' + target] = y_val_pred_10q.flatten()
                df_val['Predicted_90q_' + target] = y_val_pred_90q.flatten()

            else:
                y_val_pred = y_val_pred.flatten()
                # df_val = df_val.assign(PredictedGrowth=y_val_pred, GrowthError=y_val_pred - y_val)
                df_val['Predicted' + target] = y_val_pred
                df_val[target + 'Error'] = y_val_pred - y_val

        scores = evaluate_prediction(y_val, y_val_pred)
        log_evaluation(scores, logger)

        df_pred_list.append(df_val)

        if 'loss' in history.history.keys():
            candle.plot_history(prefix, history, 'loss')
        if args.loss == 'het':
            if 'r2_heteroscedastic' in history.history.keys():
                candle.plot_history(prefix, history, 'r2_heteroscedastic')
            if 'mae_heteroscedastic' in history.history.keys():
                candle.plot_history(prefix, history, 'mae_heteroscedastic')
            if 'meanS_heteroscedastic' in history.history.keys():
                candle.plot_history(prefix, history, 'meanS_heteroscedastic')
        elif args.loss == 'qtl':
            if 'quantile_0.5' in history.history.keys():
                candle.plot_history(prefix, history, 'quantile_0.5')
            if 'quantile_0.1' in history.history.keys():
                candle.plot_history(prefix, history, 'quantile_0.1')
            if 'quantile_0.9' in history.history.keys():
                candle.plot_history(prefix, history, 'quantile_0.9')
        else:
            if 'r2' in history.history.keys():
                candle.plot_history(prefix, history, 'r2')

    pred_fname = prefix + '.predicted.tsv'
    df_pred = pd.concat(df_pred_list)
    if args.agg_dose:
        if args.single:
            df_pred.sort_values(['Sample', 'Drug1', target], inplace=True)
        else:
            df_pred.sort_values(['Source', 'Sample', 'Drug1', 'Drug2', target], inplace=True)
    else:
        if args.single:
            df_pred.sort_values(['Sample', 'Drug1', 'Dose1', 'Growth'], inplace=True)
        else:
            df_pred.sort_values(['Sample', 'Drug1', 'Drug2', 'Dose1', 'Dose2', 'Growth'], inplace=True)
    df_pred.to_csv(pred_fname, sep='\t', index=False, float_format='%.4g')
    logger.info('Testing predictions stored in file: {}'.format(pred_fname))

    if args.cp:
        logger.info('Model stored in file: {}'.format(prefix + '.model.h5'))
        # logger.info('Model weights stored in file: {}'.format(prefix+cv_ext+'.weights.h5'))
        logger.info('Model weights stored in file: {}'.format(args.save_path + '/' + args.save_weights))

    if args.cv > 1:
        scores = evaluate_prediction(df_pred[target], df_pred['Predicted' + target])
        log_evaluation(scores, description='Combining cross validation folds:')

    for test_source in loader.test_sep_sources:
        test_gen = CombinedDataGenerator(loader, partition='test', batch_size=args.batch_size, source=test_source)
        df_test = test_gen.get_response(copy=True)
        y_test = df_test[target].values
        n_test = len(y_test)
        if n_test == 0:
            continue
        if args.no_gen:
            x_test_list, y_test = test_gen.get_slice(size=test_gen.size, single=args.single)
            y_test_pred = model.predict(x_test_list, batch_size=args.batch_size)
            if args.loss == 'het':
                y_test_pred = y_test_pred[:, 0]  # mean
            elif args.loss == 'qtl':
                y_test_pred = y_test_pred[:, 0]  # 50th quantile prediction
        else:
            y_test_pred = model.predict_generator(test_gen.flow(single=args.single), test_gen.steps)
            if args.loss == 'het':
                y_test_pred = y_test_pred[:test_gen.size, 0]  # mean
            elif args.loss == 'qtl':
                y_test_pred = y_test_pred[:test_gen.size, 0]  # 50th quantile prediction
            else:
                y_test_pred = y_test_pred[:test_gen.size]
        y_test_pred = y_test_pred.flatten()
        scores = evaluate_prediction(y_test, y_test_pred)
        log_evaluation(scores, description='Testing on data from {} ({})'.format(test_source, n_test))

    if K.backend() == 'tensorflow':
        K.clear_session()

    logger.handlers = []

    return history


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
