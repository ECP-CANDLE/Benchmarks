#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import logging
import os

import numpy as np
import pandas as pd


from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from keras.utils.vis_utils import plot_model

import data_utils_.uno as uno
import candle

import data_utils_.uno_combined_data_loader as uno_combined_data_loader
import data_utils_.uno_combined_data_generator as uno_combined_data_generator
import model_utils_.uno_model_utils as uno_model_utils

from model_utils_.uno_model_utils import heteroscedastic_loss, triple_quantile_loss

logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

additional_definitions = [
{'name':'uq_exclude_drugs_file',
    'default':argparse.SUPPRESS,
    'action':'store',
    'help':'File with drug ids to exclude from training'},
{'name':'uq_exclude_cells_file',
    'default':argparse.SUPPRESS,
    'action':'store',
    'help':'File with cell ids to exclude from training'},
{'name':'uq_exclude_indices_file',
    'default':argparse.SUPPRESS,
    'action':'store',
    'help':'File with indices to exclude from training'},
{'name':'exclude_indices', 'nargs':'+',
    'default': [],
    'help':'indices to exclude'},
{'name':'reg_l2',
    'type': float,
    'default': 0.,
    'help':'weight of regularization for l2 norm of nn weights'}
]

required = ['exclude_drugs', 'exclude_cells', 'exclude_indices']

class UQUno(candle.Benchmark):
    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(uno.required)
            self.required.update(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions + uno.additional_definitions



def initialize_parameters():

    # Build benchmark object
    unoUQBmk = UQUno(uno.file_path, 'uno_defaultUQ_model.txt', 'keras',
    prog='uno_trainUQ', desc='Build neural network based models to predict tumor response to single and paired drugs, including UQ analysis.')

    # Initialize parameters
    gParameters = candle.initialize_parameters(unoUQBmk)
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters


def run(params):
    args = candle.ArgumentStruct(**params)
    candle.set_seed(args.rng_seed)
    ext = uno.extension_from_parameters(args)
    candle.verify_path(args.save_path)
    prefix = args.save_path + 'uno' + ext
    logfile = args.logfile if args.logfile else prefix+'.log'
    uno.set_up_logger(logfile, logger, uno.loggerUno, args.verbose)
    logger.info('Params: {}'.format(params))

    # Exclude drugs / cells for UQ
    if 'uq_exclude_drugs_file' in params.keys():
        args.exclude_drugs = uno.read_IDs_file(args.uq_exclude_drugs_file)
        logger.info('Drugs to exclude: {}'.format(args.exclude_drugs))
    else:
        args.exclude_drugs = []
    if 'uq_exclude_cells_file' in params.keys():
        args.exclude_cells = uno.read_IDs_file(args.uq_exclude_cells_file)
        logger.info('Cells to exclude: {}'.format(args.exclude_cells))
    else:
        args.exclude_cells = []

    if 'uq_exclude_indices_file' in params.keys():
        exclude_indices_ = uno.read_IDs_file(args.uq_exclude_indices_file)
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

    loader = uno_combined_data_loader.CombinedDataLoader(seed=args.rng_seed)
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

    target = args.agg_dose or 'Growth'
    val_split = args.val_split
    train_split = 1 - val_split

    loader.partition_data(partition_by=args.partition_by,
                          cv_folds=args.cv, train_split=train_split, val_split=val_split,
                          cell_types=args.cell_types, by_cell=args.by_cell, by_drug=args.by_drug,
                          cell_subset_path=args.cell_subset_path,
                          drug_subset_path=args.drug_subset_path,
                          exclude_cells=args.exclude_cells,
                          exclude_drugs=args.exclude_drugs,
                          exclude_indices=args.exclude_indices
                          )

    model = uno_model_utils.build_model(loader, args, logger)
    logger.info('Combined model:')
    model.summary(print_fn=logger.info)
    # plot_model(model, to_file=prefix+'.model.png', show_shapes=True)

    if args.cp:
        model_json = model.to_json()
        with open(prefix+'.model.json', 'w') as f:
            print(model_json, file=f)

    def warmup_scheduler(epoch):
        lr = args.learning_rate or base_lr * args.batch_size/100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5-epoch) + lr * epoch) / 5)
        logger.debug('Epoch {}: lr={:.5g}'.format(epoch, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    df_pred_list = []

    cv_ext = ''
    cv = args.cv if args.cv > 1 else 1

    for fold in range(cv):
        if args.cv > 1:
            logger.info('Cross validation fold {}/{}:'.format(fold+1, cv))
            cv_ext = '.cv{}'.format(fold+1)

#        model = uno_model_utils.build_model(loader, args, logger, silent=True)

        template_model = uno_model_utils.build_model(loader, args, logger, silent=True)
        if args.initial_weights:
            logger.info("Loading weights from {}".format(args.initial_weights))
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

        if args.loss == 'heteroscedastic':
            logger.info('Training heteroscedastic model:')
            model.compile(loss=heteroscedastic_loss, optimizer=optimizer, metrics=[uno_model_utils.mae_heteroscedastic, uno_model_utils.r2_heteroscedastic, uno_model_utils.meanS_heteroscesdastic])
        elif args.loss == 'quantile':
            logger.info('Training quantile model:')
            model.compile(loss=triple_quantile_loss, optimizer=optimizer, metrics=[uno_model_utils.quantile50, uno_model_utils.quantile10, uno_model_utils.quantile90])
        else:
            logger.info('Training homoscedastic model:')
            model.compile(loss=args.loss, optimizer=optimizer, metrics=[candle.mae, candle.r2])

        # calculate trainable and non-trainable params
        params.update(candle.compute_trainable_params(model))

        candle_monitor = candle.CandleRemoteMonitor(params=params)
        timeout_monitor = candle.TerminateOnTimeOut(params['timeout'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        warmup_lr = LearningRateScheduler(warmup_scheduler)
        #checkpointer = ModelCheckpoint(prefix+cv_ext+'.weights.h5', save_best_only=True, save_weights_only=True)
        checkpointer = candle.MultiGPUCheckpoint(prefix + cv_ext + '.model.h5', save_best_only=True)
        tensorboard = TensorBoard(log_dir="tb/{}{}{}".format(args.tb_prefix, ext, cv_ext))
        history_logger = candle.LoggingCallback(logger.debug)
#        model_recorder = uno_model_utils.ModelRecorder()

        # callbacks = [history_logger, model_recorder]
        callbacks = [candle_monitor, timeout_monitor, history_logger]#, model_recorder]
        if args.reduce_lr:
            callbacks.append(reduce_lr)
        if args.warmup_lr:
            callbacks.append(warmup_lr)
        if args.cp:
            callbacks.append(checkpointer)
        if args.tb:
            callbacks.append(tensorboard)
        if args.save_weights:
            callbacks.append(uno_model_utils.SimpleWeightSaver(args.save_path + '/' + args.save_weights))


        train_gen = uno_combined_data_generator.CombinedDataGenerator(loader, fold=fold, batch_size=args.batch_size, shuffle=args.shuffle)
        val_gen = uno_combined_data_generator.CombinedDataGenerator(loader, partition='val', fold=fold, batch_size=args.batch_size, shuffle=args.shuffle)

        df_val = val_gen.get_response(copy=True)
        y_val = df_val[target].values
        y_shuf = np.random.permutation(y_val)
        uno.log_evaluation(uno.evaluate_prediction(y_val, y_shuf), logger,
                       description='Between random pairs in y_val:')

        if args.no_gen:
            x_train_list, y_train = train_gen.get_slice(size=train_gen.size, single=args.single)
            x_val_list, y_val = val_gen.get_slice(size=val_gen.size, single=args.single)
            history = model.fit(x_train_list, y_train,
                                batch_size=args.batch_size,
                                epochs=args.epochs,
                                callbacks=callbacks,
                                validation_data=(x_val_list, y_val))
        else:
            logger.info('Data points per epoch: train = %d, val = %d',train_gen.size, val_gen.size)
            logger.info('Steps per epoch: train = %d, val = %d',train_gen.steps, val_gen.steps)
            history = model.fit_generator(train_gen, train_gen.steps,
                                          epochs=args.epochs,
                                          callbacks=callbacks,
                                          validation_data=val_gen,
                                          validation_steps=val_gen.steps)

#        if args.cp:
#            model.load_weights(prefix+cv_ext+'.weights.h5')
        # model = model_recorder.best_model

        if args.no_gen:
            y_val_pred = model.predict(x_val_list, batch_size=args.batch_size)
        else:
            val_gen.reset()
            y_val_pred = model.predict_generator(val_gen, val_gen.steps + 1)
            y_val_pred = y_val_pred[:val_gen.size]

        if args.loss == 'heteroscedastic':
            y_val_pred_ = y_val_pred[:,0]
            s_val_pred = y_val_pred[:,1]

            y_val_pred = y_val_pred_.flatten()

            df_val['Predicted_'+target] = y_val_pred
            df_val[target+'_Error'] = y_val_pred-y_val
            df_val['Pred_S_'+target] = s_val_pred

        elif args.loss == 'quantile':
            y_val_pred_50q = y_val_pred[:,0]
            y_val_pred_10q = y_val_pred[:,1]
            y_val_pred_90q = y_val_pred[:,2]

            y_val_pred = y_val_pred_50q.flatten()   # 50th quantile prediction

            df_val['Predicted_50q_'+target] = y_val_pred
            df_val[target+'_Error_50q'] = y_val_pred-y_val
            df_val['Predicted_10q_'+target] = y_val_pred_10q.flatten()
            df_val['Predicted_90q_'+target] = y_val_pred_90q.flatten()

        else:
            y_val_pred = y_val_pred.flatten()

            # df_val = df_val.assign(PredictedGrowth=y_val_pred, GrowthError=y_val_pred-y_val)
            df_val['Predicted'+target] = y_val_pred
            df_val[target+'Error'] = y_val_pred-y_val

        scores = uno.evaluate_prediction(y_val, y_val_pred)
        uno.log_evaluation(scores, logger)

        df_pred_list.append(df_val)

#        if args.cp:
#            model_recorder.best_model.save(prefix+'.model.h5')

        if hasattr(history, 'loss'):
            candle.plot_history(prefix, history, 'loss')
        if args.loss == 'heteroscedastic':
            if hasattr(history, 'r2_heteroscedastic'):
                candle.plot_history(prefix, history, 'r2_heteroscedastic')
            if hasattr(history, 'meanS_heteroscedastic'):
                candle.plot_history(prefix, history, 'meanS_heteroscesdastic')
        elif args.loss == 'quantile':
            if hasattr(history, 'quantile50'):
                candle.plot_history(prefix, history, 'quantile50')
            if hasattr(history, 'quantile10'):
                candle.plot_history(prefix, history, 'quantile10')
            if hasattr(history, 'quantile90'):
                candle.plot_history(prefix, history, 'quantile90')
        else:
            if hasattr(history, 'r2'):
                candle.plot_history(prefix, history, 'r2')

    pred_fname = prefix + '.predicted.tsv'
    df_pred = pd.concat(df_pred_list)
    if args.agg_dose:
        if args.single:
#            df_pred.sort_values(['Source', 'Sample', 'Drug1', target], inplace=True)
            df_pred.sort_values(['Sample', 'Drug1', target], inplace=True)
        else:
            df_pred.sort_values(['Source', 'Sample', 'Drug1', 'Drug2', target], inplace=True)
    else:
        if args.single:
#            df_pred.sort_values(['Source', 'Sample', 'Drug1', 'Dose1', 'Growth'], inplace=True)
            df_pred.sort_values(['Sample', 'Drug1', 'Dose1', 'Growth'], inplace=True)
        else:
#            df_pred.sort_values(['Source', 'Sample', 'Drug1', 'Drug2', 'Dose1', 'Dose2', 'Growth'], inplace=True)
            df_pred.sort_values(['Sample', 'Drug1', 'Drug2', 'Dose1', 'Dose2', 'Growth'], inplace=True)
    df_pred.to_csv(pred_fname, sep='\t', index=False, float_format='%.4g')
    logger.info('Testing predictions stored in file: {}'.format(pred_fname))

    if args.cp:
        logger.info('Model stored in file: {}'.format(prefix+'.model.h5'))
#        logger.info('Model weights stored in file: {}'.format(prefix+cv_ext+'.weights.h5'))
        logger.info('Model weights stored in file: {}'.format(args.save_path + '/' + args.save_weights))

    if args.cv > 1:
        scores = uno.evaluate_prediction(df_pred[target], df_pred['Predicted'+target])
        uno.log_evaluation(scores, logger, description='Combining cross validation folds:')

    for test_source in loader.test_sep_sources:
        test_gen = uno_combined_data_generator.CombinedDataGenerator(loader, partition='test', batch_size=args.batch_size, source=test_source)
        df_test = test_gen.get_response(copy=True)
        y_test = df_test[target].values
        n_test = len(y_test)
        if n_test == 0:
            continue
        if args.no_gen:
            x_test_list, y_test = test_gen.get_slice(size=test_gen.size, single=args.single)
            y_test_pred = model.predict(x_test_list, batch_size=args.batch_size)
            if args.loss == 'heteroscedastic':
                y_test_pred = y_test_pred[:,0]
            elif args.loss == 'quantile':
                y_test_pred = y_test_pred[:,0] # 50th quantile prediction
        else:
            y_test_pred = model.predict_generator(test_gen.flow(single=args.single), test_gen.steps)
            if args.loss == 'heteroscedastic':
                y_test_pred = y_test_pred[:test_gen.size,0]
            elif args.loss == 'quantile':
                y_test_pred = y_test_pred[:test_gen.size,0] # 50th quantile prediction
            else:
                y_test_pred = y_test_pred[:test_gen.size]

        y_test_pred = y_test_pred.flatten()
        scores = uno.evaluate_prediction(y_test, y_test_pred)
        uno.log_evaluation(scores, logger, description='Testing on data from {} ({})'.format(test_source, n_test))

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
