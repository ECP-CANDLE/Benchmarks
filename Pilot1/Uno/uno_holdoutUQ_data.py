#! /usr/bin/env python

from __future__ import division, print_function

import logging
import os

from tensorflow.keras import backend as K

import uno as benchmark
import candle

from uno_trainUQ_keras2 import extension_from_parameters
from uno_data import CombinedDataLoader

logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def initialize_parameters(default_model='uno_defaultUQ_model.txt'):

    # Build benchmark object
    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, default_model, 'keras',
                                    prog='uno_holdoutUQ_data', desc='Build data split for UQ analysis in the problem of prediction of tumor response to single or drug pairs.')

    # Finalize parameters
    gParameters = candle.finalize_parameters(unoBmk)

    return gParameters


def run(params):
    args = candle.ArgumentStruct(**params)
    candle.set_seed(args.rng_seed)
    ext = extension_from_parameters(args)
    candle.verify_path(args.save_path)
    prefix = args.save_path + ext
    logfile = args.logfile if args.logfile else prefix + '.log'
    candle.set_up_logger(logfile, logger, args.verbose)
    logger.info('Params: {}'.format(params))

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
                partition_by=args.partition_by
                )

    # target = args.agg_dose or 'Growth'
    val_split = args.val_split
    train_split = 1 - val_split

    loader.partition_data(partition_by=args.partition_by,
                          cv_folds=args.cv, train_split=train_split,
                          val_split=val_split, cell_types=args.cell_types,
                          by_cell=args.by_cell, by_drug=args.by_drug,
                          cell_subset_path=args.cell_subset_path,
                          drug_subset_path=args.drug_subset_path
                          )

    print('partition_by: ', args.partition_by)
    if args.partition_by == 'drug_pair':
        fname_drugs = args.save_path + 'infer_drug_ids'
        pds = loader.get_drugs_in_val()
        with open(fname_drugs, 'w') as f:
            for item in pds:
                f.write('%s\n' % item)
        logger.info('Drug IDs in holdout set written in file: {}'.format(fname_drugs))
    elif args.partition_by == 'cell':
        fname_cells = args.save_path + 'infer_cell_ids'
        pcs = loader.get_cells_in_val()
        with open(fname_cells, 'w') as f:
            for item in pcs:
                f.write('%s\n' % item)
        logger.info('Cell IDs in holdout set written in file: {}'.format(fname_cells))
    else:
        fname_index = args.save_path + 'infer_index_ids'
        pins = loader.get_index_in_val()
        with open(fname_index, 'w') as f:
            for item in pins:
                f.write('%s\n' % item)
        logger.info('Indices in holdout set written in file: {}'.format(fname_index))


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
