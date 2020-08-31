from __future__ import absolute_import

#__version__ = '0.0.0'

#import from data_utils
from data_utils import load_csv_data
from data_utils import load_Xy_one_hot_data2
from data_utils import load_Xy_data_noheader
from data_utils import drop_impute_and_scale_dataframe
from data_utils import discretize_dataframe
from data_utils import discretize_array
from data_utils import lookup

#import from file_utils
from file_utils import get_file

#import from default_utils
from default_utils import ArgumentStruct
from default_utils import Benchmark
from default_utils import str2bool
from default_utils import finalize_parameters
from default_utils import fetch_file
from default_utils import verify_path
from default_utils import keras_default_config
from default_utils import set_up_logger
from default_utils import check_flag_conflicts

from generic_utils import Progbar

# import from viz_utils
from viz_utils import plot_history
from viz_utils import plot_scatter
from viz_utils import plot_array
from viz_utils import plot_density_observed_vs_predicted
from viz_utils import plot_2d_density_sigma_vs_error
from viz_utils import plot_histogram_error_per_sigma
from viz_utils import plot_decile_predictions
from viz_utils import plot_calibration_interpolation
from viz_utils import plot_calibrated_std
from viz_utils import plot_contamination


# import from uq_utils
from uq_utils import generate_index_distribution
from uq_utils import compute_statistics_homoscedastic_summary
from uq_utils import compute_statistics_homoscedastic
from uq_utils import compute_statistics_heteroscedastic
from uq_utils import compute_statistics_quantile
from uq_utils import split_data_for_empirical_calibration
from uq_utils import compute_empirical_calibration_interpolation

# import from profiling_utils
from profiling_utils import start_profiling
from profiling_utils import stop_profiling

# import from data_preprocessing_utils
from data_preprocessing_utils import quantile_normalization
from data_preprocessing_utils import generate_cross_validation_partition

# feature selection
from feature_selection_utils import select_features_by_missing_values
from feature_selection_utils import select_features_by_variation
from feature_selection_utils import select_decorrelated_features

# noise injection 
from noise_utils import label_flip
from noise_utils import label_flip_correlated

# P1-specific
from P1_utils import coxen_single_drug_gene_selection
from P1_utils import coxen_multi_drug_gene_selection
from P1_utils import generate_gene_set_data
from P1_utils import combat_batch_effect_removal

# import benchmark-dependent utils
import sys
if 'keras' in sys.modules:
    print ('Importing candle utils for keras')
    #import from keras_utils
    #from keras_utils import dense
    #from keras_utils import add_dense
    from keras_utils import build_initializer
    from keras_utils import build_optimizer
    from keras_utils import get_function
    from keras_utils import set_seed
    from keras_utils import set_parallelism_threads
    from keras_utils import PermanentDropout
    from keras_utils import register_permanent_dropout
    from keras_utils import LoggingCallback
    from keras_utils import MultiGPUCheckpoint
    from keras_utils import r2
    from keras_utils import mae
    from keras_utils import mse

    from viz_utils import plot_metrics

    from solr_keras import CandleRemoteMonitor
    from solr_keras import compute_trainable_params
    from solr_keras import TerminateOnTimeOut

    from uq_keras_utils import abstention_loss
    from uq_keras_utils import sparse_abstention_loss
    from uq_keras_utils import abstention_acc_metric
    from uq_keras_utils import sparse_abstention_acc_metric
    from uq_keras_utils import abstention_metric
    from uq_keras_utils import acc_class_i_metric
    from uq_keras_utils import abstention_acc_class_i_metric
    from uq_keras_utils import abstention_class_i_metric
    from uq_keras_utils import AbstentionAdapt_Callback
    from uq_keras_utils import modify_labels
    from uq_keras_utils import add_model_output
    from uq_keras_utils import r2_heteroscedastic_metric
    from uq_keras_utils import mae_heteroscedastic_metric
    from uq_keras_utils import mse_heteroscedastic_metric
    from uq_keras_utils import meanS_heteroscedastic_metric
    from uq_keras_utils import heteroscedastic_loss
    from uq_keras_utils import quantile_loss
    from uq_keras_utils import triple_quantile_loss
    from uq_keras_utils import quantile_metric
    from uq_keras_utils import add_index_to_output
    from uq_keras_utils import contamination_loss
    from uq_keras_utils import Contamination_Callback
    from uq_keras_utils import mse_contamination_metric
    from uq_keras_utils import mae_contamination_metric
    from uq_keras_utils import r2_contamination_metric

    from clr_keras_utils import CyclicLR
    from clr_keras_utils import clr_set_args
    from clr_keras_utils import clr_callback

elif 'torch' in sys.modules:
    print ('Importing candle utils for pytorch')
    from pytorch_utils import set_seed
    from pytorch_utils import build_optimizer
    from pytorch_utils import build_activation
    from pytorch_utils import get_function
    from pytorch_utils import initialize
    from pytorch_utils import xent
    from pytorch_utils import mse
    from pytorch_utils import set_parallelism_threads # for compatibility

else:
    raise Exception('No backend has been specified.')


