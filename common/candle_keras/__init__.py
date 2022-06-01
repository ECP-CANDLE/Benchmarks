from __future__ import absolute_import

# import from data_utils
from data_utils import (
    discretize_array,
    discretize_dataframe,
    drop_impute_and_scale_dataframe,
    load_csv_data,
    load_Xy_data_noheader,
    load_Xy_one_hot_data2,
    lookup,
)

# import from default_utils
from default_utils import (
    ArgumentStruct,
    Benchmark,
    fetch_file,
    finalize_parameters,
    keras_default_config,
    set_up_logger,
    str2bool,
    verify_path,
)

# import from file_utils
from file_utils import get_file
from generic_utils import Progbar

# import from keras_utils
# from keras_utils import dense
# from keras_utils import add_dense
from keras_utils import (
    LoggingCallback,
    PermanentDropout,
    build_initializer,
    build_optimizer,
    mae,
    mse,
    r2,
    register_permanent_dropout,
    set_parallelism_threads,
    set_seed,
)
from solr_keras import CandleRemoteMonitor, TerminateOnTimeOut, compute_trainable_params

# import from uq_keras_utils
from uq_keras_utils import (
    AbstentionAdapt_Callback,
    abs_acc,
    abs_acc_class1,
    abstention_loss,
    abstention_variable_initialization,
    acc_class1,
    add_model_output,
    modify_labels,
)

# import from uq_utils
from uq_utils import (
    apply_calibration,
    binning_for_calibration,
    compute_empirical_calibration,
    compute_statistics_heteroscedastic,
    compute_statistics_homoscedastic,
    compute_statistics_homoscedastic_summary,
    compute_statistics_quantile,
    compute_valid_calibration_interval,
    overprediction_check,
    split_data_for_empirical_calibration,
)

# import from viz_utils
from viz_utils import (
    plot_2d_density_sigma_vs_error,
    plot_calibration_and_errors,
    plot_density_observed_vs_predicted,
    plot_histogram_error_per_sigma,
    plot_history,
    plot_percentile_predictions,
    plot_scatter,
)

# __version__ = '0.0.0'
