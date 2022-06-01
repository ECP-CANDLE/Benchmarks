from __future__ import absolute_import

# import benchmark-dependent utils
import sys

from benchmark_def import Benchmark

# import from data_preprocessing_utils
from data_preprocessing_utils import (
    generate_cross_validation_partition,
    quantile_normalization,
)

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

# feature selection
from feature_selection_utils import (
    select_decorrelated_features,
    select_features_by_missing_values,
    select_features_by_variation,
)

# import from file_utils
from file_utils import get_file

# import from generic_utils
from generic_utils import Progbar

# import from helper_utils
from helper_utils import (
    fetch_file,
    keras_default_config,
    set_up_logger,
    str2bool,
    verify_path,
)

# noise injection
from noise_utils import (
    add_cluster_noise,
    add_column_noise,
    add_gaussian_noise,
    add_noise,
    label_flip,
    label_flip_correlated,
)

# P1-specific
from P1_utils import (
    combat_batch_effect_removal,
    coxen_multi_drug_gene_selection,
    coxen_single_drug_gene_selection,
    generate_gene_set_data,
)

# import from parsing_utils
from parsing_utils import (
    ArgumentStruct,
    check_flag_conflicts,
    finalize_parameters,
    parse_from_dictlist,
)

# import from profiling_utils
from profiling_utils import start_profiling, stop_profiling

# import from uq_utils
from uq_utils import (
    compute_empirical_calibration_interpolation,
    compute_statistics_heteroscedastic,
    compute_statistics_homoscedastic,
    compute_statistics_homoscedastic_summary,
    compute_statistics_quantile,
    generate_index_distribution,
    split_data_for_empirical_calibration,
)

# import from viz_utils
from viz_utils import (
    plot_2d_density_sigma_vs_error,
    plot_array,
    plot_calibrated_std,
    plot_calibration_interpolation,
    plot_contamination,
    plot_decile_predictions,
    plot_density_observed_vs_predicted,
    plot_histogram_error_per_sigma,
    plot_history,
    plot_scatter,
)

# __version__ = '0.0.0'


if "tensorflow.keras" in sys.modules:
    print("Importing candle utils for keras")
    # import from keras_utils
    # from keras_utils import dense
    # from keras_utils import add_dense
    from ckpt_keras_utils import CandleCheckpointCallback, MultiGPUCheckpoint, restart
    from clr_keras_utils import CyclicLR, clr_callback, clr_set_args
    from keras_utils import (
        LoggingCallback,
        PermanentDropout,
        build_initializer,
        build_optimizer,
        get_function,
        mae,
        mse,
        r2,
        register_permanent_dropout,
        set_parallelism_threads,
        set_seed,
    )
    from solr_keras import (
        CandleRemoteMonitor,
        TerminateOnTimeOut,
        compute_trainable_params,
    )
    from uq_keras_utils import (
        AbstentionAdapt_Callback,
        Contamination_Callback,
        abstention_acc_class_i_metric,
        abstention_acc_metric,
        abstention_class_i_metric,
        abstention_loss,
        abstention_metric,
        acc_class_i_metric,
        add_index_to_output,
        add_model_output,
        contamination_loss,
        heteroscedastic_loss,
        mae_contamination_metric,
        mae_heteroscedastic_metric,
        meanS_heteroscedastic_metric,
        modify_labels,
        mse_contamination_metric,
        mse_heteroscedastic_metric,
        quantile_loss,
        quantile_metric,
        r2_contamination_metric,
        r2_heteroscedastic_metric,
        sparse_abstention_acc_metric,
        sparse_abstention_loss,
        triple_quantile_loss,
    )
    from viz_utils import plot_metrics

elif "torch" in sys.modules:
    print("Importing candle utils for pytorch")
    from pytorch_utils import set_parallelism_threads  # for compatibility
    from pytorch_utils import (
        build_activation,
        build_optimizer,
        get_function,
        initialize,
        mse,
        set_seed,
        xent,
    )

else:
    raise Exception("No backend has been specified.")
