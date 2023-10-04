#! /usr/bin/env python

from __future__ import division, print_function

import logging
import os
import time

import candle
import numpy as np
import pandas as pd
import tensorflow as tf
import uno as benchmark
import uno_data
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    Callback,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from uno_data import CombinedDataGenerator, CombinedDataLoader, DataFeeder

logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.disable_eager_execution()


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ""
    ext += ".A={}".format(args.activation)
    ext += ".B={}".format(args.batch_size)
    ext += ".E={}".format(args.epochs)
    ext += ".O={}".format(args.optimizer)
    # ext += '.LEN={}'.format(args.maxlen)
    ext += ".LR={}".format(args.learning_rate)
    ext += ".CF={}".format("".join([x[0] for x in sorted(args.cell_features)]))
    ext += ".DF={}".format("".join([x[0] for x in sorted(args.drug_features)]))
    if args.feature_subsample > 0:
        ext += ".FS={}".format(args.feature_subsample)
    if args.dropout > 0:
        ext += ".DR={}".format(args.dropout)
    if args.warmup_lr:
        ext += ".wu_lr"
    if args.reduce_lr:
        ext += ".re_lr"
    if args.residual:
        ext += ".res"
    if args.use_landmark_genes:
        ext += ".L1000"
    if args.no_gen:
        ext += ".ng"
    for i, n in enumerate(args.dense):
        if n > 0:
            ext += ".D{}={}".format(i + 1, n)
    if args.dense_feature_layers != args.dense:
        for i, n in enumerate(args.dense):
            if n > 0:
                ext += ".FD{}={}".format(i + 1, n)

    return ext


def evaluate_prediction(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return {"mse": mse, "mae": mae, "r2": r2, "corr": corr}


def log_evaluation(metric_outputs, logger, description="Comparing y_true and y_pred:"):
    logger.info(description)
    for metric, value in metric_outputs.items():
        logger.info("  {}: {:.8f}".format(metric, value))


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (
            epoch,
            ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())),
        )
        self.print_fcn(msg)


class PermanentDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0.0 < self.rate < 1.0:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.rate, noise_shape)
        return x


class MultiGPUCheckpoint(ModelCheckpoint):
    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model


def build_feature_model(
    input_shape,
    name="",
    dense_layers=[1000, 1000],
    kernel_initializer="glorot_normal",
    activation="relu",
    residual=False,
    dropout_rate=0,
    permanent_dropout=True,
):
    x_input = Input(shape=input_shape)
    h = x_input
    for i, layer in enumerate(dense_layers):
        x = h
        h = Dense(layer, activation=activation, kernel_initializer=kernel_initializer)(
            h
        )
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    model = Model(x_input, h, name=name)
    return model


class SimpleWeightSaver(Callback):
    def __init__(self, fname):
        self.fname = fname

    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model

    def on_train_end(self, logs={}):
        self.model.save_weights(self.fname)


def build_model(loader, args, permanent_dropout=True, silent=False):
    input_models = {}
    dropout_rate = args.dropout

    initializer = (
        "glorot_normal"
        if hasattr(args, "initialization") is False
        else args.initialization
    )
    kernel_initializer = candle.build_initializer(
        initializer, candle.keras_default_config(), args.rng_seed
    )

    for fea_type, shape in loader.feature_shapes.items():
        base_type = fea_type.split(".")[0]
        if base_type in ["cell", "drug"]:
            if args.dense_cell_feature_layers is not None and base_type == "cell":
                dense_feature_layers = args.dense_cell_feature_layers
            elif args.dense_drug_feature_layers is not None and base_type == "drug":
                dense_feature_layers = args.dense_drug_feature_layers
            else:
                dense_feature_layers = args.dense_feature_layers

            box = build_feature_model(
                input_shape=shape,
                name=fea_type,
                dense_layers=dense_feature_layers,
                kernel_initializer=kernel_initializer,
                dropout_rate=dropout_rate,
                permanent_dropout=permanent_dropout,
            )
            if not silent:
                logger.debug("Feature encoding submodel for %s:", fea_type)
                box.summary(print_fn=logger.debug)
            input_models[fea_type] = box

    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name="input." + fea_name)
        inputs.append(fea_input)
        if fea_type in input_models:
            input_model = input_models[fea_type]
            encoded = input_model(fea_input)
        else:
            encoded = fea_input
        encoded_inputs.append(encoded)

    merged = keras.layers.concatenate(encoded_inputs)

    h = merged
    for i, layer in enumerate(args.dense):
        x = h
        h = Dense(
            layer, activation=args.activation, kernel_initializer=kernel_initializer
        )(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if args.residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    output = Dense(1, kernel_initializer=kernel_initializer)(h)

    return Model(inputs, output)


def initialize_parameters(default_model="uno_default_model.txt"):

    # Build benchmark object
    unoBmk = benchmark.BenchmarkUno(
        benchmark.file_path,
        default_model,
        "keras",
        prog="uno_baseline",
        desc="Build neural network based models to predict tumor response to single and paired drugs.",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(unoBmk)
    # benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters


def run(params):
    args = candle.ArgumentStruct(**params)
    candle.set_seed(args.rng_seed)
    ext = extension_from_parameters(args)
    candle.verify_path(args.save_path)
    # prefix = args.save_path + ext
    logfile = args.logfile if args.logfile else "save/python.log"
    candle.set_up_logger(
        logfile,
        logger,
        args.verbose,
        fmt_line="%(asctime)s %(message)s",
    )
    logger.info("UNO RUN ...")

    import tensorflow as tf

    # from tensorflow.python.client import device_lib
    print("TF version: " + tf.__version__)
    # gpus = tf.config.list_physical_devices('GPU')
    # tf.debugging.set_log_device_placement(True)
    # tf.config.set_visible_devices(gpus[0], 'GPU')
    # print(device_lib.list_local_devices())

    logger.info("Params: {}".format(params))

    if len(args.gpus) > 0:
        import tensorflow as tf

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = ",".join(map(str, args.gpus))
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    datafile = None
    if args.use_exported_data is not None:
        if os.environ["CANDLE_DATA_DIR"] is not None:
            datadir = os.environ["CANDLE_DATA_DIR"]
            datafile = os.path.join(datadir, args.use_exported_data)
        else:
            datafile = args.use_exported_data

    loader = CombinedDataLoader(seed=args.rng_seed)
    loader.load(
        cache=args.cache,
        ncols=args.feature_subsample,
        agg_dose=args.agg_dose,
        cell_features=args.cell_features,
        drug_features=args.drug_features,
        drug_median_response_min=args.drug_median_response_min,
        drug_median_response_max=args.drug_median_response_max,
        use_landmark_genes=args.use_landmark_genes,
        use_filtered_genes=args.use_filtered_genes,
        cell_feature_subset_path=args.cell_feature_subset_path
        or args.feature_subset_path,
        drug_feature_subset_path=args.drug_feature_subset_path
        or args.feature_subset_path,
        preprocess_rnaseq=args.preprocess_rnaseq,
        single=args.single,
        train_sources=args.train_sources,
        test_sources=args.test_sources,
        embed_feature_source=not args.no_feature_source,
        encode_response_source=not args.no_response_source,
        use_exported_data=args.use_exported_data,
    )

    target = args.agg_dose or "Growth"
    val_split = args.val_split
    train_split = 1 - val_split

    if args.export_csv:
        fname = args.export_csv
        loader.partition_data(
            cv_folds=args.cv,
            train_split=train_split,
            val_split=val_split,
            cell_types=args.cell_types,
            by_cell=args.by_cell,
            by_drug=args.by_drug,
            cell_subset_path=args.cell_subset_path,
            drug_subset_path=args.drug_subset_path,
        )
        train_gen = CombinedDataGenerator(
            loader, batch_size=args.batch_size, shuffle=args.shuffle
        )
        val_gen = CombinedDataGenerator(
            loader, partition="val", batch_size=args.batch_size, shuffle=args.shuffle
        )

        x_train_list, y_train = train_gen.get_slice(
            size=train_gen.size, dataframe=True, single=args.single
        )
        x_val_list, y_val = val_gen.get_slice(
            size=val_gen.size, dataframe=True, single=args.single
        )
        df_train = pd.concat([y_train] + x_train_list, axis=1)
        df_val = pd.concat([y_val] + x_val_list, axis=1)
        df = pd.concat([df_train, df_val]).reset_index(drop=True)
        if args.growth_bins > 1:
            df = uno_data.discretize(df, "Growth", bins=args.growth_bins)
        df.to_csv(fname, sep="\t", index=False, float_format="%.3g")
        return

    if args.export_data:
        fname = args.export_data
        loader.partition_data(
            cv_folds=args.cv,
            train_split=train_split,
            val_split=val_split,
            cell_types=args.cell_types,
            by_cell=args.by_cell,
            by_drug=args.by_drug,
            cell_subset_path=args.cell_subset_path,
            drug_subset_path=args.drug_subset_path,
        )
        train_gen = CombinedDataGenerator(
            loader, batch_size=args.batch_size, shuffle=args.shuffle
        )
        val_gen = CombinedDataGenerator(
            loader, partition="val", batch_size=args.batch_size, shuffle=args.shuffle
        )
        store = pd.HDFStore(fname, complevel=9, complib="blosc:snappy")

        config_min_itemsize = {"Sample": 30, "Drug1": 10}
        if not args.single:
            config_min_itemsize["Drug2"] = 10

        for partition in ["train", "val"]:
            gen = train_gen if partition == "train" else val_gen
            for i in range(gen.steps):
                x_list, y = gen.get_slice(
                    size=args.batch_size, dataframe=True, single=args.single
                )

                for j, input_feature in enumerate(x_list):
                    input_feature.columns = [""] * len(input_feature.columns)
                    store.append(
                        "x_{}_{}".format(partition, j),
                        input_feature.astype("float32"),
                        format="table",
                        data_columns=True,
                    )
                store.append(
                    "y_{}".format(partition),
                    y.astype({target: "float32"}),
                    format="table",
                    data_columns=True,
                    min_itemsize=config_min_itemsize,
                )
                logger.info(
                    "Generating {} dataset. {} / {}".format(partition, i, gen.steps)
                )

        # save input_features and feature_shapes from loader
        store.put("model", pd.DataFrame())
        store.get_storer("model").attrs.input_features = loader.input_features
        store.get_storer("model").attrs.feature_shapes = loader.feature_shapes

        store.close()
        logger.info("Completed generating {}".format(fname))
        return

    if args.use_exported_data is None:
        loader.partition_data(
            cv_folds=args.cv,
            train_split=train_split,
            val_split=val_split,
            cell_types=args.cell_types,
            by_cell=args.by_cell,
            by_drug=args.by_drug,
            cell_subset_path=args.cell_subset_path,
            drug_subset_path=args.drug_subset_path,
        )

    model = build_model(loader, args)
    logger.info("Combined model:")
    model.summary(print_fn=logger.info)
    # plot_model(model, to_file=prefix+'.model.png', show_shapes=True)

    if args.cp:
        model_json = model.to_json()
        with open("model.json", "w") as f:
            print(model_json, file=f)

    def warmup_scheduler(epoch):
        lr = args.learning_rate or base_lr * args.batch_size / 100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5 - epoch) + lr * epoch) / 5)
        logger.debug(
            "Epoch {}: lr={:.5g}".format(epoch, K.get_value(model.optimizer.lr))
        )
        return K.get_value(model.optimizer.lr)

    df_pred_list = []

    cv_ext = ""
    cv = args.cv if args.cv > 1 else 1

    for fold in range(cv):
        if args.cv > 1:
            logger.info("Cross validation fold {}/{}:".format(fold + 1, cv))
            cv_ext = ".cv{}".format(fold + 1)

        template_model = build_model(loader, args, silent=True)
        initial_epoch = 0

        logger.info("CKPT CONSTRUCT...")
        ckpt = candle.CandleCkptKeras(params, verbose=True)
        logger.info("CKPT CONSTRUCT OK.")
        logger.info("template model: " + str(template_model))

        ckpt.set_model(template_model)
        J = ckpt.restart(params)

        if J is not None:
            initial_epoch = J["epoch"]
            best_metric_last = J["best_metric_last"]
            params["ckpt_best_metric_last"] = best_metric_last
            print("initial_epoch: %i" % initial_epoch)

        elif args.initial_weights is not None:
            logger.info(
                "Loading initial weights from '{}'".format(args.initial_weights)
            )
            start = time.time()
            template_model.load_weights(args.initial_weights)
            stop = time.time()
            duration = stop - start
            logger.info("Loaded from initial_weights in %0.3f seconds." % duration)

        if len(args.gpus) > 1:
            from tensorflow.keras.utils import multi_gpu_model

            gpu_count = len(args.gpus)
            logger.info("Multi GPU with {} gpus".format(gpu_count))
            model = multi_gpu_model(template_model, cpu_merge=False, gpus=gpu_count)
        else:
            model = template_model

        optimizer = optimizers.deserialize({"class_name": args.optimizer, "config": {}})
        base_lr = args.base_lr or K.get_value(optimizer.lr)
        if args.learning_rate:
            K.set_value(optimizer.lr, args.learning_rate)

        logger.info("COMPILE")
        model.compile(
            loss=args.loss, optimizer=optimizer, metrics=[candle.mae, candle.r2]
        )

        # calculate trainable and non-trainable params
        params.update(candle.compute_trainable_params(model))

        candle_monitor = candle.CandleRemoteMonitor(params=params)
        timeout_monitor = candle.TerminateOnTimeOut(params["timeout"])
        patience = 10
        if "patience" in params:
            patience = int(params["patience"])
            logger.info("setting patience: %i" % patience)
        es_monitor = keras.callbacks.EarlyStopping(patience=patience, verbose=1)

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
        )
        warmup_lr = LearningRateScheduler(warmup_scheduler)
        # prefix + cv_ext + '.
        checkpointer = MultiGPUCheckpoint("model.h5", save_best_only=True)
        tensorboard = TensorBoard(
            log_dir="tb/{}{}{}".format(args.tb_prefix, ext, cv_ext)
        )
        history_logger = LoggingCallback(logger.debug)

        callbacks = [candle_monitor, timeout_monitor, history_logger, ckpt]
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
            callbacks.append(MultiGPUCheckpoint(args.save_weights))

        if args.use_exported_data is not None:
            train_gen = DataFeeder(
                filename=args.use_exported_data,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
                agg_dose=args.agg_dose,
                on_memory=args.on_memory_loader,
            )
            val_gen = DataFeeder(
                partition="val",
                filename=args.use_exported_data,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
                agg_dose=args.agg_dose,
                on_memory=args.on_memory_loader,
            )
            test_gen = DataFeeder(
                partition="test",
                filename=args.use_exported_data,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
                agg_dose=args.agg_dose,
                on_memory=args.on_memory_loader,
            )
        else:
            train_gen = CombinedDataGenerator(
                loader,
                fold=fold,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
            )
            val_gen = CombinedDataGenerator(
                loader,
                partition="val",
                fold=fold,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
            )
            test_gen = CombinedDataGenerator(
                loader,
                partition="test",
                fold=fold,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
            )

        df_val = val_gen.get_response(copy=True)
        y_val = df_val[target].values
        y_shuf = np.random.permutation(y_val)
        log_evaluation(
            evaluate_prediction(y_val, y_shuf),
            logger,
            description="Between random pairs in y_val:",
        )

        if args.no_gen:
            x_train_list, y_train = train_gen.get_slice(
                size=train_gen.size, single=args.single
            )
            x_val_list, y_val = val_gen.get_slice(size=val_gen.size, single=args.single)
            history = model.fit(
                x_train_list,
                y_train,
                batch_size=args.batch_size,
                epochs=args.epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                validation_data=(x_val_list, y_val),
            )
        else:
            logger.info(
                "Data points per epoch: train = %d, val = %d, test = %d",
                train_gen.size,
                val_gen.size,
                test_gen.size,
            )
            logger.info(
                "Steps per epoch: train = %d, val = %d, test = %d",
                train_gen.steps,
                val_gen.steps,
                test_gen.steps,
            )
            history = model.fit(
                train_gen,
                epochs=args.epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                validation_data=val_gen,
            )

        # prediction on holdout(test) when exists or use validation set
        if test_gen.size > 0:
            df_val = test_gen.get_response(copy=True)
            y_val = df_val[target].values
            y_val_pred = model.predict(test_gen, steps=test_gen.steps + 1)
            y_val_pred = y_val_pred[: test_gen.size]
        else:
            if args.no_gen:
                y_val_pred = model.predict(x_val_list, batch_size=args.batch_size)
            else:
                val_gen.reset()
                y_val_pred = model.predict(val_gen, steps=val_gen.steps + 1)
                y_val_pred = y_val_pred[: val_gen.size]

        y_val_pred = y_val_pred.flatten()

        if "loss" in history.history:
            history_length = len(history.history["loss"])
            logger.info("history_length: %i" % history_length)
            history_expected = args.epochs - initial_epoch
            if history_length == history_expected:
                msg = "stopping: complete"
            elif history_length < history_expected:
                msg = "stopping: early"
            else:
                msg = "stopping: unexpected extra epochs!"
            print(msg)
            logger.info(msg)

        scores = evaluate_prediction(y_val, y_val_pred)
        log_evaluation(scores, logger)

        # df_val = df_val.assign(PredictedGrowth=y_val_pred, GrowthError=y_val_pred - y_val)
        df_val["Predicted" + target] = y_val_pred
        df_val[target + "Error"] = y_val_pred - y_val
        df_pred_list.append(df_val)

        candle.plot_metrics(
            history,
            title=None,
            skip_ep=0,
            outdir=os.path.dirname(args.save_path),
            add_lr=True,
        )

    pred_fname = "predicted.tsv"  # prefix +
    df_pred = pd.concat(df_pred_list)
    if args.agg_dose:
        if args.single:
            df_pred.sort_values(["Sample", "Drug1", target], inplace=True)
        else:
            df_pred.sort_values(
                ["Source", "Sample", "Drug1", "Drug2", target], inplace=True
            )
    else:
        if args.single:
            df_pred.sort_values(["Sample", "Drug1", "Dose1", "Growth"], inplace=True)
        else:
            df_pred.sort_values(
                ["Sample", "Drug1", "Drug2", "Dose1", "Dose2", "Growth"], inplace=True
            )
    df_pred.to_csv(pred_fname, sep="\t", index=False, float_format="%.4g")

    if args.cv > 1:
        scores = evaluate_prediction(df_pred[target], df_pred["Predicted" + target])
        log_evaluation(scores, logger, description="Combining cross validation folds:")

    for test_source in loader.test_sep_sources:
        test_gen = CombinedDataGenerator(
            loader, partition="test", batch_size=args.batch_size, source=test_source
        )
        df_test = test_gen.get_response(copy=True)
        y_test = df_test[target].values
        n_test = len(y_test)
        if n_test == 0:
            continue
        if args.no_gen:
            x_test_list, y_test = test_gen.get_slice(
                size=test_gen.size, single=args.single
            )
            y_test_pred = model.predict(x_test_list, batch_size=args.batch_size)
        else:
            y_test_pred = model.predict_generator(
                test_gen.flow(single=args.single), test_gen.steps
            )
            y_test_pred = y_test_pred[: test_gen.size]
        y_test_pred = y_test_pred.flatten()
        scores = evaluate_prediction(y_test, y_test_pred)
        log_evaluation(
            scores,
            logger,
            description="Testing on data from {} ({})".format(test_source, n_test),
        )

    if K.backend() == "tensorflow":
        K.clear_session()

    logger.handlers = []

    return history


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
