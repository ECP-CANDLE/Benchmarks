# Setup

import os

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

file_path = os.path.dirname(os.path.realpath(__file__))

import candle
import smiles_transformer as st
import tensorflow.config.experimental

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
try:
    for gpu in gpus:
        print("setting memory growth")
        tensorflow.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


def initialize_parameters(default_model="regress_default_model.txt"):

    # Build benchmark object
    sctBmk = st.BenchmarkST(
        st.file_path,
        default_model,
        "keras",
        prog="p1b1_baseline",
        desc="Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(sctBmk)

    return gParameters


# Train and Evaluate


def run(params):

    x_train, y_train, x_val, y_val = st.load_data(params)

    model = st.transformer_model(params)

    kerasDefaults = candle.keras_default_config()

    optimizer = candle.build_optimizer(
        params["optimizer"], params["learning_rate"], kerasDefaults
    )

    # optimizer = optimizers.deserialize({'class_name': params['optimizer'], 'config': {}})

    # I don't know why we set base_lr. It doesn't appear to be used.
    # if 'base_lr' in params and params['base_lr'] > 0:
    #     base_lr = params['base_lr']
    # else:
    #     base_lr = K.get_value(optimizer.lr)

    # if 'learning_rate' in params and params['learning_rate'] > 0:
    #     K.set_value(optimizer.lr, params['learning_rate'])
    #     print('Done setting optimizer {} learning rate to {}'.format(
    #         params['optimizer'], params['learning_rate']))

    model.compile(loss=params["loss"], optimizer=optimizer, metrics=["mae", st.r2])

    # set up a bunch of callbacks to do work during model training..

    checkpointer = ModelCheckpoint(
        filepath="smile_regress.autosave.model.h5",
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )
    csv_logger = CSVLogger("smile_regress.training.log")
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.75,
        patience=20,
        verbose=1,
        mode="auto",
        epsilon=0.0001,
        cooldown=3,
        min_lr=0.000000001,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=100, verbose=1, mode="auto")

    history = model.fit(
        x_train,
        y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[checkpointer, csv_logger, reduce_lr, early_stop],
    )

    model.load_weights("smile_regress.autosave.model.h5")

    return history


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
