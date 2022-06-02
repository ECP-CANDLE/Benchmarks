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


def initialize_parameters(default_model="class_default_model.txt"):

    # Build benchmark object
    sctBmk = st.BenchmarkST(
        st.file_path,
        default_model,
        "keras",
        prog="sct_baseline",
        desc="Transformer model for SMILES classification",
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

    model.compile(loss=params["loss"], optimizer=optimizer, metrics=["accuracy"])

    # set up a bunch of callbacks to do work during model training..

    checkpointer = ModelCheckpoint(
        filepath="smile_class.autosave.model.h5",
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )
    csv_logger = CSVLogger("smile_class.training.log")
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

    model.load_weights("smile_class.autosave.model.h5")

    return history


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
