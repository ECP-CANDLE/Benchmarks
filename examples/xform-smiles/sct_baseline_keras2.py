# Setup

import os
import sys

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam  # RMSprop, SGD

# import gzip

# import math
# import matplotlib
# matplotlib.use('Agg')

# import matplotlib.pyplot as plt


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, "..", "..", "common"))
sys.path.append(lib_path)

import candle
import smiles_transformer as st


def initialize_parameters(default_model="class_default_model.txt"):

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

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(lr=0.000001),
        metrics=["accuracy"],
    )

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


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
