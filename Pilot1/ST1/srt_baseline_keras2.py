# Setup

import os
import tensorflow as tf
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

strategy = tf.distribute.MirroredStrategy()
print('tensorflow version: {}'.format(tf.__version__))
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

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

    EPOCH = params['epochs']
    BATCH = params['batch_size']
    GLOBAL_BATCH_SIZE = BATCH * strategy.num_replicas_in_sync

    # Load data and create distributed data sets
    x_train, y_train, x_val, y_val = st.load_data(params)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(GLOBAL_BATCH_SIZE,
                                                                drop_remainder=True,
                                                                num_parallel_calls=None,
                                                                deterministic=None,
                                                               ).repeat(EPOCH)
    print(train_ds)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(GLOBAL_BATCH_SIZE,
                                                                drop_remainder=True,
                                                                num_parallel_calls=None,
                                                                deterministic=None,).repeat(EPOCH)
    print(val_ds)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)
    train_dist = strategy.experimental_distribute_dataset(train_ds)
    val_dist = strategy.experimental_distribute_dataset(val_ds)

    steps = x_train.shape[0]//GLOBAL_BATCH_SIZE
    validation_steps = x_val.shape[0]//GLOBAL_BATCH_SIZE
    print('steps {}\nvalidation steps {}'.format(steps, validation_steps))
    print('BATCH {}\nGLOBAL_BATCH_SIZE {}'.format(BATCH, GLOBAL_BATCH_SIZE))
    with strategy.scope():
        model = st.transformer_model(params)
        # Instanciate checkpointing callback and set restart epoch
        #
        initial_epoch=0
        ckpt = candle.CandleCkptKeras(params, verbose=True)
        ckpt.set_model(model)
        J = ckpt.restart(model)
        if J is not None:
            initial_epoch = J["epoch"]
            print("restarting from ckpt: initial_epoch: %i" % initial_epoch)


        kerasDefaults = candle.keras_default_config()
        optimizer = candle.build_optimizer(
            params["optimizer"], params["learning_rate"], kerasDefaults
        )
        model.compile(loss=params["loss"], optimizer=optimizer, metrics=["mae", st.r2])

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
        #x_train,
        #y_train,
        train_dist,
        #batch_size=params["batch_size"],
        batch_size=GLOBAL_BATCH_SIZE,
        steps_per_epoch=int(steps),
        epochs=params["epochs"],
        verbose=1,
        #validation_data=(x_val, y_val),
        validation_data=val_dist,
        validation_steps=validation_steps,
        #callbacks=[checkpointer, csv_logger, reduce_lr, early_stop],
        callbacks=[ckpt, csv_logger, reduce_lr, early_stop],
        initial_epoch=initial_epoch,
    )
    ckpt.report_final()

    model.load_weights("smile_regress.autosave.model.h5")

    return history


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
