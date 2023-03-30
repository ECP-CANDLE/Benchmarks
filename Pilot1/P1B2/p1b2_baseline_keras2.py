from __future__ import print_function

import candle
import numpy as np
import p1b2
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def initialize_parameters(default_model="p1b2_default_model.txt"):

    # Build benchmark object
    p1b2Bmk = p1b2.BenchmarkP1B2(
        p1b2.file_path,
        default_model,
        "keras",
        prog="p1b2_baseline",
        desc="Train Classifier - Pilot 1 Benchmark 2",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(p1b2Bmk)

    return gParameters


def build_model(gParameters, input_dim, output_dim):

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = candle.keras_default_config()

    # Initialize weights and learning rule
    initializer_weights = candle.build_initializer(
        gParameters["initialization"], kerasDefaults, gParameters["rng_seed"]
    )
    initializer_bias = candle.build_initializer("constant", kerasDefaults, 0.0)

    activation = gParameters["activation"]

    # Define MLP architecture
    input_vector = Input(shape=(input_dim,))

    layers = gParameters["dense"]

    if layers is not None:
        if type(layers) != list:
            layers = list(layers)
        for i, l in enumerate(layers):
            if i == 0:
                x = Dense(
                    l,
                    activation=activation,
                    kernel_initializer=initializer_weights,
                    bias_initializer=initializer_bias,
                    kernel_regularizer=l2(gParameters["reg_l2"]),
                    activity_regularizer=l2(gParameters["reg_l2"]),
                )(input_vector)
            else:
                x = Dense(
                    l,
                    activation=activation,
                    kernel_initializer=initializer_weights,
                    bias_initializer=initializer_bias,
                    kernel_regularizer=l2(gParameters["reg_l2"]),
                    activity_regularizer=l2(gParameters["reg_l2"]),
                )(x)
            if gParameters["dropout"]:
                x = Dropout(gParameters["dropout"])(x)
        output = Dense(
            output_dim,
            activation=activation,
            kernel_initializer=initializer_weights,
            bias_initializer=initializer_bias,
        )(x)
    else:
        output = Dense(
            output_dim,
            activation=activation,
            kernel_initializer=initializer_weights,
            bias_initializer=initializer_bias,
        )(input_vector)

    # Build MLP model
    mlp = Model(outputs=output, inputs=input_vector)

    # Define optimizer
    optimizer = candle.build_optimizer(
        gParameters["optimizer"], gParameters["learning_rate"], kerasDefaults
    )

    # Compile model
    mlp.compile(loss=gParameters["loss"], optimizer=optimizer, metrics=[gParameters["metrics"]])

    return mlp


def run(gParameters):

    # Construct extension to save model
    ext = p1b2.extension_from_parameters(gParameters, ".keras")
    candle.verify_path(gParameters["save_path"])
    # prefix = "{}{}".format(gParameters["save_path"], ext)
    prefix = "{}/{}{}".format(gParameters["save_path"], gParameters["model_name"], ext)
    logfile = gParameters["logfile"] if gParameters["logfile"] else prefix + ".log"
    candle.set_up_logger(logfile, p1b2.logger, gParameters["verbose"])
    p1b2.logger.info("Params: {}".format(gParameters))

    # Load dataset
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = p1b2.load_data(
        gParameters
    )
    p1b2.logger.info("Shape x_train: {}".format(x_train.shape))
    p1b2.logger.info("Shape x_val:   {}".format(x_val.shape))
    p1b2.logger.info("Shape x_test:  {}".format(x_test.shape))
    p1b2.logger.info("Shape y_train: {}".format(y_train.shape))
    p1b2.logger.info("Shape y_val:   {}".format(y_val.shape))
    p1b2.logger.info("Shape y_test:  {}".format(y_test.shape))

    p1b2.logger.info(
        "Range x_train: [{:.3g}, {:.3g}]".format(np.min(x_train), np.max(x_train))
    )
    p1b2.logger.info(
        "Range x_val: [{:.3g}, {:.3g}]".format(np.min(x_val), np.max(x_val))
    )
    p1b2.logger.info(
        "Range x_test: [{:.3g}, {:.3g}]".format(np.min(x_test), np.max(x_test))
    )
    p1b2.logger.info(
        "Range y_train: [{:.3g}, {:.3g}]".format(np.min(y_train), np.max(y_train))
    )
    p1b2.logger.info(
        "Range y_val: [{:.3g}, {:.3g}]".format(np.min(y_val), np.max(y_val))
    )
    p1b2.logger.info(
        "Range y_test: [{:.3g}, {:.3g}]".format(np.min(y_test), np.max(y_test))
    )

    # Seed random generator for training
    np.random.seed(gParameters["rng_seed"])

    # Build and compile model
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    model = build_model(gParameters, input_dim, output_dim)
    p1b2.logger.debug("Model: {}".format(model.to_json()))
    p1b2.logger.info(model.summary())

    # Build checkpointing
    if gParameters["ckpt_save_interval"] > 0:
        # Specified in configuration
        ckpt = candle.CandleCkptKeras(gParameters, verbose=False)
        ckpt.set_model(model)
        J = ckpt.restart(model)
        if J is not None:
            initial_epoch = J["epoch"]
            p1b2.logger.info("restarting from ckpt: initial_epoch: {}".format(initial_epoch))
    else:
        # By default
        gParameters["ckpt_save_best"] = True
        gParameters["ckpt_directory"] = gParameters["save_path"]
        gParameters["ckpt_keep_limit"] = 1
        gParameters["ckpt_save_interval"] = 1
        ckpt = candle.CandleCkptKeras(gParameters)
        ckpt.set_model(model)
        model_json = model.to_json()
        with open(prefix + ".model.json", "w") as f:
            print(model_json, file=f)

    model.fit(
        x_train,
        y_train,
        batch_size=gParameters["batch_size"],
        epochs=gParameters["epochs"],
        validation_data=(x_val, y_val),
        callbacks=[ckpt],
    )

    # model save
    # save_filepath = "model_mlp_W_" + ext
    # mlp.save_weights(save_filepath)

    # Evalute model on test set
    y_pred = model.predict(x_test)
    scores = p1b2.evaluate_accuracy(y_pred, y_test, gParameters["one_hot_dtrep"])

    p1b2.logger.info("\nEvaluation on test data: {}".format(scores))


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
    try:
        K.clear_session()
    except AttributeError:  # theano does not have this function
        pass
