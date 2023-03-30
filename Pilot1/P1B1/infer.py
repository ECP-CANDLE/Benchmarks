import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from sklearn.manifold import TSNE

import candle
import p1b1
from p1b1_baseline_keras2 import initialize_parameters


def run(gParameters):
    # print("gParameters: ", gParameters)
    seed = gParameters["rng_seed"]

    # Construct extension to save model
    ext = p1b1.extension_from_parameters(gParameters, ".keras")
    candle.verify_path(gParameters["save_path"])
    prefix = "{}{}{}".format(gParameters["save_path"], "infer", ext)
    logfile = gParameters["logfile"] if gParameters["logfile"] else prefix + ".log"
    candle.set_up_logger(logfile, p1b1.logger, gParameters["verbose"])
    p1b1.logger.info("Params: {}".format(gParameters))

    # load the data
    x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels = p1b1.load_data(gParameters)
    # re-format
    if gParameters["model"] == "cvae":
        test_inputs = [x_test, y_test]
    else:
        test_inputs = x_test

    # load json and create model
    json_file = open(gParameters["saved_model"], "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)
    # load encoder
    json_encoder_fname = gParameters["saved_model"][:-10] + "encoder.json"
    print("json_encoder_fname: ", json_encoder_fname)
    json_encoder_file = open(json_encoder_fname, "r")
    loaded_encoder_json = json_encoder_file.read()
    json_encoder_file.close()
    loaded_encoder_json = model_from_json(loaded_encoder_json)

    # load weights into new model
    loaded_model_json.load_weights(gParameters["saved_weights"])
    p1b1.logger.info("Loaded json model from disk")
    # load encoder weights into new model
    w_encoder_file = gParameters["saved_weights"][:-10] + "encoder.h5"
    loaded_encoder_json.load_weights(w_encoder_file)
    p1b1.logger.info("Loaded json encoder model from disk")

    # predict using loaded yaml model on test data
    predict_test = loaded_model_json.predict(test_inputs)
    # x_pred = model.predict(test_inputs)
    scores = p1b1.evaluate_autoencoder(predict_test, x_test)
    p1b1.logger.info("\nEvaluation on test data: {}".format(scores))

    # x_test_encoded = encoder.predict(test_inputs, batch_size=params["batch_size"])
    x_test_encoded = loaded_encoder_json.predict(test_inputs, batch_size=gParameters["batch_size"])
    y_test_classes = np.argmax(y_test, axis=1)
    candle.plot_scatter(x_test_encoded, y_test_classes, prefix + ".latent")

    if gParameters["tsne"]:
        tsne = TSNE(n_components=2, random_state=seed)
        x_test_encoded_tsne = tsne.fit_transform(x_test_encoded)
        candle.plot_scatter(
            x_test_encoded_tsne, y_test_classes, prefix + ".latent.tsne"
        )
    return


def main():
    gParameters = initialize_parameters()
    run(gParameters)


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
