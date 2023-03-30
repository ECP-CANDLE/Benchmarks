from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json

import candle
import p1b2
from p1b2_baseline_keras2 import initialize_parameters


def run(gParameters):
    # print("gParameters: ", gParameters)

    # Construct extension to save model
    ext = p1b2.extension_from_parameters(gParameters, ".keras")
    candle.verify_path(gParameters["save_path"])
    prefix = "{}/{}{}".format(gParameters["save_path"], "infer", ext)
    logfile = gParameters["logfile"] if gParameters["logfile"] else prefix + ".log"
    candle.set_up_logger(logfile, p1b2.logger, gParameters["verbose"])
    p1b2.logger.info("Params: {}".format(gParameters))

    # load the data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = p1b2.load_data(gParameters)

    # load json and create model
    json_file = open(gParameters["saved_model"], "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model_json.load_weights(gParameters["saved_weights"])
    p1b2.logger.info("Loaded json model from disk")

    # predict using loaded yaml model on test data
    y_pred = loaded_model_json.predict(x_test)
    scores = p1b2.evaluate_accuracy(y_pred, y_test, gParameters["one_hot_dtrep"])
    p1b2.logger.info("\nEvaluation on test data: {}".format(scores))


def main():
    gParameters = initialize_parameters()
    run(gParameters)


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
