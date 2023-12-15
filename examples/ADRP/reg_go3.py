import argparse
import math
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ke
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, model_from_json, model_from_yaml
from tensorflow.keras.optimizers import SGD

file_path = os.path.dirname(os.path.realpath(__file__))

strategy = tf.distribute.MirroredStrategy(
    [
        "/xpu:0"
        #                                        ,'/xpu:1'
        #                                        ,'/xpu:2'
        #                                        ,'/xpu:3'
        #                                        ,'/xpu:4'
        #                                        ,'/xpu:5'
        #                                        ,'/xpu:6','/xpu:7'
        #                                        ,'/xpu:8','/xpu:9'
        #                                        ,'/xpu:10','/xpu:11'
    ]
)
print("tensorflow version: {}".format(tf.__version__))
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# parse args
psr = argparse.ArgumentParser(description="input csv file")
psr.add_argument("--in", default="in_file")
psr.add_argument("--ep", type=int, default=400)
args = vars(psr.parse_args())

EPOCH = args["ep"]
BATCH = 32
GLOBAL_BATCH_SIZE = BATCH * strategy.num_replicas_in_sync
DR = 0.1  # Dropout rate
data_path = args["in"]
print(args)


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def load_data():

    data_path = args["in"]

    df = (pd.read_csv(data_path, skiprows=1).values).astype("float32")
    df_y = df[:, 0].astype("float32")
    df_x = df[:, 1:PL].astype(np.float32)
    print("df_y: {}\n{}".format(df_y.shape, df_y))
    print("df_x: {}\n{}".format(df_x.shape, df_x))

    scaler = StandardScaler()
    df_x = scaler.fit_transform(df_x)

    X_train, X_test, Y_train, Y_test = train_test_split(
        df_x, df_y, test_size=0.20, random_state=42
    )

    return X_train, Y_train, X_test, Y_test


def load_data_from_parquet():

    data_path = args["in"]

    df = pd.read_parquet(data_path)
    df_y = df["reg"].values.astype("float32")
    df_x = df.iloc[:, 6:].values.astype("float32")
    print("df_y: {}\n{}".format(df_y.shape, df_y))
    print("df_x: {}\n{}".format(df_x.shape, df_x))

    scaler = StandardScaler()
    df_x = scaler.fit_transform(df_x)

    X_train, X_test, Y_train, Y_test = train_test_split(
        df_x, df_y, test_size=0.20, random_state=42
    )

    return X_train, Y_train, X_test, Y_test


# X_train, Y_train, X_test, Y_test = load_data()
X_train, Y_train, X_test, Y_test = load_data_from_parquet()
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train_shape: ", Y_train.shape)
print("Y_test shape: ", Y_test.shape)

steps = X_train.shape[0] // GLOBAL_BATCH_SIZE
validation_steps = X_test.shape[0] // GLOBAL_BATCH_SIZE
print(
    "samples {}, global_batch_size {}, steps {}".format(
        X_train.shape[0], GLOBAL_BATCH_SIZE, steps
    )
)
print(
    "val samples {}, global_batch_size {}, val_steps {}".format(
        X_test.shape[0], GLOBAL_BATCH_SIZE, validation_steps
    )
)


train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    .batch(
        GLOBAL_BATCH_SIZE,
        drop_remainder=True,
        num_parallel_calls=None,
        deterministic=None,
    )
    .repeat(EPOCH)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    .batch(
        GLOBAL_BATCH_SIZE,
        drop_remainder=True,
        num_parallel_calls=None,
        deterministic=None,
    )
    .repeat(EPOCH)
)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.DATA
)
train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)

train_dist = strategy.experimental_distribute_dataset(train_ds)
val_dist = strategy.experimental_distribute_dataset(val_ds)


with strategy.scope():
    # inputs = Input(shape=(PS,))
    inputs = Input(shape=(1826,))
    x = Dense(250, activation="relu")(inputs)
    x = Dropout(DR)(x)
    x = Dense(125, activation="relu")(x)
    x = Dropout(DR)(x)
    x = Dense(60, activation="relu")(x)
    x = Dropout(DR)(x)
    x = Dense(30, activation="relu")(x)
    x = Dropout(DR)(x)
    outputs = Dense(1, activation="relu")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(
        loss="mean_squared_error",
        optimizer=SGD(lr=0.0001, momentum=0.9),
        metrics=["mae", r2],
    )

# set up a bunch of callbacks to do work during model training..

checkpointer = ModelCheckpoint(
    filepath="reg_go.autosave.model.h5",
    verbose=1,
    save_weights_only=False,
    save_best_only=True,
)
csv_logger = CSVLogger("reg_go.training.log")
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

from datetime import datetime as dt

print(
    "{} calling model.fit".format(
        dt.fromtimestamp(dt.timestamp(dt.now())).strftime("%D %H:%M:%S.%s")
    )
)
history = model.fit(
    # X_train,
    # Y_train,
    train_dist,
    batch_size=GLOBAL_BATCH_SIZE,
    steps_per_epoch=int(steps),
    epochs=EPOCH,
    verbose=1,
    # validation_data=(X_test, Y_test),
    validation_data=val_dist,
    validation_steps=validation_steps,
    callbacks=[checkpointer, csv_logger, reduce_lr, early_stop],
)
print(
    "{} done calling model.fit".format(
        dt.fromtimestamp(dt.timestamp(dt.now())).strftime("%D %H:%M:%S.%s")
    )
)


score = model.evaluate(X_test, Y_test, verbose=0)

print(score)

print(history.history.keys())
# dict_keys(['val_loss', 'val_mae', 'val_r2', 'loss', 'mae', 'r2', 'lr'])

# summarize history for MAE
# plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history["mae"])
# plt.plot(history.history['val_mean_absolute_error'])
plt.plot(history.history["val_mae"])

plt.title("Model Mean Absolute Error")
plt.ylabel("mae")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")

plt.savefig("reg_go.mae.png", bbox_inches="tight")
plt.savefig("reg_go.mae.pdf", bbox_inches="tight")

plt.close()

# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")

plt.savefig("reg_go.loss.png", bbox_inches="tight")
plt.savefig("reg_go.loss.pdf", bbox_inches="tight")

plt.close()

print("Test val_loss:", score[0])
print("Test val_mae:", score[1])

# serialize model to JSON
model_json = model.to_json()
with open("reg_go.model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("reg_go.model.h5")
print("Saved model to disk")

# load json and create model
json_file = open("reg_go.model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model_json = model_from_json(loaded_model_json)

# load weights into new model
loaded_model_json.load_weights("reg_go.model.h5")
print("Loaded json model from disk")

# evaluate json loaded model on test data
loaded_model_json.compile(
    loss="mean_squared_error", optimizer="SGD", metrics=["mean_absolute_error"]
)
score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

print("json Validation loss:", score_json[0])
print("json Validation mae:", score_json[1])

predict_json_train = loaded_model_json.predict(X_train)

predict_json_test = loaded_model_json.predict(X_test)

pred_train = predict_json_train[:, 0]
pred_test = predict_json_test[:, 0]

np.savetxt("pred_train.csv", pred_train, delimiter=".", newline="\n", fmt="%.3f")
np.savetxt("pred_test.csv", pred_test, delimiter=",", newline="\n", fmt="%.3f")

print("Correlation prediction on test and Y_test:", np.corrcoef(pred_test, Y_test))
print("Correlation prediction on train and Y_train:", np.corrcoef(pred_train, Y_train))
