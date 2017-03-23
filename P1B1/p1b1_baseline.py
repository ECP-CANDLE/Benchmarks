from __future__ import print_function

import pandas as pd
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import Callback, ModelCheckpoint

import p1b1


EPOCH = 2
BATCH = 50

P     = 60025    # 245 x 245
N1    = 2000
NE    = 600      # encoded dim
F_MAX = 33.3
DR    = 0.2


X_train, X_test = p1b1.load_data()

input_dim = X_train.shape[1]
output_dim = input_dim

input_vector = Input(shape=(input_dim,))
x = Dense(N1, activation='sigmoid')(input_vector)
# x = Dropout(DR)(x)
x = Dense(NE, activation='sigmoid')(x)
encoded = x

x = Dense(N1, activation='sigmoid')(encoded)
# x = Dropout(DR)(x)
x = Dense(output_dim, activation='sigmoid')(x)
decoded = x

ae = Model(input_vector, decoded)
print(ae.summary())

encoded_input = Input(shape=(NE,))
encoder = Model(input_vector, encoded)
decoder = Model(encoded_input, ae.layers[-1](ae.layers[-2](encoded_input)))


# train = (pd.read_csv('breast.train.csv').values).astype('float32')
# X_train = train[:, 0:P] / F_MAX

# test = (pd.read_csv('breast.test.csv').values).astype('float32')
# X_test = test[:, 0:P] / F_MAX

ae.compile(optimizer='rmsprop', loss='mean_squared_error')

ae.fit(X_train, X_train,
       batch_size=BATCH,
       epochs=EPOCH,
       validation_split=0.2)

encoded_image = encoder.predict(X_test)
decoded_image = decoder.predict(encoded_image)
diff = decoded_image - X_test

# diff = ae.predict(X_test) - X_test
diffs = diff.ravel()


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.hist(diffs, bins='auto')
plt.title("Histogram of Errors with 'auto' bins")
plt.savefig('histogram.png')
