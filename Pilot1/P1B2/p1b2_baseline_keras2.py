from __future__ import print_function

import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import Callback, ModelCheckpoint
from keras.regularizers import l2

import p1b2


BATCH_SIZE = 64
NB_EPOCH = 20                 # number of training epochs
PENALTY = 0.00001             # L2 regularization penalty
ACTIVATION = 'sigmoid'
FEATURE_SUBSAMPLE = None
DROP = None

L1 = 1024
L2 = 512
L3 = 256
L4 = 0
LAYERS = [L1, L2, L3, L4]


class BestLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf
        self.best_model = None

    def on_epoch_end(self, batch, logs={}):
        if float(logs.get('val_loss', 0)) < self.best_val_loss:
            self.best_model = self.model
        self.best_val_loss = min(float(logs.get('val_loss', 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get('val_acc', 0)), self.best_val_acc)


def extension_from_parameters():
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(ACTIVATION)
    ext += '.B={}'.format(BATCH_SIZE)
    ext += '.D={}'.format(DROP)
    ext += '.E={}'.format(NB_EPOCH)
    if FEATURE_SUBSAMPLE:
        ext += '.F={}'.format(FEATURE_SUBSAMPLE)
    for i, n in enumerate(LAYERS):
        if n:
            ext += '.L{}={}'.format(i+1, n)
    ext += '.P={}'.format(PENALTY)
    return ext


def main():
    (X_train, y_train), (X_test, y_test) = p1b2.load_data(n_cols=FEATURE_SUBSAMPLE)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = Sequential()

    model.add(Dense(LAYERS[0], input_dim=input_dim,
                    activation=ACTIVATION,
                    kernel_regularizer=l2(PENALTY),
                    activity_regularizer=l2(PENALTY)))

    for layer in LAYERS[1:]:
        if layer:
            if DROP:
                model.add(Dropout(DROP))
            model.add(Dense(layer, activation=ACTIVATION,
                            kernel_regularizer=l2(PENALTY),
                            activity_regularizer=l2(PENALTY)))

    model.add(Dense(output_dim, activation=ACTIVATION))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    ext = extension_from_parameters()
    checkpointer = ModelCheckpoint(filepath='model'+ext+'.h5', save_best_only=True)
    history = BestLossHistory()

    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=NB_EPOCH,
              validation_split=0.2,
              callbacks=[history, checkpointer])

    y_pred = history.best_model.predict(X_test)

    print('best_val_loss={:.5f} best_val_acc={:.5f}'.format(history.best_val_loss, history.best_val_acc))
    print('Best model saved to: {}'.format('model'+ext+'.h5'))

    scores = p1b2.evaluate(y_pred, y_test)
    print('Evaluation on test data:', scores)

    submission = {'scores': scores,
                  'model': model.summary(),
                  'submitter': 'Developer Name' }

    # print('Submitting to leaderboard...')
    # leaderboard.submit(submission)


if __name__ == '__main__':
    main()
