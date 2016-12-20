from __future__ import print_function

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.regularizers import l2, activity_l2

import p1b2

# (X_train, y_train), (X_test, y_test) = p1b2.load_data(n_cols=10000)
(X_train, y_train), (X_test, y_test) = p1b2.load_data()

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

model = Sequential()

penalty = 0.01
act = 'sigmoid'

model.add(Dense(1024, input_dim=input_dim, activation=act, W_regularizer=l2(penalty), activity_regularizer=activity_l2(penalty)))
# model.add(Dropout(0.2))
model.add(Dense(512, activation=act, W_regularizer=l2(penalty), activity_regularizer=activity_l2(penalty)))
# model.add(Dropout(0.2))
model.add(Dense(256, activation=act, W_regularizer=l2(penalty), activity_regularizer=activity_l2(penalty)))
model.add(Dense(output_dim, activation=act))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train,
          batch_size = 64,
          nb_epoch = 500,
          validation_split = 0.2)

y_pred = model.predict(X_test)

scores = p1b2.evaluate(y_pred, y_test)
print(scores)

submission = {'scores': scores,
              'model': model.summary(),
              'submitter': 'Developer Name' }

print('Submitting to leaderboard...')
# leaderboard.submit(submission)
