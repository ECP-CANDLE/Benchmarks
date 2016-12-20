from __future__ import print_function

import numpy as np

from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

import p1b2

# (X_train, y_train), (X_test, y_test) = p1b2.load_data(n_cols=100)
(X_train, y_train), (X_test, y_test) = p1b2.load_data()

y_train = y_train.argmax(axis=1)
y_test = y_test.argmax(axis=1)

clf = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.05)

scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores)
print(np.mean(scores))
