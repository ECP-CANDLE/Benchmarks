import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from time import time
from alibi.explainers import CounterFactual, CounterFactualProto
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False
print(tf.test.is_gpu_available())
import pickle
model_nt3 = tf.keras.models.load_model('../nt3.autosave.model')
with open('../nt3.autosave.data.pkl', 'rb') as pickle_file:
        X_train,X_test,Y_train,Y_test = pickle.load(pickle_file)

shape_cf = (1,) + X_train.shape[1:]
print(shape_cf)
target_proba = 0.9
tol = 0.1 # want counterfactuals with p(class)>0.90
target_class = 'other' # any class other than will do
max_iter = 1000
lam_init = 1e-1
max_lam_steps = 20
learning_rate_init = 0.1
feature_range = (0,1)
cf = CounterFactual(model_nt3, shape=shape_cf, target_proba=target_proba, tol=tol,
                                        target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                                        max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                                        feature_range=feature_range)
shape = X_train[0].shape[0]
results=[]
failed_inds = []
X = np.concatenate([X_train,X_test])

for i in np.arange(0,X.shape[0]):
    print(i)
    x_sample=X[i:i+1]
    print(x_sample.shape)
    start = time()
    try:
        explanation = cf.explain(x_sample)
        print('Counterfactual prediction: {}, {}'.format(explanation.cf['class'], explanation.cf['proba']))
        print("Actual prediction: {}".format(model_nt3.predict(x_sample)))
        results.append([i, explanation.cf['X'],explanation.cf['class'], explanation.cf['proba']])
        test = model_nt3.predict(explanation.cf['X'])
        print(test, explanation.cf['proba'], explanation.cf['class'])
    except:
        print("Failed cf generation")
        failed_inds.append(i)
    if i%100 == 0 and i is not 0:
        pickle.dump(results, open("cf_{}.pkl".format(i), "wb"))
        results = []
pickle.dump(failed_inds, open("cf_failed_inds.pkl", "wb"))
