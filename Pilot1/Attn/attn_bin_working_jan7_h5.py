import itertools
import pandas as pd
import numpy as np
import os
import sys
import gzip
import argparse
import sklearn

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import tensorflow as tf

import keras as ke
from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model

from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import recall_score, auc, roc_curve, f1_score, precision_recall_curve


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)

psr = argparse.ArgumentParser(description='input agg csv file')
psr.add_argument('--in', default='in_file')
psr.add_argument('--ep', type=int, default=400)
psr.add_argument('--save_dir', default=".")
args = vars(psr.parse_args())
if not args['save_dir'].endswith('/'):
    args['save_dir'] = args['save_dir'] + '/'
print(args)

EPOCH = args['ep']
BATCH = 32
nb_classes = 2

data_path = args['in']

# df_toss = (pd.read_csv(data_path,nrows=1).values)

# print('df_toss:', df_toss.shape)

# PL = df_toss.size
# PS = PL - 1

# print('PL=',PL)

# PL     = 6213   # 38 + 60483
# PS     = 6212   # 60483
DR = 0.2      # Dropout rate


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def tf_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def auroc(y_true, y_pred):
    score = tf.py_func(lambda y_true, y_pred: roc_auc_score(y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                       [y_true, y_pred],
                       'float32',
                       stateful=False,
                       name='sklearnAUC')
    return score


def load_data():

    # start change #
    if args['in'].endswith('h5') or args['in'].endswith('hdf5'):
        print('processing h5 in file {}'.format(args['in']))

        df_x_train_0 = pd.read_hdf(args['in'], 'x_train_0').astype(np.float32)
        df_x_train_1 = pd.read_hdf(args['in'], 'x_train_1').astype(np.float32)
        X_train = pd.concat([df_x_train_0, df_x_train_1], axis=1, sort=False)
        del df_x_train_0, df_x_train_1

        df_x_test_0 = pd.read_hdf(args['in'], 'x_test_0').astype(np.float32)
        df_x_test_1 = pd.read_hdf(args['in'], 'x_test_1').astype(np.float32)
        X_test = pd.concat([df_x_test_0, df_x_test_1], axis=1, sort=False)
        del df_x_test_0, df_x_test_1

        df_x_val_0 = pd.read_hdf(args['in'], 'x_val_0').astype(np.float32)
        df_x_val_1 = pd.read_hdf(args['in'], 'x_val_1').astype(np.float32)
        X_val = pd.concat([df_x_val_0, df_x_val_1], axis=1, sort=False)
        del df_x_val_0, df_x_val_1

        Y_train = pd.read_hdf(args['in'], 'y_train')
        Y_test = pd.read_hdf(args['in'], 'y_test')
        Y_val = pd.read_hdf(args['in'], 'y_val')

        # assumes AUC is in the third column at index 2
        # df_y = df['AUC'].astype('int')
        # df_x = df.iloc[:,3:].astype(np.float32)

        # assumes dataframe has already been scaled
        # scaler = StandardScaler()
        # df_x = scaler.fit_transform(df_x)

    else:
        print('expecting in file file suffix h5')
        sys.exit()

    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


X_train, _Y_train, X_val, _Y_val, X_test, _Y_test = load_data()
# move this inside the load_data function
Y_train = _Y_train['AUC']
Y_test = _Y_test['AUC']
Y_val = _Y_val['AUC']

Y_train_neg, Y_train_pos = np.bincount(Y_train)
Y_test_neg, Y_test_pos = np.bincount(Y_test)
Y_val_neg, Y_val_pos = np.bincount(Y_val)

Y_train_total = Y_train_neg + Y_train_pos
Y_test_total = Y_test_neg + Y_test_pos
Y_val_total = Y_val_neg + Y_val_pos

total = Y_train_total + Y_test_total + Y_val_total
neg = Y_train_neg + Y_test_neg + Y_val_neg
pos = Y_train_pos + Y_test_pos + Y_val_pos

print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
Y_val = np_utils.to_categorical(Y_val, nb_classes)

# ----------------------- from stack overflow

y_integers = np.argmax(Y_train, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

PS = X_train.shape[1]
inputs = Input(shape=(PS,))

x = Dense(1000, activation='relu')(inputs)
x = BatchNormalization()(x)

a = Dense(1000, activation='relu')(x)
a = BatchNormalization()(a)

b = Dense(1000, activation='softmax')(x)
x = ke.layers.multiply([a, b])

x = Dense(500, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(DR)(x)

x = Dense(250, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(DR)(x)

x = Dense(125, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(DR)(x)

x = Dense(60, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(DR)(x)

x = Dense(30, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(DR)(x)

outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

# parallel_model = multi_gpu_model(model, gpus=4)
# parallel_model.compile(loss='mean_squared_error',
#                        optimizer=SGD(lr=0.0001, momentum=0.9),
#                        metrics=['mae', r2])

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.00001, momentum=0.9),
              # optimizer=Adam(lr=0.00001),
              # optimizer=RMSprop(lr=0.0001),
              # optimizer=Adadelta(),
              metrics=['acc', tf_auc])

# set up a bunch of callbacks to do work during model training

checkpointer = ModelCheckpoint(filepath=args['save_dir'] + 'Agg_attn_bin.autosave.model.h5', verbose=1, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger(args['save_dir'] + 'Agg_attn_bin.training.log')
reduce_lr = ReduceLROnPlateau(monitor='val_tf_auc', factor=0.20, patience=40, verbose=1, mode='auto', min_delta=0.0001, cooldown=3, min_lr=0.000000001)
early_stop = EarlyStopping(monitor='val_tf_auc', patience=200, verbose=1, mode='auto')

# history = parallel_model.fit(X_train, Y_train,
history = model.fit(X_train, Y_train, class_weight=d_class_weights,
                    batch_size=BATCH,
                    epochs=EPOCH,
                    verbose=1,
                    validation_data=(X_val, Y_val),
                    callbacks=[checkpointer, csv_logger, reduce_lr, early_stop])

score = model.evaluate(X_test, Y_test, verbose=0)

Y_predict = model.predict(X_test)

threshold = 0.5

Y_pred_int = (Y_predict[:, 0] < threshold).astype(np.int)
Y_test_int = (Y_test[:, 0] < threshold).astype(np.int)

print('creating table of predictions')
f = open(args['save_dir'] + 'Agg_attn_bin.predictions.tsv', 'w')
for index, row in _Y_test.iterrows():
    if row['AUC'] == 1:
        if Y_pred_int[index] == 1:
            call = 'TP'
        else:
            call = 'FN'
    if row['AUC'] == 0:
        if Y_pred_int[index] == 0:
            call = 'TN'
        else:
            call = 'FP'
    # 1 TN 0 0.6323 NCI60.786-0 NSC.256439 NSC.102816
    print(index, "\t", call, "\t", Y_pred_int[index], "\t", row['AUC'], "\t", row['Sample'], "\t", row['Drug1'], file=f)
f.close()

# print(Y_test[:,0])
# print(Y_predict[:,0])

false_pos_rate, true_pos_rate, thresholds = roc_curve(Y_test[:, 0], Y_predict[:, 0])

# print(thresholds)

roc_auc = auc(false_pos_rate, true_pos_rate)

auc_keras = roc_auc
fpr_keras = false_pos_rate
tpr_keras = true_pos_rate

print('creating figure 1 at ', args['save_dir'] + 'Agg_attn_bin.auroc.pdf')
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label="No Skill")
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')

plt.savefig(args['save_dir'] + 'Agg_attn_bin.auroc.pdf', bbox_inches='tight')
plt.close()


# Zoom in view of the upper left corner.
print('creating figure 2 at ', args['save_dir'] + 'Agg_attn_bin.auroc2.pdf')
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--', label="No Skill")
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')

plt.savefig(args['save_dir'] + 'Agg_attn_bin.auroc2.pdf', bbox_inches='tight')
plt.close()


f1 = f1_score(Y_test_int, Y_pred_int)

precision, recall, thresholds = precision_recall_curve(Y_test[:, 0], Y_predict[:, 0])
pr_auc = auc(recall, precision)

pr_keras = pr_auc
precision_keras = precision
recall_keras = recall

print
print

print('f1=%.3f auroc=%.3f aucpr=%.3f' % (f1, auc_keras, pr_keras))
print('creating figure 3 at ', args['save_dir'] + 'Agg_attn_bin.aurpr.pdf')
plt.figure(1)
no_skill = len(Y_test_int[Y_test_int == 1]) / len(Y_test_int)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(recall_keras, precision_keras, label='PR Keras (area = {:.3f})'.format(pr_keras))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve')
plt.legend(loc='best')

plt.savefig(args['save_dir'] + 'Agg_attn_bin.aurpr.pdf', bbox_inches='tight')

plt.close()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


class_names = ["Non-Response", "Response"]

# Compute confusion matrix
cnf_matrix = sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
print('creating figure 4 at ', args['save_dir'] + 'Agg_attn_bin.confusion_without_norm.pdf')
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig(args['save_dir'] + 'Agg_attn_bin.confusion_without_norm.pdf', bbox_inches='tight')

plt.close()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


class_names = ["Non-Response", "Response"]

# Compute confusion matrix
cnf_matrix = sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig(args['save_dir'] + 'Agg_attn_bin.confusion_without_norm.pdf', bbox_inches='tight')

plt.close()

# Plot normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig(args['save_dir'] + 'Agg_attn_bin.confusion_with_norm.pdf', bbox_inches='tight')

plt.close()


print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


print(sklearn.metrics.roc_auc_score(Y_test_int, Y_pred_int))

print(sklearn.metrics.balanced_accuracy_score(Y_test_int, Y_pred_int))

print(sklearn.metrics.classification_report(Y_test_int, Y_pred_int))

print(sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int))

print("score")
print(score)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig(args['save_dir'] + 'Agg_attn_bin.accuracy.png', bbox_inches='tight')
plt.savefig(args['save_dir'] + 'Agg_attn_bin.accuracy.pdf', bbox_inches='tight')

plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig(args['save_dir'] + 'Agg_attn_bin.loss.png', bbox_inches='tight')
plt.savefig(args['save_dir'] + 'Agg_attn_bin.loss.pdf', bbox_inches='tight')


print('Test val_loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open(args['save_dir'] + "Agg_attn_bin.model.json", "w") as json_file:
    json_file.write(model_json)

# serialize model to YAML
model_yaml = model.to_yaml()
with open(args['save_dir'] + "Agg_attn_bin.model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)


# serialize weights to HDF5
model.save_weights(args['save_dir'] + "Agg_attn_bin.model.h5")
print("Saved model to disk")

# load json and create model
json_file = open(args['save_dir'] + 'Agg_attn_bin.model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_json = model_from_json(loaded_model_json)


# load yaml and create model
yaml_file = open(args['save_dir'] + 'Agg_attn_bin.model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model_yaml = model_from_yaml(loaded_model_yaml)


# load weights into new model
loaded_model_json.load_weights(args['save_dir'] + "Agg_attn_bin.model.h5")
print("Loaded json model from disk")

# evaluate json loaded model on test data
loaded_model_json.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

print('json Validation loss:', score_json[0])
print('json Validation accuracy:', score_json[1])

print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1] * 100))


# load weights into new model
loaded_model_yaml.load_weights(args['save_dir'] + "Agg_attn_bin.model.h5")
print("Loaded yaml model from disk")

# evaluate loaded model on test data
loaded_model_yaml.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

print('yaml Validation loss:', score_yaml[0])
print('yaml Validation accuracy:', score_yaml[1])

print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1] * 100))

# predict using loaded yaml model on test and training data

predict_yaml_train = loaded_model_yaml.predict(X_train)

predict_yaml_test = loaded_model_yaml.predict(X_test)


print('Yaml_train_shape:', predict_yaml_train.shape)
print('Yaml_test_shape:', predict_yaml_test.shape)


predict_yaml_train_classes = np.argmax(predict_yaml_train, axis=1)
predict_yaml_test_classes = np.argmax(predict_yaml_test, axis=1)

np.savetxt(args['save_dir'] + "Agg_attn_bin_predict_yaml_train.csv", predict_yaml_train, delimiter=",", fmt="%.3f")
np.savetxt(args['save_dir'] + "Agg_attn_bin_predict_yaml_test.csv", predict_yaml_test, delimiter=",", fmt="%.3f")

np.savetxt(args['save_dir'] + "Agg_attn_bin_predict_yaml_train_classes.csv", predict_yaml_train_classes, delimiter=",", fmt="%d")
np.savetxt(args['save_dir'] + "Agg_attn_bin_predict_yaml_test_classes.csv", predict_yaml_test_classes, delimiter=",", fmt="%d")
