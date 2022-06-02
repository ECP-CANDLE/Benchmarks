from __future__ import print_function
from __future__ import division

import os
import pickle
import re
import warnings
import numpy as np

from sklearn import metrics
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBRegressor
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor

warnings.filterwarnings(action="ignore", category=DeprecationWarning)


def get_model(model_or_name, threads=-1, classify=False, seed=0):
    regression_models = {
        'xgboost': (XGBRegressor(max_depth=6, n_jobs=threads, random_state=seed), 'XGBRegressor'),
        'lightgbm': (LGBMRegressor(n_jobs=threads, random_state=seed, verbose=-1), 'LGBMRegressor'),
        'randomforest': (RandomForestRegressor(n_estimators=100, n_jobs=threads), 'RandomForestRegressor'),
        'adaboost': (AdaBoostRegressor(), 'AdaBoostRegressor'),
        'linear': (LinearRegression(), 'LinearRegression'),
        'elasticnet': (ElasticNetCV(positive=True), 'ElasticNetCV'),
        'lasso': (LassoCV(positive=True), 'LassoCV'),
        'ridge': (Ridge(), 'Ridge'),

        'xgb.1k': (XGBRegressor(max_depth=6, n_estimators=1000, n_jobs=threads, random_state=seed), 'XGBRegressor.1K'),
        'xgb.10k': (XGBRegressor(max_depth=6, n_estimators=10000, n_jobs=threads, random_state=seed), 'XGBRegressor.10K'),
        'lgbm.1k': (LGBMRegressor(n_estimators=1000, n_jobs=threads, random_state=seed, verbose=-1), 'LGBMRegressor.1K'),
        'lgbm.10k': (LGBMRegressor(n_estimators=10000, n_jobs=threads, random_state=seed, verbose=-1), 'LGBMRegressor.10K'),
        'rf.1k': (RandomForestRegressor(n_estimators=1000, n_jobs=threads), 'RandomForestRegressor.1K'),
        'rf.10k': (RandomForestRegressor(n_estimators=10000, n_jobs=threads), 'RandomForestRegressor.10K')
    }

    classification_models = {
        'xgboost': (XGBClassifier(max_depth=6, n_jobs=threads, random_state=seed), 'XGBClassifier'),
        'lightgbm': (LGBMClassifier(n_jobs=threads, random_state=seed, verbose=-1), 'LGBMClassifier'),
        'randomforest': (RandomForestClassifier(n_estimators=100, n_jobs=threads), 'RandomForestClassifier'),
        'adaboost': (AdaBoostClassifier(), 'AdaBoostClassifier'),
        'logistic': (LogisticRegression(), 'LogisticRegression'),
        'gaussian': (GaussianProcessClassifier(), 'GaussianProcessClassifier'),
        'knn': (KNeighborsClassifier(), 'KNeighborsClassifier'),
        'bayes': (GaussianNB(), 'GaussianNB'),
        'svm': (SVC(), 'SVC'),

        'xgb.1k': (XGBClassifier(max_depth=6, n_estimators=1000, n_jobs=threads, random_state=seed), 'XGBClassifier.1K'),
        'xgb.10k': (XGBClassifier(max_depth=6, n_estimators=10000, n_jobs=threads, random_state=seed), 'XGBClassifier.10K'),
        'lgbm.1k': (LGBMClassifier(n_estimators=1000, n_jobs=threads, random_state=seed, verbose=-1), 'LGBMClassifier.1K'),
        'lgbm.10k': (LGBMClassifier(n_estimators=1000, n_jobs=threads, random_state=seed, verbose=-1), 'LGBMClassifier.10K'),
        'rf.1k': (RandomForestClassifier(n_estimators=1000, n_jobs=threads), 'RandomForestClassifier.1K'),
        'rf.10k': (RandomForestClassifier(n_estimators=10000, n_jobs=threads), 'RandomForestClassifier.10K')
    }

    if isinstance(model_or_name, str):
        if classify:
            model_and_name = classification_models.get(model_or_name.lower())
        else:
            model_and_name = regression_models.get(model_or_name.lower())
        if not model_and_name:
            raise Exception("unrecognized model: '{}'".format(model_or_name))
        else:
            model, name = model_and_name
    else:
        model = model_or_name
        name = re.search("\\w+", str(model)).group(0)

    return model, name


def score_format(metric, score, signed=False, eol=''):
    if signed:
        return '{:<25} = {:+.5f}'.format(metric, score) + eol
    else:
        return '{:<25} =  {:.5f}'.format(metric, score) + eol


def top_important_features(model, feature_names, n_top=1000):
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
    else:
        if hasattr(model, "coef_"):
            fi = model.coef_
        else:
            return
    features = [(f, n) for f, n in zip(fi, feature_names)]
    top = sorted(features, key=lambda f: abs(f[0]), reverse=True)[:n_top]
    return top


def sprint_features(top_features, n_top=1000):
    str = ''
    for i, feature in enumerate(top_features):
        if i >= n_top:
            break
        str += '{:9.5f}\t{}\n'.format(feature[0], feature[1])
    return str


def discretize(y, bins=5, cutoffs=None, min_count=0, verbose=False, return_bins=False):
    thresholds = cutoffs
    if thresholds is None:
        if verbose:
            print('Creating {} balanced categories...'.format(bins))
        percentiles = [100 / bins * (i + 1) for i in range(bins - 1)]
        thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    good_bins = None
    if verbose:
        bc = np.bincount(classes)
        good_bins = len(bc)
        min_y = np.min(y)
        max_y = np.max(y)
        print('Category cutoffs:', ['{:.3g}'.format(t) for t in thresholds])
        print('Bin counts:')
        for i, count in enumerate(bc):
            lower = min_y if i == 0 else thresholds[i - 1]
            upper = max_y if i == len(bc) - 1 else thresholds[i]
            removed = ''
            if count < min_count:
                removed = ' .. removed (<{})'.format(min_count)
                good_bins -= 1
            print('  Class {}: {:7d} ({:.4f}) - between {:+.2f} and {:+.2f}{}'.
                  format(i, count, count / len(y), lower, upper, removed))
        # print('  Total: {:9d}'.format(len(y)))
    if return_bins:
        return classes, thresholds, good_bins
    else:
        return classes


def categorize_dataframe(df, ycol='0', bins=5, cutoffs=None, verbose=False):
    if ycol.isdigit():
        ycol = df.columns[int(ycol)]
    y = df.loc[:, ycol].values
    classes = discretize(y, bins, cutoffs, verbose)
    df.iloc[:, 0] = classes
    return df


def make_group_from_columns(df, groupcols):
    return df[groupcols].astype(str).sum(axis=1).values


def summarize(df, ycol='0', classify=False, bins=0, cutoffs=None, min_count=0):
    if ycol.isdigit():
        ycol = df.columns[int(ycol)]
    y = df.loc[:, ycol].values
    print('Target column: {}'.format(ycol))
    print('  count = {}, uniq = {}, mean = {:.3g}, std = {:.3g}'.format(len(y), len(np.unique(y)), np.mean(y), np.std(y)))
    print('  min = {:.3g}, q1 = {:.3g}, median = {:.3g}, q3 = {:.3g}, max = {:.3g}'.format(np.min(y), np.percentile(y, 25), np.median(y), np.percentile(y, 75), np.max(y)))
    # y_discrete, thresholds, _ = discretize(y, bins=4)
    # print('Quartiles of y:', ['{:.2g}'.format(t) for t in thresholds])
    good_bins = None
    if classify:
        if cutoffs is not None or bins >= 2:
            _, _, good_bins = discretize(y, bins=bins, cutoffs=cutoffs, min_count=min_count, verbose=True)
        else:
            if df[ycol].dtype in [np.dtype('float64'), np.dtype('float32')]:
                warnings.warn('Warning: classification target is float; consider using --bins or --cutoffs')
            good_bins = len(np.unique(y))
    print()
    return good_bins


def split_data(df, ycol='0', classify=False, cv=5, bins=0, cutoffs=None, groupcols=None, ignore_categoricals=False, verbose=True):
    if groupcols is not None:
        groups = make_group_from_columns(df, groupcols)

    cat_cols = df.select_dtypes(['object']).columns
    if ignore_categoricals:
        df[cat_cols] = 0
    else:
        df[cat_cols] = df[cat_cols].apply(lambda x: x.astype('category').cat.codes)

    if ycol.isdigit():
        ycol = df.columns[int(ycol)]

    y = df.loc[:, ycol].values
    x = df.drop(ycol, axis=1).values
    features = df.drop(ycol, axis=1).columns.tolist()

    if verbose:
        print('Target column: {}'.format(ycol))
        print('  count = {}, uniq = {}, mean = {:.3g}, std = {:.3g}'.format(len(y), len(np.unique(y)), np.mean(y), np.std(y)))
        print('  min = {:.3g}, q1 = {:.3g}, median = {:.3g}, q3 = {:.3g}, max = {:.3g}'.format(np.min(y), np.percentile(y, 25), np.median(y), np.percentile(y, 75), np.max(y)))

    if not classify:
        y_even = discretize(y, bins=5, verbose=False)
    elif bins >= 2:
        y = discretize(y, bins=bins, min_count=cv, verbose=verbose)
    elif cutoffs:
        y = discretize(y, cutoffs=cutoffs, min_count=cv, verbose=verbose)
    elif df[ycol].dtype in [np.dtype('float64'), np.dtype('float32')]:
        warnings.warn('Warning: classification target is float; consider using --bins or --cutoffs')
        y = y.astype(int)

    if classify:
        mask = np.ones(len(y), dtype=bool)
        unique, counts = np.unique(y, return_counts=True)
        for v, c in zip(unique, counts):
            if c < cv:
                mask[y == v] = False
        x = x[mask]
        y = y[mask]
        removed = len(mask) - np.sum(mask)
        if removed and verbose:
            print('Removed {} rows in small classes: count < {}'.format(removed, cv))

    if groupcols is None:
        if classify:
            y_even = y
        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        splits = skf.split(x, y_even)
    else:
        if classify:
            groups = groups[mask]
        gkf = GroupKFold(n_splits=cv)
        splits = gkf.split(x, y, groups)

    if verbose:
        print()

    return x, y, list(splits), features


def verify_path(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def infer(model, df, ycol='0', ignore_categoricals=False, classify=False, prefix=''):
    if type(model) == str:
        with open(model, 'rb') as f:
            model = pickle.load(f)

    cat_cols = df.select_dtypes(['object']).columns
    if ignore_categoricals:
        df[cat_cols] = 0
    else:
        df[cat_cols] = df[cat_cols].apply(lambda x: x.astype('category').cat.codes)

    if ycol.isdigit():
        ycol = df.columns[int(ycol)]

    y = df.loc[:, ycol].values
    x = df.drop(ycol, axis=1).values
    y_pred = model.predict(x)

    if classify:
        metric_names = 'accuracy_score matthews_corrcoef f1_score precision_score recall_score log_loss'.split()
    else:
        metric_names = 'r2_score explained_variance_score mean_absolute_error mean_squared_error'.split()

    scores = {}
    print('Average test metrics:')
    scores_fname = "{}.test.scores".format(prefix)
    with open(scores_fname, "w") as scores_file:
        for m in metric_names:
            try:
                s = getattr(metrics, m)(y, y_pred)
                scores[m] = s
                print(' ', score_format(m, s))
                scores_file.write(score_format(m, s, eol='\n'))
            except Exception:
                pass
        scores_file.write('\nModel:\n{}\n\n'.format(model))

    print()
    return scores


def train(model, x, y, features=None, classify=False, threads=-1, prefix='', name=None, save=False):
    verify_path(prefix)
    model, model_name = get_model(model, threads, classify=classify)
    model.fit(x, y)
    name = name or model_name
    if save:
        model_desc_fname = "{}.{}.description".format(prefix, name)
        with open(model_desc_fname, "w") as f:
            f.write('{}\n'.format(model))
        if features:
            top_features = top_important_features(model, features)
            if top_features is not None:
                fea_fname = "{}.{}.features".format(prefix, name)
                with open(fea_fname, "w") as fea_file:
                    fea_file.write(sprint_features(top_features))
        model_fname = "{}.{}.model.pkl".format(prefix, name)
        with open(model_fname, 'wb') as f:
            pickle.dump(model, f)
        return model_fname


def classify(model, x, y, splits, features, threads=-1, prefix='', seed=0, class_weight=None):
    verify_path(prefix)
    model, name = get_model(model, threads, classify=True, seed=seed)

    train_scores, test_scores = [], []
    tests, preds = None, None
    probas = None
    best_model = None
    best_score = -np.Inf

    print('>', name)
    print('Cross validation:')
    for i, (train_index, test_index) in enumerate(splits):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.set_params(class_weight=class_weight)
        model.fit(x_train, y_train)
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("  fold {}/{}: score = {:.3f}  (train = {:.3f})".format(i + 1, len(splits), test_score, train_score))
        if test_score > best_score:
            best_model = model
            best_score = test_score
        y_pred = model.predict(x_test)
        preds = np.concatenate((preds, y_pred)) if preds is not None else y_pred
        tests = np.concatenate((tests, y_test)) if tests is not None else y_test
        if hasattr(model, "predict_proba"):
            probas_ = model.predict_proba(x_test)
            probas = np.concatenate((probas, probas_)) if probas is not None else probas_

    uniques, counts = np.unique(tests, return_counts=True)
    average = 'binary' if len(uniques) <= 2 else 'weighted'

    roc_auc_score = None
    if probas is not None:
        fpr, tpr, thresholds = metrics.roc_curve(tests, probas[:, 1], pos_label=0)
        roc_auc_score = 1 - metrics.auc(fpr, tpr)
        roc_fname = "{}.{}.ROC".format(prefix, name)
        if roc_auc_score:
            with open(roc_fname, "w") as roc_file:
                roc_file.write('\t'.join(['Threshold', 'FPR', 'TPR']) + '\n')
                for ent in zip(thresholds, fpr, tpr):
                    roc_file.write('\t'.join("{0:.5f}".format(x) for x in list(ent)) + '\n')

    print('Average validation metrics:')
    naive_accuracy = max(counts) / len(tests)
    accuracy = np.sum(preds == tests) / len(tests)
    accuracy_gain = accuracy - naive_accuracy
    print(' ', score_format('accuracy_gain', accuracy_gain, signed=True))
    scores_fname = "{}.{}.scores".format(prefix, name)
    metric_names = 'accuracy_score matthews_corrcoef f1_score precision_score recall_score log_loss'.split()
    with open(scores_fname, "w") as scores_file:
        scores_file.write(score_format('accuracy_gain', accuracy_gain, signed=True, eol='\n'))
        for m in metric_names:
            s = None
            try:
                s = getattr(metrics, m)(tests, preds, average=average)
            except Exception:
                try:
                    s = getattr(metrics, m)(tests, preds)
                except Exception:
                    pass
            if s:
                print(' ', score_format(m, s))
                scores_file.write(score_format(m, s, eol='\n'))
        if roc_auc_score:
            print(' ', score_format('roc_auc_score', roc_auc_score))
            scores_file.write(score_format('roc_auc_score', roc_auc_score, eol='\n'))
        scores_file.write('\nModel:\n{}\n\n'.format(model))

    print()
    top_features = top_important_features(best_model, features)
    if top_features is not None:
        fea_fname = "{}.{}.features".format(prefix, name)
        with open(fea_fname, "w") as fea_file:
            fea_file.write(sprint_features(top_features))

    score = metrics.f1_score(tests, preds, average=average)
    return score


def regress(model, x, y, splits, features, threads=-1, prefix='', seed=0):
    verify_path(prefix)
    model, name = get_model(model, threads, seed=seed)

    train_scores, test_scores = [], []
    tests, preds = None, None
    best_model = None
    best_score = -np.Inf

    print('>', name)
    print('Cross validation:')
    for i, (train_index, test_index) in enumerate(splits):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("  fold {}/{}: score = {:.3f}  (train = {:.3f})".format(i + 1, len(splits), test_score, train_score))
        if test_score > best_score:
            best_model = model
            best_score = test_score
        y_pred = model.predict(x_test)
        preds = np.concatenate((preds, y_pred)) if preds is not None else y_pred
        tests = np.concatenate((tests, y_test)) if tests is not None else y_test

    print('Average validation metrics:')
    scores_fname = "{}.{}.scores".format(prefix, name)
    metric_names = 'r2_score explained_variance_score mean_absolute_error mean_squared_error'.split()
    with open(scores_fname, "w") as scores_file:
        for m in metric_names:
            try:
                s = getattr(metrics, m)(tests, preds)
                print(' ', score_format(m, s))
                scores_file.write(score_format(m, s, eol='\n'))
            except Exception:
                pass
        scores_file.write('\nModel:\n{}\n\n'.format(model))

    print()
    top_features = top_important_features(best_model, features)
    if top_features is not None:
        fea_fname = "{}.{}.features".format(prefix, name)
        with open(fea_fname, "w") as fea_file:
            fea_file.write(sprint_features(top_features))

    score = metrics.r2_score(tests, preds)
    return score
