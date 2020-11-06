#!/usr/bin/env Python
# -*- coding: utf-8 -*-

import numpy as np
import scipy, sympy
import pandas as pd

from matplotlib import pyplot as plt
plt.style.use("Solarize_Light2")

import sklearn
from sklearn import *

import time

data = np.load("titanic.final.npz", allow_pickle=True)

X_Train, y_Train = data["X_Train"].astype("float32"), data["y_Train"].astype("uint8")
X_Test, y_Test = data["X_Test"].astype("float32"), data["y_Test"].astype("uint8")
X_Keys, y_Keys = data["X_Keys"], data["y_Keys"]

class Constants(object):
    CROSS_VAL = sklearn.model_selection.KFold(n_splits=11, random_state=7)
    SCORING = "accuracy"

models, results = {
    "LSVC": sklearn.svm.LinearSVC(),
    #   "KNC": sklearn.neighbors.KNeighborsClassifier(),
    #   "DTC": sklearn.tree.DecisionTreeClassifier(),
    # ==Ensemble==
    "RFC": sklearn.ensemble.RandomForestClassifier(),
    #   "ETC": sklearn.ensemble.ExtraTreesClassifier(),
    #   "ABC": sklearn.ensemble.AdaBoostClassifier(),
    "GBC": sklearn.ensemble.GradientBoostingClassifier(),
    # ==Naive Bayes==
    "GNB": sklearn.naive_bayes.GaussianNB(),
    "BNB": sklearn.naive_bayes.BernoulliNB(),
    # ==SVM==
    "SVC": sklearn.svm.SVC(),
    "NuSVC": sklearn.svm.NuSVC(),
    }, {}

"""
for mod in models:
    results[mod] = sklearn.model_selection.cross_val_score(models[mod], X_Train, y_Train, cv=Constants.CROSS_VAL)
    print("%s: %.3f (%.3f)" % (mod, results[mod].mean(), results[mod].std()))
"""

"""
LSVC:
    #C: 0.01, 0.02, 0.05, 0.1
    C: 0.01

RFC:
    #n_estimators: 10, 20, 50, 100
    #n_estimators: 20, 25, 40, 50, 75
    #n_estimators: 30, 35, 40, 45, 50, 55
    n_estimators: 40

    #criterion: "gini", "entropy"
    criterion: "gini"

    #max_features: "sqrt", "auto", "log2", None
    #max_features: "sqrt", "log2"
    max_features: "sqrt"

    max_depth: None

    #min_samples_split: 2, 5, 10
    #min_samples_split: 5, 10, 15, 20
    #min_samples_split: 5, 8, 10, 12, 15
    #min_samples_split: 7, 8, 9, 10, 11
    min_samples_split: 8

GBC:
    #n_estimators: 10, 20, 50, 100
    n_estimators: 100

    #max_features: "sqrt", "auto", "log2", None
    #max_features: "sqrt", "log2"
    max_features: "sqrt"

    #max_depth: 5, 7, 9, 11, 13, 15
    max_depth: 11

    #min_samples_split: 200, 400, 600, 800, 1000
    min_samples_split: 200
"""

params = {}
gridcv = {}

for mod in params:
    gridcv[mod] = sklearn.model_selection.GridSearchCV(models[mod], params[mod])
    gridcv[mod].fit(X_Train, y_Train)
    print("%s: (%.3f) %s" % (mod, gridcv[mod].best_score_, gridcv[mod].best_params_))

fitted_models, fitted_results, y_Pred = {
    "LSVC": sklearn.svm.LinearSVC(C=0.01),
    "RFC": sklearn.ensemble.RandomForestClassifier(n_estimators=40, criterion="gini", max_features="sqrt", max_depth=None, min_samples_split=8),
    "GBC": sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, max_features="sqrt", max_depth=11, min_samples_split=200)
    }, {}, {}

for mod in fitted_models:
    fitted_results[mod] = sklearn.model_selection.cross_val_score(fitted_models[mod], X_Train, y_Train, cv=Constants.CROSS_VAL)
    fitted_models[mod].fit(X_Train, y_Train)
    y_Pred[mod] = fitted_models[mod].predict(X_Test)
    print("%s: %.3f (%.3f) acc=%.3f" % (mod, fitted_results[mod].mean(), fitted_results[mod].std(), sklearn.metrics.accuracy_score(y_Test, y_Pred[mod])))

fig_fitted = plt.figure()
fig_fitted.suptitle("Fitted Algorithm Comparision")
ax_fitted = fig_fitted.add_subplot(111)
ax_fitted.set_xticklabels(fitted_models.keys())
plt.boxplot([ fitted_results[mod] for mod in fitted_models ])
plt.savefig("fitted_algocmp.%s.jpeg" % time.time())
