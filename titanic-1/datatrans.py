#!/usr/bin/env Python
# -*- coding: utf-8 -*-

"""
Created on Sun Oct 25 12:44:49 2020

@author: sandyzikun
"""

import numpy as np
import pandas as pd

FLAG_DUMP = True

train = pd.read_csv("./train.csv").drop(["PassengerId"], axis=1)
# train = pd.read_csv("./train.csv").drop(columns=["PassengerId"])
test = pd.read_csv("./test.csv").drop(["PassengerId"], axis=1)
gsub = pd.read_csv("./gender_submission.csv")

train = train.replace({"male": 0, "female": 1})
train = train.replace({"S": 1, "C": 2, "Q": 3})

test = test.replace({"male": 0, "female": 1})
test = test.replace({"S": 1, "C": 2, "Q": 3})

import re
"""
{'Mr.': 757,
 'Mrs.': 197,
 'Miss.': 260,
 'Master.': 61,
 'Don.': 1,
 'Rev.': 8,
 'Dr.': 8,
 'Mme.': 1,
 'Ms.': 2,
 'Major.': 2,
 'Lady.': 1,
 'Sir.': 1,
 'Mlle.': 2,
 'Col.': 4,
 'Capt.': 1,
 'Countess.': 1,
 'Jonkheer.': 1,
 'Dona.': 1}
"""
PATTERN_COMPILED = re.compile(pattern="([A-Z][a-z]*\.)")
NAME_PREFIXES = {"Mr.": 1, "Mrs.": 2, "Miss.": 3, "Master.": 4, "Rev.": 5, "Dr.": 6}
prefixes_train, val_pref_train = [], []
for i in range(train.shape[0]):
    prefixes_train.append(PATTERN_COMPILED.findall(train.values[i, 2])[0])
    val_pref_train.append(NAME_PREFIXES[prefixes_train[-1]] if prefixes_train[-1] in NAME_PREFIXES else 0)
prefixes_test, val_pref_test = [], []
for i in range(test.shape[0]):
    prefixes_test.append(PATTERN_COMPILED.findall(test.values[i, 1])[0])
    val_pref_test.append(NAME_PREFIXES[prefixes_test[-1]] if prefixes_test[-1] in NAME_PREFIXES else 0)

X_Train = train.values[ : , 1 : ]
for i in range(X_Train.shape[0]):
    X_Train[i, 1] = val_pref_train[i]
    X_Train[i, -1] = int(X_Train[i, -1]) if (X_Train[i, -1] == X_Train[i, -1]) else 0
y_Train = train.values[ : , 0 ].astype("uint8")
X_Test = test.values
for i in range(X_Test.shape[0]):
    X_Test[i, 1] = val_pref_test[i]
    X_Test[i, -1] = int(X_Test[i, -1]) if (X_Test[i, -1] == X_Test[i, -1]) else 0
y_Test = gsub.values[ : , 1 ].astype("uint8")

datakeys = list(train.keys())

if FLAG_DUMP:
    import time
    """
    `X_Train`: NumPy Array, shape(891, 10), Input of training data;
    `y_Train`: NumPy Array, shape(891,), Output of training data;
    `X_Test`: NumPy Array, shape(418, 10), Input of testing data;
    `y_Test`: NumPy Array, shape(418,), Output of testing data;
    `X_Keys`: NumPy Array(List), Names of input variables;
    `y_Keys`: NumPy Array(List), Names of output variables;
    """
    np.savez("titanic.%s.npz" % time.time(),
             X_Train=X_Train, y_Train=y_Train,
             X_Test=X_Test, y_Test=y_Test,
             X_Keys=datakeys[ 1 : ], y_Keys=datakeys[ 0 : 1 ])