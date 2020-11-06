#!/usr/bin/env Python
# -*- coding: utf-8 -*-

"""
Created on Sun Oct 25 17:32:57 2020

@author: sandyzikun
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use("Solarize_Light2")

FLAG_DUMP = False

data = np.load("titanic.npz", allow_pickle=True)

X_Train, y_Train = data["X_Train"], data["y_Train"]
X_Test, y_Test = data["X_Test"], data["y_Test"]
X_Keys, y_Keys = data["X_Keys"], data["y_Keys"]

cols = np.concatenate((X_Keys, y_Keys))
train = np.concatenate((X_Train, y_Train.reshape((y_Train.shape[0], 1))), axis=1)
test = np.concatenate((X_Test, y_Test.reshape((y_Test.shape[0], 1))), axis=1)
alldata = np.concatenate((train, test), axis=0)
train = pd.DataFrame(train, columns=cols)
test = pd.DataFrame(test, columns=cols)
alldata = pd.DataFrame(alldata, columns=cols)

"""
    Training data info:
    #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
    0   Pclass    891 non-null    object
    1   Name      891 non-null    object
    2   Sex       891 non-null    object
    3   Age       714 non-null    object
    4   SibSp     891 non-null    object
    5   Parch     891 non-null    object
    6   Ticket    891 non-null    object
    7   Fare      891 non-null    object
    8   Cabin     204 non-null    object
    9   Embarked  891 non-null    object
    10  Survived  891 non-null    object

    Testing data info:
    #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
    0   Pclass    418 non-null    object
    1   Name      418 non-null    object
    2   Sex       418 non-null    object
    3   Age       332 non-null    object
    4   SibSp     418 non-null    object
    5   Parch     418 non-null    object
    6   Ticket    418 non-null    object
    7   Fare      417 non-null    object
    8   Cabin     91 non-null     object
    9   Embarked  418 non-null    object
    10  Survived  418 non-null    object

    All data info:
    #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
    0   Pclass    1309 non-null   object
    1   Name      1309 non-null   object
    2   Sex       1309 non-null   object
    3   Age       1046 non-null   object
    4   SibSp     1309 non-null   object
    5   Parch     1309 non-null   object
    6   Ticket    1309 non-null   object
    7   Fare      1308 non-null   object
    8   Cabin     295 non-null    object
    9   Embarked  1309 non-null   object
    10  Survived  1309 non-null   object
"""

# Processing Cabin
INDEX_CABIN = 8
train.values[ : , INDEX_CABIN] = np.array([ (1 if train.values[i, INDEX_CABIN] == train.values[i, INDEX_CABIN] else 0) for i in range(train.shape[0]) ])
test.values[ : , INDEX_CABIN] = np.array([ (1 if test.values[i, INDEX_CABIN] == test.values[i, INDEX_CABIN] else 0) for i in range(test.shape[0]) ])
alldata.values[ : , INDEX_CABIN] = np.array([ (1 if alldata.values[i, INDEX_CABIN] == alldata.values[i, INDEX_CABIN] else 0) for i in range(alldata.shape[0]) ])

# Processing Fare
INDEX_FARE = 7
alldata.values[1043, INDEX_FARE] = test.values[152, INDEX_FARE] = 33.295

# Processing Ticket
INDEX_TICKET = 6
def numericalonly(x: str):
    for i in range(len(x)):
        if not x[i] in "1234567890":
            return False
    return True
train.values[ : , INDEX_TICKET] = np.array([ (0 if numericalonly(train.values[i, INDEX_TICKET]) else 1) for i in range(train.shape[0]) ])
test.values[ : , INDEX_TICKET] = np.array([ (0 if numericalonly(test.values[i, INDEX_TICKET]) else 1) for i in range(test.shape[0]) ])
alldata.values[ : , INDEX_TICKET] = np.array([ (0 if numericalonly(alldata.values[i, INDEX_TICKET]) else 1) for i in range(alldata.shape[0]) ])

# Regression of Age
from sklearn import model_selection as sk_model_selection, preprocessing as sk_preprocessing
from sklearn import linear_model as sk_linear_model, \
                    ensemble as sk_ensemble, \
                    naive_bayes as sk_naive_bayes, \
                    svm as sk_svm

class Constants_AGE(object):
    NUM_SPLITS = 11
    RANDOM_STATE = 7
    SCORING = "neg_mean_squared_error"
    INDEX_AGE = 3
    TOFILL_TRAIN = 891 - 714

X_train_age, y_train_age, X_test_age = [], [], []
for i in range(alldata.shape[0]):
    if alldata.values[i, Constants_AGE.INDEX_AGE] == alldata.values[i, Constants_AGE.INDEX_AGE]:
        X_train_age.append(np.concatenate([alldata.values[i, : Constants_AGE.INDEX_AGE], alldata.values[i, (Constants_AGE.INDEX_AGE + 1) :]]))
        y_train_age.append(alldata.values[i, Constants_AGE.INDEX_AGE])
    else:
        X_test_age.append(np.concatenate([alldata.values[i, : Constants_AGE.INDEX_AGE], alldata.values[i, (Constants_AGE.INDEX_AGE + 1) :]]))

X_train_age, y_train_age, X_test_age = np.array(X_train_age), np.array(y_train_age), np.array(X_test_age)

scaler_age = sk_preprocessing.StandardScaler()
X_train_age = scaler_age.fit_transform(X_train_age, y_train_age)
X_test_age = scaler_age.transform(X_test_age)

"""
models_age = {
    "LR": sk_linear_model.ElasticNet(),
    "RF": sk_ensemble.RandomForestRegressor(),
    "NB": sk_naive_bayes.GaussianNB(),
    "SVM": sk_svm.SVR()
    }
cvscores_age = {}

print(y_train_age[ : 13])

for mod_age in models_age:
    cvscores_age[mod_age] = sk_model_selection.cross_val_score(estimator = models_age[mod_age],
                                       X = X_train_age,
                                       y = y_train_age.astype("int"),
                                       cv = sk_model_selection.KFold(n_splits = Constants_AGE.NUM_SPLITS,
                                                                   random_state = Constants_AGE.RANDOM_STATE
                                                                   ),
                                       scoring = Constants_AGE.SCORING
                                       )
fig_age = plt.figure()
fig_age.suptitle("Algorithm Comparision")
ax_age = fig_age.add_subplot(111)
ax_age.set_xticklabels(models_age.keys())
plt.boxplot([ cvscores_age[mod_age] for mod_age in models_age ])
plt.savefig("algocmp_age.jpeg")
"""
param_grid_age_svm = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "poly", "rbf", "sigmoid"]
    }
grid_age_svm = sk_model_selection.GridSearchCV(estimator = sk_svm.SVR(),
                                               param_grid = param_grid_age_svm,
                                               scoring = Constants_AGE.SCORING,
                                               cv = sk_model_selection.KFold(n_splits = Constants_AGE.NUM_SPLITS,
                                                                   random_state = Constants_AGE.RANDOM_STATE
                                                                   )
                                               )
grid_result_age_svm = grid_age_svm.fit(X_train_age, y_train_age)


machine_age = sk_svm.SVR(C=10, kernel="rbf")
machine_age.fit(X_train_age, y_train_age)
y_pred_age = machine_age.predict(X_test_age)

temp_index = 0
for i in range(train.shape[0]):
    if train.values[i, Constants_AGE.INDEX_AGE] != train.values[i, Constants_AGE.INDEX_AGE]:
        alldata.values[i, Constants_AGE.INDEX_AGE] = train.values[i, Constants_AGE.INDEX_AGE] = y_pred_age[temp_index]
        temp_index += 1
for i in range(test.shape[0]):
    if test.values[i, Constants_AGE.INDEX_AGE] != test.values[i, Constants_AGE.INDEX_AGE]:
        alldata.values[i + train.shape[0], Constants_AGE.INDEX_AGE] = test.values[i, Constants_AGE.INDEX_AGE] = y_pred_age[temp_index]
        temp_index += 1

X_Train, y_Train = train.values[ : , : (-1)], train.values[ : , (-1)]
X_Test, y_Test = test.values[ : , : (-1)], test.values[ : , (-1)]
datakeys = list(train.keys())

if FLAG_DUMP:
    import time
    np.savez("titanic.age.%s.npz" % time.time(), predicted_age=y_pred_age)
    np.savez("titanic.final.%s.npz" % time.time(),
             X_Train=X_Train, y_Train=y_Train,
             X_Test=X_Test, y_Test=y_Test,
             X_Keys=datakeys[ 1 : ], y_Keys=datakeys[ 0 : 1 ]
             )