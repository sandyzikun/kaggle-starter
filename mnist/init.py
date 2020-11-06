#!/usr/bin/env Python
# -*- coding: utf-8 -*-

"""
Created on Wed Nov  4 21:56:30 2020

@author: sandyzikun
"""

import numpy as np
import scipy, sympy
import pandas as pd

import sklearn
from sklearn import *

import tensorflow as tf
import keras, keras.backend as T

import matplotlib.pyplot as plt
plt.style.use("solarized-light")

with np.load("./k_mnist.mat32.npz", allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"]
    x_test = f["x_test"]
    f.close()
(x_keras, y_keras) = keras.datasets.mnist.load_data()[1]

def mat28expand32(input_tensor):
    assert input_tensor.shape == (28, 28)
    output_tensor = np.zeros((32, 32))
    output_tensor[ 2 : (-2) , 2 : (-2) ] = input_tensor
    return output_tensor
x_keras = np.array([ mat28expand32(x_keras[idx]) for idx in range(len(x_keras)) ])

class Constants(object):
    NUM_SPLITS = 21
    NUM_CHANNELS = 1
    IMG_SHAPE = (32, 32)
    INPUT_SHAPE = ((NUM_CHANNELS,) + IMG_SHAPE) if T.image_data_format() == "channels_first" else \
                  (IMG_SHAPE + (NUM_CHANNELS,))
    NUM_CLASSES = 10
    NUM_EPOCHES = 12
    BATCH_SIZE = 128
    LEARNING_RATE = 1.
    DROPPING_RATE = .2

x_keras = x_keras.reshape((x_keras.shape[0],) + Constants.INPUT_SHAPE).astype("float32") / 255
x_train = x_train.reshape((x_train.shape[0],) + Constants.INPUT_SHAPE).astype("float32") / 255
x_test = x_test.reshape((x_test.shape[0],) + Constants.INPUT_SHAPE).astype("float32") / 255
y_keras = keras.utils.np_utils.to_categorical(y_keras, Constants.NUM_CLASSES)
y_train = keras.utils.np_utils.to_categorical(y_train, Constants.NUM_CLASSES)


idx_fold, history, evaluation = 0, [], []
for idx_tr, idx_val in sklearn.model_selection.KFold(Constants.NUM_SPLITS).split(x_train):
    print("=====================================================================================")
    print("======== Validating on Fold[%2d] (total: %d), while training on the others... ========" % (idx_fold, Constants.NUM_SPLITS))
    print("=====================================================================================")
    x_tr, x_val = x_train[idx_tr], x_train[idx_val]
    y_tr, y_val = y_train[idx_tr], y_train[idx_val]
    lenet5 = keras.models.Sequential(layers=[
        keras.layers.Conv2D(
            filters=6, kernel_size=5,
            activation = "relu",
            input_shape = Constants.INPUT_SHAPE,
            name = "C1"
            ),
        keras.layers.MaxPooling2D(pool_size=2, name="S2"),
        keras.layers.Dropout(rate=Constants.DROPPING_RATE, name="Dropout2_3"),
        keras.layers.Conv2D(
            filters=16, kernel_size=5,
            activation = "relu",
            name = "C3"
            ),
        keras.layers.MaxPooling2D(pool_size=2, name="S4"),
        keras.layers.Conv2D(
            filters=120, kernel_size=5,
            activation = "relu",
            name = "C5"
            ),
        keras.layers.Flatten(name="Flatten5_6"),
        keras.layers.Dense(units=84, activation="relu", name="F6"),
        keras.layers.Dropout(rate=.5, name="Dropout6_7"),
        keras.layers.Dense(units=Constants.NUM_CLASSES, activation="softmax", name="Output_F7")
        ])
    lenet5.compile(
        keras.optimizers.Adadelta(Constants.LEARNING_RATE),
        loss = "categorical_crossentropy",
        metrics = ["acc"]
        )
    history.append(lenet5.fit(
        x_tr, y_tr,
        batch_size = Constants.BATCH_SIZE,
        epochs = Constants.NUM_EPOCHES,
        verbose = 1,
        #callbacks = [keras.callbacks.EarlyStopping(monitor="val_acc", patience=2, verbose=1)],
        validation_data = (x_val, y_val)
        ).history)
    evaluation.append(lenet5.evaluate(x_keras, y_keras, verbose=1))
    idx_fold += 1