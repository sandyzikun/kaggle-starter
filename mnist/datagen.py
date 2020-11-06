#!/usr/bin/env Python
# -*- coding: utf-8 -*-

"""
Created on Wed Nov  4 21:56:30 2020

@author: sandyzikun
"""

import numpy as np
import pandas as pd

import time
nowtime = time.time()

with np.load("./k_mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"]
    x_test, y_test = f["x_test"], f["y_test"]
    f.close()

class MNIST_Utils(object):
    """
    docstring
    """

    def __init__(self, name=None):
        self.name = str(name) if name else "mnistutils.%s" % time.time()
        return

    def vec2mat_(self, input_tensor):
        assert input_tensor.shape == (784,)
        return input_tensor.reshape(28, 28)

    def mat28expand32_(self, input_tensor):
        assert input_tensor.shape == (28, 28)
        output_tensor = np.zeros((32, 32))
        output_tensor[ 2 : (-2) , 2 : (-2) ] = input_tensor
        return output_tensor

    def mat28zoom112_(self, input_tensor):
        assert input_tensor.shape == (28, 28)
        output_tensor = np.zeros((112, 112))
        for i in range(112):
            for j in range(112):
                output_tensor[i, j] = input_tensor[i // 4, j // 4]
        return output_tensor

    def vec2mat(self, input_tensor):
        return np.array([ self.vec2mat_(input_tensor[idx]) for idx in range(len(input_tensor)) ])

    def mat28expand32(self, input_tensor):
        return np.array([ self.mat28expand32_(input_tensor[idx]) for idx in range(len(input_tensor)) ])

    def mat28zoom112(self, input_tensor):
        return np.array([ self.mat28zoom112_(input_tensor[idx]) for idx in range(len(input_tensor)) ])

mu = MNIST_Utils()

x_train_mat = mu.vec2mat(x_train)
x_test_mat = mu.vec2mat(x_test)
np.savez("./k_mnist.mat28.%s.npz" % nowtime,
         x_train=x_train_mat, y_train=y_train,
         x_test=x_test_mat, y_test=y_test
         )

np.savez("./k_mnist.mat32.%s.npz" % nowtime,
         x_train = mu.mat28expand32(x_train_mat),
         y_train = y_train,
         x_test = mu.mat28expand32(x_test_mat),
         y_test = y_test
         )

np.savez("./k_mnist.mat112.%s.npz" % nowtime,
         x_train = mu.mat28zoom112(x_train_mat),
         y_train = y_train,
         x_test = mu.mat28zoom112(x_test_mat),
         y_test = y_test
         )
