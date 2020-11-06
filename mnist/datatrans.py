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

train_csv = pd.read_csv("./train.csv")
test_csv = pd.read_csv("./test.csv")
sample_submission_csv = pd.read_csv("./sample_submission.csv")

np.savez("./k_mnist.%s.npz" % nowtime,
         x_train = train_csv.values[ : , 1 : ],
         y_train = train_csv.values[ : , 0 ],
         x_test = test_csv.values[ : ],
         y_test = sample_submission_csv.values[ : , 1 ]
         )

np.savez("./k_mnist.npz",
         x_train = train_csv.values[ : , 1 : ],
         y_train = train_csv.values[ : , 0 ],
         x_test = test_csv.values[ : ],
         y_test = sample_submission_csv.values[ : , 1 ]
         )
