#!/usr/bin/env Python
# -*- coding: utf-8 -*-

"""
Created on Fri Nov  6 07:56:36 2020

@author: Admin
"""

import numpy as np
import pandas as pd
import time

y_pred = np.load("./djczk.submission.titanic.GBC.npz", allow_pickle=True)["y_pred"]
data_flame = pd.DataFrame({ "PassengerId": np.arange(892, 1310), "Survived": y_pred })
data_flame.to_csv("./djczk.submission.titanic.GBC.%s.csv" % time.time(), index=False)
