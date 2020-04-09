# hyperparameter tuning for random forest on ternary classification

import pandas as pd
import numpy as np
import os
import argparse
import pickle as pkl
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from learn import tuning

X_remain = pkl.load(open("Data/X_train_HCV_binary_0.7","rb"))
y_remain = pkl.load(open("Data/y_train_HCV_binary_0.7","rb"))

params_dt = {"class_weight": [None, "balanced"],
            "min_samples_leaf":[1,2,5,10,20,30,50,100,200],"criterion":["gini","entropy"],
            "max_features":["auto","log2",None], "min_samples_split":[2,3,5,10,20,30,50,100,200],
             "splitter":["best","random"]}

#Train + Val = 70%, Test = 30%, Train:Val = 8:2
# Parameter tuning for decision tree
print("|{:3s}|{:14s}|{:16s}|{:9s}|{:12s}|{:17s}|{:8s}|".format("ind","class_weight",
                                                               "min_samples_leaf","criterion",
                                                               "max_features", "min_samples_split","splitter"))
acc = []
count = 0
for t_cw in params_dt["class_weight"]:
    for t_msl in params_dt["min_samples_leaf"]:
        for t_cri in params_dt["criterion"]:
            for t_mf in params_dt["max_features"]:
                for t_mss in params_dt["min_samples_split"]:
                    for t_ver in params_dt["splitter"]:
                        count += 1
                        print("|{:3d}|{:14s}|{:10d}|{:7s}|{:4s}|{:4d}|{:6s}|"
                                  .format(count,str(t_cw),t_msl,t_cri,str(t_mf),t_mss,t_ver))
                        tmp = []
                        for i in range(0,5):
                            X_train, X_val, y_train, y_val = train_test_split(X_remain, 
                                                                                      y_remain, test_size = 0.14)
                            ln = tuning(X_train, y_train, X_val, y_val, dict_dt = {"class_weight":t_cw, "min_samples_leaf":t_msl,
                                                            "criterion":t_cri,
                                                            "max_features":t_mf,"min_samples_split":t_mss,
                                                            "splitter":t_ver})
                            tmp.append(ln.dt_acc)
                            print("------accuracy:{}-------".format(np.mean(tmp)))
                            acc.append(np.mean(tmp))
print("----The highest accuracy is:{} which is at {}----".format(np.max(acc),np.argmax(acc)))


