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

X_remain = pkl.load(open("Data/X_train_HCV_ternary_0.7","rb"))
y_remain = pkl.load(open("Data/y_train_HCV_ternary_0.7","rb"))


params_rf = {"n_est": [10, 50, 100, 500, 1000, 5000], "class_weight": [None, "balanced", "balanced_subsample"],
            "min_samples_leaf":[1,2,5,10,20,30,50,100,200],"bootstrap":[True,False],"criterion":["gini","entropy"],
            "max_features":["auto","log2",None], "min_samples_split":[2,3,5,10,20,30,50,100,200],
             "verbose":[0,1,2,3,4,5,10,20,30,50,100]}

#Train + Val = 70%, Test = 30%, Train:Val = 8:2
# Parameter tuning for random forest
print("|{:3s}|{:5s}|{:14s}|{:16s}|{:9s}|{:9s}|{:12s}|{:17s}|{:7s}|".format("ind","n_est","class_weight",
                                                               "min_samples_leaf","bootstrap","criterion",
                                                               "max_features", "min_samples_split","verbose"))
acc = []
count = 0
for t_c in params_rf["n_est"]:
    for t_cw in params_rf["class_weight"]:
        for t_msl in params_rf["min_samples_leaf"]:
            for t_boot in params_rf["bootstrap"]:
                for t_cri in params_rf["criterion"]:
                    for t_mf in params_rf["max_features"]:
                        for t_mss in params_rf["min_samples_split"]:
                            for t_ver in params_rf["verbose"]:
                                count += 1
                                print("|{:3d}|{:.1f}|{:14s}|{:10d}|{:5s}|{:7s}|{:4s}|{:4d}|{:3d}|"
                                  .format(count,t_c,str(t_cw),t_msl,str(t_boot),t_cri,str(t_mf),t_mss,t_ver))
                                tmp = []
                                for i in range(0,5):
                                    X_train, X_val, y_train, y_val = train_test_split(X_remain, 
                                                                                      y_remain, test_size = 0.14)
                                    ln = tuning(X_train, y_train, X_val, y_val, dict_rf = {"n_est":t_c,
                                                            "class_weight":t_cw, "min_samples_leaf":t_msl,
                                                            "bootstrap":t_boot,"criterion":t_cri,
                                                            "max_features":t_mf,"min_samples_split":t_mss,
                                                            "verbose":t_ver})
                                    tmp.append(ln.rf_acc)
                                print("------accuracy:{}-------".format(np.mean(tmp)))
                                acc.append(np.mean(tmp))
print("----The highest accuracy is:{} which is at {}----".format(np.max(acc),np.argmax(acc)))


