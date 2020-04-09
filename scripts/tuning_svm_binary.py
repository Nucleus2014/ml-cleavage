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


params_svm = {"class_weight": [None, "balanced"],
            "C":[0.001,0.005, 0.01,0.05,0.1,0.5,1,1.5,2], "coef0": [0,0.01,0.1,0.5,1], 
            "decision_function_shape": ['ovr','ovo'], "degree":[1,2,3,4,5,6,7,8],
             "kernel":["linear","poly","rbf","sigmoid"],"shrinking":[True,False]}

#Train + Val = 70%, Test = 30%, Train:Val = 8:2
# Parameter tuning for random forest
print("|{:3s}|{:14s}|{:1s}|{:5s}|{:24s}|{:6s}|{:6s}|{:9s}|".format("ind","class_weight", "C", "coef0","decision_function_shape","degree","kernel","shrinking"))
acc = []
count = 0
for t_cw in params_svm["class_weight"]:
    for t_c in params_svm["C"]:
        for t_coef in params_svm["coef0"]:
            for t_dfs in params_svm["decision_function_shape"]:
                for t_de in params_svm["degree"]:
                    for t_ker in params_svm["kernel"]:
                        for t_sh in params_svm["shrinking"]:
                            count += 1
                            print("|{:3d}|{:8s}|{:.3f}|{:.2f}|{:3s}|{:1d}|{:12s}|{:5s}|"
                                  .format(count,str(t_cw),t_c,t_coef,t_dfs,t_de,t_ker,str(t_sh)))
                            tmp = []
                            for i in range(0,5):
                                X_train, X_val, y_train, y_val = train_test_split(X_remain, 
                                                                                      y_remain, test_size = 0.14)
                                ln = tuning(X_train, y_train, X_val, y_val, dict_svm = {"class_weight":t_cw, "C":t_c,
                                                            "coef0":t_coef,"decision_function_shape":t_dfs,
                                                            "degree":t_de,"kernel":t_ker,
                                                            "shrinking":t_sh})
                                tmp.append(ln.svm_acc)
                            print("------accuracy:{}-------".format(np.mean(tmp)))
                            acc.append(np.mean(tmp))
print("----The highest accuracy is:{} which is at {}----".format(np.max(acc),np.argmax(acc)))


