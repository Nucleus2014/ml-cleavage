# This script is to train and test machine learning models.
# Valid models are:
# Logistic regression
# Random forest
# Decision tree
# SVM
# Neural Network
"""
    Usage:
        python test.py -m logistic_regression -i A171T -class 3 --C 1 --solver lbfgs --penalty l2
"""

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import argparse
import pickle as pkl
from sklearn import preprocessing
from utils import *

# Argument
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices = ["logistic_regression", "random_forest", "decision_tree", "svm", "ann"])
    parser.add_argument("-i", "--dataset", type = str) # dataset name, HCV, A171T, D183A, or triple for now
    parser.add_argument("-class", "--classes", type = int) # number of classes, 2 or 3
    parser.add_argument("-C","--C", type = float, default = 1)
    parser.add_argument("-solver", "--solver", choices = ["newton-cg","lbfgs","liblinear","sag","saga"], default = 'lbfgs')
    parser.add_argument("-penalty", "--penalty", choices = ["l1","l2"], default = "l2")
    parser.add_argument("-cw", "--class_weight", choices = ["balanced", "balanced_subsample", None], default="balanced")
    parser.add_argument("-criter", "--criterion", choices = ["gini", "entropy"], default = "gini")
    parser.add_argument("-mf", "--max_features", choices = ["auto", "log2", None], default = "auto")
    parser.add_argument("-mss","--min_samples_split", type=int, default = 2)
    parser.add_argument("-msl", "--min_samples_leaf", type=int, default = 1)
    parser.add_argument("-bs", "--bootstrap", action = 'store_true')
    parser.add_argument("-ne", "--n_estimators", type=int, default=100)
    parser.add_argument("-split", "--splitter", choices = ["best", "random"], default = "best")
    parser.add_argument("-coef", "--coef0", type = float, default=0)
    parser.add_argument("-degree", "--degree", type = int, default = 3)
    parser.add_argument("-shrinking", "--shrinking", action = 'store_true')
    parser.add_argument("-kernel", "--kernel", choices = ["linear", "poly", "rbf", "sigmoid"], default="linear")
    parser.add_argument("-energy", "--energy_only", action = "store_true")
    parser.add_argument("-lr", "--learning_rate", type=float, default = 0.001)
    parser.add_argument("-dropout","--dropout", type=float, default = 0)
    parser.add_argument("-save", "--save", type = str, default = "./") # path of saving logits
    args = parser.parse_args()
    return args

def main(args):
    dataset = args.dataset
    classes = args.classes
    model = args.model
    X_train, y_train, X_test, y_test = load_data(dataset, classes)
    if args.energy_only == True:
        X_train = X_train.iloc[:,100:].copy()
        X_test = X_test.iloc[:, 100:].copy()
    X_train = scale(X_train)
    X_test = scale(X_test)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'),filepath=os.path.abspath(__file__))
    if classes == 2:
        classtype = 'binary'
    elif classes == 3:
        classtype = 'ternary'

    if model == 'logistic_regression':
        if args.C != None:
            C = args.C
        else:
            C = 1
        if args.penalty != None:
            penalty = args.penalty
        else:
            penalty = 'l2' 
        from sklearn import linear_model
        lg = linear_model.LogisticRegression(C = args.C, solver = args.solver, penalty = args.penalty, max_iter = 500)
        logger.info(lg)
        prob, acc = train_test(lg, X_train, y_train, X_test, y_test)
        logger.info('Test Accuracy:{:.4f}'.format(acc))

        np.savetxt(os.path.join(args.save, 'logits_' + args.model + '_' + str(dataset) + '_' + classtype + '_C_' + str(args.C) \
                           + '_solver_' + args.solver + '_penalty_' + str(args.penalty)), prob)
    elif model == 'random_forest':
        av_acc = 0
        for i in range(20):
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(class_weight = args.class_weight, criterion = args.criterion, max_features = args.max_features, min_samples_split = args.min_samples_split, min_samples_leaf = args.min_samples_leaf, bootstrap = args.bootstrap, n_estimators=args.n_estimators)
            logger.info(rf)
            prob, acc = train_test(rf, X_train, y_train, X_test, y_test)
            av_acc += acc
        av_acc = av_acc / 20
        logger.info('Test Accuracy:{:.4f}'.format(av_acc))
        np.savetxt(os.path.join(args.save, 'logits_' + args.model + '_' + str(dataset) + '_' + classtype + '_class_weight_' + args.class_weight + '_criterion_' + args.criterion + '_max_features_' + str(args.max_features) + '_min_samples_split_' + str(args.min_samples_split) + '_min_samples_leaf_' + \
          str(args.min_samples_leaf) + '_bootstrap_' + str(args.bootstrap) + '_n_estimators_' + str(args.n_estimators)), prob)
    elif model == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(class_weight = args.class_weight, criterion = args.criterion, max_features = args.max_features, min_samples_split = args.min_samples_split, min_samples_leaf = args.min_samples_leaf, splitter = args.splitter)
        logger.info(dt)
        prob, acc = train_test(dt, X_train, y_train, X_test, y_test)
        logger.info('Test Accuracy:{:.4f}'.format(acc))
        np.savetxt(os.path.join(args.save, 'logits_' + args.model + '_' + str(dataset) + '_' + classtype + '_class_weight_' + args.class_weight + '_criterion_' + args.criterion + '_max_features_' + str(args.max_features) + '_min_samples_split_' + str(args.min_samples_split) + '_min_samples_leaf_' + \
          str(args.min_samples_leaf) + '_splitter_' + str(args.splitter)), prob)
    elif model == 'svm':
        from sklearn import svm
        svmsvc = svm.SVC(C = args.C, class_weight = args.class_weight, coef0 = args.coef0, degree = args.degree, shrinking = args.shrinking, kernel = args.kernel, probability=True)
        logger.info(svmsvc)
        prob, acc = train_test(svmsvc, X_train, y_train, X_test, y_test)
        logger.info('Test Accuracy:{:.4f}'.format(acc))
        np.savetxt(os.path.join(args.save, 'logits_' + args.model + '_' + str(dataset) + '_' + classtype + '_C_' + str(args.C) + '_class_weight_' + args.class_weight + '_coef0_' + str(args.coef0) + '_degree_' + str(args.degree) + '_shrinking_' + str(args.shrinking) + '_kernel_' + args.kernel), prob)
    elif model == 'ann':
        import tensorflow as tf
        from tensorflow import keras
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        dropout = args.dropout
        lr = args.learning_rate
        n_class = classes
        ann = keras.Sequential([keras.layers.Dense(1024, activation=tf.nn.relu),
                                  keras.layers.Dropout(dropout, input_shape = (1024,)),
                                  keras.layers.Dense(n_class, activation=tf.nn.softmax)])

        ann.compile(optimizer=tf.train.AdamOptimizer(learning_rate = lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        logger.info(ann)
        prob, acc = train_test_ann(ann, n_class, X_train, y_train, X_test, y_test,logger)
        logger.info('Test Accuracy:{:.4f}'.format(acc))
        np.savetxt(os.path.join(args.save, 'logits_' + args.model + '_' + str(dataset) + '_' + classtype + '_lr_' + str(args.learning_rate) + '_dropout_' + str(args.dropout)), prob)

if __name__ == '__main__':
    args = parse_args()
    main(args)
