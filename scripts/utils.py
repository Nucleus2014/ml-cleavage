# train test for different regular ml tools

# Settings
import pandas as pd
import numpy as np
import os
import argparse
import pickle as pkl
from sklearn import preprocessing
#import tensorflow as tf
#from tensorflow import keras
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# initializing things
parser = argparse.ArgumentParser()
parser.add_argument("-i", type = str) # input data

args = parser.parse_args()

root = os.getcwd()
classifier_path = os.path.join(root, "Classifications")
data_path = os.path.join(root, "Data")
dict_path = os.path.join(root, "Dicts")

# get_seq is to get the list of sample names
def get_seq(df): # dataframe to be trained with labels
    aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'] # 20 aa
    sample_list = []
    for ind in range(0,df.shape[0]):
        ind_j = np.where(df.iloc[ind,0:100] == 1) # locate where indicates 1 in columns
        aa = ""
        for j in ind_j[0]:
            while j >= 20:
                j = j - 20
            aa = aa + aa_list[j]
        sample_list.append(aa)
    class_list = [sample_list, df["label"]]
    return class_list

# feature_select is to select features to be used for training, must have no missing values in the first sample,
# seq features must be np.int64 type, energy must be np.float64 type
# sel = "seq", "energy", "combine" (sequence + total_energy) is available
def feature_select(df, sel = "combine"):
    sel_set = [] # indices which indicate columns to be considered
    if (sel == "seq") or (sel == "combine"):
        for j in range(0,df.shape[1]):
            if type(df.iloc[0,j]) == np.int64:
                sel_set.append(j)
    if (sel == "energy") or (sel == "combine"):
        for j in range(0,df.shape[1]):
            if type(df.iloc[0,j]) == np.float64:
                sel_set.append(j)
    return df.iloc[:,sel_set] # X waited to be preprocessed and trained

# scale is to do data transformation
# input must be X_train without labels
def scale(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    trans = min_max_scaler.fit_transform(df)
    df_trans = pd.DataFrame(trans,columns = df.columns)
    return df_trans

if __name__ == '__main__':
            

