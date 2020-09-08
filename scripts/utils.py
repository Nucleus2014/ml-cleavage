# This script is to contain all utility functions for training machine learning models
# Author: Changpeng Lu

import pickle as pkl
from sklearn import preprocessing
import pandas as pd
import numpy as np
import logging

def train_test(model, trainset, trainy, testset, testy):
    model.fit(trainset, trainy)
    pre = model.predict(testset)
    prob = model.predict_proba(testset)
    acc = np.mean(pre == testy)
    return prob, acc

def train_test_ann(model, n_class, trainset, trainy, testset, testy):
    result = trainy.values
    newre = np.zeros(len(result))
    for i in range(0,len(result)):
        if result[i] == 'CLEAVED':
            newre[i] = 0
        elif result[i] == 'UNCLEAVED':
            newre[i] = 1.0
        elif (n_class == 3) and (result[i] == 'MIDDLE'):
            newre[i] = 2.0
    newre = newre.astype(int)

    if n_class == 2:
        class_names = ["CLEAVED","UNCLEAVED"]
    elif n_class == 3:
        class_names = ["CLEAVED","MIDDLE","UNCLEAVED"]

    model.fit(trainset.values, newre, epochs=10)

    result_test = testy.values
    newre_test = np.zeros(len(result_test))
    for i in range(0,len(result_test)):
        if result_test[i] == 'CLEAVED':
            newre_test[i] = 0
        elif result_test[i] == 'UNCLEAVED':
            newre_test[i] = 1
        elif (n_class == 3) and (result_test[i] == 'MIDDLE'):
            newre_test[i] = 2
    newre_test = newre_test.astype(int)

    test_loss, test_acc = model.evaluate(testset.values, newre_test)

    predictions = model.predict(testset.values)
    label_pre = []
    for i in range(0,len(predictions)):
        label_pre.append(np.argmax(predictions[i]))
    prob_nn = predictions
    return prob_nn, test_acc

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def scale(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    trans = min_max_scaler.fit_transform(df)
    df_trans = pd.DataFrame(trans,columns = df.columns)
    return df_trans

def load_data(dataset, classes):
    if classes == 2:
        class_type = 'binary'
    elif classes == 3:
        class_type = 'ternary'
    X_train = pkl.load(open('../Data/X_train_'+dataset+'_'+class_type+'_0.7','rb'))
    X_test = pkl.load(open('../Data/X_test_'+dataset+'_'+class_type+'_0.3','rb'))
    y_train = pkl.load(open('../Data/y_train_'+dataset+'_'+class_type+'_0.7','rb'))
    y_test = pkl.load(open('../Data/y_test_'+dataset+'_'+class_type+'_0.3','rb'))

    return X_train, y_train, X_test, y_test

def sequences_to_data(sequences):
    """
    Example input: array(['YYTTQ', 'YYTIQ', 'YYTHY', ..., 'YTATD', 'YTESW', 'YWCQH'],
      dtype='<U5')
    Example output: dtype=dataframe, first 100 columns are sequence one hot encoder, then following 4 total energy terms
    """
#     test_name = np.loadtxt(sequences_list,dtype='str')
    amino_acid_list = "ACDEFGHIKLMNPQRSTVWY"
    sequence_encoder = np.zeros((len(sequences), 100))
    for n in range(len(sequences)):
        for i in range(5):
            ind = amino_acid_list.index(sequences[n][i])
            sequence_encoder[n, 20*i+ind] = 1
    df = pd.read_csv("structure_features.csv")
    df.index = df['sequence']
    energy_df = df.loc[sequences, ['prot','pept','cst','amber']]
    seq_df = pd.DataFrame(sequence_encoder, index=sequences, columns = range(0,100))
    final_df = pd.concat([seq_df, energy_df], axis=1)
    return final_df

def generate_data(mutant, test_sequences_list):
    """
    Example input: mutant = 'Triple' or 'A171T' or 'D183A'
                   test_sequences_list = file path of testset sample names, eg., array(['YYTTQ', 'YYTIQ', 'YYTHY', ..., 'YTATD', 'YTESW', 'YWCQH'],
      dtype='<U5')
    Example output: pickle format X_train, y_train, X_test, y_test
    """
    maps = {'Triple': '011', 'A171T': '021', 'D183A': '091'}
    sample_list_path = '../../Rotation_in_Sagar_Lab/Dataset_For_All_Experimental_Sequences/'
    samples = []
    for file in os.listdir(sample_list_path):
        if file[0:3] == maps[mutant]:
            fp = open(os.path.join(sample_list_path, file),'r')
            classname = file.split('.')[0].split('_')[-1]
            if classname == 'unselected':
                continue
            for line in fp:
                samples.append([line.strip(), classname.upper()])
            fp.close()
    samples = np.asarray(samples)
    all_samples_df = sequences_to_data(samples[:,0])
    all_samples_df['result'] = samples[:,1]
    
    test_name = np.loadtxt(test_sequences_list,dtype='str')
    test_mask = np.asarray([v in test_name for v in all_samples_df.index.values])
    
    pkl.dump(all_samples_df.ix[test_name,:-1], open(os.path.join('../Data', 'X_test_' + mutant + '_ternary_0.3'),'wb'))
    pkl.dump(all_samples_df.ix[test_name,-1], open(os.path.join('../Data', 'y_test_' + mutant + '_ternary_0.3'),'wb'))
    pkl.dump(all_samples_df[~test_mask].iloc[:,:-1], open(os.path.join('../Data', 'X_train_' + mutant + '_ternary_0.7'),'wb'))
    pkl.dump(all_samples_df[~test_mask].iloc[:,-1], open(os.path.join('../Data', 'y_train_' + mutant + '_ternary_0.7'),'wb'))
