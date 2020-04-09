import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class learn:
    def __init__(self, X_train, y_train, X_test, y_test, params_lg = None, params_rf = None, params_dt = None, 
                 params_svm = None):
        # lg
        if params_lg != None:
            c = params_lg['C']
            peny = params_lg['penalty']
            weight = params_lg['class_weight']
            slv = params_lg['solver']
            n_iter = params_lg['max_iter']
            ver = params_lg['verbose']

            from sklearn import linear_model
            lg = linear_model.LogisticRegression(C = c, penalty = peny, class_weight = weight, solver = slv, 
                                             max_iter = n_iter, verbose = ver)
            lg.fit(X_train, y_train)
            self.pre_lg = lg.predict(X_test)
            self.prob_lg = lg.predict_proba(X_test)
            self.lg_acc = np.mean(self.pre_lg == y_test[1])
        
        # rf
        if params_rf != None:
            c = params_rf['n_est']
            cw = params_rf["class_weight"]
            msl = params_rf["min_samples_leaf"]
            boot = params_rf["bootstrap"]
            cri = params_rf["criterion"]
            mf = params_rf["max_features"]
            mss = params_rf["min_samples_split"]
            ver = params_rf["verbose"]
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators = c, class_weight = cw, min_samples_leaf = msl, 
                                       bootstrap = boot, criterion = cri, max_features = mf, 
                                        min_samples_split = mss, verbose = ver)
            rf.fit(X_train, y_train)
            self.pre_rf = rf.predict(X_test)
            self.prob_rf = rf.predict_proba(X_test)
            self.rf_acc = np.mean(self.pre_rf == y_test[1])
        
        # dt
        if params_dt != None:
            mss = params_dt['min_samples_split']
            msl = params_dt['min_samples_leaf']
            cw = params_dt['class_weight']
            cri = params_dt['criterion']
            mf = params_dt['max_features']
            ver = params_dt['splitter']
        
            from sklearn.tree import DecisionTreeClassifier
            dt = DecisionTreeClassifier(random_state=0,class_weight = cw, criterion = cri, max_features = mf, min_samples_split = mss, min_samples_leaf = msl, splitter = ver)
            dt.fit(X_train,y_train)
            self.pre_dt = dt.predict(X_test)
            self.prob_dt = dt.predict_proba(X_test)
            self.dt_acc = np.mean(self.pre_dt == y_test[1])
        
        #svm
        if params_svm != None:
            cw = params_svm['class_weight']
            c = params_svm['C']
            coef = params_svm['coef0']
            dfs = params_svm['decision_function_shape']
            de = params_svm['degree']
            ker = params_svm['kernel']
            sh = params_svm['shrinking']
        
            from sklearn import svm
            svm = svm.SVC(class_weight = cw, C = c, coef0 = coef, decision_function_shape = dfs, degree = de, kernel = ker, shrinking = sh, probability = True)
            svm.fit(X_train, y_train)
            self.pre_svm = svm.predict(X_test)
            self.prob_svm = svm.predict_proba(X_test)
            self.svm_acc = np.mean(self.pre_svm == y_test[1])
def tuning(X_train, y_train, X_test, y_test, dict_lg = None, dict_rf = None, dict_dt = None,  
                 dict_svm = None):
    return learn(X_train, y_train, X_test, y_test, params_lg = dict_lg, params_rf = dict_rf, params_dt = dict_dt,  
                 params_svm = dict_svm)
if __name__ == '__main__':
	learn() 
