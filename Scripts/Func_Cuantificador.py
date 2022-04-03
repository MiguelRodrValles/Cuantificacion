# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 12:00:27 2021

@author: mrvalles
"""
def error_Q(y_obs,y_predicted):
    tot=len(y_obs)
    num_obs=sum(y_obs)
    num_pred=sum(y_predicted)
    P_obs=num_obs/tot
    P_pred=num_pred/tot
    AE=abs(P_obs-P_pred)
    RAE=AE/P_obs

def Est_FPR_FNR(X,y,folds,clf):
    from sklearn.model_selection import KFold
    from sklearn import datasets, svm, metrics
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression 
    import numpy as np
    kf = KFold(n_splits=folds)
    Error=np.zeros((folds, 2))
    i=0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #param_grid = [
  #{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  #{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 #]
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        C=metrics.confusion_matrix(y_test, predicted)
        Error[i,0]=C[1,1]/(C[1,1]+C[1,0]) #TPR
        Error[i,1]=C[0,1]/(C[0,0]+C[0,1]) #FPR
        i=i+1
    return [(Error[:,0].sum())/folds,(Error[:,1].sum())/folds]


def Est_FPR_FNR_2(X,y,folds,c,g,k):
    from sklearn.model_selection import KFold
    from sklearn import datasets, svm, metrics
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    import numpy as np
    kf = KFold(n_splits=folds)
    Error=np.zeros((folds, 2))
    i=0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = svm.SVC(C=c,Kernel=k,gamma=g).fit(X_train, y_train)
        predicted = clf.predict(X_test)
        C=metrics.confusion_matrix(y_test, predicted)
        Error[i,0]=C[1,1]/(C[1,1]+C[1,0]) #TPR
        Error[i,1]=C[0,1]/(C[0,0]+C[0,1]) #FPR
        i=i+1
    return [(Error[:,0].sum())/folds,(Error[:,1].sum())/folds]
