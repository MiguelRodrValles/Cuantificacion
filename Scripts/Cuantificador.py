# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:30:35 2021

@author: mrvalles
"""
#Carga de datos
directorio_dat='C:/Users/mrvalles/Desktop/TFM/Datos/Agregacion/Files/txt'
directorio_fun='C:/Users/mrvalles/Desktop/TFM/Datos/Agregacion/Files'
import os
os.chdir(directorio_fun)
import numpy as np
import pandas as pd
import random
from Func_Cuantificador import Est_FPR_FNR,Est_FPR_FNR_2
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
os.chdir(directorio_dat)

M1=np.loadtxt('Data1.txt', delimiter="\t",skiprows=1)
#Transformacion
for i in (range(M1.shape[0])):
    if M1[i,(M1.shape[1])-1]==2:
        M1[i,(M1.shape[1])-1]=0
    else:
        M1[i,(M1.shape[1])-1]=1
X=M1[:,1:10]
y=M1[:,10]

#Dividir Train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Elección Parámetro adecuado.
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

clf = GridSearchCV(SVC(), param_grid)
clf.fit(X_train, y_train)
print(clf.best_params_)

Ajuste=Est_FPR_FNR(X_train,y_train,10)
Ajuste=Est_FPR_FNR_2(X_train,y_train,10,)

#Prediccion
Error=np.zeros((1000, 4))
for i in (range(1000)):
    p=random.uniform(0,1)
    Ind_x_pos=np.random.choice(np.where([y_test==1])[1],int(p*len(y_test)),replace=True)
    Ind_x_neg=np.random.choice(np.where([y_test==0])[1],len(y_test)-int(p*len(y_test)),replace=True)
    Ind_x=np.concatenate((Ind_x_pos,Ind_x_neg))
    X_aux=X_test[Ind_x,:]
    y_aux=y_test[Ind_x]
    predicted = clf.predict(X_aux)
    Error[i,2]=sum(y_aux)/len(y_test)
    Error[i,1]=sum(predicted)/len(y_test)    
    Error[i,0]=p
    Error[i,3]=(Error[i,1]-Ajuste[1])/(Ajuste[0]-Ajuste[1])
    

file=open('prueba.txt','w')
for i in range(len(Error)):
    file.write(str(Error[i,0])+' '+str(Error[i,1])+' '+str(Error[i,2])+' '+str(Error[i,3])+'\n')
file.close()


predicted = clf.predict(X_test)
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


predicted2 = clf2.predict(X_test)
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted2)}\n")


predicted3 = clf3.predict(X_test)
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted3)}\n")

a=clf.score(X_train, y_train)
clf2.score(X_train, y_train)
clf3.score(X_train, y_train)

metrics.confusion_matrix(y_test, predicted)