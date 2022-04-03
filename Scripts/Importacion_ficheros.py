# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 19:30:53 2021

@author: mrvalles
"""
directorio='C:/Users/mrvalles/Desktop/TFM/Datos/Agregacion/Files'
def lectura(conjunto,directorio,tipo,sheet,sep):
    import os
    import numpy as np
    import pandas as pd
    os.chdir(directorio)
    if tipo=='dat':
        a=open(conjunto,'r')
        lista=[]
        for x in a:
            aux=x.strip().split(sep)
            aux=[transform_num(i,0) for i in aux]
            lista.append(aux)
        a.close()
        M1=np.array(lista)
    elif tipo=='xls':
        M1 = pd.read_excel(io=conjunto, sheet_name=sheet)
        M1=M1.to_numpy()
    elif tipo=='csv':
        M1 = pd.read_csv(conjunto, sep=sep)
        M1=M1.to_numpy()
    return M1

M1=lectura('breast-cancer-wisconsin.data',directorio,'dat','',',')
M2=lectura('breast-cancer.data',directorio,'dat','',',')
M3=lectura('crx.data',directorio,'dat','',',')
M4=lectura('CCd.xls',directorio,'xls','Data','')
M5=lectura('winequality-red.csv',directorio,'csv','Data',";")
M6=lectura('winequality-white.csv',directorio,'csv','Data',";")
M7=lectura('german.data',directorio,'dat','',' ')
M8=lectura('australian.dat',directorio,'dat','',' ')
M9=lectura('mammographic_masses.data',directorio,'dat','',' ')

#generación M1
def genera_y(data,id_target,val_pos,val_neg):
    y=[]
    for i in range(len(data)):
        if data[i,id_target-1]==val_pos:
            y.append('+1')
        else:
            y.append('-1')
    
    return y

def genera_x(data,seq):
    lista=[]
    for i in range(len(data)):
        dictionary={}
        for j in seq:
            dictionary.update({j:data[i,j]})
        lista.append(dictionary)
    return lista

def transform_num(x,num_def):
    if x.isnumeric()==True:
        return float(x)
    else:
        return num_def



x=genera_x(M1,list(range(1,10)))
y=genera_y(M1,11,2,4)

def escribe_fich(y,x,file):
    if len(y)==len(x):
        f=open(file,'w')
        for i in range(len(y)):
            f.write(y[i] + ' ')
            for key, value in x[i].items(): 
                f.write('%s:%s ' % (key, value))
            f.write("\n")
        f.close()
        
escribe_fich(y,x,'pp.txt')    
