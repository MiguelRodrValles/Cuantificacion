# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 19:30:53 2021

@author: mrvalles
"""
directorio='C:/Users/mrvalles/Desktop/TFM/Datos/Agregacion'
def lectura(conjunto,directorio,tipo,sheet,sep):
    import os
    import numpy as np
    import pandas as pd
    os.chdir(directorio)
    if tipo=='dat':
        if sep==',':
            a=open(conjunto,'r')
            lista=[]
            for x in a:
                x.strip().split(",")
                lista.append(x.strip().split(","))
            a.close()
            M1=np.array(lista)
        elif sep=='':
            M1=np.loadtxt(conjunto, delimiter=sep)
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
M7=lectura('german.data',directorio,'dat','',',')

with open('australian.dat') as players_data:
    	players_data.read()
import numpy as np
d = np.loadtxt('australian.dat', delimiter=" ")


directorio='C:/Users/mrvalles/Desktop/TFM/Datos/Agregacion/Files'
def lectura2(conjunto,directorio,tipo,sheet,sep):
    import os
    import numpy as np
    import pandas as pd
    os.chdir(directorio)
    if tipo=='dat':
        M1=np.loadtxt(conjunto,delimiter=sep)
    elif tipo=='xls':
        M1 = pd.read_excel(io=conjunto, sheet_name=sheet)
        M1=M1.to_numpy()
    elif tipo=='csv':
        M1 = pd.read_csv(conjunto, sep=sep)
        M1=M1.to_numpy()
    return M1

M1=lectura2('breast-cancer-wisconsin.data',directorio,'dat','',',')
M2=lectura2('breast-cancer.data',directorio,'dat','',',')
M3=lectura2('crx.data',directorio,'dat','',',')
M4=lectura2('CCd.xls',directorio,'xls','Data','')
M5=lectura2('winequality-red.csv',directorio,'csv','Data',";")
M6=lectura2('winequality-white.csv',directorio,'csv','Data',";")
M7=lectura2('german.data',directorio,'dat','',',')
