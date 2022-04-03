# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:17:23 2021

@author: mrvalles
"""


import numpy as np
import os


#Directorio de funciones.
dir_func='C:/Users/mrvalles/Desktop/TFM/Scripts'
os.chdir(dir_func)
from Funciones import get_x
from Funciones import convert
from Funciones import get_y

#Directorio de datos.
dir_datos='C:/Users/mrvalles/Desktop/TFM/Datos/Agregacion/Files'
os.chdir(dir_datos)

#Carga de datos
f=open('breast-cancer-wisconsin.data')
Lines = f.readlines()
cleaned_matrix = [] 
for raw_line in Lines:
    line = raw_line.rstrip('\n')  # "1.0,2.0,3.0"
    sVals = raw_line.split(',')   # ["1.0", "2.0, "3.0"]
    fVals = list(map(convert, sVals))  # [1.0, 2.0, 3.0]
    cleaned_matrix.append(fVals)  # [[1
f.close()


x=get_x(cleaned_matrix,list(range(1,10)))
y=get_y(cleaned_matrix,10,['2'],['4'])

import os
os.chdir('C:/Users/mrvalles/Desktop/TFM/liblinear-2.42/python')
from liblinearutil import *
prob=problem(y, x)
m = train(y[:400], x[:400], '-s 0 -c 4')
p_label, p_acc, p_val = predict(y[400:], x[400:], m,'-b 1')

