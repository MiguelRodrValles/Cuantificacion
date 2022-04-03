# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:46:57 2021

@author: mrvalles
"""
def convert(x):
    import numpy as np
    if x.isalnum():
        y=np.float32(x)
    else:
        y=0
    return y
def get_x(data,seq):
    lista=[]
    for item in data:
        dictionary={}
        for j in seq:
            dictionary.update({j:item[j]})
        lista.append(dictionary)
    return lista


def get_y(data,posit,seq_pos,seq_neg):
    lista=[]
    for item in data:
        if item[posit].rstrip("\n") in seq_pos:
            lista.append(-1.0)
        else:
            lista.append(1.0)
    return lista