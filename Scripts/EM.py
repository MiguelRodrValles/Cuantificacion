# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:14:37 2021

@author: mrvalles
"""
def EM(test_x_p,test_y,train_y,niter):
    import numpy as np
    (unique, counts) = np.unique(train_y, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    P_ini=
    P_act=P_ini
    for i in range(niter):
        P_ant=np.zeros(len(test_x_p))
        for k in len(test_x_p):
            P_ant[i]=((P_act/P_ini)*test_x_p[k])/((P_act/P_ini)*test_x_p[k]+((1-P_act)/(1-P_ini))*(1-test_x_p[k]))
        P_act=sum(P_ant)/len(P_ant)
    
    