# -*- coding: utf-8 -*-
"""
Created on Mon May 20 18:01:59 2019

@author: Dragana
"""
import os
import numpy as np
import urllib.request
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

# Check PLAY
ITI1=[]
ITI2=[]
ITI3=[]
#
GO1=[]
GO2=[]
GO3=[]
#
#pax = ['3', '6', '7', '10', '11', '111']
pax = ['17']
for i in range(1,7): # 1, 2, 3, 4, 5, 6
    for k in pax:
#        data_path = 'C:/Users/Dragana/Documents/MATLAB/m_M1_internship/Multifracts/Data/'
        data_path = 'D:/ScaledTime/Matlab data/'
        data_path_subj = os.path.join(data_path, 'ScaledTime'+k)
        os.chdir(data_path_subj)
        for j in ['1.45','2.9','5.8']:
            name = 'ScaledTime_Play_subj_'+ k +'_bl_' + str(i) + '_int_' + j + '.txt'
            if os.path.isfile(name) and os.path.exists(name):
                int_length=np.genfromtxt(name, dtype='str')
                for n in range(1,16):
                    ITIstart = float(int_length[n,6])
                    ITIend = float(int_length[n,7])
                    GOsign = float(int_length[n,9])
                    PressWait = float(int_length[n,10])
                    ITI = ITIend - ITIstart
                    GOlength = PressWait - GOsign
                    if j == '1.45':
                        ITI1.append(i)
                        ITI1.append(j)
                        ITI1.append(ITI)
                        GO1.append(i)
                        GO1.append(j)
                        GO1.append(GOlength)
                    if j == '2.9':
                        ITI2.append(i)
                        ITI2.append(j)
                        ITI2.append(ITI)
                        GO2.append(i)
                        GO2.append(j)
                        GO2.append(GOlength)
                    if j == '5.8':
                        ITI3.append(i)
                        ITI3.append(j)
                        ITI3.append(ITI)
                        GO3.append(i)
                        GO3.append(j)
                        GO3.append(GOlength)
                        
ITIs = [ITI1, ITI2, ITI3]
for iti, j in enumerate(ITIs):
    print('ITI'+str(iti+1))
    for i in j:
        if float(i)>2.5 or float(i)<1.5:
            print(i)

GOs = [GO1, GO2, GO3]
for go, j in enumerate(GOs): 
    print('GO'+str(go+1))
    for i in j:
        if float(i)>0.334:
            print(i)

# Check REPLAY
ITI1=[]
ITI2=[]
ITI3=[]
#
GO1=[]
GO2=[]
GO3=[]
#
#pax = ['3', '6', '7', '10', '11', '111']
pax = ['17']
for i in range(1,7): # 1, 2, 3, 4, 5, 6
    for k in pax:
#        data_path = 'C:/Users/Dragana/Documents/MATLAB/m_M1_internship/Multifracts/Data/'
        data_path = 'D:/ScaledTime/Matlab data/'
        data_path_subj = os.path.join(data_path, 'ScaledTime'+k)
        os.chdir(data_path_subj)
        for j in ['1.45','2.9','5.8']:
            name = 'ScaledTime_Replay_subj_'+ k +'_bl_' + str(i) + '_int_' + j + '.txt'
            if os.path.isfile(name) and os.path.exists(name):
                int_length=np.genfromtxt(name, dtype='str')
                for n in range(1,16):
                    ITIstart = float(int_length[n,4])
                    GOsign = float(int_length[n,5])
                    PressWait = float(int_length[n,6])
                    ITI = GOsign - ITIstart
                    GOlength = PressWait - GOsign
                    if j == '1.45':
                        ITI1.append(i)
                        ITI1.append(j)
                        ITI1.append(ITI)
                        GO1.append(i)
                        GO1.append(j)
                        GO1.append(GOlength)
                    if j == '2.9':
                        ITI2.append(i)
                        ITI2.append(j)
                        ITI2.append(ITI)
                        GO2.append(i)
                        GO2.append(j)
                        GO2.append(GOlength)
                    if j == '5.8':
                        ITI3.append(i)
                        ITI3.append(j)
                        ITI3.append(ITI)
                        GO3.append(i)
                        GO3.append(j)
                        GO3.append(GOlength)