# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:10:37 2019

@author: Dragana
"""

"""
I used this for plotting the data from the pilots. The final pax are just commented out. 
"""
import os
import numpy as np
import urllib.request
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

"""
------- Violin + swarm plots of normalized int productions per block
"""
df = pd.DataFrame({'block': [], 'xnorm': []})
#pax = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
#       '11', '12', '13', '14', '15', '16', '17', '18']
pax = ['1','2','1369','7','8']
xnorm = 0
for i in range(1,7): # 1, 2, 3, 4, 5, 6
    for k in pax:
        data_path = 'C:/Users/Dragana/Documents/MATLAB/m_M1_internship/Multifracts/Data/'
#        data_path = 'D:/ScaledTime/Matlab data/'
        data_path_subj = os.path.join(data_path, 'ScaledTime'+k)
        os.chdir(data_path_subj)
        for j in ['1.45','2.9','5.8']:
            name = 'ScaledTime_Play_subj_'+ k +'_bl_' + str(i) + '_int_' + j + '.txt'
            if os.path.isfile(name) and os.path.exists(name):
                int_length=np.genfromtxt(name, dtype='str')
                for n in range(1,16):
                    int_single = int_length[n,15] # 17
                    int_single = float(int_single)
                    if int_single < 20:
                        xnorm = int_single/float(j)
                        df = df.append({'block': str(i), 'xnorm': xnorm}, ignore_index=True)

x_mean = df.mean(axis=0)[1]
x_sd = df.std(axis=0)[0]
minlim = (x_mean - 3*x_sd)
maxlim = (x_mean + 3*x_sd)
df1 = df.drop(df[df.xnorm >= maxlim].index)
df2 = df1.drop(df1[df1.xnorm <= minlim].index)

#%%
#with sns.color_palette("RdBu_r", n_colors=6):   
linija = df = pd.DataFrame({'x': [0,1,2,3,4,5,6], 'y': [1,1,1,1,1,1,1]})

with sns.cubehelix_palette(6, gamma=0.3, dark=0.35, hue=0.9):
#with sns.cubehelix_palette(rot=-.4, gamma=0.35, dark=0.25, hue=0.95):
    plt.title("Normalized interval productions per block, per participant", fontsize=14) 
    sns.boxplot(x="block", y="xnorm", data=df2, showfliers=False, showbox=False, whis=[25,75]) # whis=[2.5,97.5]
    ax = sns.violinplot(x="block", y="xnorm", data=df2, inner="box")
    plt.ylabel("Normalized interval production", fontsize=14) 
    plt.xlabel("Block number", fontsize=14) 
    sns.despine() # remove the top and right frame
#ax = sns.swarmplot(x="block", y="xnorm", data=df,
#                   color="black", edgecolor="gray")  
#%% Good example
    
#f, (ax1, ax2, ax3) = plt.subplots(
#    1, 3, figsize=(12, 4), sharey=True)
#sns.barplot(x='sex', y='tip', data=tips, ax=ax1)
#sns.violinplot(x='sex', y='tip', data=tips, ax=ax2)
#sns.swarmplot(x='sex', y='tip', data=tips, ax=ax3)
    
