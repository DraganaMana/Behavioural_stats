# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:16:21 2019

@author: Dragana
"""

"""
intProd --> interval production
"""
import os
import numpy as np
import urllib.request
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt


#
"""
---- Normalized distributions
"""

x1=[]
x2=[]
x3=[]
pax = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
       '11', '12', '13', '14', '15', '16', '17', '18']
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
                    int_single = int_length[n,17]
                    int_single = float(int_single)
                    if j == '1.45':
                        int_single = int_single/1.45
                        x1.append(int_single)
                    if j == '2.9':
                        int_single = int_single/2.9
                        x2.append(int_single)
                    if j == '5.8':
                        int_single = int_single/5.8
                        x3.append(int_single)

# A function to truncate the number of decimal places
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier     
               
# Calculate the means and the standard deviations 
x1_arr = np.array(x1)
x2_arr = np.array(x2)
x3_arr = np.array(x3)
#
x1_mean = np.mean(x1_arr, axis=0)
x2_mean = np.mean(x2_arr, axis=0)
x3_mean = np.mean(x3_arr, axis=0)
#
x1_sd = truncate(np.std(x1_arr, axis=0), 3)
x2_sd = truncate(np.std(x2_arr, axis=0), 3)
x3_sd = truncate(np.std(x3_arr, axis=0), 3)

# Keeps all values between mean +/- 3sd
# x1
x1_final = [x for x in x1 if (x > x1_mean - 3*x1_sd)]
x1_final = [x for x in x1_final if (x < x1_mean + 3*x1_sd)]
# x2
x2_final = [x for x in x2 if (x > x2_mean - 3*x2_sd)]
x2_final = [x for x in x2_final if (x < x2_mean + 3*x2_sd)]
# x3
x3_final = [x for x in x3 if (x > x3_mean - 3*x3_sd)]
x3_final = [x for x in x3_final if (x < x3_mean + 3*x3_sd)]

# Calculate the SDs and Means of the data without the outliers
x1_fin_arr = np.array(x1_final)
x2_fin_arr = np.array(x2_final)
x3_fin_arr = np.array(x3_final)
#
x1_fin_mean = np.mean(x1_fin_arr, axis=0)
x2_fin_mean = np.mean(x2_fin_arr, axis=0)
x3_fin_mean = np.mean(x3_fin_arr, axis=0)
#
x1_fin_sd = truncate(np.std(x1_fin_arr, axis=0), 3)
x2_fin_sd = truncate(np.std(x2_fin_arr, axis=0), 3)
x3_fin_sd = truncate(np.std(x3_fin_arr, axis=0), 3)

#
#with sns.color_palette("Blues_r"):
#    plt.title("All interval productions of all subj of all blocks")
#    plt.ylabel("Density", fontsize=12) 
#    plt.xlabel("Normalized produced intervals", fontsize=12)         
#    sns.distplot(x1_final, hist=False, label='1.45 seconds, SD='+str(x1_fin_sd))
#    sns.distplot(x2_final, hist=False, label='2.9 seconds, SD='+str(x2_fin_sd))
#    sns.distplot(x3_final, hist=False, label='5.8 seconds, SD='+str(x3_fin_sd))

#%%
%matplotlib qt
with sns.color_palette("RdBu_r", n_colors=3): # "Blues_r"
    plt.title("Normalized interval productions of all participants", fontsize=14)
    plt.ylabel("Density", fontsize=14) 
    plt.xlabel("Produced intervals", fontsize=14)  
    plt.yticks([], [])         
    sns.distplot(x1_final, hist=False, color="steelblue", kde_kws={"shade": True}, 
                 label = '1.45 seconds, SD=' + str(x1_fin_sd))
    sns.distplot(x2_final, hist=False, color="thistle", kde_kws={"shade": True},
                 label = '2.9 seconds, SD=' + str(x2_fin_sd))
    sns.distplot(x3_final, hist=False, color="indianred", kde_kws={"shade": True},
                 label = '5.8 seconds, SD=' + str(x3_fin_sd))
    plt.grid(b=None)
plt.axvline(1, ymax=0.93, linestyle='--', linewidth=1.2, color='black', 
            label='Best estimation')
plt.xticks([1])
plt.xlim(right=2.4)
sns.despine()
plt.grid(b=None)
plt.legend()
    
    
    
    
    
    
    
    
