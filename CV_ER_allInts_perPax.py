# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:23:49 2019

@author: Dragana
"""
import os
import numpy as np
import csv


#%% Calculate the SDs and means for all subjects together
    # in order to exclude the mean+/-3*SDs outliers
x1=[]
x2=[]
x3=[]


pax = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
       '11', '12', '13', '14', '15', '16', '17', '18']
for k in pax:
    for i in range(1,7): # 1, 2, 3, 4, 5, 6
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
                        x1.append(int_single)
                    if j == '2.9':
                        x2.append(int_single)
                    if j == '5.8':
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
x1_all_mean = np.mean(x1_arr, axis=0)
x2_all_mean = np.mean(x2_arr, axis=0)
x3_all_mean = np.mean(x3_arr, axis=0)
#
x1_all_sd = truncate(np.std(x1_arr, axis=0), 3)
x2_all_sd = truncate(np.std(x2_arr, axis=0), 3)
x3_all_sd = truncate(np.std(x3_arr, axis=0), 3)


#%% Calculate the CV and ER per subject, per interval
x1=[]
x2=[]
x3=[]

cvs_ints = []
ers_ints = []

rows = []

pax = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
       '11', '12', '13', '14', '15', '16', '17', '18']



for k in pax:
    for i in range(1,7): # 1, 2, 3, 4, 5, 6
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
                        x1.append(int_single)
                    if j == '2.9':
                        x2.append(int_single)
                    if j == '5.8':
                        x3.append(int_single)
                        
    # Keeps all values between mean +/- 3sd
    # x1
    x1_final = [x for x in x1 if (x > x1_all_mean - 3*x1_all_sd)]
    x1_final = [x for x in x1_final if (x < x1_all_mean + 3*x1_all_sd)]
    # x2
    x2_final = [x for x in x2 if (x > x2_all_mean - 3*x2_all_sd)]
    x2_final = [x for x in x2_final if (x < x2_all_mean + 3*x2_all_sd)]
    # x3
    x3_final = [x for x in x3 if (x > x3_all_mean - 3*x3_all_sd)]
    x3_final = [x for x in x3_final if (x < x3_all_mean + 3*x3_all_sd)] # changed this to 2, it was too long

    # Calculate the SDs and Means of the data per subject w/o the outliers

    x1_mean = np.mean(x1_final)
    x2_mean = np.mean(x2_final)
    x3_mean = np.mean(x3_final)
    
    x1_sd = np.std(x1_final)
    x2_sd = np.std(x2_final)
    x3_sd = np.std(x3_final)
    
    x1_target = 1.45
    x2_target = 2.9
    x3_target = 5.8    

    # Calculate the CVs  
    cv1 = x1_sd / x1_mean
    cv2 = x2_sd / x2_mean
    cv3 = x3_sd / x3_mean
    cv_ints = (cv1 + cv2 + cv3)/3
    
    cvs_ints.append(cv_ints)
    
    # Calculate the ERs
    er1 = x1_mean / x1_target
    er2 = x2_mean / x2_target
    er3 = x3_mean / x3_target
    er_ints = (er1 + er2 + er3)/3
    
    ers_ints.append(er_ints)


    rows.append([k,     
            cv_ints,
            er_ints])
    
    
with open('CV_ER_pax.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Subject', 'CV_ints', 'ER_ints'])
    for r in rows:
        filewriter.writerow(r)
    

#%%
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
matplotlib.style.use('ggplot')

# pick_cv = cvs1 # cvs2, cvs3, cvs_ints
# pick_er = ers1

plt.scatter(cvs_ints, ers_ints)
plt.xlabel("CV")
plt.ylabel("ER")
plt.show()   

#plt.scatter(x, y, s, c="g", alpha=0.5, marker=r'$\clubsuit$',
#            label="Luck")
#plt.xlabel("Leprechauns")
#plt.ylabel("Gold")
#plt.legend(loc='upper left') 

#%% Linear regression

#https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
#https://realpython.com/linear-regression-in-python/
#https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
#https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


x = np.array(cvs_ints).reshape((-1, 1))
y = np.array(ers_ints)

model = LinearRegression().fit(x, y)

#  the coefficient of determination (𝑅²) with .score() called on model
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

# Plot outputs
plt.scatter(x, y,  color='black')
plt.plot(x, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#%% Plot the CV-ER dots + the linear regression
import pygal


xl = np.ndarray.tolist(x) # x from the linear regress is an array, and we get list of lists
import itertools
xll = list(itertools.chain.from_iterable(xl)) # we have list of lists, and we need a flat list
tups = list(zip(xll, y_pred)) # we need a list of tuples for plotting

from pygal.style import BlueStyle, CleanStyle

# MAKING MY OWN STYLE 
from pygal.style import Style
custom_style = Style(
  background='transparent',
  plot_background='transparent',
  foreground='#000000',
  foreground_strong='#000000',
  foreground_subtle='#000000',
  opacity='.6',
  opacity_hover='.9',
  transition='400ms ease-in',
  colors=('#08cc9e', '#4860db', '#10c6d3', '#0a2628'),
  title_font_size = 20,
  label_font_size = 15,
  legend_font_size = 15)

xy_chart = pygal.XY(stroke=False, style=custom_style, 
                    title=u'Accuracy and Precission Linear Regression', 
                    x_title='Precision [Coefficient of Variation - CV]',
                    y_title='Accuracy [Error Rate - ER]',
                    legend_at_bottom=True,
                    dots_size = 5)

#xy_chart.background = '#ffffff'
#xy_chart.background = 'transparent'

xy_chart.title = 'Correlation'

t = list(zip(cvs_ints, ers_ints))
xy_chart.add('CV-ER-ints', [t[0],  t[1],  t[2],  t[3],  t[4],  t[5], 
                            t[6],  t[7],  t[8],  t[9],  t[10], t[11],  
                            t[12], t[13], t[14], t[15], t[16], t[17]])

xy_chart.add('Linear regression', [min(tups), max(tups)], stroke=True, stroke_style={'width': 3})
xy_chart.render()
xy_chart.render_to_file('CV_ER_lin-regress_allInts.svg')
#xy_chart.render_to_png('D:/ScaledTime/Matlab data/Behavioural_stats/CV_ER_lin-regress.png')


#%% Pearson correlation coefficient and p-value for testing non-correlation.

#https://www.texasgateway.org/resource/124-testing-significance-correlation-coefficient-optional
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

import scipy 
from scipy import stats

r, p = stats.pearsonr(cvs_ints, ers_ints)
#(0.6345269914848317, 2.57395430278365e-07)
print(r,p)

"""
stats.pearsonr(cvs1, ers1)
Out[50]: (-0.2508172157924429, 0.3154272113055784)

stats.pearsonr(cvs2, ers2)
Out[51]: (0.8599347534276047, 4.754429686293356e-06)

stats.pearsonr(cvs3, ers3)
Out[52]: (0.07671094282825852, 0.7622420525282299)
"""