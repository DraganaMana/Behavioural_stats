# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:55:58 2019

@author: Dragana
"""
"""
This code is also used only on the pilots data. 
"""

"""
Code plan:
    - put all the matrices from the pax 
            eg. m1 = [1.45 2.9 5.8; .....]
    - if cur_subj = '1002', use matrix m1
            or something with a for loop iterating through the subjects and taking the 
            correct matrix of intervals
    - classify the interval between:
            1. Preceeded by long
            2. Preceeded by the same
            3. Preceeded by short interval
        and assign the values of the produced intervals in the correct list: x1, x2 or x3
"""
#################################################################################


"""
------- Sequential effects - old code
"""
x1=[]
x2=[]
x3=[]
x4=[]
x5=[]
x6=[]
x_short=[]
x_long=[]
cur_int='5.8'
outliers_lim=12
for i in range(1,7): # 1, 2, 3, 4, 5, 6
        for l in ['1', '2', '7', '8', '1369']:
            data_path = 'C:/Users/Dragana/Documents/MATLAB/m_M1_internship/Multifracts/Data/'
            data_path_subj = os.path.join(data_path, 'ScaledTime'+l)
            os.chdir(data_path_subj)
            name = 'ScaledTime_Play_subj_' + l + '_bl_' + str(i) + '_int_' + cur_int + '.txt'
            if os.path.isfile(name) and os.path.exists(name):
                int_length=np.genfromtxt(name, dtype='str')
                for k in range(1,16):
                    int_single = int_length[k,15]
                    int_single = float(int_single)
                    if int_single < outliers_lim:
                        if i == 1:
                            x1.append(int_single)
                            x_long.append(int_single)
                            
                        elif i == 2:
                            x2.append(int_single)
                            x_short.append(int_single)
                            
                        elif i == 3:
                            x3.append(int_single)
                            x_long.append(int_single)
                            
                        elif i == 4:
                            x4.append(int_single)
                            x_short.append(int_single)
                            
                        elif i == 5:
                            x5.append(int_single)
#                            x_short.append(int_single)
                            
                        elif i == 6:
                            x6.append(int_single)
                            x_long.append(int_single)
                    else:
                        pass

# A function to truncate the number of decimal places
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

######################## 2.9
x_s_sd = truncate(np.std(x_short),3)
x_l_sd = truncate(np.std(x_long),3)
x3_sd = truncate(np.std(x3),3)

#with sns.color_palette("GnBu"):
with sns.diverging_palette(145, 10, s=85, l=25, n=3):
    plt.title(cur_int+" second interval productions across blocks")
    plt.ylabel("Density", fontsize=12) 
    plt.xlabel("Produced intervals", fontsize=12) 
#    plt.xlim(0,30)
    sns.distplot(x_short, rug=True, hist=False, color='green', label='Blocks 1, 4, 5 - after 1.45s, SD='+str(x_s_sd))
    sns.distplot(x3, rug=True, hist=False, color='pink', label='Block 3 - after 2.9s, SD='+str(x3_sd))
    sns.distplot(x_long, rug=True, hist=False, color='maroon', label='Blocks 2, 6 - after 5.8s, SD='+str(x_l_sd))


######################## 1.45
x_s_sd = truncate(np.std(x_short),3)
x_l_sd = truncate(np.std(x_long),3)
x1_sd = truncate(np.std(x1),3)

with sns.diverging_palette(145, 10, s=85, l=25, n=3):
    plt.title(cur_int+" second interval productions across blocks")
    plt.ylabel("Density", fontsize=12) 
    plt.xlabel("Produced intervals", fontsize=12) 
    sns.distplot(x_short, rug=True, hist=False, color='green', label='Blocks 4, 6 - after 2.9s, SD='+str(x_s_sd))
    sns.distplot(x3, rug=True, hist=False, color='pink', label='Block 1 - first interval, SD='+str(x1_sd))
    sns.distplot(x_long, rug=True, hist=False, color='maroon', label='Blocks 2, 3, 5 - after 5.8s, SD='+str(x_l_sd))

######################## 5.8
x_s_sd = truncate(np.std(x_short),3)
x_l_sd = truncate(np.std(x_long),3)
x5_sd = truncate(np.std(x5),3)

with sns.diverging_palette(145, 10, s=85, l=25, n=3):
    plt.title(cur_int+" second interval productions across blocks")
    plt.ylabel("Density", fontsize=12) 
    plt.xlabel("Produced intervals", fontsize=12) 
    sns.distplot(x_short, rug=True, hist=False, color='green', label='Blocks 2, 4 - after 1.45s, SD='+str(x_s_sd))
    sns.distplot(x3, rug=True, hist=False, color='pink', label='Block 5 - after 5.8s, SD='+str(x1_sd))
    sns.distplot(x_long, rug=True, hist=False, color='maroon', label='Blocks 1, 3, 6 - after 2.9s, SD='+str(x_l_sd))


