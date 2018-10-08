# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:42:59 2018

@author: ratin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import random
import pandas as pd
from math import sqrt
import chain
from collections import Counter
style.use('fivethirtyeight')



def KNN(data,pred):
    vote=[]
    for i in data:
        for j in data[i]:
            j=np.array(j)
            euclidean_distance = np.linalg.norm(np.array(j)-np.array(pred))
        vote.append([euclidean_distance,i])
    return min(sorted(vote))[1]

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace = True)
df.drop(['id'],axis=1,inplace=True)
temp_list=[]
temp_list_4=[]
for i in range(len(df['clump_thickness']-1)):
    s=list(chain.from_iterable(df.iloc[i:i+1,:].values))
    if s[9]==2:
        s=s[:5]+s[6:]
        temp_list.append(s[:9])
    else:
        s=s[:5]+s[6:]
        temp_list_4.append(s[:9])
d={2:temp_list,4:temp_list_4[:-1]}
pred = temp_list_4[-1]
print(KNN(d,pred))
