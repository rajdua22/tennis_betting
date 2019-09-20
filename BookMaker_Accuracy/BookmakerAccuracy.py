#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:31:51 2019

@author: rajdua

Description: Plot graphs showing accuracy of bookmakers.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('PredictionsTest(NN).csv',  encoding='latin-1', low_memory = False)
data = data.drop(columns  = ['Unnamed: 0'])

data['bmprob1'] = data['impliedodds1'] / (data['impliedodds1'] + data['impliedodds2'])
data['bmprob2'] = data['impliedodds2'] / (data['impliedodds1'] + data['impliedodds2'])

j = 0.01

cuts = np.arange(0,1+ j,j)

data['cut1'] = pd.cut(data.bmprob1, cuts)
data['cut2'] = pd.cut(data.bmprob2, cuts)

view = data.groupby('cut1').mean()['result']

plt.plot(range(len(view)),view, 'b')

plt.plot(range(len(view)),np.arange(j,1+j, j), 'r')

plt.xlabel('Implied odds')
plt.ylabel('Actual win percentage')


data['cut1Us'] = pd.cut(data.predictions1, cuts)
data['cut2Us'] = pd.cut(data.predictions2, cuts)

view2 = data.groupby('cut1Us').mean()['result']

plt.plot(range(len(view2)),view2, 'g')
plt.plot(range(len(view2)),np.arange(0,1, j), 'g')
