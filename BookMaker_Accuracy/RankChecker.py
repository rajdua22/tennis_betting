#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:12:17 2019

@author: rajdua

Description: Get percentage of better ranked playuers that win games. We use
this figure as a baseline to improve upon.
"""


import pandas as pd
import numpy as np

data = pd.read_csv('FinalData_Cleaned.csv',  encoding='latin-1', low_memory = False)
data = data.drop(columns  = ['Unnamed: 0'])

data = data.sort_values(by = data['y'])
data = data[0:37924]

X = data[['Best of', 'WRank', 'LRank', 'winner_hand', 'winner_ht', 'winner_age',
          'loser_hand', 'loser_ht', 'loser_age', 'SOT', 'Major', 'oddsw', 'oddsl',
          'CoppW', 'CoppL', 'MatchesPlayed', 'Clay', 'Hard', 'Grass', 'Carpet',
          'WInverseRank', 'LInverseRank', 'CWPW', 'CWPL', 'POSW', 'POSL', 'PINW',
          'PINL', 'GPW', 'DSWW', 'DSLW', 'CSWW', 'CSLW', 'TieWW', 'TieLW', 'FSIW', 'FSWW',
          'BPFGW', 'BPWGW', 'BPFCW', 'BPWCW', 'DSWL', 'DSLL', 'CSWL', 'CSLL',
          'TieWL', 'TieLL', 'APSGL', 'FSIL', 'FSWL', 'BPFGL', 'BPWGL', 'BPFCL',
          'BPWCL', 'TSSW', 'TSSL', 'PVPW', 'PVPL', 'PVPM', 'GPL']]

X = X.replace('Nan', np.nan).astype(float)

X = X.fillna(0)

y = data[['y']]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.3, random_state=0)
# df = pd.DataFrame(X_test)
df = pd.DataFrame(X)

def f(x):
    if (x[1] < x[2]):
        return 1
    else:
        return 0

ranks = df.apply(f, axis = 1)

total = df.shape[0]

print(ranks.sum()/total)

def g(x):
    if (x[11] < x[12]):
        return 1
    else:
        return 0
    
odds= df.apply(g, axis = 1)

print(odds.sum()/total)