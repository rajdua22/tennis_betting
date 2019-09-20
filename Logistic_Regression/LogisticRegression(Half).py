#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:17:50 2019

@author: rajdua

Description: Use logistic regression to train on data from 2002-2013 and produce
an output file with predictions for 2014+. Uses entire feature set. Doesn't artifically
double dataset.
"""


import pandas as pd
import numpy as np

data = pd.read_csv('FinalData_Cleaned.csv',  encoding='latin-1', low_memory = False)
data = data.drop(columns  = ['Unnamed: 0'])

data = data.sort_values(by = ['Date', 'players'])

data = data.reset_index()

a = data.iloc[::4, :]

b = data.iloc[3::4, :]

data = pd.concat([a, b])
    


data['Date'] = pd.to_datetime(data['Date'])

date = pd.to_datetime('2013')

dataLate = data[(data.Date > date)]

data['Date'] = pd.to_datetime(data['Date'])

date = pd.to_datetime('2002')

data = data[(data.Date > date)]

date = pd.to_datetime('2013')

data = data[(data.Date < date)]

X = data[['Best of', 'WRank', 'LRank', 'winner_hand', 'winner_ht', 'winner_age',
          'loser_hand', 'loser_ht', 'loser_age', 'SOT', 'Major', 'oddsw', 'oddsl',
          'Clay', 'Hard', 'Grass', 'Carpet', 'Wcomb', 'Lcomb', 'RTrendW', 'RTrendL',
          'WInverseRank', 'LInverseRank', 'CWPW', 'CWPL', 'POSW', 'POSL', 'PINW',
          'PINL', 'GPW', 'DSWW', 'DSLW', 'CSWW', 'CSLW', 'TieWW', 'TieLW', 'FSIW', 'FSWW',
          'BPFGW', 'BPWGW', 'BPFCW', 'BPWCW', 'DSWL', 'DSLL', 'CSWL', 'CSLL',
          'TieWL', 'TieLL', 'APSGL', 'FSIL', 'FSWL', 'BPFGL', 'BPWGL', 'BPFCL',
          'BPWCL', 'TSSW', 'TSSL', 'GPL']]

X = X.replace('Nan', np.nan).astype(float)
# X = X.fillna(0)

X = X.fillna(X.mean())

y = data[['y']]

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test2 = X_test
X_test = scaler.transform(X_test)
logreg = LogisticRegressionCV(penalty = 'elasticnet', solver = 'saga', max_iter = 50000, 
                              l1_ratios = [0.1, 0.1], Cs = 10, cv = 5)
logreg.fit(X_train, y_train.ravel())


print('Accuracy of logistic regression classifier on test set:')
print(logreg.score(X_test, y_test))

X_scaled = scaler.transform(X)
predictions = logreg.predict_proba(X_scaled)

print('Accuracy of logistic regression classifier on entire dataset:')
print(logreg.score(X_scaled, y))

a = logreg.coef_

coef_df = pd.DataFrame(np.transpose(a))
coef_df['names'] = X.columns

coef_df.to_csv('LogRegCoef3.csv')


betting = data[['oddsw', 'oddsl']]
betting['predictions1'] = predictions[:,1]
betting['result'] = data['y']
betting['predictions2'] = predictions[:,0]

betting['impliedodds1'] = betting['oddsw'].apply(lambda x: 1/ x)
betting['impliedodds2'] = betting['oddsl'].apply(lambda x: 1/ x)
betting['date'] = data['Date']
betting['Player1'] = data['Winner']
betting['Player2'] = data['Loser']
betting['Rank1'] = data['WRank']
betting['Rank2'] = data['LRank']

betting = betting.rename(columns  = {'oddsw':'odds1', 'oddsl':'odds2'})

betting.to_csv('Predictions(Log_Reg)3.csv')

data['impliedodds1'] = data['oddsw'].apply(lambda x: 1/ x)
data['impliedodds2'] = data['oddsl'].apply(lambda x: 1/ x)
data['predictions1'] = predictions[:,1]
data['predictions2'] = predictions[:,0]

data.to_csv('Data_and_predictions(Log_Reg)3.csv')


X_test2 = dataLate[['Best of', 'WRank', 'LRank', 'winner_hand', 'winner_ht', 'winner_age',
          'loser_hand', 'loser_ht', 'loser_age', 'SOT', 'Major', 'oddsw', 'oddsl',
          'Clay', 'Hard', 'Grass', 'Carpet', 'Wcomb', 'Lcomb', 'RTrendW', 'RTrendL',
          'WInverseRank', 'LInverseRank', 'CWPW', 'CWPL', 'POSW', 'POSL', 'PINW',
          'PINL', 'GPW', 'DSWW', 'DSLW', 'CSWW', 'CSLW', 'TieWW', 'TieLW', 'FSIW', 'FSWW',
          'BPFGW', 'BPWGW', 'BPFCW', 'BPWCW', 'DSWL', 'DSLL', 'CSWL', 'CSLL',
          'TieWL', 'TieLL', 'APSGL', 'FSIL', 'FSWL', 'BPFGL', 'BPWGL', 'BPFCL',
          'BPWCL', 'TSSW', 'TSSL', 'GPL']]

X_test2 = X_test2.replace('Nan', np.nan).astype(float)
# X = X.fillna(0)

X_test2 = X_test2.fillna(X_test2.mean())

ylate = dataLate[['y']]

X_testLate = scaler.transform(X_test2)


predictions = logreg.predict_proba(X_testLate)
X_test2 = pd.DataFrame(X_test2, columns = X.columns)
betting2 = X_test2[['oddsw', 'oddsl']]
betting2['predictions1'] = predictions[:,1]
betting2['result'] = ylate
betting2['predictions2'] = predictions[:,0]
betting2['impliedodds1'] = betting2['oddsw'].apply(lambda x: 1/ x)
betting2['impliedodds2'] = betting2['oddsl'].apply(lambda x: 1/ x)
betting2['date'] = dataLate['Date']
betting2['Player1'] = dataLate['Winner']
betting2['Player2'] = dataLate['Loser']
betting2['Rank1'] = X_test2['WRank']
betting2['Rank2'] = X_test2['LRank']
betting2 = betting2.rename(columns  = {'oddsw':'odds1', 'oddsl':'odds2'})
betting2.to_csv('PredictionsTest(Log_Reg)3.csv')





    