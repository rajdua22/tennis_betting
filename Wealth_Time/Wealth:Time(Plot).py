#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 00:10:53 2019

@author: rajdua

Use Kelly formula to bet on games and graph
wealth/alpha graphs for different max bet sizes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
import math

data = pd.read_csv('PredictionsTest(NN).csv', encoding = "ISO-8859-1", low_memory=False)
data['date'] = pd.to_datetime(data['date'])

def f(x):
    return frozenset({x.Player1, x.Player2})

data['players'] = data.apply(f, axis = 1)

data = data.sort_values(by = ['date', 'players'])

a = data.iloc[::2, :]

b = data.iloc[1::2, :]

a = a.reset_index()
b = b.reset_index()

a['oddsTotal'] = a.impliedodds1 + a.impliedodds2
a['impliedodds1'] = a.impliedodds1 / a.oddsTotal
a['impliedodds2'] = a.impliedodds2 / a.oddsTotal


c = (a.predictions1 + b.predictions2) / 2
d = (b.predictions1 + a.predictions2) / 2

a['predictions1'] = c
a['predictions2'] = d

a['edge1'] = a.predictions1 - a.impliedodds1
a['edge2'] = a.predictions2 - a.impliedodds2




maxBet = np.inf
wealth = 1
fraction = 0.69
wealthot = np.zeros(a.shape[0])


def kellyFormula(x):
    if x.edge1 > x.edge2:
        bet = x.odds1 - 1
        p = x.predictions1
    else:
        bet = x.odds2 - 1
        p = x.predictions2
    temp = bet + 1
    temp2 = p * temp -1
    if bet > 0:
        return fraction * (temp2 / bet)
    else:
        return 0
        
a['kelly'] = a.apply(kellyFormula, axis = 1)
    
for index, row in a.iterrows():
    kelly = row['kelly']
    odds = row['odds1'] - 1
    if kelly > 0:
        if row['edge1'] > 0:
            if (wealth * kelly) > maxBet:
                wealth += maxBet * odds
            else:
                wealth += (wealth * kelly * odds)
        if row['edge2'] > 0:
            if (wealth * kelly) > maxBet:
                wealth -= maxBet
            else:
                wealth -= (wealth * kelly)
    wealthot[index] = wealth
a['Wealth'] = wealthot


y = a['Wealth']
x = a ['date']

plt.figure(0)
plt.plot(x, y)

plt.figure(1)

plt.plot(x, y.apply(lambda x: math.log(x)))

a.to_csv('WealthOverTime.csv')


   



    


