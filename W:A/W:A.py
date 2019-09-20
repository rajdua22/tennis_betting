#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 00:10:53 2019

@author: rajdua
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



index1 = 0
wealth_alpha = np.zeros(1000)


maxBet = np.inf

for i in np.linspace(0.001, 1, 1000):
    wealth = 1
    fraction = i
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
        
    wealth_alpha[index1] = wealth
    print(index1)
    index1 += 1
    
WA_inf = pd.DataFrame(wealth_alpha)
WA_inf.index = np.linspace(0.001, 1, 1000)

WA.columns = {('wealth')}

WA.index.name = 'alpha'
 
WA_inf.to_csv('WA(inf).csv')
    
plt.plot(WA_inf)

plt.title('W/A_1_inf')
plt.xlabel('Alpha')
plt.ylabel('Wealth')

   



    


