#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:29:15 2019

@author: rajdua

Use the fractional Kelly formula to show growth of wealth
by simulating betting on games.
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

wealthA = np.zeros((25,100))
step1 = np.geomspace(1e-4, 1e20, 25)
step = np.linspace(0.01, 1, 100)
index1 = 0
beta = np.zeros(25)

for i in step1:
    maxBet = i
    index2 = 0
    beta[index2] = i

    for j in step:
        fraction = j
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
        
        # wealthot = np.zeros(a.shape[0])
        wealth  = 1
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
            # wealthot[index] = wealth
        # a['Wealth'] = wealthot
        wealthA[index1, index2] = wealth
        index2+= 1
    index1+= 1
    print (index1)
    
WA = pd.DataFrame(wealthA)
WA.columns = np.linspace(0.01, 1, 100)
WA.index = np.geomspace(1e-4, 1e13, 25)
 
WA.to_csv('WA.csv')
    
plt.plot(step, wealthA)
    

y = a['Wealth']
x = a ['date']

plt.plot(x, y)

plt.plot(x, y.apply(lambda x: math.log(x)))


a = a[a.kelly > 0]

def isFav(x):
    if x.impliedodds1 > x.impliedodds2: 
        return 1 
    else: 
        return 2
    
a['favorite'] = a.apply(isFav, axis = 1)

def OurFav(x):
    if x.edge1 > x.edge2: 
        return 1 
    else: 
        return 2
    
a['OurBet'] = a.apply(OurFav, axis = 1)

a['BetOnFavorite'] = 1*(a.OurBet == a.favorite)


def odds(x):
    if x.OurBet == 1:
        return x.odds1
    if x.OurBet == 2:
        return x.odds2
    
a['ourodds'] = a.apply(odds, axis = 1)

a['OurEdge'] = a['kelly'] * (a.ourodds - 1)

plt.hist(a.OurEdge, bins = 50)


a.to_csv('WealthOverTime.csv')
    
   



    


