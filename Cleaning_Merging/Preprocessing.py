#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:52:54 2019

@author: rajdua

Description: Loads in dataset and completes basic data cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

data = pd.read_csv('Final Merged.csv', encoding = "ISO-8859-1", low_memory=False)
data['matches'] = 1
data['Date'] = pd.to_datetime(data['Date'])


def DSW(x):
    if (x > 0) & (x < 3):
        return 1;
    else:
        return 0
    
def CSW(x):
    if (x > 2) & (x < 6):
        return 1;
    else:
        return 0

def Tie(x,y):
    if (x == 7) & (y == 6):
        return 1
    else:
        return 0
    
data['DSW_W1'] = data['L1'].apply(DSW)
data['DSW_W2'] = data['L2'].apply(DSW)
data['DSW_W3'] = data['L3'].apply(DSW)
data['DSW_W4'] = data['L4'].apply(DSW)
data['DSW_W5'] = data['L5'].apply(DSW)

data['DSW'] = data.apply(lambda row: row.DSW_W1 + row.DSW_W2 + row.DSW_W3 + row.DSW_W4 + row.DSW_W5, axis=1)

data['DSW_W1'] = data['L1'].apply(CSW)
data['DSW_W2'] = data['L2'].apply(CSW)
data['DSW_W3'] = data['L3'].apply(CSW)
data['DSW_W4'] = data['L4'].apply(CSW)
data['DSW_W5'] = data['L5'].apply(CSW)

data['CSW'] = data.apply(lambda row: row.DSW_W1 + row.DSW_W2 + row.DSW_W3 + row.DSW_W4 + row.DSW_W5, axis=1)

data['DSW_W1'] = data['W1'].apply(DSW)
data['DSW_W2'] = data['W2'].apply(DSW)
data['DSW_W3'] = data['W3'].apply(DSW)
data['DSW_W4'] = data['W4'].apply(DSW)
data['DSW_W5'] = data['W5'].apply(DSW)

data['DSL'] = data.apply(lambda row: row.DSW_W1 + row.DSW_W2 + row.DSW_W3 + row.DSW_W4 + row.DSW_W5, axis=1)

data['DSW_W1'] = data['W1'].apply(CSW)
data['DSW_W2'] = data['W2'].apply(CSW)
data['DSW_W3'] = data['W3'].apply(CSW)
data['DSW_W4'] = data['W4'].apply(CSW)
data['DSW_W5'] = data['W5'].apply(CSW)

data['CSL'] = data.apply(lambda row: row.DSW_W1 + row.DSW_W2 + row.DSW_W3 + row.DSW_W4 + row.DSW_W5, axis=1)

data['DSW_W1'] = data['L1'].apply(DSW)
data['DSW_W2'] = data['L2'].apply(DSW)
data['DSW_W3'] = data['L3'].apply(DSW)
data['DSW_W4'] = data['L4'].apply(DSW)
data['DSW_W5'] = data['L5'].apply(DSW)

data['DSW'] = data.apply(lambda row: row.DSW_W1 + row.DSW_W2 + row.DSW_W3 + row.DSW_W4 + row.DSW_W5, axis=1)

data['DSW_W1'] = data['L1'].apply(CSW)
data['DSW_W2'] = data['L2'].apply(CSW)
data['DSW_W3'] = data['L3'].apply(CSW)
data['DSW_W4'] = data['L4'].apply(CSW)
data['DSW_W5'] = data['L5'].apply(CSW)

data['CSW'] = data.apply(lambda row: row.DSW_W1 + row.DSW_W2 + row.DSW_W3 + row.DSW_W4 + row.DSW_W5, axis=1)

data['DSW_W1'] = data.apply(lambda row: Tie(row['W1'], row['L1']), axis = 1)
data['DSW_W2'] = data.apply(lambda row: Tie(row['W2'], row['L2']), axis = 1)
data['DSW_W3'] = data.apply(lambda row: Tie(row['W3'], row['L3']), axis = 1)
data['DSW_W4'] = data.apply(lambda row: Tie(row['W4'], row['L4']), axis = 1)
data['DSW_W5'] = data.apply(lambda row: Tie(row['W5'], row['L5']), axis = 1)

data['TieW'] = data.apply(lambda row: row.DSW_W1 + row.DSW_W2 + row.DSW_W3 + row.DSW_W4 + row.DSW_W5, axis=1)

data['DSW_W1'] = data.apply(lambda row: Tie(row['L1'], row['W1']), axis = 1)
data['DSW_W2'] = data.apply(lambda row: Tie(row['L2'], row['W2']), axis = 1)
data['DSW_W3'] = data.apply(lambda row: Tie(row['L3'], row['W3']), axis = 1)
data['DSW_W4'] = data.apply(lambda row: Tie(row['L4'], row['W4']), axis = 1)
data['DSW_W5'] = data.apply(lambda row: Tie(row['L5'], row['W5']), axis = 1)

data['TieL'] = data.apply(lambda row: row.DSW_W1 + row.DSW_W2 + row.DSW_W3 + row.DSW_W4 + row.DSW_W5, axis=1)


data['SetsCompleted'] = data['DSW'] + data['DSL'] + data['CSW'] + data['CSL'] + data['TieW'] + data['TieL']



# '1' indicates quarterfinal or more important match
def stage(x):
    if (x == 'F') | (x == 'SF') | (x == 'QF'):
        return 1;
    else:
        return 0
    
data['SOT'] = data['round'].apply(stage)

    
data['Major'] = (data['Series'] == 'Grand Slam').astype(int)

data = data.drop(columns = ['Unnamed: 0', 'DSW_W1', 'DSW_W2', 'DSW_W3', 'DSW_W4', 'DSW_W5'])

data['games'] = data[['W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4','L4', 'W5', 'L5']].sum(axis = 1)

data['oddsw'] = data[['CBW', 'GBW', 'IWW',  'SBW', 
            'B365W', 'B&WW', 'EXW', 'PSW', 'UBW', 'LBW', 'SJW']].mean(axis = 1)

data['oddsl'] = data[['CBL', 'GBL', 'IWL',  'SBL', 
            'B365L', 'B&WL', 'EXL', 'PSL', 'UBL', 'LBL', 'SJL']].mean(axis = 1)

data = data.dropna(subset = ['oddsw'])

data = data.dropna(subset = ['oddsl'])

data = data.reset_index(drop = True)

def underdog (x):
    if (x.oddsl > x.oddsw):
        return 0
    else:
        return 1
    
data['underdogWon'] = data.apply(underdog, axis = 1)



# Copp

data= data[~data['WRank'].isnull()]
data= data[~data['LRank'].isnull()]
data= data[data['WRank'].str.isnumeric()]
data= data[data['LRank'].str.isnumeric()]
data['WRank'] = pd.to_numeric(data['WRank'])
data['LRank'] = pd.to_numeric(data['LRank'])

data = data.reset_index(drop = True)

# Added 6/16 - Get win percentage for all ranks agasint each other
winners  = data['WRank']
winners = pd.unique(winners)
winners = winners.tolist()
winners.sort()
losers = data['LRank']
losers = pd.unique(losers)
losers = losers.tolist()
losers.sort()
players = list (set(winners) | set(losers))
players.sort()

winners = data['WRank']
losers = data['LRank']

old_dict = dict(enumerate(players))
new_dict = dict([(value, key) for key, value in old_dict.items()])
winners = winners.map(new_dict)
losers = losers.map(new_dict)
matches = pd.concat([winners, losers], axis=1)

m = len(players)

results = np.zeros(shape=(m,m))

for index, row in matches.iterrows():
    results[row['WRank'], row['LRank']] += 1

n = len(matches)
percent = np.zeros(n)
totalM = np.zeros(n)

for index, row in matches.iterrows():
    currentM = 0
    wins  = 0 
    losses = 0
    wrank  = row['WRank']
    print("wrank: ",  wrank)
    lrank  = row['LRank']
    print ("lrank: ", lrank)
    if (wrank < m) & (wrank >= 0):
        if (lrank < m) & (lrank >= 0):
            wins = results[wrank, lrank]
            print("wins: ", wins)
            losses = results[lrank, wrank]
            print("losses: ", losses)
    currentM += wins
    currentM += losses
    print("currentM: ", currentM)
    i = 1
    while currentM < 20:
        twins  = 0
        tlosses = 0
        for j in range(wrank - i, wrank + i):
            for k in range(lrank - i, lrank + i):
                if ((j < m) & (j >= 0)):
                    if ((k < m) & (k >= 0)):
                        twins += results[j, k]     
                        tlosses += results[k, j]
        currentM += twins
        currentM += tlosses
        wins += twins
        losses += tlosses
        i+= 1 
    print ("wins: ", wins)
    print ("losses ", losses)
    print(currentM)
    if index < n:
        totalM[index] = wins + losses
        if wins + losses  == 0:
            percent[index] = np.nan
        elif losses == 0:
            percent[index] = 1
        elif wins == 0:
            percent[index] = 0
        elif (wins>0) & (losses > 0):
            percent[index]  = wins / (wins + losses)
    


data['CoppW'] = percent
data['CoppL'] = 1 - percent
data['MatchesPlayed'] = totalM

# Create a 0/1 column for surface
data['Clay'] = data.Surface == 'Clay'
data['Clay'] *= 1 

data['Hard'] = data.Surface == 'Hard'
data['Hard'] *= 1

data['Grass'] = data.Surface == 'Grass'
data['Grass'] *= 1 

data['Carpet'] = data.Surface == 'Carpet'
data['Carpet'] *= 1 

data['WInverseRank'] =  data['WRank'].apply(lambda x: 1 /x) 
data['LInverseRank'] = data['LRank'].apply(lambda x: 1 /x) 

data.to_csv('Final Merged1.csv')



    