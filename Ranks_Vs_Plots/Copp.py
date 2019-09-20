#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 08:49:59 2019

@author: rajdua

Description: Generate win percentages for ranks vs. rank ranges.
"""

import pandas as pd
import numpy as np
data2= pd.read_csv('Final Merged1.csv', encoding = "ISO-8859-1", low_memory=False)


data2= data2[~data2['WRank'].isnull()]
data2= data2[~data2['LRank'].isnull()]
data2= data2[data2['WRank'].str.isnumeric()]
data2= data2[data2['LRank'].str.isnumeric()]
data2['WRank'] = pd.to_numeric(data2['WRank'])
data2['LRank'] = pd.to_numeric(data2['LRank'])

# Added 6/16 - Get win percentage for all ranks agasint each other
winners  = data2['WRank']
winners = pd.unique(winners)
winners = winners.tolist()
winners.sort()
losers = data2['LRank']
losers = pd.unique(losers)
losers = losers.tolist()
losers.sort()
players = list (set(winners) | set(losers))
players.sort()

winners = data2['WRank']
losers = data2['LRank']

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
    # print("wrank: ",  wrank)
    lrank  = row['LRank']
    # print ("lrank: ", lrank)
    if (wrank < m) & (wrank >= 0):
        if (wrank < m) & (wrank >= 0):
            wins = results[wrank, lrank]
            # print("wins: ", wins)
            losses = results[lrank, wrank]
            # print("losses: ", losses)
    currentM += wins
    currentM += losses
    # print("currentM: ", currentM)
    i = 1
    while currentM < 20:
        twins  = 0
        tlosses = 0
        for j in range(wrank - i, wrank + i):
            for k in range(lrank - i, lrank + i):
                if (j < m) & (j >= 0):
                    if (k < m) & (k >= 0):
                        twins += results[j, k]     
                        tlosses += results[k, j]
        currentM += twins
        currentM += tlosses
        wins += twins
        losses += tlosses
        i+= 1 
    # print ("wins: ", wins)
    # print ("losses ", losses)
    # print(currentM)
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
    

#WPvP is winner's percentage against opponent
#LPvP is loser's percentage against opponent
data2['CoppW'] = percent
data2['CoppL'] = 1 - percent
data2['MatchesPlayed'] = totalM

data2['Date'] = pd.to_datetime(data2['Date'])
data2.index = data2['Date'] ##do I need to set the index by times? or should I just keep a column of dates?
data2= data2.sort_index(ascending=True)


# Output to file
data2.to_csv('Final Merged1.csv')