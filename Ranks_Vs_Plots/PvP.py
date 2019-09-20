#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 00:44:43 2019

@author: rajdua

Description: Get win percentages vs. specific opponents (not time dependent).
"""

import pandas as pd
import numpy as np
data = pd.read_csv('Data.csv', encoding = "ISO-8859-1", low_memory=False)

# Added 6/16 - Get win percentage vs. specific opponent
winners  = data['Winner']
winners = pd.unique(winners)
winners = winners.tolist()
winners.sort()
losers = data['Loser']
losers = pd.unique(losers)
losers = losers.tolist()
losers.sort()
players = list (set(winners) | set(losers))
players.sort()

winners = data['Winner']
losers = data['Loser']

old_dict = dict(enumerate(players))
new_dict = dict([(value, key) for key, value in old_dict.items()])
winners = winners.map(new_dict)
losers = losers.map(new_dict)
matches = pd.concat([winners, losers], axis=1)

m = len(players)

results = np.zeros(shape=(m,m))

for index, row in matches.iterrows():
    results[row['Winner'], row['Loser']] += 1

n = len(matches)
percent = np.zeros(n)
totalM = np.zeros(n)

for index, row in matches.iterrows():
    wins = results[row['Winner'], row['Loser']]
    losses = results[row['Loser'], row['Winner']]
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
data['WPvP'] = percent
data['LPvP'] = 1 - percent
data['MatchesPlayedPvP'] = totalM

# Output to file
data.to_csv('PvP.csv')




