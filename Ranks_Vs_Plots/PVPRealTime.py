#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:20:44 2019

@author: rajdua

Description: Generate player vs. player win percentages in real time. 
"""

import numpy as np 
import pandas as pd

df = pd.read_csv('Data.csv', encoding = "ISO-8859-1", low_memory=False)
    
NRW = df[df['WRank']=='NR'] 
NRL = df[df['LRank']=='NR'] 
NAW = df[df['WRank'].isnull()]
NAL = df[df['LRank'].isnull()]

#union the above indexes so we can drop them
bad_index = NRW.index.union(NRL.index).union(NAW.index).union(NAL.index)
#now we would like to drop these bad rows from df
df = df.drop(bad_index)

#change string values to numeric values fixes common bugs
df[['WRank','LRank']] = df[['WRank','LRank']].apply(pd.to_numeric, errors='coerce')

df['Date'] = pd.to_datetime(df['Date'])
df.index = df['Date']
df = df.sort_index(ascending=True)

m = len(df)
df['players'] = df[['Winner','Loser']].apply(frozenset, axis=1)
df['Wins'] = 0
df['Loss'] = 0

for i in range(0,m):
    winner = df['Winner'][i]
    loser = df['Loser'][i]
    wins = 0
    loss = 0
        
    player = frozenset([winner, loser])
    for j in range(0,i):
        if player == df['players'][j]:
            if df['Winner'][j] == winner:
                wins = wins + 1
                loss = loss + 1
    df['Wins'][i] = wins
    df['Loss'][i] = loss
    print (i)
        