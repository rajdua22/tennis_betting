#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:20:52 2019

@author: rajdua

Description: Generate features before training.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Final Merged1.csv', encoding = "ISO-8859-1", low_memory=False)
data['Date'] = pd.to_datetime(data['Date'])

# data = data.tail(1000)
start = time.time()




#Features 4,5,6

player_melt = data[['Winner', 'Loser', 'Date', 'matches', 'games', 'Surface', 'Tournament']].melt(['Date', 'matches', 'games', 'Surface', 'Tournament'])
win_games = player_melt.groupby(['Date','value', 'variable', 'Surface', 'Tournament']).sum().reset_index()

# Win percent (career) before match
def winpercentW(x):
    df = win_games[win_games.value == x.Winner]
    df2 = df[(df.Date < x.Date)]
    df3 = df2[(df2.variable == 'Winner')]
    df4 = df2[(df2.variable == 'Loser')]
    if df4.matches.sum() + df3.matches.sum() == 0:
        return 'Nan'
    else:
        return (df3.matches.sum() / (df4.matches.sum() + df3.matches.sum()))

data['CWPW'] = data.apply(winpercentW, axis = 1)


def winpercentL(x):
    df = win_games[win_games.value == x.Loser]
    df2 = df[(df.Date < x.Date)]
    df3 = df2[(df2.variable == 'Winner')]
    df4 = df2[(df2.variable == 'Loser')]
    if df4.matches.sum() + df3.matches.sum() == 0:
        return 'Nan'
    else:
        return (df3.matches.sum() / (df4.matches.sum() + df3.matches.sum()))

data['CWPL'] = data.apply(winpercentL, axis = 1)


# Win percent (surface) before match
def surfacepercentW(x):
    df5 = win_games[win_games.value == x.Winner]
    df = df5[df5.Surface == x.Surface]
    df2 = df[(df.Date < x.Date)]
    df3 = df2[(df2.variable == 'Winner')]
    df4 = df2[(df2.variable == 'Loser')]
    if df4.matches.sum() + df3.matches.sum() == 0:
        return 'Nan'
    else:
        return (df3.matches.sum() / (df4.matches.sum() + df3.matches.sum()))

data['POSW'] = data.apply(surfacepercentW, axis = 1)


def surfacepercentL(x):
    df5 = win_games[win_games.value == x.Loser]
    df = df5[df5.Surface == x.Surface]
    df2 = df[(df.Date < x.Date)]
    df3 = df2[(df2.variable == 'Winner')]
    df4 = df2[(df2.variable == 'Loser')]
    if df4.matches.sum() + df3.matches.sum() == 0:
        return 'Nan'
    else:
        return (df3.matches.sum() / (df4.matches.sum() + df3.matches.sum()))

data['POSL'] = data.apply(surfacepercentL, axis = 1)


# Win percent (tourney) before match
date_lag = pd.Timedelta('50 days')

def tourneypercentW(x):
    df5 = win_games[win_games.value == x.Winner]
    df = df5[df5.Tournament == x.Tournament]
    df2 = df[(df.Date < x.Date - date_lag)]
    df3 = df2[(df2.variable == 'Winner')]
    df4 = df2[(df2.variable == 'Loser')]
    if df4.matches.sum() + df3.matches.sum() == 0:
        return 'Nan'
    else:
        return (df3.matches.sum() / (df4.matches.sum() + df3.matches.sum()))

data['PINW'] = data.apply(tourneypercentW, axis = 1)


def tourneypercentL(x):
    df5 = win_games[win_games.value == x.Loser]
    df = df5[df5.Tournament == x.Tournament]
    df2 = df[(df.Date < x.Date - date_lag)]
    df3 = df2[(df2.variable == 'Winner')]
    df4 = df2[(df2.variable == 'Loser')]
    if df4.matches.sum() + df3.matches.sum() == 0:
        return 'Nan'
    else:
        return (df3.matches.sum() / (df4.matches.sum() + df3.matches.sum()))

data['PINL'] = data.apply(tourneypercentL, axis = 1)



# Games played in the last 7 days
date_lag = pd.Timedelta('7 days')

def gamesPlayedW(x):
    df = win_games[win_games.value == x.Winner]
    df2 = df[( df.Date < x.Date) & (df.Date > x.Date - date_lag)]
    return df2.games.sum()

data['GPW'] = data.apply(gamesPlayedW, axis = 1)


def gamesPlayedL(x):
    df = win_games[win_games.value == x.Loser]
    df2 = df[( df.Date < x.Date) & (df.Date > x.Date - date_lag)]
    return df2.games.sum()

data['GPL'] = data.apply(gamesPlayedL, axis = 1)

end = time.time()
print(end-start)



# data = pd.read_csv('Final Merged1.csv', encoding = "ISO-8859-1", low_memory=False)
# data['Date'] = pd.to_datetime(data['Date'])
# start = time.time()

# Features 7,8,9,10,11,12

player_melt = data[['Winner', 'Loser', 'Date', 'matches', 'games', 'DSW', 'DSL','CSW', 'CSL', 'TieW', 'TieL', 'SetsCompleted', 'w_ace', 'l_ace',
                    'w_1stIn', 'l_1stIn', 'w_1stWon', 'l_1stWon', 'w_bpFaced', 'w_bpSaved', 'l_bpFaced', 'l_bpSaved', 'w_SvGms', 'l_SvGms']].melt([
        'Date', 'matches', 'games', 'DSW', 'DSL','CSW', 'CSL', 'TieW', 'TieL', 'SetsCompleted', 'w_ace', 'l_ace', 'w_1stIn', 'l_1stIn', 'w_1stWon', 'l_1stWon',
        'w_bpFaced', 'w_bpSaved', 'l_bpFaced', 'l_bpSaved', 'w_SvGms', 'l_SvGms' ])

def DSW(x):
    if (x.variable == 'Winner'):
        return x.DSW
    if (x.variable == 'Loser'):
        return x.DSL

def DSL(x):
    if (x.variable == 'Winner'):
        return x.DSL
    if (x.variable == 'Loser'):
        return x.DSW
    
def CSW(x):
    if (x.variable == 'Winner'):
        return x.CSW
    if (x.variable == 'Loser'):
        return x.CSL

def CSL(x):
    if (x.variable == 'Winner'):
        return x.CSL
    if (x.variable == 'Loser'):
        return x.CSW

def TieW(x):
    if (x.variable == 'Winner'):
        return x.TieW
    if (x.variable == 'Loser'):
        return x.TieL

def TieL(x):
    if (x.variable == 'Winner'):
        return x.TieL
    if (x.variable == 'Loser'):
        return x.TieW

def FstIn(x):
    if (x.variable == 'Winner'):
        return x.w_1stIn
    if (x.variable == 'Loser'):
        return x.l_1stIn

def aces(x):
    if (x.variable == 'Winner'):
        return x.w_ace
    if (x.variable == 'Loser'):
        return x.l_ace

def FstWon(x):
    if (x.variable == 'Winner'):
        return x.w_1stWon
    if (x.variable == 'Loser'):
        return x.l_1stWon
    
def BpFaced(x):
    if (x.variable == 'Winner'):
        return x.w_bpFaced
    if (x.variable == 'Loser'):
        return x.l_bpFaced

def BpSaved(x):
    if (x.variable == 'Winner'):
        return x.w_bpSaved
    if (x.variable == 'Loser'):
        return x.l_bpSaved
    
def SvGms(x):
    if (x.variable == 'Winner'):
        return x.w_SvGms
    if (x.variable == 'Loser'):
        return x.l_SvGms

    
player_melt['DSWx']= player_melt.apply(DSW, axis = 1)
player_melt['DSLx']= player_melt.apply(DSL, axis = 1)
player_melt['CSWx']= player_melt.apply(CSW, axis = 1)
player_melt['CSLx']= player_melt.apply(CSL, axis = 1)
player_melt['TieWx']= player_melt.apply(TieW, axis = 1)
player_melt['TieLx']= player_melt.apply(TieL, axis = 1)
player_melt['aces']= player_melt.apply(aces, axis = 1)
player_melt['FstIn']= player_melt.apply(FstIn, axis = 1)
player_melt['FstWon']= player_melt.apply(FstWon, axis = 1)
player_melt['bp_Faced']= player_melt.apply(BpFaced, axis = 1)
player_melt['bp_Saved']= player_melt.apply(BpSaved, axis = 1)
player_melt['SvGms']= player_melt.apply(SvGms, axis = 1)


sets_games = player_melt.groupby(['Date','value']).sum().reset_index()

# data = data.tail(1000)

# Lots of diffferent features for stats in last 15 matches - refer to documentation
date_lag = pd.Timedelta('20 days')

def W(x):
    df = sets_games[sets_games.value == x.Winner]
    df2 = df[(df.Date < x.Date)]
    size = df2.shape[0]
    if size  == 0:
        return pd.Series(['Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'])
    BPFC = df2.bp_Faced.sum() / df2.games.sum()
    BPWC = (df2.bp_Saved.sum() / df2.bp_Faced.sum())
    df2 = df2.reindex(index=df2.index[::-1])
    df2 = df2.reset_index()
    df2 = df2[0:15]
    sums = df2.SetsCompleted.sum()
    if sums == 0:
         return pd.Series(['Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'])
    else:
        DSW = (df2.DSWx.sum() / sums)
        DSL = (df2.DSLx.sum() / sums)
        CSW = (df2.CSWx.sum() / sums)
        CSL = (df2.CSLx.sum() / sums)
        TieW = (df2.TieWx.sum() / sums)
        TieL = (df2.TieLx.sum() / sums)
        Aces = (df2.aces.sum() / df2.SvGms.sum())
        FstIn = (df2.FstIn.sum() / df2.SvGms.sum())
        FSW = (df2.FstWon.sum() / df2.FstIn.sum())
        BPFG = (df2.bp_Faced.sum() / df2.games.sum())
        BPWG = (df2.bp_Saved.sum() / df2.bp_Faced.sum())
        return pd.Series([DSW, DSL, CSW, CSL, TieW, TieL, Aces, FstIn, FSW, BPFG, BPWG, BPFC, BPWC])

    
data[['DSWW', 'DSLW', 'CSWW','CSLW', 'TieWW', 'TieLW', 'APSGW', 'FSIW', 'FSWW', 'BPFGW', 'BPWGW', 'BPFCW', 'BPWCW']] = data.apply(W, axis = 1)


def L(x):
    df = sets_games[sets_games.value == x.Loser]
    df2 = df[(df.Date < x.Date)]
    size = df2.shape[0]
    if size  == 0:
        return pd.Series(['Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'])
    BPFC = df2.bp_Faced.sum() / df2.games.sum()
    BPWC = (df2.bp_Saved.sum() / df2.bp_Faced.sum())
    df2 = df2.reindex(index=df2.index[::-1])
    df2 = df2.reset_index()
    df2 = df2[0:15]
    sums = df2.SetsCompleted.sum()
    if sums == 0:
         return pd.Series(['Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'])
    else:
        DSW = (df2.DSWx.sum() / sums)
        DSL = (df2.DSLx.sum() / sums)
        CSW = (df2.CSWx.sum() / sums)
        CSL = (df2.CSLx.sum() / sums)
        TieW = (df2.TieWx.sum() / sums)
        TieL = (df2.TieLx.sum() / sums)
        Aces = (df2.aces.sum() / df2.SvGms.sum())
        FstIn = (df2.FstIn.sum() / df2.SvGms.sum())
        FSW = (df2.FstWon.sum() / df2.FstIn.sum())
        BPFG = (df2.bp_Faced.sum() / df2.games.sum())
        BPWG = (df2.bp_Saved.sum() / df2.bp_Faced.sum())
        return pd.Series([DSW, DSL, CSW, CSL, TieW, TieL, Aces, FstIn, FSW, BPFG, BPWG, BPFC, BPWC])

data[['DSWL', 'DSLL', 'CSWL','CSLL', 'TieWL', 'TieLL', 'APSGL', 'FSIL', 'FSWL', 'BPFGL', 'BPWGL', 'BPFCL', 'BPWCL']] = data.apply(L, axis = 1)

end = time.time()
print(end-start)






# data = pd.read_csv('Final Merged1.csv', encoding = "ISO-8859-1", low_memory=False)
# data['Date'] = pd.to_datetime(data['Date'])
# start = time.time()

# Similar spot in tournament percentage - stage and underdog/favorite

player_melt = data[['Winner', 'Loser', 'Date', 'matches', 'underdogWon', 'SOT']].melt(['Date', 'matches', 'underdogWon', 'SOT'])

def p(x):
    if (x.underdogWon == 1) & (x.variable == 'Winner'):
        return 1
    elif (x.underdogWon == 0) & (x.variable == 'Loser'):
        return 1
    else:
        return 0
        
player_melt['underdog'] = player_melt.apply(p, axis = 1)

sets_games = player_melt.groupby(['Date','value', 'variable']).sum().reset_index()


# data = data.tail(1000)

def TSSW(x):
    df = sets_games[sets_games.value == x.Winner]
    df2 = df[(df.Date < x.Date)]
    df2 = df2[df2.SOT == x.SOT]
    if (x.underdogWon == 1):
        df2 = df2[df2.underdog > 0]
        if (df2.matches.sum() == 0):
            SS = 'NaN'
        else:
            SS = (df2.underdogWon.sum() / df2.matches.sum())
    else:
        df2 = df2[df2.underdog == 0]
        if (df2.matches.sum() == 0):
            SS = 'NaN'
        SS = 1 - (df2.underdogWon.sum() / df2.matches.sum())
    return SS

data['TSSW'] = data.apply(TSSW, axis = 1)

def TSSL(x):
    df = sets_games[sets_games.value == x.Loser]
    df2 = df[(df.Date < x.Date)]
    df2 = df2[df2.SOT == x.SOT]
    if (x.underdogWon == 0):
        df2 = df2[df2.underdog > 0]
        if (df2.matches.sum() == 0):
            SS = 'NaN'
        else:
            SS = (df2.underdogWon.sum() / df2.matches.sum())
    else:
        df2 = df2[df2.underdog == 0]
        if (df2.matches.sum() == 0):
            SS = 'NaN'
        SS = 1 - (df2.underdogWon.sum() / df2.matches.sum())
    return SS

data['TSSL'] = data.apply(TSSL, axis = 1)



# Copp & PVP
def f(x):
    return frozenset({x.Winner, x.Loser})

data['players'] = data.apply(f, axis =1)


def b(x):
    df = data[data.players == x.players]
    df2 = df[(df.Date < x.Date)]
    df3 = df2[(x.Winner == df2.Winner)]
    matches = df2.matches.sum()
    W = (df3.matches.sum() / matches)
    L = 1 - W
    return pd.Series([W, L, matches])

data[['PVPW', 'PVPL', 'PVPM']] = data.apply(b, axis = 1)

# end = time.time()
# print(end-start)





# data = pd.read_csv('Final Merged1.csv', encoding = "ISO-8859-1", low_memory=False)
# data['Date'] = pd.to_datetime(data['Date'])

# Rank trends
data.sort_values(by = ['Date'])
player_melt = data[['Winner', 'Loser', 'Date', 'WRank', 'LRank']].melt(['Date', 'WRank', 'LRank'])

def rank(x):
    if x.variable == 'Winner':
        return x.WRank
    else:
        return x.LRank
    
player_melt['Rank'] = player_melt.apply(rank, axis = 1)
player_melt = player_melt.sort_values(by = ['Date'])

# data = data.tail(1000)

# start = time.time()

def rankTrend(x):
    df = player_melt[player_melt.value == x.Winner]
    df2 = df[(df.Date < x.Date)]
    df2 = df2.sort_values(by = 'Date')
    df2 = df2.reindex(index=df2.index[::-1])
    df2 = df2.reset_index()
    size = df2.shape[0]
    if (size == 0):
        return 'NaN'
    elif (size < 15):
        df2 = df2[0:size]
    else:
        df2 = df2[0:15]    
    df2 = df2.reindex(index=df2.index[::-1])
    df2 = df2.reset_index()
    start = df2['Date'].iloc[0]
    X = (df2['Date'] -  start).dt.days.values.reshape(-1, 1)
    Y = df2['Rank'].values.reshape(-1,1)
    reg = LinearRegression().fit(X, Y)
    return reg.coef_

data['RTrendW'] = data.apply(rankTrend, axis = 1)


def rankTrendL(x):
    df = player_melt[player_melt.value == x.Loser]
    df2 = df[(df.Date < x.Date)]
    df2 = df2.sort_values(by = 'Date')
    df2 = df2.reindex(index=df2.index[::-1])
    df2 = df2.reset_index()
    size = df2.shape[0]
    if (size == 0):
        return 'NaN'
    elif (size < 15):
        df2 = df2[0:size]
    else:
        df2 = df2[0:15]      
    df2 = df2.reindex(index=df2.index[::-1])
    df2 = df2.reset_index()
    start = df2['Date'].iloc[0]
    X = (df2['Date'] -  start).dt.days.values.reshape(-1, 1)
    Y = df2['Rank'].values.reshape(-1,1)
    reg = LinearRegression().fit(X, Y)
    return reg.coef_

data['RTrendL'] = data.apply(rankTrendL, axis = 1)   


end = time.time()
print(end-start)

data.to_csv('FinalData.csv')
    
    

