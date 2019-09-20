#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 3 8:01:09 2019

@author: rajdua

Decritption: Take in preprocessed dataset, fill in missing values, and double the rows.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

df = pd.read_csv('FinalData.csv', encoding = "ISO-8859-1", low_memory=False)

NRW = df[df['WRank']=='NR'] 
NRL = df[df['LRank']=='NR'] 
NAW = df[df['WRank'].isnull()]
NAL = df[df['LRank'].isnull()]
#
##union the above indexes so we can drop them
bad_index = NRW.index.union(NRL.index).union(NAW.index).union(NAL.index)
##now we would like to drop these bad rows from df
df = df.drop(bad_index)

#change string values to numeric values fixes common bugs
df[['WRank','LRank']] = df[['WRank','LRank']].apply(pd.to_numeric, errors='coerce')

df = df.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1', 'tourney_date', 'Changed_Date', 'Winner_Loser_x',
                        'wl_number_x', 'merge_number', 'wfirstname', 'el_date_y', 'wl_number_y'])

df['Best of'] = 1 * (df['Best of'] == 3)
df['winner_hand'] = 1 * (df['winner_hand'] == 'R')
df['loser_hand'] = 1 * (df['loser_hand'] == 'R')

def f(x):
    n = x.PVPM
    if n == 0:
        return x.CoppW
    else:   
        return (x.CoppW / (n + 1)) + ((n* x.PVPW) / (n+1))

df['Wcomb'] = df.apply(f, axis = 1)


def g(x):
    n = x.PVPM
    if n == 0:
        return x.CoppL
    else:   
        return (x.CoppL / (n + 1)) + ((n* x.PVPL) / (n+1))

df['Lcomb'] = df.apply(g, axis = 1)


df['RTrendW'] = df['RTrendW'].replace(np.nan, 'NAN')
def p(x):
    rank  = x.RTrendW
    if rank == 'NAN':
        return np.nan
    i = 2
    while (rank[i] != ']'):
        i += 1
    return rank[2:i]

rankw  = df.apply(p, axis = 1)
rankw = np.asarray(rankw, dtype=np.float64)

df['RTrendL'] = df['RTrendL'].replace(np.nan, 'NAN')
def l(x):
    rank  = x.RTrendL
    if rank == 'NAN':
        return np.nan
    i = 2
    while (rank[i] != ']'):
        i += 1
    return rank[2:i]

rankl  = df.apply(l, axis = 1)
rankl = np.asarray(rankl, dtype=np.float64)

df['RTrendW'] = rankw
df['RTrendL'] = rankl


df['y'] = 1

data = df.copy()

data['y'] = 0

data['Wcomb'] = df['Lcomb']
data['Lcomb'] = df['Wcomb']

data['LRank'] = df['WRank']
data['WRank'] = df['LRank']

data['Loser'] = df['Winner']
data['Winner'] = df['Loser']

data['W1'] = df['L1']
data['W2'] = df['L2']
data['W3'] = df['L1']
data['W4'] = df['L2']
data['W5'] = df['L1']
data['L1'] = df['W1']
data['L2'] = df['W2']
data['L3'] = df['W3']
data['L4'] = df['W4']
data['L5'] = df['W5']

data['Wsets'] = df['Lsets']
data['Lsets'] = df['Wsets']

data['CBW'] = df['CBL']
data['CBL'] = df['CBW']
data['GBW'] = df['GBL']
data['GBL'] = df['GBW']
data['IWW'] = df['IWL']
data['IWL'] = df['IWW']
data['SBW'] = df['SBL']
data['SBL'] = df['SBW']
data['B365W'] = df['B365L']
data['B365L'] = df['B365W']
data['B&WW'] = df['B&WL']
data['B&WL'] = df['B&WW']
data['EXW'] = df['EXL']
data['EXL'] = df['EXW']
data['PSW'] = df['PSL']
data['PSL'] = df['PSW']
data['WPts'] = df['LPts']
data['LPts'] = df['WPts']
data['UBW'] = df['UBL']
data['UBL'] = df['UBW']
data['LBW'] = df['LBL']
data['LBL'] = df['LBW']
data['SJW'] = df['SJL']
data['SJL'] = df['SJW']
data['MaxW'] = df['MaxL']
data['MaxL'] = df['MaxW']
data['AvgW'] = df['AvgL']
data['AvgL'] = df['AvgW']

data['winner_id'] = df['loser_id']
data['loser_id'] = df['winner_id']
data['winner_seed'] = df['loser_seed']
data['loser_seed'] = df['winner_seed']
data['winner_entry'] = df['loser_entry']
data['loser_entry'] = df['winner_entry']
data['winner_hand'] = df['loser_hand']
data['loser_hand'] = df['winner_hand']

data['winner_ht'] = df['loser_ht']
data['loser_ht'] = df['winner_ht']
data['winner_ioc'] = df['loser_ioc']
data['loser_ioc'] = df['winner_ioc']
data['winner_age']= df['loser_age']
data['loser_age'] = df['winner_age']
data['w_ace'] = df['l_ace']
data['l_ace'] = df['w_ace']
data['w_df'] = df['l_df']
data['l_df'] = df['w_df']
data['w_svpt'] = df['l_svpt']
data['l_svpt'] = df['w_svpt']
data['w_1stIn'] = df['l_1stIn']
data['l_1stIn'] = df['w_1stIn']
data['w_1stWon'] = df['l_1stWon']
data['l_1stWon'] = df['w_1stWon']
data['w_2ndWon'] = df['l_2ndWon']
data['l_2ndWon'] = df['w_2ndWon']
data['w_SvGms'] = df['l_SvGms']
data['l_SvGms'] = df['w_SvGms']
data['w_bpSaved'] = df['l_bpSaved']
data['l_bpSaved'] = df['w_bpSaved']
data['w_bpFaced'] = df['l_bpFaced']
data['l_bpFaced'] = df['w_bpFaced']
data['winner_rank'] = df['loser_rank']
data['loser_rank'] = df['winner_rank']
data['winner_rank_points'] = df['loser_rank_points']
data['loser_rank_points'] = df['winner_rank_points']


data['DSW'] = df['DSL']
data['DSL'] = df['DSW']
data['CSW'] = df['CSL']
data['CSL'] = df['CSW']
data['TieW'] = df['TieL']
data['TieL'] = df['TieW']
data['oddsw'] = df['oddsl']
data['oddsl'] = df['oddsw']
data['CoppW'] = df['CoppL']
data['CoppL'] = df['CoppW']
data['WInverseRank'] = df['LInverseRank']
data['LInverseRank'] = df['WInverseRank']
data['CWPW'] = df['CWPL']
data['CWPL'] = df['CWPW']
data['POSW'] = df['POSL']
data['POSL'] = df['POSW']
data['PINW'] = df['PINL']
data['PINL'] = df['PINW']
data['GPW'] = df['GPL']
data['GPL'] = df['GPW']
data['DSWW'] = df['DSWL']
data['DSWL'] = df['DSWW']
data['CSWW'] = df['CSWL']
data['CSWL'] = df['CSWW']
data['TieWW'] = df['TieWL']
data['TieWL'] = df['TieWW']

data['DSLW'] = df['DSLL']
data['DSLL'] = df['DSLW']
data['CSLW'] = df['CSLL']
data['CSLL'] = df['CSLW']
data['TieLW'] = df['TieLL']
data['TieLL'] = df['TieLW']

data['APSGW'] = df['APSGL']
data['APSGL'] = df['APSGW']
data['FSIW'] = df['FSIL']
data['FSIL'] = df['FSIW']
data['FSWW'] = df['FSWL']
data['FSWL'] = df['FSWW']
data['BPFGW'] = df['BPFGL']
data['BPFGL'] = df['BPFGW']
data['BPWGW'] = df['BPWGL']
data['BPWGL'] = df['BPWGW']
data['BPFCW'] = df['BPFCL']
data['BPFCL'] = df['BPFCW']
data['BPWCW'] = df['BPWCL']
data['BPWCL'] = df['BPWCW']

data['TSSW'] = df['TSSL']
data['TSSL'] = df['TSSW']
data['PVPW'] = df['PVPL']
data['PVPL'] = df['PVPW']
data['RTrendW'] = df['RTrendL']
data['RTrendL'] = df['RTrendW']


merged_df = merged_df = pd.concat([df, data])

merged_df.to_csv('FinalData_Cleaned.csv')