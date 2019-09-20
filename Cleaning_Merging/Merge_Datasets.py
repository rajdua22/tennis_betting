# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:09:58 2019

@author: Neel Tiruviluamala

Description: More efficient way to merge datasets. Note: This code was written
by Dr. Tiruviluamala of the USC Math department.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df1 = pd.read_csv('Data.csv', encoding = "ISO-8859-1", low_memory=False)
df2 = pd.read_csv('Data_stats.csv', encoding = "ISO-8859-1", low_memory=False)

#Create a base date so that we can recast dates in df1 and df2 as integers 
#that represent the number of days since the base date
base_date = pd.to_datetime('1-1-2000')

#Create the elapsed days (from the base date) field in df1 and df2
df1.Date = pd.to_datetime(df1.Date, format = '%d/%m/%Y')
df1['el_date'] = (df1.Date - base_date).dt.days
df2['el_date'] = (pd.to_datetime(df2.Changed_Date, format = '%m/%d/%Y') - base_date).dt.days

#Create a list of winner/loser pairs apparent in both df1 and df2
df1['Winner_Loser'] = df1.apply(lambda x: str(x.Winner)+str(x.Loser), axis = 1)
df2['Winner_Loser'] = df2.apply(lambda x: str(x.wchangedname) + str(x.lchangedname), axis = 1)

A = set(df1.Winner_Loser.unique())
B = set(df2.Winner_Loser.unique())
winner_loser_pairs = A.union(B)

winner_loser_pairs = list(winner_loser_pairs)

#Create a dictionary mapping between winner_loser_pairs and multiples of 100,000
#This is done because there are ~7000 integers in the 'el_date' fields and we want
#to place the winner_loser_pairs at different orders of magnitude

stp = 100000
dict_vals = np.arange(0, stp*len(winner_loser_pairs),stp)
wl_dict = dict(zip(winner_loser_pairs, dict_vals))

#Create the corresponding fields in df1 and df2 that contain the integer values associated
#with each winner/loser pair
df1['wl_number'] = df1.apply(lambda x: wl_dict[x.Winner_Loser],axis = 1)
df2['wl_number'] = df2.apply(lambda x: wl_dict[x.Winner_Loser],axis = 1)

#Create a merge number in df1 and df2 that adds the wl_number to the el_date number
#Note that the only way a merge number in df1 will be "close" to a merge number in df2 is if
#they correspond to the same row (and should thus be merged together)
df1['merge_number'] = df1.wl_number + df1.el_date
df2['merge_number'] = df2.wl_number + df2.el_date

df1 = df1.sort_values('merge_number')
df2 = df2.sort_values('merge_number')

#pd.merge_asof will merge two data frames based on "close" values.  We have done
#all the work to create this close value (merge_number) and so we proceed naturally

df1.to_csv('df1.csv')
df2.to_csv('df2.csv')
# df_merged = pd.merge_asof(df1, df2, on  = "merge_number", tolerance = 10)

# df_merged.to_csv('merger.csv')

left = pd.read_csv('df1.csv', encoding = "ISO-8859-1", low_memory=False)

right = pd.read_csv('df2.csv', encoding = "ISO-8859-1", low_memory=False)

df_merged = pd.merge_asof(left, right, on  = "merge_number", tolerance = 25)

df3 = df_merged.dropna(subset = ['wchangedname'])

df3 = df3.drop(columns = ['Unnamed: 0_x', 'el_date_x', 'Unnamed: 0_y', 'Source.Name', 'surface',
                    'Unnamed: 13', 'Unnamed: 14', 'wlastname', 'wchangedname', 'winner_name',
                    'llastname', 'lmiddlename', 'lfirstname', 'lchangedname', 'loser_name',
                    'best_of', 'Winner_Loser_y'])

df3.to_csv('Final Merged.csv')


                                        
