# Author: Austin Pollok

### Note: This code was written by Austin Pollok, a PhD student at the USC Mathematics Department

### Description: Perform some EDA. An early attempt at extracting features. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

###note we have cross-sectional(spatial) data and time-series data

##there is an error with the data, Rankings have NR values which I manually changed to N/A
df = pd.read_csv('DataCleaned2.csv', encoding = "ISO-8859-1", low_memory=False)

#print(df.WRank.isnull().sum(), df.LRank.isnull().sum(), df.WRank.isnull().sum() + df.LRank.isnull().sum())
#we're isolating the NR and NaN in the rankings so we can get rid of them
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

##change index to date so we can make some features time dependent
df['Date'] = pd.to_datetime(df['Date'])
df.index = df['Date'] ##do I need to set the index by times? or should I just keep a column of dates?
df = df.sort_index(ascending=True)

df['y']  = 1

##we need to  create new dataframes of series to store our new features then merge them later to create a feature vector/matrix of covariates
##we could just add them onto the original dataframe by df['newfeaturename'] = whatever
##we create a features dataframe

##we split features into game features which are time invariant
##this houses features that you can get solely from the game info without a history of the players
##this is for the cross-sectional(spatial) data
game_features = pd.DataFrame()

game_features['Winner'] = df['Winner']
game_features['Loser'] = df['Loser']

#features['Player0Rank'] = df[['WRank','LRank']].max(axis=1) 
#features['Player1Rank'] = df[['WRank','LRank']].min(axis=1)
game_features['WRank'] = df['WRank']
game_features['LRank'] = df['LRank']
game_features['RankDiff'] = (df['WRank']-df['LRank']).abs()

game_features['Hard'] = np.where(df['Surface'] == 'Hard', 1, 0)
game_features['Clay'] = np.where(df['Surface'] == 'Clay', 1, 0)
game_features['Grass'] = np.where(df['Surface'] == 'Grass', 1, 0)
game_features['Carpet'] = np.where(df['Surface'] == 'Carpet', 1, 0)

game_features['Indoor'] = np.where(df['Court'] == 'Indoor', 1, 0)
game_features['Outdoor'] = np.where(df['Court']=='Outdoor', 1, 0)

###note that DataFrame.groupby() is like in Excel when you select the column
###so that the rows are ordered by the selected column, then the brackets [''] after
###groupby is like selecting the column called '' ordered by the selected column

#note there are some duplicate players such as Federer or Ferrero, possibly b/c of spacing?
winners = pd.DataFrame(game_features.groupby('Winner').Winner.count())
losers = pd.DataFrame(game_features.groupby('Loser').Loser.count())

##this is for the time-series data
players_features = pd.merge(winners, losers, how='outer', left_index=True, right_index=True).fillna(0.)
players_features.columns = ['Wins', 'Losses']
players_features['WPercentage'] = players_features['Wins']/(players_features['Wins']+players_features['Losses'])


clay_wins = game_features.groupby('Winner').Clay.sum().rename('ClayWins')
clay_losses = game_features.groupby('Loser').Clay.sum().rename('ClayLosses')
clay_record = pd.merge(clay_wins, clay_losses, how='outer',  left_index=True,right_index=True).fillna(0.)
players_features = pd.merge(players_features, clay_record, left_index=True, right_index=True)
players_features['ClWPercentage'] = players_features['ClayWins']/(players_features['ClayWins']+players_features['ClayLosses'])

hard_wins = game_features.groupby('Winner').Hard.sum().rename('HardWins')
hard_losses = game_features.groupby('Loser').Hard.sum().rename('HardLosses')
hard_record = pd.merge(hard_wins, hard_losses, how='outer', left_index=True, right_index=True).fillna(0.)
players_features = pd.merge(players_features, hard_record, left_index=True, right_index=True)
players_features['HWPercentage'] = players_features['HardWins']/(players_features['HardWins']+players_features['HardLosses'])

grass_wins = game_features.groupby('Winner').Grass.sum().rename('GrassWins')
grass_losses = game_features.groupby('Loser').Grass.sum().rename('GrassLosses')
grass_record = pd.merge(grass_wins, grass_losses, how='outer', left_index=True, right_index=True).fillna(0.)
players_features = pd.merge(players_features, grass_record, left_index=True, right_index=True)
players_features['GWPercentage'] = players_features['GrassWins']/(players_features['GrassWins']+players_features['GrassLosses'])

carpet_wins = game_features.groupby('Winner').Carpet.sum().rename('CarpetWins')
carpet_losses = game_features.groupby('Loser').Carpet.sum().rename('CarpetLosses')
carpet_record = pd.merge(carpet_wins, carpet_losses, how='outer', left_index=True, right_index=True).fillna(0.)
players_features = pd.merge(players_features, carpet_record, left_index=True, right_index=True)
players_features['CarpWPercentage'] = players_features['CarpetWins']/(players_features['CarpetWins']+players_features['CarpetLosses'])

indoor_wins = game_features.groupby('Winner').Indoor.sum().rename('IndoorWins')
indoor_losses = game_features.groupby('Loser').Indoor.sum().rename('IndoorLosses')
indoor_record = pd.merge(indoor_wins, indoor_losses, how='outer', left_index=True, right_index=True).fillna(0.)
players_features = pd.merge(players_features, indoor_record, left_index=True, right_index=True)
players_features['IndWPercentage'] = players_features['IndoorWins']/(players_features['IndoorWins']+players_features['IndoorLosses'])
#
outdoor_wins = game_features.groupby('Winner').Outdoor.sum().rename('OutdoorWins')
outdoor_losses = game_features.groupby('Loser').Outdoor.sum().rename('OutdoorLosses')
outdoor_record = pd.merge(outdoor_wins, outdoor_losses, how='outer', left_index=True, right_index=True).fillna(0.)
players_features = pd.merge(players_features, outdoor_record, left_index=True, right_index=True)
players_features['OutWPercentage'] = players_features['OutdoorWins']/(players_features['OutdoorWins']+players_features['OutdoorLosses'])

#player_rank is just a middle dataframe used to store some values in between computing the desired values
sum_winner_rank = game_features.groupby('Winner').WRank.sum().rename('TotalWRank')
sum_loser_rank = game_features.groupby('Loser').LRank.sum().rename('TotalLRank')
player_rank = pd.merge(sum_winner_rank, sum_loser_rank, how='outer', left_index=True, right_index=True).fillna(0.)
player_rank['TotalRank'] = player_rank['TotalWRank'] + player_rank['TotalLRank']
player_rank['AvgRank'] = player_rank['TotalRank'] / (players_features['Wins'] + players_features['Losses'])
players_features['AvgRank'] = player_rank['AvgRank']

##these are all at final time, need to make time dependent


#here I consider the average rank from 2012 on, this is ad-hoc and should be handled for any time
winners2012 = game_features['2012':].groupby('Winner').Winner.count().rename('Wins2012+')
losers2012 = game_features['2012':].groupby('Loser').Loser.count().rename('Losses2012+')
players_features[['Wins2012+', 'Losses2012+']] = pd.merge(winners2012, losers2012, how='outer', left_index=True, right_index=True).fillna(0.)
players_features[['Wins2012+', 'Losses2012+']] = players_features[['Wins2012+', 'Losses2012+']].fillna(0.)

sum_winner_rank2012 = game_features['2012':].groupby('Winner').WRank.sum().rename('TotalWRank2012+')
sum_loser_rank2012 = game_features['2012':].groupby('Loser').LRank.sum().rename('TotalLRank2012+')
player_rank[['TotalWRank2012+', 'TotalLRank2012+']] = pd.merge(sum_winner_rank2012, sum_loser_rank2012, how='outer', left_index=True, right_index=True).fillna(0.)
player_rank[['TotalWRank2012+', 'TotalLRank2012+']] = player_rank[['TotalWRank2012+', 'TotalLRank2012+']].fillna(0.)

player_rank['TotalRank2012+'] = player_rank['TotalWRank2012+'] + player_rank['TotalLRank2012+']
player_rank['AvgRank2012+'] = player_rank['TotalRank2012+'] / (players_features['Wins2012+'] + players_features['Losses2012+'])
players_features['AvgRank2012+'] = player_rank['AvgRank2012+']

##here we attempt to make the players features dynamic
game_features.groupby(['Date', 'Winner']).WRank.sum() #or cumsum()
game_features.groupby(['Date', 'Winner']).LRank.sum() # .unstack()

exampledoublegroupby = game_features.groupby(['Winner','Loser'])[['WRank','LRank']].mean() #this finds the average winner and loser ranking for every matchup, can also replace with .count()
exdoublegroupby1 = game_features.groupby(['Date','Winner']).WRank.mean()
doubgrouptoDF = exdoublegroupby1.unstack()

pivottable1 = game_features.pivot_table(values='WRank', index=game_features.index, columns='Winner')
pivottable2 = game_features.pivot_table(values='LRank', index=game_features.index, columns='Loser')
mergepivottables = pd.merge(pivottable1, pivottable2, how='outer', left_index=True, right_index=True).fillna(0.)

examplepivottable = game_features.pivot_table(values=game_features.columns, index=game_features.Winner, columns=game_features.index)

players_features['Winner'] = players_features.index
data = df.merge(players_features, on = 'Winner')
players_features.columns = ['WinsL', 'LossesL', 'WPercentageL', 'ClayWinsL', 'ClayLossesL', 'ClWPercentageL', 'HardWinsL', 'HardLossesL', 'HWPercentageL', 'GrassWinsL', 'GrassLossesL', 'GWPercentageL', 'CarpetWinsL', 
                            'CareptLossesL', 'CarpWPercentageL', 'IndoorWinsL','IndoorLossesL', 'IndWPercentageL', 'OUtdoorWins', 'OutdoorLossesL', 'OutWPercentageL', 'AvgrankL', 'Wins2012+L', 'Losses2012+L',
                            'AvgRank2012+L', 'Loser']

data = data.merge(players_features, on = 'Loser')

# Create a 0/1 column for clay
data['Clay'] = data.Surface == 'Clay'
data['Clay'] *= 1 

data['Hard'] = data.Surface == 'Hard'
data['Hard'] *= 1

data['Grass'] = data.Surface == 'Grass'
data['Grass'] *= 1 

data['Carpet'] = data.Surface == 'Carpet'
data['Carpet'] *= 1 


data['PercentOnSurface'] = data['Grass'] * data['GWPercentage'] + data['Clay'] * data['ClWPercentage'] + data['Carpet'] * data['CarpWPercentage'] + data['Hard'] * data['HWPercentage']
data['PercentOnSurfaceL'] = data['Grass'] * data['GWPercentageL'] + data['Clay'] * data['ClWPercentageL'] + data['Carpet'] * data['CarpWPercentageL'] + data['Hard'] * data['HWPercentageL']

data['WInverseRank'] =  data['WRank'].apply(lambda x: 1 /x) 
data['LInverseRank'] = data['LRank'].apply(lambda x: 1 /x) 

# data['Date'] = pd.to_datetime(data['Date'])
#data.index = data['Date'] 
#data = data.sort_index(ascending=True)

data2 = data

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
data2['MatchesPlayedCopp'] = totalM




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
data2['WPvP'] = percent
data2['LPvP'] = 1 - percent
data2['MatchesPlayedPvP'] = totalM

data2['Date'] = pd.to_datetime(data2['Date'])
data2.index = data2['Date'] ##do I need to set the index by times? or should I just keep a column of dates?
data2= data2.sort_index(ascending=True)

#we put the features together to get one set of features

##normalize the features we're accepting

data2.to_csv('DataFeatures2.csv')