
"""
@author: JacobS
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance
import timeit
import matplotlib.pyplot as plt


#%% - import
# Importing data set at 20 Hz including:
# Timestamp, X, Y, speed, acceleration, relative speed, heartrate, relative heartrate, frequency, ID, name
start = timeit.default_timer()

df = pd.read_csv("RawData17122020Version3.csv", quotechar = ",", encoding = "ISO-8859-1")  # RawDataJacob2  RawData17122020Version3
df = df.replace('"', '', regex=True)
df.columns = df.columns.str.replace('"', "")
pd.set_option('display.max_columns', 50)

#%% - pre-processing data

df['X'].fillna(method='pad', inplace=True)  # remove 0 with previous row value
df['Y'].fillna(method='pad', inplace=True)  # remove 0 with previous row value

# df = df[(df['relative speed'] < 100)] # Verwijderen van grove outliers

# df = df.drop(df[(df['Speed'] == 0.0) & (df['Euclidean Distance Diff'] > 5.0)].index)  # remove outliers
# df = df.drop(df[(df['Speed'] == 0.0) & (df['relative speed'] > 50.0)].index)  # remove outliers

#%% pre-process incorrect data

df.loc[df['X'] > 20.5, ['Speed', 'Acceleration']] = 0
df.loc[df['X'] < -20.5, ['Speed', 'Acceleration']] = 0
df.loc[df['Y'] > 11.0, ['Speed', 'Acceleration']] = 0
df.loc[df['Y'] < -11.0, ['Speed', 'Acceleration']] = 0
df['Speed'].values[df['Speed'] > 8.0] = 0 
df['Acceleration'].values[df['Acceleration'] < -5.0] = 0
df['Acceleration'].values[df['Acceleration'] > 5.0] = 0  

#%% - making dictionary and list 
df['Name'] = df['Name'].astype('category')
playerName = df['Name'].unique().tolist()
playerName = [playerName for playerName in playerName if str(playerName) != 'nan']  # remove nan

df_dict = {name: df.loc[df['Name'] == name] for name in playerName}
df_list = [df.loc[df['Name'] == name] for name in playerName]

df = df.apply(pd.to_numeric, errors='coerce')  # object to int64
 
#%% - calculate variables per player

# calculate Euclidean Distance 
def calc_Euclidean(X, Y):  # calculate distance covered
    return distance.euclidean(X, Y)
for p in df_list:
    p['Euclidean Distance'] = p.apply(lambda row: calc_Euclidean(row.X, row.Y), axis=1) 
    p['Euclidean Distance Diff'] = abs(p['Euclidean Distance'] - p['Euclidean Distance'].shift(1))
    #%% remove wrong distances due to leaving the field
    p['Timestamp'] = p['Timestamp'].apply(pd.to_numeric, errors='coerce')  # object to int64
    p['Diff Timestamp'] = p['Timestamp'] - p['Timestamp'].shift(1) 
    p.loc[(p['Diff Timestamp'] > 100.0),  'Euclidean Distance Diff'] = 0
    p['Total Euclidean Distance'] = np.cumsum(p['Euclidean Distance Diff'])
    #%% calculate HID
    p['relative speed'] = p['relative speed'].apply(pd.to_numeric, errors='coerce')  # object to int64
    p['High Intensity Distance'] = p.loc[(p['relative speed'] >= 70), 'Euclidean Distance Diff']  # find distance covered when speed > 70%
    p['High Intensity Distance'] = p['High Intensity Distance'].fillna(value=0)
    p['HID'] = p.groupby(['Name'])['High Intensity Distance'].cumsum()  # cumulative sum of all HID meters
    #%% calculate Acc > 2.0 m/s2
    # What is missing is the min acceleration time = 100 ms -) 
    p['Acc > 2.0'] = 0 # add a class column with 0 as default value
    p.loc[(p['Acceleration'] >= 2) & (p['Speed'] >= 2) , 'Acc > 2.0'] = 1 # find all rows that fulfills your conditions and set Acc > 2.0 m/s2 to 1
    p['Diff Acc'] = p['Acc > 2.0'] - p['Acc > 2.0'].shift(1)
    p.loc[(p['Diff Acc'] >= 1), 'Acc'] = 1
    p['Total Acc'] = np.cumsum(p['Acc'])
    p.drop(['Acc > 2.0', 'Acc', 'Diff Acc'], axis=1, inplace=True)  # drop first 3 acc columns
    #%% calculate Decc < -2.0 m/s2
    # What is missing is the min decceleration time = 100 ms
    p['Decc > 2.0'] = 0 # add a column with 0 as default value
    p.loc[(p['Acceleration'] <= -2) & (p['Speed'] >= 2) , 'Decc > 2.0'] = 1 #  & (p['Speed'] >= 2)  find all rows that fulfills your conditions and set Acc > 2.0 m/s2 to 1
    p['Diff Decc'] = p['Decc > 2.0'] - p['Decc > 2.0'].shift(1)
    p.loc[(p['Diff Decc'] >= 1), 'Decc'] = 1
    p['Total Decc'] = np.cumsum(p['Decc'])
    p.drop(['Decc > 2.0', 'Decc', 'Diff Decc'], axis=1, inplace=True)  # drop first 3 acc columns   
    #%% calculate MP
    p['alpha'] = np.degrees(np.arctan(9.81/p['Acceleration']))
    p['g'] = np.sqrt(p['Acceleration'] ** 2 + 9.81 ** 2)
    p['ES'] = np.tan(0.5*np.pi - np.arctan(9.81/p['Acceleration']))
    p['EM'] = p['g']/9.81
    p['EC'] = (155.4*p['ES'] ** 5 - 30.4*p['ES'] ** 4 - 43.3*p['ES'] ** 3 + 46.3*p['ES'] ** 2 + 19.5*p['ES'] + 3.6)*p['EM']
    p['MP'] = p['EC'] * p['Speed']
    p['MP Cumulative'] = p.groupby(['Name'])['MP'].cumsum()
    p.drop(['alpha', 'g', 'ES', 'EM'], axis=1, inplace=True)  # drop columns
    #%% calculate TRIMP
    p['heartbeat relative'] = p['heartbeat relative'].apply(pd.to_numeric, errors='coerce')  # object to int64
    p['HR zones'] = 0
    p.loc[(p['heartbeat relative'] >= 65) & (p['heartbeat relative'] < 72) , 'HR zones'] = 1  # find all rows that fulfills your conditions and set HR zones to 1
    p.loc[(p['heartbeat relative'] >= 72) & (p['heartbeat relative'] < 79) , 'HR zones'] = 2  # find all rows that fulfills your conditions and set HR zones to 2
    p.loc[(p['heartbeat relative'] >= 79) & (p['heartbeat relative'] < 86) , 'HR zones'] = 3  # find all rows that fulfills your conditions and set HR zones to 3
    p.loc[(p['heartbeat relative'] >= 86) & (p['heartbeat relative'] < 93) , 'HR zones'] = 4  # find all rows that fulfills your conditions and set HR zones to 4
    p.loc[(p['heartbeat relative'] >= 93), 'HR zones'] = 5  # find all rows that fulfills your conditions and set HR zones to 5
    p['TRIMP'] = np.cumsum(p['HR zones'] / 1200)  # calculate back to minutes (60000 / 50 ms)
    
df2 = pd.concat(df_list)  # make new Dataframe with new variables

#%% select period of training session
# https://www.epochconverter.com/

# df2['Timestamp'] = df2['Timestamp'].apply(pd.to_numeric, errors='coerce')  # object to int64

# df3 = df2[df2['Timestamp'].between(xxx,xxx)]  # fill xxx's with start timestamp and end timestamp

#%% - new Dataframe with calculated variables per player
TrainingVariables = pd.DataFrame(df2.groupby(['Name'], as_index = False)['TRIMP', 'Total Euclidean Distance', 'HID', 'Total Acc', 'Total Decc', 'MP Cumulative', 'MP'].max())
TrainingVariables2 = pd.DataFrame(df2.groupby(['Name'], as_index = False)['MP'].mean())
TrainingVariables['mean MP'] = TrainingVariables2['MP']

print(TrainingVariables)

#%% - plotting

# dfName1 = df2[df2['Name'] == playerName[0]]  # make dataframe per handballer
# dfName2 = df2[df2['Name'] == playerName[1]]
# dfName3 = df2[df2['Name'] == playerName[2]]
# dfName4 = df2[df2['Name'] == playerName[3]]
# dfName5 = df2[df2['Name'] == playerName[4]]
# dfName6 = df2[df2['Name'] == playerName[5]]

# CountAcc = dfName1['Acceleration'] > 2.0

# plt.plot(dfName1['MP'])

stop = timeit.default_timer()
print('Time: ', stop - start)



