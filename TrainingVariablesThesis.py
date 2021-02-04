
"""
@author: JacobS
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance
import timeit
import matplotlib.pyplot as plt

start = timeit.default_timer()

#%% - import
# Importing data set at 20 Hz including:
# True time, heartbeat relative, heartbeat, speed relative, speed, acceleration, X, Y, Name

df = pd.read_csv("26-01-2021meiden.csv", sep = ',', encoding = "ISO-8859-1")  # using encoding due to Loïs instead of Lois
pd.set_option('display.max_columns', 50)

# check if the manually removed errors are still in the raw data file # yes they are!
# df3 = df[(df['X'] < 20.5) & (df['X'] > -20.5) & (df['Y'] < 11.0) & (df['Y'] > -11.0) & (df['Speed'] == 0) & (df['Acceleration'] == 0) & (df['relative speed'] > 0)]

df['Time'] = pd.to_datetime(df['Date/Time'])  # rename Date/Time to Time 
df.drop(['Date/Time'], axis=1, inplace=True)  # drop Date/Time

#%% calculate Timestamp
def transform (dtime):  # use function
    return dtime.timestamp()*1000
df['Timestamp'] = (df.apply(lambda row: transform(row.Time), axis = 1))  # calculate timestamp of every row

#%% - pre-processing data
df['X'].fillna(method='pad', inplace=True)  # remove 0 with previous row value
df['Y'].fillna(method='pad', inplace=True)  # remove 0 with previous row value

#%% pre-process incorrect data
df.loc[df['X'] > 20.5, ['Speed', 'Acceleration', 'heartbeat relative', 'heartbeat']] = 0
df.loc[df['X'] < -20.5, ['Speed', 'Acceleration', 'heartbeat relative', 'heartbeat']] = 0
df.loc[df['Y'] > 11.0, ['Speed', 'Acceleration', 'heartbeat relative', 'heartbeat']] = 0
df.loc[df['Y'] < -11.0, ['Speed', 'Acceleration', 'heartbeat relative', 'heartbeat']] = 0
df.loc[df['Speed'] > 8.0, ['Speed', 'relative speed']] = 0
df.loc[df['Acceleration'] < -5.0, ['Acceleration', 'relative speed']] = 0
df.loc[df['Acceleration'] > 5.0, ['Acceleration', 'relative speed']] = 0

# Python to csv  

#%% - making dictionary and list 
df['Name'] = df['Name'].astype('category')
playerName = df['Name'].unique().tolist()

df_dict = {name: df.loc[df['Name'] == name] for name in playerName}
df_list = [df.loc[df['Name'] == name] for name in playerName]

 
#%% - calculate variables per player

# calculate Euclidean Distance 
def calc_Euclidean(X, Y):  # calculate distance covered
    return distance.euclidean(X, Y)
for p in df_list:
    p['Euclidean Distance'] = p.apply(lambda row: calc_Euclidean(row.X, row.Y), axis=1) 
    p['Euclidean Distance Diff'] = abs(p['Euclidean Distance'] - p['Euclidean Distance'].shift(1))
    p.loc[(p['Speed'] == 0) & (p['Acceleration'] == 0) , 'Euclidean Distance Diff'] = 0
    p.loc[(p['Euclidean Distance Diff'] > 0.8), 'Euclidean Distance Diff'] = 0
    #%% remove wrong distances due to leaving the field
    p['Diff Timestamp'] = p['Timestamp'] - p['Timestamp'].shift(1) 
    p.loc[(p['Diff Timestamp'] > 100.0),  'Euclidean Distance Diff'] = 0
    p['Total Euclidean Distance'] = np.cumsum(p['Euclidean Distance Diff'])
    #%% calculate training duration
    p['On Field'] = 1
    p.loc[(p['X'] > 20.5) | (p['X'] < -20.5) | (p['Y'] > 11.0) | (p['Y'] < -11.0), 'On Field'] = 0  # if players are outside the field
    p['heartbeat'] = p['heartbeat'].fillna(0)  # change nan to 0
    p.loc[(p['Speed'] == 0) & (p['Acceleration'] == 0) & (p['heartbeat'] == 0), 'On Field'] = 0  # On field is 0 when there is no signal
    p['On Field'] = p.groupby(['Name'])['On Field'].cumsum() / 20  # calculate seconds on field
    p['Training Time'] = pd.to_datetime(p["On Field"], unit='s').dt.strftime("%H:%M:%S")
    p.drop(['On Field'], axis=1, inplace=True)
    #%% calculate HID
    # p['relative speed'] = p['relative speed'].apply(pd.to_numeric, errors='coerce')  # object to int64
    p['High Intensity Distance'] = p.loc[(p['relative speed'] >= 70), 'Euclidean Distance Diff']  # find distance covered when speed > 70%
    p['High Intensity Distance'] = p['High Intensity Distance'].fillna(value=0)
    p.loc[(p['High Intensity Distance'] > 0.8) , 'High Intensity Distance'] = 0  # choose which value is preferred to exclude outliers
    p['HID'] = p.groupby(['Name'])['High Intensity Distance'].cumsum()  # cumulative sum of all HID meters
    #%% calculate Acc > 2.0 m/s2
    p['Acc > 2.0'] = 0 # add a class column with 0 as default value
    p['Acceleration'].replace(to_replace=0, method='ffill')
    p.loc[(p['Acceleration'] > 2.2) & (p['Speed'] > 2) , 'Acc > 2.0'] = 1 # find all rows that fulfills your conditions and set Acc > 2.0 m/s2 to 1
    p['Diff Acc'] = p['Acc > 2.0'] + p['Acc > 2.0'].shift(1)  # if 2 then acc >= 100 ms, if 0 or 1 then acc <= 50 ms
    p['Diff Acc'] = p['Diff Acc']**2  # square the values to 0, 1, 4 to calculate differences
    p['Diff2 Acc'] = p['Diff Acc'] - p['Diff Acc'].shift(1)  # if 3 then acc condition is met
    p.loc[(p['Diff2 Acc'] == 3), 'Acc'] = 1  # find every acc
    p['Total Acc'] = np.cumsum(p['Acc'])  # count all acc
    p.drop(['Acc > 2.0', 'Acc', 'Diff Acc', 'Diff2 Acc'], axis=1, inplace=True)  # drop acc calculation columns
    #%% calculate Decc < -2.0 m/s2
    p['Decc > 2.0'] = 0 # add a column with 0 as default value
    p.loc[(p['Acceleration'] < -2.2) & (p['Speed'] > 2) , 'Decc > 2.0'] = 1 #  & (p['Speed'] >= 2)  find all rows that fulfills your conditions and set Acc > 2.0 m/s2 to 1
    p['Diff Decc'] = p['Decc > 2.0'] + p['Decc > 2.0'].shift(1)
    p['Diff Decc'] = p['Diff Decc']**2
    p['Diff2 Decc'] = p['Diff Decc'] - p['Diff Decc'].shift()
    p.loc[(p['Diff2 Decc'] == 3), 'Decc'] = 1
    p['Total Decc'] = np.cumsum(p['Decc'])
    p.drop(['Decc > 2.0', 'Decc', 'Diff Decc', 'Diff2 Decc'], axis=1, inplace=True)  # drop decc calculation columns   
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
    p['HR zone 0'] = 0
    p['HR zone 1'] = 0
    p['HR zone 2'] = 0 
    p['HR zone 3'] = 0
    p['HR zone 4'] = 0
    p['HR zone 5'] = 0
    p.loc[(p['heartbeat relative'] < 65), 'HR zone 0'] = 1  # find all rows that fulfills your conditions and set HR zones to 5
    p.loc[(p['heartbeat relative'] >= 65) & (p['heartbeat relative'] < 72) , 'HR zone 1'] = 1  # find all rows that fulfills your conditions and set HR zones to 1
    p.loc[(p['heartbeat relative'] >= 72) & (p['heartbeat relative'] < 79) , 'HR zone 2'] = 1  # find all rows that fulfills your conditions and set HR zones to 2
    p.loc[(p['heartbeat relative'] >= 79) & (p['heartbeat relative'] < 86) , 'HR zone 3'] = 1  # find all rows that fulfills your conditions and set HR zones to 3
    p.loc[(p['heartbeat relative'] >= 86) & (p['heartbeat relative'] < 93) , 'HR zone 4'] = 1  # find all rows that fulfills your conditions and set HR zones to 4
    p.loc[(p['heartbeat relative'] >= 93), 'HR zone 5'] = 1  # find all rows that fulfills your conditions and set HR zones to 5
    p['Time zone 0'] = np.cumsum(p['HR zone 0'] / 20)  # calculate seconds in HR zone
    p['Time zone 0'] = pd.to_datetime(p["Time zone 0"], unit='s').dt.strftime("%H:%M:%S")  # seconds to HH:MM:SS
    p['Time zone 1'] = np.cumsum(p['HR zone 1'] / 20)  # calculate seconds in HR zone
    p['Time zone 1'] = pd.to_datetime(p["Time zone 1"], unit='s').dt.strftime("%H:%M:%S")  # seconds to HH:MM:SS    
    p['Time zone 2'] = np.cumsum(p['HR zone 2'] / 20)  # calculate seconds in HR zone
    p['Time zone 2'] = pd.to_datetime(p["Time zone 2"], unit='s').dt.strftime("%H:%M:%S")  # seconds to HH:MM:SS    
    p['Time zone 3'] = np.cumsum(p['HR zone 3'] / 20)  # calculate seconds in HR zone
    p['Time zone 3'] = pd.to_datetime(p["Time zone 3"], unit='s').dt.strftime("%H:%M:%S")  # seconds to HH:MM:SS    
    p['Time zone 4'] = np.cumsum(p['HR zone 4'] / 20)  # calculate seconds in HR zone
    p['Time zone 4'] = pd.to_datetime(p["Time zone 4"], unit='s').dt.strftime("%H:%M:%S")  # seconds to HH:MM:SS    
    p['Time zone 5'] = np.cumsum(p['HR zone 5'] / 20)  # calculate seconds in HR zone
    p['Time zone 5'] = pd.to_datetime(p["Time zone 5"], unit='s').dt.strftime("%H:%M:%S")  # seconds to HH:MM:SS
    p['TRIMP'] = np.cumsum((p['HR zone 1'] * 1 + p['HR zone 2'] * 2 + p['HR zone 3'] * 3 + p['HR zone 4'] * 4 + p['HR zone 5'] * 5) / 1200)  # calculate TRIMP by calculating in minutes (60000 / 50 ms)
    
df2 = pd.concat(df_list)  # make new Dataframe with new variables

#%% select period of training session
# https://www.epochconverter.com/ 

# df2['Timestamp'] = df2['Timestamp'].apply(pd.to_numeric, errors='coerce')  # object to int64

# df3 = df2[df2['Timestamp'].between(xxx,xxx)]  # fill xxx's with start timestamp and end timestamp

#%% - new Dataframe with calculated variables per player
TrainingVariables = pd.DataFrame(df2.groupby(['Name'], as_index = False)['Training Time', 'Time zone 0', 'Time zone 1', 'Time zone 2', 'Time zone 3', 'Time zone 4', 'Time zone 5', 'TRIMP', 'Total Euclidean Distance', 'HID', 'Total Acc', 'Total Decc', 'MP Cumulative', 'MP'].max())

meanMP = pd.DataFrame(df2.groupby(['Name'], as_index = False)['MP'].mean())  # calculate mean MP in new Dataframe
TrainingVariables['mean MP'] = meanMP['MP']  # add mean MP to Training Variables


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



