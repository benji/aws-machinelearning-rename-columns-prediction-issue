import pandas as pd
import math, json, sys
import numpy as np

df1 = pd.read_csv('data/1stFlrSF/sample_predictions.csv')
df2 = pd.read_csv('data/FirstFlrSF/sample_predictions.csv')

df =  pd.DataFrame()
df['tag'] = df1['tag']
df['score1'] = df1['score']
df['score2'] = df2['score']

df['diff'] = (df['score2'] - df['score1']) / df['score1'] * 100

print df.sort_values('diff', ascending=False).head()