import sqlite3
import pandas as pd
import numpy as np

def encode(A):
	classes = np.unique(A)
	dictionary = dict(zip(classes, range(len(classes))))
	A_numeric = list(map(dictionary.get, A))
	return(A_numeric, dictionary)


filepath = '/home/mroman/Master/RI_deliv_2/gdp_growth_forecast/'
con = sqlite3.connect(filepath + 'db.sqlite3') 

df = pd.read_sql_query('SELECT c.CountryCode, i.IndicatorCode, ci.Year, ci.Value FROM CountryIndicators as ci, Countries as c, Indicators as i WHERE ci.CountryCode=c.CountryCode AND i.IndicatorCode=ci.IndicatorCode ', con)

test = df.pivot_table(index=['CountryCode','Year'], columns="IndicatorCode", values='Value')
test['nextyearGDPgrowth'] = test['NY.GDP.MKTP.KD.ZG'].shift(periods=-1)

#test.to_sql("formatted", con, if_exists="replace")

classList = np.array(list(map(list, test.index)))

countryCodes_num, countryDict = encode(classList[:,0])
years_num = np.array(list(map(int, classList[:,1])))[:,np.newaxis]
countryCodes_num = np.array(countryCodes_num)[:,np.newaxis]

values = test.to_numpy()
dataset = np.concatenate((countryCodes_num,years_num,values), axis=1)
'''
dataset[:,0] #numeric country code
dataset[:,1] #year
dataset[:,-1] #GDP to predict
'''

x = dataset[:,:-1]
y = dataset[:,-1]

indexes_2010 = np.where((x[:,1]==2010))
x_2010 = x[indexes_2010]
x = np.delete(x, indexes_2010, 0)
y = np.delete(y, indexes_2010, 0)
