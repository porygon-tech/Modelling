import sqlite3
import pandas as pd
import numpy as np
import math 

def encode(A):
	classes = np.unique(A)
	dictionary = dict(zip(classes, range(len(classes))))
	A_numeric = list(map(dictionary.get, A))
	return(A_numeric, dictionary)

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def splitdata(a, b, fraction):
	assert len(a) == len(b)
	length = len(a)
	train_num = math.floor(length * (1-fraction))
	return(a[0:train_num],b[0:train_num],a[train_num:],b[train_num:])



filepath = '/home/mroman/Master/RI_deliv_2/gdp_growth_forecast/'
con = sqlite3.connect(filepath + 'db.sqlite3') 

df = pd.read_sql_query('SELECT c.CountryCode, i.IndicatorCode, ci.Year, ci.Value FROM CountryIndicators as ci, Countries as c, Indicators as i WHERE ci.CountryCode=c.CountryCode AND i.IndicatorCode=ci.IndicatorCode ', con)

test = df.pivot_table(index=['CountryCode','Year'], columns="IndicatorCode", values='Value')
test['nextyearGDPgrowth'] = test['NY.GDP.MKTP.KD.ZG'].shift(periods=-1)


def fixvalue(value, fixedvalue, isNA=False):
	if isNA:
		return(fixedvalue)
	else:
		return(value)

def fixcol(col):
	mu = np.mean(col)
	sd = np.std(col)
	length = len(col)
	fixedvalues = np.random.normal(mu, sd, length)
	NAs = col.isin(['NaN'])
	return list(map(fixvalue, col,fixedvalues,NAs))

def fixtab(tab):
	for i in tab.columns:
		tab[i] = fixcol(tab[i])
	return tab


filtered = test.dropna(thresh=2/3*test.shape[0], axis=1)
filtered = fixtab(filtered)





classList = np.array(list(map(list, filtered.index)))

countryCodes_num, countryDict = encode(classList[:,0])
years_num = np.array(list(map(int, classList[:,1])))[:,np.newaxis]
countryCodes_num = np.array(countryCodes_num)[:,np.newaxis]

values = filtered.to_numpy()
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
