from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd
import numpy as np 
from math import sqrt

np.seterr(divide='ignore', invalid='ignore')
#np.seterr(divide='raise')

url = "../data/1000_split_rf.csv"
val_url = "../data/115_split_rf.csv"


cols = ["Store", "Customers", "Open", "Promo", "SchoolHoliday"]
cols += ["StateHoliday_0", "StateHoliday_a", "StateHoliday_b"]
cols += ["StateHoliday_c", "Weekends", "Weekdays", "StoreType_a", "StoreType_b"]
cols += ["StoreType_c", "StoreType_d", "Assortment_a", "Assortment_b", "Assortment_c"]
cols += ["HasCompetition", "IsDoingPromo2", "CompetitionDistance"]

colsRes = ["Sales"]

col = len(cols)+len(colsRes)
column = cols + colsRes
col2 = ["toWrite"] + column

"""normalize training data"""
for i in range(1,1116):                    #for each store
    print i
    val_df= pd.read_csv(val_url).astype(int)
    val = val_df.loc[val_df["Store"] == i]
    valArr = val.as_matrix(column)
    store = np.empty((len(valArr),1))
    store.fill(i)
    toWrite = np.zeros((len(valArr),1))
    toWrite = np.column_stack((toWrite, store))
    if len(valArr)>0:
        for j in range(1,len(valArr[0])):   #for each column, store column excluded
            #temp = []
            z=[]
            n=[]
            lo = float(min(valArr[:,j]))    #find the minimum value in the column
            hi = float(max(valArr[:,j]))    #find the maximum value in the column
            z = valArr[:,j].astype(float)
            n=np.empty((len(valArr), 1))
            denom = hi-lo
            div = np.divide(np.subtract(z,lo), denom)
            n = np.subtract(np.dot(2.0, div), 1.0)
            n = np.nan_to_num(n)
            toWrite = np.column_stack((toWrite, n))
        df = pd.DataFrame(toWrite, columns=col2)
        df.to_csv('normalised_valData.csv', mode='a')
    else:
        continue
