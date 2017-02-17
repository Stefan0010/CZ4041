from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np 
from math import sqrt

url = "../data/1000_split_rf.csv"
val = "../data/115_split_rf.csv"


train_df = pd.read_csv(url)
train = train_df.astype(int)
train_df.head()

val_df = pd.read_csv(val)
val = val_df.astype(int)



cols = ["Store", "DayOfWeek", "Customer", "Open", "Promo"]
cols += ["StateHoliday", "StateHoliday_0", "StateHoliday_a", "StateHoliday_b"]
cols += ["StateHoliday_c", "Weekends", "Weekdays", "StoreType_a", "StoreType_b"]
cols += ["StoreType_c", "StoreType_d", "Assortment_a", "Assortment_b", "Assortment_c"]
cols += ["HasComptetion", "IsDoingPromo2", "CompetitionDistance"]

colsRes = ["Sales"]

trainArr = train.as_matrix(cols)
trainRes = train.as_matrix(colsRes)


valArr = val.as_matrix(cols)
valRes = val.as_matrix(colsRes)

n = [10,15,20]
for i in range(len(n)):
	rf = RandomForestRegressor(n_estimators = n[i])
	rf.fit(trainArr, trainRes.ravel())

	results = rf.predict(valArr)

	print ("number of trees =  " + str(n[i]))
	for i in range (50):
		print (str(i) + "  valRes:  " + str(valRes[i]))
		print (str(i) + "  results:  " + str(results[i]))

	error = sqrt(mean_squared_error(valRes, results))
	print (error)
	print """"""""""""""""""""



