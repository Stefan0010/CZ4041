from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np 
from math import sqrt

url = "../data/1000_split_rf.csv"
val = "../data/115_split_rf.csv"


train_df = pd.read_csv(url)
train = train_df.astype(int)

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

#alpha=27
en = ElasticNetCV()
en.fit(trainArr, trainRes.ravel())

results = en.predict(valArr)

filtr = (valRes!=0.).ravel()
temp1 = valRes[filtr].ravel()
temp2 = results[filtr].ravel()
rmspe = np.sqrt( np.sum(((temp1-temp2)/temp1)**2/len(temp1)) )

print rmspe
