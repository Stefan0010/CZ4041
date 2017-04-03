from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lars
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd
import numpy as np 
from math import sqrt

np.seterr(divide='ignore', invalid='ignore')
#np.seterr(divide='raise')

cols = ["Customers", "Open", "Promo", "SchoolHoliday"]
cols += ["StateHoliday_0", "StateHoliday_a", "StateHoliday_b"]
cols += ["StateHoliday_c", "Weekends", "Weekdays", "StoreType_a", "StoreType_b"]
cols += ["StoreType_c", "StoreType_d", "Assortment_a", "Assortment_b", "Assortment_c"]
cols += ["HasCompetition", "IsDoingPromo2", "CompetitionDistance"]

colsRes = ["Sales"]


df = pd.read_csv("normalised_data.csv")
train_df = df.loc[df["Store"] != "Store"].astype(float)
train = train_df.as_matrix(cols)
trainRes = train_df.as_matrix(colsRes)

df2 = pd.read_csv("normalised_valData.csv")
val_df = df2.loc[df2["Store"] != "Store"].astype(float)
val = val_df.as_matrix(cols)
valRes = val_df.as_matrix(colsRes)


df3 =pd.read_csv("../data/115_split_rf.csv").astype(float)
valRes_d = df3.as_matrix(colsRes)

#rf = ElasticNet()
#rf = RandomForestRegressor(n_estimators=30, max_features=0.7)
#rf = Lars()
rf = svm.SVR()
rf.fit(train, trainRes.ravel())

results = rf.predict(val)

"""
for i in range(len(val)):
    print(val[i])
    print("results: " + str(results[i]) + "    valRes: " + str(valRes[i]))
"""


print "normalised_data"
print(results)
error = sqrt(mean_squared_error(valRes, results))
print (error)

filtr = (valRes!=0.).ravel()
temp1 = valRes[filtr].ravel()
temp2 = results[filtr].ravel()
rmspe = np.sqrt( np.sum(((temp1-temp2)/temp1)**2/len(temp1)) )
print rmspe

print "\n"
print "denormalised data"
hi = float(max(valRes_d))
lo = float(min(valRes_d))

m = hi-lo
results_d = np.add(np.multiply(np.divide(np.add(results, 1.0), 2.0), m), lo)

print results_d

error2 = sqrt(mean_squared_error(valRes_d, results_d))
print (error2)

filtr_ = (valRes_d!=0.).ravel()
temp1_ = valRes_d[filtr_].ravel()
temp2_ = results_d[filtr_].ravel()
rmspe_ = np.sqrt( np.sum(((temp1_-temp2_)/temp1_)**2/len(temp1_)) )
print rmspe_

