from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np 
from math import sqrt


cols = ["Store", "Open", "Promo", "SchoolHoliday"]
cols += ["StateHoliday_0", "StateHoliday_a", "StateHoliday_b"]
cols += ["StateHoliday_c", "Weekends", "Weekdays", "StoreType_a", "StoreType_b"]
cols += ["StoreType_c", "StoreType_d", "Assortment_a", "Assortment_b", "Assortment_c"]
cols += ["HasCompetition", "IsDoingPromo2", "CompetitionDistance"]

colsRes = ["Sales"]



train_df = pd.read_csv("../data/train_merged.csv")
train = train_df.as_matrix(cols)
train = train.astype(int)
trainRes = train_df.as_matrix(colsRes).astype(int)

test_df = pd.read_csv("../data/test_merged.csv")
test = test_df.as_matrix(cols)
test = test.astype(int)




#rf = RandomForestRegressor(n_estimators = 200, max_features= 0.7, max_depth = 20)
rf = ElasticNet()
rf.fit(train, trainRes.ravel())

results = rf.predict(test)

df = pd.DataFrame(results, columns = ["Sales"])
df.to_csv("predicted_resultEN.csv")




