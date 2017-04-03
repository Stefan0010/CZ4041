from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd
import numpy as np 
from math import sqrt

np.seterr(divide='ignore', invalid='ignore')
#np.seterr(divide='raise')

testing = "../data/test_merged.csv"
training = "../data/train_merged.csv"


cols = ["Open", "Promo", "SchoolHoliday"]
cols += ["StateHoliday_0", "StateHoliday_a", "StateHoliday_b"]
cols += ["StateHoliday_c", "Weekends", "Weekdays", "StoreType_a", "StoreType_b"]
cols += ["StoreType_c", "StoreType_d", "Assortment_a", "Assortment_b", "Assortment_c"]
cols += ["HasCompetition", "IsDoingPromo2", "CompetitionDistance"]

colsRes = ["Sales"]

col = cols + colsRes
train_df = pd.read_csv(training)
trainArr = train_df.as_matrix(col)
trainRes = train_df.as_matrix(colsRes)
print len(trainArr[0])
toWrite = np.zeros((len(trainArr),1))

"""normalise training data"""

print "normalise training data"

for i in range(len(trainArr[0])):
	z=[]
	n=[]
	z = trainArr[:,i].astype(float)
	std = np.std(z)
	mean = np.mean(z)
	n = np.divide(np.subtract(z, mean), std)
	n= np.nan_to_num(n)
	toWrite = np.column_stack((toWrite, n))
	"""
	lo = float(min(trainArr[:,i]))    #find the minimum value in the column
	hi = float(max(trainArr[:,i]))    #find the maximum value in the column
	z = trainArr[:,i].astype(float)
	n=np.empty((len(trainArr), 1))
	denom = hi-lo
	div = np.divide(np.subtract(z,lo), denom)
	#n = np.subtract(np.dot(2.0, div), 1.0)
	n = np.nan_to_num(div)
	toWrite = np.column_stack((toWrite, n))
	"""


"""train data"""

print "train data"
print toWrite
rf = RandomForestRegressor(n_estimators=30, max_features=0.7)
rf.fit(toWrite[:, 1:19], toWrite[:, 20])


"""normalise testing data"""

print "normalise testing data"
test_df = pd.read_csv(testing)
colID = test_df.as_matrix(["Id"])
testArr = test_df.as_matrix(cols)

toTest = np.zeros((len(testArr), 1))

for j in range(0, len(testArr[0])):
	z2=[]
	n2=[]
	z2 = testArr[:,j].astype(float)
	mean2 = np.mean(z2)
	std2 = np.std(z2)
	n2 = np.divide(np.subtract(z2, mean2), std2)
	n2 = np.nan_to_num(n2)
	toTest = np.column_stack((toTest, n2))

	"""
	lo2 = float(min(testArr[:,j]))    #find the minimum value in the column
	hi2 = float(max(testArr[:,j]))    #find the maximum value in the column
	z2 = testArr[:,j].astype(float)
	n2=np.empty((len(testArr), 1))
	denom2 = hi2-lo2
	div2 = np.divide(np.subtract(z2,lo2), denom2)
	#n2 = np.subtract(np.dot(2.0, div2), 1.0)
	n2 = np.nan_to_num(div2)
	toTest = np.column_stack((toTest, n2))
	"""

"""predict testing data"""

print "predict testing data"
results = rf.predict(toTest[:,1:19])


"""denormalise predicted result"""

print "denormalise predicted result"
"""
hi3 = float(max(trainRes))
lo3 = float(min(trainRes))

m = hi3-lo3
#results_d = np.add(np.multiply(np.divide(np.add(results, 1.0), 2.0), m), lo3)
results_d = np.add(np.multiply(results, m), lo3)
"""

mean3 = np.mean(trainRes)
std3 = np.std(trainRes)

results_d = np.add(np.multiply(results, std3), mean3)
withID = np.column_stack((colID, results_d))


"""write to csv"""

print "write to csv"
df2 = pd.DataFrame(withID, columns = ["Id", "Sales"])
df2.to_csv('random_forest_one_model_std.csv')

print "done"