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


cols = ["Store", "Open", "Promo", "SchoolHoliday"]
cols += ["StateHoliday_0", "StateHoliday_a", "StateHoliday_b"]
cols += ["StateHoliday_c", "Weekends", "Weekdays", "StoreType_a", "StoreType_b"]
cols += ["StoreType_c", "StoreType_d", "Assortment_a", "Assortment_b", "Assortment_c"]
cols += ["HasCompetition", "IsDoingPromo2", "CompetitionDistance"]

colsRes = ["Sales"]

stores_df = pd.read_csv(testing)
stores = stores_df.as_matrix(["Store"])
stores = np.unique(stores)

col = len(cols)+len(colsRes)
column = cols + colsRes
col2 = ["toWrite"] + column

rmspe = list()
rmspe2 = list()

"""normalize training data"""
for store in stores:                    #for each store
    print store
    train_df = pd.read_csv(training)
    train = train_df.loc[train_df["Store"] == store]
    trainArr = train.as_matrix(column)
    trainArr = trainArr.astype(int)
    #store = np.empty((len(trainArr),1))
    #store.fill(i)
    toWrite = np.zeros((len(trainArr),1))
    #toWrite = np.column_stack((toWrite, store))
    if len(trainArr)>0:
        for j in range(1,len(trainArr[0])):   #for each column, store column excluded
            #temp = []
            z=[]
            n=[]
            lo = float(min(trainArr[:,j]))    #find the minimum value in the column
            hi = float(max(trainArr[:,j]))    #find the maximum value in the column
            z = trainArr[:,j].astype(float)
            n=np.empty((len(trainArr), 1))
            denom = hi-lo
            div = np.divide(np.subtract(z,lo), denom)
            n = np.subtract(np.dot(2.0, div), 1.0)
            n = np.nan_to_num(n)
            toWrite = np.column_stack((toWrite, n))
        #df = pd.DataFrame(toWrite, columns=col2)
        #df.to_csv('normalised_data.csv', mode='a')
    else:
        continue

    print "use all training data"
    print "----------------------------------"
    print "training"
    print "----------------------------------"

    """use all training data"""
    tw_len = len(toWrite)
    tr = 4*tw_len//5
    rf = RandomForestRegressor(n_estimators=30, max_features=0.7)
    rf.fit(toWrite[:tr,1:19], toWrite[:tr,20])

    """normalise testing data"""
    print "normalise testing data"
    print "----------------------------------"


    test_df = pd.read_csv(testing)
    test = test_df.loc[test_df["Store"] == store]

    
    testArr = test.as_matrix(cols)
    colID = test.as_matrix(["Id"]).astype(int)
    #print colID
    testArr = testArr.astype(int)
    toTest = np.zeros((len(testArr),1))
    for k in range(1, len(testArr[0])):
        z2=[]
        n2=[]
        lo2 = float(min(testArr[:,k]))    #find the minimum value in the column
        hi2 = float(max(testArr[:,k]))    #find the maximum value in the column
        z2 = testArr[:,k].astype(float)
        n2=np.empty((len(testArr), 1))
        denom2 = hi2-lo2
        div2 = np.divide(np.subtract(z2,lo2), denom2)
        n2 = np.subtract(np.dot(2.0, div2), 1.0)
        n2 = np.nan_to_num(n2)
        toTest = np.column_stack((toTest, n2))

    #toTest = np.column_stack((colID, toTest))
    print "predict testing data"
    print "----------------------------------"
    results = rf.predict(toTest[:,1:19])
    #results = rf.predict(toWrite[tr:,1:19])

    res = train.as_matrix(colsRes)
    testRes = res[:,-1]
    hi3 = float(max(testRes))
    lo3 = float(min(testRes))

    m = hi3-lo3
    results_d = np.add(np.multiply(np.divide(np.add(results, 1.0), 2.0), m), lo3)
    withID = np.column_stack((colID, results_d))
    #print withID

    print "write to result1"
    print "----------------------------------"
    df = pd.DataFrame(withID, columns = ["Id", "Sales"])
    df.to_csv('result1.csv', mode='a')

    #print results_d
    """
    error2 = sqrt(mean_squared_error(testRes, results_d))
    print (error2)

    filtr_ = (testRes!=0.).ravel()
    temp1_ = testRes[filtr_].ravel()
    temp2_ = results_d[filtr_].ravel()
    rmspe_ = np.sqrt( np.sum(((temp1_-temp2_)/temp1_)**2/len(temp1_)) )
    print rmspe_
    rmspe.append(rmspe_)
    """

    """use half of training data"""
    print "use half of training data"
    print "----------------------------------"
    tr2 = tw_len//2

    rf2 = RandomForestRegressor(n_estimators=30, max_features=0.7)
    rf2.fit(toWrite[:tr2,1:-2], toWrite[:tr2,-1])

    results2 = rf2.predict(toTest[:,1:19])

    res2 = train.as_matrix(colsRes)
    testRes2 = res2[:,-1]
    hi3 = float(max(testRes2))
    lo3 = float(min(testRes2))

    m = hi3-lo3
    results_d2 = np.add(np.multiply(np.divide(np.add(results2, 1.0), 2.0), m), lo3)
    withID2 = np.column_stack((colID, results_d2))
    #print withID2

    print "write to result2"
    print "----------------------------------"

    df2 = pd.DataFrame(withID2, columns = ["Id", "Sales"])
    df2.to_csv('result2.csv', mode='a')

    """
    error2 = sqrt(mean_squared_error(testRes2, results_d2))
    print (error2)

    filtr_ = (testRes2!=0.).ravel()
    temp1_ = testRes2[filtr_].ravel()
    temp2_ = results_d2[filtr_].ravel()
    rmspe_ = np.sqrt( np.sum(((temp1_-temp2_)/temp1_)**2/len(temp1_)) )
    print rmspe_
    rmspe2.append(rmspe_)
    """


"""
avg_rmspe = sum(rmspe)/len(rmspe)
avg_rmspe2 = sum(rmspe2)/len(rmspe2)
print "avg rmspe: " + str(avg_rmspe)
print "avg rmspe2: " + str(avg_rmspe2)

"""

