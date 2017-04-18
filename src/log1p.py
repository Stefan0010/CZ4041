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
cols += ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

colsRes = ["Sales"]

stores_df = pd.read_csv(testing)
stores = stores_df.as_matrix(["Store"])
stores = np.unique(stores)

col = len(cols)+len(colsRes)
column = cols + colsRes
col2 = ["toWrite"] + column


"""normalize training data"""
for store in stores:                    #for each store
    print store
    train_df = pd.read_csv(training)
    train = train_df.loc[train_df["Store"] == store]
    date = pd.to_datetime(train['Date'], format="%d/%m/%Y")
    month = date.dt.month
    year = date.dt.year
    trainArr = train.as_matrix(cols)   #without sales
    trainSales = train.as_matrix(colsRes)
    trainArr = trainArr.astype(int)
    trainArr = np.column_stack((trainArr, month, year))

    if len(trainArr)>0:
        z=[]
        z=trainSales.astype(float)
        log=np.log1p(z)
    else:
        continue
    toWrite = trainArr[:, :]
    print len(toWrite[0])
    print "use all training data"
    print "----------------------------------"
    print "training"
    print "----------------------------------"

    """use 80 percent of training data"""
    tw_len = len(toWrite)
    tr = 4*tw_len//5
    rf = RandomForestRegressor(n_estimators=100, max_depth=30)
    rf.fit(toWrite[:tr, :], log[:tr])

    """validating data"""
    print "validating 20 percent of data"
    print "----------------------------------"
    val_result = rf.predict(toWrite[tr:, :])

    """normalise testing data"""
    print "normalise testing data"
    print "----------------------------------"


    test_df = pd.read_csv(testing)
    test = test_df.loc[test_df["Store"] == store]

    
    testArr = test.as_matrix(cols)
    colID = test.as_matrix(["Id"]).astype(int)
    testArr = testArr.astype(int)

    date_test = pd.to_datetime(test['Date'], format="%d/%m/%Y")
    month_test = date_test.dt.month
    year_test = date_test.dt.year
    testArr = np.column_stack((testArr, month_test, year_test))
    print len(testArr[0])

    print "predict testing data"
    print "----------------------------------"
    results = rf.predict(testArr)

    results_d = np.expm1(results)
    withID = np.column_stack((colID, results_d))


    print "write to result1"
    print "----------------------------------"
    df = pd.DataFrame(withID, columns = ["Id", "Sales"])
    df.to_csv('result1_log_with_month.csv', mode='a', index=False)

    """use half of training data"""
    print "use half of training data"
    print "----------------------------------"
    tr2 = tw_len//2

    rf2 = RandomForestRegressor(n_estimators=100, max_depth = 30)
    rf2.fit(toWrite[:tr2,:], log[:tr2])

    """validating data with 50 percent of data"""
    print "validating 50 percent of data"
    print "----------------------------------"

    val_result2 = rf2.predict(toWrite[tr2:, :])

    """predict testing data"""
    print "predict testing data"
    print "----------------------------------"

    results2 = rf2.predict(testArr)
    results_d2 = np.expm1(results2)

    withID2 = np.column_stack((colID, results_d2))

    print "write to result2"
    print "----------------------------------"

    df2 = pd.DataFrame(withID2, columns = ["Id", "Sales"])
    df2.to_csv('result2_log_with_month.csv', mode='a', index=False)


