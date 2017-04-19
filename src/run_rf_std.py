from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd
import numpy as np 
from math import sqrt

np.seterr(divide='ignore', invalid='ignore')


testing = "../data/test_merged.csv"
training = "../data/train_merged.csv"


cols = ["Store", "Open", "Promo", "SchoolHoliday"]
cols += ["StateHoliday_0", "StateHoliday_a", "StateHoliday_b"]
cols += ["StateHoliday_c", "Weekends", "Weekdays", "StoreType_a", "StoreType_b"]
cols += ["StoreType_c", "StoreType_d", "Assortment_a", "Assortment_b", "Assortment_c"]
cols += ["HasCompetition", "IsDoingPromo2", "CompetitionDistance"]
cols += ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

colsRes = ["Sales"]

column = cols + colsRes

stores_df = pd.read_csv(testing)
stores = stores_df.as_matrix(["Store"])
stores = np.unique(stores)

def predict():
    """normalize training data"""
    for store in stores:                    #for each store
        print store
        train_df = pd.read_csv(training)
        train = train_df.loc[train_df["Store"] == store]
        trainArr = train.as_matrix(column)
        trainArr = trainArr.astype(int)
        toWrite = np.zeros((len(trainArr),1))
        if len(trainArr)>0:
            for j in range(1,len(trainArr[0])):   #for each column, store column excluded
                #temp = []
                z=[]
                n=[]
                z = trainArr[:,j].astype(float)
                std = np.std(z)
                mean = np.mean(z)
                n = np.divide(np.subtract(z, mean), std)
                n = np.nan_to_num(n)
                toWrite = np.column_stack((toWrite, n))
        else:
            continue

        print "use all training data"
        print "----------------------------------"
        print "training"
        print "----------------------------------"

        """use 80 percent of training data"""
        tw_len = len(toWrite)
        tr = 4*tw_len//5
        rf = RandomForestRegressor(n_estimators=100, max_depth=30)
        rf.fit(toWrite[:tr,1:26], toWrite[:tr,27])

        """validating data"""
        print "validating 20 percent of data"
        print "----------------------------------"
        val_result = rf.predict(toWrite[tr:, 1:26])

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
            z2 = testArr[:,k].astype(float)
            mean2 = np.mean(z2)
            std2 = np.std(z2)
            n2 = np.divide(np.subtract(z2, mean2), std2)
            n2 = np.nan_to_num(n2)

            toTest = np.column_stack((toTest, n2))

        print "predict testing data"
        print "----------------------------------"
        results = rf.predict(toTest[:,1:26])

        res = train.as_matrix(colsRes)

        mean3 = np.mean(res)
        std3 = np.std(res)
        results_d = np.add(np.multiply(results, std3), mean3)
        withID = np.column_stack((colID, results_d))


        print "write to result1"
        print "----------------------------------"
        df = pd.DataFrame(withID, columns = ["Id", "Sales"])
        df.to_csv('result1_std.csv', mode='a', index=False)


        """use half of training data"""
        print "use half of training data"
        print "----------------------------------"
        tr2 = tw_len//2

        rf2 = RandomForestRegressor(n_estimators=100, max_depth = 30)
        rf2.fit(toWrite[:tr2,1:26], toWrite[:tr2,27])

        """validating data with 50 percent of data"""
        print "validating 50 percent of data"
        print "----------------------------------"

        val_result2 = rf2.predict(toWrite[tr2:, 1:26])

        """predict testing data"""
        print "predict testing data"
        print "----------------------------------"

        results2 = rf2.predict(toTest[:,1:26])

        res2 = train.as_matrix(colsRes)
        mean4 = np.mean(res2)
        std4 = np.std(res2)

        results_d2 = np.add(np.multiply(results2, std4), mean4)

        withID2 = np.column_stack((colID, results_d2))


        print "write to result2"
        print "----------------------------------"

        df2 = pd.DataFrame(withID2, columns = ["Id", "Sales"])
        df2.to_csv('result2_std.csv', mode='a', index=False)

def sort_result():
    df = pd.read_csv('result1_std.csv')
    result = df.loc[df["Id"] != 'Id'] 
    Id = result.as_matrix(["Id"])
    Id = list(Id.flatten())
    Id = map(float, Id)
    Id = map(int, Id)
    sales = result.as_matrix(["Sales"])
    toWrite = np.column_stack((Id, sales))
    write = pd.DataFrame(toWrite, columns = ["Id", "Sales"])
    write = write.sort_values("Id", ascending=True)
    write["Id"] = write["Id"].astype(int)
    write.to_csv("result1_std.csv", encoding="utf-8", index=False)

    df2 = pd.read_csv('result2.csv')
    result2 = df2.loc[df["Id"] != 'Id']
    Id2 = result2.as_matrix(["Id"])
    Id2 = list(Id2.flatten())
    Id2 = map(float, Id2)
    Id2 = map(int, Id2)
    sales2 = result2.as_matrix(["Sales"])
    toWrite2 = np.column_stack((Id2, sales2))
    write2 = pd.DataFrame(toWrite2, columns = ["Id", "Sales"]) 
    write2 = write2.sort_values("Id", ascending=True)
    write2["Id"] = write2["Id"].astype(int)
    result2.to_csv("result2_std.csv", encoding="utf-8", index=False)

def run():
    #predict()
    sort_result()

if __name__ == "__main__":
    run()