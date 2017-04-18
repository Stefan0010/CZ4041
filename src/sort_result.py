import pandas as pd
import numpy as np

df = pd.read_csv('result1_log_with_month.csv')
result = df.loc[df["Id"] != 'Id'] 
Id = result.as_matrix(["Id"])
sales = result.as_matrix(["Sales"])
sales = sales.clip(0)
toWrite = np.column_stack((Id, sales))
write = pd.DataFrame(toWrite, columns = ["Id", "Sales"])
write.to_csv("rf1_log_with_month.csv", encoding="utf-8")

df2 = pd.read_csv('result2_log_with_month.csv')
result2 = df2.loc[df["Id"] != 'Id']
Id2 = result2.as_matrix(["Id"])
sales2 = result2.as_matrix(["Sales"])
sales2 = sales2.clip(0)
toWrite2 = np.column_stack((Id2, sales2))
write2 = pd.DataFrame(toWrite2, columns = ["Id", "Sales"]) 
result2.to_csv("rf2_log_with_month.csv", encoding="utf-8")