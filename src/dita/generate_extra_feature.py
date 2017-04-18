import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

testing = "../data/test_merged.csv"
training = "../../data/train_merged.csv"

tr_df= pd.read_csv(training)
dow = tr_df.as_matrix(["DayOfWeek"]).astype(int)
row = len(tr_df)
print dow[1]
features = np.zeros((row, 7))
c = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

lb = LabelBinarizer()
lb.fit(range(1,max(dow)+1))
onecode = lb.transform(dow)

print('{0}'.format(onecode[1]))


for day in c:
	i = c.index(day)
	tr_df[day] = pd.DataFrame(onecode[:,i], columns = [day])

tr_df.to_csv(training, encoding="utf-8", index=False)

