from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time

time_before = time.time()
df = pd.read_csv("../data/dt-data.csv")
print(df)

df_obj = df.select_dtypes(['object'])
df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
df.columns = df.columns.str.replace('\W','')
df['Occupied'] = df['Occupied'].str.replace('[^[a-zA-Z]', '')
df['Enjoy'] = df['Enjoy'].str.replace('[^[a-zA-Z]', '')
print(df)

x_feature = df.iloc[:, :-1]
print(x_feature)

x_dict = x_feature.T.to_dict().values()
#print(x_dict)
v = DictVectorizer(sparse=False)
x_training = v.fit_transform(x_dict)
print(x_training)

y_feature = df.iloc[:,-1]
le = LabelEncoder()

y_training = le.fit_transform(y_feature)
print(y_training)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(x_training, y_training)
time_after = time.time()

print(time_after-time_before)

df = pd.read_csv("../data/test.csv")

df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
df.columns = df.columns.str.replace('\W','')
df['Occupied'] = df['Occupied'].str.replace('[^[a-zA-Z]', '')
print(df)