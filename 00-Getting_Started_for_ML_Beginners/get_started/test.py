print("==========string split测试==========")
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
print("TRAIN_URL:")
print(TRAIN_URL)
print("TRAIN_URL.split('/'):")
print(TRAIN_URL.split('/'))
print("TRAIN_URL.split('/')[-1]:")
print(TRAIN_URL.split('/')[-1])

print("==========pandas.DataFrame.pop测试==========")
import pandas as pd
df = pd.DataFrame([('falcon', 'bird',    389.0),
    ('parrot', 'bird',     24.0),
    ('lion',   'mammal',   80.5),
    ('monkey', 'mammal',   100.0)],
    columns=('name', 'class', 'max_speed'))
print("DataFrame:") 
print(df)
print("DataFrame.pop('class'):")
print(df.pop('class'))

print("==========feature column测试==========")
print("DataFrame.keys():",df.keys())
import tensorflow as tf
my_feature_columns = []
for key in df.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print("feature column:")
print(my_feature_columns)
print("type of feature column:",type(my_feature_columns))

print("==========zip测试==========")
a=[1,2,3]
b=[1,3,5]
print("a:",a)
print("b:",b)
print("zip(a,b):")
for z in zip(a,b):
    print(z)

