from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import time
t1 = time.time()
train_data = pd.read_csv('D:\\iot\\competition\\data\\dealed\\data_15_dealed.csv')
test_data = pd.read_csv('D:\\iot\\competition\\data\\dealed\\data_21_dealed.csv')

y_train_label = train_data['label']
train_data.drop('label',axis=1, inplace=True)

y_test_label = test_data['label']
test_data.drop('label',axis=1, inplace=True)

cl = model(train_data,y_train_label)
cl.score(test_data,y_test_label)
t2 = time.time()
print(t2-t1)



def model(x,y):
    weight_list = []
    for j in range(len(y)):
        if y[j] == 1:
            weight_list.append(15)
        if y[j] == 0:
            weight_list.append(1)
    clf = GradientBoostingClassifier(
          loss='deviance', n_estimators=300,
          learning_rate=0.1,
          max_depth=10, random_state=0,
          min_samples_split=200,
          min_samples_leaf=250,
          subsample=1.0,
          max_features='sqrt').fit(x, y, weight_list)
    return clf