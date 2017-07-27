

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# 读取数据
train_data = pd.read_csv('D:\\Files\\competion\\data_preprocess\\new_feature\\final_15.csv') 
test_data = pd.read_csv('D:\\Files\\competion\\data_preprocess\\new_feature\\final_21.csv') 
train_label = train_data['label']
test_label = test_data['label']
# 保存group信息
train_group = train_data['group']
test_group = test_data['group']

train_data.drop('group',axis=1, inplace=True)
train_data.drop('group_label',axis=1, inplace=True)
train_data.drop('label',axis=1, inplace=True)
test_data.drop('label',axis=1, inplace=True)
test_data.drop('group',axis=1, inplace=True)
test_data.drop('group_label',axis=1, inplace=True)


# 主函数
def main():
    t1 = time.time()
    cllf = model(train_data,train_label)
    score,TN,FN = model_roc_score(cllf,test_data,test_label)
    print ("score = "+ str(score) +", TN = "+ str(TN) +
          ", FN =" + str(FN))
    t2 = time.time()
    print("used time :"+ str(t2-t1))
    #保存模型
    joblib.dump(cllf, 'Randomforest.model')

# 定义model函数 ，最好分类结果：TF = 4654, FP = 5044 Score=70.77450485449361
def model(train_data,train_label):
    weight_list=[]
    for j in range(len(train_label)):
        if train_label[j] == 0:
            weight_list.append(1)
        if train_label[j] == 1:
            weight_list.append(24)
    clf = RandomForestClassifier(
                            n_estimators = 300,max_features='log2', 
                            max_depth = 4 ,
                            min_samples_leaf = 25,
                            min_samples_split = 8,
                            criterion='gini',
                            min_impurity_split=1e-7,class_weight={0:1,1:18}
                            ).fit(train_data,train_label)
    return clf

#ROC打分函数
def model_roc_score(clf,test_data,test_label):
    pre_label = clf.predict(test_data)
    N_fault = 0
    FP = 0
    TN = 0
    N_normal = 0
    FN = 0
    for i in range(len(test_label)):
        if test_label[i] == 1:  #实际标签为1
            N_fault = N_fault + 1
            if pre_label[i] == 0: # 错分为0
                FP = FP + 1
            else:
                TN = TN+1
        else: #实际标签为0
            N_normal = N_normal + 1
            if pre_label[i] == 1: #错分为1
                FN = FN + 1
    score = (1-0.5*(FN/N_normal)-0.5*(FP/N_fault))*100
    return score,TN,FN





