from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import time
def main():	
	train_data = pd.read_csv('D:\\iot\\competition\\data\\new_feature\\final_15.csv')
	test_data = pd.read_csv('D:\\iot\\competition\\data\\new_feature\\final_21.csv')
	#train_data = pd.read_csv('D:\\iot\\competition\\data\\new_feature\\final_21.csv')
	#test_data = pd.read_csv('D:\\iot\\competition\\data\\new_feature\\final_15.csv')

	y_train_label = train_data['label']
	train_data.drop('label',axis=1, inplace=True)
	train_data.drop('group',axis=1, inplace=True)
	train_data.drop('group_label',axis=1, inplace=True)

	y_test_label = test_data['label']
	y_test_group = test_data['group']
	test_data.drop('label',axis=1, inplace=True)
	test_data.drop('group',axis=1, inplace=True)
	test_data.drop('group_label',axis=1, inplace=True)

	
	t1 = time.time()   
	clff = model(train_data,y_train_label)
	s,n,TP = cal_score(clff,test_data,y_test_label,0.75)
	print( "score = " + str(s) + ", Negtive =" + str(n) + 
		  "(" + str(TP) + ")" + "\n")


	t2 = time.time()
	print("used time :" + str(t2-t1))

	
	#保存模型
	joblib.dump(clff, 'LR.model')

#定义model函数
def model(train_data,y_train_label):
    weight_list = []
    for j in range(len(y_train_label)):
        if y_train_label[j] == 1:
            weight_list.append(10)
        if y_train_label[j] == 0:
            weight_list.append(1)
    clf = LogisticRegression(penalty = 'l2',
                            class_weight = 'balanced',
                            C = 0.2,
                            solver = 'liblinear'
                            ).fit(train_data, y_train_label, weight_list)
    return clf
	

#按照group重新标记label,同一个group中所有样本都有同样的label
#计算最终的分数	
def cal_score(cl,test_data,y_test_label,threshold):
    #pre_label = cl.predict(test_data)
    pre_label = [0]*len(y_test_label)
    pre_label_pro = cl.predict_proba(test_data)
#     for i in range(len(pre_label_pro)):
#         if pre_label_pro[i][1] > threshold:
#             pre_label[i] = 1    
    temp = 0
    times = 0
    group_start = 0
    for i in range(len(y_test_group)-1):
        if y_test_group[i] == y_test_group[i+1]:
            temp = temp + pre_label_pro[i][1]
            times = times + 1
        else:
			temp = temp + pre_label_pro[i][1]
            times = times + 1
            if temp/times >= threshold:
                for j in range(group_start,i):
                    pre_label[j] = 1
            group_start = i+1
			temp = 0
			times = 0
        if i == len(y_test_group)-2:
            if temp/times >= threshold:
                for j in range(group_start,i):
                    pre_label[j] = 1
    
    n_f = 0
    n_n = 0
    FP = 0
    FN = 0
    TP = 0
    n_label = 0
    for i in range(0,len(y_test_label)):
        if pre_label[i] == 1:
            n_label = n_label+1
        if y_test_label[i] == 1:
            n_f = n_f + 1
            if pre_label[i] == 0:
                FP = FP + 1
            else:
                TP = TP + 1
        else: #y_test_label[i] == 0
            n_n = n_n + 1
            if pre_label[i] == 1:
                FN = FN + 1
    score = (1-0.5*(FN/n_n)-0.5*(FP/n_f))*100
    return score,n_label,TP