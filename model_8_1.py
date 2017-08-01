from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
import numpy as np
import pandas as pd
import time    


def mian():
	#训练集，交叉验证数据集
	train_data = pd.read_csv('new_train_40.csv')
	y = train_data['label']
	train_group = train_data['group']
	x = train_data.drop(['label','group'],axis=1,inplace = False)


	#按照比例划分训练集
	X_train, X_test, y_train, y_test = cross_validation.train_test_split( x, y, stratify=y, test_size=0.3, random_state=1)
	X_train = X_train.reset_index(drop=True)
	y_train = y_train.reset_index(drop=True)
	X_test = X_test.reset_index(drop=True)
	y_test = y_test.reset_index(drop=True)
	
	
	#读取预测集
	test_data = pd.read_csv('new_test_40.csv')
	temp_time = test_data['time']
	temp_time = temp_time.astype(int)
	temp_group = test_data['group']
	test_data = test_data.drop(['time','group'],axis=1,inplace = False)
	
	#预测结果并输出
	pre_label = predict(clff,test_data,0.65)

	#按照group信息做调整，保证每一个group都是同一个label
	group_start = 0
	is_one = 0
	for i in range(len(pre_label)-1):
		if temp_group[i] != temp_group[i+1]:
			is_one = 0
			for j in range(0,i-group_start+1):
				if pre_label[i-j] == 1:
					is_one += 1
			if is_one >= 2:
				for j in range(0,i-group_start+1):
					pre_label[i-j] = 1
			if is_one == 1:
				for j in range(0,i-group_start+1):
					pre_label[i-j] = 0
			group_start = i + 1

	output_name = 'new_test_RF.csv'
	output(output_name,temp_time,pre_label)
	
	
#调参专用
def tiaocan():
	
	t1 = time.time()  
	#parameter = range(1,40,3)
	parameter =[0.6,0.65,0.7,0.75,0.8,0.85]
	#parameter =[0.05,0.1,0.2,1,1.5,2,2.5,3,3.5,4,4.5,5]
	#parameter=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	for i in range(len(parameter)):
		#clff = LR_model(X_train,y_train,parameter[i])
		#clff = LR_model(X_train,list(y_train),parameter[i])
		#clff = xgb_model(X_train,y_train,1)
		clff = RF_model(X_train,y_train)

		s,n,TP,pre_label = cal_score(clff,X_test,y_test,parameter[i])
		print( "min_samples_split = "+ str(parameter[i]) + ", score = " + str(s) + ", Negtive =" + str(n) + 
			  "(" + str(TP) + ")" + "\n")


	t2 = time.time()
	print("used time :" + str(t2-t1))
	
	
#RF_model
def RF_model(train_data,train_label):
    weight_list=[]
    for j in range(len(train_label)):
        if train_label[j] == 0:
            weight_list.append(1)
        if train_label[j] == 1:
            weight_list.append(40)
    clf = RandomForestClassifier(
                            n_estimators = 32,#13
                            #max_features='sqrt', 
                            max_depth = 8 ,
                            min_samples_leaf = 20,#25
                            min_samples_split = 6,#8
                            criterion='entropy',
                            ).fit(train_data,train_label,weight_list)
    return clf
	
	
	
#xgb模型
def xgb_model(train_data,y_train_label,paramter):
    weight_list = []
    for j in range(len(y_train_label)):
        if y_train_label[j] == 1:
            weight_list.append(20)
        if y_train_label[j] == 0:
            weight_list.append(1)     
    clf = XGBClassifier(
    silent=0 ,
    learning_rate= 0.3, # 如同学习率
    min_child_weight=1, 
    max_depth = 5, # 构建树的深度，越大越容易过拟合
    gamma = 0.2,  
    subsample=1, # 随机采样训练样本 训练实例的子采样比
    max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
    colsample_bytree=1, # 生成树时进行的列采样 
    reg_lambda = 1.2,  # 控制模型复杂度的权重值的L2正则化项参数
    reg_alpha = 0.7, # L1 正则项参数
    scale_pos_weight = 2.2, 
    n_estimators = 65, #树的个数
    seed=0 #随机种子
    )
    clf.fit(train_data,y_train_label,weight_list)
    return clf



#LR模型
def LR_model(train_data,y_train_label):
    weight_list = []
    for j in range(len(y_train_label)):
        if y_train_label[j] == 1:
            weight_list.append(5)
        if y_train_label[j] == 0:
            weight_list.append(1)
    clf = LogisticRegression(penalty = 'l2',
                            class_weight = 'balanced',
                            C = 0.2,
                            solver = 'liblinear'
                            ).fit(train_data, y_train_label, weight_list)
    return clf	
	
	
	
	
#计算交叉验证的分数	
def cal_score(cl,test_data,y_test_label,threshold):
    #pre_label = cl.predict(test_data)
    pre_label = [0]*len(y_test_label)
    pre_label_pro = cl.predict_proba(test_data)
	
    for i in range(len(pre_label_pro)):
        if pre_label_pro[i][1] > threshold:
            pre_label[i] = 1   

#     group_start = 0
#     is_one = 0
#     for i in range(len(pre_label)-1):
#         if train_group[i] != train_group[i+1]:
#             is_one = 0
#             for j in range(0,i-group_start+1):
#                 if pre_label[i-j] == 1:
#                     is_one += 1
#             if is_one >= 2:
#                 for j in range(0,i-group_start+1):
#                     pre_label[i-j] = 1
#             if is_one == 1:
#                 for j in range(0,i-group_start+1):
#                     pre_label[i-j] = 0
#             group_start = i + 1
        
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
    return score,n_label,TP,pre_label	
	
	
	
	
	
#预测
def predict(cl,test_data,threshold):
    pre_label = [0]*len(test_data)
    pre_label_pro = cl.predict_proba(test_data)
    for i in range(len(pre_label_pro)):
        if pre_label_pro[i][1] > threshold:
            pre_label[i] = 1 
    return pre_label

	
	
#输出结果	
def output(name,time,final_label):
    start_time =[]
    end_time = []
    for i in range(len(final_label)-1):
        if (final_label[i] == 0) & (final_label[i+1] == 1):
            start_time.append(time[i+1])
        if (final_label[i] == 1) & (final_label[i+1] == 0):
            end_time.append(time[i+1]-1)

    if (final_label[len(final_label)-2] == 0) & (final_label[len(final_label)-1] == 1):
        start_time.append(time[len(final_label)-1])
        end_time.append(0)     #如果是这种情况，就最后手动加


    if (final_label[len(final_label)-2] == 1) & (final_label[len(final_label)-1] == 1):
        end_time.append(0)

    final_pre_result = pd.DataFrame(start_time,columns = ['startTime'])
    final_pre_result['endTime'] = end_time
    final_pre_result.to_csv(name,index = False)	
	
	





	
	
	
	
