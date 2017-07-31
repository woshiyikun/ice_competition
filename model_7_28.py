from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
import numpy as np
import pandas as pd
import time    

def main():
	train_data = pd.read_csv('train_data_final_40.csv')
	y = train_data['label']
	x = train_data.drop(['label'],axis=1,inplace = False)

	#���ձ����������ݼ�
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(x,y,stratify=y,test_size=0.2, 
                                  random_state=100)
	t1 = time.time()   
	clff = LR_model(X_train,list(y_train))
	#clff = LR_model(X_train,list(y_train),parameter[i])
    #clff = xgb_model(X_train,list(y_train),parameter[i])
    clff = RF_model(X_train,list(y_train),parameter[i])
	
	s,n,TP = cal_score(clff,X_test,list(y_test),0.75)
	print( "score = " + str(s) + ", Negtive =" + str(n) + 
		  "(" + str(TP) + ")" + "\n")


	t2 = time.time()
	print("used time :" + str(t2-t1))


def save_model():
	clff = xgb_model_final(x,list(y))
	#����ģ��
	joblib.dump(clff, 'xgb_7_31.model')
	
#�����ò����Ժ��ģ��
def xgb_model_final(train_data,y_train_label):
    weight_list = []
    for j in range(len(y_train_label)):
        if y_train_label[j] == 1:
            weight_list.append(26)
        if y_train_label[j] == 0:
            weight_list.append(1)     
    clf = XGBClassifier(
    silent=0 ,
    learning_rate= 0.3, # ��ͬѧϰ��
    min_child_weight=1, 
    max_depth = 5, # ����������ȣ�Խ��Խ���׹����
    gamma = 0.2,  
    subsample=1, # �������ѵ������ ѵ��ʵ�����Ӳ�����
    max_delta_step=0,#���������������������ÿ������Ȩ�ع��ơ�
    colsample_bytree=1, # ������ʱ���е��в��� 
    reg_lambda = 1.2,  # ����ģ�͸��Ӷȵ�Ȩ��ֵ��L2���������
    reg_alpha = 0.7, # L1 ���������
    scale_pos_weight = 2.2, 
    n_estimators = 65, #���ĸ���
    seed=0 #�������
    )
    clf.fit(train_data,y_train_label,weight_list)
    return clf
	

def RF_model(train_data,train_label,parameter):
    weight_list=[]
    for j in range(len(train_label)):
        if train_label[j] == 0:
            weight_list.append(1)
        if train_label[j] == 1:
            weight_list.append(10)
    clf = RandomForestClassifier(
                            n_estimators = 100,max_features='log2', 
                            max_depth = parameter ,
                            min_samples_leaf = 10,
                            min_samples_split = 8,
                            criterion='gini',
                            min_impurity_split=1e-7,
                            class_weight= 'balanced'
                            ).fit(train_data,train_label,weight_list)
    return clf
	
def xgb_model(train_data,y_train_label,parameter):
    weight_list = []
    for j in range(len(y_train_label)):
        if y_train_label[j] == 1:
            weight_list.append(parameter)
        if y_train_label[j] == 0:
            weight_list.append(1)
    #class_weight={0:1,1:20}        
    clf = XGBClassifier(
    silent=0 ,#���ó�1��û��������Ϣ��������������Ϊ0.�Ƿ�����������ʱ��ӡ��Ϣ��
    learning_rate= 0.3, # ��ͬѧϰ��
    min_child_weight=1, 
    # �������Ĭ���� 1����ÿ��Ҷ������ h �ĺ������Ƕ��٣�����������������ʱ�� 0-1 �������
    #������ h �� 0.01 ������min_child_weight Ϊ 1 ��ζ��Ҷ�ӽڵ���������Ҫ���� 100 ��������
    #��������ǳ�Ӱ����������Ҷ�ӽڵ��ж��׵��ĺ͵���Сֵ���ò���ֵԽС��Խ���� overfitting��
    max_depth = 7, # ����������ȣ�Խ��Խ���׹����
    gamma = 0.2,  # ����Ҷ�ӽڵ�������һ�������������С��ʧ����,Խ��Խ���أ�һ��0.1��0.2�����ӡ�
    subsample=1, # �������ѵ������ ѵ��ʵ�����Ӳ�����
    max_delta_step=0,#���������������������ÿ������Ȩ�ع��ơ�
    colsample_bytree=1, # ������ʱ���е��в��� 
    reg_lambda=2,  # ����ģ�͸��Ӷȵ�Ȩ��ֵ��L2���������������Խ��ģ��Խ�����׹���ϡ�
    reg_alpha=0.7, # L1 ���������
    scale_pos_weight=1, #���ȡֵ����0�Ļ��������������ƽ�������������ڿ���������ƽ������Ȩ��
    #objective= 'multi:softmax', #���������� ָ��ѧϰ�������Ӧ��ѧϰĿ��
    #num_class=10, # �������������� multisoftmax ����
    n_estimators = 165, #���ĸ���
    seed=0 #�������
    #eval_metric= 'auc'
    )

    clf.fit(train_data,y_train_label,weight_list)
    #������֤���� verbose=False����ӡ����
    return clf
	

#����model����
def LR_model(train_data,y_train_label):
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

def cal_score(cl,test_data,y_test_label,threshold):
    #pre_label = cl.predict(test_data)
    pre_label = [0]*len(y_test_label)
    pre_label_pro = cl.predict_proba(test_data)
	
    for i in range(len(pre_label_pro)):
        if pre_label_pro[i][1] > threshold:
            pre_label[i] = 1   
			
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