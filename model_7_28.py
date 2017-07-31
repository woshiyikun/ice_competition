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

	#按照比例划分数据集
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
	#保存模型
	joblib.dump(clff, 'xgb_7_31.model')
	
#调整好参数以后的模型
def xgb_model_final(train_data,y_train_label):
    weight_list = []
    for j in range(len(y_train_label)):
        if y_train_label[j] == 1:
            weight_list.append(26)
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
    silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
    learning_rate= 0.3, # 如同学习率
    min_child_weight=1, 
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    max_depth = 7, # 构建树的深度，越大越容易过拟合
    gamma = 0.2,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    subsample=1, # 随机采样训练样本 训练实例的子采样比
    max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
    colsample_bytree=1, # 生成树时进行的列采样 
    reg_lambda=2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    reg_alpha=0.7, # L1 正则项参数
    scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
    #objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
    #num_class=10, # 类别数，多分类与 multisoftmax 并用
    n_estimators = 165, #树的个数
    seed=0 #随机种子
    #eval_metric= 'auc'
    )

    clf.fit(train_data,y_train_label,weight_list)
    #设置验证集合 verbose=False不打印过程
    return clf
	

#定义model函数
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