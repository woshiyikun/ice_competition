
#����ʵ���µ��뷨����ÿ��group��һ���������Ϊһ��������
#������ͳ�����ԡ�
import numpy as np
import pandas as pd
import time
import math
from tqdm import *

def main():
	data_15 = pd.read_csv('D:\\iot\\competition\\ice\\train\\15\\15_data_train.csv')
	data_15.drop(['pitch2_angle','pitch3_angle','pitch1_moto_tmp','pitch3_moto_tmp','pitch2_speed','pitch3_speed'],axis=1, inplace=True)
	d_f = pd.read_csv('D:\\iot\\competition\\ice\\train\\15\\15_failureInfo.csv')
	d_n = pd.read_csv('D:\\iot\\competition\\ice\\train\\15\\15_normalInfo.csv')
		

	data_15 = add_label(data_15,d_n,d_f)
	
	data_21 = pd.read_csv('D:\\iot\\competition\\ice\\train\\21\\21_data_train.csv')
	data_21.drop(['pitch2_angle','pitch3_angle','pitch1_moto_tmp','pitch3_moto_tmp','pitch2_speed','pitch3_speed'],axis=1, inplace=True)
	d_f = pd.read_csv('D:\\iot\\competition\\ice\\train\\21\\21_failureInfo.csv')
	d_n = pd.read_csv('D:\\iot\\competition\\ice\\train\\21\\21_normalInfo.csv')
		

	data_21 = add_label(data_21,d_n,d_f)
	train_data = pd.concat([data_15,data_21],axis = 0)
	train_data = train_data[train_data['label'].isin([1])].append(train_data[train_data['label'].isin([0])])
	
	
	#��������
	train_data = train_data.reset_index(drop=True)
	
	train_data = add_group(train_data)
	
	#����һЩ�Ӽ������㣬Ȼ������ƽ��ֵ֮��ġ�
	train_data = add_columns(train_data)

	#�޸�����
	columns_list = list(train_data)
	columns_list.remove('label')
	columns_list.remove('group')
	columns_list.remove('new_group')
	columns_list.append('new_group')
	columns_list.append('label')
	new_train_data = train_data.loc[:,columns_list]
	
	new_train_data_stat = cal_statist(new_train_data)
	new_data_normal_label = new_train_data_stat['label']
	new_data_normal = normalization(new_train_data_stat)
	new_data_normal['label'] = new_data_normal_label
	
	#��Ҫɾ������ȱʧֵ���У���Ϊ��Щgroup��ֻ��һ����Ч������������׼�����Nan���������group��Ҫɾ��
	new_data_normal_del = new_data_normal.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
	
	new_data_normal_del.to_csv('train_data_7_28_all.csv',index = False)
	#new_data_normal_del������ǹ�һ��������������ֵ�ˡ���һ���ǿ�������Ҫ��
	new_columns_name = fea_importance(new_data_normal_del,40)
	fianl_train_all = new_data_normal_del[new_columns_name]
	fianl_train_all.to_csv('train_data_final_40.csv',index = False)
	
	
#����������Ҫ�ԣ�������ѡ��ǰK������
#GBDT����Ҫ��
def fea_importance(new_data_normal_del,k):
	from sklearn.ensemble import GradientBoostingClassifier
	y_label = new_data_normal_del['label']
	xtrain = new_data_normal_del.drop('label',axis=1, inplace=False)
	#GBDT
	clf = GradientBoostingClassifier(
									loss='deviance', n_estimators=200,
									 learning_rate=0.1,
									 max_depth=7, 
									 random_state=0,
									 min_samples_split=200,
									 min_samples_leaf=250,
									 subsample=1.0,
									 #max_features=variables.max_feature_gdbt
									).fit(xtrain, y_label)

	#feature importance
	importance_list = clf.feature_importances_

	name_list = new_data_normal.columns.values
	impor_dict = {}
	for i in range(len(importance_list)):
		impor_dict[i] = importance_list[i]

	impor_dict_list = sorted(impor_dict.items(), key=lambda d: d[1], reverse=True)

	for i in range(len(impor_dict_list)):
		print ("fea" + str(impor_dict_list[i][0]) + " " + str(name_list[impor_dict_list[i][0]]) + " : " + str(
			impor_dict_list[i][1]))
	
	#��ѡ��������Ҫ������ǰ40������	
	new_columns_name = []
	for i in range(0,k):
		new_columns_name.append(name_list[impor_dict_list[i][0]])
	new_columns_name.append('label')
	return new_columns_name
	
	

#���㲻ͬ��֮��Ĳ�ֵ����������	
def add_columns(train_data):
	#���㲻ͬ��֮��Ĳ�ֵ	
	#def cal_diff(train_data):
    #�����¶ȼ�
    train_data['e_p2_ng_t'] = train_data['environment_tmp'] - train_data['pitch2_ng5_tmp']    
    train_data['e_p2_m_t'] = train_data['environment_tmp'] - train_data['pitch2_moto_tmp']
   
    #�����¶ȼ�
    train_data['i_p1_ng_t'] = train_data['int_tmp'] - train_data['pitch1_ng5_tmp']
    train_data['i_p2_m_t'] = train_data['int_tmp'] - train_data['pitch2_moto_tmp']
	
	#�����¶Ⱥͻ����¶ȵĲ�
	train_data['e_i_t'] = train_data['environment_tmp'] - train_data['int_tmp']
	
	
		
	#����ƫ��ϵͳ������ֵ
	#def cal_yaw(train_data):
	# ��������ٶȵĴη�
	train_data['wind_speed_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_speed']]
	train_data['wind_direction_mean_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in train_data['wind_direction_mean']]
	train_data['wind_direction_mean_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_direction_mean_cos']]
	# ����ƫ��λ������
	train_data['yaw_ch1'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_position']]
	train_data['yaw_ch2'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_speed']]
	#����������ܶ�
	train_data['air_density_20'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos_2']*train_data['yaw_ch2'])

	train_data.drop(['yaw_ch2','wind_direction_mean_cos_2','wind_direction_mean_cos'],axis=1, inplace=True)
	
	
	return train_data

	
	
#��һ��group�ֳ����ݣ����Ҹ��µ����
def add_group(train_data):
	
	train_data = train_data.reset_index(drop=True)
	old_group = train_data['group']
	new_group = []
	group_start = 0
	new_group_index = 0
	for i in tqdm(range(len(train_data)-1)):
		if old_group[i] != old_group[i+1]:
			for j in range(0,int((i-group_start)/2)):
				new_group.append(new_group_index)
			new_group_index += 1
			for j in range(int((i-group_start)/2),i-group_start+1):
				new_group.append(new_group_index)
			new_group_index += 1
			group_start = i+1
		if i == len(train_data)-2 :
			for j in range(0,int((i-group_start)/2)):
				new_group.append(new_group_index)
			new_group_index += 1
			for j in range(int((i-group_start)/2),i-group_start+2):
				new_group.append(new_group_index)
	train_data['new_group'] = new_group
	return train_data



#����ÿ���»��ֵ�group����ͳ��ֵ
def cal_statist(train_data):
	#modû���ֳɵĺ������Ե��ã�����ֻ���Լ�д������������
	mod = lambda x : np.argmax(x.value_counts())
	columns_name = train_data.columns
	new_data = pd.DataFrame(train_data['label'].groupby(train_data['new_group']).mean())
	new_data = new_data.reset_index( drop=True)
	for i in tqdm(range(1,len(train_data.columns)-2)):
		mean_name = columns_name[i]+'_mean'
		med_name = columns_name[i]+'_med'
		max_name = columns_name[i]+'_max'
		min_name = columns_name[i]+'_min'
		std_name = columns_name[i]+'_std'
		mod_name = columns_name[i]+'_mod'
		
		t_mean = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).mean().rename(mean_name)).reset_index( drop=True)
		t_median = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).median().rename(med_name)).reset_index( drop=True)
		t_max = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).max().rename(max_name)).reset_index( drop=True)
		t_min = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).min().rename(min_name)).reset_index( drop=True)
		t_std = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).std().rename(std_name)).reset_index( drop=True)
		t_mod = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).apply(mod).rename(mod_name)).reset_index( drop=True)
		
		new_data = pd.concat([new_data,t_mean,t_median,t_max,t_min,t_std,t_mod],axis = 1)
		
	return new_data	
	
	
#��һ��
def normalization(train_data):
	#�⼸����ʽ��ֵ�Ĳ��̫���ˣ�Ҫ����һ��
	#ȫ��������һ������
	#���ֲ�ͬ�Ĺ�һ������������һ��
	#train_data_normal = train_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
	
	train_data_normal = train_data.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
	return train_data_normal
	

#��ԭʼ�������label
def add_label(train_data,d_n,d_f):
	train_data_new_time = [(lambda x : time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))(x) for x in train_data['time'] ]

	#ת��normal��failure��ʱ���ʽ
	d_n_start_time = [(lambda x : time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))(x) for x in d_n['startTime'] ]
	d_n_end_time= [(lambda x : time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))(x) for x in d_n['endTime'] ]
	d_f_start_time = [(lambda x : time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))(x) for x in d_f['startTime'] ]
	d_f_end_time = [(lambda x : time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))(x) for x in d_f['endTime'] ]

	label_list_2=[]
	t1 = 0
	t2 = 0
	l1 = 0
	l2 = 0
	for i in range(0,len(train_data)):
		if ((train_data_new_time[i] >= d_n_start_time[t1]) &
		(train_data_new_time[i] <= d_n_end_time[t1])):
			label_list_2.append(0)
			l1 = 1
		elif ((train_data_new_time[i] >= d_f_start_time[t2]) &
		(train_data_new_time[i] <= d_f_end_time[t2])):
			label_list_2.append(1)
			l2 = 2
		else:
			if l1 == 1: 
				t1 = t1 +1 
				if t1 == len(d_n_end_time):
					t1 = len(d_n_end_time)-1
			if l2 == 2:
				t2 = t2 + 1
				if t2 == len(d_f_end_time):
					t1 = len(d_f_end_time)-1
			l1 = 0 
			l2 = 0
			label_list_2.append(2)

	label_s=pd.Series(label_list_2)
	train_data['label'] = label_s
	return train_data
	
	
	
