
def main():
	import numpy as np
	import pandas as pd
	import math
	import time
	from tqdm import *


	train_data = pd.read_csv('D:\\iot\\competition\\ice\\train\\15\\15_data_train.csv')

	#�м�����������ضȷǳ��ߣ����Կ���ֱ��ɾ��6������
	d_f = pd.read_csv('D:\\iot\\competition\\ice\\train\\15\\15_failureInfo.csv')
	d_n = pd.read_csv('D:\\iot\\competition\\ice\\train\\15\\15_normalInfo.csv')
	
	
	
	train_data.drop(['pitch2_angle','pitch3_angle','pitch1_moto_tmp','pitch3_moto_tmp','pitch2_speed','pitch3_speed'],axis=1, inplace=True)
	
	
	train_data = find_group(train_data)
	train_data = add_label(train_data,d_n,d_f)
	train_data = cal_statist(train_data)
	train_data = cal_diff(train_data)
	train_data = cal_yaw(train_data)
	
	#�������ɭ�ֵ�������Ҫ��������ѡ����Ҫ������ǰ40������
	new_columns_name = ['e_p2_m_t', 'e_i_t', 'p_avr_10min', 'air_density_20', 'p1_a_10med',
       'yaw_ch1', 'p1_a_10max', 'y_p_10avr', 'w_avr_10avr', 'p2_mt_10min',
       'e_p2_m_t_avr', 'y_p_10max', 'int_tmp_10min', 'p2_mt_10avr',
       'i_p2_m_t_avr', 'y_p_10min', 'p_avr_10max', 'y_p_10med', 'w_avr_10max',
       'e_tmp_10avr', 'i_p2_m_t', 'p1_a_10avr', 'w_avr_10min', 'yaw_position',
       'int_tmp_10avr', 'p2_mt_10max', 'p2_mt_10med', 'e_tmp_10min',
       'p1_a_10min', 'pitch1_angle', 'p_avr_10avr', 'pitch2_moto_tmp',
       'e_tmp_10max', 'w_avr_10med', 'environment_tmp', 'int_tmp_10med',
       'e_p2_ng_t', 'g_s_avr_10max', 'wind_speed_2', 'e_tmp_10med','group',
	   'group_label','label']
	#�Ȱ���Ӧ������������Ȼ����ɾ��ĳЩ��
	final_data = train_data[new_columns_name]

	#ɾ��ÿ��group��ǰʮ������
	final_data_del = final_data[final_data['group_label'].isin([0])]

	#ɾ����Ч������label = 2��
	final_data_del = final_data_del[final_data_del['label'].isin([1])].append(final_data_del[final_data_del['label'].isin([0])])


	final_data_del.to_csv('final_15.csv',index = None)	
	
	
	



#����ÿһ��group����ͳ��ֵ,����ɾ��ÿ��group��ǰʮ������

def find_group(train_data):
    group_list = train_data['group']
    group_label = [0]*len(train_data)

    for i in range(0,len(train_data)-1):
        if i==0:
            for j in range(0,10):
                group_label[i+j] = 1
        elif (group_list[i] != group_list[i-1]):
            for j in range(0,10):
                group_label[i+j] = 1

    group_label_d = pd.DataFrame(group_label,columns=['group_label'])	
    train_data = pd.concat([train_data,group_label_d],axis=1)
    return train_data


#����ͳ��ֵ
def cal_statist(train_data):

	
	new_avr_10 = []   # ֮ǰ�ļ���ʱ����ƽ��ֵ
	new_max_10 = []   #ǰ10�����ֵ
	new_min_10 = []   #ǰ10����Сֵ
	new_std_10 = []   #ǰ10�ı�׼��
	new_med_10 = []   #ǰ10����λ��
	
	group_label = train_data['group_label']

	#��20������������
	for col_index in tqdm(range(1,len(train_data.columns)-3)):  #�Ա����������б���,ɾ����һ�к��������
		temp_index = train_data.columns[col_index]
		temp_col = train_data[temp_index]
		temp_len = len(temp_col)
		for row_index in range(temp_len):    #��ÿһ�����ݽ��б���
			if(group_label[row_index] == 1):
				new_avr_10.append(temp_col[row_index])   
				new_max_10.append(temp_col[row_index])
				new_min_10.append(temp_col[row_index])
				new_std_10.append(0)
				new_med_10.append(temp_col[row_index]) 
			else:

				temp_10 = 0
				temp_list = []

				for i in range(0,10):
				   temp_list.append(temp_col[row_index-i])

				new_max_10.append(np.max(temp_list))
				new_min_10.append(np.min(temp_list))
				new_std_10.append(np.std(temp_list))
				new_avr_10.append(np.mean(temp_list))
				new_med_10.append(np.median(temp_list))

	#����������listת����dataframe��ʽ
	#max_10
	temp_array = np.array(new_max_10)
	max_10_over_array = np.transpose(np.reshape(temp_array,(20,int(len(temp_array)/20))))
	i = 10
	j='max'
	columns_name7 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j), 'p2_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_'+str(i)+str(j)]
	max_10 = pd.DataFrame(max_10_over_array,train_data.index,
					   columns=columns_name7)				   


	#min_10
	temp_array = np.array(new_min_10)
	min_10_over_array = np.transpose(np.reshape(temp_array,(20,int(len(temp_array)/20))))
	i = 10
	j='min'
	columns_name8 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j), 'p2_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_'+str(i)+str(j)]
	min_10 = pd.DataFrame(min_10_over_array,train_data.index,
					   columns=columns_name8)

	#std_10
	temp_array = np.array(new_std_10)
	std_10_over_array = np.transpose(np.reshape(temp_array,(20,int(len(temp_array)/20))))
	i = 10
	j='std'
	columns_name9 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j), 'p2_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_'+str(i)+str(j)]
	std_10 = pd.DataFrame(std_10_over_array,train_data.index,
					   columns=columns_name9)

	#med_10
	temp_array = np.array(new_med_10)
	med_10_over_array = np.transpose(np.reshape(temp_array,(20,int(len(temp_array)/20))))
	i = 10
	j='med'
	columns_name1 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j), 'p2_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_'+str(i)+str(j)]
	med_10 = pd.DataFrame(med_10_over_array,train_data.index,
					   columns=columns_name1)

	#std_10
	temp_array = np.array(new_avr_10)
	avr_10_over_array = np.transpose(np.reshape(temp_array,(20,int(len(temp_array)/20))))
	i = 10
	j='avr'
	columns_name2 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j), 'p2_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_'+str(i)+str(j)]
	avr_10 = pd.DataFrame(avr_10_over_array,train_data.index,
					   columns=columns_name2)

	#���Ӹ���dataframe
	train_data = pd.concat([train_data,med_10,
					  avr_10,max_10,min_10,std_10],axis = 1)#��������
	return train_data


					  
					  
					  
#���㲻ͬ��֮��Ĳ�ֵ	
def cal_diff(train_data):
    #�����¶ȼ�
    train_data['e_p1_ng_t'] = train_data['environment_tmp'] - train_data['pitch1_ng5_tmp']
    train_data['e_p2_ng_t'] = train_data['environment_tmp'] - train_data['pitch2_ng5_tmp']
    train_data['e_p3_ng_t'] = train_data['environment_tmp'] - train_data['pitch3_ng5_tmp']
    train_data['e_p2_m_t'] = train_data['environment_tmp'] - train_data['pitch2_moto_tmp']

    
    #�����¶ȼ�
    train_data['i_p1_ng_t'] = train_data['int_tmp'] - train_data['pitch1_ng5_tmp']
    train_data['i_p2_ng_t'] = train_data['int_tmp'] - train_data['pitch2_ng5_tmp']
    train_data['i_p3_ng_t'] = train_data['int_tmp'] - train_data['pitch3_ng5_tmp']
    train_data['i_p2_m_t'] = train_data['int_tmp'] - train_data['pitch2_moto_tmp']
	
	#�����¶Ⱥͻ����¶ȵĲ�
	train_data['e_i_t'] = train_data['environment_tmp'] - train_data['int_tmp']
	
	#ǰʮ��ʱ����ƽ��ֵ֮��
	train_data['i_p1_ng_t_avr'] = train_data['int_tmp_10avr'] - train_data['p1_ngt_10avr']
	train_data['i_p2_m_t_avr'] = train_data['int_tmp_10avr'] - train_data['p2_mt_10avr']
	train_data['e_p2_m_t_avr'] = train_data['e_tmp_10avr'] - train_data['p2_mt_10avr']
	train_data['e_p1_ng_t_avr'] = train_data['e_tmp_10avr'] - train_data['p1_ngt_10avr']
	
	return train_data

		
	
	
#����ƫ��ϵͳ������ֵ
def cal_yaw(train_data):
	# ��������ٶȵĴη�
	train_data['wind_speed_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_speed']]
	train_data['wind_speed_3'] = [(lambda x : math.pow(x,3))(x) for x in train_data['wind_speed']]
	#����Է�ǵ�cos
	train_data['wind_direction_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in train_data['wind_direction']]
	train_data['wind_direction_mean_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in train_data['wind_direction_mean']]
	#����Է�ǵ�ƽ��
	train_data['wind_direction_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_direction_cos']]
	train_data['wind_direction_mean_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_direction_mean_cos']]
	# ����ƫ��λ������
	train_data['yaw_ch1'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_position']]
	train_data['yaw_ch2'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_speed']]
	#����������ܶ�

	train_data['air_density_20'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos_2']*train_data['yaw_ch2'])
	return train_data


#���label
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
	

	
#ɾ��time��group����
#̽��������Ҫ�ԣ�������������
def fea_importance():

	y_label = f_data['label']
	f_data.drop('label',axis=1, inplace=True)
	f_data.drop('time',axis=1, inplace=True)
	f_data.drop('group',axis=1, inplace=True)

	#RF
	t1 = time.time()
	clf = RandomForestClassifier(
									n_estimators=300,
									criterion = 'gini',
									max_features = None,
									max_depth = 1000
									).fit(f_data, y_label)
	t2 = time.time()
	print(t2-t1)
	#feature importance
	importance_list = clf.feature_importances_

	name_list = f_data.columns.values
	impor_dict = {}
	for i in range(len(importance_list)):
		impor_dict[i] = importance_list[i]

	impor_dict_list = sorted(impor_dict.items(), key=lambda d: d[1], reverse=True)

	for i in range(len(impor_dict_list)):
		print ( str(name_list[impor_dict_list[i][0]]) + " : " + str(impor_dict_list[i][1]))
    
	
	
#��ѡ��������Ҫ������ǰ40������	
new_columns_name = []
for i in range(0,40):
    new_columns_name.append(name_list[impor_dict_list[i][0]])
new_columns_name.append('label')
new_columns_name.append('group')
new_columns_name.append('group_label')


	
	
	
	
