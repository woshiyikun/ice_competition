
import numpy as np
import pandas as pd
import math
import time

train_data = pd.read_csv('D:\\iot\\competition\\ice\\train\\21\\21_data_train.csv')

#有几个特征的相关度非常高，所以可以直接删除6个特征
train_data.drop(['pitch2_angle','pitch3_angle','pitch1_moto_tmp','pitch3_moto_tmp','pitch2_speed','pitch3_speed'],axis=1, inplace=True)



#每个group标注前十个样本为1，训练和预测的时候直接删除这些样本

def find_group():
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
#记得添加到train_data里面


#计算所有列的统计值
def cal_statist():

	from tqdm import *
	new_avr_10 = []   # 之前的几个时间点的平均值
	new_max_10 = []   #前10的最大值
	new_min_10 = []   #前10的最小值
	new_std_10 = []   #前10的标准差
	new_med_10 = []   #前10的中位数

	#对20个特征做处理
	for col_index in tqdm(range(1,len(train_data.columns)-2)):  #对表格的列名进行遍历,删除第一列和最后一列
		temp_index = train_data.columns[col_index]
		temp_col = train_data[temp_index]
		temp_len = len(temp_col)
		for row_index in range(temp_len):    #对每一列数据进行遍历
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

	#将上述各个list转换成dataframe格式
	#max_10
	temp_array = np.array(new_max_10)
	max_10_over_array = np.transpose(np.reshape(temp_array,(20,190494)))
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
	min_10_over_array = np.transpose(np.reshape(temp_array,(20,190494)))
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
	std_10_over_array = np.transpose(np.reshape(temp_array,(20,190494)))
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
	med_10_over_array = np.transpose(np.reshape(temp_array,(20,190494)))
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
	avr_10_over_array = np.transpose(np.reshape(temp_array,(20,190494)))
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

	#连接各个dataframe
	train_data = pd.concat([train_data,med_10,
					  avr_10,max_10,min_10,std_10],axis = 1)#横轴连接


					  
					  
					  
#计算不同列之间的差值	
def cal_diff():
    #环境温度减
    train_data['e_p1_ng_t'] = train_data['environment_tmp'] - train_data['pitch1_ng5_tmp']
    train_data['e_p2_ng_t'] = train_data['environment_tmp'] - train_data['pitch2_ng5_tmp']
    train_data['e_p3_ng_t'] = train_data['environment_tmp'] - train_data['pitch3_ng5_tmp']
    train_data['e_p2_m_t'] = train_data['environment_tmp'] - train_data['pitch2_moto_tmp']

    
    #机舱温度减
    train_data['i_p1_ng_t'] = train_data['int_tmp'] - train_data['pitch1_ng5_tmp']
    train_data['i_p2_ng_t'] = train_data['int_tmp'] - train_data['pitch2_ng5_tmp']
    train_data['i_p3_ng_t'] = train_data['int_tmp'] - train_data['pitch3_ng5_tmp']
    train_data['i_p2_m_t'] = train_data['int_tmp'] - train_data['pitch2_moto_tmp']
	
	#机舱温度和环境温度的差
	train_data['e_i_t'] = train_data['environment_tmp'] - train_data['int_tmp']
	
	
	
	#30天平均值减20天，20天减10天，10天减5天
	train_data['w_30_20'] = train_data['w_avr_30b'] - train_data['w_avr_20b']
	train_data['w_20_10'] = train_data['w_avr_20b'] - train_data['w_avr_10b']
	train_data['w_10_5'] = train_data['w_avr_10b'] - train_data['w_avr_5b']

	train_data['g_30_20'] = train_data['g_s_avr_30b'] - train_data['g_s_avr__20b']
	train_data['g_20_10'] = train_data['g_s_avr__20b'] - train_data['g_s_avr__10b']
	train_data['g_10_5'] = train_data['g_s_avr__10b'] - train_data['g_s_avr__5b']
	
	train_data['p_30_20'] = train_data['p_avr_30b'] - train_data['p_avr_20b']
	train_data['p_20_10'] = train_data['p_avr_20b'] - train_data['p_avr_10b']
	train_data['p_10_5'] = train_data['p_avr_10b'] - train_data['p_avr_5b']
	
	train_data['w_d_30_20'] = train_data['w_d_avr_30b'] - train_data['w_d_avr_20b']
	train_data['w_d_20_10'] = train_data['w_d_avr_20b'] - train_data['w_d_avr_10b']
	train_data['w_d_10_5'] = train_data['w_d_avr_10b'] - train_data['w_d_avr_5b']
	
	train_data['y_s_30_20'] = train_data['y_s_30b'] - train_data['y_s_20b']
	train_data['y_s_20_10'] = train_data['y_s_20b'] - train_data['y_s_10b']
	train_data['y_s_10_5'] = train_data['y_s_10b'] - train_data['y_s_5b']
	
	train_data['p1_a_30_20'] = train_data['p1_a_30b'] - train_data['p1_a_20b']
	train_data['p1_a_20_10'] = train_data['p1_a_20b'] - train_data['p1_a_10b']
	train_data['p1_a_10_5'] = train_data['p1_a_10b'] - train_data['p1_a_5b']
	
	train_data['p1_s_30_20'] = train_data['p1_s_30b'] - train_data['p1_s_20b']
	train_data['p1_s_20_10'] = train_data['p1_s_20b'] - train_data['p1_s_10b']
	train_data['p1_s_10_5'] = train_data['p1_s_10b'] - train_data['p1_s_5b']
	
	train_data['p2_mt_30_20'] = train_data['p2_mt_30b'] - train_data['p2_mt_20b']
	train_data['p2_mt_20_10'] = train_data['p2_mt_20b'] - train_data['p2_mt_10b']
	train_data['p2_mt_10_5'] = train_data['p2_mt_10b'] - train_data['p2_mt_5b']
	
	train_data['acc_x_30_20'] = train_data['acc_x_30b'] - train_data['acc_x_20b']
	train_data['acc_x_20_10'] = train_data['acc_x_20b'] - train_data['acc_x_10b']
	train_data['acc_x_10_5'] = train_data['acc_x_10b'] - train_data['acc_x_5b']
	
	train_data['acc_y_30_20'] = train_data['acc_y_30b'] - train_data['acc_y_20b']
	train_data['acc_y_20_10'] = train_data['acc_y_20b'] - train_data['acc_y_10b']
	train_data['acc_y_10_5'] = train_data['acc_y_10b'] - train_data['acc_y_5b']
	
	train_data['e_tmp_30_20'] = train_data['e_tmp_30b'] - train_data['e_tmp_20b']
	train_data['e_tmp_20_10'] = train_data['e_tmp_20b'] - train_data['e_tmp_10b']
	train_data['e_tmp_10_5'] = train_data['e_tmp_10b'] - train_data['e_tmp_5b']
	
	train_data['int_tmp_30_20'] = train_data['int_tmp_30b'] - train_data['int_tmp_20b']
	train_data['int_tmp_20_10'] = train_data['int_tmp_20b'] - train_data['int_tmp_10b']
	train_data['int_tmp_10_5'] = train_data['int_tmp_10b'] - train_data['int_tmp_5b']
	
	train_data['p1_ngt_30_20'] = train_data['p1_ngt_30b'] - train_data['p1_ngt_20b']
	train_data['p1_ngt_20_10'] = train_data['p1_ngt_20b'] - train_data['p1_ngt_10b']
	train_data['p1_ngt_10_5'] = train_data['p1_ngt_10b'] - train_data['p1_ngt_5b']

	train_data['p2_ngt_30_20'] = train_data['p2_ngt_30b'] - train_data['p2_ngt_20b']
	train_data['p2_ngt_20_10'] = train_data['p2_ngt_20b'] - train_data['p2_ngt_10b']
	train_data['p2_ngt_10_5'] = train_data['p2_ngt_10b'] - train_data['p2_ngt_5b']
	
	train_data['p3_ngt_30_20'] = train_data['p3_ngt_30b'] - train_data['p3_ngt_20b']
	train_data['p3_ngt_20_10'] = train_data['p3_ngt_20b'] - train_data['p3_ngt_10b']
	train_data['p3_ngt_10_5'] = train_data['p3_ngt_10b'] - train_data['p3_ngt_5b']
	
	train_data['p1_ngD_30_20'] = train_data['p1_ngD_30b'] - train_data['p1_ngD_20b']
	train_data['p1_ngD_20_10'] = train_data['p1_ngD_20b'] - train_data['p1_ngD_10b']
	train_data['p1_ngD_10_5'] = train_data['p1_ngD_10b'] - train_data['p1_ngD_5b']
	
	train_data['p2_ngD_30_20'] = train_data['p2_ngD_30b'] - train_data['p2_ngD_20b']
	train_data['p2_ngD_20_10'] = train_data['p2_ngD_20b'] - train_data['p2_ngD_10b']
	train_data['p2_ngD_10_5'] = train_data['p2_ngD_10b'] - train_data['p2_ngD_5b']
	
	train_data['p3_ngD_30_20'] = train_data['p3_ngD_30b'] - train_data['p3_ngD_20b']
	train_data['p3_ngD_20_10'] = train_data['p3_ngD_20b'] - train_data['p3_ngD_10b']
	train_data['p3_ngD_10_5'] = train_data['p3_ngD_10b'] - train_data['p3_ngD_5b']
	
	
	
#计算偏航系统的特征值
def cal_yaw():
	# 计算风速速度的次方
	train_data['wind_speed_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_speed']]
	train_data['wind_speed_3'] = [(lambda x : math.pow(x,3))(x) for x in train_data['wind_speed']]
	#计算对风角的cos
	train_data['wind_direction_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in train_data['wind_direction']]
	train_data['wind_direction_mean_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in train_data['wind_direction_mean']]
	#计算对风角的平方
	train_data['wind_direction_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_direction_cos']]
	train_data['wind_direction_mean_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_direction_mean_cos']]
	# 计算偏航位置特征
	train_data['yaw_ch1'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_position']]
	train_data['yaw_ch2'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_speed']]
	#计算空气的密度

	train_data['air_density_20'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos_2']*train_data['yaw_ch2'])


#转换原始表格的时间格式
def add_label():
	train_data_new_time = [(lambda x : time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))(x) for x in train_data['time'] ]

	#转换normal和failure的时间格式
	d_n_start_time = [(lambda x : time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))(x) for x in d_n['startTime'] ]
	d_n_end_time= [(lambda x : time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))(x) for x in d_n['endTime'] ]
	d_f_start_time = [(lambda x : time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))(x) for x in d_f['startTime'] ]
	d_f_end_time = [(lambda x : time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))(x) for x in d_f['endTime'] ]

	#添加label，把dataframe转换成list，大大提升for循环的速度
	label_list=[]
	t1 = 0
	t2 = 0
	for i in range(0,len(train_data)):
		if ((train_data_new_time[i] >= d_n_start_time[t1]) &
		(train_data_new_time[i] < d_n_end_time[t1])):
			label_list.append(0)
		elif train_data_new_time[i] == d_n_end_time[t1]:  
			t1 = t1+1
			print("t1="+ str(t1))
			if t1 == len(d_n_end_time):
				t1 = len(d_n_end_time)-1
			label_list.append(0)
		elif ((train_data_new_time[i] >= d_f_start_time[t2]) &
		(train_data_new_time[i] < d_f_end_time[t2])):
			label_list.append(1)
		elif train_data_new_time[i] == d_f_end_time[t2]:    
			t2 = t2+1
			print("t2="+ str(t2))
			if t1 == len(d_f_end_time):
				t1 = len(d_f_end_time)-1
			label_list.append(1)
		else:
			label_list.append(2)

	#添加到data_15里
	label_s=pd.Series(label_list)
	train_data['label'] = label_s
	
	
#剔除无效数据，保留label为1和0的数据	
def choose_label():
	f_data = train_data[train_data['label'].isin([1])].append(train_data[train_data['label'].isin([0])])
	
	
	
#删除time和group两列
#探索特征重要性，减少特征数量
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
    
	
	
#挑选出特征重要性排名前40的特征	
new_columns_name = []
for i in range(0,40):
    new_columns_name.append(name_list[impor_dict_list[i][0]])
new_columns_name.append('label')
new_columns_name.append('group')
new_columns_name.append('group_label')


#先把相应的列挑出来，然后再删除某些行
final_data = train_data[new_columns_name]

#先选group_label
final_data_del = final_data[final_data['group_label'].isin([0])]

#再选label
final_data_del = final_data_del[final_data_del['label'].isin([1])].append(final_data_del[final_data_del['label'].isin([0])])


final_data_del.to_csv('final_21.csv',index = None)	
	
	
	
	
