

import math
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier


#处理21_data这个文件
train_data = pd.read_csv('D:\\iot\\competition\\ice\\train\\21\\21_data_train.csv')
d_f = pd.read_csv('D:\\iot\\competition\\ice\\train\\21\\21_failureInfo.csv')
d_n = pd.read_csv('D:\\iot\\competition\\ice\\train\\21\\21_normalInfo.csv')



cal_statist()
read_AS()
cal_diff()
cal_yaw()
add_label()
choose_label()


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


#计算所有列的统计值
def cal_statist():

	new_avr_5_before = []   # 之前的几天的平均值
	new_avr_10_before = [] 
	new_avr_15_before = [] 

	new_avr_5_after = []   # 之后的几天的平均值
	new_avr_10_after = [] 
	new_avr_15_after = [] 

	new_max_10_over = []   #前后十天的最大值
	new_min_10_over = []   #前后十天的最小值
	new_std_10_over = []   #前后十天的标准差

	for col_index in range(1,len(train_data.columns)-1):  #对表格的列名进行遍历
		temp_index = train_data.columns[col_index]
		temp_col = train_data[temp_index]
		temp_len = len(temp_col)
		for row_index in range(temp_len):    #对每一列数据进行遍历
			if(row_index < 14 || row_index > temp_len - 15):
				new_avr_5_before.append(temp_col[row_index])   
				new_avr_10_before.append(temp_col[row_index])   
				new_avr_15_before.append(temp_col[row_index])
				new_avr_5_after.append(temp_col[row_index])   
				new_avr_10_after.append(temp_col[row_index])   
				new_avr_15_after.append(temp_col[row_index])   
				new_max_10_over.append(temp_col[row_index])
				new_min_10_over.append(temp_col[row_index])
				new_std_10_over.append(temp_col[row_index])		 
			else:
			
			
				temp_add_before = 0
				temp_add_after = 0
				for i in range(0,5):
					temp_add_before += temp_col[row_index-i]
					temp_add_after += temp_col[row_index+i]
				
				new_avr_5_before.append(temp_add_before/5)  
				new_avr_5_after.append(temp_add_after/5) 
				
				temp_add_before = 0
				temp_add_after = 0
				temp_list = []
				for i in range(0,10):
					temp_add_before += temp_col[row_index-i]
					temp_add_after += temp_col[row_index+i]
					temp_list.append(temp_col[row_index-i])
					temp_list.append(temp_col[row_index+i])
					
					
				new_avr_10_before.append(temp_add_before/10) 
				new_avr_10_after.append(temp_add_after/10) 
				
				new_max_10_over.append(np.max(temp_list))
				new_min_10_over.append(np.min(temp_list))
				new_std_10_over.append(np.std(temp_list))
				
				
				temp_add_before = 0
				temp_add_after = 0
				for i in range(0,15):
					temp_add_before += temp_col[row_index-i]
					temp_add_after += temp_col[row_index+i]
				new_avr_15_before.append(temp_add_before/15) 
				new_avr_15_after.append(temp_add_after/15)
	
	
	
	#将上述各个list转换成dataframe格式
	#avr_5b
	temp_array = np.array(new_avr_5_before)
	avr_5_before_array = np.transpose(np.reshape(temp_array,(26,190494)))
	i = 5
	j='b'
	columns_name1 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p2_a_'+str(i)+str(j),'p3_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j),'p2_s_'+str(i)+str(j),
					 'p3_s_'+str(i)+str(j),'p1_mt_'+str(i)+str(j),
					 'p2_mt_'+str(i)+str(j),'p3_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]
	avr_5b = pd.DataFrame(avr_5_before_array,train_data.index,
					   columns=columns_name1)
					   
	#avr_10b
	temp_array = np.array(new_avr_10_before)
	avr_10_before_array = np.transpose(np.reshape(temp_array,(26,190494)))
	i = 10
	j='b'
	columns_name2 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p2_a_'+str(i)+str(j),'p3_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j),'p2_s_'+str(i)+str(j),
					 'p3_s_'+str(i)+str(j),'p1_mt_'+str(i)+str(j),
					 'p2_mt_'+str(i)+str(j),'p3_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]
	avr_10b = pd.DataFrame(avr_10_before_array,train_data.index,
					   columns=columns_name2)

	#avr_15b
	temp_array = np.array(new_avr_15_before)
	avr_15_before_array = np.transpose(np.reshape(temp_array,(26,190494)))
	i = 15
	j='b'
	columns_name3 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p2_a_'+str(i)+str(j),'p3_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j),'p2_s_'+str(i)+str(j),
					 'p3_s_'+str(i)+str(j),'p1_mt_'+str(i)+str(j),
					 'p2_mt_'+str(i)+str(j),'p3_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]
	avr_15b = pd.DataFrame(avr_15_before_array,train_data.index,
					   columns=columns_name3)

	#avr_5a
	temp_array = np.array(new_avr_5_after)
	avr_5_after_array = np.transpose(np.reshape(temp_array,(26,190494)))
	i = 5
	j='a'
	columns_name4 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p2_a_'+str(i)+str(j),'p3_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j),'p2_s_'+str(i)+str(j),
					 'p3_s_'+str(i)+str(j),'p1_mt_'+str(i)+str(j),
					 'p2_mt_'+str(i)+str(j),'p3_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]
	avr_5a = pd.DataFrame(avr_5_after_array,train_data.index,
					   columns=columns_name4)

	#avr_10a
	temp_array = np.array(new_avr_10_after)
	avr_10_after_array = np.transpose(np.reshape(temp_array,(26,190494)))
	i = 10
	j='a'
	columns_name5 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p2_a_'+str(i)+str(j),'p3_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j),'p2_s_'+str(i)+str(j),
					 'p3_s_'+str(i)+str(j),'p1_mt_'+str(i)+str(j),
					 'p2_mt_'+str(i)+str(j),'p3_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]
	avr_10a = pd.DataFrame(avr_10_after_array,train_data.index,
					   columns=columns_name5)

	#avr_15a
	temp_array = np.array(new_avr_15_after)
	avr_15_after_array = np.transpose(np.reshape(temp_array,(26,190494)))
	i = 15
	j='a'
	columns_name6 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p2_a_'+str(i)+str(j),'p3_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j),'p2_s_'+str(i)+str(j),
					 'p3_s_'+str(i)+str(j),'p1_mt_'+str(i)+str(j),
					 'p2_mt_'+str(i)+str(j),'p3_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]
	avr_15a = pd.DataFrame(avr_15_after_array,train_data.index,
					   columns=columns_name6)

					   
	#max_10
	temp_array = np.array(new_max_10_over)
	max_10_over_array = np.transpose(np.reshape(temp_array,(26,190494)))
	i = 10
	j='max'
	columns_name7 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p2_a_'+str(i)+str(j),'p3_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j),'p2_s_'+str(i)+str(j),
					 'p3_s_'+str(i)+str(j),'p1_mt_'+str(i)+str(j),
					 'p2_mt_'+str(i)+str(j),'p3_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]
	max_10 = pd.DataFrame(max_10_over_array,train_data.index,
					   columns=columns_name7)				   


	#min_10
	temp_array = np.array(new_min_10_over)
	min_10_over_array = np.transpose(np.reshape(temp_array,(26,190494)))
	i = 10
	j='min'
	columns_name8 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p2_a_'+str(i)+str(j),'p3_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j),'p2_s_'+str(i)+str(j),
					 'p3_s_'+str(i)+str(j),'p1_mt_'+str(i)+str(j),
					 'p2_mt_'+str(i)+str(j),'p3_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]
	min_10 = pd.DataFrame(min_10_over_array,train_data.index,
					   columns=columns_name8)
					   
	#std_10
	temp_array = np.array(new_std_10_over)
	std_10_over_array = np.transpose(np.reshape(temp_array,(26,190494)))
	i = 10
	j='std'
	columns_name9 = ['w_avr_'+str(i)+str(j),'g_s_avr_'+str(i)+str(j),
					 'p_avr_'+str(i)+str(j),'w_d_avr_'+str(i)+str(j),
					 'w_d_mean_'+str(i)+str(j),'y_p_'+str(i)+str(j),
					 'y_s_'+str(i)+str(j),'p1_a_'+str(i)+str(j),
					 'p2_a_'+str(i)+str(j),'p3_a_'+str(i)+str(j),
					 'p1_s_'+str(i)+str(j),'p2_s_'+str(i)+str(j),
					 'p3_s_'+str(i)+str(j),'p1_mt_'+str(i)+str(j),
					 'p2_mt_'+str(i)+str(j),'p3_mt_'+str(i)+str(j),
					 'acc_x_'+str(i)+str(j),'acc_y_'+str(i)+str(j),
					 'e_tmp_'+str(i)+str(j),
					 'int_tmp_'+str(i)+str(j),'p1_ngt_'+str(i)+str(j),
					 'p2_ngt_'+str(i)+str(j),'p3_ngt_'+str(i)+str(j),
					 'p1_ngD_'+str(i)+str(j),'p2_ngD_'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]
	std_10 = pd.DataFrame(std_10_over_array,train_data.index,
					   columns=columns_name9)



	#连接各个dataframe
	train_data = pd.concat([train_data,
					  avr_5b,avr_10b,avr_15b,
					  avr_5a,avr_10a,avr_15a,
					  max_10,min_10,std_10],axis = 1)#横轴连接


					  
					  
#读晓聪的文件并改名(加速度信息)
def read_AS():
	
	#AS_1_b
	AS_1_b = pd.read_csv('D:\\iot\\competition\\data\\AS\\21\\data_21_accelerated .csv')
	i = 1
	j='b'
	AS_1_b.columns = ['w_AS'+str(i)+str(j),'g_s_AS'+str(i)+str(j),
					 'p_AS'+str(i)+str(j),'w_d_AS'+str(i)+str(j),
					 'w_d_mean_AS'+str(i)+str(j),'y_p_AS'+str(i)+str(j),
					 'y_s_AS'+str(i)+str(j),'p1_a_AS'+str(i)+str(j),
					 'p2_a_AS'+str(i)+str(j),'p3_a_AS'+str(i)+str(j),
					 'p1_s_AS'+str(i)+str(j),'p2_s_AS'+str(i)+str(j),
					 'p3_s_AS'+str(i)+str(j),'p1_mt_AS'+str(i)+str(j),
					 'p2_mt_AS'+str(i)+str(j),'p3_mt_AS'+str(i)+str(j),
					 'acc_x_AS'+str(i)+str(j),'acc_y_AS'+str(i)+str(j),
					 'e_tmp_AS'+str(i)+str(j),
					 'int_tmp_AS'+str(i)+str(j),'p1_ngt_AS'+str(i)+str(j),
					 'p2_ngt_AS'+str(i)+str(j),'p3_ngt_AS'+str(i)+str(j),
					 'p1_ngD_AS'+str(i)+str(j),'p2_ngD_AS'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]

	#AS_5_b
	AS_5_b = pd.read_csv('D:\\iot\\competition\\data\\AS\\21\\data_21_accelerated_5.csv')
	i = 5
	j='b'
	AS_5_b.columns = ['w_AS'+str(i)+str(j),'g_s_AS'+str(i)+str(j),
					 'p_AS'+str(i)+str(j),'w_d_AS'+str(i)+str(j),
					 'w_d_mean_AS'+str(i)+str(j),'y_p_AS'+str(i)+str(j),
					 'y_s_AS'+str(i)+str(j),'p1_a_AS'+str(i)+str(j),
					 'p2_a_AS'+str(i)+str(j),'p3_a_AS'+str(i)+str(j),
					 'p1_s_AS'+str(i)+str(j),'p2_s_AS'+str(i)+str(j),
					 'p3_s_AS'+str(i)+str(j),'p1_mt_AS'+str(i)+str(j),
					 'p2_mt_AS'+str(i)+str(j),'p3_mt_AS'+str(i)+str(j),
					 'acc_x_AS'+str(i)+str(j),'acc_y_AS'+str(i)+str(j),
					 'e_tmp_AS'+str(i)+str(j),
					 'int_tmp_AS'+str(i)+str(j),'p1_ngt_AS'+str(i)+str(j),
					 'p2_ngt_AS'+str(i)+str(j),'p3_ngt_AS'+str(i)+str(j),
					 'p1_ngD_AS'+str(i)+str(j),'p2_ngD_AS'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]



	#AS_10_b
	AS_10_b = pd.read_csv('D:\\iot\\competition\\data\\AS\\21\\data_21_accelerated_10.csv')
	i = 10
	j='b'
	AS_10_b.columns = ['w_AS'+str(i)+str(j),'g_s_AS'+str(i)+str(j),
					 'p_AS'+str(i)+str(j),'w_d_AS'+str(i)+str(j),
					 'w_d_mean_AS'+str(i)+str(j),'y_p_AS'+str(i)+str(j),
					 'y_s_AS'+str(i)+str(j),'p1_a_AS'+str(i)+str(j),
					 'p2_a_AS'+str(i)+str(j),'p3_a_AS'+str(i)+str(j),
					 'p1_s_AS'+str(i)+str(j),'p2_s_AS'+str(i)+str(j),
					 'p3_s_AS'+str(i)+str(j),'p1_mt_AS'+str(i)+str(j),
					 'p2_mt_AS'+str(i)+str(j),'p3_mt_AS'+str(i)+str(j),
					 'acc_x_AS'+str(i)+str(j),'acc_y_AS'+str(i)+str(j),
					 'e_tmp_AS'+str(i)+str(j),
					 'int_tmp_AS'+str(i)+str(j),'p1_ngt_AS'+str(i)+str(j),
					 'p2_ngt_AS'+str(i)+str(j),'p3_ngt_AS'+str(i)+str(j),
					 'p1_ngD_AS'+str(i)+str(j),'p2_ngD_AS'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]



	#AS_1_a
	AS_1_a = pd.read_csv('D:\\iot\\competition\\data\\AS\\21_after\\data_21_after_accelerated_1.csv')
	i = 1
	j='a'
	AS_1_a.columns = ['w_AS'+str(i)+str(j),'g_s_AS'+str(i)+str(j),
					 'p_AS'+str(i)+str(j),'w_d_AS'+str(i)+str(j),
					 'w_d_mean_AS'+str(i)+str(j),'y_p_AS'+str(i)+str(j),
					 'y_s_AS'+str(i)+str(j),'p1_a_AS'+str(i)+str(j),
					 'p2_a_AS'+str(i)+str(j),'p3_a_AS'+str(i)+str(j),
					 'p1_s_AS'+str(i)+str(j),'p2_s_AS'+str(i)+str(j),
					 'p3_s_AS'+str(i)+str(j),'p1_mt_AS'+str(i)+str(j),
					 'p2_mt_AS'+str(i)+str(j),'p3_mt_AS'+str(i)+str(j),
					 'acc_x_AS'+str(i)+str(j),'acc_y_AS'+str(i)+str(j),
					 'e_tmp_AS'+str(i)+str(j),
					 'int_tmp_AS'+str(i)+str(j),'p1_ngt_AS'+str(i)+str(j),
					 'p2_ngt_AS'+str(i)+str(j),'p3_ngt_AS'+str(i)+str(j),
					 'p1_ngD_AS'+str(i)+str(j),'p2_ngD_AS'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]

	#AS_5_a
	AS_5_a = pd.read_csv('D:\\iot\\competition\\data\\AS\\21_after\\data_21_after_accelerated_5.csv')
	i = 5
	j='a'
	AS_5_a.columns = ['w_AS'+str(i)+str(j),'g_s_AS'+str(i)+str(j),
					 'p_AS'+str(i)+str(j),'w_d_AS'+str(i)+str(j),
					 'w_d_mean_AS'+str(i)+str(j),'y_p_AS'+str(i)+str(j),
					 'y_s_AS'+str(i)+str(j),'p1_a_AS'+str(i)+str(j),
					 'p2_a_AS'+str(i)+str(j),'p3_a_AS'+str(i)+str(j),
					 'p1_s_AS'+str(i)+str(j),'p2_s_AS'+str(i)+str(j),
					 'p3_s_AS'+str(i)+str(j),'p1_mt_AS'+str(i)+str(j),
					 'p2_mt_AS'+str(i)+str(j),'p3_mt_AS'+str(i)+str(j),
					 'acc_x_AS'+str(i)+str(j),'acc_y_AS'+str(i)+str(j),
					 'e_tmp_AS'+str(i)+str(j),
					 'int_tmp_AS'+str(i)+str(j),'p1_ngt_AS'+str(i)+str(j),
					 'p2_ngt_AS'+str(i)+str(j),'p3_ngt_AS'+str(i)+str(j),
					 'p1_ngD_AS'+str(i)+str(j),'p2_ngD_AS'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]



	#AS_10_a
	AS_10_a = pd.read_csv('D:\\iot\\competition\\data\\AS\\21_after\\data_21_after_accelerated_10.csv')
	i = 10
	j='a'
	AS_10_a.columns = ['w_AS'+str(i)+str(j),'g_s_AS'+str(i)+str(j),
					 'p_AS'+str(i)+str(j),'w_d_AS'+str(i)+str(j),
					 'w_d_mean_AS'+str(i)+str(j),'y_p_AS'+str(i)+str(j),
					 'y_s_AS'+str(i)+str(j),'p1_a_AS'+str(i)+str(j),
					 'p2_a_AS'+str(i)+str(j),'p3_a_AS'+str(i)+str(j),
					 'p1_s_AS'+str(i)+str(j),'p2_s_AS'+str(i)+str(j),
					 'p3_s_AS'+str(i)+str(j),'p1_mt_AS'+str(i)+str(j),
					 'p2_mt_AS'+str(i)+str(j),'p3_mt_AS'+str(i)+str(j),
					 'acc_x_AS'+str(i)+str(j),'acc_y_AS'+str(i)+str(j),
					 'e_tmp_AS'+str(i)+str(j),
					 'int_tmp_AS'+str(i)+str(j),'p1_ngt_AS'+str(i)+str(j),
					 'p2_ngt_AS'+str(i)+str(j),'p3_ngt_AS'+str(i)+str(j),
					 'p1_ngD_AS'+str(i)+str(j),'p2_ngD_AS'+str(i)+str(j),
					 'p3_ngD_AS'+str(i)+str(j)]


	train_data = pd.concat([train_data,
					  AS_1_b,AS_5_b,AS_10_b,
					  AS_1_a,AS_5_a,AS_10_a],axis = 1)#横轴连接块
	len(list(train_data))


	
#计算不同列之间的差值	
def cal_diff():
    #环境温度减
    train_data['e_p1_ng_t'] = train_data['environment_tmp'] - train_data['pitch1_ng5_tmp']
    train_data['e_p2_ng_t'] = train_data['environment_tmp'] - train_data['pitch2_ng5_tmp']
    train_data['e_p3_ng_t'] = train_data['environment_tmp'] - train_data['pitch3_ng5_tmp']
    train_data['e_p1_m_t'] = train_data['environment_tmp'] - train_data['pitch1_moto_tmp']
    train_data['e_p2_m_t'] = train_data['environment_tmp'] - train_data['pitch2_moto_tmp']
    train_data['e_p3_m_t'] = train_data['environment_tmp'] - train_data['pitch3_moto_tmp']
    
    #机舱温度减
    train_data['i_p1_ng_t'] = train_data['int_tmp'] - train_data['pitch1_ng5_tmp']
    train_data['i_p2_ng_t'] = train_data['int_tmp'] - train_data['pitch2_ng5_tmp']
    train_data['i_p3_ng_t'] = train_data['int_tmp'] - train_data['pitch3_ng5_tmp']
    train_data['i_p1_m_t'] = train_data['int_tmp'] - train_data['pitch1_moto_tmp']
    train_data['i_p2_m_t'] = train_data['int_tmp'] - train_data['pitch2_moto_tmp']
    train_data['i_p3_m_t'] = train_data['int_tmp'] - train_data['pitch3_moto_tmp']
	
	
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
	train_data['air_density_1'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_cos'])
	train_data['air_density_2'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos'])
	train_data['air_density_3'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_cos_2'])
	train_data['air_density_4'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos_2'])

	train_data['air_density_5'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_cos'])
	train_data['air_density_6'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_mean_cos'])
	train_data['air_density_7'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_cos_2'])
	train_data['air_density_8'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_mean_cos_2'])

	train_data['air_density_9'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_cos']*train_data['yaw_ch1'])
	train_data['air_density_10'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos']*train_data['yaw_ch1'])
	train_data['air_density_11'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_cos_2']*train_data['yaw_ch1'])
	train_data['air_density_12'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos_2']*train_data['yaw_ch1'])

	train_data['air_density_13'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_cos']*train_data['yaw_ch1'])
	train_data['air_density_14'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_mean_cos']*train_data['yaw_ch1'])
	train_data['air_density_15'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_cos_2']*train_data['yaw_ch1'])
	train_data['air_density_16'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_mean_cos_2']*train_data['yaw_ch1'])

	train_data['air_density_17'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_cos']*train_data['yaw_ch2'])
	train_data['air_density_18'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos']*train_data['yaw_ch2'])
	train_data['air_density_19'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_cos_2']*train_data['yaw_ch2'])
	train_data['air_density_20'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos_2']*train_data['yaw_ch2'])

	train_data['air_density_21'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_cos']*train_data['yaw_ch2'])
	train_data['air_density_22'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_mean_cos']*train_data['yaw_ch2'])
	train_data['air_density_23'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_cos_2']*train_data['yaw_ch2'])
	train_data['air_density_24'] = train_data['power']/(train_data['wind_speed_3']*train_data['wind_direction_mean_cos_2']*train_data['yaw_ch2'])


#剔除无效数据，保留label为1和0的数据	
def choose_label():
	f_data = train_data[train_data['label'].isin([1])].append(train_data[train_data['label'].isin([0])])
	
	
#特征数量461，太多了，先手动减少一些没有意义的特征
def drop_fea():
	f_data.drop(['group','acc_x','acc_y',
				'pitch1_ng5_DC','pitch2_ng5_DC','pitch3_ng5_DC',
				'p2_a_5b','p2_a_10b','p2_a_15b','p3_a_5b','p3_a_10b','p3_a_15b',
				'p2_a_5a','p2_a_10a','p2_a_15a','p3_a_5a','p3_a_10a','p3_a_15a',
				'p2_ngD_5b','p2_ngD_10b','p2_ngD_15b','p3_ngD_5b','p3_ngD_10b','p3_ngD_15b',
				'p2_ngD_5a','p2_ngD_10a','p2_ngD_15a','p3_ngD_5a','p3_ngD_10a','p3_ngD_15a',
				'p1_a_AS1b','p1_a_AS5b','p1_a_AS10b','p1_a_AS1a','p1_a_AS5a','p1_a_AS10a',
				'p2_a_AS1b','p2_a_AS5b','p2_a_AS10b','p2_a_AS1a','p2_a_AS5a','p2_a_AS10a',
				'p3_a_AS1b','p3_a_AS5b','p3_a_AS10b','p3_a_AS1a','p3_a_AS5a','p3_a_AS10a'],axis=1, inplace=True)
	
	
#探索特征重要性，减少特征数量
def fea_importance():
	y_label = f_data['label']
	f_data.drop('label',axis=1, inplace=True)
	f_data.drop('time',axis=1, inplace=True)

	#RF
	t1 = time.time()
	clf = RandomForestClassifier(
									n_estimators=200,
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
    
	

#找到重要性排名前130的特征，
def find_important_feature():

	new_columns_name = []
	for i in range(0,130):
		new_columns_name.append(name_list[impor_dict_list[i][0]])
		
	f_data_del = f_data[new_columns_name]
	y_label_d = pd.DataFrame(y_label,columns =['label'])
	f_data_del = pd.concat([f_data_del,y_label_d],axis = 1)

	#f_data_del就是最后130维，20万数据，带label的

	#过滤一些特征，对名称相似的列计算相关系数，若相关性较高，则只保留一列
	new_columns_name.sort()  #排序


	f_data_del.drop(['air_density_1','air_density_2','air_density_3','air_density_4','air_density_17',
					'air_density_18','air_density_19','e_p3_m_t','e_p2_m_t','e_tmp_10a', 'e_tmp_10b','e_tmp_10max',
					'e_tmp_15a', 'e_tmp_15b','e_tmp_5a','e_tmp_5b','g_s_avr_10a', 'g_s_avr_10b','g_s_avr_10min', 'g_s_avr_15a', 'g_s_avr_5a',
					'g_s_avr_5b','i_p1_m_t','i_p2_m_t','int_tmp', 'int_tmp_10a','int_tmp_10b','int_tmp_10max', 'int_tmp_15a',
					'int_tmp_15b','int_tmp_5a','int_tmp_5b','p1_a_10a', 'p1_a_10b', 'p1_a_15a', 'p1_a_5a', 'p1_a_5b', 'p1_mt_10a',
					'p1_mt_10b', 'p1_mt_10max', 'p1_mt_10min', 'p1_mt_15b', 'p1_mt_5a', 'p1_mt_5b', 'p2_mt_10a', 'p2_mt_10b', 'p2_mt_10max', 'p2_mt_10min',
					'p2_mt_15a', 'p2_mt_15b', 'p2_mt_5a', 'p2_mt_5b','p2_ngt_10b','p2_ngt_5b','p3_mt_10a','p3_mt_10b', 'p3_mt_10max', 'p3_mt_10min',
					'p3_mt_15a', 'p3_mt_15b', 'p3_mt_5a', 'p3_mt_5b', 'p_avr_10a', 'p_avr_10b','p_avr_15a', 'p_avr_15b', 'p_avr_5a', 'p_avr_5b','pitch1_angle',
					'pitch2_moto_tmp', 'pitch3_angle', 'pitch3_moto_tmp','w_avr_10a', 'w_avr_10b','w_avr_15a', 'w_avr_15b', 'w_avr_5a', 'w_avr_5b',
					'y_p_10a', 'y_p_10b', 'y_p_10max', 'y_p_10min', 'y_p_15a', 'y_p_5a', 'y_p_5b'],axis=1, inplace=True)




	#计算相关性，晓聪的这几个公式相关性非常高，所以保留那个排名最高的，只保留20
	for i in range(1,8):   
		print(f_data_del['air_density_1'].corr(f_data_del[new_columns_name[i]]))


	#e_p1_m_t，e_p2_m_t，e_p3_m_t相关性也很高,只保留1
	print(f_data_del['e_p1_m_t'].corr(f_data_del['e_p3_m_t']))

	#这个相关性0.95，不知要要不要保留。
	print(f_data_del['e_p1_ng_t'].corr(f_data_del['e_p2_ng_t']))
	
	#时间间隔取的太短了，好多数据都没有差别，这些所有的数据都几乎一样。
	print(f_data_del['e_tmp_10a'].corr(f_data_del['e_tmp_15a']))
	
	#最后只剩下46个特征了，可以统一算一下相关性
	#继续看相关性，还要继续删除特征。。。
	f_data_del.corr().to_csv('corr.csv')
	f_data_del.drop(['p2_a_10max','p3_a_10max','p1_a_10min','p3_a_10min'],axis=1, inplace=True)

	
#最后剩下42个特征，把特征名写到文件中。
import csv
with open('D:\\iot\\competition\\columns_name.csv','w',newline='') as data:
    writer = csv.writer(data)
    writer.writerow(list(f_data_del.columns))
	
#写train_data这个数据表到csv文件
f_data_del.to_csv('f_data_del.csv',index = None)

	
#训练模型，用GBDT，可以设置正负样本的权重	
def train_model():
	
	
	
	

	
	
	
	
	
	
