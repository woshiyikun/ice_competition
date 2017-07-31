import numpy as np
import pandas as pd
import math
import time
from sklearn.externals import joblib
from tqdm import *


test_data = pd.read_csv('D:\\iot\\competition\\ice\\test\\08\\08_data.csv')


def main():
	test_data.drop(['pitch2_angle','pitch3_angle','pitch1_moto_tmp','pitch3_moto_tmp','pitch2_speed','pitch3_speed'],axis=1, inplace=True)
	
	test_data = find_group(test_data)
	test_data = cal_statist(test_data)
	test_data = cal_diff(test_data)
	test_data = cal_yaw(test_data)
		
	new_columns_name = ['e_p2_m_t', 'e_i_t', 'p_avr_10min', 'air_density_20', 'p1_a_10med',
       'yaw_ch1', 'p1_a_10max', 'y_p_10avr', 'w_avr_10avr', 'p2_mt_10min',
       'e_p2_m_t_avr', 'y_p_10max', 'int_tmp_10min', 'p2_mt_10avr',
       'i_p2_m_t_avr', 'y_p_10min', 'p_avr_10max', 'y_p_10med', 'w_avr_10max',
       'e_tmp_10avr', 'i_p2_m_t', 'p1_a_10avr', 'w_avr_10min', 'yaw_position',
       'int_tmp_10avr', 'p2_mt_10max', 'p2_mt_10med', 'e_tmp_10min',
       'p1_a_10min', 'pitch1_angle', 'p_avr_10avr', 'pitch2_moto_tmp',
       'e_tmp_10max', 'w_avr_10med', 'environment_tmp', 'int_tmp_10med',
       'e_p2_ng_t', 'g_s_avr_10max', 'wind_speed_2', 'e_tmp_10med','group_label',
	   'group']
	
	
	test_data = test_data[new_columns_name]
	
	#然后选group_label=0的列出来计算
	test_data_del = test_data[test_data['group_label'].isin([0])]
	y_test_group_del = test_data_del['group']
	test_data_del.drop(['group','group_label'],axis=1, inplace=True)
	
	#test_data_del就是预测集的特征
	pre_label = run_model(test_data_del)
	
	#把test_data_del写成文件
	test_data_del.to_csv('test_data_del_08.csv',index = False)
	
	#根据group信息，把同一个gourp都变成同样的label(添加每个group的前十个样本)
	final_label = cal_final_label(pre_label,y_test_group_del)
	
	#输出成最后的格式
	output(final_label)
	
	
def output(final_label):
	start_time =[]
	end_time = []
	for i in range(len(final_label)-1):
		if (final_label[i] == 0) & (final_label[i+1] == 1):
			start_time.append(i+2)
		if (final_label[i] == 1) & (final_label[i+1] == 0):
			end_time.append(i+2)
	final_pre_result = pd.DataFrame(start_time,columns = ['startTime'])
	final_pre_result['endTime'] = end_time
	final_pre_result.to_csv('test_08_results.csv',index = False)
		
	
def cal_final_label():
	final_label = []
	group_time = 0
	y_test_group_del_list = list(y_test_group_del)
	for i in range(len(y_test_group_del)-1):
		if y_test_group_del_list[i] == y_test_group_del_list[i+1]:
			final_label.append(pre_label[i])
		else:
			group_time += 1
			for j in range(0,11):
				final_label.append(pre_label[i-1])
		if i == len(y_test_group_del)-2:
			for j in range(0,11):
				final_label.append(pre_label[i-1])
	group_time
	
	#把完整的label信息写到文件里。
	final_label_d = pd.DataFrame(final_label,columns = ['final_label'])
	final_label_d.to_csv('final_label.csv',index = False)

	return final_label

		
	
#按照每一个group计算统计值
#每个group标注前十个样本为1

def find_group():
	group_list = test_data['group']
	group_label = [0]*len(test_data)
	
	for i in range(0,len(test_data)):
		if i==0:
			for j in range(0,10):
				group_label[i+j] = 1
		elif (group_list[i] != group_list[i-1]):
			for j in range(0,10):
				group_label[i+j] = 1

	group_label_d = pd.DataFrame(group_label,columns=['group_label'])	
	test_data = pd.concat([test_data,group_label_d],axis=1)
	return test_data
#记得添加到test_data里面


#计算所有列的统计值
def cal_statist(test_data):

	
	new_avr_10 = []   # 之前的几个时间点的平均值
	new_max_10 = []   #前10的最大值
	new_min_10 = []   #前10的最小值
	new_std_10 = []   #前10的标准差
	new_med_10 = []   #前10的中位数
	
	group_label = test_data['group_label']

	#对20个特征做处理
	for col_index in tqdm(range(1,len(test_data.columns)-2)):  #对表格的列名进行遍历,删除第一列和最后两列（因为没有label）
		temp_index = test_data.columns[col_index]
		temp_col = test_data[temp_index]
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
	max_10_over_array = np.transpose(np.reshape(temp_array,(20,202328)))
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
	max_10 = pd.DataFrame(max_10_over_array,test_data.index,
					   columns=columns_name7)				   


	#min_10
	temp_array = np.array(new_min_10)
	min_10_over_array = np.transpose(np.reshape(temp_array,(20,202328)))
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
	min_10 = pd.DataFrame(min_10_over_array,test_data.index,
					   columns=columns_name8)

	#std_10
	temp_array = np.array(new_std_10)
	std_10_over_array = np.transpose(np.reshape(temp_array,(20,202328)))
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
	std_10 = pd.DataFrame(std_10_over_array,test_data.index,
					   columns=columns_name9)

	#med_10
	temp_array = np.array(new_med_10)
	med_10_over_array = np.transpose(np.reshape(temp_array,(20,202328)))
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
	med_10 = pd.DataFrame(med_10_over_array,test_data.index,
					   columns=columns_name1)

	#std_10
	temp_array = np.array(new_avr_10)
	avr_10_over_array = np.transpose(np.reshape(temp_array,(20,202328)))
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
	avr_10 = pd.DataFrame(avr_10_over_array,test_data.index,
					   columns=columns_name2)

	#连接各个dataframe
	test_data = pd.concat([test_data,med_10,
					  avr_10,max_10,min_10,std_10],axis = 1)#横轴连接
	return test_data


#计算不同列之间的差值	
def cal_diff(test_data):
    #环境温度减
    test_data['e_p1_ng_t'] = test_data['environment_tmp'] - test_data['pitch1_ng5_tmp']
    test_data['e_p2_ng_t'] = test_data['environment_tmp'] - test_data['pitch2_ng5_tmp']
    test_data['e_p3_ng_t'] = test_data['environment_tmp'] - test_data['pitch3_ng5_tmp']
    test_data['e_p2_m_t'] = test_data['environment_tmp'] - test_data['pitch2_moto_tmp']

    
    #机舱温度减
    test_data['i_p1_ng_t'] = test_data['int_tmp'] - test_data['pitch1_ng5_tmp']
    test_data['i_p2_ng_t'] = test_data['int_tmp'] - test_data['pitch2_ng5_tmp']
    test_data['i_p3_ng_t'] = test_data['int_tmp'] - test_data['pitch3_ng5_tmp']
   test_data['i_p2_m_t'] = test_data['int_tmp'] - test_data['pitch2_moto_tmp']
	
	#机舱温度和环境温度的差
	test_data['e_i_t'] = test_data['environment_tmp'] - test_data['int_tmp']
	
	#前十个时间点的平均值之差
	test_data['i_p1_ng_t_avr'] = test_data['int_tmp_10avr'] - test_data['p1_ngt_10avr']
	test_data['i_p2_m_t_avr'] = test_data['int_tmp_10avr'] - test_data['p2_mt_10avr']
	test_data['e_p2_m_t_avr'] = test_data['e_tmp_10avr'] - test_data['p2_mt_10avr']
	test_data['e_p1_ng_t_avr'] = test_data['e_tmp_10avr'] - test_data['p1_ngt_10avr']
	
	return test_data

		
	
	
#计算偏航系统的特征值
def cal_yaw(test_data):
	# 计算风速速度的次方
	test_data['wind_speed_2'] = [(lambda x : math.pow(x,2))(x) for x in test_data['wind_speed']]
	test_data['wind_speed_3'] = [(lambda x : math.pow(x,3))(x) for x in test_data['wind_speed']]
	#计算对风角的cos
	test_data['wind_direction_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in test_data['wind_direction']]
	test_data['wind_direction_mean_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in test_data['wind_direction_mean']]
	#计算对风角的平方
	test_data['wind_direction_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in test_data['wind_direction_cos']]
	test_data['wind_direction_mean_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in test_data['wind_direction_mean_cos']]
	# 计算偏航位置特征
	test_data['yaw_ch1'] = [(lambda x : math.exp(-abs(x)))(x) for x in test_data['yaw_position']]
	test_data['yaw_ch2'] = [(lambda x : math.exp(-abs(x)))(x) for x in test_data['yaw_speed']]
	#计算空气的密度

	test_data['air_density_20'] = test_data['power']/(test_data['wind_speed_2']*test_data['wind_direction_mean_cos_2']*test_data['yaw_ch2'])
	return test_data



	
	
#跑模型
#记得删除每个group的前10个数据
def run_model(test_data):
	#从硬盘读取模型
	
	LR = joblib.load('C:\\Users\\q81022760\\competition\\model\\LR.model')
	threshold = 0.7
	y_test_group = test_data['group']
	test_data.drop('group',axis=1, inplace=True)
	test_data.drop('group_label',axis=1, inplace=True)

	pre_label = [0]*len(test_data)
	pre_label_pro = LR.predict_proba(test_data)
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
			if temp/times >= threshold:
				for j in range(group_start,i):
					pre_label[j] = 1
			group_start = i+1
			times = 0
            temp = 0
		if i == len(y_test_group)-2:
			if temp/times >= threshold:
				for j in range(group_start,i):
					pre_label[j] = 1
	return pre_label
	
	
	
	
	
	
	

