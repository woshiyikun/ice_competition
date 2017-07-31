
#重新实现新的想法，把每个group的一半的样本作为一个样本，
#来计算统计特性。
import numpy as np
import pandas as pd
import time
import math
from tqdm import *

def main():
	test_data = pd.read_csv('D:\\iot\\competition\\ice\\test\\08\\08_data.csv')
	
	test_data = add_group(test_data)
	
	#先做一些加减的运算，然后再求平均值之类的。
	test_data = add_columns(test_data)

	#修改列名
	columns_list = list(test_data)	
	columns_list.remove('group')
	columns_list.remove('new_group')
	columns_list.append('new_group')
	new_test_data = test_data.loc[:,columns_list]
	
	new_test_data_stat = cal_statist(new_test_data)	
	temp_time = new_test_data_stat['time']
	new_data_normal = normalization(new_test_data_stat)
	new_data_normal['time'] = temp_time
	
	#还要删除存在缺失值的行，因为有些group中只有一个有效样本，这样标准差就是Nan，所以这个group就要删掉
	new_data_normal_del = new_data_normal.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
	
	new_data_normal_del.to_csv('test_data_7_31_all.csv',index = False)
	#new_data_normal_del里面就是归一化的所有特征的值了。下一步是看特征重要性
	new_columns_name = ['time','e_i_t_max','air_density_20_med', 'power_max', 'pitch2_moto_tmp_mean', 'e_p2_m_t_max',
		 'power_min', 'yaw_position_mean', 'acc_x_mean', 'e_i_t_min', 'e_i_t_mod', 'power_mean', 'generator_speed_min',
		 'yaw_ch1_mean', 'environment_tmp_mean', 'air_density_20_mean', 'pitch1_angle_mean', 'air_density_20_min', 'pitch2_moto_tmp_min',
		 'acc_y_mean', 'e_p2_m_t_min', 'i_p2_m_t_max', 'e_p2_m_t_mean', 'air_density_20_std', 'int_tmp_mean', 'e_p2_m_t_mod',
		 'air_density_20_max', 'power_mod', 'environment_tmp_min', 'pitch1_angle_max', 'environment_tmp_max', 'wind_speed_max',
		 'e_i_t_mean', 'wind_direction_mean_std', 'power_med', 'yaw_position_min', 'pitch1_angle_min', 'int_tmp_min', 'yaw_position_max',
		 'wind_speed_std', 'pitch2_moto_tmp_max']
	fianl_train_all = new_data_normal_del[new_columns_name]
	fianl_train_all.to_csv('test_data_final_40.csv',index = False)
	
	
	

#计算不同列之间的差值，增加特征	
def add_columns(test_data):
	#计算不同列之间的差值	
	#def cal_diff(test_data):
    #环境温度减
    test_data['e_p2_ng_t'] = test_data['environment_tmp'] - test_data['pitch2_ng5_tmp']    
    test_data['e_p2_m_t'] = test_data['environment_tmp'] - test_data['pitch2_moto_tmp']
   
    #机舱温度减
    test_data['i_p1_ng_t'] = test_data['int_tmp'] - test_data['pitch1_ng5_tmp']
    test_data['i_p2_m_t'] = test_data['int_tmp'] - test_data['pitch2_moto_tmp']
	
	#机舱温度和环境温度的差
	test_data['e_i_t'] = test_data['environment_tmp'] - test_data['int_tmp']
	
	
		
	#计算偏航系统的特征值
	#def cal_yaw(test_data):
	# 计算风速速度的次方
	test_data['wind_speed_2'] = [(lambda x : math.pow(x,2))(x) for x in test_data['wind_speed']]
	test_data['wind_direction_mean_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in test_data['wind_direction_mean']]
	test_data['wind_direction_mean_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in test_data['wind_direction_mean_cos']]
	# 计算偏航位置特征
	test_data['yaw_ch1'] = [(lambda x : math.exp(-abs(x)))(x) for x in test_data['yaw_position']]
	test_data['yaw_ch2'] = [(lambda x : math.exp(-abs(x)))(x) for x in test_data['yaw_speed']]
	#计算空气的密度
	test_data['air_density_20'] = test_data['power']/(test_data['wind_speed_2']*test_data['wind_direction_mean_cos_2']*test_data['yaw_ch2'])

	test_data.drop(['yaw_ch2','wind_direction_mean_cos_2','wind_direction_mean_cos'],axis=1, inplace=True)
	
	
	return test_data

	
	
#把一个group分成两份，并且赋新的组号
def add_group(test_data):
	
	test_data = test_data.reset_index(drop=True)
	old_group = test_data['group']
	new_group = []
	group_start = 0
	new_group_index = 0
	for i in tqdm(range(len(test_data)-1)):
		if old_group[i] != old_group[i+1]:
			for j in range(0,int((i-group_start)/2)):
				new_group.append(new_group_index)
			new_group_index += 1
			for j in range(int((i-group_start)/2),i-group_start+1):
				new_group.append(new_group_index)
			new_group_index += 1
			group_start = i+1
		if i == len(test_data)-2 :
			for j in range(0,int((i-group_start)/2)):
				new_group.append(new_group_index)
			new_group_index += 1
			for j in range(int((i-group_start)/2),i-group_start+2):
				new_group.append(new_group_index)
	test_data['new_group'] = new_group
	return test_data



def cal_statist(test_data):
    #mod没有现成的函数可以调用，所以只能自己写函数。求众数
    mod = lambda x : np.argmax(x.value_counts())
    columns_name = test_data.columns
    #第一列是time，先生成这个dataframe，最后也不能删
    new_data = pd.DataFrame(test_data['time'].groupby(test_data['new_group']).head(1))
	new_data = new_data.reset_index( drop=True)
    
    for i in tqdm(range(1,len(test_data.columns)-1)):
        mean_name = columns_name[i]+'_mean'
        med_name = columns_name[i]+'_med'
        max_name = columns_name[i]+'_max'
        min_name = columns_name[i]+'_min'
        std_name = columns_name[i]+'_std'
        mod_name = columns_name[i]+'_mod'

        t_mean = pd.DataFrame(test_data[columns_name[i]].groupby(test_data['new_group']).mean().rename(mean_name)).reset_index( drop=True)
        t_median = pd.DataFrame(test_data[columns_name[i]].groupby(test_data['new_group']).median().rename(med_name)).reset_index( drop=True)
        t_max = pd.DataFrame(test_data[columns_name[i]].groupby(test_data['new_group']).max().rename(max_name)).reset_index( drop=True)
        t_min = pd.DataFrame(test_data[columns_name[i]].groupby(test_data['new_group']).min().rename(min_name)).reset_index( drop=True)
        t_std = pd.DataFrame(test_data[columns_name[i]].groupby(test_data['new_group']).std().rename(std_name)).reset_index( drop=True)
        t_mod = pd.DataFrame(test_data[columns_name[i]].groupby(test_data['new_group']).apply(mod).rename(mod_name)).reset_index( drop=True)

        new_data = pd.concat([new_data,t_mean,t_median,t_max,t_min,t_std,t_mod],axis = 1)

    return new_data	

	
#归一化
def normalization(test_data):
	#这几个公式的值的差别太大了，要做归一化
	#全部都做归一化算了
	#两种不同的归一化，都可以试一下
	#test_data_normal = test_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
	
	test_data_normal = test_data.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
	return test_data_normal
	


	
	
