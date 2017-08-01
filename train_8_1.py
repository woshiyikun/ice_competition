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

	#重新索引
	train_data = train_data.reset_index(drop=True)

	train_data = add_group(train_data)

	#先做一些加减的运算，然后再求平均值之类的。
	train_data = add_columns(train_data)

	#修改列名
	columns_list = list(train_data)
	columns_list.remove('label')
	columns_list.remove('group')
	columns_list.remove('new_group')
	columns_list.append('group')
	columns_list.append('new_group')
	columns_list.append('label')
	new_train_data = train_data.loc[:,columns_list]

	new_train_data_stat = cal_statist(new_train_data)
	temp_label = new_train_data_stat['label']
	temp_group = new_train_data_stat['group']

	new_data_normal = normalization(new_train_data_stat)
	new_data_normal['label'] = temp_label
	new_data_normal['group'] = temp_group

	#还要删除存在缺失值的行，因为有些group中只有一个有效样本，这样标准差就是Nan，所以这个group就要删掉\n",
	new_data_normal_del = new_data_normal.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

	#new_data_normal_del.to_csv('train_data_7_28_all.csv',index = False)\n",
	#new_data_normal_del里面就是归一化的所有特征的值了。下一步是看特征重要性\n",
	new_columns_name = ['label','group','e_i_t_max','air_density_20_med', 'power_max', 'pitch2_moto_tmp_mean', 'e_p2_m_t_max',
			 'power_min', 'yaw_position_mean', 'acc_x_mean', 'e_i_t_min', 'e_i_t_mod', 'power_mean', 'generator_speed_min',
			 'yaw_ch1_mean', 'environment_tmp_mean', 'air_density_20_mean', 'pitch1_angle_mean', 'air_density_20_min', 'pitch2_moto_tmp_min',
			 'acc_y_mean', 'e_p2_m_t_min', 'i_p2_m_t_max', 'e_p2_m_t_mean', 'air_density_20_std', 'int_tmp_mean', 'e_p2_m_t_mod',
			 'air_density_20_max', 'power_mod', 'environment_tmp_min', 'pitch1_angle_max', 'environment_tmp_max', 'wind_speed_max',
			 'e_i_t_mean', 'wind_direction_mean_std', 'power_med', 'yaw_position_min', 'pitch1_angle_min', 'int_tmp_min', 'yaw_position_max',
			 'wind_speed_std', 'pitch2_moto_tmp_max']
	fianl_train_all = new_data_normal_del[new_columns_name]
	fianl_train_all.to_csv('new_train_40.csv',index = False)
	
	
	
#把一个group分成三份，并且赋新的组号
def add_group(train_data):

    train_data = train_data.reset_index(drop=True)
    old_group = train_data['group']
    new_group = []
    group_start = 0
    new_group_index = 0
    for i in tqdm(range(len(train_data)-1)):
        if old_group[i] != old_group[i+1]:
            if (i-group_start)>5:
                for j in range(0,int((i-group_start)/3)):
                    new_group.append(new_group_index)
                new_group_index += 1
                for j in range(int((i-group_start)/3),int((i-group_start)/3*2)):
                    new_group.append(new_group_index)
                new_group_index += 1
                for j in range(int((i-group_start)/3*2),i-group_start+1):
                    new_group.append(new_group_index)
                new_group_index += 1
            else:
                for j in range(0,(i-group_start)+1):
                    new_group.append(new_group_index)
                new_group_index += 1
            group_start = i+1
        if i == len(train_data)-2 :
            for j in range(0,int((i-group_start)/3)):
                new_group.append(new_group_index)
            new_group_index += 1
            for j in range(int((i-group_start)/3),int((i-group_start)/3*2)):
                new_group.append(new_group_index)
            new_group_index += 1
            for j in range(int((i-group_start)/3*2),i-group_start+2):
                new_group.append(new_group_index)
            new_group_index += 1
    train_data['new_group'] = new_group
    return train_data

#构造特征
def add_columns(train_data):
    #计算不同列之间的差值	
    #def cal_diff(train_data):
    #环境温度减
    train_data['e_p2_ng_t'] = train_data['environment_tmp'] - train_data['pitch2_ng5_tmp']    
    train_data['e_p2_m_t'] = train_data['environment_tmp'] - train_data['pitch2_moto_tmp']

    #机舱温度减
    train_data['i_p1_ng_t'] = train_data['int_tmp'] - train_data['pitch1_ng5_tmp']
    train_data['i_p2_m_t'] = train_data['int_tmp'] - train_data['pitch2_moto_tmp']

    #机舱温度和环境温度的差
    train_data['e_i_t'] = train_data['environment_tmp'] - train_data['int_tmp']



    #计算偏航系统的特征值
    #def cal_yaw(train_data):
    # 计算风速速度的次方
    train_data['wind_speed_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_speed']]
    train_data['wind_direction_mean_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in train_data['wind_direction_mean']]
    train_data['wind_direction_mean_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_direction_mean_cos']]
    # 计算偏航位置特征
    train_data['yaw_ch1'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_position']]
    train_data['yaw_ch2'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_speed']]
    #计算空气的密度
    train_data['air_density_20'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos_2']*train_data['yaw_ch2'])

    train_data.drop(['yaw_ch2','wind_direction_mean_cos_2','wind_direction_mean_cos'],axis=1, inplace=True)
    return train_data	
	
	
	
#归一化	
def normalization(train_data):

    #train_data_normal = train_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    train_data_normal = train_data.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    return train_data_normal	
	
	
	
	
#给原始数据添加label
def add_label(train_data,d_n,d_f):
    train_data_new_time = [(lambda x : time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))(x) for x in train_data['time'] ]

    #转换normal和failure的时间格式
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
	
	
	


#按照每个新划分的group计算统计值
def cal_statist(train_data):
    #mod没有现成的函数可以调用，所以只能自己写函数。求众数
    mod = lambda x : np.argmax(x.value_counts())
    columns_name = train_data.columns
    new_data = pd.DataFrame(train_data['label'].groupby(train_data['new_group']).head(1))
    new_data = new_data.reset_index( drop=True)
    old_group = pd.DataFrame(train_data['group'].groupby(train_data['new_group']).head(1))
    old_group = old_group.reset_index( drop=True)
    new_data = pd.concat([new_data,old_group],axis = 1)
    
    for i in tqdm(range(1,len(train_data.columns)-3)):
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



	
	
	
	
	
	
