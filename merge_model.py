import numpy as np
import pandas as pd
import time
import math
from sklearn.externals import joblib
from tqdm import *


#先只用xgb看看分数如何
def main():
	test_data_40 = pd.read_csv('test_data_final_40.csv')
	
	time = test_data['time']
	test_data.drop('time',axis=1, inplace=True)
	final_label = run_model(test_data_40)
	output(time,final_label)
	






def run_model(test_data):
	#从硬盘读取模型
	
	clff = joblib.load('C:\\Users\\q81022760\\competition\\model\\xgb_7_31.model')
	threshold = 0.75

	
	pre_label = [0]*len(test_data)
    pre_label_pro = clff.predict_proba(test_data)
	
    for i in range(len(pre_label_pro)):
        if pre_label_pro[i][1] > threshold:
            pre_label[i] = 1   

	return pre_label
	
	
def output(time,final_label):
	start_time =[]
	end_time = []
	for i in range(len(final_label)-1):
		if (final_label[i] == 0) & (final_label[i+1] == 1):
			start_time.append(time[i+1])
		if (final_label[i] == 1) & (final_label[i+1] == 0):
			end_time.append(time[i+1]-1)
			
	if (final_label[len(final_label)-2] == 0) & (final_label[len(final_label)-1] == 1):
		start_time.append(time[len(final_label)-1])
		end_time.append(0)			#如果是这种情况，就最后手动加
		
		
	if (final_label[len(final_label)-2] == 1) & (final_label[len(final_label)-1] == 1):
		end_time.append(0)
		
	final_pre_result = pd.DataFrame(start_time,columns = ['startTime'])
	final_pre_result['endTime'] = end_time
	final_pre_result.to_csv('xgb_results.csv',index = False)
	
	
	
	
	