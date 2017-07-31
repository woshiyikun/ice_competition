{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import time\n",
    "import math\n",
    "from tqdm import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|███████████████████████████████| 553710/553710 [00:21<00:00, 25473.55it/s]\n",
      "100%|██████████████████████████████████████████| 29/29 [06:33<00:00, 13.49s/it]\n"
     ]
    }
   ],
   "source": [
    "data_15 = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\15\\\\15_data.csv')\n",
    "data_15.drop(['pitch2_angle','pitch3_angle','pitch1_moto_tmp','pitch3_moto_tmp','pitch2_speed','pitch3_speed'],axis=1, inplace=True)\n",
    "d_f = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\15\\\\15_failureInfo.csv')\n",
    "d_n = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\15\\\\15_normalInfo.csv')\n",
    "\n",
    "\n",
    "data_15 = add_label(data_15,d_n,d_f)\n",
    "\n",
    "data_21 = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\21\\\\21_data.csv')\n",
    "data_21.drop(['pitch2_angle','pitch3_angle','pitch1_moto_tmp','pitch3_moto_tmp','pitch2_speed','pitch3_speed'],axis=1, inplace=True)\n",
    "d_f = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\21\\\\21_failureInfo.csv')\n",
    "d_n = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\21\\\\21_normalInfo.csv')\n",
    "\n",
    "\n",
    "data_21 = add_label(data_21,d_n,d_f)\n",
    "train_data = pd.concat([data_15,data_21],axis = 0)\n",
    "train_data = train_data[train_data['label'].isin([1])].append(train_data[train_data['label'].isin([0])])\n",
    "\n",
    "\n",
    "#重新索引\n",
    "train_data = train_data.reset_index(drop=True)\n",
    "\n",
    "train_data = add_group(train_data)\n",
    "\n",
    "#先做一些加减的运算，然后再求平均值之类的。\n",
    "train_data = add_columns(train_data)\n",
    "\n",
    "#修改列名\n",
    "columns_list = list(train_data)\n",
    "columns_list.remove('label')\n",
    "columns_list.remove('group')\n",
    "columns_list.remove('new_group')\n",
    "columns_list.append('group')\n",
    "columns_list.append('new_group')\n",
    "columns_list.append('label')\n",
    "new_train_data = train_data.loc[:,columns_list]\n",
    "\n",
    "\n",
    "new_train_data_stat = cal_statist(new_train_data)\n",
    "new_data_normal_label = new_train_data_stat['label']\n",
    "#new_data_normal_group = new_train_data_stat['group']\n",
    "new_data_normal = normalization(new_train_data_stat)\n",
    "new_data_normal['label'] = new_data_normal_label\n",
    "\n",
    "#还要删除存在缺失值的行，因为有些group中只有一个有效样本，这样标准差就是Nan，所以这个group就要删掉\n",
    "new_data_normal_del = new_data_normal.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)\n",
    "\n",
    "#new_data_normal_del.to_csv('train_data_7_28_all.csv',index = False)\n",
    "#new_data_normal_del里面就是归一化的所有特征的值了。下一步是看特征重要性\n",
    "new_columns_name = ['label','e_i_t_max','air_density_20_med', 'power_max', 'pitch2_moto_tmp_mean', 'e_p2_m_t_max',\n",
    "         'power_min', 'yaw_position_mean', 'acc_x_mean', 'e_i_t_min', 'e_i_t_mod', 'power_mean', 'generator_speed_min',\n",
    "         'yaw_ch1_mean', 'environment_tmp_mean', 'air_density_20_mean', 'pitch1_angle_mean', 'air_density_20_min', 'pitch2_moto_tmp_min',\n",
    "         'acc_y_mean', 'e_p2_m_t_min', 'i_p2_m_t_max', 'e_p2_m_t_mean', 'air_density_20_std', 'int_tmp_mean', 'e_p2_m_t_mod',\n",
    "         'air_density_20_max', 'power_mod', 'environment_tmp_min', 'pitch1_angle_max', 'environment_tmp_max', 'wind_speed_max',\n",
    "         'e_i_t_mean', 'wind_direction_mean_std', 'power_med', 'yaw_position_min', 'pitch1_angle_min', 'int_tmp_min', 'yaw_position_max',\n",
    "         'wind_speed_std', 'pitch2_moto_tmp_max']\n",
    "fianl_train_all = new_data_normal_del[new_columns_name]\n",
    "fianl_train_all.to_csv('train_data_40.csv',index = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'group'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   1944\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1945\u001b[0;31m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1946\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4154)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4018)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12368)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12322)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mKeyError\u001b[0m: 'group'",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-25-0306a9de7f3e>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mnew_data_normal_group\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnew_train_data_stat\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'group'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m   1995\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_multilevel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1996\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1997\u001b[0;31m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_column\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1998\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1999\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_getitem_column\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m_getitem_column\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m   2002\u001b[0m         \u001b[1;31m# get column\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   2003\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mis_unique\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2004\u001b[0;31m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_item_cache\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2005\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   2006\u001b[0m         \u001b[1;31m# duplicate columns & possible reduce dimensionality\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m_get_item_cache\u001b[0;34m(self, item)\u001b[0m\n\u001b[1;32m   1348\u001b[0m         \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcache\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1349\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mres\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1350\u001b[0;31m             \u001b[0mvalues\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_data\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1351\u001b[0m             \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_box_item_values\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mvalues\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1352\u001b[0m             \u001b[0mcache\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mres\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\internals.py\u001b[0m in \u001b[0;36mget\u001b[0;34m(self, item, fastpath)\u001b[0m\n\u001b[1;32m   3288\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   3289\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0misnull\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3290\u001b[0;31m                 \u001b[0mloc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3291\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   3292\u001b[0m                 \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0misnull\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   1945\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1946\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1947\u001b[0;31m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_maybe_cast_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1948\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1949\u001b[0m         \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmethod\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmethod\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtolerance\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtolerance\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4154)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4018)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12368)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12322)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mKeyError\u001b[0m: 'group'"
     ]
    }
   ],
   "source": [
    "new_data_normal_group = new_train_data_stat['group']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "new_data_normal.drop(['group_mean','group_med','group_max','group_min','group_std','group_mod'],axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#还要删除存在缺失值的行，因为有些group中只有一个有效样本，这样标准差就是Nan，所以这个group就要删掉\n",
    "new_data_normal_del = new_data_normal.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)\n",
    "\n",
    "#new_data_normal_del.to_csv('train_data_7_28_all.csv',index = False)\n",
    "#new_data_normal_del里面就是归一化的所有特征的值了。下一步是看特征重要性\n",
    "new_columns_name = ['label','e_i_t_max','air_density_20_med', 'power_max', 'pitch2_moto_tmp_mean', 'e_p2_m_t_max',\n",
    "         'power_min', 'yaw_position_mean', 'acc_x_mean', 'e_i_t_min', 'e_i_t_mod', 'power_mean', 'generator_speed_min',\n",
    "         'yaw_ch1_mean', 'environment_tmp_mean', 'air_density_20_mean', 'pitch1_angle_mean', 'air_density_20_min', 'pitch2_moto_tmp_min',\n",
    "         'acc_y_mean', 'e_p2_m_t_min', 'i_p2_m_t_max', 'e_p2_m_t_mean', 'air_density_20_std', 'int_tmp_mean', 'e_p2_m_t_mod',\n",
    "         'air_density_20_max', 'power_mod', 'environment_tmp_min', 'pitch1_angle_max', 'environment_tmp_max', 'wind_speed_max',\n",
    "         'e_i_t_mean', 'wind_direction_mean_std', 'power_med', 'yaw_position_min', 'pitch1_angle_min', 'int_tmp_min', 'yaw_position_max',\n",
    "         'wind_speed_std', 'pitch2_moto_tmp_max']\n",
    "fianl_train_all = new_data_normal_del[new_columns_name]\n",
    "fianl_train_all.to_csv('train_data_40.csv',index = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#给原始数据添加label\n",
    "def add_label(train_data,d_n,d_f):\n",
    "    train_data_new_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in train_data['time'] ]\n",
    "\n",
    "    #转换normal和failure的时间格式\n",
    "    d_n_start_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_n['startTime'] ]\n",
    "    d_n_end_time= [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_n['endTime'] ]\n",
    "    d_f_start_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_f['startTime'] ]\n",
    "    d_f_end_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_f['endTime'] ]\n",
    "\n",
    "    label_list_2=[]\n",
    "    t1 = 0\n",
    "    t2 = 0\n",
    "    l1 = 0\n",
    "    l2 = 0\n",
    "    for i in range(0,len(train_data)):\n",
    "        if ((train_data_new_time[i] >= d_n_start_time[t1]) &\n",
    "        (train_data_new_time[i] <= d_n_end_time[t1])):\n",
    "            label_list_2.append(0)\n",
    "            l1 = 1\n",
    "        elif ((train_data_new_time[i] >= d_f_start_time[t2]) &\n",
    "        (train_data_new_time[i] <= d_f_end_time[t2])):\n",
    "            label_list_2.append(1)\n",
    "            l2 = 2\n",
    "        else:\n",
    "            if l1 == 1: \n",
    "                t1 = t1 +1 \n",
    "                if t1 == len(d_n_end_time):\n",
    "                    t1 = len(d_n_end_time)-1\n",
    "            if l2 == 2:\n",
    "                t2 = t2 + 1\n",
    "                if t2 == len(d_f_end_time):\n",
    "                    t1 = len(d_f_end_time)-1\n",
    "            l1 = 0 \n",
    "            l2 = 0\n",
    "            label_list_2.append(2)\n",
    "\n",
    "    label_s=pd.Series(label_list_2)\n",
    "    train_data['label'] = label_s\n",
    "    return train_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#归一化\n",
    "def normalization(train_data):\n",
    "    #这几个公式的值的差别太大了，要做归一化\n",
    "    #全部都做归一化算了\n",
    "    #两种不同的归一化，都可以试一下\n",
    "    #train_data_normal = train_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))\n",
    "\n",
    "    train_data_normal = train_data.apply(lambda x: (x - np.mean(x)) / (np.std(x)))\n",
    "    return train_data_normal"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#按照每个新划分的group计算统计值\n",
    "def cal_statist(train_data):\n",
    "    #mod没有现成的函数可以调用，所以只能自己写函数。求众数\n",
    "    mod = lambda x : np.argmax(x.value_counts())\n",
    "    columns_name = train_data.columns\n",
    "    new_data = pd.DataFrame(train_data['label'].groupby(train_data['new_group']).mean())\n",
    "    new_data = new_data.reset_index( drop=True)\n",
    "    for i in tqdm(range(1,len(train_data.columns)-2)):\n",
    "        mean_name = columns_name[i]+'_mean'\n",
    "        med_name = columns_name[i]+'_med'\n",
    "        max_name = columns_name[i]+'_max'\n",
    "        min_name = columns_name[i]+'_min'\n",
    "        std_name = columns_name[i]+'_std'\n",
    "        mod_name = columns_name[i]+'_mod'\n",
    "\n",
    "        t_mean = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).mean().rename(mean_name)).reset_index( drop=True)\n",
    "        t_median = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).median().rename(med_name)).reset_index( drop=True)\n",
    "        t_max = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).max().rename(max_name)).reset_index( drop=True)\n",
    "        t_min = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).min().rename(min_name)).reset_index( drop=True)\n",
    "        t_std = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).std().rename(std_name)).reset_index( drop=True)\n",
    "        t_mod = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).apply(mod).rename(mod_name)).reset_index( drop=True)\n",
    "\n",
    "        new_data = pd.concat([new_data,t_mean,t_median,t_max,t_min,t_std,t_mod],axis = 1)\n",
    "\n",
    "    return new_data\t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#把一个group分成三份，并且赋新的组号\n",
    "def add_group(train_data):\n",
    "\n",
    "    train_data = train_data.reset_index(drop=True)\n",
    "    old_group = train_data['group']\n",
    "    new_group = []\n",
    "    group_start = 0\n",
    "    new_group_index = 0\n",
    "    for i in tqdm(range(len(train_data)-1)):\n",
    "        if old_group[i] != old_group[i+1]:\n",
    "            if (i-group_start)>5:\n",
    "                for j in range(0,int((i-group_start)/3)):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "                for j in range(int((i-group_start)/3),int((i-group_start)/3*2)):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "                for j in range(int((i-group_start)/3*2),i-group_start+1):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "            else:\n",
    "                for j in range(0,(i-group_start)+1):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "            group_start = i+1\n",
    "        if i == len(train_data)-2 :\n",
    "            for j in range(0,int((i-group_start)/3)):\n",
    "                new_group.append(new_group_index)\n",
    "            new_group_index += 1\n",
    "            for j in range(int((i-group_start)/3),int((i-group_start)/3*2)):\n",
    "                new_group.append(new_group_index)\n",
    "            new_group_index += 1\n",
    "            for j in range(int((i-group_start)/3*2),i-group_start+2):\n",
    "                new_group.append(new_group_index)\n",
    "            new_group_index += 1\n",
    "    train_data['new_group'] = new_group\n",
    "    return train_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "new_group_d = pd.DataFrame(new_group,columns=['new_group'])\n",
    "new_group_d.to_csv('new_group_d.csv',index=False)\n",
    "train_data_d = pd.DataFrame(list(train_data['group']),columns=['train_data_d'])\n",
    "train_data_d.to_csv('train_data_d.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#计算不同列之间的差值，增加特征\t\n",
    "def add_columns(train_data):\n",
    "    #计算不同列之间的差值\t\n",
    "    #def cal_diff(train_data):\n",
    "    #环境温度减\n",
    "    train_data['e_p2_ng_t'] = train_data['environment_tmp'] - train_data['pitch2_ng5_tmp']    \n",
    "    train_data['e_p2_m_t'] = train_data['environment_tmp'] - train_data['pitch2_moto_tmp']\n",
    "\n",
    "    #机舱温度减\n",
    "    train_data['i_p1_ng_t'] = train_data['int_tmp'] - train_data['pitch1_ng5_tmp']\n",
    "    train_data['i_p2_m_t'] = train_data['int_tmp'] - train_data['pitch2_moto_tmp']\n",
    "\n",
    "    #机舱温度和环境温度的差\n",
    "    train_data['e_i_t'] = train_data['environment_tmp'] - train_data['int_tmp']\n",
    "\n",
    "\n",
    "\n",
    "    #计算偏航系统的特征值\n",
    "    #def cal_yaw(train_data):\n",
    "    # 计算风速速度的次方\n",
    "    train_data['wind_speed_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_speed']]\n",
    "    train_data['wind_direction_mean_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in train_data['wind_direction_mean']]\n",
    "    train_data['wind_direction_mean_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_direction_mean_cos']]\n",
    "    # 计算偏航位置特征\n",
    "    train_data['yaw_ch1'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_position']]\n",
    "    train_data['yaw_ch2'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_speed']]\n",
    "    #计算空气的密度\n",
    "    train_data['air_density_20'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos_2']*train_data['yaw_ch2'])\n",
    "\n",
    "    train_data.drop(['yaw_ch2','wind_direction_mean_cos_2','wind_direction_mean_cos'],axis=1, inplace=True)\n",
    "\n",
    "\n",
    "    return train_data"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [default]",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import time\n",
    "import math\n",
    "from tqdm import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|███████████████████████████████| 553710/553710 [00:21<00:00, 25473.55it/s]\n",
      "100%|██████████████████████████████████████████| 29/29 [06:33<00:00, 13.49s/it]\n"
     ]
    }
   ],
   "source": [
    "data_15 = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\15\\\\15_data.csv')\n",
    "data_15.drop(['pitch2_angle','pitch3_angle','pitch1_moto_tmp','pitch3_moto_tmp','pitch2_speed','pitch3_speed'],axis=1, inplace=True)\n",
    "d_f = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\15\\\\15_failureInfo.csv')\n",
    "d_n = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\15\\\\15_normalInfo.csv')\n",
    "\n",
    "\n",
    "data_15 = add_label(data_15,d_n,d_f)\n",
    "\n",
    "data_21 = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\21\\\\21_data.csv')\n",
    "data_21.drop(['pitch2_angle','pitch3_angle','pitch1_moto_tmp','pitch3_moto_tmp','pitch2_speed','pitch3_speed'],axis=1, inplace=True)\n",
    "d_f = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\21\\\\21_failureInfo.csv')\n",
    "d_n = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\21\\\\21_normalInfo.csv')\n",
    "\n",
    "\n",
    "data_21 = add_label(data_21,d_n,d_f)\n",
    "train_data = pd.concat([data_15,data_21],axis = 0)\n",
    "train_data = train_data[train_data['label'].isin([1])].append(train_data[train_data['label'].isin([0])])\n",
    "\n",
    "\n",
    "#重新索引\n",
    "train_data = train_data.reset_index(drop=True)\n",
    "\n",
    "train_data = add_group(train_data)\n",
    "\n",
    "#先做一些加减的运算，然后再求平均值之类的。\n",
    "train_data = add_columns(train_data)\n",
    "\n",
    "#修改列名\n",
    "columns_list = list(train_data)\n",
    "columns_list.remove('label')\n",
    "columns_list.remove('group')\n",
    "columns_list.remove('new_group')\n",
    "columns_list.append('group')\n",
    "columns_list.append('new_group')\n",
    "columns_list.append('label')\n",
    "new_train_data = train_data.loc[:,columns_list]\n",
    "\n",
    "\n",
    "new_train_data_stat = cal_statist(new_train_data)\n",
    "new_data_normal_label = new_train_data_stat['label']\n",
    "#new_data_normal_group = new_train_data_stat['group']\n",
    "new_data_normal = normalization(new_train_data_stat)\n",
    "new_data_normal['label'] = new_data_normal_label\n",
    "\n",
    "#还要删除存在缺失值的行，因为有些group中只有一个有效样本，这样标准差就是Nan，所以这个group就要删掉\n",
    "new_data_normal_del = new_data_normal.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)\n",
    "\n",
    "#new_data_normal_del.to_csv('train_data_7_28_all.csv',index = False)\n",
    "#new_data_normal_del里面就是归一化的所有特征的值了。下一步是看特征重要性\n",
    "new_columns_name = ['label','e_i_t_max','air_density_20_med', 'power_max', 'pitch2_moto_tmp_mean', 'e_p2_m_t_max',\n",
    "         'power_min', 'yaw_position_mean', 'acc_x_mean', 'e_i_t_min', 'e_i_t_mod', 'power_mean', 'generator_speed_min',\n",
    "         'yaw_ch1_mean', 'environment_tmp_mean', 'air_density_20_mean', 'pitch1_angle_mean', 'air_density_20_min', 'pitch2_moto_tmp_min',\n",
    "         'acc_y_mean', 'e_p2_m_t_min', 'i_p2_m_t_max', 'e_p2_m_t_mean', 'air_density_20_std', 'int_tmp_mean', 'e_p2_m_t_mod',\n",
    "         'air_density_20_max', 'power_mod', 'environment_tmp_min', 'pitch1_angle_max', 'environment_tmp_max', 'wind_speed_max',\n",
    "         'e_i_t_mean', 'wind_direction_mean_std', 'power_med', 'yaw_position_min', 'pitch1_angle_min', 'int_tmp_min', 'yaw_position_max',\n",
    "         'wind_speed_std', 'pitch2_moto_tmp_max']\n",
    "fianl_train_all = new_data_normal_del[new_columns_name]\n",
    "fianl_train_all.to_csv('train_data_40.csv',index = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'group'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   1944\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1945\u001b[0;31m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1946\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4154)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4018)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12368)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12322)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mKeyError\u001b[0m: 'group'",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-25-0306a9de7f3e>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mnew_data_normal_group\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnew_train_data_stat\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'group'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m   1995\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_multilevel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1996\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1997\u001b[0;31m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_column\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1998\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1999\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_getitem_column\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m_getitem_column\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m   2002\u001b[0m         \u001b[1;31m# get column\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   2003\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mis_unique\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2004\u001b[0;31m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_item_cache\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2005\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   2006\u001b[0m         \u001b[1;31m# duplicate columns & possible reduce dimensionality\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m_get_item_cache\u001b[0;34m(self, item)\u001b[0m\n\u001b[1;32m   1348\u001b[0m         \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcache\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1349\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mres\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1350\u001b[0;31m             \u001b[0mvalues\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_data\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1351\u001b[0m             \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_box_item_values\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mvalues\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1352\u001b[0m             \u001b[0mcache\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mres\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\internals.py\u001b[0m in \u001b[0;36mget\u001b[0;34m(self, item, fastpath)\u001b[0m\n\u001b[1;32m   3288\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   3289\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0misnull\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3290\u001b[0;31m                 \u001b[0mloc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3291\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   3292\u001b[0m                 \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0misnull\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   1945\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1946\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1947\u001b[0;31m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_maybe_cast_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1948\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1949\u001b[0m         \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmethod\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmethod\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtolerance\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtolerance\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4154)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4018)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12368)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12322)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mKeyError\u001b[0m: 'group'"
     ]
    }
   ],
   "source": [
    "new_data_normal_group = new_train_data_stat['group']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "new_data_normal.drop(['group_mean','group_med','group_max','group_min','group_std','group_mod'],axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#还要删除存在缺失值的行，因为有些group中只有一个有效样本，这样标准差就是Nan，所以这个group就要删掉\n",
    "new_data_normal_del = new_data_normal.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)\n",
    "\n",
    "#new_data_normal_del.to_csv('train_data_7_28_all.csv',index = False)\n",
    "#new_data_normal_del里面就是归一化的所有特征的值了。下一步是看特征重要性\n",
    "new_columns_name = ['label','e_i_t_max','air_density_20_med', 'power_max', 'pitch2_moto_tmp_mean', 'e_p2_m_t_max',\n",
    "         'power_min', 'yaw_position_mean', 'acc_x_mean', 'e_i_t_min', 'e_i_t_mod', 'power_mean', 'generator_speed_min',\n",
    "         'yaw_ch1_mean', 'environment_tmp_mean', 'air_density_20_mean', 'pitch1_angle_mean', 'air_density_20_min', 'pitch2_moto_tmp_min',\n",
    "         'acc_y_mean', 'e_p2_m_t_min', 'i_p2_m_t_max', 'e_p2_m_t_mean', 'air_density_20_std', 'int_tmp_mean', 'e_p2_m_t_mod',\n",
    "         'air_density_20_max', 'power_mod', 'environment_tmp_min', 'pitch1_angle_max', 'environment_tmp_max', 'wind_speed_max',\n",
    "         'e_i_t_mean', 'wind_direction_mean_std', 'power_med', 'yaw_position_min', 'pitch1_angle_min', 'int_tmp_min', 'yaw_position_max',\n",
    "         'wind_speed_std', 'pitch2_moto_tmp_max']\n",
    "fianl_train_all = new_data_normal_del[new_columns_name]\n",
    "fianl_train_all.to_csv('train_data_40.csv',index = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#给原始数据添加label\n",
    "def add_label(train_data,d_n,d_f):\n",
    "    train_data_new_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in train_data['time'] ]\n",
    "\n",
    "    #转换normal和failure的时间格式\n",
    "    d_n_start_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_n['startTime'] ]\n",
    "    d_n_end_time= [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_n['endTime'] ]\n",
    "    d_f_start_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_f['startTime'] ]\n",
    "    d_f_end_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_f['endTime'] ]\n",
    "\n",
    "    label_list_2=[]\n",
    "    t1 = 0\n",
    "    t2 = 0\n",
    "    l1 = 0\n",
    "    l2 = 0\n",
    "    for i in range(0,len(train_data)):\n",
    "        if ((train_data_new_time[i] >= d_n_start_time[t1]) &\n",
    "        (train_data_new_time[i] <= d_n_end_time[t1])):\n",
    "            label_list_2.append(0)\n",
    "            l1 = 1\n",
    "        elif ((train_data_new_time[i] >= d_f_start_time[t2]) &\n",
    "        (train_data_new_time[i] <= d_f_end_time[t2])):\n",
    "            label_list_2.append(1)\n",
    "            l2 = 2\n",
    "        else:\n",
    "            if l1 == 1: \n",
    "                t1 = t1 +1 \n",
    "                if t1 == len(d_n_end_time):\n",
    "                    t1 = len(d_n_end_time)-1\n",
    "            if l2 == 2:\n",
    "                t2 = t2 + 1\n",
    "                if t2 == len(d_f_end_time):\n",
    "                    t1 = len(d_f_end_time)-1\n",
    "            l1 = 0 \n",
    "            l2 = 0\n",
    "            label_list_2.append(2)\n",
    "\n",
    "    label_s=pd.Series(label_list_2)\n",
    "    train_data['label'] = label_s\n",
    "    return train_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#归一化\n",
    "def normalization(train_data):\n",
    "    #这几个公式的值的差别太大了，要做归一化\n",
    "    #全部都做归一化算了\n",
    "    #两种不同的归一化，都可以试一下\n",
    "    #train_data_normal = train_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))\n",
    "\n",
    "    train_data_normal = train_data.apply(lambda x: (x - np.mean(x)) / (np.std(x)))\n",
    "    return train_data_normal"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#按照每个新划分的group计算统计值\n",
    "def cal_statist(train_data):\n",
    "    #mod没有现成的函数可以调用，所以只能自己写函数。求众数\n",
    "    mod = lambda x : np.argmax(x.value_counts())\n",
    "    columns_name = train_data.columns\n",
    "    new_data = pd.DataFrame(train_data['label'].groupby(train_data['new_group']).mean())\n",
    "    new_data = new_data.reset_index( drop=True)\n",
    "    for i in tqdm(range(1,len(train_data.columns)-2)):\n",
    "        mean_name = columns_name[i]+'_mean'\n",
    "        med_name = columns_name[i]+'_med'\n",
    "        max_name = columns_name[i]+'_max'\n",
    "        min_name = columns_name[i]+'_min'\n",
    "        std_name = columns_name[i]+'_std'\n",
    "        mod_name = columns_name[i]+'_mod'\n",
    "\n",
    "        t_mean = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).mean().rename(mean_name)).reset_index( drop=True)\n",
    "        t_median = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).median().rename(med_name)).reset_index( drop=True)\n",
    "        t_max = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).max().rename(max_name)).reset_index( drop=True)\n",
    "        t_min = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).min().rename(min_name)).reset_index( drop=True)\n",
    "        t_std = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).std().rename(std_name)).reset_index( drop=True)\n",
    "        t_mod = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).apply(mod).rename(mod_name)).reset_index( drop=True)\n",
    "\n",
    "        new_data = pd.concat([new_data,t_mean,t_median,t_max,t_min,t_std,t_mod],axis = 1)\n",
    "\n",
    "    return new_data\t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#把一个group分成三份，并且赋新的组号\n",
    "def add_group(train_data):\n",
    "\n",
    "    train_data = train_data.reset_index(drop=True)\n",
    "    old_group = train_data['group']\n",
    "    new_group = []\n",
    "    group_start = 0\n",
    "    new_group_index = 0\n",
    "    for i in tqdm(range(len(train_data)-1)):\n",
    "        if old_group[i] != old_group[i+1]:\n",
    "            if (i-group_start)>5:\n",
    "                for j in range(0,int((i-group_start)/3)):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "                for j in range(int((i-group_start)/3),int((i-group_start)/3*2)):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "                for j in range(int((i-group_start)/3*2),i-group_start+1):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "            else:\n",
    "                for j in range(0,(i-group_start)+1):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "            group_start = i+1\n",
    "        if i == len(train_data)-2 :\n",
    "            for j in range(0,int((i-group_start)/3)):\n",
    "                new_group.append(new_group_index)\n",
    "            new_group_index += 1\n",
    "            for j in range(int((i-group_start)/3),int((i-group_start)/3*2)):\n",
    "                new_group.append(new_group_index)\n",
    "            new_group_index += 1\n",
    "            for j in range(int((i-group_start)/3*2),i-group_start+2):\n",
    "                new_group.append(new_group_index)\n",
    "            new_group_index += 1\n",
    "    train_data['new_group'] = new_group\n",
    "    return train_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "new_group_d = pd.DataFrame(new_group,columns=['new_group'])\n",
    "new_group_d.to_csv('new_group_d.csv',index=False)\n",
    "train_data_d = pd.DataFrame(list(train_data['group']),columns=['train_data_d'])\n",
    "train_data_d.to_csv('train_data_d.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#计算不同列之间的差值，增加特征\t\n",
    "def add_columns(train_data):\n",
    "    #计算不同列之间的差值\t\n",
    "    #def cal_diff(train_data):\n",
    "    #环境温度减\n",
    "    train_data['e_p2_ng_t'] = train_data['environment_tmp'] - train_data['pitch2_ng5_tmp']    \n",
    "    train_data['e_p2_m_t'] = train_data['environment_tmp'] - train_data['pitch2_moto_tmp']\n",
    "\n",
    "    #机舱温度减\n",
    "    train_data['i_p1_ng_t'] = train_data['int_tmp'] - train_data['pitch1_ng5_tmp']\n",
    "    train_data['i_p2_m_t'] = train_data['int_tmp'] - train_data['pitch2_moto_tmp']\n",
    "\n",
    "    #机舱温度和环境温度的差\n",
    "    train_data['e_i_t'] = train_data['environment_tmp'] - train_data['int_tmp']\n",
    "\n",
    "\n",
    "\n",
    "    #计算偏航系统的特征值\n",
    "    #def cal_yaw(train_data):\n",
    "    # 计算风速速度的次方\n",
    "    train_data['wind_speed_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_speed']]\n",
    "    train_data['wind_direction_mean_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in train_data['wind_direction_mean']]\n",
    "    train_data['wind_direction_mean_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_direction_mean_cos']]\n",
    "    # 计算偏航位置特征\n",
    "    train_data['yaw_ch1'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_position']]\n",
    "    train_data['yaw_ch2'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_speed']]\n",
    "    #计算空气的密度\n",
    "    train_data['air_density_20'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos_2']*train_data['yaw_ch2'])\n",
    "\n",
    "    train_data.drop(['yaw_ch2','wind_direction_mean_cos_2','wind_direction_mean_cos'],axis=1, inplace=True)\n",
    "\n",
    "\n",
    "    return train_data"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [default]",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import time\n",
    "import math\n",
    "from tqdm import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|███████████████████████████████| 553710/553710 [00:21<00:00, 25473.55it/s]\n",
      "100%|██████████████████████████████████████████| 29/29 [06:33<00:00, 13.49s/it]\n"
     ]
    }
   ],
   "source": [
    "data_15 = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\15\\\\15_data.csv')\n",
    "data_15.drop(['pitch2_angle','pitch3_angle','pitch1_moto_tmp','pitch3_moto_tmp','pitch2_speed','pitch3_speed'],axis=1, inplace=True)\n",
    "d_f = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\15\\\\15_failureInfo.csv')\n",
    "d_n = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\15\\\\15_normalInfo.csv')\n",
    "\n",
    "\n",
    "data_15 = add_label(data_15,d_n,d_f)\n",
    "\n",
    "data_21 = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\21\\\\21_data.csv')\n",
    "data_21.drop(['pitch2_angle','pitch3_angle','pitch1_moto_tmp','pitch3_moto_tmp','pitch2_speed','pitch3_speed'],axis=1, inplace=True)\n",
    "d_f = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\21\\\\21_failureInfo.csv')\n",
    "d_n = pd.read_csv('E:\\\\competition\\\\ice\\\\train\\\\21\\\\21_normalInfo.csv')\n",
    "\n",
    "\n",
    "data_21 = add_label(data_21,d_n,d_f)\n",
    "train_data = pd.concat([data_15,data_21],axis = 0)\n",
    "train_data = train_data[train_data['label'].isin([1])].append(train_data[train_data['label'].isin([0])])\n",
    "\n",
    "\n",
    "#重新索引\n",
    "train_data = train_data.reset_index(drop=True)\n",
    "\n",
    "train_data = add_group(train_data)\n",
    "\n",
    "#先做一些加减的运算，然后再求平均值之类的。\n",
    "train_data = add_columns(train_data)\n",
    "\n",
    "#修改列名\n",
    "columns_list = list(train_data)\n",
    "columns_list.remove('label')\n",
    "columns_list.remove('group')\n",
    "columns_list.remove('new_group')\n",
    "columns_list.append('group')\n",
    "columns_list.append('new_group')\n",
    "columns_list.append('label')\n",
    "new_train_data = train_data.loc[:,columns_list]\n",
    "\n",
    "\n",
    "new_train_data_stat = cal_statist(new_train_data)\n",
    "new_data_normal_label = new_train_data_stat['label']\n",
    "#new_data_normal_group = new_train_data_stat['group']\n",
    "new_data_normal = normalization(new_train_data_stat)\n",
    "new_data_normal['label'] = new_data_normal_label\n",
    "\n",
    "#还要删除存在缺失值的行，因为有些group中只有一个有效样本，这样标准差就是Nan，所以这个group就要删掉\n",
    "new_data_normal_del = new_data_normal.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)\n",
    "\n",
    "#new_data_normal_del.to_csv('train_data_7_28_all.csv',index = False)\n",
    "#new_data_normal_del里面就是归一化的所有特征的值了。下一步是看特征重要性\n",
    "new_columns_name = ['label','e_i_t_max','air_density_20_med', 'power_max', 'pitch2_moto_tmp_mean', 'e_p2_m_t_max',\n",
    "         'power_min', 'yaw_position_mean', 'acc_x_mean', 'e_i_t_min', 'e_i_t_mod', 'power_mean', 'generator_speed_min',\n",
    "         'yaw_ch1_mean', 'environment_tmp_mean', 'air_density_20_mean', 'pitch1_angle_mean', 'air_density_20_min', 'pitch2_moto_tmp_min',\n",
    "         'acc_y_mean', 'e_p2_m_t_min', 'i_p2_m_t_max', 'e_p2_m_t_mean', 'air_density_20_std', 'int_tmp_mean', 'e_p2_m_t_mod',\n",
    "         'air_density_20_max', 'power_mod', 'environment_tmp_min', 'pitch1_angle_max', 'environment_tmp_max', 'wind_speed_max',\n",
    "         'e_i_t_mean', 'wind_direction_mean_std', 'power_med', 'yaw_position_min', 'pitch1_angle_min', 'int_tmp_min', 'yaw_position_max',\n",
    "         'wind_speed_std', 'pitch2_moto_tmp_max']\n",
    "fianl_train_all = new_data_normal_del[new_columns_name]\n",
    "fianl_train_all.to_csv('train_data_40.csv',index = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'group'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   1944\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1945\u001b[0;31m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1946\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4154)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4018)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12368)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12322)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mKeyError\u001b[0m: 'group'",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-25-0306a9de7f3e>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mnew_data_normal_group\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnew_train_data_stat\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'group'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m   1995\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_multilevel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1996\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1997\u001b[0;31m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_column\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1998\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1999\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_getitem_column\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m_getitem_column\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m   2002\u001b[0m         \u001b[1;31m# get column\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   2003\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mis_unique\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2004\u001b[0;31m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_item_cache\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2005\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   2006\u001b[0m         \u001b[1;31m# duplicate columns & possible reduce dimensionality\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m_get_item_cache\u001b[0;34m(self, item)\u001b[0m\n\u001b[1;32m   1348\u001b[0m         \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcache\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1349\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mres\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1350\u001b[0;31m             \u001b[0mvalues\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_data\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1351\u001b[0m             \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_box_item_values\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mvalues\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1352\u001b[0m             \u001b[0mcache\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mres\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\internals.py\u001b[0m in \u001b[0;36mget\u001b[0;34m(self, item, fastpath)\u001b[0m\n\u001b[1;32m   3288\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   3289\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0misnull\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3290\u001b[0;31m                 \u001b[0mloc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3291\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   3292\u001b[0m                 \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0misnull\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   1945\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1946\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1947\u001b[0;31m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_maybe_cast_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1948\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   1949\u001b[0m         \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmethod\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmethod\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtolerance\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtolerance\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4154)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\index.pyx\u001b[0m in \u001b[0;36mpandas.index.IndexEngine.get_loc (pandas\\index.c:4018)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12368)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas\\hashtable.pyx\u001b[0m in \u001b[0;36mpandas.hashtable.PyObjectHashTable.get_item (pandas\\hashtable.c:12322)\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mKeyError\u001b[0m: 'group'"
     ]
    }
   ],
   "source": [
    "new_data_normal_group = new_train_data_stat['group']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "new_data_normal.drop(['group_mean','group_med','group_max','group_min','group_std','group_mod'],axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#还要删除存在缺失值的行，因为有些group中只有一个有效样本，这样标准差就是Nan，所以这个group就要删掉\n",
    "new_data_normal_del = new_data_normal.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)\n",
    "\n",
    "#new_data_normal_del.to_csv('train_data_7_28_all.csv',index = False)\n",
    "#new_data_normal_del里面就是归一化的所有特征的值了。下一步是看特征重要性\n",
    "new_columns_name = ['label','e_i_t_max','air_density_20_med', 'power_max', 'pitch2_moto_tmp_mean', 'e_p2_m_t_max',\n",
    "         'power_min', 'yaw_position_mean', 'acc_x_mean', 'e_i_t_min', 'e_i_t_mod', 'power_mean', 'generator_speed_min',\n",
    "         'yaw_ch1_mean', 'environment_tmp_mean', 'air_density_20_mean', 'pitch1_angle_mean', 'air_density_20_min', 'pitch2_moto_tmp_min',\n",
    "         'acc_y_mean', 'e_p2_m_t_min', 'i_p2_m_t_max', 'e_p2_m_t_mean', 'air_density_20_std', 'int_tmp_mean', 'e_p2_m_t_mod',\n",
    "         'air_density_20_max', 'power_mod', 'environment_tmp_min', 'pitch1_angle_max', 'environment_tmp_max', 'wind_speed_max',\n",
    "         'e_i_t_mean', 'wind_direction_mean_std', 'power_med', 'yaw_position_min', 'pitch1_angle_min', 'int_tmp_min', 'yaw_position_max',\n",
    "         'wind_speed_std', 'pitch2_moto_tmp_max']\n",
    "fianl_train_all = new_data_normal_del[new_columns_name]\n",
    "fianl_train_all.to_csv('train_data_40.csv',index = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#给原始数据添加label\n",
    "def add_label(train_data,d_n,d_f):\n",
    "    train_data_new_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in train_data['time'] ]\n",
    "\n",
    "    #转换normal和failure的时间格式\n",
    "    d_n_start_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_n['startTime'] ]\n",
    "    d_n_end_time= [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_n['endTime'] ]\n",
    "    d_f_start_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_f['startTime'] ]\n",
    "    d_f_end_time = [(lambda x : time.mktime(time.strptime(x,\"%Y-%m-%d %H:%M:%S\")))(x) for x in d_f['endTime'] ]\n",
    "\n",
    "    label_list_2=[]\n",
    "    t1 = 0\n",
    "    t2 = 0\n",
    "    l1 = 0\n",
    "    l2 = 0\n",
    "    for i in range(0,len(train_data)):\n",
    "        if ((train_data_new_time[i] >= d_n_start_time[t1]) &\n",
    "        (train_data_new_time[i] <= d_n_end_time[t1])):\n",
    "            label_list_2.append(0)\n",
    "            l1 = 1\n",
    "        elif ((train_data_new_time[i] >= d_f_start_time[t2]) &\n",
    "        (train_data_new_time[i] <= d_f_end_time[t2])):\n",
    "            label_list_2.append(1)\n",
    "            l2 = 2\n",
    "        else:\n",
    "            if l1 == 1: \n",
    "                t1 = t1 +1 \n",
    "                if t1 == len(d_n_end_time):\n",
    "                    t1 = len(d_n_end_time)-1\n",
    "            if l2 == 2:\n",
    "                t2 = t2 + 1\n",
    "                if t2 == len(d_f_end_time):\n",
    "                    t1 = len(d_f_end_time)-1\n",
    "            l1 = 0 \n",
    "            l2 = 0\n",
    "            label_list_2.append(2)\n",
    "\n",
    "    label_s=pd.Series(label_list_2)\n",
    "    train_data['label'] = label_s\n",
    "    return train_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#归一化\n",
    "def normalization(train_data):\n",
    "    #这几个公式的值的差别太大了，要做归一化\n",
    "    #全部都做归一化算了\n",
    "    #两种不同的归一化，都可以试一下\n",
    "    #train_data_normal = train_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))\n",
    "\n",
    "    train_data_normal = train_data.apply(lambda x: (x - np.mean(x)) / (np.std(x)))\n",
    "    return train_data_normal"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#按照每个新划分的group计算统计值\n",
    "def cal_statist(train_data):\n",
    "    #mod没有现成的函数可以调用，所以只能自己写函数。求众数\n",
    "    mod = lambda x : np.argmax(x.value_counts())\n",
    "    columns_name = train_data.columns\n",
    "    new_data = pd.DataFrame(train_data['label'].groupby(train_data['new_group']).mean())\n",
    "    new_data = new_data.reset_index( drop=True)\n",
    "    for i in tqdm(range(1,len(train_data.columns)-2)):\n",
    "        mean_name = columns_name[i]+'_mean'\n",
    "        med_name = columns_name[i]+'_med'\n",
    "        max_name = columns_name[i]+'_max'\n",
    "        min_name = columns_name[i]+'_min'\n",
    "        std_name = columns_name[i]+'_std'\n",
    "        mod_name = columns_name[i]+'_mod'\n",
    "\n",
    "        t_mean = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).mean().rename(mean_name)).reset_index( drop=True)\n",
    "        t_median = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).median().rename(med_name)).reset_index( drop=True)\n",
    "        t_max = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).max().rename(max_name)).reset_index( drop=True)\n",
    "        t_min = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).min().rename(min_name)).reset_index( drop=True)\n",
    "        t_std = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).std().rename(std_name)).reset_index( drop=True)\n",
    "        t_mod = pd.DataFrame(train_data[columns_name[i]].groupby(train_data['new_group']).apply(mod).rename(mod_name)).reset_index( drop=True)\n",
    "\n",
    "        new_data = pd.concat([new_data,t_mean,t_median,t_max,t_min,t_std,t_mod],axis = 1)\n",
    "\n",
    "    return new_data\t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#把一个group分成三份，并且赋新的组号\n",
    "def add_group(train_data):\n",
    "\n",
    "    train_data = train_data.reset_index(drop=True)\n",
    "    old_group = train_data['group']\n",
    "    new_group = []\n",
    "    group_start = 0\n",
    "    new_group_index = 0\n",
    "    for i in tqdm(range(len(train_data)-1)):\n",
    "        if old_group[i] != old_group[i+1]:\n",
    "            if (i-group_start)>5:\n",
    "                for j in range(0,int((i-group_start)/3)):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "                for j in range(int((i-group_start)/3),int((i-group_start)/3*2)):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "                for j in range(int((i-group_start)/3*2),i-group_start+1):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "            else:\n",
    "                for j in range(0,(i-group_start)+1):\n",
    "                    new_group.append(new_group_index)\n",
    "                new_group_index += 1\n",
    "            group_start = i+1\n",
    "        if i == len(train_data)-2 :\n",
    "            for j in range(0,int((i-group_start)/3)):\n",
    "                new_group.append(new_group_index)\n",
    "            new_group_index += 1\n",
    "            for j in range(int((i-group_start)/3),int((i-group_start)/3*2)):\n",
    "                new_group.append(new_group_index)\n",
    "            new_group_index += 1\n",
    "            for j in range(int((i-group_start)/3*2),i-group_start+2):\n",
    "                new_group.append(new_group_index)\n",
    "            new_group_index += 1\n",
    "    train_data['new_group'] = new_group\n",
    "    return train_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "new_group_d = pd.DataFrame(new_group,columns=['new_group'])\n",
    "new_group_d.to_csv('new_group_d.csv',index=False)\n",
    "train_data_d = pd.DataFrame(list(train_data['group']),columns=['train_data_d'])\n",
    "train_data_d.to_csv('train_data_d.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#计算不同列之间的差值，增加特征\t\n",
    "def add_columns(train_data):\n",
    "    #计算不同列之间的差值\t\n",
    "    #def cal_diff(train_data):\n",
    "    #环境温度减\n",
    "    train_data['e_p2_ng_t'] = train_data['environment_tmp'] - train_data['pitch2_ng5_tmp']    \n",
    "    train_data['e_p2_m_t'] = train_data['environment_tmp'] - train_data['pitch2_moto_tmp']\n",
    "\n",
    "    #机舱温度减\n",
    "    train_data['i_p1_ng_t'] = train_data['int_tmp'] - train_data['pitch1_ng5_tmp']\n",
    "    train_data['i_p2_m_t'] = train_data['int_tmp'] - train_data['pitch2_moto_tmp']\n",
    "\n",
    "    #机舱温度和环境温度的差\n",
    "    train_data['e_i_t'] = train_data['environment_tmp'] - train_data['int_tmp']\n",
    "\n",
    "\n",
    "\n",
    "    #计算偏航系统的特征值\n",
    "    #def cal_yaw(train_data):\n",
    "    # 计算风速速度的次方\n",
    "    train_data['wind_speed_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_speed']]\n",
    "    train_data['wind_direction_mean_cos'] = [(lambda x : math.cos(x*(math.pi)/180))(x) for x in train_data['wind_direction_mean']]\n",
    "    train_data['wind_direction_mean_cos_2'] = [(lambda x : math.pow(x,2))(x) for x in train_data['wind_direction_mean_cos']]\n",
    "    # 计算偏航位置特征\n",
    "    train_data['yaw_ch1'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_position']]\n",
    "    train_data['yaw_ch2'] = [(lambda x : math.exp(-abs(x)))(x) for x in train_data['yaw_speed']]\n",
    "    #计算空气的密度\n",
    "    train_data['air_density_20'] = train_data['power']/(train_data['wind_speed_2']*train_data['wind_direction_mean_cos_2']*train_data['yaw_ch2'])\n",
    "\n",
    "    train_data.drop(['yaw_ch2','wind_direction_mean_cos_2','wind_direction_mean_cos'],axis=1, inplace=True)\n",
    "\n",
    "\n",
    "    return train_data"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [default]",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
